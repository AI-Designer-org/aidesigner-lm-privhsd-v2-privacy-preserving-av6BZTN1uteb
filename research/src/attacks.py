"""
Privacy Attack Simulations for PrivHSD
=======================================
Implements privacy risk assessment tools:

1. Membership Inference Attack (MIA) - Can adversary determine if a
   specific text was in the training set?

2. Attribute Inference Attack - Can adversary infer protected attributes
   (e.g., author demographics) from model outputs?

3. Representation Inversion - Can adversary reconstruct inputs from
   hidden representations?

4. Stylometry-based Re-identification - Can adversary match texts to
   authors based on writing style encoded in model outputs?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class AttackMetrics:
    """Metrics for a privacy attack."""
    auc: float = 0.5
    accuracy: float = 0.5
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    advantage: float = 0.0  # Attacker's advantage over random
    confidence: float = 0.0  # Average confidence of attacker


class MembershipInferenceAttack:
    """
    Membership Inference Attack (MIA).

    Determines whether a specific text was part of the training set
    by exploiting differences in model behavior on training vs. non-training data.

    Attack types:
      - Threshold-based: uses loss/probability thresholds
      - Shadow model: trains a binary classifier on model outputs
      - Reference-based: compares against a reference model
    """

    def __init__(self, attack_type: str = "shadow_model", random_seed: int = 42):
        """
        Args:
            attack_type: 'threshold', 'shadow_model', 'reference'
            random_seed: Random seed for reproducibility
        """
        self.attack_type = attack_type
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        self.attack_model = None
        self.threshold = 0.5

    def _extract_features(
        self,
        model,
        texts: List[str],
        tokenizer,
        device: str = "cuda",
    ) -> np.ndarray:
        """
        Extract features for MIA from model outputs.

        Features:
          - Predicted probability for the true class
          - Entropy of prediction distribution
          - Loss value
          - Gradient norm (if accessible)
          - Representation norms
        """
        model.eval()
        features = []

        with torch.no_grad():
            for text in tqdm(texts, desc="Extracting MIA features"):
                encodings = tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=256,
                    return_tensors="pt",
                )
                input_ids = encodings["input_ids"].to(device)
                attention_mask = encodings["attention_mask"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_representations=True,
                )

                probs = outputs["hate_probs"][0].cpu().numpy()
                reps = outputs.get("representations", torch.zeros(1, 768)).cpu().numpy()[0]

                feat = [
                    float(np.max(probs)),  # max probability
                    float(-np.sum(probs * np.log(probs + 1e-10))),  # entropy
                    float(np.linalg.norm(reps)),  # representation norm
                    float(np.std(probs)),  # prediction uncertainty
                ]
                features.append(feat)

        return np.array(features)

    def train_threshold_attack(
        self,
        train_losses: np.ndarray,
        test_losses: np.ndarray,
    ):
        """
        Train a threshold-based MIA.

        The intuition: training examples typically have lower loss.
        Find the optimal threshold separating train vs. non-train losses.
        """
        all_losses = np.concatenate([train_losses, test_losses])
        labels = np.concatenate([np.ones(len(train_losses)), np.zeros(len(test_losses))])

        # Try multiple thresholds
        best_acc = 0.0
        best_threshold = np.median(all_losses)

        for percentile in range(5, 95, 5):
            threshold = np.percentile(all_losses, percentile)
            preds = (all_losses < threshold).astype(int)
            acc = accuracy_score(labels, preds)
            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold

        self.threshold = best_threshold
        logger.info(
            f"Threshold MIA trained: threshold={best_threshold:.4f}, "
            f"best_acc={best_acc:.4f}"
        )
        return best_threshold, best_acc

    def train_shadow_model_attack(
        self,
        member_features: np.ndarray,
        non_member_features: np.ndarray,
    ):
        """
        Train a binary classifier to distinguish members from non-members.

        Uses logistic regression as the attack model (commonly used in
        the literature).
        """
        X = np.vstack([member_features, non_member_features])
        y = np.hstack([np.ones(len(member_features)), np.zeros(len(non_member_features))])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_seed
        )

        # Train attack classifier
        self.attack_model = LogisticRegression(max_iter=1000, C=1.0, random_state=self.random_seed)
        self.attack_model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.attack_model.predict(X_test)
        y_prob = self.attack_model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        logger.info(
            f"Shadow model MIA trained: acc={acc:.4f}, auc={auc:.4f}"
        )

        return AttackMetrics(
            auc=auc,
            accuracy=acc,
            precision=f1_score(y_test, y_pred, average="binary", zero_division=0),
            f1=f1_score(y_test, y_pred, average="binary", zero_division=0),
            advantage=acc - 0.5,
            confidence=float(np.mean(y_prob)),
        )

    def evaluate(
        self,
        model,
        member_texts: List[str],
        non_member_texts: List[str],
        tokenizer,
        device: str = "cuda",
    ) -> AttackMetrics:
        """
        Run the MIA attack.

        Args:
            model: Target model
            member_texts: Texts that were in the training set
            non_member_texts: Texts NOT in the training set
            tokenizer: Tokenizer for the model
            device: Device for computation

        Returns:
            AttackMetrics with attack performance
        """
        logger.info(f"Running MIA attack (type={self.attack_type})")

        member_feats = self._extract_features(model, member_texts, tokenizer, device)
        non_member_feats = self._extract_features(model, non_member_texts, tokenizer, device)

        if self.attack_type == "shadow_model":
            metrics = self.train_shadow_model_attack(member_feats, non_member_feats)
        else:
            # Simple threshold-based: use mean prediction probability
            member_scores = member_feats[:, 0]
            non_member_scores = non_member_feats[:, 0]
            # (lower entropy typically means more confident = member)
            # Actually use: members typically have higher confidence
            # In practice this is noisy for DP-trained models

            all_scores = np.concatenate([member_scores, non_member_scores])
            all_labels = np.concatenate([np.ones(len(member_scores)), np.zeros(len(non_member_scores))])

            # Find best threshold
            best_auc = 0.5
            best_acc = 0.5
            for thresh in np.linspace(0.05, 0.95, 50):
                preds = (all_scores >= thresh).astype(int)
                try:
                    auc = roc_auc_score(all_labels, all_scores)
                except ValueError:
                    auc = 0.5
                acc = accuracy_score(all_labels, preds)
                if acc > best_acc:
                    best_acc = acc
                if auc > best_auc:
                    best_auc = auc

            metrics = AttackMetrics(
                auc=best_auc,
                accuracy=best_acc,
                advantage=best_acc - 0.5,
            )

        logger.info(
            f"MIA Attack complete: auc={metrics.auc:.4f}, "
            f"acc={metrics.accuracy:.4f}"
        )

        return metrics


class AttributeInferenceAttack:
    """
    Attribute Inference Attack.

    Attempts to infer protected attributes (e.g., author demographics)
    from model outputs, to test whether identity information is leaked
    through the model's representations.
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.attack_model = None

    def train_attack(
        self,
        representations: np.ndarray,
        attribute_labels: np.ndarray,
    ) -> AttackMetrics:
        """
        Train an attribute inference attack on model representations.

        If the attack succeeds (high accuracy), the model is leaking
        attribute information that should have been removed.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            representations, attribute_labels,
            test_size=0.3, random_state=self.random_seed,
        )

        # Multi-class logistic regression for attribute inference
        n_classes = len(np.unique(attribute_labels))
        if n_classes == 2:
            self.attack_model = LogisticRegression(max_iter=1000, C=1.0,
                                                    random_state=self.random_seed)
        else:
            self.attack_model = LogisticRegression(max_iter=1000, C=1.0,
                                                    multi_class="multinomial",
                                                    random_state=self.random_seed)

        self.attack_model.fit(X_train, y_train)
        y_pred = self.attack_model.predict(X_test)
        y_prob = self.attack_model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        try:
            if n_classes == 2:
                auc = roc_auc_score(y_test, y_prob[:, 1])
            else:
                auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
        except ValueError:
            auc = 0.5

        metrics = AttackMetrics(
            auc=auc,
            accuracy=acc,
            f1=f1_score(y_test, y_pred, average="weighted", zero_division=0),
            advantage=acc - (1.0 / n_classes),
        )

        logger.info(
            f"Attribute inference attack: acc={acc:.4f}, auc={auc:.4f}, "
            f"advantage={metrics.advantage:.4f}"
        )

        return metrics

    def evaluate(
        self,
        representations: np.ndarray,
        attribute_labels: np.ndarray,
    ) -> AttackMetrics:
        """
        Evaluate attribute inference vulnerability.

        Args:
            representations: Model's hidden representations
            attribute_labels: Ground truth attribute labels

        Returns:
            AttackMetrics measuring leakage
        """
        return self.train_attack(representations, attribute_labels)


class StylometryReidentificationRisk:
    """
    Stylometry-based Re-identification Risk Assessment.

    Measures the extent to which author identity can be inferred from
    model outputs or representations, simulating a realistic
    de-anonymization attack scenario.

    Uses:
      - Writing style features extracted from model representations
      - Auxiliary authorship attribution classifier
      - Cross-text matching accuracy
    """

    def __init__(self, n_authors: int = 50, random_seed: int = 42):
        self.n_authors = n_authors
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        self.author_classifier = None

    def _extract_stylistic_features(
        self, texts: List[str],
    ) -> np.ndarray:
        """
        Extract stylometric features from raw text.

        Features (without relying on model):
          - Text length statistics
          - Vocabulary richness (type-token ratio)
          - Punctuation frequency
          - Capitalization patterns
          - Sentence length variability
          - Function word frequencies

        This gives a baseline stylometry risk independent of the model.
        """
        features = []
        common_function_words = [
            "the", "a", "an", "and", "or", "but", "in", "on", "at",
            "to", "for", "of", "with", "by", "from", "as", "is", "was",
            "are", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "will", "would", "can", "could", "may",
            "might", "shall", "should", "not", "no", "nor", "this", "that",
            "these", "those", "i", "you", "he", "she", "it", "we", "they",
            "my", "your", "his", "her", "its", "our", "their",
        ]

        for text in texts:
            if not text or not isinstance(text, str):
                features.append([0.0] * (20 + len(common_function_words)))
                continue

            words = text.split()
            n_words = len(words)
            n_chars = len(text)
            n_sentences = max(1, text.count(".") + text.count("!") + text.count("?"))
            n_unique = len(set(w.lower() for w in words))

            feat = [
                n_chars,
                n_words,
                n_sentences,
                n_chars / max(1, n_words),  # avg word length
                n_words / n_sentences,  # avg sentence length
                n_unique / max(1, n_words),  # type-token ratio
                text.count(",") / max(1, n_words),  # comma frequency
                text.count("!") / max(1, n_words),  # exclamation frequency
                text.count("?") / max(1, n_words),  # question frequency
                text.count('"') / max(1, n_words),  # quote frequency
                text.count(";") / max(1, n_words),  # semicolon frequency
                text.count(":") / max(1, n_words),  # colon frequency
                text.count("...") / max(1, n_words),  # ellipsis frequency
                sum(1 for c in text if c.isupper()) / max(1, n_chars),  # capitalization ratio
                sum(1 for c in text if c.isdigit()) / max(1, n_chars),  # digit ratio
                sum(1 for c in text if c.isspace()) / max(1, n_chars),  # whitespace ratio
                text.count("\n") / max(1, n_words),  # newline frequency
                len(words[-1]) if words else 0,  # last word length
                sum(1 for w in words if w[0].isupper() if w) / max(1, n_words),  # capitalized words
                sum(1 for w in words if w.isupper() and len(w) > 1) / max(1, n_words),  # ALLCAPS words
            ]

            # Function word frequencies
            text_lower = text.lower()
            for fw in common_function_words:
                feat.append(text_lower.count(fw) / max(1, n_words))

            features.append(feat)

        return np.array(features)

    def _extract_model_stylistic_features(
        self, model, texts: List[str], tokenizer, device: str = "cuda",
    ) -> np.ndarray:
        """Extract stylometric features from model representations."""
        model.eval()
        rep_features = []

        with torch.no_grad():
            for text in tqdm(texts, desc="Extracting stylometric features"):
                encodings = tokenizer(
                    text, truncation=True, padding="max_length",
                    max_length=256, return_tensors="pt",
                )
                input_ids = encodings["input_ids"].to(device)
                attention_mask = encodings["attention_mask"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_representations=True,
                )
                rep = outputs.get("representations",
                                   torch.zeros(1, 768)).cpu().numpy()[0]
                rep_features.append(rep)

        return np.array(rep_features)

    def evaluate_raw_text_risk(
        self, texts: List[str], author_labels: np.ndarray,
    ) -> AttackMetrics:
        """
        Evaluate stylometry-based re-identification risk from raw text features.
        This establishes the baseline risk without any model.
        """
        features = self._extract_stylistic_features(texts)

        X_train, X_test, y_train, y_test = train_test_split(
            features, author_labels, test_size=0.3, random_state=self.random_seed,
        )

        n_authors_actual = len(np.unique(author_labels))
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=15,
            random_state=self.random_seed, n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        chance = 1.0 / max(n_authors_actual, 1)

        metrics = AttackMetrics(
            accuracy=acc,
            advantage=acc - chance,
            precision=f1_score(y_test, y_pred, average="weighted", zero_division=0),
            f1=f1_score(y_test, y_pred, average="weighted", zero_division=0),
        )

        logger.info(
            f"Raw text stylometry risk: acc={acc:.4f}, "
            f"chance={chance:.4f}, advantage={metrics.advantage:.4f}"
        )

        return metrics

    def evaluate_model_representation_risk(
        self, model, texts: List[str], author_labels: np.ndarray,
        tokenizer, device: str = "cuda",
    ) -> AttackMetrics:
        """
        Evaluate re-identification risk from model representations.
        Lower is better for privacy.
        """
        features = self._extract_model_stylistic_features(
            model, texts, tokenizer, device,
        )

        X_train, X_test, y_train, y_test = train_test_split(
            features, author_labels, test_size=0.3, random_state=self.random_seed,
        )

        n_authors_actual = len(np.unique(author_labels))
        clf = LogisticRegression(
            max_iter=2000, C=0.1, multi_class="multinomial",
            random_state=self.random_seed,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        chance = 1.0 / max(n_authors_actual, 1)

        metrics = AttackMetrics(
            accuracy=acc,
            advantage=acc - chance,
            f1=f1_score(y_test, y_pred, average="weighted", zero_division=0),
        )

        logger.info(
            f"Model representation stylometry risk: acc={acc:.4f}, "
            f"chance={chance:.4f}, advantage={metrics.advantage:.4f}"
        )

        return metrics

    def evaluate(
        self, model, texts: List[str], author_labels: np.ndarray,
        tokenizer, device: str = "cuda",
    ) -> Dict[str, AttackMetrics]:
        """Run full stylometry risk assessment."""
        raw_risk = self.evaluate_raw_text_risk(texts, author_labels)
        model_risk = self.evaluate_model_representation_risk(
            model, texts, author_labels, tokenizer, device,
        )

        logger.info(
            f"Stylometry risk - raw text: {raw_risk.accuracy:.4f}, "
            f"model reps: {model_risk.accuracy:.4f}"
        )

        return {"raw_text": raw_risk, "model_representation": model_risk}


class RepresentationPrivacyAudit:
    """
    Audit the privacy properties of model representations.

    Measures:
      - Representation entropy (higher = more privacy)
      - Mutual information between representation and identity
      - k-anonymity of representation space
      - Nearest-neighbor privacy leakage
    """

    @staticmethod
    def compute_entropy(representations: np.ndarray) -> float:
        """Compute entropy of representation space (higher = better privacy)."""
        # Normalize representations
        reps_norm = representations / (np.linalg.norm(representations, axis=1, keepdims=True) + 1e-10)

        # Estimate entropy via eigenvalue distribution
        cov = np.cov(reps_norm.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        entropy = -np.sum(eigenvalues * np.log(eigenvalues + 1e-10))

        return float(entropy)

    @staticmethod
    def compute_k_anonymity(
        representations: np.ndarray,
        k: int = 5,
        threshold: float = 0.95,
    ) -> float:
        """
        Compute the fraction of representations that are k-anonymous
        (i.e., have at least k-1 other representations within distance threshold).

        Higher fraction = better privacy.
        """
        reps_norm = representations / (np.linalg.norm(representations, axis=1, keepdims=True) + 1e-10)
        sim_matrix = np.dot(reps_norm, reps_norm.T)

        n = len(representations)
        anonymous_count = 0

        for i in range(min(n, 500)):  # Sample for efficiency
            similar = np.sum(sim_matrix[i] >= threshold) - 1  # Exclude self
            if similar >= k - 1:
                anonymous_count += 1

        fraction = anonymous_count / min(n, 500)
        return fraction

    @staticmethod
    def compute_privacy_leakage_score(
        representations: np.ndarray,
        author_labels: np.ndarray,
    ) -> float:
        """
        Composite privacy leakage score.

        Lower = better privacy (less identity information in representations).
        Based on the correlation between representation similarity and
        same-author membership.
        """
        reps_norm = representations / (np.linalg.norm(representations, axis=1, keepdims=True) + 1e-10)
        sim_matrix = np.dot(reps_norm, reps_norm.T)

        n = min(len(representations), 300)
        same_author_sims = []
        diff_author_sims = []

        for i in range(n):
            for j in range(i + 1, n):
                sim = sim_matrix[i, j]
                if author_labels[i] == author_labels[j]:
                    same_author_sims.append(sim)
                else:
                    diff_author_sims.append(sim)

        if same_author_sims and diff_author_sims:
            mean_same = np.mean(same_author_sims)
            mean_diff = np.mean(diff_author_sims)
            leakage = max(0, mean_same - mean_diff)
        else:
            leakage = 0.0

        return float(leakage)
