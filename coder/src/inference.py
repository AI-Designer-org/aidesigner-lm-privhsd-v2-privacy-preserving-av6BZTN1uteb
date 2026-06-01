"""
PrivHSD v2 Inference Engine
============================
Deployable inference interface for the trained PrivHSD model.

Privacy-preserving inference features:
  - No identity information stored or logged
  - Single-sample inference (no batching that could leak cross-sample info)
  - Configurable privacy mode for API deployment
  - Representations are NEVER returned to callers
  - Confidence score limited in privacy mode
"""

import torch
import json
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PrivHSDInference:
    """Inference wrapper for PrivHSD v2 model.

    Provides hate speech detection with privacy-preserving inference:
      - Representations never exposed
      - No text logging in privacy mode
      - Configurable output verbosity
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        privacy_mode: bool = True,
    ):
        """
        Args:
            model_path: Path to saved model checkpoint (.pt)
            device: Device for inference
            privacy_mode: If True, enables privacy-preserving features
                          (no logging of text, no caching, bounded output)
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.privacy_mode = privacy_mode

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get("model_config", {})

        # Initialize model
        from src.model import PrivHSDConfig, PrivHSDModelV2

        model_cfg = PrivHSDConfig(
            model_name=config.get("model_name", "albert-base-v2"),
            model_type=config.get("model_type", "albert"),
            num_hate_classes=config.get("num_hate_classes", 2),
            num_authors=config.get("num_authors", 100),
            alpha_final=config.get("alpha_final", 0.5),
            disentanglement_weight=config.get("disentanglement_weight", 0.3),
            mim_weight=config.get("mim_weight", 0.0),
        )
        self.model = PrivHSDModelV2(model_cfg)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Load tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.get("model_name", "albert-base-v2")
        )

        self.final_epsilon = checkpoint.get("final_epsilon", float("inf"))
        self.best_f1 = checkpoint.get("best_val_f1", 0.0)

        logger.info(
            f"Inference engine ready: model={config.get('model_name', 'unknown')}, "
            f"ε={self.final_epsilon:.2f}, F1={self.best_f1:.4f}, "
            f"privacy_mode={privacy_mode}"
        )

    @torch.no_grad()
    def predict(self, text: str) -> Dict:
        """Predict hate speech probability for a single text.

        In privacy mode, no text is logged or stored.
        Representations are NEVER returned to callers.

        Args:
            text: Input text to classify.

        Returns:
            Dict with 'hate_probability' (float in [0, 1]) and
            'is_hate' (bool, threshold at 0.5).
            In non-privacy mode, also includes 'confidence'.
            NEVER returns representations or intermediate features.

        Shape invariants:
            Single string input; output is a dict (not a tensor).
        """
        if not text or not isinstance(text, str):
            return {"error": "Invalid input", "hate_probability": 0.0, "is_hate": False}

        encodings = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        hate_prob = float(outputs["hate_probs"][0, 1].cpu().numpy())

        result = {
            "hate_probability": round(hate_prob, 4),
            "is_hate": hate_prob > 0.5,
        }

        # In privacy mode, limit output detail
        if not self.privacy_mode:
            result["confidence"] = round(abs(hate_prob - 0.5) * 2, 4)

        # Representations are NEVER returned
        return result

    @torch.no_grad()
    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """Predict hate speech for multiple texts.

        Args:
            texts: List of input texts
            batch_size: Batch size (for efficiency, limited to prevent
                       cross-sample privacy leakage)

        Returns:
            List of prediction dicts
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]
            for text in batch_texts:
                results.append(self.predict(text))
        return results

    def validate_identity_agnostic(self, texts_by_author: Dict[str, List[str]]) -> Dict:
        """Validate that predictions are identity-agnostic.

        Tests whether the same hateful content produces consistent predictions
        regardless of author-specific phrasing.

        Args:
            texts_by_author: Dict mapping author_id -> list of their texts

        Returns:
            Dict with cross-author consistency metrics.
            'identity_agnostic_score' of 1.0 = perfect invariance.
        """
        all_probs = []
        author_means = {}

        for author_id, author_texts in texts_by_author.items():
            probs = [self.predict(t)["hate_probability"] for t in author_texts]
            all_probs.extend(probs)
            author_means[str(author_id)] = {
                "mean_prob": float(np.mean(probs)),
                "std_prob": float(np.std(probs)),
                "n_samples": len(probs),
            }

        author_var = float(np.var([m["mean_prob"] for m in author_means.values()]))
        return {
            "overall_mean": float(np.mean(all_probs)),
            "overall_std": float(np.std(all_probs)),
            "author_variance": author_var,
            "author_means": author_means,
            "identity_agnostic_score": max(0.0, 1.0 - author_var),
        }
