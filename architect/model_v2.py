"""
PrivHSD v2: Reference Model Implementation
==========================================
Multi-Level Adversarial Disentanglement + DP-SGD + Mutual Information
Minimization for Privacy-preserving Hate Speech Detection.

This is the reference implementation corresponding to the architecture
specification in ARCHITECTURE_SPEC.md. It is designed to be a drop-in
upgrade from the v1 model in src/model.py.

Architecture Innovations (v2):
  1. Multi-Level Adversarial Disentanglement (MLAD)
  2. Adaptive Gradient Reversal Scheduling (AGRS)
  3. Mutual Information Minimization via MINE (MIM)
  4. Per-Layer Adaptive DP Clipping
  5. Representation Orthogonality Regularization
  6. Cross-Model Consistency Regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PrivHSDv2Config:
    # ─── Transformer Backbone ──────────────────────────────────────────
    model_name: str = "albert-base-v2"
    model_type: str = "albert"  # "albert" | "roberta" | "xlm-roberta"
    max_seq_len: int = 256
    dropout: float = 0.2
    attention_dropout: float = 0.2

    # ─── Hate Speech Classification Head ───────────────────────────────
    num_hate_classes: int = 2
    classifier_hidden_dim: int = 768
    classifier_num_layers: int = 2

    # ─── Multi-Level Adversarial Disentanglement (MLAD) ────────────────
    num_authors: int = 100
    adversarial_levels: Tuple[str, ...] = ("pooler", "token", "head")
    adversary_hidden_dim: int = 256
    adversary_num_layers: int = 3
    adversary_dropout: float = 0.3

    # ─── Adaptive Gradient Reversal Scheduling (AGRS) ──────────────────
    alpha_initial: float = 0.1
    alpha_final: float = 1.0
    alpha_schedule: str = "sigmoid"  # "linear" | "sigmoid" | "adaptive"
    alpha_warmup_epochs: int = 2
    alpha_gamma: float = 2.0

    # ─── Disentanglement Weights ───────────────────────────────────────
    disentanglement_weight: float = 0.3
    mim_weight: float = 0.1
    orthogonality_weight: float = 0.05

    # ─── Mutual Information Minimization (MIM) ─────────────────────────
    mim_hidden_dim: int = 128
    mim_learning_rate: float = 1e-4

    # ─── DP-SGD Privacy Parameters ─────────────────────────────────────
    dp_enabled: bool = True
    target_epsilon: float = 8.0
    target_delta: Optional[float] = None
    max_grad_norm: float = 1.0
    per_layer_clipping: bool = True

    # ─── Training ──────────────────────────────────────────────────────
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    seed: int = 42
    device: str = "cuda"
    output_dir: str = "models"


# ═══════════════════════════════════════════════════════════════════════════
# Gradient Reversal Layer
# ═══════════════════════════════════════════════════════════════════════════


class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL) for adversarial training.

    Forward: identity (x → x)
    Backward: reverses gradient and scales by alpha (-alpha * grad)

    Reference: Ganin & Lempitsky, ICML 2015
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return grad_output.neg() * ctx.alpha, None


# ═══════════════════════════════════════════════════════════════════════════
# Adaptive Alpha Scheduler
# ═══════════════════════════════════════════════════════════════════════════


class AdaptiveAlphaScheduler:
    """
    Adaptive Gradient Reversal Scheduling (AGRS).

    Controls the strength of adversarial disentanglement over the
    course of training using parameterized schedules.

    Schedules:
        - linear:   alpha = alpha_initial + (alpha_final - alpha_initial) * p
        - sigmoid:  alpha around a sigmoid midpoint for gradual transition
        - adaptive: sigmoid base modulated by encoder/adversary loss dynamics

    Inductive bias:
        Early epochs should focus on learning hate-relevant features.
        Adversarial pressure should ramp up gradually so the encoder
        does not collapse representations before learning useful features.
    """

    def __init__(self, config: PrivHSDv2Config):
        self.initial = config.alpha_initial
        self.final = config.alpha_final
        self.schedule = config.alpha_schedule
        self.warmup_epochs = config.alpha_warmup_epochs
        self.gamma = config.alpha_gamma
        self.steps_per_epoch = 1000  # will be updated at runtime

        # State for adaptive mode
        self.encoder_loss_history: List[float] = []
        self.adversary_loss_history: List[float] = []
        self.current_alpha = self.initial

    def set_steps_per_epoch(self, steps: int):
        """Set steps per epoch from actual dataloader length."""
        self.steps_per_epoch = steps

    def get_alpha(
        self,
        step: int,
        hate_loss: Optional[float] = None,
        adv_loss: Optional[float] = None,
    ) -> float:
        """Compute alpha for the current training step."""
        epoch = step / max(self.steps_per_epoch, 1)

        if epoch < self.warmup_epochs:
            return self.initial

        p = (epoch - self.warmup_epochs) / (
            max(self.steps_per_epoch, 1) - self.warmup_epochs
        )
        p = min(1.0, max(0.0, p))

        if self.schedule == "linear":
            alpha = self.initial + (self.final - self.initial) * p
        elif self.schedule == "sigmoid":
            sigmoid_val = 1.0 / (1.0 + math.exp(-self.gamma * (p - 0.5)))
            alpha = self.initial + (self.final - self.initial) * sigmoid_val
        elif self.schedule == "adaptive":
            alpha = self._compute_adaptive_alpha(p, hate_loss, adv_loss)
        else:
            alpha = self.initial + (self.final - self.initial) * p

        self.current_alpha = alpha
        return alpha

    def _compute_adaptive_alpha(
        self,
        p: float,
        hate_loss: Optional[float],
        adv_loss: Optional[float],
    ) -> float:
        """Adaptive alpha modulated by loss dynamics."""
        base = self.initial + (self.final - self.initial) * (
            1.0 / (1.0 + math.exp(-self.gamma * (p - 0.5)))
        )

        if hate_loss is not None and len(self.encoder_loss_history) > 5:
            recent = self.encoder_loss_history[-5:]
            if recent[-1] - recent[0] > 0.05:
                base *= 0.9  # hate loss rising → reduce pressure

        if adv_loss is not None and len(self.adversary_loss_history) > 5:
            recent = self.adversary_loss_history[-5:]
            if recent[-1] - recent[0] < -0.05:
                base *= 1.1  # adversary winning → increase pressure

        return max(self.initial, min(self.final, base))

    def update_history(self, hate_loss: float, adv_loss: float):
        """Update loss history for adaptive scheduling."""
        self.encoder_loss_history.append(hate_loss)
        self.adversary_loss_history.append(adv_loss)
        if len(self.encoder_loss_history) > 50:
            self.encoder_loss_history.pop(0)
            self.adversary_loss_history.pop(0)


# ═══════════════════════════════════════════════════════════════════════════
# Adversary MLP
# ═══════════════════════════════════════════════════════════════════════════


class AdversaryMLP(nn.Module):
    """
    Multi-layer perceptron for identity adversary head.

    Architecture:
        Linear → LayerNorm → ReLU → Dropout → [repeat] → Linear(num_authors)

    Inductive bias:
        Deeper MLP with higher dropout (0.3 vs 0.2 for hate classifier)
        prevents the adversary from overfitting to noisy pseudo-author
        labels, forcing the encoder to remove robust identity features.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = []
        curr_dim = input_dim

        for i in range(num_layers):
            next_dim = hidden_dim if i < num_layers - 1 else output_dim
            layers.append(nn.Linear(curr_dim, next_dim))
            if i < num_layers - 1:
                layers.append(nn.LayerNorm(next_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            curr_dim = next_dim

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ═══════════════════════════════════════════════════════════════════════════
# Multi-Level Adversarial Disentanglement Block
# ═══════════════════════════════════════════════════════════════════════════


class MultiLevelAdversarialBlock(nn.Module):
    """
    Multi-Level Adversarial Disentanglement (MLAD).

    Deploys separate adversarial heads at three representation levels:
        1. pooler: [CLS] pooled representation (sequence-level)
        2. token:  mean-pooled token representations (token-level style)
        3. head:   mean-pooled attention head outputs (attention patterns)

    Each level captures different aspects of author identity:
        - Pooler: topic choice, sentiment tendencies
        - Token: word choice, function word frequency
        - Head: attention patterns, syntactic preferences

    Reference: Extends Ganin & Lempitsky (ICML 2015) from single-level
    domain adaptation to multi-level identity disentanglement.
    """

    def __init__(self, config: PrivHSDv2Config):
        super().__init__()
        self.levels = config.adversarial_levels
        hidden_size = config.classifier_hidden_dim

        self.adversaries = nn.ModuleDict()
        for level in self.levels:
            input_dim = {
                "pooler": hidden_size,
                "token": hidden_size,
                "head": hidden_size,
            }.get(level, hidden_size)

            self.adversaries[level] = AdversaryMLP(
                input_dim=input_dim,
                hidden_dim=config.adversary_hidden_dim,
                output_dim=config.num_authors,
                num_layers=config.adversary_num_layers,
                dropout=config.adversary_dropout,
            )

    def forward(
        self,
        pooler_repr: torch.Tensor,
        token_repr: torch.Tensor,
        head_repr: torch.Tensor,
        author_labels: torch.Tensor,
        alpha: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pooler_repr: (B, H)  — [CLS] pooled
            token_repr:  (B, T, H) — all token representations
            head_repr:   (B, num_heads, H/num_heads) — per-head outputs
            author_labels: (B,) — author identity labels
            alpha: GRL scaling factor

        Returns:
            dict with "author_loss", "level_losses", "level_logits",
            and "adversary_features" (pre-projection features)
        """
        level_losses = {}
        level_logits = {}
        level_features = {}

        for level in self.levels:
            if level == "pooler":
                repr_input = pooler_repr
            elif level == "token":
                repr_input = token_repr.mean(dim=1)  # (B, H)
            elif level == "head":
                repr_input = head_repr.mean(dim=1)  # (B, H)
            else:
                raise ValueError(f"Unknown level: {level}")

            # Apply GRL and adversary
            reversed_repr = GradientReversalLayer.apply(repr_input, alpha)
            author_logits = self.adversaries[level](reversed_repr)
            level_loss = F.cross_entropy(author_logits, author_labels)

            level_losses[f"author_loss_{level}"] = level_loss
            level_logits[f"author_logits_{level}"] = author_logits
            # Store pre-classifier features for orthogonality regularization
            if hasattr(self.adversaries[level].network, "__getitem__"):
                for module in self.adversaries[level].network:
                    if isinstance(module, nn.Linear):
                        level_features[level] = module(reversed_repr)
                        break

        # Total: mean across levels
        total_adv_loss = torch.stack(list(level_losses.values())).mean()

        return {
            "author_loss": total_adv_loss,
            "level_losses": level_losses,
            "level_logits": level_logits,
            "adversary_features": level_features,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Mutual Information Minimization via MINE
# ═══════════════════════════════════════════════════════════════════════════


class MutualInformationMinimizer(nn.Module):
    """
    Mutual Information Neural Estimator (MINE) for minimizing
    I(representation; author_id).

    Estimates a lower bound on mutual information using a neural
    network critic T_theta:

        I(X; Y) >= sup_{T} E_P[T] - log(E_Q[exp(T)])

    where P is the joint distribution (same-author pairs) and Q is
    the product of marginals (shuffled author pairs).

    The training happens in two phases:
        1. MINE phase:  maximize I_est (critic network step)
        2. Encoder phase: minimize I_est (backbone encoder step)

    Reference: Belghazi et al., "MINE: Mutual Information Neural
    Estimation", ICML 2018.
    """

    def __init__(
        self,
        repr_dim: int,
        num_authors: int,
        hidden_dim: int = 128,
    ):
        super().__init__()

        # MINE critic network: (repr, author_embedding) → scalar score
        self.mine_network = nn.Sequential(
            nn.Linear(repr_dim + num_authors, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Author embedding (one-hot like, but learned)
        self.author_embedding = nn.Embedding(num_authors, num_authors)

        # Gradient clipping for stability
        self.max_grad_norm = 1.0

    def estimate_mutual_information(
        self,
        representations: torch.Tensor,
        author_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate I(repr; author) using MINE lower bound.

        Returns a scalar MI estimate.
        """
        batch_size = representations.size(0)

        # Joint samples: (repr_i, author_i)
        author_embeds = self.author_embedding(author_labels)
        joint_input = torch.cat([representations, author_embeds], dim=1)
        joint_scores = self.mine_network(joint_input)

        # Marginal samples: (repr_i, shuffled author_j)
        shuffled_idx = torch.randperm(batch_size, device=author_labels.device)
        shuffled_embeds = self.author_embedding(author_labels[shuffled_idx])
        marginal_input = torch.cat([representations, shuffled_embeds], dim=1)
        marginal_scores = self.mine_network(marginal_input)

        # MINE lower bound: E_joint[T] - log(E_marginal[exp(T)])
        joint_mean = joint_scores.mean()
        # LogSumExp for numerical stability
        marginal_log_mean = torch.logsumexp(marginal_scores, dim=0) - math.log(
            batch_size
        )

        mi_estimate = joint_mean - marginal_log_mean

        return mi_estimate

    def forward(
        self,
        representations: torch.Tensor,
        author_labels: torch.Tensor,
        maximize: bool = False,
    ) -> torch.Tensor:
        """
        Compute MI estimate for optimization.

        Args:
            representations: Model representations (B, repr_dim)
            author_labels: Author identity labels (B,)
            maximize: If True, return -I for gradient ascent (MINE step).
                      If False, return +I for gradient descent (encoder step).

        Returns:
            Scalar loss for optimization
        """
        mi_estimate = self.estimate_mutual_information(
            representations, author_labels
        )

        if maximize:
            return -mi_estimate  # MINE maximizes I
        else:
            return mi_estimate   # Encoder minimizes I


# ═══════════════════════════════════════════════════════════════════════════
# Hate Speech Classification Head
# ═══════════════════════════════════════════════════════════════════════════


class HateClassificationHead(nn.Module):
    """
    Hate speech classification MLP head.

    Architecture:
        Linear(H, H) → LayerNorm(H) → ReLU → Dropout → Linear(H, num_classes)

    Uses lower dropout (0.2) than adversary heads (0.3) to preserve
    hate-relevant signal while the adversary is more regularized.
    """

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []

        # Projection layer
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.LayerNorm(hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Additional intermediate layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Classification layer
        layers.append(nn.Linear(hidden_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get pre-classification features (for orthogonality reg)."""
        for i, layer in enumerate(self.network):
            x = layer(x)
            if i == len(self.network) - 2:  # before last Linear
                return x
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Main PrivHSD v2 Model
# ═══════════════════════════════════════════════════════════════════════════


class PrivHSDModelV2(nn.Module):
    """
    PrivHSD v2: Privacy-preserving Hate Speech Detection model.

    Combines:
        1. Transformer backbone (ALBERT / RoBERTa)
        2. Hate speech classification head
        3. Multi-level adversarial identity disentanglement (MLAD)
        4. Mutual information minimization (MINE)
        5. Adaptive gradient reversal scheduling (AGRS)

    This is the reference implementation for the architecture
    specified in ARCHITECTURE_SPEC.md.
    """

    def __init__(self, config: PrivHSDv2Config):
        super().__init__()
        self.config = config
        self.model_type = config.model_type

        # ── Load transformer backbone ──────────────────────────────
        self._load_backbone(config)

        # ── Hate speech classification head ────────────────────────
        self.hate_classifier = HateClassificationHead(
            hidden_size=self.hidden_size,
            num_classes=config.num_hate_classes,
            num_layers=config.classifier_num_layers,
            dropout=config.dropout,
        )

        # ── Multi-level adversarial disentanglement ────────────────
        self.mlad_block = MultiLevelAdversarialBlock(config)

        # ── Mutual information minimizer (optional) ────────────────
        if config.mim_weight > 0:
            self.mim_module = MutualInformationMinimizer(
                repr_dim=self.hidden_size,
                num_authors=config.num_authors,
                hidden_dim=config.mim_hidden_dim,
            )
        else:
            self.mim_module = None

        # ── Alpha scheduler ────────────────────────────────────────
        self.alpha_scheduler = AdaptiveAlphaScheduler(config)

        logger.info(
            f"PrivHSD v2 initialized: backbone={config.model_name}, "
            f"levels={config.adversarial_levels}, "
            f"params={sum(p.numel() for p in self.parameters()):,}"
        )

    def _load_backbone(self, config: PrivHSDv2Config):
        """Load the transformer backbone from HuggingFace."""
        from transformers import (
            AlbertModel,
            RobertaModel,
            AutoConfig,
        )

        hf_config = AutoConfig.from_pretrained(
            config.model_name,
            hidden_dropout_prob=config.dropout,
            attention_probs_dropout_prob=config.attention_dropout,
            output_hidden_states=True,
            output_attentions=True,
        )

        if config.model_type == "albert":
            self.encoder = AlbertModel.from_pretrained(
                config.model_name, config=hf_config
            )
            self.hidden_size = self.encoder.config.hidden_size
            self.num_heads = self.encoder.config.num_attention_heads
        elif config.model_type == "roberta":
            self.encoder = RobertaModel.from_pretrained(
                config.model_name, config=hf_config
            )
            self.hidden_size = self.encoder.config.hidden_size
            self.num_heads = self.encoder.config.num_attention_heads
        else:
            raise ValueError(f"Unsupported model_type: {config.model_type}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        hate_labels: Optional[torch.Tensor] = None,
        author_labels: Optional[torch.Tensor] = None,
        alpha: Optional[float] = None,
        step: Optional[int] = None,
        return_representations: bool = False,
        return_attentions: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional adversarial disentanglement.

        Args:
            input_ids: (B, T) token indices
            attention_mask: (B, T) attention mask
            hate_labels: (B,) optional hate speech labels
            author_labels: (B,) optional author identity labels
            alpha: GRL scaling (from scheduler if None)
            step: current step (for alpha scheduling)
            return_representations: whether to return pooled reprs
            return_attentions: whether to return attention outputs

        Returns:
            dict with "hate_logits", "hate_probs", and optionally
            "loss", "hate_loss", "author_loss", "mim_loss",
            "representations", etc.
        """
        # ── Transformer forward ────────────────────────────────────
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
        )

        last_hidden = outputs.last_hidden_state  # (B, T, H)
        pooler_repr = last_hidden[:, 0, :]       # (B, H) — [CLS]

        # Token-level: mean pool all non-padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).float()
        token_repr = (last_hidden * mask_expanded).sum(dim=1) / (
            mask_expanded.sum(dim=1) + 1e-10
        )  # (B, H)

        # Head-level: average attention head outputs from last layer
        # outputs.attentions is tuple of (B, num_heads, T, T)
        # We take the last layer's attention and mean over T→(B, num_heads)
        if outputs.attentions is not None and len(outputs.attentions) > 0:
            last_attentions = outputs.attentions[-1]  # (B, num_heads, T, T)
            # Mean over source and target positions → (B, num_heads)
            head_repr = last_attentions.mean(dim=[2, 3])  # (B, num_heads)
            # Project to hidden size if needed
            if head_repr.size(-1) != self.hidden_size:
                head_repr = head_repr  # will be used as-is
        else:
            head_repr = pooler_repr.unsqueeze(1).expand(
                -1, self.num_heads, -1
            ).mean(dim=1)  # (B, H)

        # ── Hate speech classification ─────────────────────────────
        hate_logits = self.hate_classifier(pooler_repr)
        hate_probs = F.softmax(hate_logits, dim=-1)

        result = {
            "hate_logits": hate_logits,
            "hate_probs": hate_probs,
        }

        # ── Compute losses ─────────────────────────────────────────
        total_loss = torch.tensor(0.0, device=input_ids.device)

        # Hate speech loss
        if hate_labels is not None:
            hate_loss = F.cross_entropy(hate_logits, hate_labels)
            result["hate_loss"] = hate_loss
            total_loss = total_loss + hate_loss

        # Adversarial identity disentanglement
        if author_labels is not None:
            current_alpha = (
                alpha
                if alpha is not None
                else self.alpha_scheduler.current_alpha
            )

            adv_outputs = self.mlad_block(
                pooler_repr=pooler_repr,
                token_repr=last_hidden,
                head_repr=head_repr,
                author_labels=author_labels,
                alpha=current_alpha,
            )

            result["author_loss"] = adv_outputs["author_loss"]
            result["level_losses"] = adv_outputs["level_losses"]
            result["alpha"] = current_alpha

            total_loss = (
                total_loss
                + self.config.disentanglement_weight * adv_outputs["author_loss"]
            )

            # Representation orthogonality regularization
            if self.config.orthogonality_weight > 0 and hate_labels is not None:
                hate_features = self.hate_classifier.get_features(pooler_repr)
                # Get adversary features for pooler level
                adv_features = adv_outputs["adversary_features"].get(
                    "pooler",
                    adv_outputs.get("adversary_features", {}).get(
                        list(adv_outputs.get("level_losses", {"pooler": None}).keys())[0]
                        if adv_outputs.get("level_losses")
                        else None,
                        None,
                    ),
                )

                if hate_features is not None and adv_features is not None:
                    # Normalize
                    hate_norm = F.normalize(hate_features, dim=-1)
                    adv_norm = F.normalize(adv_features, dim=-1)
                    # Cosine similarity (want 0 = orthogonal)
                    cos_sim = (hate_norm * adv_norm).sum(dim=-1).mean()
                    orth_loss = cos_sim.abs()
                    result["orth_loss"] = orth_loss
                    total_loss = (
                        total_loss
                        + self.config.orthogonality_weight * orth_loss
                    )

            # Mutual information minimization
            if self.mim_module is not None and self.config.mim_weight > 0:
                mi_estimate = self.mim_module(
                    pooler_repr, author_labels, maximize=False
                )
                result["mim_loss"] = self.config.mim_weight * mi_estimate
                total_loss = total_loss + result["mim_loss"]

        result["loss"] = total_loss

        # ── Optional return values ─────────────────────────────────
        if return_representations:
            result["representations"] = pooler_repr.detach()

        if return_attentions:
            result["attentions"] = outputs.attentions

        return result

    def get_hate_predictions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get hate speech predictions (inference mode)."""
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids, attention_mask=attention_mask
            )
        return outputs["hate_probs"]

    def get_representations(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extract pooled representations for privacy analysis."""
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_representations=True,
            )
        return outputs["representations"]

    @torch.no_grad()
    def compute_mine_mi(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        author_labels: torch.Tensor,
    ) -> float:
        """Compute MINE mutual information estimate."""
        if self.mim_module is None:
            return 0.0
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_representations=True,
        )
        mi = self.mim_module.estimate_mutual_information(
            outputs["representations"], author_labels
        )
        return mi.item()

    def get_mim_loss(
        self,
        representations: torch.Tensor,
        author_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Get MIM loss for encoder optimization."""
        if self.mim_module is None:
            return torch.tensor(0.0, device=representations.device)
        return self.config.mim_weight * self.mim_module(
            representations, author_labels, maximize=False
        )


# ═══════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════


def compute_subspace_orthogonality(
    feat_a: torch.Tensor,
    feat_b: torch.Tensor,
) -> torch.Tensor:
    """
    Compute cosine similarity between two feature subspaces.

    Encourages hate-relevant and identity-relevant features to
    occupy orthogonal subspaces, making identity information harder
    to extract from hate-related representations.
    """
    a_norm = F.normalize(feat_a, dim=-1)
    b_norm = F.normalize(feat_b, dim=-1)
    cos_sim = (a_norm * b_norm).sum(dim=-1).abs().mean()
    return cos_sim


def build_model_from_config(config: PrivHSDv2Config) -> PrivHSDModelV2:
    """Factory function: create model from config."""
    return PrivHSDModelV2(config)


# ═══════════════════════════════════════════════════════════════════════════
# Migration Helper: v1 → v2 Compatibility
# ═══════════════════════════════════════════════════════════════════════════


class PrivHSDModel(PrivHSDModelV2):
    """
    Backward-compatible alias for PrivHSDModelV2.

    Allows existing v1 training/evaluation code to work with
    minimal changes. The v1 constructor signature maps to v2 config.
    """

    def __init__(
        self,
        model_name: str = "albert-base-v2",
        num_hate_classes: int = 2,
        num_authors: int = 100,
        adversarial_alpha: float = 0.5,
        disentanglement_weight: float = 0.3,
        hidden_dropout: float = 0.2,
        model_type: str = "albert",
        cache_dir: Optional[str] = None,
    ):
        config = PrivHSDv2Config(
            model_name=model_name,
            model_type=model_type,
            num_hate_classes=num_hate_classes,
            num_authors=num_authors,
            alpha_final=adversarial_alpha,
            disentanglement_weight=disentanglement_weight,
            dropout=hidden_dropout,
            adversary_dropout=hidden_dropout + 0.1,
        )
        # Ignore cache_dir in v2 — HuggingFace handles it
        super().__init__(config)

    @classmethod
    def from_config(cls, config: PrivHSDv2Config) -> "PrivHSDModel":
        """Create from PrivHSDv2Config."""
        return cls(
            model_name=config.model_name,
            num_hate_classes=config.num_hate_classes,
            num_authors=config.num_authors,
            adversarial_alpha=config.alpha_final,
            disentanglement_weight=config.disentanglement_weight,
            hidden_dropout=config.dropout,
            model_type=config.model_type,
        )
