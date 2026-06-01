"""
PrivHSD v2: Privacy-preserving Hate Speech Detection Model
===========================================================
Multi-Level Adversarial Disentanglement (MLAD) + DP-SGD + Mutual Information
Minimization (MINE) for author-identity-agnostic hate speech detection.

Architecture Innovations (v2):
  1. Multi-Level Adversarial Disentanglement (MLAD)
  2. Adaptive Gradient Reversal Scheduling (AGRS)
  3. Mutual Information Minimization via MINE (MIM)
  4. Per-Layer Adaptive DP Clipping interface
  5. Representation Orthogonality Regularization

Shape convention: (B, T, D) — batch, seq_len, d_model
Domain: LM + Privacy ML
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PrivHSDConfig:
    """Complete configuration for PrivHSD v2 model.

    All hyperparameters are centralized here for systematic ablation.
    """

    # ─── Transformer Backbone ──────────────────────────────────────────
    model_name: str = "albert-base-v2"       # HF model identifier
    model_type: str = "albert"               # "albert" | "roberta" | "xlm-roberta"
    d_model: int = 768                        # hidden dimension
    n_layers: int = 12                        # transformer layers
    n_heads: int = 12                         # attention heads
    d_ff: int = 3072                          # feed-forward dimension
    max_seq_len: int = 256                    # truncation length
    dropout: float = 0.2                      # hidden dropout
    attention_dropout: float = 0.2            # attention dropout

    # ─── Hate Speech Classification Head ───────────────────────────────
    num_hate_classes: int = 2                 # binary hate/not-hate
    classifier_hidden_dim: int = 768          # MLP hidden dim
    classifier_num_layers: int = 2            # MLP depth

    # ─── Multi-Level Adversarial Disentanglement (MLAD) ────────────────
    num_authors: int = 100                    # pseudo-author classes
    adversarial_levels: Tuple[str, ...] = (
        "pooler", "token", "head"
    )                                          # levels for adversaries
    adversary_hidden_dim: int = 256           # adversary MLP hidden size
    adversary_num_layers: int = 3             # adversary MLP depth
    adversary_dropout: float = 0.3            # adversary dropout (higher=more regularization)

    # ─── Adaptive Gradient Reversal Scheduling (AGRS) ──────────────────
    alpha_initial: float = 0.1                # initial GRL scaling
    alpha_final: float = 1.0                  # final GRL scaling
    alpha_schedule: str = "sigmoid"           # "linear" | "sigmoid" | "adaptive"
    alpha_warmup_epochs: int = 2              # warmup before alpha increases
    alpha_gamma: float = 2.0                  # sigmoid steepness

    # ─── Disentanglement Loss Weights ──────────────────────────────────
    disentanglement_weight: float = 0.3       # main adversarial loss weight
    mim_weight: float = 0.1                   # mutual information minimization weight
    orthogonality_weight: float = 0.05        # representation orthogonality reg
    consistency_weight: float = 0.05          # cross-model consistency (future)

    # ─── Mutual Information Minimization (MIM) ─────────────────────────
    mim_estimator: str = "mine"               # "mine" | "nwj" | "info_nce"
    mim_hidden_dim: int = 128                 # MINE network hidden dim
    mim_learning_rate: float = 1e-4           # separate LR for MINE
    mim_momentum: float = 0.9                 # MINE optimizer momentum

    # ─── DP-SGD Privacy Parameters ─────────────────────────────────────
    dp_enabled: bool = True                   # master switch for DP-SGD
    target_epsilon: float = 8.0               # target privacy budget
    target_delta: Optional[float] = None      # auto = 1/|D| if None
    max_grad_norm: float = 1.0                # global clipping norm
    per_layer_clipping: bool = True           # adaptive per-layer clip norms
    noise_multiplier: Optional[float] = None  # auto-computed if None
    poisson_sampling: bool = True             # Poisson sampling for tighter DP

    # ─── Privacy-Augmented Data ────────────────────────────────────────
    privacy_augment_level: Optional[str] = None  # None | "low" | "medium" | "high"
    label_flip_prob: float = 0.05             # label noise probability
    word_dropout_prob: float = 0.10           # word dropout probability
    synonym_replacement_prob: float = 0.05    # synonym replacement
    entity_masking: bool = True               # mask named entities

    # ─── Training ──────────────────────────────────────────────────────
    batch_size: int = 16                      # per-GPU batch size
    learning_rate: float = 2e-5               # peak learning rate
    lr_schedule: str = "linear"               # "linear" | "cosine" | "constant"
    warmup_ratio: float = 0.1                 # fraction of steps for warmup
    weight_decay: float = 0.01                # AdamW weight decay
    num_epochs: int = 10                      # training epochs
    gradient_accumulation_steps: int = 1      # gradient accumulation
    early_stopping_patience: int = 5          # epochs without improvement
    eval_every: int = 1                       # evaluate every N epochs

    # ─── Optimization ──────────────────────────────────────────────────
    optimizer: str = "adamw"                  # "adamw" | "adam" | "sgd"
    beta1: float = 0.9                        # Adam beta1
    beta2: float = 0.999                      # Adam beta2
    epsilon: float = 1e-8                     # optimizer epsilon

    # ─── System ────────────────────────────────────────────────────────
    device: str = "cuda"
    mixed_precision: str = "fp16"             # "fp16" | "bf16" | "fp32"
    use_ghost_clipping: bool = True           # Opacus ghost clipping
    num_workers: int = 4                      # DataLoader workers
    seed: int = 42                            # random seed
    output_dir: str = "models"                # output directory

    # ─── Evaluation ────────────────────────────────────────────────────
    eval_metrics: Tuple[str, ...] = (
        "f1", "auc", "accuracy", "precision", "recall", "mcc", "specificity"
    )
    privacy_attack_types: Tuple[str, ...] = (
        "mia_shadow", "mia_threshold", "attribute_inference", "stylometry"
    )
    pareto_metrics: Tuple[str, ...] = (
        "epsilon", "f1_score", "mia_auc"
    )

    def __post_init__(self):
        if self.target_delta is None:
            self.target_delta = 1e-5  # overridden with dataset size


# ═══════════════════════════════════════════════════════════════════════════
# Gradient Reversal Layer
# ═══════════════════════════════════════════════════════════════════════════


class GradientReversalLayer(torch.autograd.Function):
    """Gradient Reversal Layer (GRL) for adversarial training.

    Forward: identity  (x → x)
    Backward: reverses and scales gradient  (-alpha * grad)

    Reference: Ganin & Lempitsky, ICML 2015
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """Forward: identity. x: (B, D) → (B, D) same shape."""
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Backward: negated and scaled gradient."""
        return grad_output.neg() * ctx.alpha, None


# ═══════════════════════════════════════════════════════════════════════════
# Adaptive Alpha Scheduler
# ═══════════════════════════════════════════════════════════════════════════


class AdaptiveAlphaScheduler:
    """Adaptive Gradient Reversal Scheduling (AGRS).

    Controls the strength of adversarial disentanglement over training
    using parameterized schedules. Early epochs focus on hate feature
    learning; adversarial pressure ramps up gradually.

    Schedule types:
        linear:   alpha = initial + (final - initial) * p
        sigmoid:  sigmoid ramp around midpoint p=0.5
        adaptive: sigmoid base modulated by encoder/adversary loss dynamics

    Inductive bias:
        Early training = learn hate features (alpha ≈ 0).
        Late training = max disentanglement (alpha ≈ 1).
    """

    def __init__(self, config: PrivHSDConfig):
        self.initial = config.alpha_initial       # 0.1
        self.final = config.alpha_final            # 1.0
        self.schedule = config.alpha_schedule      # "sigmoid"
        self.warmup_epochs = config.alpha_warmup_epochs  # 2
        self.gamma = config.alpha_gamma            # 2.0
        self.total_steps = 10000                   # will be updated at runtime
        self.warmup_steps = self.warmup_epochs * (self.total_steps // 10)

        # State for adaptive mode
        self.encoder_loss_history: List[float] = []
        self.adversary_loss_history: List[float] = []
        self.current_alpha = self.initial

    def set_total_steps(self, total_steps: int):
        """Set total training steps from actual dataloader."""
        self.total_steps = max(total_steps, 1)
        self.warmup_steps = self.warmup_epochs * (self.total_steps // 10)

    def get_alpha(
        self,
        step: int,
        hate_loss: Optional[float] = None,
        adv_loss: Optional[float] = None,
    ) -> float:
        """Compute alpha for current training step.

        Args:
            step: Current gradient step (0-indexed)
            hate_loss: Current hate loss (for adaptive mode)
            adv_loss: Current adversary loss (for adaptive mode)

        Returns:
            alpha: GRL scaling factor in [initial, final]
        """
        if self.total_steps <= 0:
            return self.initial

        p = step / self.total_steps  # progress in [0, 1]
        p = min(1.0, max(0.0, p))

        # Warmup phase: no adversarial pressure
        if step < self.warmup_steps:
            return self.initial

        # Ramp phase: map [warmup_steps, total_steps] → [0, 1]
        ramp_p = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        ramp_p = min(1.0, max(0.0, ramp_p))

        if self.schedule == "linear":
            alpha = self.initial + (self.final - self.initial) * ramp_p
        elif self.schedule == "sigmoid":
            sigmoid_val = 1.0 / (1.0 + math.exp(-self.gamma * (ramp_p - 0.5)))
            alpha = self.initial + (self.final - self.initial) * sigmoid_val
        elif self.schedule == "adaptive":
            alpha = self._compute_adaptive_alpha(ramp_p, hate_loss, adv_loss)
        else:
            # Default: sigmoid
            sigmoid_val = 1.0 / (1.0 + math.exp(-self.gamma * (ramp_p - 0.5)))
            alpha = self.initial + (self.final - self.initial) * sigmoid_val

        self.current_alpha = alpha
        return alpha

    def _compute_adaptive_alpha(
        self,
        p: float,
        hate_loss: Optional[float],
        adv_loss: Optional[float],
    ) -> float:
        """Adaptive alpha: sigmoid base modulated by loss dynamics.

        - If adversary loss decreasing (adversary winning) → increase alpha.
        - If hate loss increasing (disentanglement hurting utility) → decrease alpha.
        """
        # Sigmoid base
        base = self.initial + (self.final - self.initial) * (
            1.0 / (1.0 + math.exp(-self.gamma * (p - 0.5)))
        )

        # Modulate based on hate loss trend
        if hate_loss is not None and len(self.encoder_loss_history) > 5:
            recent = self.encoder_loss_history[-5:]
            if recent[-1] - recent[0] > 0.05:
                base *= 0.9  # hate loss rising → reduce pressure

        # Modulate based on adversary loss trend
        if adv_loss is not None and len(self.adversary_loss_history) > 5:
            recent = self.adversary_loss_history[-5:]
            if recent[-1] - recent[0] < -0.05:
                base *= 1.1  # adversary winning → increase pressure

        return max(self.initial, min(self.final, base))

    def update_history(self, hate_loss: float, adv_loss: float):
        """Update loss history for adaptive scheduling."""
        self.encoder_loss_history.append(hate_loss)
        self.adversary_loss_history.append(adv_loss)
        # Keep bounded window
        if len(self.encoder_loss_history) > 50:
            self.encoder_loss_history.pop(0)
            self.adversary_loss_history.pop(0)


# ═══════════════════════════════════════════════════════════════════════════
# Adversary MLP
# ═══════════════════════════════════════════════════════════════════════════


class AdversaryMLP(nn.Module):
    """Multi-layer perceptron for identity adversary head.

    Architecture: Linear → LayerNorm → ReLU → Dropout → ... → Linear(num_authors).

    Higher dropout (0.3 vs 0.2 for hate classifier) prevents overfitting to
    noisy pseudo-author labels.

    Shape: (B, D) → (B, num_authors)
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
            layers.append(nn.Linear(curr_dim, next_dim))  # (B, curr_dim) → (B, next_dim)
            if i < num_layers - 1:
                layers.append(nn.LayerNorm(next_dim))      # (B, next_dim) → (B, next_dim)
                layers.append(nn.ReLU())                   # (B, next_dim) → (B, next_dim)
                layers.append(nn.Dropout(dropout))          # (B, next_dim) → (B, next_dim)
            curr_dim = next_dim

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input to author logits.

        Args:
            x: (B, D) — adversary input representation.

        Returns:
            (B, num_authors) — author identity logits.

        Shape invariants: B ≥ 1; dtype in {float32, bfloat16}.
        """
        return self.network(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract pre-classification features for orthogonality regularization.

        Returns the hidden representation just before the final Linear
        projection layer.

        Args:
            x: (B, D) — adversary input representation.

        Returns:
            (B, hidden_dim) — pre-projection features.
        """
        # Track the last output before the final Linear layer
        last_hidden = None
        for module in self.network:
            if isinstance(module, nn.Linear):
                # Check if this is the last Linear layer (output_dim != hidden_dim)
                if module.out_features != module.in_features or (
                    last_hidden is not None and module.out_features != last_hidden.size(-1)
                ):
                    # This is likely the final projection; return cached pre-projection features
                    if last_hidden is not None:
                        return last_hidden
                x = module(x)
                last_hidden = x
            else:
                x = module(x)
                if isinstance(module, nn.ReLU) or isinstance(module, nn.Tanh):
                    last_hidden = x
        # Fallback: return the last non-final hidden state
        return last_hidden if last_hidden is not None else x


# ═══════════════════════════════════════════════════════════════════════════
# Multi-Level Adversarial Disentanglement Block
# ═══════════════════════════════════════════════════════════════════════════


class MultiLevelAdversarialBlock(nn.Module):
    """Multi-Level Adversarial Disentanglement (MLAD).

    Deploys separate adversarial heads at three representation levels:
        1. pooler: [CLS] pooled representation (sequence-level topic/style)
        2. token:  mean-pooled token representations (word-level style)
        3. head:   mean-pooled attention head outputs (attention patterns)

    Each level captures different aspects of author identity:
        - Pooler: topic choice, sentiment tendencies
        - Token: word choice, function word frequency
        - Head: attention patterns, syntactic preferences

    Reference: Extends Ganin & Lempitsky (ICML 2015) to multi-level identity
    disentanglement for privacy-preserving NLP.
    """

    def __init__(self, config: PrivHSDConfig):
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
        pooler_repr: torch.Tensor,   # (B, D)
        token_repr: torch.Tensor,     # (B, T, D)
        head_repr: torch.Tensor,      # (B, D)
        author_labels: torch.Tensor,  # (B,)
        alpha: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Multi-level adversarial forward pass.

        At each level (pooler/token/head), the representation is passed
        through a GRL (identity forward, -alpha*grad backward) and then
        an AdversaryMLP predicting author identity.

        Args:
            pooler_repr: (B, D) — [CLS] pooled representation.
            token_repr: (B, T, D) — all token representations (mean-pooled
                       internally to (B, D) before adversary).
            head_repr: (B, D) — per-head attention output averages.
            author_labels: (B,) — author identity labels (long).
            alpha: GRL scaling factor.

        Returns:
            dict with:
                "author_loss": scalar — mean of all level CE losses.
                "level_losses": dict — per-level CE losses.
                "level_logits": dict — per-level author logits.
                "adversary_features": dict — per-level pre-projection features.

        Shape invariants:
            B ≥ 1; T ≤ config.max_seq_len; dtype float32 or bfloat16.
        """
        level_losses = {}
        level_logits = {}
        level_features = {}

        for level in self.levels:
            if level == "pooler":
                repr_input = pooler_repr                           # (B, D)
            elif level == "token":
                repr_input = token_repr.mean(dim=1)                # (B, T, D) → (B, D)
            elif level == "head":
                # head_repr from get_transformer_outputs is already (B, D)
                repr_input = head_repr.view(head_repr.size(0), -1)  # (B, D)
            else:
                raise ValueError(f"Unknown level: {level}")

            # Apply GRL and adversary
            reversed_repr = GradientReversalLayer.apply(repr_input, alpha)  # (B, D)
            author_logits = self.adversaries[level](reversed_repr)          # (B, num_authors)

            # Compute CE loss — use float32 for numerical safety
            level_loss = F.cross_entropy(
                author_logits.float(), author_labels
            ).to(author_logits.dtype)                                         # scalar

            level_losses[f"author_loss_{level}"] = level_loss
            level_logits[f"author_logits_{level}"] = author_logits

            # Extract pre-classifier features for orthogonality regularization
            level_features[level] = self.adversaries[level].get_features(reversed_repr)

        # Total: mean across levels
        total_adv_loss = torch.stack(list(level_losses.values())).mean()  # scalar

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
    """Mutual Information Neural Estimator (MINE) for minimizing
    I(representation; author_id).

    Estimates a lower bound on mutual information using a neural
    network critic T_theta:

        I(X; Y) >= sup_T E_P[T] - log(E_Q[exp(T)])

    where P is the joint distribution (same-author pairs) and Q is
    the product of marginals (shuffled author pairs).

    Two-phase training:
        1. MINE phase:  maximize I_est (update MINE network)
        2. Encoder phase: minimize I_est (update backbone encoder)

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

        # MINE critic: (repr, author_embedding) → scalar score
        # Shape: (B, repr_dim + num_authors) → (B, 1)
        self.mine_network = nn.Sequential(
            nn.Linear(repr_dim + num_authors, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Author embedding (one-hot-like, learned)
        self.author_embedding = nn.Embedding(num_authors, num_authors)

    def estimate_mutual_information(
        self,
        representations: torch.Tensor,  # (B, repr_dim)
        author_labels: torch.Tensor,    # (B,)
    ) -> torch.Tensor:
        """Estimate I(repr; author) using MINE lower bound.

        Returns a scalar MI estimate (higher = more identity leakage).
        """
        batch_size = representations.size(0)

        # Joint samples: (repr_i, author_i)
        author_embeds = self.author_embedding(author_labels)       # (B, num_authors)
        joint_input = torch.cat([representations, author_embeds], dim=1)  # (B, repr_dim + num_authors)
        joint_scores = self.mine_network(joint_input)              # (B, 1)

        # Marginal samples: (repr_i, shuffled_author_j)
        shuffled_idx = torch.randperm(batch_size, device=author_labels.device)
        shuffled_embeds = self.author_embedding(author_labels[shuffled_idx])  # (B, num_authors)
        marginal_input = torch.cat([representations, shuffled_embeds], dim=1) # (B, repr_dim + num_authors)
        marginal_scores = self.mine_network(marginal_input)                  # (B, 1)

        # MINE lower bound: E_joint[T] - log(E_marginal[exp(T)])
        joint_mean = joint_scores.mean()  # scalar
        # LogSumExp for numerical stability
        marginal_log_mean = torch.logsumexp(marginal_scores, dim=0) - math.log(batch_size)  # scalar

        mi_estimate = joint_mean - marginal_log_mean  # scalar

        return mi_estimate

    def forward(
        self,
        representations: torch.Tensor,  # (B, repr_dim)
        author_labels: torch.Tensor,    # (B,)
        maximize: bool = False,
    ) -> torch.Tensor:
        """Compute MI estimate for optimization.

        Args:
            representations: (B, repr_dim) — model's pooled representations.
            author_labels: (B,) — author identity labels.
            maximize: If True, return -I for gradient ascent (MINE step).
                      If False, return +I for gradient descent (encoder step).

        Returns:
            Scalar MI estimate. Maximized by MINE network (grad ascent),
            minimized by encoder (grad descent).

        Shape invariants:
            B ≥ 1; representations dtype in {float32, bfloat16}.
        """
        mi_estimate = self.estimate_mutual_information(representations, author_labels)

        if maximize:
            return -mi_estimate  # MINE maximizes I → gradient ascent
        else:
            return mi_estimate   # Encoder minimizes I → gradient descent


# ═══════════════════════════════════════════════════════════════════════════
# Hate Speech Classification Head
# ═══════════════════════════════════════════════════════════════════════════


class HateClassificationHead(nn.Module):
    """Hate speech classification MLP head.

    Architecture:
        Linear(D, D) → LayerNorm(D) → ReLU → Dropout → Linear(D, num_classes)

    Uses lower dropout (0.2) than adversary heads (0.3) to preserve
    hate-relevant signal.
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

        # First projection
        layers.append(nn.Linear(hidden_size, hidden_size))  # (B, D) → (B, D)
        layers.append(nn.LayerNorm(hidden_size))             # (B, D) → (B, D)
        layers.append(nn.ReLU())                             # (B, D) → (B, D)
        layers.append(nn.Dropout(dropout))                   # (B, D) → (B, D)

        # Additional intermediate layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Final classification
        layers.append(nn.Linear(hidden_size, num_classes))   # (B, D) → (B, num_classes)

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify hate speech from pooled representations.

        Args:
            x: (B, D) — pooled representation from encoder.

        Returns:
            (B, num_classes) — logits, residual NOT added.

        Shape invariants:
            B ≥ 1; D must match d_model; dtype in {float32, bfloat16}.
        """
        return self.network(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get pre-classification features for orthogonality regularization.

        Args:
            x: (B, D) — pooled representation from encoder.

        Returns:
            (B, D) — hidden features after intermediate layers, before final Linear.
        """
        for i, layer in enumerate(self.network):
            x = layer(x)
            # Return after the last ReLU (before final Linear)
            if i == len(self.network) - 2:
                return x
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Per-Layer DP Clipping Adapter
# ═══════════════════════════════════════════════════════════════════════════


class PerLayerDPAdapter:
    """DP-SGD with per-layer adaptive gradient clipping.

    Different layers have different sensitivity to DP noise.
    Early layers (general linguistic features) need tighter clipping;
    later task-specific layers may need looser bounds.

    Method:
        1. Run a few non-private warmup steps to estimate per-layer gradient variance.
        2. Compute per-layer clip factors inversely proportional to variance.
        3. Apply per-layer clipping, scaling noise accordingly.

    Note: Full per-layer clipping requires Opacus internals modification.
    This adapter provides the estimation and interface; falls back to
    uniform clipping if unsupported.
    """

    @staticmethod
    def estimate_layer_sensitivity(
        model: nn.Module,
        sample_batch: Dict[str, torch.Tensor],
        n_steps: int = 5,
    ) -> List[float]:
        """Estimate per-layer gradient sensitivity via variance.

        Returns clip factors (one per named parameter).
        Higher sensitivity → tighter clipping (lower factor).
        """
        model.train()
        param_norms = {name: [] for name, _ in model.named_parameters()}

        for _ in range(n_steps):
            outputs = model(**sample_batch)
            loss = outputs.get("loss", torch.tensor(0.0))
            if loss.grad_fn is not None:
                loss.backward()
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param_norms[name].append(param.grad.norm().item())
                model.zero_grad()

        # Compute clip factors
        clip_factors = []
        for name in param_norms:
            norms = param_norms[name]
            if len(norms) > 1:
                var = float(np.var(norms))  # noqa: NPY  (imported at call site)
                # Higher variance → smaller clip factor (tighter clip)
                factor = 1.0 / (1.0 + var)
                clip_factors.append(max(0.1, min(2.0, factor)))
            else:
                clip_factors.append(1.0)

        return clip_factors


# ═══════════════════════════════════════════════════════════════════════════
# Main PrivHSD v2 Model
# ═══════════════════════════════════════════════════════════════════════════


class PrivHSDModelV2(nn.Module):
    """PrivHSD v2: Privacy-preserving Hate Speech Detection model.

    Combines:
        1. Transformer backbone (ALBERT / RoBERTa / XLM-RoBERTa)
        2. Hate speech classification head
        3. Multi-Level Adversarial Disentanglement (MLAD)
        4. Adaptive Gradient Reversal Scheduling (AGRS)
        5. Mutual Information Minimization via MINE (MIM)
        6. Representation Orthogonality Regularization

    Shape convention: All hidden states use (B, T, D) throughout.
    """

    def __init__(self, config: PrivHSDConfig):
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

    def _load_backbone(self, config: PrivHSDConfig):
        """Load transformer backbone from HuggingFace."""
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

    def get_transformer_outputs(
        self,
        input_ids: torch.Tensor,      # (B, T)
        attention_mask: torch.Tensor,  # (B, T)
    ) -> Dict[str, torch.Tensor]:
        """Extract multi-level representations from the transformer backbone.

        Produces three representation levels for multi-level adversarial
        disentanglement: pooler ([CLS] token), token (mean-pooled all tokens),
        and head (attention-pattern-based aggregate).

        Args:
            input_ids: (B, T) — token indices.
            attention_mask: (B, T) — binary attention mask.

        Returns:
            dict with:
                "pooler_repr": (B, D) — [CLS] pooled representation.
                "token_repr": (B, T, D) — all token hidden states.
                "head_repr": (B, D) — attention-pattern-based representation.
                "last_hidden": (B, T, D) — complete last hidden state.

        Shape invariants:
            T ≤ config.max_seq_len; B ≥ 1; dtype in {float32, bfloat16}.
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
        )

        last_hidden = outputs.last_hidden_state  # (B, T, D)
        B, T, D = last_hidden.shape

        # Pooler: [CLS] token
        pooler_repr = last_hidden[:, 0, :]        # (B, D)

        # Token-level: mean pool over non-padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).float()           # (B, T, 1)
        token_repr = (last_hidden * mask_expanded).sum(dim=1) / (      # (B, D)
            mask_expanded.sum(dim=1) + 1e-10
        )

        # Head-level: use last layer's attention pattern
        if outputs.attentions is not None and len(outputs.attentions) > 0:
            last_attentions = outputs.attentions[-1]     # (B, n_heads, T, T)
            # Mean over source and target positions → (B, n_heads)
            # Then expand to (B, D) by repeating across d_head dimension
            n_heads = last_attentions.size(1)
            d_head = D // n_heads
            head_attn = last_attentions.mean(dim=[2, 3])  # (B, n_heads)
            # Interpolate from n_heads to D for consistency
            head_repr = head_attn.unsqueeze(-1).expand(-1, -1, d_head).reshape(B, D)  # (B, D)
        else:
            head_repr = pooler_repr  # (B, D)

        return {
            "pooler_repr": pooler_repr,
            "token_repr": last_hidden,
            "head_repr": head_repr,
            "last_hidden": last_hidden,
        }

    def forward(
        self,
        input_ids: torch.Tensor,          # (B, T)
        attention_mask: torch.Tensor,      # (B, T)
        hate_labels: Optional[torch.Tensor] = None,    # (B,)
        author_labels: Optional[torch.Tensor] = None,  # (B,)
        alpha: Optional[float] = None,                 # GRL scaling
        step: Optional[int] = None,                    # current step
        return_representations: bool = False,
        return_attentions: bool = False,
        use_checkpoint: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with optional adversarial disentanglement.

        Args:
            input_ids: (B, T) token indices
            attention_mask: (B, T) attention mask
            hate_labels: (B,) optional hate speech labels
            author_labels: (B,) optional author identity labels
            alpha: GRL scaling (from scheduler if None)
            step: current training step (for alpha scheduling)
            return_representations: return pooled representations for analysis
            return_attentions: return attention outputs
            use_checkpoint: use gradient checkpointing

        Returns:
            dict with "hate_logits", "hate_probs", and optionally
            "loss", "hate_loss", "author_loss", "mim_loss", etc.
        """
        # ── Transformer forward ────────────────────────────────────
        transformer_out = self.get_transformer_outputs(input_ids, attention_mask)
        pooler_repr = transformer_out["pooler_repr"]    # (B, D)
        last_hidden = transformer_out["last_hidden"]    # (B, T, D)
        head_repr = transformer_out["head_repr"]        # (B, D)

        # ── Hate speech classification ─────────────────────────────
        hate_logits = self.hate_classifier(pooler_repr)     # (B, num_classes)
        hate_probs = F.softmax(hate_logits.float(), dim=-1).to(hate_logits.dtype)  # (B, num_classes)

        result = {
            "hate_logits": hate_logits,
            "hate_probs": hate_probs,
        }

        # ── Compute losses ─────────────────────────────────────────
        total_loss = torch.tensor(0.0, device=input_ids.device)

        # Hate speech loss
        if hate_labels is not None:
            hate_loss = F.cross_entropy(hate_logits.float(), hate_labels).to(hate_logits.dtype)
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
            result["level_losses"] = {k: v.item() if isinstance(v, torch.Tensor) else v
                                       for k, v in adv_outputs["level_losses"].items()}
            result["alpha"] = current_alpha

            total_loss = total_loss + self.config.disentanglement_weight * adv_outputs["author_loss"]

            # ── Representation orthogonality regularization ──────────
            if self.config.orthogonality_weight > 0 and hate_labels is not None:
                hate_features = self.hate_classifier.get_features(pooler_repr)  # (B, D)
                # Use pooler-level adversary features
                adv_features = adv_outputs["adversary_features"].get(
                    "pooler",
                    list(adv_outputs["adversary_features"].values())[0]
                    if adv_outputs["adversary_features"]
                    else None,
                )

                if hate_features is not None and adv_features is not None:
                    orth_loss = compute_subspace_orthogonality(hate_features, adv_features)
                    result["orth_loss"] = orth_loss
                    total_loss = total_loss + self.config.orthogonality_weight * orth_loss

            # ── Mutual information minimization ──────────────────────
            if self.mim_module is not None and self.config.mim_weight > 0:
                mi_estimate = self.mim_module(
                    pooler_repr, author_labels, maximize=False
                )
                mim_loss = self.config.mim_weight * mi_estimate
                result["mim_loss"] = mim_loss
                total_loss = total_loss + mim_loss

        result["loss"] = total_loss

        # ── Optional return values ─────────────────────────────────
        if return_representations:
            result["representations"] = pooler_repr.detach()  # (B, D)

        if return_attentions:
            result["attentions"] = transformer_out.get("attentions")

        # NaN check in training mode
        if self.training and torch.isnan(total_loss):
            logger.warning("NaN loss detected in PrivHSDModelV2.forward")

        return result

    @torch.no_grad()
    def get_hate_predictions(
        self,
        input_ids: torch.Tensor,      # (B, T)
        attention_mask: torch.Tensor,  # (B, T)
    ) -> torch.Tensor:
        """Get hate speech probabilities for inference.

        Convenience method wrapping forward() with no labels.
        Representations are NOT returned (privacy by default).

        Args:
            input_ids: (B, T) — token indices.
            attention_mask: (B, T) — binary attention mask.

        Returns:
            (B, num_classes) — softmax probabilities.

        Shape invariants:
            B ≥ 1; T ≤ config.max_seq_len.
        """
        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        return outputs["hate_probs"]

    @torch.no_grad()
    def get_representations(
        self,
        input_ids: torch.Tensor,      # (B, T)
        attention_mask: torch.Tensor,  # (B, T)
    ) -> torch.Tensor:
        """Extract pooled representations for privacy analysis.

        Used by the privacy attack suite to evaluate residual identity
        leakage (attribute inference, stylometry). NOT returned during
        standard inference (privacy-preserving default).

        Args:
            input_ids: (B, T) — token indices.
            attention_mask: (B, T) — binary attention mask.

        Returns:
            (B, D) — [CLS] pooled representations (detached).

        Shape invariants:
            B ≥ 1; T ≤ config.max_seq_len.
        """
        outputs = self.forward(
            input_ids=input_ids, attention_mask=attention_mask,
            return_representations=True,
        )
        return outputs["representations"]

    def compute_mine_mi(
        self,
        input_ids: torch.Tensor,      # (B, T)
        attention_mask: torch.Tensor,  # (B, T)
        author_labels: torch.Tensor,  # (B,)
    ) -> float:
        """Compute MINE mutual information estimate.

        Returns scalar MI estimate (higher = more identity leakage).
        """
        if self.mim_module is None:
            return 0.0
        representations = self.get_representations(input_ids, attention_mask)
        with torch.no_grad():
            mi = self.mim_module.estimate_mutual_information(
                representations, author_labels
            )
        return mi.item()


# ═══════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════


def compute_subspace_orthogonality(
    feat_a: torch.Tensor,  # (B, D1)
    feat_b: torch.Tensor,  # (B, D2)
) -> torch.Tensor:
    """Compute cosine similarity between two feature subspaces.

    Encourages hate-relevant and identity-relevant features to occupy
    orthogonal subspaces, making identity information harder to extract.
    If feature dimensions differ, projects the larger one down.

    Args:
        feat_a: (B, D1) — features from first subspace (e.g., hate).
        feat_b: (B, D2) — features from second subspace (e.g., identity).

    Returns:
        Scalar in [0, 1]. Lower = more orthogonal = better disentanglement.

    Shape invariants:
        Batch dimension B must match for both inputs.
        dtype in {float32, bfloat16}. Cast to float32 internally.
    """
    if feat_a.size(-1) != feat_b.size(-1):
        # Project to common dimension (smaller of the two)
        min_dim = min(feat_a.size(-1), feat_b.size(-1))
        if feat_a.size(-1) > min_dim:
            feat_a = feat_a[..., :min_dim]
        if feat_b.size(-1) > min_dim:
            feat_b = feat_b[..., :min_dim]

    a_norm = F.normalize(feat_a.float(), dim=-1)   # (B, D)
    b_norm = F.normalize(feat_b.float(), dim=-1)   # (B, D)
    cos_sim = (a_norm * b_norm).sum(dim=-1).abs().mean()  # scalar
    return cos_sim


def count_params(model: nn.Module) -> None:
    """Print parameter counts for a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total:,} | Trainable: {trainable:,}")


# ═══════════════════════════════════════════════════════════════════════════
# Backward-Compatible v1 Alias
# ═══════════════════════════════════════════════════════════════════════════


class PrivHSDModel(PrivHSDModelV2):
    """Backward-compatible alias for PrivHSDModelV2.

    Allows existing v1 training/evaluation code to work with the v2
    architecture using the same constructor signature.
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
        config = PrivHSDConfig(
            model_name=model_name,
            model_type=model_type,
            num_hate_classes=num_hate_classes,
            num_authors=num_authors,
            alpha_final=adversarial_alpha,
            disentanglement_weight=disentanglement_weight,
            dropout=hidden_dropout,
            adversary_dropout=hidden_dropout + 0.1,
        )
        super().__init__(config)


# ═══════════════════════════════════════════════════════════════════════════
# Standalone Smoke Test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Minimal config for smoke test
    cfg = PrivHSDConfig(
        model_name="albert-base-v2",
        d_model=768,
        num_authors=10,
        disentanglement_weight=0.3,
        mim_weight=0.1,
        orthogonality_weight=0.05,
    )

    model = PrivHSDModelV2(cfg).to(device)
    model.eval()

    B, T = 2, 64
    input_ids = torch.randint(0, 100, (B, T), device=device)
    attention_mask = torch.ones(B, T, device=device)
    hate_labels = torch.randint(0, 2, (B,), device=device)
    author_labels = torch.randint(0, 10, (B,), device=device)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            hate_labels=hate_labels,
            author_labels=author_labels,
        )

    assert out["hate_logits"].shape == (B, 2), f"Bad hate_logits shape: {out['hate_logits'].shape}"
    assert out["hate_probs"].shape == (B, 2), f"Bad hate_probs shape: {out['hate_probs'].shape}"
    assert "loss" in out, "Missing loss in output"
    assert "author_loss" in out, "Missing author_loss in output"
    print(f"[SMOKE OK] PrivHSDModelV2 forward: logits={out['hate_logits'].shape}, loss={out['loss'].item():.4f}")

    count_params(model)
