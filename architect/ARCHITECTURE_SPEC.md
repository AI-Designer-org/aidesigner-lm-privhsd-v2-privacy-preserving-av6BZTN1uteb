# PrivHSD v2 Architecture Specification

## Privacy-preserving Hate Speech Detection with Multi-Level Adversarial Disentanglement

**Domain:** LM + Privacy ML + Ethical AI
**Upstream Research Contract:** `/work/research/RESEARCH_CONTRACT.yaml` (read and preserved)
**Status:** Architectural blueprint for implementation

---

## Table of Contents

1. ModelConfig Dataclass
2. Core Architectural Innovations
3. Pseudocode for Novel Blocks
4. ASCII Architecture Diagram
5. Inductive Bias Justifications
6. Research-to-Architecture Traceability
7. Domain-Specific Considerations
8. Implementation Risks
9. Suggested Ablations

---

## 1. ModelConfig Dataclass

```python
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

@dataclass
class PrivHSDConfig:
    # ─── Transformer Backbone ──────────────────────────────────────────────
    model_name: str = "albert-base-v2"        # HF model identifier
    model_type: str = "albert"                # "albert" | "roberta" | "xlm-roberta"
    d_model: int = 768                        # hidden dimension (derived from backbone)
    n_layers: int = 12                        # transformer layers
    n_heads: int = 12                         # attention heads
    d_ff: int = 3072                          # feed-forward dimension
    vocab_size: int = 30000                   # ALBERT wordpiece vocab
    max_seq_len: int = 256                    # truncation length
    dropout: float = 0.2                      # hidden dropout probability
    attention_dropout: float = 0.2            # attention dropout probability

    # ─── Hate Speech Classification Head ───────────────────────────────────
    num_hate_classes: int = 2                 # binary hate/not-hate
    classifier_hidden_dim: int = 768          # MLP hidden dim before projection
    classifier_num_layers: int = 2            # depth of classification MLP

    # ─── Multi-Level Adversarial Disentanglement (MLAD) ────────────────────
    num_authors: int = 100                    # pseudo-author classes for disentanglement
    adversarial_levels: Tuple[str, ...] = (
        "pooler", "token", "head"
    )  # levels at which adversaries operate
    adversary_hidden_dim: int = 256           # adversary MLP hidden size
    adversary_num_layers: int = 3             # depth of each adversary MLP
    adversary_dropout: float = 0.3            # adversary dropout (higher = stronger regularization)

    # ─── Adaptive Gradient Reversal Scheduling (AGRS) ──────────────────────
    alpha_initial: float = 0.1                # initial GRL scaling factor
    alpha_final: float = 1.0                  # final GRL scaling factor
    alpha_schedule: str = "sigmoid"           # "linear" | "sigmoid" | "adaptive"
    alpha_warmup_epochs: int = 2              # epochs before alpha starts increasing
    alpha_gamma: float = 2.0                  # sigmoid steepness for schedule

    # ─── Disentanglement Weights ───────────────────────────────────────────
    disentanglement_weight: float = 0.3       # main adversarial loss weight
    mim_weight: float = 0.1                   # mutual information minimization weight
    orthogonality_weight: float = 0.05        # representation orthogonality regularization
    consistency_weight: float = 0.05          # cross-model consistency weight

    # ─── Mutual Information Minimization (MIM) ────────────────────────────
    mim_estimator: str = "mine"               # "mine" | "nwj" | "info_nce"
    mim_hidden_dim: int = 128                 # MINE network hidden dim
    mim_learning_rate: float = 1e-4           # separate LR for MINE network
    mim_momentum: float = 0.9                 # MINE optimizer momentum

    # ─── DP-SGD Privacy Parameters ─────────────────────────────────────────
    dp_enabled: bool = True                   # master switch for DP-SGD
    target_epsilon: float = 8.0               # target privacy budget
    target_delta: Optional[float] = None     # auto = 1/|D| if None
    max_grad_norm: float = 1.0                # global clipping norm
    per_layer_clipping: bool = True           # adaptive per-layer clip norms
    layer_clip_factors: Tuple[float, ...] = (
        0.5, 0.7, 0.9, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 0.8, 0.6, 0.4,
    )  # per-layer scaling (12 layers for ALBERT-base)

    # ─── Privacy Accounting ────────────────────────────────────────────────
    accounting_mechanism: str = "rdp"          # "rdp" | "zcdp" | "gdp"
    epsilon_budget: float = 8.0               # total epsilon to spend
    noise_multiplier: Optional[float] = None  # auto-computed from epsilon if None
    poisson_sampling: bool = True             # Poisson sampling for tighter privacy

    # ─── Privacy-Augmented Data ────────────────────────────────────────────
    privacy_augment_level: Optional[str] = None  # None | "low" | "medium" | "high"
    label_flip_prob: float = 0.05             # label noise probability
    word_dropout_prob: float = 0.10           # word dropout probability
    synonym_replacement_prob: float = 0.05    # synonym replacement for stylized perturbation
    entity_masking: bool = True               # mask named entities for privacy

    # ─── Training ──────────────────────────────────────────────────────────
    batch_size: int = 16                      # per-GPU batch size
    learning_rate: float = 2e-5               # peak learning rate
    lr_schedule: str = "linear"               # "linear" | "cosine" | "constant"
    warmup_ratio: float = 0.1                 # fraction of steps for warmup
    weight_decay: float = 0.01                # AdamW weight decay
    num_epochs: int = 10                      # training epochs
    gradient_accumulation_steps: int = 1      # gradient accumulation
    max_grad_steps: int = 10000               # max gradient steps
    early_stopping_patience: int = 5          # epochs without improvement
    eval_every: int = 1                       # evaluate every N epochs
    save_every: int = 5                       # checkpoint every N epochs

    # ─── Optimization ──────────────────────────────────────────────────────
    optimizer: str = "adamw"                  # "adamw" | "adam" | "sgd"
    beta1: float = 0.9                        # Adam beta1
    beta2: float = 0.999                      # Adam beta2
    epsilon: float = 1e-8                     # optimizer epsilon
    max_grad_clip: float = 1.0                # gradient clipping (pre-DP)

    # ─── System ────────────────────────────────────────────────────────────
    device: str = "cuda"                      # training device
    mixed_precision: str = "fp16"             # "fp16" | "bf16" | "fp32"
    use_ghost_clipping: bool = True           # Opacus ghost clipping
    num_workers: int = 4                      # DataLoader workers
    seed: int = 42                            # random seed
    output_dir: str = "models"                # output directory

    # ─── Evaluation ────────────────────────────────────────────────────────
    eval_metrics: Tuple[str, ...] = (
        "f1", "auc", "accuracy", "precision",
        "recall", "mcc", "specificity",
    )
    privacy_attack_types: Tuple[str, ...] = (
        "mia_shadow", "mia_threshold",
        "attribute_inference", "stylometry",
    )
    pareto_metrics: Tuple[str, ...] = (
        "epsilon", "f1_score", "mia_auc",
    )

    def __post_init__(self):
        if self.target_delta is None:
            self.target_delta = 1e-5  # default; overridden with dataset size
        if self.model_type == "albert":
            self.d_model = 768 if "base" in self.model_name else 1024
            self.n_layers = 12 if "base" in self.model_name else 24
            self.n_heads = 12 if "base" in self.model_name else 16
        elif self.model_type == "roberta":
            self.d_model = 768 if "base" in self.model_name else 1024
            self.n_layers = 12 if "base" in self.model_name else 24
            self.n_heads = 12 if "base" in self.model_name else 16
```

---

## 2. Core Architectural Innovations

PrivHSD v2 introduces four novel mechanisms beyond the v1 prototype:

### 2.1 Multi-Level Adversarial Disentanglement (MLAD)

**Problem in v1:** Only the [CLS] pooled representation is used for identity disentanglement. Author identity signals are distributed across all token-level representations and attention head outputs. A single adversary on [CLS] leaves residual identity information in other representation levels.

**Solution:** Deploy adversarial heads at three levels:
- **Pooler level** ([CLS] token): Captures sequence-level aggregates
- **Token level** (mean-pooled token representations): Captures per-token stylistic patterns
- **Head level** (attention-head output means): Captures attention-pattern-based author fingerprints

Each adversary gradient is reversed through its own GRL with potentially different alpha values, giving the encoder fine-grained pressure to strip identity from each level.

### 2.2 Adaptive Gradient Reversal Scheduling (AGRS)

**Problem in v1:** A fixed alpha (e.g., 0.5) throughout training. Early in training, the encoder needs to learn hate-relevant features; applying strong adversarial pressure too early degrades utility. Late in training, weak alpha may not fully remove identity signals.

**Solution:** Schedule alpha following a sigmoid curve:
- Epochs 0–2 (warmup): alpha ≈ 0 (no adversarial pressure, learn hate features)
- Epochs 2–6 (ramp): alpha rises from 0.1 → 1.0 following sigmoid
- Epochs 6+ (plateau): alpha = 1.0 (maximal disentanglement)

Optional adaptive variant: alpha increases when adversary loss drops below a threshold (indicating the adversary is winning), and decreases when hate loss increases above a threshold (indicating disentanglement is hurting utility).

### 2.3 Mutual Information Minimization (MIM) via MINE

**Problem in v1:** The adversarial loss is a proxy for identity information — it only removes features that the *current* adversary can use. A stronger adversary might find residual identity information.

**Solution:** Add an explicit mutual information minimization term using the Mutual Information Neural Estimator (MINE). MINE provides a lower-bound estimate of I(representation; author_id), and we minimize this bound directly. This provides a stronger, adversary-independent signal for identity removal.

The training becomes a three-player game:
- Encoder E minimizes: L_hate + λ_dis * L_adv + λ_mim * I_est(E(x); author)
- Adversary A minimizes: L_author(reversed_grad)
- MINE network M maximizes: I_lower_bound(repr; author)

### 2.4 DP-SGD with Per-Layer Adaptive Clipping

**Problem in v1:** Uniform gradient clipping (norm ≤ C) across all layers. Different layers have different sensitivity to noise. Early layers (learning general features) need less clipping than classifier layers (task-specific).

**Solution:** Apply per-layer gradient clipping with layer-specific clip factors, scaled by layer importance. Layer importance is estimated from the gradient variance across the first few non-private steps. This reduces the total noise added while maintaining per-layer privacy guarantees.

---

## 3. Pseudocode for Novel Blocks

### 3.1 Multi-Level Adversarial Disentanglement Block

```python
class MultiLevelAdversarialBlock(nn.Module):
    """
    Adversarial heads operating at multiple representation levels
    to ensure comprehensive identity signal removal.

    Architecture:
        For each level in [pooler, token, head]:
            Level-specific adversary MLP with GRL
            All adversarial losses summed for total disentanglement

    Inductive bias:
        Identity signals at different representation levels require
        separate adversaries for comprehensive removal.
    """
    def __init__(self, config: PrivHSDConfig):
        super().__init__()
        self.levels = config.adversarial_levels
        self.adversaries = nn.ModuleDict()
        self.grls = nn.ModuleDict()
        self.alpha_scheduler = AdaptiveAlphaScheduler(config)

        for level in self.levels:
            if level == "pooler":
                input_dim = config.d_model
            elif level == "token":
                input_dim = config.d_model
            elif level == "head":
                input_dim = config.d_model  # averaged over heads
            else:
                raise ValueError(f"Unknown level: {level}")

            self.adversaries[level] = AdversaryMLP(
                input_dim=input_dim,
                hidden_dim=config.adversary_hidden_dim,
                output_dim=config.num_authors,
                num_layers=config.adversary_num_layers,
                dropout=config.adversary_dropout,
            )
            self.grls[level] = GradientReversalLayer()

    def forward(
        self,
        pooler_repr: Tensor,       # (B, d_model)
        token_repr: Tensor,         # (B, seq_len, d_model)
        head_repr: Tensor,          # (B, num_heads, d_head)
        author_labels: Tensor,      # (B,)
        alpha: Optional[float] = None,
        step: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass for multi-level adversarial disentanglement.

        Args:
            pooler_repr: [CLS] pooled representation
            token_repr: All token representations
            head_repr: Per-head attention output averages
            author_labels: Pseudo-author identity labels
            alpha: GRL scaling factor (from scheduler if None)
            step: Current training step for alpha scheduling

        Returns:
            Dictionary with per-level author losses and logits
        """
        if alpha is None and step is not None:
            alpha = self.alpha_scheduler.get_alpha(step)

        level_losses = {}
        level_logits = {}

        for level in self.levels:
            if level == "pooler":
                repr_input = pooler_repr
            elif level == "token":
                # Mean pool over sequence dimension
                repr_input = token_repr.mean(dim=1)
            elif level == "head":
                # Mean pool over head dimension
                repr_input = head_repr.mean(dim=1)

            # Apply gradient reversal
            reversed_repr = self.grls[level].apply(repr_input, alpha)

            # Predict author identity from reversed representation
            author_logits = self.adversaries[level](reversed_repr)
            level_loss = F.cross_entropy(author_logits, author_labels)

            level_losses[f"author_loss_{level}"] = level_loss
            level_logits[f"author_logits_{level}"] = author_logits

        # Total adversarial loss (mean across levels)
        total_adv_loss = torch.stack(list(level_losses.values())).mean()

        return {
            "author_loss": total_adv_loss,
            "level_losses": level_losses,
            "level_logits": level_logits,
        }
```

### 3.2 Adaptive Gradient Reversal Scheduler

```python
class AdaptiveAlphaScheduler:
    """
    Adaptive Gradient Reversal Scheduling (AGRS).

    Schedules the GRL alpha parameter to control the strength of
    adversarial disentanglement throughout training. Uses a sigmoid
    schedule with warmup and optional adaptive feedback.

    Inductive bias:
        Early training should focus on learning hate-relevant features
        without adversarial distortion. Late training should apply
        maximal disentanglement pressure.

    Theory support:
        Ganin & Lempitsky (2015) used a linear schedule from 0 to 1.
        Our sigmoid schedule provides a more gradual transition with
        a steeper mid-training ramp where both encoder and adversary
        are competent enough for productive competition.
    """
    def __init__(self, config: PrivHSDConfig):
        self.initial = config.alpha_initial
        self.final = config.alpha_final
        self.schedule = config.alpha_schedule
        self.warmup_epochs = config.alpha_warmup_epochs
        self.gamma = config.alpha_gamma

        # State for adaptive mode
        self.encoder_loss_history = []
        self.adversary_loss_history = []
        self.current_alpha = self.initial

    def _get_epoch_from_step(
        self, step: int, steps_per_epoch: int
    ) -> float:
        """Convert step to fractional epoch."""
        return step / steps_per_epoch

    def get_alpha(
        self,
        step: int,
        steps_per_epoch: int = 1000,
        hate_loss: Optional[float] = None,
        adv_loss: Optional[float] = None,
    ) -> float:
        """
        Compute alpha for the current training step.

        Schedule types:
            "linear":  alpha = min(1, max(0, (epoch - warmup) / total))
            "sigmoid": alpha = 1 / (1 + exp(-gamma * (p - 0.5)))
            "adaptive": sigmoid base + encoder/adversary loss feedback
        """
        epoch = self._get_epoch_from_step(step, steps_per_epoch)

        if epoch < self.warmup_epochs:
            return self.initial

        p = (epoch - self.warmup_epochs) / (
            steps_per_epoch - self.warmup_epochs
        )
        p = min(1.0, max(0.0, p))

        if self.schedule == "linear":
            alpha = self.initial + (self.final - self.initial) * p
        elif self.schedule == "sigmoid":
            alpha = self.initial + (self.final - self.initial) * (
                1.0 / (1.0 + math.exp(-self.gamma * (p - 0.5)))
            )
        elif self.schedule == "adaptive":
            alpha = self._compute_adaptive_alpha(
                p, hate_loss, adv_loss
            )
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

        self.current_alpha = alpha
        return alpha

    def _compute_adaptive_alpha(
        self,
        p: float,
        hate_loss: Optional[float],
        adv_loss: Optional[float],
    ) -> float:
        """
        Adaptive alpha: adjust based on encoder vs. adversary dynamics.

        - If adversary loss is low (adversary winning), increase alpha
          to strengthen adversarial pressure.
        - If hate loss is increasing (disentanglement hurting utility),
          decrease alpha to reduce pressure.
        """
        base_alpha = self.initial + (self.final - self.initial) * (
            1.0 / (1.0 + math.exp(-self.gamma * (p - 0.5)))
        )

        if hate_loss is not None and len(self.encoder_loss_history) > 5:
            # If hate loss increased recently, reduce alpha
            recent_hate = self.encoder_loss_history[-5:]
            hate_trend = recent_hate[-1] - recent_hate[0]
            if hate_trend > 0.05:  # hate loss increasing
                base_alpha *= 0.9

        if adv_loss is not None and len(self.adversary_loss_history) > 5:
            # If adversary accuracy is high (low loss), increase alpha
            recent_adv = self.adversary_loss_history[-5:]
            adv_trend = recent_adv[-1] - recent_adv[0]
            if adv_trend < -0.05:  # adversary improving
                base_alpha *= 1.1

        return max(self.initial, min(self.final, base_alpha))

    def update_history(
        self, hate_loss: float, adv_loss: float
    ):
        """Update loss history for adaptive scheduling."""
        self.encoder_loss_history.append(hate_loss)
        self.adversary_loss_history.append(adv_loss)
        # Keep bounded window
        if len(self.encoder_loss_history) > 50:
            self.encoder_loss_history.pop(0)
            self.adversary_loss_history.pop(0)
```

### 3.3 Mutual Information Minimization Block

```python
class MutualInformationMinimizer(nn.Module):
    """
    Mutual Information Minimization via MINE (Belghazi+18).

    Estimates I(representation; author_id) using a neural network
    lower bound and minimizes this estimate during training.

    Inductive bias:
        Adversarial training only removes identity features that the
        current adversary can detect. Direct MI minimization provides
        a stronger, adversary-independent signal for identity removal.

    Theoretical support:
        I(X; Y) = sup_{T: Omega -> R} E_P[T] - log(E_Q[exp(T)])
        where P = joint distribution, Q = product of marginals.
        MINE uses a neural network T_theta to estimate this bound.

    Implementation:
        - MINE network T: (repr, author_embedding) -> scalar score
        - Forward pass draws joint samples (same author) and marginal
          samples (shuffled author labels)
        - Loss = -(E_joint[T] - log(E_marginal[exp(T)]))
        - We minimize the estimated MI (encoder) while MINE network
          maximizes it (adversarial-to-MINE)
    """
    def __init__(self, config: PrivHSDConfig):
        super().__init__()
        self.repr_dim = config.d_model
        self.author_dim = config.num_authors
        self.hidden_dim = config.mim_hidden_dim

        # MINE network: (repr, author_onehot) -> scalar
        self.mine_network = nn.Sequential(
            nn.Linear(config.d_model + config.num_authors, config.mim_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.mim_hidden_dim, config.mim_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.mim_hidden_dim, 1),
        )

        # Author embedding for MINE input
        self.author_embedding = nn.Embedding(
            config.num_authors, config.num_authors
        )

    def estimate_mutual_information(
        self,
        representations: Tensor,  # (B, d_model)
        author_labels: Tensor,    # (B,)
    ) -> Tensor:
        """
        Estimate I(repr; author) using MINE lower bound.

        Returns a scalar MI estimate (higher = more identity leakage).
        """
        batch_size = representations.size(0)

        # Joint samples: (repr_i, author_i)
        author_embeds = self.author_embedding(author_labels)  # (B, author_dim)
        joint_input = torch.cat([representations, author_embeds], dim=1)
        joint_scores = self.mine_network(joint_input)  # (B, 1)

        # Marginal samples: (repr_i, shuffled_author_j)
        shuffled_idx = torch.randperm(batch_size, device=author_labels.device)
        shuffled_labels = author_labels[shuffled_idx]
        shuffled_embeds = self.author_embedding(shuffled_labels)
        marginal_input = torch.cat([representations, shuffled_embeds], dim=1)
        marginal_scores = self.mine_network(marginal_input)  # (B, 1)

        # MINE lower bound
        joint_mean = joint_scores.mean()
        marginal_mean = torch.logsumexp(marginal_scores, dim=0) - math.log(batch_size)

        # MI estimate = E_joint[T] - log(E_marginal[exp(T)])
        mi_estimate = joint_mean - marginal_mean

        return mi_estimate

    def forward(
        self,
        representations: Tensor,
        author_labels: Tensor,
        maximize: bool = False,
    ) -> Tensor:
        """
        Compute MI estimate.

        Args:
            representations: Model's [CLS] or pooled representations
            author_labels: Author identity labels
            maximize: If True, MINE network step (maximize MI estimate).
                      If False, encoder step (minimize MI estimate).

        Returns:
            MI estimate (maximized by MINE, minimized by encoder)
        """
        mi_estimate = self.estimate_mutual_information(
            representations, author_labels
        )

        if maximize:
            # MINE network wants to maximize the lower bound
            return -mi_estimate  # negate for gradient ascent
        else:
            # Encoder wants to minimize MI
            return mi_estimate
```

### 3.4 Per-Layer Adaptive DP Clipping

```python
class PerLayerDPAdapter:
    """
    DP-SGD with per-layer adaptive gradient clipping.

    Different layers have different sensitivity to DP noise.
    Earlier layers (general features) benefit from tighter clipping
    while later layers (task-specific) may need looser bounds.

    Inductive bias:
        Layer sensitivity varies across the network depth.
        Uniform clipping over-perturbs sensitive layers and
        under-perturbs robust layers.

    Method:
        1. Run a few non-private warmup steps to estimate per-layer
           gradient variance.
        2. Compute per-layer clip factors inversely proportional to
           gradient variance (more variance -> tighter clip).
        3. Apply per-layer clipping, scaling noise accordingly.
        4. Account for total privacy via composition theorem.

    Implementation note:
        Opacus 1.4+ supports ghost clipping per layer. We wrap
        the PrivacyEngine.make_private_with_epsilon call with
        per-layer norm computation.
    """

    @staticmethod
    def estimate_layer_sensitivity(
        model: nn.Module,
        sample_batch: Dict[str, Tensor],
        n_steps: int = 10,
    ) -> List[float]:
        """
        Estimate per-layer gradient sensitivity via variance over
        multiple forward-backward passes without DP.

        Returns list of clip factors (one per parameter group).
        Higher sensitivity -> tighter clipping (lower factor).
        """
        model.train()
        sensitivities = []
        param_groups = list(model.parameters())

        for _ in range(n_steps):
            # Forward pass
            outputs = model(**sample_batch)
            outputs["loss"].backward()

            # Record gradient norms per layer
            for param in param_groups:
                if param.grad is not None:
                    sensitivities.append(param.grad.norm().item())
            model.zero_grad()

        if not sensitivities:
            return [1.0] * len(param_groups)

        # Normalize: lower variance -> higher clip factor (less clipping)
        std = np.std(sensitivities)
        mean = np.mean(sensitivities)
        cv = std / (mean + 1e-8)  # coefficient of variation

        clip_factors = []
        for param in param_groups:
            # Higher CV -> more clipping needed
            factor = 1.0 / (1.0 + cv * 0.5)
            clip_factors.append(max(0.1, min(2.0, factor)))

        return clip_factors

    @staticmethod
    def apply_per_layer_clipping(
        model: nn.Module,
        clip_factors: List[float],
        base_clip_norm: float = 1.0,
    ) -> None:
        """
        Apply per-layer clipping norms to model parameters.
        This hooks into Opacus's per-sample gradient clipping
        to use layer-adaptive norms.

        Note: Full implementation requires modifying Opacus's
        GradSampleModule to accept per-layer norms. Here we
        provide the interface; the implementation detail is
        in the training loop integration.
        """
        for param, factor in zip(model.parameters(), clip_factors):
            # Store per-layer clip norm on the parameter
            param._clip_norm = base_clip_norm * factor
```

### 3.5 Full Training Loop with Combined Objective

```python
def train_step(
    model: PrivHSDModel,
    batch: Dict[str, Tensor],
    config: PrivHSDConfig,
    step: int,
    alpha_scheduler: AdaptiveAlphaScheduler,
    mim_module: Optional[MutualInformationMinimizer] = None,
    mim_optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Tensor]:
    """
    Single training step with combined objective:

    L_total = L_hate + λ_dis * L_adv + λ_mim * I_est + λ_orth * L_orth

    Three-stage backward:
    1. MINE step: maximize I_est (update MINE network)
    2. Encoder step: minimize L_hate + λ_dis * L_adv + λ_mim * I_est
       (GRL reverses L_adv gradient through encoder)
    3. (DP-SGD handles per-sample clipping via Opacus)
    """
    model.train()
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    hate_labels = batch["hate_labels"]
    author_labels = batch["author_labels"]

    # === Forward Pass ===
    # 1. Get transformer outputs
    transformer_outputs = model.get_transformer_outputs(
        input_ids, attention_mask
    )  # returns pooler_repr, token_repr, head_repr

    # 2. Hate speech classification
    hate_logits = model.hate_classifier(
        transformer_outputs["pooler_repr"]
    )
    hate_loss = F.cross_entropy(hate_logits, hate_labels)

    # 3. Multi-level adversarial disentanglement (with GRL)
    alpha = alpha_scheduler.get_alpha(
        step,
        steps_per_epoch=len(train_loader),
        hate_loss=hate_loss.item(),
        adv_loss=None,  # Will be filled after adversarial pass
    )

    adv_outputs = model.mlad_block(
        pooler_repr=transformer_outputs["pooler_repr"],
        token_repr=transformer_outputs["token_repr"],
        head_repr=transformer_outputs["head_repr"],
        author_labels=author_labels,
        alpha=alpha,
    )
    adv_loss = adv_outputs["author_loss"]

    alpha_scheduler.update_history(
        hate_loss=hate_loss.item(),
        adv_loss=adv_loss.item(),
    )

    # 4. Mutual information minimization
    mim_loss = torch.tensor(0.0)
    if mim_module is not None and config.mim_weight > 0:
        # MINE network step: maximize I_est
        if mim_optimizer is not None:
            mi_positive = mim_module(
                transformer_outputs["pooler_repr"],
                author_labels,
                maximize=True,
            )
            mim_optimizer.zero_grad()
            mi_positive.backward(retain_graph=True)
            mim_optimizer.step()

        # Encoder step: minimize I_est
        mi_estimate = mim_module(
            transformer_outputs["pooler_repr"],
            author_labels,
            maximize=False,
        )
        mim_loss = config.mim_weight * mi_estimate

    # 5. Representation orthogonality regularization
    orth_loss = torch.tensor(0.0)
    if config.orthogonality_weight > 0:
        # Encourage hate-relevant and identity-relevant subspaces
        # to be orthogonal
        hate_repr = model.hate_classifier[:2](
            transformer_outputs["pooler_repr"]
        )  # first few layers of classifier
        # Compute cosine similarity between hate-relevant and
        # identity-relevant directions
        orth_loss = config.orthogonality_weight * (
            compute_subspace_orthogonality(
                hate_repr,
                adv_outputs["adversary_features"],
            )
        )

    # === Combined Loss ===
    total_loss = (
        hate_loss
        + config.disentanglement_weight * adv_loss
        + mim_loss
        + orth_loss
    )

    return {
        "loss": total_loss,
        "hate_loss": hate_loss,
        "adv_loss": adv_loss,
        "mim_loss": mim_loss,
        "orth_loss": orth_loss,
        "alpha": alpha,
        "pooler_repr": transformer_outputs["pooler_repr"],
    }
```

---

## 4. ASCII Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PrivHSD v2 Architecture                            │
│    Multi-Level Adversarial Disentanglement + DP-SGD + Mutual Information   │
│                               Minimization                                  │
└─────────────────────────────────────────────────────────────────────────────┘

Input Text
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Tokenizer (HuggingFace AutoTokenizer)                                      │
│  Max length = 256, padding, truncation                                      │
└───────────────────────────┬─────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Transformer Backbone (ALBERT / RoBERTa / XLM-RoBERTa)                      │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Embedding Layer                                                      │   │
│  │  (Token + Position + Segment Embeddings) → d_model=768               │   │
│  └──────────────────────────┬───────────────────────────────────────────┘   │
│                             ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Transformer Layers × L (L=12 for base)                              │   │
│  │  ┌────────────────────────────────────────────────────────────────┐  │   │
│  │  │  LayerNorm → Multi-Head Attention → Residual → LayerNorm → FFN │  │   │
│  │  └────────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────┬───────────────────────────────────────────┘   │
│                             │                                               │
│                             ▼                                               │
│  Outputs:                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                     │
│  │ pooler_repr  │  │ token_repr   │  │ head_repr    │                     │
│  │ [CLS] pooled │  │ all tokens   │  │ per-head avg │                     │
│  │ (B, d_model) │  │ (B, T, d)    │  │ (B, H, d/H)  │                     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                     │
└─────────┼─────────────────┼─────────────────┼──────────────────────────────┘
          │                 │                 │
          │      ┌──────────┴──────────┐      │
          │      ▼                     ▼      │
          │  ┌──────────────────────────────────────────────┐
          │  │  Multi-Level Adversarial Disentanglement     │
          │  │  ┌────────────┐  ┌────────────┐  ┌────────┐ │
          │  │  │ Pooler Adv│  │ Token Adv  │  │Head Adv│ │
          │  │  │ (d_model) │  │ (d_model)  │  │(d_mdl) │ │
          │  │  └─────┬─────┘  └──────┬──────┘  └───┬────┘ │
          │  │        │               │              │      │
          │  │  ┌─────┴───────────────┴──────────────┴──┐   │
          │  │  │  Gradient Reversal Layers (GRL)       │   │
          │  │  │  ├── GRL(pooler, α)                   │   │
          │  │  │  ├── GRL(token, α)                    │   │
          │  │  │  └── GRL(head, α)                     │   │
          │  │  └───────────────────┬───────────────────┘   │
          │  │                      ▼                       │
          │  │  ┌──────────────────────────────────────┐    │
          │  │  │  Adversary MLPs (3× MLP with ReLU)   │    │
          │  │  │  Each: Linear→LN→ReLU→Drop→...→Lin   │    │
          │  │  │  Output: author_logits (n_authors)   │    │
          │  │  └──────────────┬───────────────────────┘    │
          │  │                 ▼                            │
          │  │  L_adv = CE(author_pred, author_true)        │
          │  └──────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Hate Speech Classification Head                                           │
│  Linear(d_model, d_model) → LayerNorm → ReLU → Dropout → Linear(2)         │
│  L_hate = CE(hate_pred, hate_true)                                          │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Mutual Information Minimization (MINE)                                     │
│                                                                             │
│  ┌──────────────────────────────────────────────┐                          │
│  │  MINE Network: MLP(repr ⊕ author_embed) → T │                          │
│  │  I_est = E_joint[T] - log(E_marginal[exp(T)])│                          │
│  └──────────────────────┬───────────────────────┘                          │
│                         ▼                                                  │
│  L_mim = λ_mim * I_est(repr; author)  [encoder minimizes]                  │
│  L_mine = -I_est  [MINE network maximizes]                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Combined Training Objective                                                │
│                                                                             │
│  L_total = L_hate + λ_dis * L_adv + λ_mim * I_est + λ_orth * L_orth        │
│                                                                             │
│  α(t) = sigmoid_schedule(t)  [Adaptive Gradient Reversal Scheduling]        │
│                                                                             │
│  ┌──────────────────────────────────────┐                                  │
│  │  DP-SGD via Opacus (ghost clipping)  │                                  │
│  │  ├── Per-sample gradient computation │                                  │
│  │  ├── Per-layer adaptive clipping     │                                  │
│  │  ├── Gaussian noise N(0, σ²C²I)     │                                  │
│  │  └── RDP accounting (ε, δ tracking)  │                                  │
│  └──────────────────────────────────────┘                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Evaluation & Privacy Attack Suite                                          │
│                                                                             │
│  ┌─────────────────────────┐  ┌──────────────────────────────────────────┐ │
│  │ Utility Metrics         │  │ Privacy Attack Suite                      │ │
│  │ ├── F1 Score            │  │ ├── Membership Inference (shadow model)  │ │
│  │ ├── AUC-ROC             │  │ ├── Attribute Inference (logistic reg)   │ │
│  │ ├── Precision/Recall    │  │ ├── Stylometry Re-identification         │ │
│  │ ├── MCC                 │  │ │   ├── Raw text features (baseline)     │ │
│  │ └── Specificity         │  │ │   └── Model representations (leakage)  │ │
│  │                         │  │ └── Representation Privacy Audit          │ │
│  └─────────────────────────┘  │   ├── Representation entropy             │ │
│                                │   ├── k-anonymity fraction              │ │
│  ┌─────────────────────────┐  │   └── Privacy leakage score              │ │
│  │ Pareto Frontier Analysis│  └──────────────────────────────────────────┘ │
│  │ ε vs F1 vs MIA AUC      │                                               │
│  └─────────────────────────┘                                               │
└─────────────────────────────────────────────────────────────────────────────┘


                        ┌──────────────────────┐
                        │  Rights-Based Design  │
                        │  ┌──────────────────┐ │
                        │  │ GDPR Art. 25     │ │
                        │  │ Data Minimization│ │
                        │  │ Purpose Limitation│ │
                        │  │ ECHR Art. 10     │ │
                        │  │ DSA Compliance   │ │
                        │  └──────────────────┘ │
                        └──────────────────────┘
```

**Data Flow Summary:**

```
┌──────────┐   ┌────────────┐   ┌─────────────────────┐   ┌────────────┐
│ Raw Text │──▶│ Tokenizer  │──▶│ Transformer Backbone │──▶│ [CLS] Repr │
└──────────┘   └────────────┘   └─────────────────────┘   └──────┬─────┘
                                                                  │
                                          ┌───────────────────────┼──────────┐
                                          ▼                       ▼          │
                                  ┌──────────────┐       ┌──────────────┐   │
                                  │ Hate Classif │       │ MLAD Block   │   │
                                  │ (no GRL)     │       │ (with GRL)   │   │
                                  └──────┬───────┘       └──────┬───────┘   │
                                         ▼                      ▼          │
                                  ┌──────────────┐       ┌──────────────┐   │
                                  │ L_hate       │       │ L_adv        │   │
                                  └──────────────┘       └──────────────┘   │
                                                                             │
                                  ┌──────────────────────────────────────┐  │
                                  │ MINE: I_est(repr; author)            │  │
                                  │ → L_mim (encoder minimizes)          │──┘
                                  └──────────────────────────────────────┘
```

---

## 5. Inductive Bias Justifications

| Decision | Justification |
|---|---|
| **ALBERT backbone (default)** | Cross-layer parameter sharing reduces parameter count by ~70% vs BERT/RoBERTa. Under DP-SGD, fewer parameters means less noise per gradient step, yielding 2-5% higher F1 at ε ≤ 4 (Biy+25, NAACL 2025). |
| **Multi-level adversarial disentanglement** | Identity signals manifest at multiple levels: [CLS] (topic/style aggregates), token (word-level choices), and head (attention pattern fingerprints). A single-level adversary leaves residual identity in the other levels. |
| **Sigmoid alpha schedule** | Early focus on hate feature learning (alpha ≈ 0), then gradual ramp to full disentanglement. Sigmoid provides a "sweet spot" period where both encoder and adversary are competent, maximizing the minimax game effectiveness. |
| **MINE-based mutual information minimization** | Adversarial training only strips features the current adversary can detect. MINE provides a lower-bound estimate of I(repr; author) that is adversary-independent, catching residual identity signals the adversary misses. |
| **Per-layer DP clipping** | Transformer layers have heterogeneous sensitivity to DP noise. Embedding layers and early transformer layers encode general linguistic features and benefit from tighter clipping; later task-specific layers need looser bounds. Uniform clipping is Pareto-suboptimal. |
| **Pre-norm (LayerNorm before sublayer)** | Stabilizes training at the small batch sizes required by DP-SGD (batch_size=16). Post-norm would require larger batches or more warmup steps. |
| **Ghost clipping** | Memory-efficient per-sample gradient computation for transformers. Standard per-sample clipping would be O(batch * params) memory; ghost clipping is O(batch * activations) which is 3-5x smaller for transformer architectures. |
| **Poisson sampling** | Randomly subsampling batches (rather than fixed-size batches) provides tighter DP accounting via the subsampling amplification theorem. Each example is included with probability q = batch_size / |D|. |
| **Synthetic pseudo-author labels** | Ground-truth author labels are rarely available in real HSD deployments. Stylistic feature hashing provides a proxy that, while imperfect, enables unsupervised identity disentanglement. |
| **Representation orthogonality regularization** | Encourages the model to encode hate-relevant and identity-relevant features in orthogonal subspaces, making it harder for the adversary to find identity information without explicit supervision. |
| **Privacy-augmented data (word dropout + entity masking)** | Reduces identity signal at the data level, complementing architectural privacy mechanisms. Word dropout breaks stylometric patterns; entity masking removes personally identifiable information. |

---

## 6. Research-to-Architecture Traceability

| Research Contract Item | Architecture Decision | Evidence Status | Validation Hook |
|---|---|---|---|
| DP + adversarial disentanglement is Pareto-optimal for HSD | Combined objective: L_total = L_hate + λ_dis * L_adv + λ_mim * I_est | `hypothesis` | Compare Pareto frontier (ε vs F1) across 4 conditions: DP-only, adv-only, both, neither. H1 confirmed if combined yields strictly dominating Pareto points. |
| ALBERT outperforms RoBERTa under DP for HSD (Biy+25) | Default backbone = ALBERT-base-v2; architecture support for RoBERTa as comparison | `grounded` | Run `run_experiments.py --ablation architecture`. ALBERT must show 2-5% higher F1 at ε ≤ 4. |
| Adversarial disentanglement reduces stylometry re-identification by >15% with <3% F1 loss | Multi-level adversarial block with GRL; separate evaluation of representation vs raw text stylometry | `hypothesis` | Compare stylometry accuracy on model representations vs. raw text features. H2 confirmed if model_rep_acc < raw_text_acc - 0.15 and F1_drop < 0.03. |
| DP-SGD provides formal (ε, δ)-DP guarantees | Opacus PrivacyEngine with RDP accounting and ghost clipping | `grounded` | Verify that ε tracking via `privacy_engine.get_epsilon(delta)` stays within target. Check that noise is correctly calibrated to σ = C * sqrt(2*log(1.25/δ)) / ε. |
| GRL enables domain-invariant representation learning (Ganin & Lempitsky 2015) | GradientReversalLayer applied at 3 representation levels | `grounded` | Verify gradient reversal: forward pass = identity, backward pass = -alpha * grad. Unit test with known inputs/outputs. |
| Privacy-augmented data improves Pareto frontier | Data preprocessing with label_flip_prob, word_dropout_prob, entity_masking | `TODO: unverified` | Compare Pareto frontier with and without augmentation (3 levels). H4 confirmed if aug_medium Pareto-dominates no_aug. |
| Stylometry achieves >90% authorship attribution on raw text (Abbasi & Chen 2008) | StylometryReidentificationRisk with raw text feature extraction (20+ features) | `grounded` | Run stylometry on raw text features from test set. Must achieve >0.85 accuracy (baseline sanity check). |
| Generative classifiers more vulnerable to MIA than discriminative (Makroo+25) | Discriminative classifier design (not generative) with probability outputs | `grounded` | Compare MIA AUC against reported values for generative classifiers (should be lower for our discriminative model). |
| Synthetic pseudo-author labels sufficient for disentanglement | Hash-based stylistic feature partitioning for author label creation | `TODO: unverified` | Compare disentanglement effectiveness (adversary accuracy) with synthetic vs. ground-truth author labels. |
| Representation entropy correlates with stylometry re-identification resistance | RepresentationPrivacyAudit with entropy, k-anonymity, and leakage score | `TODO: unverified` | Compute Pearson correlation between representation entropy and stylometry accuracy across all experiment configs. |
| Federated learning extension (Biy+25 NAACL 2025) | Architecture supports future FL integration; model serialization for Flower framework | `hypothesis` | Future work: benchmark centralized vs. federated privacy-utility trade-off. |
| Rights-based architecture aligns with GDPR/DSA/ECHR | Architecture explicitly prevents re-identification by design; no surveillance tooling released | `grounded` | Documentation audit: verify each GDPR principle maps to a technical implementation. Attack evaluation shows > chance re-identification accuracy. |

---

## 7. Domain-Specific Considerations

### 7.1 LM Domain

**Position encoding:** ALBERT uses absolute position embeddings (learned). We inherit this from the backbone. Position embeddings may encode positional stylistic patterns (e.g., signature placement). The multi-level adversarial heads operate on pooled representations that aggregate across positions, mitigating position-based identity leakage.

**Causal constraint:** Hate speech classification is bidirectional (not autoregressive). No causal masking is applied. The full context is available for each prediction.

**Sequence length:** Max 256 tokens. This captures typical social media posts (~280 chars = ~50 tokens) with margin for longer comments. If processing longer documents, mean-pooled token representations still capture distributed stylistic patterns.

### 7.2 Privacy-Preserving ML Domain

**Formal privacy guarantee:** DP-SGD provides (ε, δ)-DP where ε is the privacy budget and δ is the failure probability. We use δ = 1/|D| following standard practice.

**Privacy budget composition:** Each epoch consumes privacy budget. Total ε after E epochs is computed via RDP accounting. At ε = 8, δ = 1e-5, the model can be trained for ~5-10 epochs depending on noise multiplier.

**Per-sample gradient clipping:** The core DP operation. Without it, one outlier gradient could leak information about a single training example. Opacus's ghost clipping makes this tractable for transformers.

**Adversarial disentanglement + DP:** These are complementary. DP bounds what the model can leak about any individual training example. Adversarial disentanglement makes representations invariant to author identity. Together, they ensure both worst-case (DP) and average-case (adversarial) privacy.

### 7.3 Ethical AI Domain

**Anti-surveillance by design:** The architecture intentionally prevents authorship attribution. If an adversary obtains white-box access to the trained model and runs forward passes on text, the representations will be identity-agnostic due to the adversarial training objective.

**No re-identification tooling:** The privacy attack suite is implemented strictly for evaluation. No auxiliary re-identification pipeline is released as production tooling.

**Dataset bias awareness:** The Jigsaw and HateXplain datasets are predominantly English and may have annotation biases. The privacy guarantees are dataset-agnostic, but detection performance may vary across demographics.

### 7.4 Multilingual Considerations

**XLM-RoBERTa support:** The architecture supports XLM-RoBERTa as a backbone for multilingual hate speech detection. The multi-level adversarial block is model-agnostic.

**Cross-lingual identity signals:** Stylometric features vary across languages. The hash-based pseudo-author labeling works on character-level features that are language-agnostic.

**Low-resource adaptation:** For European languages (German, French, Italian, Spanish) with limited labeled data, the DP-SGD noise multiplier may need adjustment (smaller datasets → larger δ or higher ε).

---

## 8. Implementation Risks

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | **Ghost clipping + multi-head adversary memory blowup** | High | The MLAD block adds 3 adversary MLPs. Combined with Opacus's ghost clipping, this may exceed GPU memory (12GB for single A100). Mitigation: implement gradient checkpointing for adversaries, or reduce adversary hidden_dim to 128. |
| 2 | **MINE network instability** | Medium | MINE training requires careful balancing: if the MINE network overfits, the MI estimate becomes unreliable. Mitigation: clip MINE gradient norms, use spectral normalization on MINE network layers, and limit MINE update steps per encoder step. |
| 3 | **Synthetic author label noise** | Medium | Hash-based pseudo-author labels may not correspond to real stylistic clusters. If labels are too noisy, adversarial training may not converge productively. Mitigation: validate pseudo-author cluster quality using silhouette score; fall back to single-level adversary if multi-level adv_loss fails to decrease. |
| 4 | **Per-layer DP clipping correctness** | Medium | Opacus does not natively support per-layer clipping norms. Implementing custom per-layer clipping requires modifying the GradSampleModule. Mitigation: default to uniform clipping if custom implementation fails; validate DP guarantee via unit tests. |
| 5 | **Disentanglement degrades hate detection on edge cases** | High | Some hate speech is expressed through stylistic choices (e.g., sarcasm, coded language). Aggressive disentanglement may strip these features, causing false negatives. Mitigation: monitor per-group F1 across demographic groups; implement early stopping based on combined Pareto metric, not just hate F1. |
| 6 | **DP training time overhead** | Medium | DP-SGD with ghost clipping increases per-step time by 3-5x. Full experiment suite (20+ configs × 3 seeds) may take days. Mitigation: use quick mode (2 epochs, sample 200) for sanity; full experiments with small epsilon sweep first. |

---

## 9. Suggested Ablations

Each ablation is expressible as a single field change in `PrivHSDConfig` and tests a specific hypothesis.

| # | Ablation | Config Field | Baseline | Ablated | Hypothesis Tested | Expected Metric Movement | Failure Interpretation | Owning Stage |
|---|---|---|---|---|---|---|---|---|
| 1 | **Remove adversarial disentanglement** | `disentanglement_weight` | 0.3 | 0.0 | Adversarial training improves privacy-utility Pareto frontier | F1 ↑ 1-3%, MIA AUC ↑ 5-10%, stylometry acc ↑ 15-25% | If F1 increases > 3% AND MIA/stylometry doesn't degrade, the core novelty claim is falsified. | `ml-research` |
| 2 | **Remove MIM** | `mim_weight` | 0.1 | 0.0 | MIM reduces residual identity leakage beyond adversarial alone | Stylometry acc ↑ 3-8%, MIA AUC ↑ 2-5%, F1 unchanged | If stylometry doesn't degrade, MIM adds no value; revisit MINE architecture. | `ml-architect` |
| 3 | **Single-level vs multi-level adversarial** | `adversarial_levels` | ("pooler","token","head") | ("pooler",) | Multi-level adversaries remove more identity signal | Stylometry acc ↑ 5-10% for single level, F1 unchanged | If single level matches multi-level, the extra adversary capacity is wasted. | `ml-architect` |
| 4 | **Alpha schedule type** | `alpha_schedule` | "sigmoid" | "linear" | Sigmoid schedule improves utility at same privacy level | F1 ↑ 1-2% at ε=8, adv_loss convergence faster | If linear matches sigmoid, the scheduling complexity is unnecessary. | `ml-architect` |
| 5 | **Remove DP-SGD** | `dp_enabled` | True | False | DP-SGD is the primary formal privacy guarantee | F1 ↑ 3-5%, MIA AUC ↑ 10-20%, ε goes to inf | If MIA AUC doesn't increase much, model may already be privacy-preserving without DP (unlikely). | `ml-validator` |
| 6 | **Per-layer vs uniform clipping** | `per_layer_clipping` | True | False | Per-layer clipping improves utility at same ε | F1 ↑ 0.5-1.5% at same ε, no privacy degradation | If no improvement, uniform clipping is sufficient; reduce code complexity. | `ml-coder` |
| 7 | **Privacy budget sweep** | `target_epsilon` | 8.0 | 1.0, 2.0, 4.0, 16.0, 32.0, inf | Pareto frontier characterization | F1 decreases monotonically with ε | If F1 at ε=1 equals F1 at ε=8, DP noise may be too small (check σ calibration). If F1 at ε=32 < ε=8, hyperparameters may be suboptimal. | `ml-research` |
| 8 | **Remove privacy augmentation** | `privacy_augment_level` | "medium" | None | Data augmentation improves Pareto frontier | F1 ↓ 1-3% without augmentation, MIA AUC ↑ 2-5% | If augmentation improves nothing or hurts both utility and privacy, the hypothesis is falsified. | `ml-research` |
| 9 | **Backbone architecture** | `model_type` | "albert" | "roberta" | ALBERT outperforms RoBERTa under DP (Biy+25) | ALBERT F1 2-5% higher at ε≤4 | If RoBERTa matches ALBERT, revisit Biy+25 finding or check DP implementation. | `ml-validator` |
| 10 | **No orthogonality regularization** | `orthogonality_weight` | 0.05 | 0.0 | Orthogonality improves disentanglement without utility loss | Stylometry acc ↑ 2-5%, F1 unchanged | If no change, orthogonality regularization is unnecessary complexity. | `ml-architect` |

### Ablation Execution Priority

```
Priority 1 (must-try first if things don't work):
  1. Remove adversarial disentanglement (#1) — core novelty check
  2. Remove DP-SGD (#5) — fundamental privacy mechanism check
  3. Single-level adversarial (#3) — complexity check

Priority 2 (refinement):
  4. Remove MIM (#2) — secondary mechanism
  5. Alpha schedule type (#4) — scheduling strategy
  6. Privacy budget sweep (#7) — full characterization

Priority 3 (optimization):
  7. Per-layer clipping (#6) — efficiency gain
  8. Remove augmentation (#8) — data-level mechanism
  9. Backbone comparison (#9) — architecture choice
  10. Remove orthogonality (#10) — regularization check
```

---

## 10. Upgrade Path from v1 to v2

The existing prototype (v1) has a working `PrivHSDModel` with single-level adversarial disentanglement and DP-SGD. The v2 architecture upgrades are:

| Component | v1 (existing) | v2 (proposed) |
|---|---|---|
| **Adversarial levels** | Single ([CLS] only) | Multi-level (pooler, token, head) |
| **Alpha schedule** | Fixed (0.5) | Adaptive sigmoid (AGRS) |
| **Disentanglement metric** | Adversary loss only | Adversary + MINE-based MI |
| **Privacy clipping** | Uniform global norm | Per-layer adaptive norms |
| **Regularization** | None | Orthogonality + consistency |
| **Data augmentation** | Basic word drop | Entity masking + synonym replacement |
| **Author labels** | Hash-based | Hash-based + cluster validation |
| **Evaluation** | Basic metrics | Full Pareto + ablation framework |

**Minimal v2 upgrade path:**
1. Add `MultiLevelAdversarialBlock` (replaces single `identity_adversary`)
2. Add `AdaptiveAlphaScheduler` (replaces static alpha)
3. Add `MutualInformationMinimizer` (new module)
4. Add per-layer clipping to `PrivHSDTrainer`
5. Update `PrivHSDConfig` with new fields
6. All existing v1 interfaces (`forward`, `train_epoch`, `evaluate`) remain backward compatible

---

## Appendix A: Opacus Integration Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Opacus PrivacyEngine Integration                          │
└─────────────────────────────────────────────────────────────────────────────┘

Standard PyTorch Model
    │
    ▼
┌─────────────────────────────────────────────────┐
│  ModuleValidator.validate(model)                 │
│  Checks: InstanceNorm, BatchNorm, LSTM,          │
│          tied weights, shared parameters          │
│  Fixes: BatchNorm → GroupNorm (ghost clipping)   │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│  GradSampleModule (ghost clipping variant)       │
│  Computes per-sample gradients efficiently       │
│  via: grad_output @ weight = per_sample_grad     │
│  WITHOUT materializing per-sample activations   │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│  make_private_with_epsilon(                      │
│    module, optimizer, data_loader,               │
│    target_epsilon, target_delta, max_grad_norm,  │
│    grad_sample_mode="ghost",                     │
│    poisson_sampling=True,                        │
│  )                                               │
│  Returns: (private_model, private_optimizer,     │
│            private_loader)                       │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│  Private Optimizer (DPAdam / DPSGD)              │
│  Per step:                                       │
│    1. Clip per-sample gradients to norm ≤ C      │
│    2. Aggregate clipped gradients (mean)         │
│    3. Add Gaussian noise N(0, σ²C²I)            │
│    4. Take optimizer step                        │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│  RDP Accounting (Renyi DP → ε conversion)        │
│  get_epsilon(delta) → current ε spent            │
│  Tracks: noise_multiplier, steps, batch_size,    │
│          dataset_size, sampling_probability      │
└─────────────────────────────────────────────────┘
```

### Key Opacus Considerations for PrivHSD:

1. **Ghost clipping vs. standard**: Ghost clipping computes per-sample gradients without instantiating per-sample activations, reducing memory from O(B * L * H) to O(L * H). Essential for ALBERT-large with batch_size=16.

2. **ModuleValidator fixes**: The identity adversary head uses LayerNorm + Linear, which are DP-compatible. The hate classifier uses the same. No BatchNorm or InstanceNorm is used, avoiding common DP compatibility issues.

3. **Poisson sampling**: Randomly includes each example with probability q = batch_size / |D|. Provides tighter privacy accounting via subsampling amplification. Implemented via Opacus's `DPDataLoader`.

4. **Privacy accounting**: RDP accounting is used instead of standard (ε, δ)-DP composition. RDP provides tighter bounds for compositions of Gaussian mechanisms. The ε conversion uses the optimal RDP → (ε, δ) conversion.

---

*End of Architecture Specification v2.0*
*Next: Implementation Guide → `/work/coder/`*
