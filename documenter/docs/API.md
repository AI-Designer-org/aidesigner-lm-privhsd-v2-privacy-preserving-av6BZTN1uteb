# API Reference

## `src.model` — Model architecture

### `class PrivHSDConfig`
Complete configuration dataclass for PrivHSD v2. All hyperparameters centralized for systematic ablation.

**Fields:**
| Field | Type | Default | Rationale |
|---|---|---|---|
| `model_name` | `str` | `"albert-base-v2"` | HF model identifier |
| `model_type` | `str` | `"albert"` | `"albert"` \| `"roberta"` \| `"xlm-roberta"` |
| `d_model` | `int` | 768 | Hidden dimension (derived from backbone) |
| `n_layers` | `int` | 12 | Transformer layers |
| `n_heads` | `int` | 12 | Attention heads |
| `d_ff` | `int` | 3072 | Feed-forward dimension |
| `max_seq_len` | `int` | 256 | Truncation length |
| `dropout` | `float` | 0.2 | Hidden dropout |
| `attention_dropout` | `float` | 0.2 | Attention dropout |
| `num_hate_classes` | `int` | 2 | Binary hate/not-hate |
| `classifier_hidden_dim` | `int` | 768 | Hate classifier MLP hidden dim |
| `classifier_num_layers` | `int` | 2 | Hate classifier MLP depth |
| `num_authors` | `int` | 100 | Pseudo-author classes |
| `adversarial_levels` | `Tuple[str]` | `("pooler","token","head")` | Levels for adversaries |
| `adversary_hidden_dim` | `int` | 256 | Adversary MLP hidden size |
| `adversary_num_layers` | `int` | 3 | Adversary MLP depth |
| `adversary_dropout` | `float` | 0.3 | Adversary dropout (higher=more regularization) |
| `alpha_initial` | `float` | 0.1 | Initial GRL scaling |
| `alpha_final` | `float` | 1.0 | Final GRL scaling |
| `alpha_schedule` | `str` | `"sigmoid"` | `"linear"` \| `"sigmoid"` \| `"adaptive"` |
| `alpha_warmup_epochs` | `int` | 2 | Warmup before alpha increases |
| `alpha_gamma` | `float` | 2.0 | Sigmoid steepness |
| `disentanglement_weight` | `float` | 0.3 | Main adversarial loss weight |
| `mim_weight` | `float` | 0.1 | MI minimization weight |
| `orthogonality_weight` | `float` | 0.05 | Orthogonality regularization |
| `consistency_weight` | `float` | 0.05 | Cross-model consistency (future) |
| `mim_estimator` | `str` | `"mine"` | `"mine"` \| `"nwj"` \| `"info_nce"` |
| `mim_hidden_dim` | `int` | 128 | MINE network hidden dim |
| `mim_learning_rate` | `float` | 1e-4 | Separate LR for MINE |
| `dp_enabled` | `bool` | `True` | Master switch for DP-SGD |
| `target_epsilon` | `float` | 8.0 | Target privacy budget |
| `target_delta` | `Optional[float]` | `None` | Auto = 1/\|D\| if None |
| `max_grad_norm` | `float` | 1.0 | Global clipping norm |
| `per_layer_clipping` | `bool` | `True` | Adaptive per-layer clip norms (interface) |
| `privacy_augment_level` | `Optional[str]` | `None` | `None` \| `"low"` \| `"medium"` \| `"high"` |
| `label_flip_prob` | `float` | 0.05 | Label noise probability |
| `word_dropout_prob` | `float` | 0.10 | Word dropout probability |
| `batch_size` | `int` | 16 | Per-GPU batch size |
| `learning_rate` | `float` | 2e-5 | Peak learning rate |
| `num_epochs` | `int` | 10 | Training epochs |
| `seed` | `int` | 42 | Random seed |

### `class GradientReversalLayer(torch.autograd.Function)`
Gradient Reversal Layer (GRL) for adversarial training. Forward: identity. Backward: negates and scales gradient by -α.

**Methods:**
- `forward(ctx, x, alpha=1.0)` — `(B, D) → (B, D)`. Identity forward.
- `backward(ctx, grad_output)` — `(B, D) → (B, D)`. Returns `-α · grad_output`.

Shape invariants: Input and output have identical shape. Alpha is a scalar float.

### `class AdaptiveAlphaScheduler`
Adaptive Gradient Reversal Scheduling (AGRS). Controls GRL alpha over training steps using parameterized schedules.

**Constructor:** `AdaptiveAlphaScheduler(config: PrivHSDConfig)`

**Methods:**
- `set_total_steps(total_steps: int)` — Set total training steps from actual dataloader.
- `get_alpha(step: int, hate_loss=None, adv_loss=None) → float` — Compute alpha for current step. Returns α ∈ [α_initial, α_final].
- `update_history(hate_loss: float, adv_loss: float)` — Update loss history for adaptive scheduling.

### `class AdversaryMLP(nn.Module)`
Multi-layer perceptron for identity adversary head.

**Constructor:** `AdversaryMLP(input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.3)`

**Methods:**
- `forward(x)` — `(B, D) → (B, output_dim)`. Passes through MLP.
- `get_features(x)` — `(B, D) → (B, hidden_dim)`. Extracts pre-classification features for orthogonality regularization.

### `class MultiLevelAdversarialBlock(nn.Module)`
Multi-Level Adversarial Disentanglement (MLAD). Deploys separate adversarial heads at pooler, token, and head representation levels.

**Constructor:** `MultiLevelAdversarialBlock(config: PrivHSDConfig)`

**Methods:**
- `forward(pooler_repr, token_repr, head_repr, author_labels, alpha=1.0) → Dict[str, Tensor]`
  - `pooler_repr`: `(B, D)` — [CLS] pooled representation
  - `token_repr`: `(B, T, D)` — all token representations
  - `head_repr`: `(B, D)` — per-head attention output averages
  - `author_labels`: `(B,)` — author identity labels
  - Returns dict with `"author_loss"`, `"level_losses"`, `"level_logits"`, `"adversary_features"`

### `class MutualInformationMinimizer(nn.Module)`
Mutual Information Neural Estimator (MINE) for minimizing I(representation; author_id).

**Constructor:** `MutualInformationMinimizer(repr_dim, num_authors, hidden_dim=128)`

**Methods:**
- `estimate_mutual_information(representations, author_labels) → Tensor` — Estimate I(repr; author) using MINE lower bound. Returns scalar.
- `forward(representations, author_labels, maximize=False) → Tensor` — If `maximize=True`, returns -I for gradient ascent (MINE step). If `False`, returns +I (encoder step).

### `class HateClassificationHead(nn.Module)`
Hate speech classification MLP head.

**Constructor:** `HateClassificationHead(hidden_size, num_classes, num_layers=2, dropout=0.2)`

**Methods:**
- `forward(x)` — `(B, D) → (B, num_classes)`. Hate logits.
- `get_features(x)` — `(B, D) → (B, D)`. Pre-classification features for orthogonality regularization.

### `class PerLayerDPAdapter`
DP-SGD per-layer adaptive gradient clipping interface.

**Methods (static):**
- `estimate_layer_sensitivity(model, sample_batch, n_steps=5) → List[float]` — Estimate per-layer sensitivity via gradient variance. Returns clip factors.
- `apply_per_layer_clipping(model, clip_factors, base_clip_norm=1.0)` — Apply per-layer clipping norms (interface; may fall back to uniform).

### `class PrivHSDModelV2(nn.Module)`
Main PrivHSD v2 model. Combines transformer backbone, hate classification head, MLAD block, MINE module, and AGRS scheduler.

**Constructor:** `PrivHSDModelV2(config: PrivHSDConfig)`

**Methods:**
- `forward(input_ids, attention_mask, hate_labels=None, author_labels=None, alpha=None, step=None, return_representations=False, return_attentions=False, use_checkpoint=False) → Dict[str, Tensor]`
  - `input_ids`: `(B, T)` — token indices
  - `attention_mask`: `(B, T)` — attention mask
  - `hate_labels`: `(B,)` — optional hate speech labels
  - `author_labels`: `(B,)` — optional author identity labels
  - `alpha`: float — GRL scaling (from scheduler if None)
  - `step`: int — current training step
  - Returns dict with `"hate_logits"` `(B, 2)`, `"hate_probs"` `(B, 2)`, and optionally `"loss"`, `"hate_loss"`, `"author_loss"`, `"mim_loss"`, `"orth_loss"`, `"alpha"`, `"level_losses"`, `"representations"`, `"attentions"`
- `get_transformer_outputs(input_ids, attention_mask) → Dict[str, Tensor]` — Extract multi-level representations. Returns dict with `"pooler_repr"` `(B, D)`, `"token_repr"` `(B, T, D)`, `"head_repr"` `(B, D)`, `"last_hidden"` `(B, T, D)`.
- `get_hate_predictions(input_ids, attention_mask) → Tensor` — `(B, T) → (B, 2)`. Hate probabilities for inference.
- `get_representations(input_ids, attention_mask) → Tensor` — `(B, T) → (B, D)`. Pooled representations for privacy analysis.
- `compute_mine_mi(input_ids, attention_mask, author_labels) → float` — Compute MINE mutual information estimate.

### `class PrivHSDModel(PrivHSDModelV2)`
Backward-compatible v1 alias. Allows existing v1 training code to work with v2 architecture.

**Constructor:** `PrivHSDModel(model_name="albert-base-v2", num_hate_classes=2, num_authors=100, adversarial_alpha=0.5, disentanglement_weight=0.3, hidden_dropout=0.2, model_type="albert", cache_dir=None)`

### `compute_subspace_orthogonality(feat_a, feat_b) → Tensor`
Compute cosine similarity between two feature subspaces. `(B, D1)` and `(B, D2) → scalar`. Lower = more orthogonal = better disentanglement.

### `count_params(model) → None`
Print total and trainable parameter counts.

---

## `src.train` — Training pipeline

### `class PrivHSDTrainer`
Trainer for PrivHSD v2 with DP-SGD and adversarial disentanglement.

**Constructor:** `PrivHSDTrainer(model, train_loader, val_loader, test_loader, config=None, learning_rate=2e-5, target_epsilon=8.0, target_delta=None, max_grad_norm=1.0, dp_enabled=True, device="cuda", output_dir="models/checkpoints", use_ghost_clipping=True)`

**Methods:**
- `train_step(batch, step, use_adversarial=True) → Dict` — Single training step with three-phase backward (MINE → encoder → DP-SGD).
- `train_epoch(epoch, use_adversarial=True) → Dict[str, float]` — Train for one epoch. Returns metrics dict.
- `evaluate(loader, split_name="val") → Dict[str, float]` — Evaluate on a data loader. Returns loss, acc, F1, AUC, precision, recall.
- `train(num_epochs=10, use_adversarial=True, eval_every=1, save_every=5, early_stopping_patience=5) → Dict` — Full training loop with early stopping and checkpointing.
- `get_privacy_spent() → float` — Current ε spent via Opacus RDP accounting.
- `save_checkpoint(epoch)` — Save model checkpoint.
- `save_final_model(path)` — Save final trained model.

---

## `src.data_utils` — Data loading

### `class HateSpeechDataset(Dataset)`
Unified hate speech dataset with author identity labels.

**Constructor:** `HateSpeechDataset(texts, hate_labels, author_ids=None, tokenizer=None, max_length=256, is_test=False)`

Each item returns dict with `"input_ids"` `(T,)`, `"attention_mask"` `(T,)`, `"hate_labels"` `(,)`, and optionally `"author_labels"` `(,)`.

### `create_author_labels(texts, n_authors=100, random_seed=42) → List[int]`
Create synthetic pseudo-author labels from stylistic features (text length, punctuation, capitalization). Divides dataset into n_authors groups via MD5 hashing of feature vectors.

### `load_jigsaw_dataset(data_dir, tokenizer, max_length=256, sample_size=None, random_seed=42) → Tuple[Dataset, Dataset, Dataset, int]`
Load Jigsaw Toxic Comment Classification dataset. Returns (train, val, test, n_authors). Generates synthetic data if CSV not found.

### `load_hatexplain_dataset(data_dir, tokenizer, max_length=256, sample_size=None, random_seed=42) → Tuple[Dataset, Dataset, Dataset, int]`
Load HateXplain dataset. Returns (train, val, test, n_authors). Generates synthetic data if JSON not found.

### `create_privacy_augmented_variant(dataset, epsilon_level="medium", random_seed=42) → HateSpeechDataset`
Create privacy-augmented variant with label noise and word dropout. Levels: `"low"` (0.02/0.05), `"medium"` (0.05/0.10), `"high"` (0.10/0.20).

### `get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=16, num_workers=2) → Tuple[DataLoader, DataLoader, DataLoader]`
Create standard DataLoaders for train/val/test splits.

---

## `src.evaluate` — Evaluation framework

### `class UtilityMetrics`
Container for hate speech detection utility metrics.

**Fields:** `accuracy`, `f1_score`, `precision`, `recall`, `roc_auc`, `average_precision`, `specificity`, `mcc`

### `class PrivacyMetrics`
Container for privacy guarantee metrics.

**Fields:** `epsilon`, `delta`, `mechanism`, `membership_inference_auc`, `attribute_inference_auc`, `stylometry_reid_accuracy`, `representation_entropy`

### `class EvaluationResult`
Full evaluation result for a single experiment configuration.

**Fields:** `config` (dict), `utility` (UtilityMetrics), `privacy` (PrivacyMetrics), `predictions`, `probabilities`, `labels`

**Methods:**
- `privacy_utility_ratio() → float` — F1 / max(ε, 0.01). Higher = better trade-off.

### `compute_utility_metrics(y_true, y_pred, y_prob) → UtilityMetrics`
Compute comprehensive utility metrics from numpy arrays.

### `evaluate_model(model, data_loader, device="cuda") → EvaluationResult`
Run full utility evaluation of a model on a dataset.

### `class ParetoFrontierAnalyzer`
Analyze the privacy-utility Pareto frontier across experiment configs.

**Constructor:** `ParetoFrontierAnalyzer(output_dir="models/results")`

**Methods:**
- `add_result(result, config=None)` — Add evaluation result.
- `compute_pareto_frontier(privacy_metric="epsilon", utility_metric="f1_score") → List[EvaluationResult]` — Compute Pareto-optimal points.
- `plot_pareto_frontier(save_name, privacy_metric, utility_metric)` — Plot frontier.
- `plot_ablation_study(save_name)` — Plot ablation comparison.
- `save_results_table(save_name) → pd.DataFrame` — Save results CSV.
- `generate_report(save_name) → Dict` — Generate JSON evaluation report.

---

## `src.attacks` — Privacy attack suite

### `class AttackMetrics`
Metrics for a privacy attack.

**Fields:** `auc`, `accuracy`, `precision`, `recall`, `f1`, `advantage`, `confidence`

### `class MembershipInferenceAttack`
MIA via shadow model or threshold approaches.

**Constructor:** `MembershipInferenceAttack(attack_type="shadow_model", random_seed=42)`

**Methods:**
- `evaluate(model, member_texts, non_member_texts, tokenizer, device) → AttackMetrics` — Run MIA attack.
- `train_shadow_model_attack(member_features, non_member_features) → AttackMetrics` — Train shadow classifier.
- `train_threshold_attack(train_losses, test_losses) → Tuple[float, float]` — Find optimal threshold.

### `class AttributeInferenceAttack`
Attribute inference from model representations.

**Constructor:** `AttributeInferenceAttack(random_seed=42)`

**Methods:**
- `evaluate(representations, attribute_labels) → AttackMetrics` — Evaluate attribute inference vulnerability.

### `class StylometryReidentificationRisk`
Stylometry-based re-identification risk assessment (raw text + model representations).

**Constructor:** `StylometryReidentificationRisk(n_authors=50, random_seed=42)`

**Methods:**
- `evaluate(model, texts, author_labels, tokenizer, device) → Dict[str, AttackMetrics]` — Full stylometry assessment.
- `evaluate_raw_text_risk(texts, author_labels) → AttackMetrics` — Baseline risk from raw text features.
- `evaluate_model_representation_risk(model, texts, author_labels, tokenizer, device) → AttackMetrics` — Risk from model representations.

### `class RepresentationPrivacyAudit`
Audit privacy properties of model representations.

**Static methods:**
- `compute_entropy(representations) → float` — Eigenvalue entropy of representation covariance. Higher = better.
- `compute_k_anonymity(representations, k=5, threshold=0.95) → float` — Fraction of k-anonymous representations. Higher = better.
- `compute_privacy_leakage_score(representations, author_labels) → float` — Composite leakage score. Lower = better.

---

## `src.inference` — Inference engine

### `class PrivHSDInference`
Deployable inference wrapper with privacy-preserving features.

**Constructor:** `PrivHSDInference(model_path, device="cuda", privacy_mode=True)`

**Methods:**
- `predict(text) → Dict` — Predict hate speech for single text. Returns `{"hate_probability", "is_hate"}`. Never returns representations.
- `predict_batch(texts, batch_size=32) → List[Dict]` — Batch prediction.
- `validate_identity_agnostic(texts_by_author) → Dict` — Validate predictions are identity-agnostic. Returns `identity_agnostic_score`.

---

## `src.__init__` — Package exports

Public API re-exports all above classes and functions. Version: `"2.0.0"`.
