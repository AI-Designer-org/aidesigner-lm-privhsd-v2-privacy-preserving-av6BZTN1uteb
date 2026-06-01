# PrivHSD v2 — Experiment Coverage

## Mapping Required Experiments to Implemented Artifacts

This document compares the experiments required by the upstream research contract
and architecture specification against what is implemented in the codebase. Each
entry documents: (1) what is required, (2) what is implemented, (3) how to run it,
and (4) any gaps.

---

## 1. Baseline Requirements

### From RESEARCH_CONTRACT.yaml

| # | Required Baseline | Status | Implementation | Run Command |
|---|---|---|---|---|
| 1 | Non-DP BERT/RoBERTa/ALBERT classifier without adversarial training | ✅ Implemented | `run_experiments.py` — `get_baseline_configs()` → `baseline_no_privacy` | `python run_experiments.py` (default includes baselines) |
| 2 | DP-only ALBERT classifier (ε=8) without adversarial disentanglement | ✅ Implemented | `run_experiments.py` — `get_baseline_configs()` → `dp_only` | `python run_experiments.py --ablation adversarial` |
| 3 | Adversarial-only ALBERT classifier without DP | ✅ Implemented | `run_experiments.py` — `get_baseline_configs()` → `adv_only` | `python run_experiments.py --ablation adversarial` |
| 4 | DP-ALBERT with ε ∈ {1, 2, 4, 8, 16, 32} | ✅ Implemented | `run_experiments.py` — `get_privacy_sweep_configs()` (6 epsilon values) | `python run_experiments.py --ablation privacy` |
| 5 | Non-DP RoBERTa-base for architecture comparison | ✅ Implemented | `run_experiments.py` — `get_architecture_comparison_configs()` | `python run_experiments.py --ablation architecture` |

### From ARCHITECTURE_SPEC.md

| # | Required Ablation | Status | Implementation | Run Command |
|---|---|---|---|---|
| 1 | Remove adversarial disentanglement (weight=0) | ✅ Implemented | `ablation_runner.py` → `no_adversarial` | `python ablation_runner.py --ablation 1` |
| 2 | Remove MIM (mim_weight=0) | ✅ Implemented | `ablation_runner.py` → `no_mim` | `python ablation_runner.py --ablation 4` |
| 3 | Single-level vs multi-level adversarial | ✅ Implemented | `ablation_runner.py` → `single_level_adversarial` | `python ablation_runner.py --ablation 3` |
| 4 | Alpha schedule type (linear vs sigmoid) | ✅ Implemented | `ablation_runner.py` → `alpha_linear` | `python ablation_runner.py --ablation 5` |
| 5 | Remove DP-SGD (dp_enabled=False) | ✅ Implemented | `ablation_runner.py` → `no_dp` | `python ablation_runner.py --ablation 2` |
| 6 | Per-layer vs uniform clipping | ✅ Implemented | `ablation_runner.py` → `uniform_clipping` | `python ablation_runner.py --ablation 7` |
| 7 | Privacy budget sweep | ✅ Implemented | `ablation_runner.py` → `strong_privacy_eps1`, `weak_privacy_eps32` | `python ablation_runner.py --ablation 10,11` |
| 8 | Remove privacy augmentation | ✅ Implemented | `ablation_runner.py` → `no_augmentation` | `python ablation_runner.py --ablation 8` |
| 9 | Backbone architecture (ALBERT vs RoBERTa) | ✅ Implemented | `ablation_runner.py` → `roberta_backbone` | `python ablation_runner.py --ablation 9` |
| 10 | Remove orthogonality regularization | ✅ Implemented | `ablation_runner.py` → `no_orthogonality` | `python ablation_runner.py --ablation 6` |

---

## 2. Evaluation Requirements

### Utility Metrics

| Metric | Status | Implementation | File |
|---|---|---|---|
| F1-score (binary) | ✅ Implemented | `compute_utility_metrics()` → `f1_score` | `src/evaluate.py` |
| AUC-ROC | ✅ Implemented | `compute_utility_metrics()` → `roc_auc` | `src/evaluate.py` |
| Accuracy | ✅ Implemented | `compute_utility_metrics()` → `accuracy` | `src/evaluate.py` |
| Precision | ✅ Implemented | `compute_utility_metrics()` → `precision` | `src/evaluate.py` |
| Recall | ✅ Implemented | `compute_utility_metrics()` → `recall` | `src/evaluate.py` |
| MCC | ✅ Implemented | `compute_utility_metrics()` → `mcc` | `src/evaluate.py` |
| Specificity | ✅ Implemented | `compute_utility_metrics()` → `specificity` | `src/evaluate.py` |

### Privacy Metrics

| Metric | Status | Implementation | File |
|---|---|---|---|
| ε, δ DP accounting | ✅ Implemented | `PrivHSDTrainer.get_privacy_spent()` via Opacus RDP | `src/train.py` |
| MIA (shadow model) AUC | ✅ Implemented | `MembershipInferenceAttack.train_shadow_model_attack()` | `src/attacks.py` |
| MIA (threshold) | ✅ Implemented | `MembershipInferenceAttack.train_threshold_attack()` | `src/attacks.py` |
| Attribute inference accuracy | ✅ Implemented | `AttributeInferenceAttack.evaluate()` | `src/attacks.py` |
| Stylometry (raw text) accuracy | ✅ Implemented | `StylometryReidentificationRisk.evaluate_raw_text_risk()` | `src/attacks.py` |
| Stylometry (model reps) accuracy | ✅ Implemented | `StylometryReidentificationRisk.evaluate_model_representation_risk()` | `src/attacks.py` |
| Representation entropy | ✅ Implemented | `RepresentationPrivacyAudit.compute_entropy()` | `src/attacks.py` |
| k-anonymity fraction | ✅ Implemented | `RepresentationPrivacyAudit.compute_k_anonymity()` | `src/attacks.py` |
| Privacy leakage score | ✅ Implemented | `RepresentationPrivacyAudit.compute_privacy_leakage_score()` | `src/attacks.py` |
| Pareto frontier | ✅ Implemented | `ParetoFrontierAnalyzer.compute_pareto_frontier()` + plotting | `src/evaluate.py` |

---

## 3. Synthetic Benchmarks

| Benchmark | Status | Implementation | File |
|---|---|---|---|
| Shape inference test | ✅ Implemented | `TestShapes.test_output_shape_inference()` | `validator/test_model.py` |
| Variable batch/sequence tests | ✅ Implemented | `TestShapes.test_variable_batch_size()`, `test_variable_sequence_length()` | `validator/test_model.py` |
| Gradient flow (all params) | ✅ Implemented | `TestGradients.test_all_params_receive_gradients()` | `validator/test_model.py` |
| GRL reversal correctness | ✅ Implemented | `TestGradients.test_grl_reverses_gradient()` | `validator/test_model.py` |
| Numerical stability (bf16, extreme inputs) | ✅ Implemented | `TestNumerics` (3 tests) | `validator/test_model.py` |
| Disentanglement effectiveness | ✅ Implemented | `TestCorrectness.test_disentanglement_reduces_author_accuracy()` | `validator/test_model.py` |
| Orthogonality regularization | ✅ Implemented | `TestCorrectness` (4 orthogonality tests) | `validator/test_model.py` |
| MINE MI estimate stability | ✅ Implemented | `TestCorrectness.test_mim_estimate_range()` | `validator/test_model.py` |
| Stylometry feature extraction | ✅ Implemented | `TestBenchmarks.test_stylometry_risk_baseline()` | `validator/test_model.py` |
| Pareto frontier correctness | ✅ Implemented | `TestBenchmarks.test_pareto_frontier_computation()` | `validator/test_model.py` |
| MIA shadow model | ✅ Implemented | `TestPrivacyML.test_mia_shadow_model()` | `validator/test_model.py` |
| Representation privacy audit | ✅ Implemented | `TestPrivacyML.test_representation_privacy_audit()` | `validator/test_model.py` |
| Rights-based architecture | ✅ Implemented | `TestRightsBasedArchitecture` (4 tests) | `validator/test_model.py` |

---

## 4. Ablation Coverage

| Ablation | Status | Syntactic Sugar | Run Command |
|---|---|---|---|
| no_adversarial (#1) | ✅ ✅ | `disentanglement_weight=0.0, mim_weight=0.0, orthogonality_weight=0.0` | `python ablation_runner.py --ablation 1` |
| no_dp (#5) | ✅ ✅ | `dp_enabled=False` | `python ablation_runner.py --ablation 2` |
| single_level_adversarial (#3) | ✅ ✅ | `adversarial_levels=("pooler",)` | `python ablation_runner.py --ablation 3` |
| no_mim (#2) | ✅ ✅ | `mim_weight=0.0` | `python ablation_runner.py --ablation 4` |
| alpha_linear (#4) | ✅ ✅ | `alpha_schedule="linear"` | `python ablation_runner.py --ablation 5` |
| no_orthogonality (#10) | ✅ ✅ | `orthogonality_weight=0.0` | `python ablation_runner.py --ablation 6` |
| uniform_clipping (#6) | ✅ ✅ | `per_layer_clipping=False` | `python ablation_runner.py --ablation 7` |
| no_augmentation (#8) | ✅ ✅ | `privacy_augment_level=None` | `python ablation_runner.py --ablation 8` |
| roberta_backbone (#9) | ✅ ✅ | `model_name="roberta-base", model_type="roberta"` | `python ablation_runner.py --ablation 9` |
| strong_privacy_eps1 (#7) | ✅ ✅ | `target_epsilon=1.0` | `python ablation_runner.py --ablation 10` |
| weak_privacy_eps32 (#7) | ✅ ✅ | `target_epsilon=32.0` | `python ablation_runner.py --ablation 11` |

**All 11 architect-specified ablations implemented.** ✅

---

## 5. Profiling Coverage

| Capability | Status | Implementation | Run Command |
|---|---|---|---|
| Forward-pass profiling | ✅ Implemented | `profile_model.py --mode forward` | `python profile_model.py --mode forward --steps 10` |
| Training step profiling | ✅ Implemented | `profile_model.py --mode train` | `python profile_model.py --mode train --steps 10` |
| Full DP training profiling | ✅ Implemented | `profile_model.py --mode full` | `python profile_model.py --mode full --steps 10` |
| FLOP estimation | ✅ Implemented | `estimate_flops()` (Kaplan scaling: 2N forward, 6N train) | `python profile_model.py` |
| CUDA memory tracking | ✅ Implemented | `torch.cuda.memory_allocated()`, `max_memory_allocated()` | `python profile_model.py` |
| Chrome trace export | ✅ Implemented | `prof.export_chrome_trace()` | `python profile_model.py --html` |
| Memory budget check | ✅ Implemented | `memory_budget_check()` estimates if config fits on GPU | `python profile_model.py --memory-check` |

---

## 6. Gaps and TODO Items

| Gap | Severity | Notes | Required Action |
|---|---|---|---|
| No empirical results | 🔴 High | All current results are projected from literature | Run `python run_experiments.py --full` on GPU |
| No multi-seed experiments | 🔴 High | Single seed (42) used throughout | Add multi-seed loop to `run_experiments.py` |
| No multilingual evaluation | 🟡 Medium | European focus requires DE/FR/IT/ES data | Implement XLM-RoBERTa data pipeline |
| No federated learning | 🟡 Medium | FL extension mentioned as future work | Integrate Flower framework + Opacus |
| No Docker environment | 🟡 Medium | Reproducibility without container spec | Create Dockerfile with pinned versions |
| Per-layer clipping fallback | 🟡 Medium | May fall back to uniform clipping | Verify with Opacus grad_sample_mode |
| No real dataset validation | 🔴 High | Uses synthetic data by default | Download Jigsaw + HateXplain, update paths |
| No related work section | 🟢 Low | Research note lacks structured comparison | Add "Related Work" section to RESEARCH_NOTE.md |
| No hyperparameter sensitivity | 🟢 Low | No automated hyperparameter search | Add Optuna/Ray Tune integration |

---

## 7. Can the Benchmark Suite Distinguish the Proposed Architecture from a Trivial Baseline?

**Yes**, for the following reasons:

1. **Ablation runner computes F1_delta** — each ablation's utility is compared to baseline,
   making it immediately visible whether removing a component degrades performance.

2. **Pareto frontier analysis** — the combined DP+adversarial configuration can be directly
   compared against DP-only, adversarial-only, and no-privacy baselines. If the combined
   config Pareto-dominates, the novelty claim is supported.

3. **Privacy attack suite provides multiple dimensions** — even if F1 is similar, the
   combined config must show lower MIA AUC and lower stylometry accuracy to justify
   the additional complexity.

4. **Synthetic benchmarks with known ground truth** — the `TestBenchmarks` tests verify
   that the model can learn hate speech patterns above random baseline, and that privacy
   metrics behave as expected.

5. **Statistical comparison via deltas** — the ablation runner automatically computes
   f1_delta and eps_delta from baseline, providing quantitative comparison.
