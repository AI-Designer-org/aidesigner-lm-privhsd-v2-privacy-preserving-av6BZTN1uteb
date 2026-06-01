# Changelog

## [2.0.0] — 2026-06-01
### Added
- Initial implementation of PrivHSD v2 with multi-level adversarial disentanglement (MLAD).
- DP-SGD training via Opacus PrivacyEngine with ghost clipping.
- Mutual Information Minimization via MINE (MIM) module.
- Adaptive Gradient Reversal Scheduling (AGRS) with sigmoid schedule.
- Representation orthogonality regularization.
- Per-layer adaptive DP clipping interface.
- Hate speech classification head with pre-norm architecture.
- Backward-compatible v1 alias (`PrivHSDModel`).
- Unit test suite (9 test classes, 45+ tests):
  - Shape tests (output dimensions, variable sizes, representation shapes).
  - Gradient flow tests (encoder, classifier, adversary, MINE, GRL).
  - Correctness tests (loss values, disentanglement, scheduler, orthogonality, MINE).
  - Numerical stability tests (bf16, fp16 guard, extreme inputs, zero mask).
  - Privacy ML tests (DP budget, non-DP mode, representation audit, stylometry, MIA).
  - Data pipeline tests (dataset, dataloader, augmentation).
  - Benchmark tests (synthetic hate detection, entropy, Pareto, stylometry risk).
  - Evaluation pipeline tests (model evaluation, result serialization).
  - Rights-based architecture tests (representation leak, inference privacy mode).
- Privacy attack suite: membership inference (shadow + threshold), attribute inference, stylometry re-identification (raw text + model representations), representation privacy audit (entropy, k-anonymity, leakage score).
- Privacy-augmented data variant creation (low/medium/high noise levels).
- Synthetic pseudo-author label generation via stylistic feature hashing.
- Systematic experiment runner with ablation config factories:
  - 7-level privacy budget sweep (ε ∈ {1, 2, 4, 8, 16, 32, ∞}).
  - 4-condition 2×2 adversarial ablation (DP × adversarial).
  - 3-backbone architecture comparison (ALBERT-base, ALBERT-large, RoBERTa-base).
  - 4-level privacy augmentation comparison (none, low, medium, high).
- Ablation runner with 11 single-field configs mapped to architecture traceability table.
- Pareto frontier analyzer with plotting, results CSV/JSON export.
- Profile script with torch.profiler and FLOP estimation (Kaplan scaling).
- CI-friendly smoke test with synthetic data (no HF download required).
- Config dataclass as single source of truth for all hyperparameters.
- Comprehensive error handling, logging, and checkpointing.
- Documentation: README, ARCHITECTURE, TRAINING, BENCHMARKS, API, CHANGELOG.
- Requirements.txt with pinned dependencies.

### Rights-based design features
- Inference engine never returns representations by default.
- Privacy mode prevents text logging.
- Representations intentionally identity-agnostic via adversarial training.
- No re-identification tooling released.
- Architecture aligned with GDPR Art. 25, EU Digital Services Act, ECHR Art. 10.

### Known gaps (documented)
- No empirical results yet — all values projected from literature.
- Multi-seed experiments not yet implemented (currently single seed 42).
- Multilingual evaluation not yet included (DE/FR/IT/ES).
- Federated learning extension not implemented.
- Docker/Singularity container not yet provided.
