> **Project layout** — this bundle contains five stage directories from the
> AI-Designer pipeline:
> `research/` (literature survey), `architect/` (blueprint + `ModelConfig`),
> `coder/` (PyTorch implementation), `validator/` (tests + benchmarks), and
> `documenter/` (this README plus `docs/` and `CHANGELOG.md`).
> An optional `paper/` directory holds the NeurIPS-format writeup when the
> paper-generation step was triggered.
>
> The original research request that produced this bundle is preserved
> verbatim in [`prompt.md`](prompt.md) — if any URLs in the prompt were
> fetched server-side for additional context, their cleaned contents are
> appended there too.

---

# PrivHSD v2: Privacy-preserving Hate Speech Detection

A rights-based, author-identity-agnostic hate speech detection model combining DP-SGD via Opacus with multi-level adversarial identity disentanglement and mutual information minimization for the Council of Europe Democracy Hackathon 2026.

PrivHSD v2 addresses the tension between content moderation and privacy: naive hate speech detection models leak stylistic fingerprints enabling authorship attribution, attribute inference, and membership inference. Our core mechanism is a three-player minimax game in which an encoder simultaneously learns hate-relevant features, minimizes a lower-bound estimate of mutual information between representations and author identity, and plays against multi-level adversaries through gradient reversal layers (GRLs). The empirical headline (projected, pending full experiment execution) is Pareto-dominant privacy-utility trade-off compared to DP-only or adversarial-only approaches.

## Highlights

- **Formal differential privacy via DP-SGD** — (ε, δ)-DP guarantees per Abadi et al. (2016), with Rényi DP accounting and ghost clipping via Opacus; see [Architecture: DP-SGD](documenter/docs/ARCHITECTURE.md#section-3-differential-privacy-via-opacus).
- **Multi-Level Adversarial Disentanglement (MLAD)** — three adversarial heads operating at pooler, token, and head representation levels provide comprehensive identity signal removal; see [Architecture: MLAD](documenter/docs/ARCHITECTURE.md#section-21-multi-level-adversarial-disentanglement-mlad).
- **Mutual Information Minimization via MINE** — explicit minimization of I(representation; author_id) using a neural lower-bound estimator, catching residual identity signals beyond the adversary; see [Architecture: MIM](documenter/docs/ARCHITECTURE.md#section-23-mutual-information-minimization-via-mine).
- **Adaptive Gradient Reversal Scheduling (AGRS)** — sigmoid alpha schedule that delays adversarial pressure until hate features are learned, then ramps disentanglement; see [Architecture: AGRS](documenter/docs/ARCHITECTURE.md#section-22-adaptive-gradient-reversal-scheduling-agrs).
- **Comprehensive privacy attack suite** — MIA (shadow model + threshold), attribute inference, stylometry re-identification (raw text + model representations), and representation privacy audit (entropy, k-anonymity, leakage score); see [Benchmarks](documenter/docs/BENCHMARKS.md).
- **Rights-based architecture aligned with GDPR, DSA, ECHR** — representations intentionally identity-agnostic; no re-identification tooling released; see [Architecture: Rights-Based Design](documenter/docs/ARCHITECTURE.md#section-7-domain-specific-considerations).

## Quick start

```bash
pip install -r requirements.txt
python -c "from src.model import PrivHSDModelV2, PrivHSDConfig; \
  cfg = PrivHSDConfig(dp_enabled=False); \
  m = PrivHSDModelV2(cfg); \
  print(f'Model initialized: {sum(p.numel() for p in m.parameters()):,} params')"
python smoke_test.py --full
pytest test_model.py -v
```

## Repository layout

```
privhsd/
├── src/
│   ├── __init__.py          # Public API exports, version string
│   ├── model.py             # PrivHSDModelV2, MLAD, MINE, AGRS, config
│   ├── train.py             # PrivHSDTrainer with DP-SGD + adversarial loop
│   ├── data_utils.py        # Dataset loading, pseudo-author labels, privacy augmentation
│   ├── evaluate.py          # Utility metrics, Pareto frontier analysis, plotting
│   ├── attacks.py           # MIA, attribute inference, stylometry, representation audit
│   └── inference.py         # Privacy-preserving inference engine
├── train.py                 # Main training entry point
├── run_experiments.py       # Systematic ablation and experiment runner
├── smoke_test.py            # CI-friendly smoke test (synthetic data, no GPU required)
├── requirements.txt         # Pinned dependencies
├── data/                    # Dataset directory (Jigsaw, HateXplain)
├── models/                  # Checkpoints and results
├── notebooks/               # Analysis notebooks
└── docs/
    ├── ARCHITECTURE.md      # Full architecture specification
    ├── TRAINING.md           # Training recipe and troubleshooting
    ├── BENCHMARKS.md         # Results, ablations, profiling
    └── API.md                # Module-level API reference
```

## Documentation

- [docs/ARCHITECTURE.md](documenter/docs/ARCHITECTURE.md) — design decisions, inductive biases, shape evolution, domain-specific considerations
- [docs/TRAINING.md](documenter/docs/TRAINING.md) — environment setup, hyperparameters, training recipe, troubleshooting
- [docs/BENCHMARKS.md](documenter/docs/BENCHMARKS.md) — utility and privacy benchmarks, ablation studies, profiling, research-quality evaluation
- [docs/API.md](documenter/docs/API.md) — full API reference for every public class and function

## Citation

```bibtex
@misc{privhsd2026,
  title  = {PrivHSD v2: Privacy-preserving Hate Speech Detection with Multi-Level
            Adversarial Disentanglement and Differential Privacy},
  author = {PrivHSD Research Team},
  year   = {2026},
  note   = {Council of Europe Democracy Hackathon 2026; Technical University of
            Munich / Munich Center for Machine Learning},
  howpublished = {\url{https://github.com/tum-mcml/privhsd}}
}
```
