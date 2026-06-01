# PrivHSD: Privacy-preserving Hate Speech Detection

A privacy-by-design hate speech detection system that operates completely agnostically to the author's identity, maximizing detection performance while strictly minimizing privacy leakage.

**Council of Europe Democracy Hackathon 2026**
**Mentored by:** Technical University of Munich / Munich Center for Machine Learning (MCML)

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Quick sanity check (uses synthetic data if no real data available)
python run_experiments.py --quick

# Train a model with DP-SGD + adversarial disentanglement
python train.py --model albert-base-v2 --target-epsilon 8.0 --adversarial
```

## Core Innovation

PrivHSD combines three complementary privacy mechanisms:

1. **DP-SGD via Opacus**: Formal (ε, δ)-differential privacy guarantees
2. **Adversarial Identity Disentanglement**: GRL-based training to strip author identity signals from representations
3. **Privacy-Augmented Data**: Preprocessing to reduce identity leakage at the data level

## Repository Structure

```
privhsd/
├── src/
│   ├── model.py          # PrivHSDModel with adversarial disentanglement
│   ├── train.py          # DP-SGD training pipeline with Opacus
│   ├── data_utils.py     # Dataset loading & preprocessing
│   ├── evaluate.py       # Utility & privacy evaluation + Pareto analysis
│   └── attacks.py        # MIA, attribute inference, stylometry attacks
├── train.py              # Main training entrypoint
├── run_experiments.py    # Systematic ablation & experiment runner
├── RESEARCH_NOTE.md      # Full research documentation
├── requirements.txt      # Dependencies
```

## Experiments

```bash
# Privacy budget sweep (ε ∈ {1, 2, 4, 8, 16, 32})
python run_experiments.py --ablation privacy

# Adversarial disentanglement ablation
python run_experiments.py --ablation adversarial

# Architecture comparison (ALBERT vs RoBERTa)
python run_experiments.py --ablation architecture

# Full experiment suite
python run_experiments.py --full
```

## Privacy Guarantees

| Mechanism | Guarantee |
|---|---|
| **DP-SGD** | (ε, δ)-DP with ε ∈ [1, 32], δ = 1/|D| |
| **Adversarial Disentanglement** | Representation invariance to author identity |
| **Membership Inference** | Evaluated via shadow model attack (AUC reported) |
| **Stylometry Re-identification** | Evaluated via authorship attribution classifier |

## Ethical Design

- Architecture prevents re-use as surveillance tool
- No re-identification pipeline released as production tooling
- Aligned with GDPR, EU DSA, and ECHR principles
- Full transparency via published privacy budgets

## Citation

```bibtex
@software{privhsd2026,
  author = {PrivHSD Research Team},
  title = {PrivHSD: Privacy-preserving Hate Speech Detection},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/tum-privhsd/privhsd}
}
```
