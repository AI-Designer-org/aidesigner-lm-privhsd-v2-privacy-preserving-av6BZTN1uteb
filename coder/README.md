# PrivHSD v2: Privacy-preserving Hate Speech Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Multi-Level Adversarial Disentanglement + DP-SGD + Mutual Information Minimization for Author-Identity-Agnostic Hate Speech Detection.**

PrivHSD v2 is a privacy-preserving hate speech detection system built on a **rights-based architecture** aligned with GDPR, EU Digital Services Act, and Council of Europe human rights standards. It achieves high hate speech detection performance while provably minimizing identity signal leakage.

> **Target:** Council of Europe Democracy Hackathon 2026 (deadline 5 June 2026)
> **Mentored by:** Technical University of Munich / Munich Center for Machine Learning

---

## Architecture at a Glance

```
Input Text → Tokenizer → Transformer Backbone (ALBERT)
                              │
                    ┌─────────┴─────────┐
                    │                   │
               Hate Classifier     MLAD Block (GRL × 3)
               (CE Loss)           (Pooler + Token + Head)
                    │                   │
               Hate Loss           Author Loss (reversed)
                    │                   │
                    └─────┬─────────────┘
                          │
              ┌───────────┴───────────┐
              │    Mutual Info (MINE)  │
              │    Orthogonality Reg   │
              └───────────┬───────────┘
                          │
                    L_total = L_hate + λ_dis*L_adv + λ_mim*I_est + λ_orth*L_orth
                          │
                    DP-SGD (Opacus ghost clipping)
```

## Core Innovations (v2)

| Innovation | What | Why |
|---|---|---|
| **MLAD** | Adversarial heads at pooler, token, and head levels | Identity signals distributed across representations; need multi-level removal |
| **AGRS** | Sigmoid/adaptive alpha schedule | Early: learn hate features (α≈0); Late: max disentanglement (α≈1) |
| **MINE** | Neural MI estimation and minimization | Catches residual identity the adversary misses |
| **Per-Layer DP** | Layer-adaptive clip norms | Different layers have different DP sensitivity |
| **Orthogonality Reg** | Orthogonal hate/identity subspaces | Makes identity info harder to extract |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Smoke test (verifies all components with synthetic data)
python smoke_test.py

# 3. Train a model (uses synthetic data if real datasets unavailable)
python train.py --model albert-base-v2 --target-epsilon 8.0 --adversarial

# 4. Run experiments
python run_experiments.py --quick                        # Quick sanity
python run_experiments.py --ablation privacy             # Privacy budget sweep
python run_experiments.py --ablation adversarial         # Adversarial ablation
python run_experiments.py --ablation architecture        # ALBERT vs RoBERTa
python run_experiments.py --full                         # Full experiment suite
```

## Usage Examples

### Training

```bash
# DP-SGD + adversarial disentanglement (recommended)
python train.py --model albert-base-v2 --target-epsilon 8.0 --adversarial

# DP-only (baseline)
python train.py --model albert-base-v2 --target-epsilon 8.0 --no-adversarial

# Non-DP + adversarial (for ablation)
python train.py --model roberta-base --no-dp --adversarial

# With privacy-augmented data
python train.py --model albert-base-v2 --privacy-augment medium
```

### Inference

```python
from src.inference import PrivHSDInference

engine = PrivHSDInference(model_path="models/privhsd_final.pt")
result = engine.predict("This is a sample text to classify")
print(f"Hate probability: {result['hate_probability']:.4f}")
print(f"Is hate: {result['is_hate']}")
```

### Programmatic API

```python
from src.model import PrivHSDConfig, PrivHSDModelV2

config = PrivHSDConfig(model_name="albert-base-v2", num_authors=100)
model = PrivHSDModelV2(config)

outputs = model(input_ids, attention_mask,
                hate_labels=hate_labels, author_labels=author_labels)
loss = outputs["loss"]
hate_probs = outputs["hate_probs"]
```

## Repository Structure

```
privhsd/
├── src/
│   ├── model.py          # PrivHSDModelV2 (MLAD, AGRS, MINE, orthogonality)
│   ├── train.py          # DP-SGD training with Opacus + adversarial loop
│   ├── data_utils.py     # Dataset loading, pseudo-author labels, augmentation
│   ├── evaluate.py       # Utility metrics, Pareto frontier analysis, plots
│   ├── attacks.py        # MIA, attribute inference, stylometry, representation audit
│   └── inference.py      # Deployable inference engine (privacy mode)
├── train.py              # Main training script
├── run_experiments.py    # Systematic ablation experiment runner
├── smoke_test.py         # Component smoke tests
├── RESEARCH_NOTE.md      # Full research note with methodology
├── requirements.txt      # Python dependencies
├── data/                 # Dataset storage
└── models/               # Output directory
    ├── checkpoints/
    ├── results/
    └── ablations/
```

## Ablation Studies

| Command | Configs | Scope |
|---|---|---|
| `--ablation privacy` | 7 (ε ∈ {1, 2, 4, 8, 16, 32, ∞}) | Full privacy budget sweep |
| `--ablation adversarial` | 4 (DP/adv/DP+adv/neither) | Component contribution |
| `--ablation architecture` | 3 (ALBERT-base, ALBERT-large, RoBERTa) | Backbone comparison |
| `--ablation augmentation` | 4 (none, low, medium, high) | Data augmentation effect |
| `--full` | All of the above | Complete characterization |

## Privacy Attack Suite

| Attack | Method | Metric |
|---|---|---|
| **Membership Inference** | Shadow model on prediction features | AUC (lower = better) |
| **Attribute Inference** | LogisticRegression on representations | Accuracy (lower = better) |
| **Stylometry (raw text)** | RandomForest on 80+ features | Accuracy (baseline) |
| **Stylometry (model reps)** | LogisticRegression on representations | Accuracy (lower = better) |
| **Representation Audit** | Entropy, k-anonymity, leakage score | Composite (higher entropy = better) |

## Rights-Based Architecture

PrivHSD v2 is explicitly aligned with:

- **GDPR Art. 5, 25**: Data minimization + privacy by design/default
- **EU Digital Services Act**: Proportionate moderation without surveillance
- **ECHR Art. 8, 10**: Private life + freedom of expression
- **Council of Europe CM/Rec(2021)8**: Democratic values in content moderation

## Key References

- Biy et al. (2025). Privacy-Preserving FL for HSD. *NAACL 2025 SRW.*
- Ganin & Lempitsky (2015). Domain Adaptation by Backpropagation. *ICML.*
- Belghazi et al. (2018). MINE: Mutual Information Neural Estimation. *ICML.*
- Abadi et al. (2016). Deep Learning with Differential Privacy. *CCS.*
- Abbasi & Chen (2008). Writerpints. *ACM TOIS.*

## License

MIT License — see LICENSE file for details.
