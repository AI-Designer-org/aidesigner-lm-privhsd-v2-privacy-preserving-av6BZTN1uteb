# Training & Reproduction

## Environment

- Python: 3.10+
- PyTorch: 2.0+
- Transformers: 4.30+
- Opacus: 1.4+
- CUDA: 11.7+, tested on NVIDIA A100 (40 GB) or equivalent
- Other: flash-attn (optional, for larger models)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch; import opacus; print(torch.__version__, opacus.__version__)"
```

`requirements.txt` includes:
- `torch>=2.0.0`, `transformers>=4.30.0`, `opacus>=1.4.0`
- `scikit-learn`, `scipy`, `numpy`, `pandas`
- `matplotlib`, `seaborn` (for evaluation plots)
- `tqdm`, `pyyaml` (utilities)
- `sentencepiece`, `tokenizers` (tokenization)

## Default hyperparameters

All hyperparameters are centralized in `PrivHSDConfig` (`src/model.py`, lines 34-139).

| Field | Default | Rationale |
|---|---|---|
| `model_name` | `"albert-base-v2"` | Parameter-efficient backbone under DP (Biy+25, NAACL 2025) |
| `model_type` | `"albert"` | Cross-layer sharing, ~70% fewer params than BERT/RoBERTa |
| `d_model` | 768 | ALBERT-base hidden dimension |
| `max_seq_len` | 256 | Covers typical social media posts |
| `dropout` | 0.2 | Standard regularization for hate classifier |
| `adversarial_levels` | `("pooler","token","head")` | Multi-level identity coverage |
| `adversary_hidden_dim` | 256 | Capacity for identity classification |
| `adversary_num_layers` | 3 | Depth for learned adversary |
| `adversary_dropout` | 0.3 | Higher than hate head to prevent overfitting to noisy labels |
| `num_authors` | 100 | Pseudo-author classes for disentanglement |
| `alpha_initial` | 0.1 | Minimal adversarial pressure during warmup |
| `alpha_final` | 1.0 | Full disentanglement at end of training |
| `alpha_schedule` | `"sigmoid"` | Gradual ramp, steep mid-training transition |
| `alpha_warmup_epochs` | 2 | Learn hate features before adversarial pressure |
| `disentanglement_weight` | 0.3 | Balances hate accuracy vs identity removal |
| `mim_weight` | 0.1 | MINE-based MI minimization contribution |
| `orthogonality_weight` | 0.05 | Regulates hate/identity subspace overlap |
| `dp_enabled` | `True` | Master switch for DP-SGD |
| `target_epsilon` | 8.0 | Moderate privacy budget |
| `max_grad_norm` | 1.0 | Per-sample gradient clipping norm |
| `batch_size` | 16 | Per-GPU batch (limited by ghost clipping memory) |
| `learning_rate` | 2e-5 | Standard fine-tuning LR |
| `lr_schedule` | `"linear"` | Linear warmup + decay |
| `warmup_ratio` | 0.1 | 10% of steps for LR warmup |
| `weight_decay` | 0.01 | AdamW regularization |
| `num_epochs` | 10 | Sufficient for convergence with DP |
| `mixed_precision` | `"fp16"` | Mixed precision for speed; bf16 also supported |
| `seed` | 42 | Reproducibility |

## Recommended training recipe

| Setting | Value | Notes |
|---|---|---|
| Optimizer | AdamW | β₁=0.9, β₂=0.999, ε=1e-8 |
| Peak LR | 2e-5 | Linear warmup over 10% of steps, cosine decay |
| Batch size | 16 | Gradient accumulation if needed for larger effective batch |
| Weight decay | 0.01 | Excluded from bias/LayerNorm parameters |
| Grad clip (DP) | 1.0 | Per-sample L2 norm; handled by Opacus |
| Grad clip (MINE) | 1.0 | Separate clip for MINE network stability |
| Precision | fp16 mixed | Master weights in fp32; Opacus handles autocast |
| Privacy budget | ε=8.0, δ=1e-5 | Moderate; sweep ε ∈ {1,2,4,8,16,32} for analysis |
| Ghost clipping | Enabled | Memory-efficient per-sample gradients (Opacus) |
| Poisson sampling | Enabled | Tighter DP accounting via subsampling amplification |

### Standard training command

```bash
python train.py \
    --model albert-base-v2 \
    --dataset jigsaw \
    --target-epsilon 8.0 \
    --adversarial \
    --alpha-schedule sigmoid \
    --epochs 10 \
    --batch-size 16 \
    --learning-rate 2e-5
```

### Privacy budget sweep

```bash
python run_experiments.py --ablation privacy
```

### Adversarial ablation (2×2 factorial)

```bash
python run_experiments.py --ablation adversarial
```

### Full experiment suite

```bash
python run_experiments.py --full
```

### Quick sanity check (CI-friendly, ~2 min on CPU)

```bash
python run_experiments.py --quick
```

### All single-field ablations

```bash
python ablation_runner.py --ablation all
```

## Domain-specific training considerations

### Privacy-Preserving ML

- **DP-SGD batch size:** Per-sample gradient clipping is memory-intensive. Batch size 16 is the default. Use gradient accumulation (e.g., `gradient_accumulation_steps=4`) for an effective batch of 64 without increasing memory.
- **Epsilon accounting:** Each epoch consumes privacy budget. At ε=8, the model can train for ~5-10 epochs depending on noise multiplier. Monitor with `PrivHSDTrainer.get_privacy_spent()`.
- **Ghost clipping:** If ghost clipping fails (memory issues), the trainer falls back to standard per-sample clipping. Expect ~3× slower training but correct DP guarantees.
- **Module validation:** Opacus's `ModuleValidator` checks for DP-incompatible layers (BatchNorm, InstanceNorm, tied weights). The PrivHSD model uses only LayerNorm and Linear layers, which are DP-compatible.

### LM (Hate Speech Detection)

- **Sequence length:** 256 tokens max. Social media posts are typically shorter, but truncation preserves stylistic patterns in the first 256 tokens.
- **Label balance:** Hate speech datasets are typically imbalanced (~10% toxic). The binary cross-entropy handles this implicitly; no class weighting is applied by default.
- **Data augmentation:** Privacy-augmented variants (label noise, word dropout) may reduce F1 by 1-3% but improve privacy metrics. Start with `--privacy-augment medium`.

### Ethical AI

- **Inference privacy mode:** The inference engine (`PrivHSDInference`) defaults to `privacy_mode=True`, which prevents text logging and limits output detail.
- **Representation audit:** Run `RepresentationPrivacyAudit` on test-set representations to verify identity-agnostic properties (entropy, k-anonymity, leakage score).

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Loss NaN in first steps | bf16/fp16 overflow in attention | Use fp32 for loss computation; Opacus handles autocast |
| Opacus `ModuleValidator` errors | BatchNorm or tied weights in custom head | Use only LayerNorm + Linear in all heads |
| Ghost clipping OOM | Multi-level adversaries + transformer exceed GPU memory | Reduce `adversary_hidden_dim` to 128, or disable ghost clipping |
| MINE loss not decreasing | MINE network overfitting | Reduce MINE learning rate to 5e-5, increase clipping to 0.5 |
| Adversary loss = 0 immediately | Alpha too low or adversary too strong | Check alpha schedule; reduce `adversary_num_layers` to 2 |
| Validation F1 << Train F1 | Overfitting due to DP noise | Increase ε (weaker privacy), or use privacy-augmented data |
| DP training very slow | Standard per-sample clipping fallback | Ensure ghost clipping works; check Opacus version compatibility |
| Author loss not decreasing | Pseudo-author labels too noisy | Check `create_author_labels` feature distribution; increase `num_authors` |
| Stylometry acc ≈ random | Disentanglement working correctly | Good! This is the desired outcome — model is identity-agnostic |
| Stylometry acc ≈ raw text baseline | Disentanglement not working | Increase `disentanglement_weight` or check adversary gradient flow |
