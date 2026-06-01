# Architecture

## 1. Motivation

Hate speech detection (HSD) is essential for healthier online discourse and regulatory compliance (EU Digital Services Act, Council of Europe standards). However, naive HSD models pose severe privacy risks. Writing style is a unique behavioral biometric — stylometric re-identification achieves >90% accuracy given sufficient candidate authors (Abbasi & Chen, 2008; Narayanan et al., 2012). Model representations can leak protected attributes (gender, age, location) even when not explicitly modeled, and membership inference attacks can determine whether a specific text was in the training set (Shokri et al., 2017; Makroo et al., 2025).

These risks transform HSD tools from protective instruments into surveillance infrastructure, potentially violating GDPR principles (data minimization, purpose limitation, privacy by design) and chilling free expression — exactly the democratic values hate speech regulation aims to protect.

**The central hypothesis** of PrivHSD v2 is that combining differential privacy via DP-SGD with multi-level adversarial identity disentanglement and mutual information minimization yields a Pareto-optimal privacy-utility trade-off that neither mechanism achieves independently: strong hate speech detection performance (F1, AUC) with provably minimal leakage of author identity signals (epsilon, MIA AUC, stylometry accuracy).

## 2. At a glance

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PrivHSD v2 Architecture                            │
│    Multi-Level Adversarial Disentanglement + DP-SGD + Mutual Information   │
│                               Minimization                                  │
└─────────────────────────────────────────────────────────────────────────────┘

Input Text
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Tokenizer (HuggingFace AutoTokenizer)                                  │
│  Max length = 256, padding, truncation                                  │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Transformer Backbone (ALBERT / RoBERTa / XLM-RoBERTa)                  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Embedding Layer                                                  │   │
│  │  (Token + Position + Segment Embeddings) → d_model=768           │   │
│  └──────────────────────────┬───────────────────────────────────────┘   │
│                             ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Transformer Layers × L (L=12 for base)                          │   │
│  │  ┌────────────────────────────────────────────────────────────┐  │   │
│  │  │  LayerNorm → Multi-Head Attention → Residual → LN → FFN   │  │   │
│  │  └────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────┬───────────────────────────────────────┘   │
│                             │                                           │
│                             ▼                                           │
│  Outputs:                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │
│  │ pooler_repr  │  │ token_repr   │  │ head_repr    │                 │
│  │ [CLS] pooled │  │ all tokens   │  │ per-head avg │                 │
│  │ (B, d_model) │  │ (B, T, d)    │  │ (B, H, d/H)  │                 │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                 │
└─────────┼─────────────────┼─────────────────┼──────────────────────────┘
          │                 │                 │
          │      ┌──────────┴──────────┐      │
          │      ▼                     ▼      │
          │  ┌──────────────────────────────────────────────┐
          │  │  Multi-Level Adversarial Disentanglement     │
          │  │  ┌────────────┐  ┌────────────┐  ┌────────┐ │
          │  │  │ Pooler Adv│  │ Token Adv  │  │Head Adv│ │
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
┌─────────────────────────────────────────────────────────────────────────┐
│  Hate Speech Classification Head                                       │
│  Linear(d_model, d_model) → LayerNorm → ReLU → Dropout → Linear(2)     │
│  L_hate = CE(hate_pred, hate_true)                                      │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Mutual Information Minimization (MINE)                                 │
│  ┌────────────────────────────────────────────────────┐                 │
│  │  MINE Network: MLP(repr ⊕ author_embed) → T       │                 │
│  │  I_est = E_joint[T] - log(E_marginal[exp(T)])     │                 │
│  └──────────────────────┬─────────────────────────────┘                 │
│                         ▼                                              │
│  L_mim = λ_mim * I_est(repr; author)  [encoder minimizes]              │
│  L_mine = -I_est  [MINE network maximizes]                             │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Combined Training Objective                                            │
│  L_total = L_hate + λ_dis * L_adv + λ_mim * I_est + λ_orth * L_orth    │
│  α(t) = sigmoid_schedule(t)  [Adaptive Gradient Reversal Scheduling]    │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  DP-SGD via Opacus (ghost clipping)                              │   │
│  │  ├── Per-sample gradient computation                              │   │
│  │  ├── Per-layer adaptive clipping (interface)                      │   │
│  │  ├── Gaussian noise N(0, σ²C²I)                                  │   │
│  │  └── RDP accounting (ε, δ tracking)                              │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Evaluation & Privacy Attack Suite                                      │
│  ┌─────────────────────────┐  ┌──────────────────────────────────────┐  │
│  │ Utility Metrics         │  │ Privacy Attack Suite                  │  │
│  │ ├── F1 Score            │  │ ├── Membership Inference (shadow)    │  │
│  │ ├── AUC-ROC             │  │ ├── Attribute Inference (logistic)   │  │
│  │ ├── Precision/Recall    │  │ ├── Stylometry Re-identification     │  │
│  │ ├── MCC                 │  │ │   ├── Raw text features (baseline) │  │
│  │ └── Specificity         │  │ │   └── Model reps (leakage)        │  │
│  │                         │  │ └── Representation Privacy Audit     │  │
│  └─────────────────────────┘  │   ├── Representation entropy        │  │
│                                │   ├── k-anonymity fraction         │  │
│  ┌─────────────────────────┐  │   └── Privacy leakage score        │  │
│  │ Pareto Frontier Analysis │  └──────────────────────────────────────┘  │
│  │ ε vs F1 vs MIA AUC      │                                           │
│  └─────────────────────────┘                                           │
└─────────────────────────────────────────────────────────────────────────┘

                        ┌──────────────────────┐
                        │  Rights-Based Design  │
                        │  ├── GDPR Art. 25     │
                        │  ├── Data Minimization│
                        │  ├── Purpose Limitation│
                        │  ├── ECHR Art. 10     │
                        │  └── DSA Compliance   │
                        └──────────────────────┘
```

| Property | Value |
|---|---|
| Parameter count (default ALBERT-base-v2 config) | ~12M |
| Parameter count (PrivHSDModelV2, all modules) | ~15.3M |
| Time complexity (forward) | O(B · T · D² · L) — standard transformer |
| Time complexity (DP training step) | O(3× forward) due to ghost clipping |
| Space complexity | O(B · T · D + B · params) ghost clipping |
| Hardware minimum | 12 GB GPU (A100 recommended for ghost clipping) |

## 3. The core component

### 3.1 Multi-Level Adversarial Disentanglement (MLAD)

**Intuition:** Author identity signals are distributed across multiple levels of representation in a transformer. High-level topic choices and sentiment tendencies appear in the [CLS] pooled vector. Word-level stylistic patterns — function-word frequency, punctuation habits — are encoded in per-token representations. Attention-head patterns reveal syntactic preferences that serve as behavioral biometrics.

A single adversary operating on the [CLS] vector leaves residual identity information at the token and head levels. MLAD deploys three separate adversarial heads — one per representation level — each with its own gradient reversal layer. The encoder must strip identity from all three levels simultaneously, producing representations that are comprehensively identity-agnostic.

**Reference:** Extends the domain-adversarial training of Ganin & Lempitsky (ICML 2015) to multi-level identity disentanglement for privacy-preserving NLP.

### 3.2 Adaptive Gradient Reversal Scheduling (AGRS)

**Intuition:** Applying full adversarial pressure from the first training step is destructive — the encoder needs to learn hate-relevant features before it can be forced to forget identity signals. AGRS uses a sigmoid schedule that keeps alpha near zero during a warmup phase (epochs 0-2), ramps alpha from 0.1 to 1.0 during a productive competition phase (epochs 2-6), and sustains maximal disentanglement pressure thereafter.

An optional adaptive variant monitors encoder and adversary loss dynamics: if the adversary is winning (loss decreasing rapidly), alpha increases; if hate loss rises (disentanglement damaging utility), alpha decreases.

### 3.3 Mutual Information Minimization via MINE

**Intuition:** The adversarial loss only removes identity features that the *current* adversary can detect. A stronger adversary might find residual information. Mutual Information Neural Estimation (MINE) provides a lower-bound estimate of I(representation; author_id) that is adversary-independent. Minimizing this bound directly catches identity signals the adversary misses.

This creates a three-player minimax game:
- **Encoder** minimizes L_total = L_hate + λ_dis · L_adv + λ_mim · I_est + λ_orth · L_orth
- **Adversary** minimizes L_author (gradient reversed through GRL)
- **MINE network** maximizes I_est (upper bound by adversarial training)

### 3.4 Equations

**Combined training objective:**

```
L_total = L_hate(y_pred, y_true) + λ_dis · L_author(a_pred, a_true) + λ_mim · I(repr; author) + λ_orth · orth(hate_feats, adv_feats)
```

where:
- L_hate = CE(hate_logits, hate_labels) — cross-entropy for binary hate classification
- L_author = CE(author_logits, author_labels) — cross-entropy for pseudo-author classification (gradient reversed through GRL)
- I(repr; author) = sup_T E_P[T(repr, author)] - log(E_Q[exp(T(repr, author))]) — MINE lower bound (Belghazi et al., 2018)
- orth(a, b) = |cosine_sim(a, b)| — absolute cosine similarity between hate features and adversary features (encourages orthogonal subspaces)

**GRL forward/backward:**
```
forward:  GRL(x, α) = x
backward: ∂GRL/∂x = -α · ∂L/∂x
```

**DP-SGD update (per step):**
```
For each example i in batch B:
    g_i = ∇θ L(f(x_i), y_i)                    # per-sample gradient
    g̃_i = g_i / max(1, ||g_i||₂ / C)           # clip to norm C
    g_avg = (1/|B|) · (Σ_i g̃_i + N(0, σ²C²I)) # average + noise
    θ ← optimizer_step(θ, g_avg)
```

**Privacy accounting (RDP → ε):**
```
RDP_λ(mechanism) = λ · σ² / 2                      # RDP for Gaussian mechanism
ε(δ) = min_λ [ RDP_λ + (log(1/δ) - log(λ)) / (λ-1) - log(λ) / (λ-1) ]  # optimal conversion
```

### 3.5 Reference implementation walk-through

```python
def forward(self, input_ids, attention_mask, hate_labels=None, author_labels=None, ...):
    # ── Transformer backbone ────────────────────────────────────
    # input_ids: (B, T)         attention_mask: (B, T)
    transformer_out = self.get_transformer_outputs(input_ids, attention_mask)
    pooler_repr = transformer_out["pooler_repr"]   # (B, D) — [CLS] pooled
    last_hidden = transformer_out["last_hidden"]   # (B, T, D) — all tokens
    head_repr = transformer_out["head_repr"]       # (B, D) — attention head avg

    # ── Hate speech classification ──────────────────────────────
    # Linear(D, D) → LayerNorm → ReLU → Dropout → Linear(D, 2)
    hate_logits = self.hate_classifier(pooler_repr)   # (B, 2)
    hate_probs = F.softmax(hate_logits, dim=-1)       # (B, 2)

    # ── Multi-level adversarial disentanglement ────────────────
    # Three adversaries: pool, token, head. Each:
    #   GRL_forward: identity   GRL_backward: -alpha * grad
    #   AdversaryMLP: Linear → LN → ReLU → Dropout → ... → Linear(n_authors)
    adv_outputs = self.mlad_block(
        pooler_repr=pooler_repr,          # (B, D)
        token_repr=last_hidden,            # (B, T, D)
        head_repr=head_repr,               # (B, D)
        author_labels=author_labels,       # (B,)
        alpha=current_alpha,               # scalar ∈ [0.1, 1.0]
    )
    author_loss = adv_outputs["author_loss"]  # scalar (mean of 3 CE losses)

    # ── Mutual information minimization ─────────────────────────
    if self.mim_module is not None:
        mi_estimate = self.mim_module(
            pooler_repr, author_labels, maximize=False
        )  # scalar — lower-bound I(repr; author)
        mim_loss = config.mim_weight * mi_estimate

    # ── Orthogonality regularization ────────────────────────────
    hate_feats = self.hate_classifier.get_features(pooler_repr)  # (B, D)
    adv_feats = adv_outputs["adversary_features"]["pooler"]      # (B, D)
    orth_loss = config.orthogonality_weight * \
        |cosine_sim(hate_feats, adv_feats)|

    # ── Combined objective ──────────────────────────────────────
    loss = hate_loss + λ_dis * author_loss + mim_loss + orth_loss
```

Source: `src/model.py`, lines 814-928 (`PrivHSDModelV2.forward`).

## 4. Tensor shape evolution

| Stage | Shape | Notes |
|---|---|---|
| Input | (B, T) | B=batch, T=seq_len (max 256); dtype=torch.long |
| Token embeddings | (B, T, D) | D=d_model (768 for base); dtype depending on precision |
| Transformer output | (B, T, D) | Last hidden state, all positions |
| pooler_repr | (B, D) | [CLS] token from last hidden state |
| token_repr | (B, T, D) | All token representations (mean-pooled for adversary: (B, D)) |
| head_repr | (B, D) | Attention-pattern-based aggregated per-head output |
| Hate classifier features | (B, D) | Pre-classification hidden (for orthogonality regularization) |
| Hate logits | (B, 2) | Binary classification logits |
| Hate probs | (B, 2) | Softmax probabilities |
| Adversary features | (B, hidden_dim) | Pre-projection features from each adversary MLP |
| Author logits | (B, num_authors) | Per-level adversarial predictions |
| MI estimate | scalar | MINE lower-bound I(repr; author) |

## 5. Design decisions

| Decision | Alternative considered | Why we chose this | Trade-off accepted |
|---|---|---|---|
| **ALBERT backbone (default)** | RoBERTa, BERT, XLM-RoBERTa | Cross-layer parameter sharing reduces params by ~70%, yielding 2-5% higher F1 at ε ≤ 4 under DP (Biy+25, NAACL 2025) | Slightly lower absolute accuracy at ε=∞ compared to RoBERTa |
| **Multi-level adversarial heads** | Single [CLS] adversary | Identity signals at token and head levels would be missed | 3× adversary parameters; memory overhead |
| **Sigmoid alpha schedule** | Linear schedule, fixed alpha | Sigmoid provides steep mid-training ramp when both encoder and adversary are competent | Requires warmup epochs tuning |
| **MINE-based MI minimization** | Adversary alone | Catches residual identity signals the adversary misses | Three-player game instability risk |
| **DP-SGD via Opacus** | Custom DP implementation | Production-quality ghost clipping, RDP accounting, Poisson sampling | Opacus version constraints |
| **Ghost clipping** | Standard per-sample clipping | O(batch × activations) vs O(batch × params) memory — 3-5× reduction | Requires GradSampleModule compatibility |
| **Pre-norm (LN before sublayer)** | Post-norm | Stabilizes training at DP batch sizes (16) | Slightly different convergence properties |
| **Poisson sampling** | Fixed-size batches | Tighter DP accounting via subsampling amplification theorem | Variable batch sizes complicate throughput |
| **Synthetic pseudo-author labels** | Ground-truth author IDs | Author metadata rarely available in real deployments; stylistic feature hashing provides unsupervised proxy | Cluster quality depends on feature engineering |
| **Representation orthogonality** | No regularization | Encourages hate/identity subspaces to be orthogonal, reducing residual leakage | Additional loss term to tune |

## 6. Domain-specific considerations

### 6.1 Language Modeling (LM)

**Position encoding:** ALBERT uses learned absolute position embeddings. Position embeddings may encode signature-like positional patterns (e.g., where a user places their sign-off). Pooled representations aggregate across positions, mitigating this.

**Causal constraint:** Hate speech classification is bidirectional (non-autoregressive). No causal masking is applied — full context is available for each prediction.

**Sequence length:** 256 tokens max. Sufficient for social media posts (~280 chars = ~50 tokens) with margin. Longer documents are truncated; mean-pooled token representations still capture distributed stylistic patterns.

### 6.2 Privacy-Preserving ML

**Formal privacy guarantee:** DP-SGD provides (ε, δ)-DP where ε is the privacy budget and δ is the failure probability. δ = 1/|D| per standard practice. Privacy budget is tracked via Rényi DP accounting for tighter composition bounds.

**Per-sample gradient clipping:** The core DP operation. Without it, one outlier gradient could leak information about a single training example. Opacus ghost clipping makes this tractable for transformer architectures.

**Adversarial + DP complementarity:** DP bounds what the model can leak about any individual training example (worst-case). Adversarial disentanglement makes representations invariant to author identity (average-case). Together they cover both regimes.

### 6.3 Ethical AI / Rights-Based Design

**Anti-surveillance by design:** The architecture intentionally prevents authorship attribution. If an adversary obtains white-box access, representations will be identity-agnostic due to the adversarial training objective. The inference engine never returns representations.

**No re-identification tooling:** The privacy attack suite is implemented strictly for evaluation. No auxiliary re-identification pipeline is released as production tooling.

**Regulatory alignment:**
- **GDPR Art. 25** (Privacy by Design): DP + adversarial disentanglement built into architecture
- **EU Digital Services Act:** Enables proportionate hate speech moderation
- **ECHR Art. 10:** Protects freedom of expression by limiting surveillance capability
- **Council of Europe:** Democratic values — safety without mass surveillance

### 6.4 Multilingual extension

XLM-RoBERTa is supported as a backbone for multilingual hate speech detection. The MLAD block is model-agnostic. Hash-based pseudo-author labeling uses character-level features that are language-agnostic. Low-resource languages may require adjusted noise multipliers (smaller datasets → larger δ or higher ε).

## 7. Known limitations

- **No empirical results yet** (projected from literature) — all H1-H4 hypotheses require experiment execution (`python run_experiments.py --full`) to validate; see [BENCHMARKS.md](BENCHMARKS.md) for status.
- **Synthetic author labels** may limit disentanglement effectiveness vs. ground-truth authorship data.
- **Computational cost** — DP-SGD with ghost clipping increases training time by 3-5× compared to non-DP training.
- **Dataset scope** — primarily English; multilingual validation (DE/FR/IT/ES) not yet implemented despite European hackathon focus.
- **Federated learning** — not yet integrated; the TUM/MCML FL+DP framework (Biy+25) provides a natural extension path but is not implemented.
- **MINE network instability** — three-player game convergence is not theoretically characterized; MINE may require careful tuning.
- **Per-layer DP clipping** — Opacus integration may fall back to uniform clipping; per-layer benefit is unverified.
- **No multi-seed experiments** — all configurations use a single seed (42); statistical significance pending.
