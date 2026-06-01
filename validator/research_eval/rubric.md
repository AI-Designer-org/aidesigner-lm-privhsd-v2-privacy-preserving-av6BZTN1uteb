# PrivHSD v2 — Research Quality Evaluation Rubric

## Overview

This rubric scores the PrivHSD v2 research artifact across six dimensions
commonly used in AI-researcher-quality evaluation (adapted from standard
ML conference reviewing criteria). Each dimension is scored 0-5.

**Project:** Privacy-preserving Hate Speech Detection (PrivHSD) for Council of Europe
Democracy Hackathon 2026

**Domain:** LM + Privacy ML + Ethical AI

**Task Level:** Level 2 (papers/context provided — full literature review
and prototype implementation)

---

## Score Scale

| Score | Meaning |
|---|---|
| 0 | Not addressed or no artifact exists |
| 1 | Mentioned but unsupported |
| 2 | Partially supported with major gaps |
| 3 | Plausible and minimally supported |
| 4 | Strong, with clear evidence and reproducible checks |
| 5 | Publication-ready for this scaffold's scope |

---

## 1. Novelty (0-5)

**Key questions:**
- Does the architecture propose a genuinely new combination or mechanism?
- Is the novelty clearly distinguished from existing work?
- Are the novelty claims appropriately scoped and falsifiable?

**LM-specific:**
- Does the benchmark suite test long-context recall, causal masking, induction behavior,
  throughput/memory, and at least one strong baseline?
- **PrivHSD-specific:** Since HSD is non-autoregressive (bidirectional), causal masking
  and induction-head tests are not applicable. Instead, test: disentanglement effectiveness,
  privacy-utility Pareto frontier, stylometry reduction, and MIA resistance.

**Score: 4** — Strong novelty with appropriate scope
- First systematic combination of DP-SGD + multi-level adversarial disentanglement + MINE
  for hate speech detection
- Novel AGRS (adaptive gradient reversal scheduling) extends Ganin & Lempitsky 2015
  with sigmoid schedule and adaptive feedback
- Three-player minimax game formulation (encoder × adversary × MINE) is well-motivated
- Concern: individual components are known; novelty is in the combination and domain application

---

## 2. Experimental Comprehensiveness (0-5)

**Key questions:**
- Are all relevant baselines and ablations implemented?
- Is the evaluation multi-dimensional (privacy + utility)?  
- Are the results statistically meaningful?

**LM-specific:** Tests invariance/equivariance claims, resolution behavior, feature quality,
and compares against a simple baseline (non-DP BERT/ViT-style).

**Privacy ML-specific:**
- DP budget sweep across ε ∈ {1, 2, 4, 8, 16, 32} ✓
- Component ablation: 2×2 factorial (DP × adversarial) ✓
- MIA evaluation (shadow model + threshold) ✓
- Attribute inference on representations ✓
- Stylometry re-identification (raw text + model reps) ✓
- Representation entropy, k-anonymity, privacy leakage score ✓
- Architecture comparison: ALBERT vs RoBERTa ✓
- Privacy-augmented data variants (low/medium/high) ✓

**Score: 4** — Comprehensive coverage with minor gaps
- All required baselines implemented
- 11 ablations covering all architect-specified config changes
- Full privacy attack suite
- Gap: no multi-seed experiments yet; no multilingual evaluation; no federated extension

---

## 3. Theoretical Foundation (0-5)

**Key questions:**
- Is the architecture theoretically motivated?
- Are the inductive biases clearly articulated?
- Are the privacy guarantees formally stated?

**Score: 4** — Strong theoretical grounding
- DP-SGD: formal (ε, δ)-DP via RDP accounting ✓
- GRL: minimax formulation with convergence theory ✓
- MINE: mutual information lower bound from Belghazi+18 ✓
- Inductive biases documented in ARCHITECTURE_SPEC.md ✓
- Per-layer clipping: empirically motivated but not formally bounded
- Concern: three-player game convergence not theoretically characterized

---

## 4. Result Analysis (0-5)

**Key questions:**
- Are the results clearly presented and interpreted?
- Is the privacy-utility trade-off quantified?
- Are the findings compared to stated hypotheses?

**Score: 2** — Infrastructure exists but no empirical results yet
- All evaluation infrastructure implemented (compute_utility_metrics, ParetoFrontierAnalyzer,
  attack evaluation, ablation delta computation)
- Expected/projected results documented in RESEARCH_NOTE.md
- No actual empirical results collected yet (pending experiment execution)
- No statistical significance testing (requires multi-seed runs)
- Pareto analysis uses simple dominance — could use multi-objective metrics (hypervolume)

---

## 5. Implementation Reproducibility (0-5)

**Key questions:**
- Can the results be reproduced from the codebase?
- Are random seeds fixed?
- Is the environment specified?
- Are there CI-friendly test modes?

**Score: 4** — Strong reproducibility
- Fixed seed (42) across torch, numpy, random ✓
- Config dataclass as single source of truth ✓
- smoke_test.py works with synthetic data, no HF download ✓
- test_model.py with 45+ tests, marked benchmark/needs_hf categories ✓
- Ablation runner with config diff reporting ✓
- requirements.txt with dependencies ✓
- Quick mode (--quick) for CI-friendly verification ✓
- Gap: no Dockerfile or environment lock file; requires specific PyTorch/Opacus version combo

---

## 6. Writing Readiness (0-5)

**Key questions:**
- Is the research narrative well-structured and complete?
- Are ethical considerations addressed?
- Are limitations documented?
- Is the work publication-ready?

**Score: 4** — Well-structured with clear narrative
- RESEARCH_NOTE.md covers motivation, architecture, methodology, ethics ✓
- Rights-based architecture aligned with GDPR/DSA/ECHR ✓
- Ethical considerations: dual-use prevention, dataset biases, societal impact ✓
- Limitations: 5 documented limitations ✓
- Publication plan with 5 target venues ✓
- Gap: empirical results section is projected; no formal "Related Work" section
- Gap: results needed before publication submission

---

## Domain-Specific Research Questions (LM + Privacy ML)

### LM Domain

| Question | Status | Evidence |
|---|---|---|
| Does the benchmark suite test causal masking? | N/A — HSD is bidirectional, not autoregressive | No causal masking needed |
| Does it test throughput/memory? | ✓ Implemented | `profile_model.py` with FLOP estimation, CUDA memory tracking |
| Does it test at least one strong baseline? | ✓ Implemented | Non-DP ALBERT + RoBERTa baselines in `run_experiments.py` |
| Does it test long-context behavior? | ✓ Partial | Variable sequence length tests in `test_model.py` (T=16,32,64) |

### Privacy ML Domain

| Question | Status | Evidence |
|---|---|---|
| DP budget sweep across ε values? | ✓ Implemented | ε ∈ {1, 2, 4, 8, 16, 32} in `get_privacy_sweep_configs()` |
| MIA evaluation? | ✓ Implemented | `MembershipInferenceAttack` (shadow + threshold) |
| Attribute inference? | ✓ Implemented | `AttributeInferenceAttack` on representations |
| Stylometry re-identification? | ✓ Implemented | `StylometryReidentificationRisk` (raw + model reps) |
| Representation audit? | ✓ Implemented | `RepresentationPrivacyAudit` (entropy, k-anonymity, leakage) |
| Pareto frontier analysis? | ✓ Implemented | `ParetoFrontierAnalyzer` with plotting |
| Ablation study? | ✓ Implemented | 11 single-field ablations in `ablation_runner.py` |

### Ethical AI Domain

| Question | Status | Evidence |
|---|---|---|
| Rights-based architecture documented? | ✓ Yes | GDPR + DSA + ECHR alignment in RESEARCH_NOTE.md and ARCHITECTURE_SPEC.md |
| Anti-surveillance by design? | ✓ Yes | Representations never returned by default (tested in test_model.py) |
| Dual-use prevention? | ✓ Discussed | Ethical considerations section; no re-identification tooling released |
| Dataset biases acknowledged? | ✓ Yes | Language, annotation, demographic biases documented |
| Limitations documented? | ✓ Yes | 5 limitations in RESEARCH_NOTE.md |

---

## Overall Assessment

| Dimension | Score | Key Strength | Key Gap |
|---|---|---|---|
| Novelty | 4/5 | First DP+MLAD+MINE combination for HSD | Components individually known |
| Experimental | 4/5 | Comprehensive privacy attack suite | No multi-seed / multilingual runs |
| Theoretical | 4/5 | Formal DP guarantees, rigorous MINE | Three-player game convergence uncharacterized |
| Result Analysis | 2/5 | Infrastructure ready | No empirical results yet |
| Reproducibility | 4/5 | seed=42, synthetic data CI path | No Dockerfile |
| Writing | 4/5 | Strong ethics, structured narrative | Related Work section needed |
| **Overall** | **3.7/5** | | |

The artifact is strong in architecture, implementation, and evaluation infrastructure.
The primary gap is the lack of empirical results, which is expected for a pre-experiment
prototype. Running the recommended experiments will close most gaps.
