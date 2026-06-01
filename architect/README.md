# PrivHSD v2: Architecture Design

## Overview

Complete architecture specification for **PrivHSD v2**, a privacy-preserving hate speech detection system with multi-level adversarial disentanglement, differential privacy, and mutual information minimization.

## Files

| File | Description |
|---|---|
| `ARCHITECTURE_SPEC.md` | Full architecture specification: innovations, pseudocode, diagrams, bias justifications, traceability, risks, and ablations |
| `model_v2.py` | Reference implementation: `PrivHSDModelV2` with MLAD, AGRS, MINE, and backward-compatible `PrivHSDModel` alias |

## Core Innovations (v2 over v1)

1. **Multi-Level Adversarial Disentanglement (MLAD)** — Adversaries at pooler, token, and head levels
2. **Adaptive Gradient Reversal Scheduling (AGRS)** — Sigmoid/adaptive alpha schedules
3. **Mutual Information Minimization (MINE)** — Direct I(repr; author) estimation and minimization
4. **Per-Layer Adaptive DP Clipping** — Layer-specific clip norms for better utility at same ε
5. **Representation Orthogonality Regularization** — Orthogonal hate/identity subspaces

## Research-to-Architecture Traceability

The traceability table in `ARCHITECTURE_SPEC.md` maps every research contract item (from `RESEARCH_CONTRACT.yaml`) to implementation decisions, evidence status, and validation hooks.

## Backward Compatibility

`model_v2.py` provides `PrivHSDModel` as a subclass of `PrivHSDModelV2` with the same constructor signature as v1. The existing training/evaluation pipeline can import `PrivHSDModel` with zero code changes.

## Next Steps

The architecture is ready for implementation. The coder agent should:

1. Replace `src/model.py` with `model_v2.py` (or merge the MLAD/AGRS/MIM blocks)
2. Update `PrivHSDConfig` in `train.py` to include new v2 fields
3. Add MINE optimizer to the training loop
4. Implement per-layer clipping in the Opacus integration
5. Validate backward compatibility with existing experiment scripts
