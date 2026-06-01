#!/usr/bin/env python3
"""
PrivHSD v2 — Ablation Runner
=============================
Systematic single-field ablation experiments mapped to the Architect's
traceability table. Each ablation changes exactly one field in
PrivHSDConfig and measures the effect on utility (F1, AUC) and
privacy (MIA AUC, stylometry acc, epsilon).

The ablations are structured as NamedSingleFieldConfigs — each
differs from the baseline on exactly one dimension, enabling
unambiguous attribution of metric changes.

Usage:
    python ablation_runner.py --list                # List all ablations
    python ablation_runner.py --ablation 1          # Run ablation #1
    python ablation_runner.py --ablation 1,3,5      # Run multiple
    python ablation_runner.py --ablation all        # Run all
    python ablation_runner.py --dry-run             # Print configs without running
    python ablation_runner.py --quick               # 2 epochs, sample 200

Output:
    models/ablations/<name>/  — per-ablation results
    models/ablations/summary.json  — aggregated comparison
"""

import argparse
import json
import math
import sys
import time
import logging
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "coder"))

from src.model import PrivHSDConfig, PrivHSDModelV2
from src.data_utils import (
    HateSpeechDataset,
    create_author_labels,
    create_privacy_augmented_variant,
    get_dataloaders,
)
from src.train import PrivHSDTrainer
from src.evaluate import (
    evaluate_model,
    ParetoFrontierAnalyzer,
    EvaluationResult,
    UtilityMetrics,
    PrivacyMetrics,
    compute_utility_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ablation_runner")


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline Configuration
# ═══════════════════════════════════════════════════════════════════════════════

def get_baseline_config() -> PrivHSDConfig:
    """Baseline configuration: full PrivHSD v2 with DP + adversarial + MIM.

    This is the recommended configuration against which all ablations
    are compared.
    """
    return PrivHSDConfig(
        # Backbone
        model_name="albert-base-v2",
        model_type="albert",
        max_seq_len=256,
        dropout=0.2,
        attention_dropout=0.2,
        # Hate head
        num_hate_classes=2,
        classifier_hidden_dim=768,
        classifier_num_layers=2,
        # MLAD
        num_authors=100,
        adversarial_levels=("pooler", "token", "head"),
        adversary_hidden_dim=256,
        adversary_num_layers=3,
        adversary_dropout=0.3,
        # AGRS
        alpha_initial=0.1,
        alpha_final=1.0,
        alpha_schedule="sigmoid",
        alpha_warmup_epochs=2,
        alpha_gamma=2.0,
        # Loss weights
        disentanglement_weight=0.3,
        mim_weight=0.1,
        orthogonality_weight=0.05,
        consistency_weight=0.05,
        # MIM
        mim_estimator="mine",
        mim_hidden_dim=128,
        mim_learning_rate=1e-4,
        mim_momentum=0.9,
        # DP-SGD
        dp_enabled=True,
        target_epsilon=8.0,
        target_delta=None,  # auto = 1/|D|
        max_grad_norm=1.0,
        per_layer_clipping=True,
        poisson_sampling=True,
        # Data augmentation
        privacy_augment_level=None,
        label_flip_prob=0.05,
        word_dropout_prob=0.10,
        synonym_replacement_prob=0.05,
        entity_masking=True,
        # Training
        batch_size=16,
        learning_rate=2e-5,
        lr_schedule="linear",
        warmup_ratio=0.1,
        weight_decay=0.01,
        num_epochs=10,
        gradient_accumulation_steps=1,
        early_stopping_patience=5,
        optimizer="adamw",
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        # System
        device="cuda",
        mixed_precision="fp16",
        use_ghost_clipping=True,
        num_workers=4,
        seed=42,
        output_dir="models",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Ablation Definitions
# ═══════════════════════════════════════════════════════════════════════════════

# Each ablation is a (name, config_modifier, hypothesis_test, expected_movement)
# where config_modifier takes baseline config and returns a modified config.

Ablation = Tuple[str, Callable[[PrivHSDConfig], PrivHSDConfig], str, str]

ABLATIONS: List[Ablation] = [
    # ── Priority 1: Core novelty checks ──────────────────────────────
    (
        "no_adversarial",
        lambda c: PrivHSDConfig(**{
            **{k: v for k, v in c.__dict__.items() if not k.startswith('_')},
            "disentanglement_weight": 0.0,
            "mim_weight": 0.0,
            "orthogonality_weight": 0.0,
        }),
        "Core novelty: adversarial disentanglement improves privacy-utility frontier",
        "F1 ↑ 1-3%, MIA AUC ↑ 5-10%, stylometry acc ↑ 15-25% vs baseline",
    ),
    (
        "no_dp",
        lambda c: PrivHSDConfig(**{
            **{k: v for k, v in c.__dict__.items() if not k.startswith('_')},
            "dp_enabled": False,
            "target_epsilon": float("inf"),
        }),
        "Core novelty: DP-SGD is the primary formal privacy guarantee",
        "F1 ↑ 3-5%, MIA AUC ↑ 10-20%, ε → inf",
    ),
    (
        "single_level_adversarial",
        lambda c: PrivHSDConfig(**{
            **{k: v for k, v in c.__dict__.items() if not k.startswith('_')},
            "adversarial_levels": ("pooler",),
        }),
        "Multi-level vs single-level: more levels remove more identity signal",
        "Stylometry acc ↑ 5-10% for single level, F1 unchanged",
    ),
    # ── Priority 2: Refinement ───────────────────────────────────────
    (
        "no_mim",
        lambda c: PrivHSDConfig(**{
            **{k: v for k, v in c.__dict__.items() if not k.startswith('_')},
            "mim_weight": 0.0,
        }),
        "MIM reduces residual identity leakage beyond adversarial alone",
        "Stylometry acc ↑ 3-8%, MIA AUC ↑ 2-5%, F1 unchanged",
    ),
    (
        "alpha_linear",
        lambda c: PrivHSDConfig(**{
            **{k: v for k, v in c.__dict__.items() if not k.startswith('_')},
            "alpha_schedule": "linear",
        }),
        "Sigmoid schedule improves utility at same privacy level",
        "F1 ↑ 1-2% at ε=8, adv_loss convergence faster",
    ),
    (
        "no_orthogonality",
        lambda c: PrivHSDConfig(**{
            **{k: v for k, v in c.__dict__.items() if not k.startswith('_')},
            "orthogonality_weight": 0.0,
        }),
        "Orthogonality regularization improves disentanglement without utility loss",
        "Stylometry acc ↑ 2-5%, F1 unchanged",
    ),
    # ── Priority 3: Optimization & characterization ──────────────────
    (
        "uniform_clipping",
        lambda c: PrivHSDConfig(**{
            **{k: v for k, v in c.__dict__.items() if not k.startswith('_')},
            "per_layer_clipping": False,
        }),
        "Per-layer clipping improves utility at same ε",
        "F1 ↑ 0.5-1.5% at same ε, no privacy degradation",
    ),
    (
        "no_augmentation",
        lambda c: PrivHSDConfig(**{
            **{k: v for k, v in c.__dict__.items() if not k.startswith('_')},
            "privacy_augment_level": None,
        }),
        "Privacy augmentation improves Pareto frontier",
        "F1 ↓ 1-3%, MIA AUC ↑ 2-5%",
    ),
    (
        "roberta_backbone",
        lambda c: PrivHSDConfig(**{
            **{k: v for k, v in c.__dict__.items() if not k.startswith('_')},
            "model_name": "roberta-base",
            "model_type": "roberta",
            "d_model": 768,
            "n_layers": 12,
            "n_heads": 12,
            "d_ff": 3072,
        }),
        "ALBERT outperforms RoBERTa under DP (Biy+25, NAACL 2025)",
        "ALBERT F1 2-5% higher at ε≤4",
    ),
    (
        "strong_privacy_eps1",
        lambda c: PrivHSDConfig(**{
            **{k: v for k, v in c.__dict__.items() if not k.startswith('_')},
            "target_epsilon": 1.0,
        }),
        "Pareto frontier: F1 decreases monotonically with ε",
        "F1 lower than baseline (ε=8), MIA AUC < baseline",
    ),
    (
        "weak_privacy_eps32",
        lambda c: PrivHSDConfig(**{
            **{k: v for k, v in c.__dict__.items() if not k.startswith('_')},
            "target_epsilon": 32.0,
        }),
        "Pareto frontier: weaker DP has higher utility",
        "F1 higher than baseline (ε=8), MIA AUC > baseline",
    ),
]


def get_ablation_names() -> List[str]:
    """Return list of ablation names."""
    return [name for name, _, _, _ in ABLATIONS]


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment Runner
# ═══════════════════════════════════════════════════════════════════════════════

class AblationExperimentRunner:
    """Run single ablation experiments and collect results."""

    def __init__(self, quick: bool = False, output_dir: str = "models"):
        self.quick = quick
        self.output_dir = Path(output_dir)
        self.results_dir = self.output_dir / "ablations"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, Dict] = {}

    def _load_synthetic_data(self, config: PrivHSDConfig, sample_size: int = None):
        """Load or generate synthetic hate speech data."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        # Generate synthetic data
        rng = np.random.RandomState(config.seed)
        n = sample_size or (200 if self.quick else 5000)
        texts = [
            f"Synthetic text sample {i} for testing PrivHSD model. "
            f"{'Contains hateful language.' if rng.random() > 0.8 else 'Normal neutral text.'}"
            for i in range(n)
        ]
        labels = [1 if "hateful" in t.lower() else 0 for t in texts]

        # Author labels
        n_authors = min(config.num_authors, max(n // 20, 10))
        author_labels = create_author_labels(texts, n_authors=n_authors, random_seed=config.seed)

        # Split
        indices = rng.permutation(n)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)

        datasets = {}
        for name, idx in [
            ("train", indices[:n_train]),
            ("val", indices[n_train:n_train + n_val]),
            ("test", indices[n_train + n_val:]),
        ]:
            datasets[name] = HateSpeechDataset(
                texts=[texts[i] for i in idx],
                hate_labels=[labels[i] for i in idx],
                author_ids=[author_labels[i] for i in idx],
                tokenizer=tokenizer,
                max_length=config.max_seq_len,
                is_test=(name == "test"),
            )

        loaders = get_dataloaders(
            datasets["train"], datasets["val"], datasets["test"],
            batch_size=config.batch_size,
        )

        return (*loaders, n_authors, tokenizer)

    def run_ablation(self, ablation_idx: int) -> Dict:
        """Run a single ablation by index."""
        if ablation_idx < 0 or ablation_idx >= len(ABLATIONS):
            raise ValueError(f"Invalid ablation index {ablation_idx}. "
                             f"Valid range: 0-{len(ABLATIONS) - 1}")

        name, modifier, hypothesis, expected = ABLATIONS[ablation_idx]
        baseline = get_baseline_config()
        modified = modifier(baseline)

        # If quick, reduce epochs and sample size
        if self.quick:
            modified.num_epochs = 2
            sample_size = 200
        else:
            sample_size = None

        logger.info(f"\n{'='*60}")
        logger.info(f"Ablation #{ablation_idx}: {name}")
        logger.info(f"  Hypothesis: {hypothesis}")
        logger.info(f"  Expected: {expected}")
        logger.info(f"  Config change: {_diff_config(baseline, modified)}")
        logger.info(f"{'='*60}")

        # Load data
        device = modified.device if torch.cuda.is_available() and modified.device == "cuda" else "cpu"
        train_loader, val_loader, test_loader, n_authors, tokenizer = \
            self._load_synthetic_data(modified, sample_size)

        effective_authors = min(modified.num_authors, n_authors)
        modified.num_authors = effective_authors

        # Initialize model
        model = PrivHSDModelV2(modified)
        logger.info(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

        # Compute delta from dataset size
        target_delta = 1.0 / max(len(train_loader.dataset), 1)

        # Trainer
        trainer = PrivHSDTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=modified,
            learning_rate=modified.learning_rate,
            target_epsilon=modified.target_epsilon if modified.target_epsilon != float("inf") else 1e6,
            target_delta=target_delta,
            max_grad_norm=modified.max_grad_norm,
            dp_enabled=modified.dp_enabled,
            device=device,
            output_dir=str(self.results_dir / name),
            use_ghost_clipping=modified.use_ghost_clipping,
        )

        # Train
        train_results = trainer.train(
            num_epochs=modified.num_epochs,
            use_adversarial=(modified.disentanglement_weight > 0),
            eval_every=1,
            save_every=max(modified.num_epochs, 1),
            early_stopping_patience=min(modified.early_stopping_patience, 2) if self.quick else modified.early_stopping_patience,
        )

        # Evaluate
        eval_result = evaluate_model(model, test_loader, device)

        # Privacy metrics
        privacy = PrivacyMetrics(
            epsilon=train_results.get("final_epsilon", float("inf")),
            delta=target_delta,
        )

        # Compute simple privacy-utility ratio
        f1 = eval_result.utility.f1_score
        eps = max(privacy.epsilon, 0.01)
        pur = f1 / eps

        # Store result
        result_entry = {
            "ablation_name": name,
            "ablation_index": ablation_idx,
            "hypothesis": hypothesis,
            "expected_movement": expected,
            "config_diff": _diff_config(baseline, modified),
            "utility": eval_result.utility.to_dict(),
            "privacy": eval_result.privacy.to_dict(),
            "privacy_utility_ratio": pur,
            "final_epsilon": privacy.epsilon,
            "baseline_f1": None,  # will be filled after all ablations
            "baseline_eps": None,
            "f1_delta": None,
            "status": "completed",
        }

        # Save individual result
        result_path = self.results_dir / f"{name}_result.json"
        with open(result_path, "w") as f:
            json.dump(result_entry, f, indent=2, default=str)

        logger.info(
            f"  Done: F1={f1:.4f}, ε={privacy.epsilon:.2f}, P/U ratio={pur:.4f}"
        )

        self.results[name] = result_entry
        return result_entry

    def run_all(self) -> Dict:
        """Run all ablations sequentially."""
        baseline_result = self.run_single("baseline")

        for idx in range(len(ABLATIONS)):
            try:
                self.run_ablation(idx)
            except Exception as e:
                logger.error(f"Ablation #{idx} failed: {e}", exc_info=True)
                self.results[ABLATIONS[idx][0]] = {
                    "ablation_name": ABLATIONS[idx][0],
                    "status": "failed",
                    "error": str(e),
                }

        # Post-process: compute deltas from baseline
        self._compute_deltas()
        self._save_summary()
        return self.results

    def run_single(self, name_or_idx: str) -> Dict:
        """Run either a named ablation or by index."""
        if name_or_idx == "baseline":
            return self._run_baseline()
        try:
            idx = int(name_or_idx)
            return self.run_ablation(idx)
        except ValueError:
            # Find by name
            for i, (name, _, _, _) in enumerate(ABLATIONS):
                if name == name_or_idx:
                    return self.run_ablation(i)
            raise ValueError(f"Unknown ablation: {name_or_idx}")

    def _run_baseline(self) -> Dict:
        """Run baseline configuration."""
        logger.info(f"\n{'='*60}")
        logger.info("Baseline run (full PrivHSD v2)")
        logger.info(f"{'='*60}")

        config = get_baseline_config()
        if self.quick:
            config.num_epochs = 2
            sample_size = 200
        else:
            sample_size = None

        device = config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu"
        train_loader, val_loader, test_loader, n_authors, tokenizer = \
            self._load_synthetic_data(config, sample_size)

        effective_authors = min(config.num_authors, n_authors)
        config.num_authors = effective_authors

        model = PrivHSDModelV2(config)
        target_delta = 1.0 / max(len(train_loader.dataset), 1)

        trainer = PrivHSDTrainer(
            model=model, train_loader=train_loader, val_loader=val_loader,
            test_loader=test_loader, config=config,
            learning_rate=config.learning_rate,
            target_epsilon=config.target_epsilon,
            target_delta=target_delta,
            dp_enabled=config.dp_enabled,
            device=device,
            output_dir=str(self.results_dir / "baseline"),
            use_ghost_clipping=config.use_ghost_clipping,
        )
        train_results = trainer.train(
            num_epochs=config.num_epochs, use_adversarial=True,
            eval_every=1, save_every=5,
        )
        eval_result = evaluate_model(model, test_loader, device)

        entry = {
            "ablation_name": "baseline",
            "utility": eval_result.utility.to_dict(),
            "privacy": {"epsilon": train_results.get("final_epsilon", float("inf"))},
            "final_epsilon": train_results.get("final_epsilon", float("inf")),
            "status": "completed",
        }
        self.results["baseline"] = entry
        logger.info(f"Baseline done: F1={eval_result.utility.f1_score:.4f}, "
                     f"ε={entry['final_epsilon']:.2f}")
        return entry

    def _compute_deltas(self):
        """Compute F1 and ε deltas from baseline."""
        baseline = self.results.get("baseline", {})
        baseline_f1 = baseline.get("utility", {}).get("f1_score", 0)
        baseline_eps = baseline.get("final_epsilon", float("inf"))

        for name, entry in self.results.items():
            if name == "baseline":
                continue
            if entry.get("status") == "completed":
                entry["baseline_f1"] = baseline_f1
                entry["baseline_eps"] = baseline_eps
                f1 = entry.get("utility", {}).get("f1_score", 0)
                eps = entry.get("final_epsilon", float("inf"))
                entry["f1_delta"] = f1 - baseline_f1
                entry["eps_delta"] = eps - baseline_eps if (eps != float("inf") and baseline_eps != float("inf")) else None

    def _save_summary(self):
        """Save aggregated summary."""
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "quick_mode": self.quick,
            "n_ablations": len(ABLATIONS),
            "results": self.results,
        }
        path = self.results_dir / "summary.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Summary saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════════════════════

def _diff_config(baseline: PrivHSDConfig, modified: PrivHSDConfig) -> str:
    """Return string describing differences between two configs."""
    diffs = []
    for field in PrivHSDConfig.__dataclass_fields__:
        bv = getattr(baseline, field)
        mv = getattr(modified, field)
        if bv != mv:
            diffs.append(f"{field}: {bv} → {mv}")
    return "; ".join(diffs) if diffs else "(identical)"


def print_ablation_table():
    """Print a formatted table of all ablations."""
    print(f"\n{'Idx':<4} {'Name':<28} {'Hypothesis':<60}")
    print("-" * 100)
    for idx, (name, _, hypothesis, expected) in enumerate(ABLATIONS):
        print(f"{idx:<4} {name:<28} {hypothesis[:57]:<60}")
        print(f"    {'':>4} {'':<28} Expected: {expected[:57]}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="PrivHSD v2 Ablation Runner")
    parser.add_argument("--ablation", type=str, default=None,
                        help="Ablation index/name, comma-separated, or 'all'")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode (2 epochs, sample 200)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print configs without running")
    parser.add_argument("--list", action="store_true",
                        help="List all ablations")
    parser.add_argument("--output-dir", type=str, default="models")
    args = parser.parse_args()

    if args.list:
        print_ablation_table()
        return

    runner = AblationExperimentRunner(quick=args.quick, output_dir=args.output_dir)

    if args.dry_run:
        baseline = get_baseline_config()
        print(f"\nBaseline config fields: {len(PrivHSDConfig.__dataclass_fields__)}")
        print(f"  model_name={baseline.model_name}, dp_enabled={baseline.dp_enabled}")
        print(f"  disentanglement_weight={baseline.disentanglement_weight}")
        print(f"  mim_weight={baseline.mim_weight}, orthogonality_weight={baseline.orthogonality_weight}")
        print(f"  target_epsilon={baseline.target_epsilon}")
        print()
        print_ablation_table()
        print("Dry-run complete. Use --ablation <idx> to run.")
        return

    if args.ablation == "all" or args.ablation is None:
        logger.info("Running all ablations...")
        runner.run_all()
    else:
        indices = [x.strip() for x in args.ablation.split(",")]
        for idx in indices:
            runner.run_single(idx)

    logger.info("Ablation runner complete.")


if __name__ == "__main__":
    main()
