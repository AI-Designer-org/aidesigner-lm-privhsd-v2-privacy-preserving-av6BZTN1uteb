#!/usr/bin/env python3
"""
PrivHSD v2: Systematic Experiment Runner
=========================================
Runs ablation studies across privacy budgets, model configurations,
adversarial levels, and dataset variants. Generates Pareto frontier
analysis and comprehensive evaluation reports.

Usage:
    python run_experiments.py --quick                       # Quick sanity check
    python run_experiments.py --full                        # Full experiment suite
    python run_experiments.py --ablation privacy            # Privacy budget sweep
    python run_experiments.py --ablation architecture       # Architecture comparison
    python run_experiments.py --ablation adversarial        # Adversarial ablation
    python run_experiments.py --ablation augmentation       # Data augmentation ablation
"""

import argparse
import torch
import numpy as np
import random
import logging
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.model import PrivHSDConfig, PrivHSDModelV2
from src.data_utils import (
    load_jigsaw_dataset,
    load_hatexplain_dataset,
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
)
from src.attacks import (
    MembershipInferenceAttack,
    AttributeInferenceAttack,
    StylometryReidentificationRisk,
    RepresentationPrivacyAudit,
    AttackMetrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    name: str
    model_name: str = "albert-base-v2"
    model_type: str = "albert"
    target_epsilon: float = 8.0
    dp_enabled: bool = True
    adversarial: bool = True
    adversarial_alpha: float = 1.0
    alpha_schedule: str = "sigmoid"
    disentanglement_weight: float = 0.3
    mim_weight: float = 0.1
    orthogonality_weight: float = 0.05
    dataset: str = "jigsaw"
    learning_rate: float = 2e-5
    batch_size: int = 16
    epochs: int = 10
    num_authors: int = 100
    privacy_augment: Optional[str] = None
    sample_size: Optional[int] = None

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "target_epsilon": self.target_epsilon,
            "dp_enabled": self.dp_enabled,
            "adversarial": self.adversarial,
            "adversarial_alpha": self.adversarial_alpha,
            "alpha_schedule": self.alpha_schedule,
            "disentanglement_weight": self.disentanglement_weight,
            "mim_weight": self.mim_weight,
            "orthogonality_weight": self.orthogonality_weight,
            "dataset": self.dataset,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "num_authors": self.num_authors,
            "privacy_augment": self.privacy_augment,
        }


def train_and_evaluate(
    config: ExperimentConfig,
    device: str = "cuda",
    output_dir: str = "models",
) -> EvaluationResult:
    """Run a single experiment end-to-end.

    Args:
        config: Experiment configuration
        device: Device for training
        output_dir: Output directory

    Returns:
        EvaluationResult with utility and privacy metrics
    """
    logger.info(f"\n{'='*60}\nExperiment: {config.name}\n{'='*60}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Load dataset
    if config.dataset == "jigsaw":
        train_dataset, val_dataset, test_dataset, n_authors = load_jigsaw_dataset(
            data_dir="data/jigsaw", tokenizer=tokenizer,
            max_length=256, sample_size=config.sample_size,
        )
    else:
        train_dataset, val_dataset, test_dataset, n_authors = load_hatexplain_dataset(
            data_dir="data/hatexplain", tokenizer=tokenizer,
            max_length=256, sample_size=config.sample_size,
        )

    # Privacy augmentation
    if config.privacy_augment:
        train_dataset = create_privacy_augmented_variant(
            train_dataset, epsilon_level=config.privacy_augment
        )

    effective_authors = min(config.num_authors, n_authors)
    train_loader, val_loader, test_loader = get_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=config.batch_size,
    )

    # Model config
    model_config = PrivHSDConfig(
        model_name=config.model_name,
        model_type=config.model_type,
        num_hate_classes=2,
        num_authors=effective_authors,
        alpha_final=config.adversarial_alpha,
        alpha_schedule=config.alpha_schedule,
        disentanglement_weight=config.disentanglement_weight,
        mim_weight=config.mim_weight,
        orthogonality_weight=config.orthogonality_weight,
        num_epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        dp_enabled=config.dp_enabled,
    )

    model = PrivHSDModelV2(model_config)

    # Target delta
    target_delta = 1.0 / max(len(train_dataset), 1)

    # Trainer
    eps = config.target_epsilon if config.target_epsilon != float("inf") else 1e6
    trainer = PrivHSDTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=model_config,
        learning_rate=config.learning_rate,
        target_epsilon=eps,
        target_delta=target_delta,
        max_grad_norm=1.0,
        dp_enabled=config.dp_enabled,
        device=device,
        output_dir=f"{output_dir}/checkpoints/{config.name}",
        use_ghost_clipping=True,
    )

    # Train
    results = trainer.train(
        num_epochs=config.epochs,
        use_adversarial=config.adversarial,
        eval_every=1,
        save_every=5,
        early_stopping_patience=3,
    )

    # Utility evaluation
    logger.info("Evaluating utility...")
    eval_result = evaluate_model(model, test_loader, device)

    # Privacy metrics
    privacy_metrics = PrivacyMetrics()
    privacy_metrics.epsilon = results.get("final_epsilon", float("inf"))
    privacy_metrics.delta = target_delta

    # --- Membership Inference Attack ---
    try:
        mia = MembershipInferenceAttack(attack_type="shadow_model")
        member_texts = [train_dataset.texts[i] for i in range(min(100, len(train_dataset)))]
        non_member_texts = [test_dataset.texts[i] for i in range(min(100, len(test_dataset)))]
        mia_metrics = mia.evaluate(model, member_texts, non_member_texts, tokenizer, device)
        privacy_metrics.membership_inference_auc = mia_metrics.auc
    except Exception as e:
        logger.warning(f"MIA failed: {e}")

    # --- Attribute Inference ---
    try:
        attr_attack = AttributeInferenceAttack()
        representations = []
        attr_labels = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if batch_idx >= 5:
                    break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                reps = model.get_representations(input_ids, attention_mask)
                representations.append(reps.cpu().numpy())
                if "author_labels" in batch:
                    attr_labels.extend(batch["author_labels"].cpu().numpy())

        if representations and attr_labels:
            reps_array = np.vstack(representations)
            attr_labels_array = np.array(attr_labels)
            attr_metrics = attr_attack.evaluate(reps_array, attr_labels_array)
            privacy_metrics.attribute_inference_auc = attr_metrics.auc
    except Exception as e:
        logger.warning(f"Attribute inference failed: {e}")

    # --- Stylometry Re-identification ---
    try:
        stylo = StylometryReidentificationRisk(n_authors=effective_authors)
        sample_texts = [test_dataset.texts[i] for i in range(min(200, len(test_dataset)))]
        sample_authors = [test_dataset.author_ids[i] for i in range(min(200, len(test_dataset)))]
        if sample_authors and all(a is not None for a in sample_authors):
            stylo_metrics = stylo.evaluate(
                model, sample_texts, np.array(sample_authors), tokenizer, device
            )
            privacy_metrics.stylometry_reid_accuracy = (
                stylo_metrics.get("model_representation", AttackMetrics()).accuracy
            )
    except Exception as e:
        logger.warning(f"Stylometry failed: {e}")

    # --- Representation Privacy Audit ---
    try:
        audit = RepresentationPrivacyAudit()
        if representations:
            reps_array = np.vstack(representations)
            privacy_metrics.representation_entropy = audit.compute_entropy(reps_array)
    except Exception as e:
        logger.warning(f"Representation audit failed: {e}")

    # Bundle
    eval_result.privacy = privacy_metrics
    eval_result.config = config.to_dict()

    # Save individual result
    result_path = Path(f"{output_dir}/results/{config.name}_result.json")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump({
            "config": config.to_dict(),
            "utility": eval_result.utility.to_dict(),
            "privacy": eval_result.privacy.to_dict(),
        }, f, indent=2)

    logger.info(
        f"Experiment {config.name} done: F1={eval_result.utility.f1_score:.4f}, "
        f"AUC={eval_result.utility.roc_auc:.4f}, "
        f"ε={eval_result.privacy.epsilon:.2f}, "
        f"MIA AUC={eval_result.privacy.membership_inference_auc:.4f}"
    )

    return eval_result


# ── Ablation config factories ──────────────────────────────────────────


def get_privacy_sweep_configs() -> List[ExperimentConfig]:
    """Privacy budget sweep: ε ∈ {1, 2, 4, 8, 16, 32, ∞}."""
    configs = []
    for eps in [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]:
        configs.append(ExperimentConfig(
            name=f"dp_eps_{int(eps)}",
            target_epsilon=eps,
            dp_enabled=True,
            adversarial=True,
            adversarial_alpha=1.0,
            alpha_schedule="sigmoid",
        ))
    # Non-DP baseline for comparison
    configs.append(ExperimentConfig(
        name="dp_eps_inf",
        target_epsilon=float("inf"),
        dp_enabled=True,
        adversarial=True,
        adversarial_alpha=1.0,
    ))
    return configs


def get_adversarial_ablation_configs() -> List[ExperimentConfig]:
    """Adversarial disentanglement ablation (4 conditions)."""
    return [
        ExperimentConfig(
            name="no_dp_no_adv", dp_enabled=False, adversarial=False,
            disentanglement_weight=0.0, mim_weight=0.0, orthogonality_weight=0.0,
            target_epsilon=float("inf"),
        ),
        ExperimentConfig(
            name="no_dp_with_adv", dp_enabled=False, adversarial=True,
            target_epsilon=float("inf"),
        ),
        ExperimentConfig(
            name="dp_no_adv", dp_enabled=True, adversarial=False,
            disentanglement_weight=0.0, mim_weight=0.0, orthogonality_weight=0.0,
            target_epsilon=8.0,
        ),
        ExperimentConfig(
            name="dp_with_adv", dp_enabled=True, adversarial=True,
            target_epsilon=8.0,
        ),
    ]


def get_architecture_comparison_configs() -> List[ExperimentConfig]:
    """Compare ALBERT vs RoBERTa backbones at ε=8."""
    return [
        ExperimentConfig(
            name="albert_base_dp8",
            model_name="albert-base-v2", model_type="albert",
            target_epsilon=8.0,
        ),
        ExperimentConfig(
            name="roberta_base_dp8",
            model_name="roberta-base", model_type="roberta",
            target_epsilon=8.0,
        ),
        ExperimentConfig(
            name="albert_large_dp8",
            model_name="albert-large-v2", model_type="albert",
            target_epsilon=8.0, batch_size=8,  # smaller batch for large
        ),
    ]


def get_privacy_augmentation_configs() -> List[ExperimentConfig]:
    """Privacy-augmented data variants."""
    return [
        ExperimentConfig(name=f"augment_{level}", privacy_augment=level, target_epsilon=8.0)
        for level in ["low", "medium", "high"]
    ]


def get_baseline_configs() -> List[ExperimentConfig]:
    """Baseline comparisons."""
    return [
        ExperimentConfig(
            name="baseline_no_privacy",
            dp_enabled=False, adversarial=False,
            disentanglement_weight=0.0, mim_weight=0.0, orthogonality_weight=0.0,
            target_epsilon=float("inf"),
        ),
        ExperimentConfig(
            name="dp_only",
            dp_enabled=True, adversarial=False,
            disentanglement_weight=0.0, mim_weight=0.0, orthogonality_weight=0.0,
            target_epsilon=8.0,
        ),
        ExperimentConfig(
            name="adv_only",
            dp_enabled=False, adversarial=True, target_epsilon=float("inf"),
        ),
    ]


# ── Main ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="PrivHSD v2 Experiment Runner")
    parser.add_argument("--full", action="store_true", help="Full experiment suite")
    parser.add_argument("--quick", action="store_true", help="Quick sanity (1 config)")
    parser.add_argument("--ablation", type=str, default=None,
                        choices=["privacy", "adversarial", "architecture", "augmentation"],
                        help="Specific ablation study")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    logger.info(f"Device: {device}")

    # Determine configs
    configs = []
    if args.quick:
        logger.info("Quick sanity check (2 epochs, sample 200)")
        configs = [ExperimentConfig(name="sanity_check", epochs=2, sample_size=200)]
    elif args.ablation == "privacy":
        configs = get_privacy_sweep_configs()
    elif args.ablation == "adversarial":
        configs = get_adversarial_ablation_configs()
    elif args.ablation == "architecture":
        configs = get_architecture_comparison_configs()
    elif args.ablation == "augmentation":
        configs = get_privacy_augmentation_configs()
    elif args.full:
        logger.info("Full experiment suite")
        configs = (
            get_baseline_configs()
            + get_privacy_sweep_configs()
            + get_adversarial_ablation_configs()
            + get_architecture_comparison_configs()
            + get_privacy_augmentation_configs()
        )
    else:
        logger.info("Default: baseline + privacy sweep")
        configs = get_baseline_configs() + get_privacy_sweep_configs()

    # Run experiments
    analyzer = ParetoFrontierAnalyzer(output_dir=f"{args.output_dir}/results")

    for config in configs:
        try:
            result = train_and_evaluate(config, device, args.output_dir)
            analyzer.add_result(result, config.to_dict())
        except Exception as e:
            logger.error(f"Experiment {config.name} failed: {e}", exc_info=True)

    # Generate analysis
    if analyzer.results:
        logger.info(f"\nGenerating analysis for {len(analyzer.results)} experiments...")

        analyzer.plot_pareto_frontier()
        analyzer.plot_ablation_study()
        df = analyzer.save_results_table()
        logger.info(f"\nResults:\n{df.to_string()}")
        report = analyzer.generate_report()

        logger.info(f"\n{'='*60}")
        logger.info("EXPERIMENTS COMPLETE")
        logger.info(f"Configs: {len(analyzer.results)}")
        logger.info(f"Pareto-optimal: {len(analyzer.compute_pareto_frontier())}")
        logger.info(f"Results: {args.output_dir}/results/")
        logger.info(f"{'='*60}")
    else:
        logger.warning("No experiments completed.")


if __name__ == "__main__":
    main()
