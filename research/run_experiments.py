#!/usr/bin/env python3
"""
PrivHSD: Systematic Experiment Runner
======================================
Runs ablation studies across privacy budgets, model configurations,
and dataset variants. Generates Pareto frontier analysis and
comprehensive evaluation reports.

Usage:
    python run_experiments.py --quick                       # Quick sanity check
    python run_experiments.py --full                        # Full experiment suite
    python run_experiments.py --ablation privacy            # Privacy budget sweep
    python run_experiments.py --ablation architecture       # Architecture comparison
"""

import argparse
import torch
import numpy as np
import random
import logging
import json
import os
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.model import PrivHSDModel
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
    """Configuration for a single experiment."""
    name: str
    model_name: str = "albert-base-v2"
    model_type: str = "albert"
    target_epsilon: float = 8.0
    dp_enabled: bool = True
    adversarial: bool = True
    adversarial_alpha: float = 0.5
    disentanglement_weight: float = 0.3
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
            "disentanglement_weight": self.disentanglement_weight,
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
    """
    Run a single experiment configuration end-to-end.

    Args:
        config: Experiment configuration
        device: Device for training
        output_dir: Output directory for model checkpoints

    Returns:
        EvaluationResult with utility and privacy metrics
    """
    logger.info(f"\n{'='*60}\nRunning experiment: {config.name}\n{'='*60}")
    logger.info(f"Config: {config.to_dict()}")

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

    # Privacy augment
    if config.privacy_augment:
        train_dataset = create_privacy_augmented_variant(
            train_dataset, epsilon_level=config.privacy_augment
        )

    effective_authors = min(config.num_authors, n_authors)
    train_loader, val_loader, test_loader = get_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=config.batch_size,
    )

    # Initialize model
    model = PrivHSDModel(
        model_name=config.model_name,
        num_hate_classes=2,
        num_authors=effective_authors,
        adversarial_alpha=config.adversarial_alpha,
        disentanglement_weight=config.disentanglement_weight,
        model_type=config.model_type,
    )

    # Target delta
    target_delta = 1.0 / len(train_dataset) if len(train_dataset) > 0 else 1e-5

    # Trainer
    trainer = PrivHSDTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=config.learning_rate,
        target_epsilon=config.target_epsilon,
        target_delta=target_delta,
        max_grad_norm=1.0,
        adversarial_alpha=config.adversarial_alpha,
        disentanglement_weight=config.disentanglement_weight,
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

    # Evaluate utility
    logger.info("Evaluating utility metrics...")
    eval_result = evaluate_model(model, test_loader, device)

    # Evaluate privacy
    logger.info("Evaluating privacy metrics...")
    privacy_metrics = PrivacyMetrics()
    privacy_metrics.epsilon = results.get("final_epsilon", float("inf"))
    privacy_metrics.delta = target_delta

    # --- Membership Inference Attack ---
    try:
        mia = MembershipInferenceAttack(attack_type="shadow_model")
        # Sample member and non-member texts
        member_texts = [train_dataset.texts[i] for i in range(min(100, len(train_dataset)))]
        non_member_texts = [test_dataset.texts[i] for i in range(min(100, len(test_dataset)))]
        mia_metrics = mia.evaluate(model, member_texts, non_member_texts, tokenizer, device)
        privacy_metrics.membership_inference_auc = mia_metrics.auc
    except Exception as e:
        logger.warning(f"MIA evaluation failed: {e}")

    # --- Attribute Inference ---
    try:
        attr_attack = AttributeInferenceAttack()
        representations = []
        attr_labels = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if batch_idx >= 5:
                    break  # Sample first 5 batches
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
        logger.warning(f"Attribute inference evaluation failed: {e}")

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
        logger.warning(f"Stylometry evaluation failed: {e}")

    # --- Representation Privacy Audit ---
    try:
        audit = RepresentationPrivacyAudit()
        if representations:
            reps_array = np.vstack(representations)
            privacy_metrics.representation_entropy = audit.compute_entropy(reps_array)
    except Exception as e:
        logger.warning(f"Representation audit failed: {e}")

    # Bundle results
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

    logger.info(f"Experiment {config.name} complete:")
    logger.info(f"  F1: {eval_result.utility.f1_score:.4f}")
    logger.info(f"  AUC: {eval_result.utility.roc_auc:.4f}")
    logger.info(f"  ε: {eval_result.privacy.epsilon:.2f}")
    logger.info(f"  MIA AUC: {eval_result.privacy.membership_inference_auc:.4f}")

    return eval_result


def get_privacy_sweep_configs() -> List[ExperimentConfig]:
    """Budget sweep: vary epsilon from strong to no privacy."""
    configs = []
    for eps in [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]:
        configs.append(ExperimentConfig(
            name=f"dp_eps_{eps}",
            target_epsilon=eps,
            dp_enabled=True,
            adversarial=True,
        ))
    return configs


def get_adversarial_ablation_configs() -> List[ExperimentConfig]:
    """Adversarial disentanglement ablation."""
    configs = [
        ExperimentConfig(
            name="no_dp_no_adv",
            dp_enabled=False, adversarial=False,
            target_epsilon=float("inf"),
        ),
        ExperimentConfig(
            name="no_dp_with_adv",
            dp_enabled=False, adversarial=True,
            target_epsilon=float("inf"),
        ),
        ExperimentConfig(
            name="dp_no_adv",
            dp_enabled=True, adversarial=False,
            target_epsilon=8.0,
        ),
        ExperimentConfig(
            name="dp_with_adv",
            dp_enabled=True, adversarial=True,
            target_epsilon=8.0,
        ),
    ]
    return configs


def get_architecture_comparison_configs() -> List[ExperimentConfig]:
    """Compare ALBERT vs RoBERTa backbones."""
    configs = [
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
            target_epsilon=8.0,
        ),
    ]
    return configs


def get_privacy_augmentation_configs() -> List[ExperimentConfig]:
    """Privacy-augmented data variants."""
    configs = []
    for level in ["low", "medium", "high"]:
        configs.append(ExperimentConfig(
            name=f"augment_{level}",
            privacy_augment=level,
            target_epsilon=8.0,
        ))
    return configs


def get_baseline_configs() -> List[ExperimentConfig]:
    """Baseline comparisons (no privacy)."""
    return [
        ExperimentConfig(
            name="baseline_no_privacy",
            dp_enabled=False, adversarial=False,
            target_epsilon=float("inf"),
        ),
        ExperimentConfig(
            name="dp_only",
            dp_enabled=True, adversarial=False,
            target_epsilon=8.0,
        ),
        ExperimentConfig(
            name="adv_only",
            dp_enabled=False, adversarial=True,
            target_epsilon=float("inf"),
        ),
    ]


def main():
    parser = argparse.ArgumentParser(description="PrivHSD Experiment Runner")
    parser.add_argument("--full", action="store_true", help="Run full experiment suite")
    parser.add_argument("--quick", action="store_true", help="Quick sanity check (1 config)")
    parser.add_argument("--ablation", type=str, default=None,
                        choices=["privacy", "adversarial", "architecture", "augmentation"],
                        help="Run specific ablation study")
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
        logger.info("Running quick sanity check")
        configs = [
            ExperimentConfig(name="sanity_check", epochs=2, sample_size=200),
        ]
    elif args.ablation == "privacy":
        logger.info("Running privacy budget sweep")
        configs = get_privacy_sweep_configs()
    elif args.ablation == "adversarial":
        logger.info("Running adversarial disentanglement ablation")
        configs = get_adversarial_ablation_configs()
    elif args.ablation == "architecture":
        logger.info("Running architecture comparison")
        configs = get_architecture_comparison_configs()
    elif args.ablation == "augmentation":
        logger.info("Running privacy augmentation ablation")
        configs = get_privacy_augmentation_configs()
    elif args.full:
        logger.info("Running full experiment suite")
        configs = (
            get_baseline_configs()
            + get_privacy_sweep_configs()
            + get_adversarial_ablation_configs()
            + get_architecture_comparison_configs()
            + get_privacy_augmentation_configs()
        )
    else:
        logger.info("No experiment type specified. Running baseline + privacy sweep.")
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
    if len(analyzer.results) > 0:
        logger.info(f"\nGenerating analysis for {len(analyzer.results)} experiments...")

        # Pareto frontier
        analyzer.plot_pareto_frontier()
        analyzer.plot_ablation_study()

        # Results table
        df = analyzer.save_results_table()
        logger.info(f"\nResults summary:\n{df.to_string()}")

        # JSON report
        report = analyzer.generate_report()

        logger.info(f"\n{'='*60}")
        logger.info("EXPERIMENTS COMPLETE")
        logger.info(f"Total configurations: {len(analyzer.results)}")
        logger.info(f"Pareto-optimal: {len(analyzer.compute_pareto_frontier())}")
        logger.info(f"Results saved to: {args.output_dir}/results/")
        logger.info(f"{'='*60}")
    else:
        logger.warning("No experiments completed successfully.")


if __name__ == "__main__":
    main()
