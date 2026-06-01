#!/usr/bin/env python3
"""
PrivHSD: Privacy-preserving Hate Speech Detection
==================================================
Main training script.

Trains an identity-disentangled differentially private transformer
for hate speech detection with formal privacy guarantees.

Usage:
    python train.py --model albert-base-v2 --target-epsilon 8.0 --adversarial
    python train.py --model roberta-base --target-epsilon 4.0 --no-adversarial
    python train.py --dataset hatexplain --model albert-base-v2 --dp-enabled
"""

import argparse
import torch
import numpy as np
import random
import logging
import os
from pathlib import Path

from src.model import PrivHSDModel
from src.data_utils import (
    load_jigsaw_dataset,
    load_hatexplain_dataset,
    create_privacy_augmented_variant,
    get_dataloaders,
)
from src.train import PrivHSDTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="PrivHSD: Privacy-preserving Hate Speech Detection"
    )
    # Model configuration
    parser.add_argument(
        "--model", type=str, default="albert-base-v2",
        choices=["albert-base-v2", "albert-large-v2", "roberta-base", "roberta-large"],
        help="Transformer backbone (default: albert-base-v2)"
    )
    parser.add_argument(
        "--model-type", type=str, default="albert",
        choices=["albert", "roberta"],
        help="Model type (default: albert)"
    )

    # Privacy parameters
    parser.add_argument(
        "--target-epsilon", type=float, default=8.0,
        help="Target privacy budget epsilon (default: 8.0)"
    )
    parser.add_argument(
        "--target-delta", type=float, default=None,
        help="Target delta (default: 1/|dataset|)"
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=1.0,
        help="Max gradient norm for DP-SGD clipping (default: 1.0)"
    )
    parser.add_argument(
        "--dp-enabled", action="store_true", default=True,
        help="Enable DP-SGD training via Opacus (default: True)"
    )
    parser.add_argument(
        "--no-dp", action="store_false", dest="dp_enabled",
        help="Disable DP-SGD training"
    )

    # Adversarial disentanglement
    parser.add_argument(
        "--adversarial", action="store_true", default=True,
        help="Enable adversarial identity disentanglement (default: True)"
    )
    parser.add_argument(
        "--no-adversarial", action="store_false", dest="adversarial",
        help="Disable adversarial disentanglement"
    )
    parser.add_argument(
        "--adversarial-alpha", type=float, default=0.5,
        help="Gradient reversal alpha (default: 0.5)"
    )
    parser.add_argument(
        "--disentanglement-weight", type=float, default=0.3,
        help="Weight of adversarial loss (default: 0.3)"
    )

    # Dataset
    parser.add_argument(
        "--dataset", type=str, default="jigsaw",
        choices=["jigsaw", "hatexplain"],
        help="Dataset to use (default: jigsaw)"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Data directory (default: data)"
    )
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Limit training samples (default: use all)"
    )
    parser.add_argument(
        "--privacy-augment", type=str, default=None,
        choices=["low", "medium", "high"],
        help="Apply privacy augmentation to dataset"
    )

    # Training
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-5,
        help="Learning rate (default: 2e-5)"
    )
    parser.add_argument(
        "--max-length", type=int, default=256,
        help="Max sequence length (default: 256)"
    )
    parser.add_argument(
        "--num-authors", type=int, default=100,
        help="Number of author classes for disentanglement (default: 100)"
    )

    # System
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (default: cuda)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="models",
        help="Output directory (default: models)"
    )
    parser.add_argument(
        "--use-ghost-clipping", action="store_true", default=True,
        help="Use ghost clipping for memory efficiency"
    )
    parser.add_argument(
        "--no-ghost-clipping", action="store_false", dest="use_ghost_clipping",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    logger.info(f"Using device: {device}")
    logger.info(f"Arguments: {vars(args)}")

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    logger.info(f"Tokenizer loaded: {args.model}")

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    if args.dataset == "jigsaw":
        train_dataset, val_dataset, test_dataset, n_authors = load_jigsaw_dataset(
            data_dir=f"{args.data_dir}/jigsaw",
            tokenizer=tokenizer,
            max_length=args.max_length,
            sample_size=args.sample_size,
            random_seed=args.seed,
        )
    elif args.dataset == "hatexplain":
        train_dataset, val_dataset, test_dataset, n_authors = load_hatexplain_dataset(
            data_dir=f"{args.data_dir}/hatexplain",
            tokenizer=tokenizer,
            max_length=args.max_length,
            sample_size=args.sample_size,
            random_seed=args.seed,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Apply privacy augmentation if requested
    if args.privacy_augment:
        logger.info(f"Applying privacy augmentation: {args.privacy_augment}")
        train_dataset = create_privacy_augmented_variant(
            train_dataset, epsilon_level=args.privacy_augment
        )

    # Override num_authors if dataset is smaller
    effective_authors = min(args.num_authors, n_authors)
    logger.info(f"Using {effective_authors} author classes for disentanglement")

    # Create dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=args.batch_size,
    )

    # Initialize model
    logger.info(f"Initializing PrivHSD model with backbone: {args.model}")
    model = PrivHSDModel(
        model_name=args.model,
        num_hate_classes=2,
        num_authors=effective_authors,
        adversarial_alpha=args.adversarial_alpha,
        disentanglement_weight=args.disentanglement_weight,
        model_type=args.model_type,
    )
    logger.info(f"Model parameter count: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize trainer
    trainer = PrivHSDTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=args.learning_rate,
        target_epsilon=args.target_epsilon,
        target_delta=args.target_delta,
        max_grad_norm=args.max_grad_norm,
        adversarial_alpha=args.adversarial_alpha,
        disentanglement_weight=args.disentanglement_weight,
        dp_enabled=args.dp_enabled,
        device=device,
        output_dir=f"{args.output_dir}/checkpoints",
        use_ghost_clipping=args.use_ghost_clipping,
    )

    # Train
    logger.info(f"Starting training for {args.epochs} epochs")
    results = trainer.train(
        num_epochs=args.epochs,
        use_adversarial=args.adversarial,
        eval_every=1,
        save_every=5,
        early_stopping_patience=5,
    )

    # Save final model
    trainer.save_final_model(f"{args.output_dir}/privhsd_final.pt")

    # Print final results
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE - FINAL RESULTS:")
    logger.info(f"  Best Validation F1: {results['best_val_f1']:.4f}")
    logger.info(f"  Final Test F1:      {results['final_test'].get('test_f1', 0):.4f}")
    logger.info(f"  Final Test AUC:     {results['final_test'].get('test_auc', 0):.4f}")
    logger.info(f"  Privacy Budget (ε): {results['final_epsilon']:.2f}")
    logger.info(f"  Privacy Delta (δ):  {trainer.target_delta:.2e}")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    main()
