#!/usr/bin/env python3
"""
PrivHSD v2: Privacy-preserving Hate Speech Detection
=====================================================
Main training script for the v2 model with multi-level adversarial
disentanglement, DP-SGD, and mutual information minimization.

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
from pathlib import Path

from src.model import PrivHSDConfig, PrivHSDModelV2
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
        description="PrivHSD v2: Privacy-preserving Hate Speech Detection"
    )

    # Model
    parser.add_argument("--model", type=str, default="albert-base-v2",
                        choices=["albert-base-v2", "albert-large-v2",
                                 "roberta-base", "roberta-large"],
                        help="Transformer backbone")
    parser.add_argument("--model-type", type=str, default="albert",
                        choices=["albert", "roberta"],
                        help="Model type")

    # Privacy
    parser.add_argument("--target-epsilon", type=float, default=8.0,
                        help="Target privacy budget")
    parser.add_argument("--target-delta", type=float, default=None,
                        help="Target delta (default: 1/|dataset|)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Max gradient norm for DP clipping")
    parser.add_argument("--dp-enabled", action="store_true", default=True)
    parser.add_argument("--no-dp", action="store_false", dest="dp_enabled")

    # Adversarial
    parser.add_argument("--adversarial", action="store_true", default=True)
    parser.add_argument("--no-adversarial", action="store_false", dest="adversarial")
    parser.add_argument("--adversarial-alpha", type=float, default=0.5,
                        help="GRL alpha final value")
    parser.add_argument("--alpha-schedule", type=str, default="sigmoid",
                        choices=["linear", "sigmoid", "adaptive"])
    parser.add_argument("--disentanglement-weight", type=float, default=0.3)
    parser.add_argument("--mim-weight", type=float, default=0.1,
                        help="Mutual information minimization weight")
    parser.add_argument("--orthogonality-weight", type=float, default=0.05,
                        help="Orthogonality regularization weight")

    # Dataset
    parser.add_argument("--dataset", type=str, default="jigsaw",
                        choices=["jigsaw", "hatexplain"])
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--privacy-augment", type=str, default=None,
                        choices=["low", "medium", "high"])

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--num-authors", type=int, default=100)

    # System
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--use-ghost-clipping", action="store_true", default=True)
    parser.add_argument("--no-ghost-clipping", action="store_false",
                        dest="use_ghost_clipping")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    logger.info(f"Device: {device}")
    logger.info(f"Args: {vars(args)}")

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    if args.dataset == "jigsaw":
        train_dataset, val_dataset, test_dataset, n_authors = load_jigsaw_dataset(
            data_dir=f"{args.data_dir}/jigsaw", tokenizer=tokenizer,
            max_length=args.max_length, sample_size=args.sample_size,
            random_seed=args.seed,
        )
    elif args.dataset == "hatexplain":
        train_dataset, val_dataset, test_dataset, n_authors = load_hatexplain_dataset(
            data_dir=f"{args.data_dir}/hatexplain", tokenizer=tokenizer,
            max_length=args.max_length, sample_size=args.sample_size,
            random_seed=args.seed,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Privacy augmentation
    if args.privacy_augment:
        logger.info(f"Privacy augmentation: {args.privacy_augment}")
        train_dataset = create_privacy_augmented_variant(
            train_dataset, epsilon_level=args.privacy_augment
        )

    effective_authors = min(args.num_authors, n_authors)

    # DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=args.batch_size,
    )

    # Model config
    config = PrivHSDConfig(
        model_name=args.model,
        model_type=args.model_type,
        num_hate_classes=2,
        num_authors=effective_authors,
        alpha_final=args.adversarial_alpha,
        alpha_schedule=args.alpha_schedule,
        disentanglement_weight=args.disentanglement_weight,
        mim_weight=args.mim_weight,
        orthogonality_weight=args.orthogonality_weight,
        dropout=0.2,
        adversary_dropout=0.3,
        max_seq_len=args.max_length,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dp_enabled=args.dp_enabled,
        target_epsilon=args.target_epsilon,
        target_delta=args.target_delta,
        use_ghost_clipping=args.use_ghost_clipping,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    # Initialize model
    model = PrivHSDModelV2(config)
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Trainer
    target_delta_val = args.target_delta or (1.0 / max(len(train_dataset), 1))
    trainer = PrivHSDTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        learning_rate=args.learning_rate,
        target_epsilon=args.target_epsilon,
        target_delta=target_delta_val,
        max_grad_norm=args.max_grad_norm,
        dp_enabled=args.dp_enabled,
        device=device,
        output_dir=f"{args.output_dir}/checkpoints",
        use_ghost_clipping=args.use_ghost_clipping,
    )

    # Train
    logger.info(f"Training for {args.epochs} epochs (adversarial={args.adversarial})")
    results = trainer.train(
        num_epochs=args.epochs,
        use_adversarial=args.adversarial,
        eval_every=1,
        save_every=5,
        early_stopping_patience=5,
    )

    # Save final
    trainer.save_final_model(f"{args.output_dir}/privhsd_final.pt")

    # Final results
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"  Best Val F1:  {results['best_val_f1']:.4f}")
    logger.info(f"  Test F1:      {results['final_test'].get('test_f1', 0):.4f}")
    logger.info(f"  Test AUC:     {results['final_test'].get('test_auc', 0):.4f}")
    logger.info(f"  Final ε:      {results['final_epsilon']:.2f}")
    logger.info(f"  Final δ:      {target_delta_val:.2e}")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    main()
