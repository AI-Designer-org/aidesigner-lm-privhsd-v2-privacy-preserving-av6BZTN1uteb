"""
Training Pipeline for PrivHSD
=============================
Implements DP-SGD training via Opacus with adversarial identity
disentanglement for privacy-preserving hate speech detection.

Key features:
  - Per-sample gradient clipping (Opacus PrivacyEngine)
  - Adversarial training for identity disentanglement
  - Privacy budget accounting (ε, δ)
  - Checkpointing and experiment tracking
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
import json
import time
from typing import Dict, Optional, Tuple, Callable
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from src.model import PrivHSDModel

logger = logging.getLogger(__name__)


class PrivHSDTrainer:
    """
    Trainer for PrivHSD with DP-SGD and adversarial disentanglement.

    Args:
        model: PrivHSDModel instance
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        learning_rate: Learning rate
        target_epsilon: Target privacy budget (ε)
        target_delta: Target delta for DP
        max_grad_norm: Max gradient norm for DP clipping
        adversarial_alpha: GRL scaling factor
        disentanglement_weight: Weight of adversarial loss
        dp_enabled: Whether to use DP-SGD (Opacus)
        device: Device to train on
        output_dir: Directory for saving outputs
        use_ghost_clipping: Whether to use ghost clipping (for large models)
    """

    def __init__(
        self,
        model: PrivHSDModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        learning_rate: float = 2e-5,
        target_epsilon: float = 8.0,
        target_delta: Optional[float] = None,
        max_grad_norm: float = 1.0,
        adversarial_alpha: float = 0.5,
        disentanglement_weight: float = 0.3,
        dp_enabled: bool = True,
        device: str = "cuda",
        output_dir: str = "models/checkpoints",
        use_ghost_clipping: bool = True,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.learning_rate = learning_rate
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta or (1.0 / len(train_loader.dataset))
        self.max_grad_norm = max_grad_norm
        self.adversarial_alpha = adversarial_alpha
        self.disentanglement_weight = disentanglement_weight
        self.dp_enabled = dp_enabled
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_ghost_clipping = use_ghost_clipping

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )

        # Privacy engine
        self.privacy_engine = None
        if dp_enabled:
            self._setup_privacy_engine()

        # History
        self.train_history = []
        self.val_history = []
        self.privacy_history = []
        self.best_val_f1 = 0.0
        self.best_model_state = None

        logger.info(
            f"Trainer initialized: dp={dp_enabled}, ε={target_epsilon}, "
            f"δ={self.target_delta:.2e}, lr={learning_rate}, "
            f"alpha={adversarial_alpha}, dis_weight={disentanglement_weight}"
        )

    def _setup_privacy_engine(self):
        """Setup Opacus PrivacyEngine with ghost clipping for transformers."""
        logger.info("Setting up Opacus PrivacyEngine...")

        # Validate and fix model for DP compatibility
        self.model = ModuleValidator.fix(self.model)
        errors = ModuleValidator.validate(self.model, strict=False)
        if errors:
            logger.warning(f"Module validation errors (will attempt fix): {errors}")
            self.model = ModuleValidator.fix(self.model)

        # Check if the model is compatible
        self.model.train()

        # Create privacy engine
        self.privacy_engine = PrivacyEngine()

        # Handle ghost clipping for transformer models
        try:
            if self.use_ghost_clipping:
                # Ghost clipping: more memory efficient for transformers
                self.model, self.optimizer, self.train_loader = (
                    self.privacy_engine.make_private_with_epsilon(
                        module=self.model,
                        optimizer=self.optimizer,
                        data_loader=self.train_loader,
                        epochs=10,  # Will be overridden
                        target_epsilon=self.target_epsilon,
                        target_delta=self.target_delta,
                        max_grad_norm=self.max_grad_norm,
                        grad_sample_mode="ghost",
                    )
                )
            else:
                self.model, self.optimizer, self.train_loader = (
                    self.privacy_engine.make_private_with_epsilon(
                        module=self.model,
                        optimizer=self.optimizer,
                        data_loader=self.train_loader,
                        epochs=10,
                        target_epsilon=self.target_epsilon,
                        target_delta=self.target_delta,
                        max_grad_norm=self.max_grad_norm,
                    )
                )
        except Exception as e:
            logger.warning(f"Ghost clipping setup failed ({e}), falling back to standard mode")
            self.model, self.optimizer, self.train_loader = (
                self.privacy_engine.make_private_with_epsilon(
                    module=self.model,
                    optimizer=self.optimizer,
                    data_loader=self.train_loader,
                    epochs=10,
                    target_epsilon=self.target_epsilon,
                    target_delta=self.target_delta,
                    max_grad_norm=self.max_grad_norm,
                )
            )

        logger.info("PrivacyEngine setup complete.")

    def get_privacy_spent(self) -> float:
        """Get current privacy budget spent (ε)."""
        if self.privacy_engine is None:
            return 0.0
        try:
            epsilon = self.privacy_engine.get_epsilon(delta=self.target_delta)
            return epsilon
        except Exception:
            return float("inf")

    def train_epoch(
        self,
        epoch: int,
        use_adversarial: bool = True,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number
            use_adversarial: Whether to include adversarial disentanglement

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_hate_loss = 0.0
        total_adv_loss = 0.0
        total_loss = 0.0
        all_preds = []
        all_labels = []
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch in pbar:
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            hate_labels = batch["hate_labels"].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            forward_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "hate_labels": hate_labels,
            }

            if use_adversarial and "author_labels" in batch:
                forward_kwargs["author_labels"] = batch["author_labels"].to(self.device)
                forward_kwargs["alpha"] = self.adversarial_alpha

            outputs = self.model(**forward_kwargs)

            # Backward pass
            outputs["loss"].backward()
            self.optimizer.step()

            # Track metrics
            total_loss += outputs["loss"].item()
            total_hate_loss += outputs.get("hate_loss", torch.tensor(0.0)).item()
            if "author_loss" in outputs:
                total_adv_loss += outputs["author_loss"].item()

            preds = outputs["hate_logits"].argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(hate_labels.cpu().numpy())

            n_batches += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": outputs["loss"].item(),
                "hate_loss": outputs.get("hate_loss", torch.tensor(0.0)).item(),
            })

        # Compute epoch metrics
        avg_loss = total_loss / n_batches
        avg_hate_loss = total_hate_loss / n_batches
        avg_adv_loss = total_adv_loss / n_batches if use_adversarial else 0.0
        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds, average="binary")

        # Get privacy budget
        epsilon_spent = self.get_privacy_spent()

        metrics = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_hate_loss": avg_hate_loss,
            "train_adv_loss": avg_adv_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "epsilon_spent": epsilon_spent,
        }

        self.train_history.append(metrics)
        self.privacy_history.append({
            "epoch": epoch,
            "epsilon": epsilon_spent,
            "delta": self.target_delta,
        })

        logger.info(
            f"Epoch {epoch} Train: loss={avg_loss:.4f}, "
            f"hate_loss={avg_hate_loss:.4f}, acc={train_acc:.4f}, "
            f"f1={train_f1:.4f}, ε={epsilon_spent:.2f}"
        )

        return metrics

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        split_name: str = "val",
    ) -> Dict[str, float]:
        """
        Evaluate the model on a data loader.

        Args:
            loader: DataLoader to evaluate on
            split_name: Name for logging

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []
        n_batches = 0

        for batch in tqdm(loader, desc=f"Evaluating [{split_name}]"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            hate_labels = batch["hate_labels"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                hate_labels=hate_labels,
            )

            total_loss += outputs["hate_loss"].item()
            preds = outputs["hate_logits"].argmax(dim=-1).cpu().numpy()
            probs = outputs["hate_probs"][:, 1].cpu().numpy()

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(hate_labels.cpu().numpy())
            n_batches += 1

        # Compute metrics
        avg_loss = total_loss / n_batches
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="binary")

        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.5

        precision = f1_score(all_labels, all_preds, average="binary", zero_division=0)
        recall = f1_score(all_labels, all_preds, average="binary", zero_division=0)

        metrics = {
            f"{split_name}_loss": avg_loss,
            f"{split_name}_acc": acc,
            f"{split_name}_f1": f1,
            f"{split_name}_auc": auc,
            f"{split_name}_precision": precision,
        }

        self.val_history.append(metrics)

        logger.info(
            f"[{split_name}] loss={avg_loss:.4f}, acc={acc:.4f}, "
            f"f1={f1:.4f}, auc={auc:.4f}"
        )

        return metrics

    def train(
        self,
        num_epochs: int = 10,
        use_adversarial: bool = True,
        eval_every: int = 1,
        save_every: int = 5,
        early_stopping_patience: int = 5,
    ) -> Dict:
        """
        Full training loop with periodic evaluation and checkpointing.

        Returns:
            Dictionary of final results
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(
                epoch=epoch,
                use_adversarial=use_adversarial,
            )

            # Evaluate
            if epoch % eval_every == 0:
                val_metrics = self.evaluate(
                    self.val_loader, split_name="val"
                )
                test_metrics = self.evaluate(
                    self.test_loader, split_name="test"
                )

                # Check for improvement
                val_f1 = val_metrics.get("val_f1", 0.0)
                if val_f1 > self.best_val_f1:
                    self.best_val_f1 = val_f1
                    self.best_model_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                    patience_counter = 0
                    logger.info(f"New best val F1: {val_f1:.4f}")
                else:
                    patience_counter += 1

                # Early stopping
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

            # Save checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(epoch)

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        # Final evaluation
        final_val = self.evaluate(self.val_loader, "val")
        final_test = self.evaluate(self.test_loader, "test")

        results = {
            "final_val": final_val,
            "final_test": final_test,
            "best_val_f1": self.best_val_f1,
            "final_epsilon": self.get_privacy_spent(),
            "train_history": self.train_history,
            "privacy_history": self.privacy_history,
        }

        logger.info(
            f"Training complete. Best val F1: {self.best_val_f1:.4f}, "
            f"Final ε: {results['final_epsilon']:.2f}"
        )

        return results

    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_f1": self.best_val_f1,
            "train_history": self.train_history,
            "privacy_spent": self.get_privacy_spent(),
        }, path)
        logger.info(f"Checkpoint saved: {path}")

    def save_final_model(self, path: str = "models/privhsd_final.pt"):
        """Save the final trained model."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_config": {
                "model_name": self.model.model_name,
                "num_hate_classes": self.model.num_hate_classes,
                "num_authors": self.model.num_authors,
                "adversarial_alpha": self.model.adversarial_alpha,
                "disentanglement_weight": self.model.disentanglement_weight,
            },
            "final_epsilon": self.get_privacy_spent(),
            "best_val_f1": self.best_val_f1,
        }, save_path)
        logger.info(f"Final model saved: {save_path}")
