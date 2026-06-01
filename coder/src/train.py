"""
Training Pipeline for PrivHSD v2
=================================
Implements DP-SGD training via Opacus with multi-level adversarial identity
disentanglement for privacy-preserving hate speech detection.

Key features:
  - Per-sample gradient clipping (Opacus PrivacyEngine with ghost clipping)
  - Multi-Level Adversarial Disentanglement (MLAD) training
  - Adaptive Gradient Reversal Scheduling (AGRS)
  - Mutual Information Minimization (MINE) training
  - Privacy budget accounting (ε, δ) via RDP
  - Gradient checkpointing support
  - Checkpointing and experiment tracking
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
import json
import time
from typing import Dict, Optional, Tuple, Callable, List
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from src.model import PrivHSDModelV2, PrivHSDConfig

logger = logging.getLogger(__name__)


class PrivHSDTrainer:
    """Trainer for PrivHSD v2 with DP-SGD and adversarial disentanglement.

    Handles the combined optimization of:
      1. Hate speech classification (primary task)
      2. Multi-Level Adversarial Disentanglement (MLAD)
      3. Mutual Information Minimization via MINE (MIM)
      4. Representation orthogonality regularization

    The combined training objective is:
        L_total = L_hate + λ_dis * L_adv + λ_mim * I_est + λ_orth * L_orth

    where L_adv gradient is reversed through GRL.
    """

    def __init__(
        self,
        model: PrivHSDModelV2,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Optional[PrivHSDConfig] = None,
        learning_rate: float = 2e-5,
        target_epsilon: float = 8.0,
        target_delta: Optional[float] = None,
        max_grad_norm: float = 1.0,
        dp_enabled: bool = True,
        device: str = "cuda",
        output_dir: str = "models/checkpoints",
        use_ghost_clipping: bool = True,
    ):
        self.model = model.to(device)
        self.config = config or model.config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.learning_rate = learning_rate
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta or (1.0 / max(len(train_loader.dataset), 1))
        self.max_grad_norm = max_grad_norm
        self.dp_enabled = dp_enabled
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_ghost_clipping = use_ghost_clipping

        # Optimizer for main model
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(config.beta1 if config else 0.9, config.beta2 if config else 0.999),
        )

        # MINE optimizer (separate, following Belghazi+18)
        self.mim_optimizer = None
        if (config and config.mim_weight > 0 and model.mim_module is not None):
            self.mim_optimizer = torch.optim.Adam(
                model.mim_module.parameters(),
                lr=config.mim_learning_rate,
                betas=(0.9, config.mim_momentum),
            )

        # Privacy engine
        self.privacy_engine = None
        if dp_enabled:
            self._setup_privacy_engine()

        # Configure AGRS scheduler
        total_steps = len(train_loader) * (config.num_epochs if config else 10)
        self.model.alpha_scheduler.set_total_steps(total_steps)

        # History
        self.train_history = []
        self.val_history = []
        self.privacy_history = []
        self.best_val_f1 = 0.0
        self.best_model_state = None

        logger.info(
            f"Trainer initialized: dp={dp_enabled}, ε={target_epsilon}, "
            f"δ={self.target_delta:.2e}, lr={learning_rate}, "
            f"levels={config.adversarial_levels if config else ('pooler',)}, "
            f"mim_weight={config.mim_weight if config else 0.0}"
        )

    def _setup_privacy_engine(self):
        """Setup Opacus PrivacyEngine with ghost clipping."""
        logger.info("Setting up Opacus PrivacyEngine...")

        from opacus import PrivacyEngine
        from opacus.validators import ModuleValidator

        # Validate and fix model for DP compatibility
        self.model = ModuleValidator.fix(self.model)
        errors = ModuleValidator.validate(self.model, strict=False)
        if errors:
            logger.warning(f"Module validation errors: {errors}")
            self.model = ModuleValidator.fix(self.model)

        self.model.train()
        self.privacy_engine = PrivacyEngine()

        try:
            eps = self.target_epsilon if self.target_epsilon != float("inf") else 1e6
            if self.use_ghost_clipping:
                self.model, self.optimizer, self.train_loader = (
                    self.privacy_engine.make_private_with_epsilon(
                        module=self.model,
                        optimizer=self.optimizer,
                        data_loader=self.train_loader,
                        epochs=self.config.num_epochs if self.config else 10,
                        target_epsilon=eps,
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
                        epochs=self.config.num_epochs if self.config else 10,
                        target_epsilon=eps,
                        target_delta=self.target_delta,
                        max_grad_norm=self.max_grad_norm,
                    )
                )
        except Exception as e:
            logger.warning(f"Ghost clipping setup failed ({e}), falling back")
            self.model, self.optimizer, self.train_loader = (
                self.privacy_engine.make_private_with_epsilon(
                    module=self.model,
                    optimizer=self.optimizer,
                    data_loader=self.train_loader,
                    epochs=self.config.num_epochs if self.config else 10,
                    target_epsilon=eps,
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

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        step: int,
        use_adversarial: bool = True,
    ) -> Dict:
        """Single training step with combined objective.

        Three-phase backward:
          1. MINE network forward + backward (maximize I_est)
          2. Encoder forward + combined backward (minimize L_total)
          3. (DP-SGD handles per-sample clipping via Opacus)

        Args:
            batch: dict with "input_ids" (B, T), "attention_mask" (B, T),
                   "hate_labels" (B,), optionally "author_labels" (B,).
            step: Current global training step (0-indexed).
            use_adversarial: If True, include adversarial disentanglement
                             and MINE objectives in the loss.

        Returns:
            Dict with loss values, logits, and alpha from model forward.
        """
        input_ids = batch["input_ids"].to(self.device)          # (B, T)
        attention_mask = batch["attention_mask"].to(self.device)  # (B, T)
        hate_labels = batch["hate_labels"].to(self.device)       # (B,)

        self.optimizer.zero_grad()

        # ── Phase 1: MINE network step (if applicable) ──────────────
        if (
            use_adversarial
            and self.mim_optimizer is not None
            and "author_labels" in batch
        ):
            author_labels = batch["author_labels"].to(self.device)

            # Get representations for MINE (detached from encoder)
            with torch.no_grad():
                transformer_out = self.model.get_transformer_outputs(
                    input_ids, attention_mask
                )
                pooler_repr = transformer_out["pooler_repr"]  # (B, D)

            # MINE maximizes I(repr; author)
            mi_positive = self.model.mim_module(
                pooler_repr, author_labels, maximize=True
            )
            self.mim_optimizer.zero_grad()
            mi_positive.backward()
            # Clip MINE gradients for stability
            torch.nn.utils.clip_grad_norm_(
                self.model.mim_module.parameters(), max_norm=1.0
            )
            self.mim_optimizer.step()
        else:
            author_labels = batch.get("author_labels", None)
            if author_labels is not None:
                author_labels = author_labels.to(self.device)

        # ── Phase 2: Encoder forward + combined backward ────────────
        # Get alpha from AGRS scheduler
        alpha = None
        if use_adversarial and author_labels is not None:
            alpha = self.model.alpha_scheduler.get_alpha(step)

        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            hate_labels=hate_labels,
            author_labels=author_labels if use_adversarial else None,
            alpha=alpha,
            step=step,
        )

        # Backward pass
        loss = outputs["loss"]
        loss.backward()

        # Update AGRS scheduler with loss values
        if use_adversarial and "author_loss" in outputs:
            self.model.alpha_scheduler.update_history(
                hate_loss=outputs.get("hate_loss", torch.tensor(0.0)).item(),
                adv_loss=outputs["author_loss"].item(),
            )

        # Optimizer step (DP-SGD handles clipping internally)
        self.optimizer.step()

        return outputs

    def train_epoch(
        self,
        epoch: int,
        use_adversarial: bool = True,
    ) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            epoch: Current epoch number (1-indexed)
            use_adversarial: Whether to include adversarial disentanglement

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_hate_loss = 0.0
        total_adv_loss = 0.0
        total_mim_loss = 0.0
        total_loss = 0.0
        all_preds = []
        all_labels = []
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, batch in enumerate(pbar):
            global_step = (epoch - 1) * len(self.train_loader) + batch_idx

            outputs = self.train_step(
                batch=batch,
                step=global_step,
                use_adversarial=use_adversarial,
            )

            # Track metrics
            total_loss += outputs["loss"].item()
            total_hate_loss += outputs.get("hate_loss", torch.tensor(0.0)).item()

            if "author_loss" in outputs:
                total_adv_loss += outputs["author_loss"].item()
            if "mim_loss" in outputs:
                total_mim_loss += outputs["mim_loss"].item()

            preds = outputs["hate_logits"].argmax(dim=-1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["hate_labels"].cpu().numpy())

            n_batches += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{outputs['loss'].item():.4f}",
                "hate": f"{outputs.get('hate_loss', 0):.4f}",
                "adv": f"{outputs.get('author_loss', 0):.4f}",
                "α": f"{outputs.get('alpha', 0):.3f}",
            })

        # Compute epoch metrics
        avg_loss = total_loss / max(n_batches, 1)
        avg_hate_loss = total_hate_loss / max(n_batches, 1)
        avg_adv_loss = total_adv_loss / max(n_batches, 1)
        avg_mim_loss = total_mim_loss / max(n_batches, 1)
        train_acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
        train_f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0) if all_labels else 0.0

        # Get privacy budget spent
        epsilon_spent = self.get_privacy_spent()

        metrics = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_hate_loss": avg_hate_loss,
            "train_adv_loss": avg_adv_loss,
            "train_mim_loss": avg_mim_loss,
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
            f"hate={avg_hate_loss:.4f}, adv={avg_adv_loss:.4f}, "
            f"mim={avg_mim_loss:.4f}, acc={train_acc:.4f}, "
            f"f1={train_f1:.4f}, ε={epsilon_spent:.2f}"
        )

        return metrics

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        split_name: str = "val",
    ) -> Dict[str, float]:
        """Evaluate model on a data loader.

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
        avg_loss = total_loss / max(n_batches, 1)
        acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
        f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0) if all_labels else 0.0

        try:
            auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
        except ValueError:
            auc = 0.5

        from sklearn.metrics import precision_score, recall_score
        precision = precision_score(all_labels, all_preds, zero_division=0) if all_labels else 0.0
        recall = recall_score(all_labels, all_preds, zero_division=0) if all_labels else 0.0

        metrics = {
            f"{split_name}_loss": avg_loss,
            f"{split_name}_acc": acc,
            f"{split_name}_f1": f1,
            f"{split_name}_auc": auc,
            f"{split_name}_precision": precision,
            f"{split_name}_recall": recall,
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
        """Full training loop with periodic evaluation and checkpointing.

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
                val_metrics = self.evaluate(self.val_loader, split_name="val")
                test_metrics = self.evaluate(self.test_loader, split_name="test")

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
            logger.info("Restored best model state")

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
            "config": self.config,
        }, path)
        logger.info(f"Checkpoint saved: {path}")

    def save_final_model(self, path: str = "models/privhsd_final.pt"):
        """Save final trained model."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_config": {
                "model_name": self.config.model_name if self.config else "albert-base-v2",
                "model_type": self.config.model_type if self.config else "albert",
                "num_hate_classes": self.config.num_hate_classes if self.config else 2,
                "num_authors": self.config.num_authors if self.config else 100,
                "alpha_final": self.config.alpha_final if self.config else 0.5,
                "disentanglement_weight": self.config.disentanglement_weight if self.config else 0.3,
                "mim_weight": self.config.mim_weight if self.config else 0.0,
            },
            "final_epsilon": self.get_privacy_spent(),
            "best_val_f1": self.best_val_f1,
        }, save_path)
        logger.info(f"Final model saved: {save_path}")
