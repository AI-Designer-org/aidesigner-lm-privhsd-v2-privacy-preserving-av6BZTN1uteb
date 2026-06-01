"""
Evaluation Framework for PrivHSD v2
====================================
Comprehensive evaluation of hate speech detection performance and
privacy guarantees, including:
  - Utility metrics (F1, AUC, precision, recall, MCC, specificity)
  - Privacy metrics (ε, δ accounting)
  - Pareto frontier analysis of privacy-utility trade-off
  - Ablation study visualization
  - Results export (CSV, JSON)
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import label_binarize
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class UtilityMetrics:
    """Container for hate speech detection utility metrics."""
    accuracy: float = 0.0
    f1_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    roc_auc: float = 0.5
    average_precision: float = 0.0
    specificity: float = 0.0
    mcc: float = 0.0  # Matthews correlation coefficient

    def to_dict(self) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "f1_score": self.f1_score,
            "precision": self.precision,
            "recall": self.recall,
            "roc_auc": self.roc_auc,
            "average_precision": self.average_precision,
            "specificity": self.specificity,
            "mcc": self.mcc,
        }


@dataclass
class PrivacyMetrics:
    """Container for privacy guarantee metrics."""
    epsilon: float = float("inf")
    delta: float = 1e-5
    mechanism: str = "dp-sgd"
    membership_inference_auc: float = 0.5
    attribute_inference_auc: float = 0.5
    stylometry_reid_accuracy: float = 0.0
    representation_entropy: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "membership_inference_auc": self.membership_inference_auc,
            "attribute_inference_auc": self.attribute_inference_auc,
            "stylometry_reid_accuracy": self.stylometry_reid_accuracy,
            "representation_entropy": self.representation_entropy,
        }


@dataclass
class EvaluationResult:
    """Full evaluation result for a single experiment configuration."""
    config: Dict = field(default_factory=dict)
    utility: UtilityMetrics = field(default_factory=UtilityMetrics)
    privacy: PrivacyMetrics = field(default_factory=PrivacyMetrics)
    predictions: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None

    def privacy_utility_ratio(self) -> float:
        """Higher is better: how much utility per unit of privacy cost."""
        eps = max(self.privacy.epsilon, 0.01)
        return self.utility.f1_score / eps


def compute_utility_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> UtilityMetrics:
    """Compute comprehensive utility metrics from predictions.

    Args:
        y_true: (N,) — ground truth binary labels (0 or 1).
        y_pred: (N,) — predicted binary labels (0 or 1).
        y_prob: (N,) — predicted probability of positive class (0 to 1).

    Returns:
        UtilityMetrics with accuracy, F1, precision, recall, ROC AUC,
        average precision, specificity, and MCC.

    Shape invariants:
        All arrays must have the same length N ≥ 1.
    """
    metrics = UtilityMetrics()
    metrics.accuracy = accuracy_score(y_true, y_pred)
    metrics.f1_score = f1_score(y_true, y_pred, average="binary", zero_division=0)
    metrics.precision = precision_score(y_true, y_pred, zero_division=0)
    metrics.recall = recall_score(y_true, y_pred, zero_division=0)

    try:
        metrics.roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics.roc_auc = 0.5

    metrics.average_precision = average_precision_score(y_true, y_prob)

    # Specificity = TN / (TN + FP)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Matthews correlation coefficient
    numerator = (tp * tn) - (fp * fn)
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    metrics.mcc = numerator / denominator if denominator > 0 else 0.0

    return metrics


def evaluate_model(
    model,
    data_loader,
    device: str = "cuda",
) -> EvaluationResult:
    """Run full evaluation of a model on a dataset.

    Args:
        model: PrivHSDModelV2 instance
        data_loader: DataLoader with hate_labels
        device: Device for computation

    Returns:
        EvaluationResult with all metrics
    """
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["hate_labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            probs = outputs["hate_probs"][:, 1].cpu().numpy()
            preds = outputs["hate_logits"].argmax(dim=-1).cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    utility = compute_utility_metrics(y_true, y_pred, y_prob)

    return EvaluationResult(
        utility=utility,
        predictions=y_pred,
        probabilities=y_prob,
        labels=y_true,
    )


class ParetoFrontierAnalyzer:
    """Analyze the privacy-utility Pareto frontier across experiment configs.

    A point is Pareto-optimal if no other point has both better privacy
    (lower epsilon) AND better utility (higher F1/metric).
    """

    def __init__(self, output_dir: str = "models/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[EvaluationResult] = []

    def add_result(self, result: EvaluationResult, config: Optional[Dict] = None):
        """Add evaluation result to the frontier analysis."""
        if config:
            result.config = config
        self.results.append(result)

    def compute_pareto_frontier(
        self,
        privacy_metric: str = "epsilon",
        utility_metric: str = "f1_score",
    ) -> List[EvaluationResult]:
        """Compute Pareto-optimal points.

        Points are Pareto-optimal if no other point has both better
        privacy (lower privacy_metric) AND better utility (higher utility_metric).
        """
        if not self.results:
            return []

        points = []
        for r in self.results:
            privacy_val = getattr(r.privacy, privacy_metric)
            utility_val = getattr(r.utility, utility_metric)
            points.append((privacy_val, utility_val, r))

        # Sort by privacy (lower = better)
        points.sort(key=lambda x: x[0])

        pareto_points = []
        max_utility = -float("inf")
        for privacy_val, utility_val, result in points:
            if utility_val > max_utility:
                max_utility = utility_val
                pareto_points.append(result)

        return pareto_points

    def plot_pareto_frontier(
        self,
        save_name: str = "pareto_frontier.png",
        privacy_metric: str = "epsilon",
        utility_metric: str = "f1_score",
    ):
        """Plot the privacy-utility Pareto frontier."""
        if not self.results:
            logger.warning("No results to plot.")
            return

        privacy_vals = [getattr(r.privacy, privacy_metric) for r in self.results]
        utility_vals = [getattr(r.utility, utility_metric) for r in self.results]
        config_names = [
            r.config.get("name", r.config.get("model_name", f"Config {i}"))
            for i, r in enumerate(self.results)
        ]

        pareto = self.compute_pareto_frontier(privacy_metric, utility_metric)

        fig, ax = plt.subplots(figsize=(10, 7))

        # Plot all points
        ax.scatter(privacy_vals, utility_vals, c="steelblue", s=80, alpha=0.7,
                   label="All configurations")

        # Highlight Pareto frontier
        pareto_privacy = [getattr(r.privacy, privacy_metric) for r in pareto]
        pareto_utility = [getattr(r.utility, utility_metric) for r in pareto]
        ax.scatter(pareto_privacy, pareto_utility, c="red", s=120,
                   marker="*", zorder=5, label="Pareto-optimal")

        # Connect Pareto points
        pareto_points = sorted(
            zip(pareto_privacy, pareto_utility),
            key=lambda x: x[0]
        )
        if len(pareto_points) > 1:
            pp, pu = zip(*pareto_points)
            ax.plot(pp, pu, "r--", alpha=0.5, label="Pareto frontier")

        # Annotate points
        for i, name in enumerate(config_names):
            ax.annotate(name, (privacy_vals[i], utility_vals[i]),
                        xytext=(5, 5), textcoords="offset points", fontsize=8)

        ax.set_xlabel("Privacy Budget (ε)" if privacy_metric == "epsilon" else privacy_metric)
        ax.set_ylabel(f"Utility ({utility_metric})")
        ax.set_title("Privacy-Utility Pareto Frontier")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Annotate ideal region
        ax.annotate(
            "Ideal region\n(high utility,\nstrong privacy)",
            xy=(min(privacy_vals), max(utility_vals)),
            xytext=(min(privacy_vals) + 0.1, max(utility_vals) - 0.05),
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.3),
        )

        save_path = self.output_dir / save_name
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Pareto frontier plot saved: {save_path}")

    def plot_ablation_study(
        self,
        save_name: str = "ablation_study.png",
    ):
        """Plot ablation study results."""
        if not self.results:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: F1 vs Epsilon per configuration
        ax1 = axes[0]
        names = [
            r.config.get("name", r.config.get("model_name", f"Config {i}"))
            for i, r in enumerate(self.results)
        ]
        f1_scores = [r.utility.f1_score for r in self.results]
        epsilons = [r.privacy.epsilon for r in self.results]

        x = np.arange(len(names))
        width = 0.35
        ax1.bar(x - width / 2, f1_scores, width, label="F1 Score", color="steelblue")
        ax1.bar(x + width / 2, epsilons, width, label="ε", color="coral")
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=45, ha="right")
        ax1.set_ylabel("Score")
        ax1.set_title("Ablation: Configurations")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Privacy-Utility scatter with region coloring
        ax2 = axes[1]
        colors = [
            "green" if r.privacy.epsilon <= 3 else
            "orange" if r.privacy.epsilon <= 8 else
            "red"
            for r in self.results
        ]
        ax2.scatter(
            [r.privacy.epsilon for r in self.results],
            [r.utility.f1_score for r in self.results],
            c=colors, s=100, alpha=0.7,
        )
        ax2.set_xlabel("Privacy Budget (ε)")
        ax2.set_ylabel("F1 Score")
        ax2.set_title("Privacy-Utility Trade-off by Privacy Regime")
        ax2.grid(True, alpha=0.3)

        # Region annotations
        max_eps = max([r.privacy.epsilon for r in self.results]) + 1
        ax2.axvspan(0, 3, alpha=0.1, color="green", label="Strong (ε ≤ 3)")
        ax2.axvspan(3, 8, alpha=0.1, color="orange", label="Moderate (3 < ε ≤ 8)")
        ax2.axvspan(8, max_eps, alpha=0.1, color="red", label="Weak (ε > 8)")
        ax2.legend(fontsize=8)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Ablation study plot saved: {save_path}")

    def save_results_table(self, save_name: str = "results_table.csv") -> pd.DataFrame:
        """Save results as a structured CSV table."""
        rows = []
        for r in self.results:
            row = {
                "config": r.config.get("name", r.config.get("model_name", "unknown")),
                "f1_score": round(r.utility.f1_score, 4),
                "accuracy": round(r.utility.accuracy, 4),
                "roc_auc": round(r.utility.roc_auc, 4),
                "precision": round(r.utility.precision, 4),
                "recall": round(r.utility.recall, 4),
                "mcc": round(r.utility.mcc, 4),
                "specificity": round(r.utility.specificity, 4),
                "epsilon": round(r.privacy.epsilon, 2),
                "delta": r.privacy.delta,
                "mia_auc": round(r.privacy.membership_inference_auc, 4),
                "attr_inf_auc": round(r.privacy.attribute_inference_auc, 4),
                "stylometry_acc": round(r.privacy.stylometry_reid_accuracy, 4),
                "repr_entropy": round(r.privacy.representation_entropy, 4),
                "purity_ratio": round(r.privacy_utility_ratio(), 4),
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        save_path = self.output_dir / save_name
        df.to_csv(save_path, index=False)
        logger.info(f"Results table saved: {save_path}")
        return df

    def generate_report(self, save_name: str = "evaluation_report.json") -> Dict:
        """Generate a comprehensive JSON evaluation report."""
        report = {
            "n_configurations": len(self.results),
            "pareto_optimal_count": len(self.compute_pareto_frontier()),
            "results": [],
        }

        for r in self.results:
            report["results"].append({
                "config": r.config,
                "utility": r.utility.to_dict(),
                "privacy": r.privacy.to_dict(),
                "privacy_utility_ratio": r.privacy_utility_ratio(),
            })

        save_path = self.output_dir / save_name
        with open(save_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Evaluation report saved: {save_path}")

        return report
