"""
PrivHSD Model Architecture
==========================
Identity-Disentangled Differentially Private Transformer for
Privacy-preserving Hate Speech Detection.

Architecture design:
  - Transformer backbone (ALBERT / RoBERTa) for hate speech detection
  - Adversarial identity disentanglement module that strips author
    identity signals from the representation space
  - DP-SGD integration via Opacus for formal privacy guarantees
  - Multi-task learning: primary HSD task + adversarial identity task
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    AlbertForSequenceClassification,
    RobertaForSequenceClassification,
)
from typing import Optional, Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)


class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL) for adversarial training.

    During forward pass, acts as identity.
    During backward pass, reverses and scales the gradient.
    This encourages the encoder to learn representations that
    *cannot* predict the protected attribute (author identity).
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return grad_output.neg() * ctx.alpha, None


class IdentityDisentanglementHead(nn.Module):
    """
    Adversarial head that attempts to predict author identity
    from the representation. The GRL ensures the encoder learns
    identity-agnostic representations.

    Architecture:
        Hidden → LayerNorm → ReLU → Dropout → Hidden → LayerNorm → ReLU → Dropout → Linear(num_authors)
    """
    def __init__(
        self,
        hidden_size: int,
        num_authors: int,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_layers = num_layers
        layers = []
        for i in range(num_layers):
            in_dim = hidden_size if i == 0 else hidden_size
            out_dim = hidden_size if i < num_layers - 1 else num_authors
            layers.extend([
                nn.Linear(in_dim, hidden_size) if i < num_layers - 1 else nn.Linear(in_dim, out_dim),
                nn.LayerNorm(hidden_size) if i < num_layers - 1 else nn.Identity(),
                nn.ReLU() if i < num_layers - 1 else nn.Identity(),
                nn.Dropout(dropout) if i < num_layers - 1 else nn.Identity(),
            ])
        self.network = nn.Sequential(*layers) if num_layers > 1 else nn.Linear(hidden_size, num_authors)

        # Simplify: just use a simple MLP
        if num_layers > 1:
            layers = []
            curr_dim = hidden_size
            for i in range(num_layers - 1):
                layers.append(nn.Linear(curr_dim, hidden_size))
                layers.append(nn.LayerNorm(hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                curr_dim = hidden_size
            layers.append(nn.Linear(hidden_size, num_authors))
            self.network = nn.Sequential(*layers)
        else:
            self.network = nn.Linear(hidden_size, num_authors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict author identity from representation."""
        return self.network(x)


class PrivHSDModel(nn.Module):
    """
    Privacy-preserving Hate Speech Detection Model.

    Core innovation: Identity-disentangled representations learned
    via adversarial training, combined with differential privacy.

    Args:
        model_name: HuggingFace transformer model name
        num_hate_classes: Number of hate speech classification labels
        num_authors: Number of distinct authors (for disentanglement)
        adversarial_alpha: Gradient reversal scaling factor
        disentanglement_weight: Weight of adversarial loss
        hidden_dropout: Dropout probability
        model_type: 'albert' or 'roberta'
        cache_dir: Cache directory for pretrained models
    """
    def __init__(
        self,
        model_name: str = "albert-base-v2",
        num_hate_classes: int = 2,
        num_authors: int = 100,
        adversarial_alpha: float = 0.5,
        disentanglement_weight: float = 0.3,
        hidden_dropout: float = 0.2,
        model_type: str = "albert",
        cache_dir: Optional[str] = None,
    ):
        super().__init__()

        self.model_name = model_name
        self.num_hate_classes = num_hate_classes
        self.num_authors = num_authors
        self.adversarial_alpha = adversarial_alpha
        self.disentanglement_weight = disentanglement_weight
        self.model_type = model_type

        # Load transformer backbone
        logger.info(f"Loading backbone: {model_name}")
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_hate_classes,
            hidden_dropout_prob=hidden_dropout,
            attention_probs_dropout_prob=hidden_dropout,
            cache_dir=cache_dir,
        )

        if model_type == "albert":
            self.encoder = AlbertForSequenceClassification.from_pretrained(
                model_name, config=config, cache_dir=cache_dir,
            )
            self.hidden_size = config.hidden_size
        elif model_type == "roberta":
            self.encoder = RobertaForSequenceClassification.from_pretrained(
                model_name, config=config, cache_dir=cache_dir,
            )
            self.hidden_size = config.hidden_size
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        # Override the classifier head to use hidden states directly
        # We'll extract the [CLS] representation manually
        self.hate_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(self.hidden_size, num_hate_classes),
        )

        # Identity disentanglement adversary
        self.identity_adversary = IdentityDisentanglementHead(
            hidden_size=self.hidden_size,
            num_authors=num_authors,
            dropout=hidden_dropout,
        )

        # Gradient reversal layer
        self.grl = GradientReversalLayer()

        logger.info(
            f"PrivHSD Model initialized: backbone={model_name}, "
            f"hate_classes={num_hate_classes}, authors={num_authors}, "
            f"alpha={adversarial_alpha}, dis_weight={disentanglement_weight}"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        hate_labels: Optional[torch.Tensor] = None,
        author_labels: Optional[torch.Tensor] = None,
        alpha: Optional[float] = None,
        return_representations: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional adversarial disentanglement.

        Args:
            input_ids: Token indices
            attention_mask: Attention mask
            token_type_ids: Token type IDs (for BERT-based models)
            hate_labels: Ground truth hate speech labels
            author_labels: Ground truth author identity labels
            alpha: GRL scaling (overrides default if provided)
            return_representations: Whether to return pooled representations

        Returns:
            Dictionary with logits, losses, and optionally representations
        """
        # Get transformer outputs
        transformer_outputs = self.encoder.albert if self.model_type == "albert" else self.encoder.roberta

        if self.model_type == "albert":
            outputs = transformer_outputs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_hidden_states=True,
            )
        else:  # roberta
            outputs = transformer_outputs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # Get pooled representation (last hidden state of [CLS] token)
        # outputs.last_hidden_state shape: (batch, seq_len, hidden)
        last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)
        cls_representation = last_hidden[:, 0, :]  # (batch, hidden)

        # Hate speech classification
        hate_logits = self.hate_classifier(cls_representation)

        result = {
            "hate_logits": hate_logits,
            "hate_probs": F.softmax(hate_logits, dim=-1),
        }

        # Compute hate speech classification loss
        if hate_labels is not None:
            hate_loss = F.cross_entropy(hate_logits, hate_labels)
            result["hate_loss"] = hate_loss

        # Adversarial identity disentanglement
        if author_labels is not None:
            current_alpha = alpha if alpha is not None else self.adversarial_alpha
            # Apply gradient reversal
            reversed_repr = self.grl.apply(cls_representation, current_alpha)
            # Predict author identity
            author_logits = self.identity_adversary(reversed_repr)
            author_loss = F.cross_entropy(author_logits, author_labels)
            result["author_loss"] = author_loss
            result["author_logits"] = author_logits

            # Total loss = hate_loss + disentanglement_weight * author_loss
            if hate_labels is not None:
                result["loss"] = hate_loss + self.disentanglement_weight * author_loss

        if return_representations:
            result["representations"] = cls_representation.detach()

        return result

    def get_hate_predictions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get hate speech predictions only (inference mode)."""
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        return outputs["hate_probs"]

    def get_representations(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract pooled representations for privacy analysis."""
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_representations=True,
            )
        return outputs["representations"]


# Alias for backwards compatibility
IdentityDisentangledDPTransformer = PrivHSDModel
