#!/usr/bin/env python3
"""
PrivHSD Inference Script
========================
Deployable inference interface for the trained PrivHSD model.

Usage:
    python inference.py --model-path models/privhsd_final.pt --text "Your text here"
    python inference.py --model-path models/privhsd_final.pt --interactive
    python inference.py --model-path models/privhsd_final.pt --batch-file texts.txt
"""

import argparse
import torch
import json
import sys
from pathlib import Path
from typing import List, Dict, Union

from src.model import PrivHSDModel


class PrivHSDInference:
    """
    Inference wrapper for the PrivHSD model.

    Provides hate speech detection with privacy-preserving inference:
      - No identity information stored or logged
      - Single-sample inference (no batching that could leak cross-sample info)
      - Configurable privacy budget tracking for API deployment
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        privacy_mode: bool = True,
    ):
        """
        Initialize the inference engine.

        Args:
            model_path: Path to saved model checkpoint
            device: Device for inference
            privacy_mode: If True, enables privacy-preserving inference features
                          (no logging of text, no caching, bounded output detail)
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.privacy_mode = privacy_mode

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get("model_config", {})

        # Initialize model
        self.model = PrivHSDModel(
            model_name=config.get("model_name", "albert-base-v2"),
            num_hate_classes=config.get("num_hate_classes", 2),
            num_authors=config.get("num_authors", 100),
            adversarial_alpha=config.get("adversarial_alpha", 0.5),
            disentanglement_weight=config.get("disentanglement_weight", 0.3),
            model_type=config.get("model_type", "albert"),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Load tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.get("model_name", "albert-base-v2")
        )

        self.final_epsilon = checkpoint.get("final_epsilon", float("inf"))
        self.best_f1 = checkpoint.get("best_val_f1", 0.0)

        print(f"PrivHSD Inference Engine initialized")
        print(f"  Model: {config.get('model_name', 'unknown')}")
        print(f"  Device: {self.device}")
        print(f"  Training ε: {self.final_epsilon:.2f}")
        print(f"  Best F1: {self.best_f1:.4f}")
        print(f"  Privacy mode: {'ENABLED' if privacy_mode else 'DISABLED'}")
        print()

    @torch.no_grad()
    def predict(self, text: str) -> Dict:
        """
        Predict hate speech probability for a single text.

        In privacy mode, no text is logged or stored.

        Args:
            text: Input text to classify

        Returns:
            Dictionary with prediction results
        """
        if not text or not isinstance(text, str):
            return {"error": "Invalid input", "hate_probability": 0.0, "is_hate": False}

        encodings = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        hate_prob = float(outputs["hate_probs"][0, 1].cpu().numpy())

        result = {
            "hate_probability": round(hate_prob, 4),
            "is_hate": hate_prob > 0.5,
            "confidence": round(abs(hate_prob - 0.5) * 2, 4),
        }

        # In privacy mode, limit detail to prevent misuse
        if self.privacy_mode:
            # Only return minimal information
            result.pop("confidence", None)

        # NEVER return representations or intermediate features
        return result

    @torch.no_grad()
    def predict_batch(
        self, texts: List[str], batch_size: int = 32
    ) -> List[Dict]:
        """
        Predict hate speech for a batch of texts.

        Args:
            texts: List of input texts
            batch_size: Batch size for inference

        Returns:
            List of prediction dictionaries
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            for text in batch_texts:
                results.append(self.predict(text))
        return results

    def interactive_mode(self):
        """Interactive hate speech detection loop."""
        print("=" * 50)
        print("PrivHSD Interactive Detection")
        print("Type 'quit' to exit, 'toggle-privacy' to toggle privacy mode")
        print("=" * 50)
        print()

        while True:
            try:
                text = input("Enter text: ")
                if text.lower() == "quit":
                    break
                elif text.lower() == "toggle-privacy":
                    self.privacy_mode = not self.privacy_mode
                    print(f"Privacy mode: {'ON' if self.privacy_mode else 'OFF'}")
                    continue

                result = self.predict(text)
                label = "HATE SPEECH" if result["is_hate"] else "NON-HATE"
                print(f"Prediction: {label}")
                print(f"Probability: {result['hate_probability']:.4f}")
                print()

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                continue

    def validate_identity_agnostic(self, texts_by_author: Dict[str, List[str]]) -> Dict:
        """
        Validate that predictions are identity-agnostic.

        Tests whether the same hateful content produces consistent predictions
        regardless of author-specific phrasing.

        Args:
            texts_by_author: Dict mapping author_id -> list of their texts

        Returns:
            Dict with cross-author consistency metrics
        """
        all_probs = []
        author_means = {}

        for author_id, author_texts in texts_by_author.items():
            probs = [self.predict(t)["hate_probability"] for t in author_texts]
            all_probs.extend(probs)
            author_means[str(author_id)] = {
                "mean_prob": float(np.mean(probs)),
                "std_prob": float(np.std(probs)),
                "n_samples": len(probs),
            }

        return {
            "overall_mean": float(np.mean(all_probs)),
            "overall_std": float(np.std(all_probs)),
            "author_variance": float(np.var([m["mean_prob"] for m in author_means.values()])),
            "author_means": author_means,
            "identity_agnostic_score": max(
                0, 1 - float(np.var([m["mean_prob"] for m in author_means.values()]))
            ),
        }


def main():
    parser = argparse.ArgumentParser(description="PrivHSD Inference")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--text", type=str, default=None,
                        help="Single text to classify")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode")
    parser.add_argument("--batch-file", type=str, default=None,
                        help="File with texts (one per line)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for batch results")
    parser.add_argument("--no-privacy-mode", action="store_true",
                        help="Disable privacy-preserving inference features")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    engine = PrivHSDInference(
        model_path=args.model_path,
        device=args.device,
        privacy_mode=not args.no_privacy_mode,
    )

    if args.text:
        result = engine.predict(args.text)
        print(json.dumps(result, indent=2))

    elif args.interactive:
        engine.interactive_mode()

    elif args.batch_file:
        with open(args.batch_file, "r") as f:
            texts = [line.strip() for line in f if line.strip()]
        results = engine.predict_batch(texts)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")
        else:
            for text, result in zip(texts, results):
                print(f"{text[:60]:60s} -> {'HATE' if result['is_hate'] else 'OK'} ({result['hate_probability']:.3f})")

    else:
        print("No input provided. Use --text, --interactive, or --batch-file.")
        parser.print_help()


if __name__ == "__main__":
    import numpy as np  # For validation utility
    main()
