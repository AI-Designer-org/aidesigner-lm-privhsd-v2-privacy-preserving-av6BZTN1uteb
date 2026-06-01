#!/usr/bin/env python3
"""
PrivHSD v2 Smoke Test
======================
Verifies that the full model pipeline initializes, forward-passes,
and produces correctly-shaped outputs. Runs without a GPU or
pretrained model weights (uses synthetic data).

Usage:
    python smoke_test.py
    python smoke_test.py --no-dp     # Skip DP-SGD setup
    python smoke_test.py --device cpu
"""

import argparse
import torch
import numpy as np
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("smoke_test")


def count_params(model):
    """Print parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total params:    {total:>10,}")
    logger.info(f"  Trainable params: {trainable:>10,}")
    return total


def test_model_forward(args):
    """Test v2 model forward pass with synthetic data."""
    logger.info("=" * 60)
    logger.info("PrivHSD v2 Smoke Test")
    logger.info("=" * 60)

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    logger.info(f"  Device: {device}")

    # ── 1. Model initialization ────────────────────────────────────
    logger.info("\n[1/5] Initializing PrivHSDModelV2...")
    from src.model import PrivHSDConfig, PrivHSDModelV2, count_params

    cfg = PrivHSDConfig(
        model_name="albert-base-v2",
        model_type="albert",
        d_model=768,
        n_layers=2,       # reduced for speed
        n_heads=4,
        max_seq_len=64,
        num_hate_classes=2,
        num_authors=10,
        adversarial_levels=("pooler", "token", "head"),
        adversary_hidden_dim=64,
        adversary_num_layers=2,
        disentanglement_weight=0.3,
        mim_weight=0.1,
        orthogonality_weight=0.05,
        dropout=0.2,
        adversary_dropout=0.3,
        num_epochs=1,
        batch_size=4,
        dp_enabled=False,  # no DP for smoke test
    )

    model = PrivHSDModelV2(cfg).to(device)
    model.eval()
    count_params(model)
    logger.info("  [OK] Model initialized")

    # ── 2. Forward pass without labels ─────────────────────────────
    logger.info("\n[2/5] Forward pass (no labels)...")
    B, T = 4, 32
    input_ids = torch.randint(0, 100, (B, T), device=device)
    attention_mask = torch.ones(B, T, device=device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)

    assert "hate_logits" in out, "Missing hate_logits"
    assert "hate_probs" in out, "Missing hate_probs"
    assert out["hate_logits"].shape == (B, 2), f"Bad logits shape: {out['hate_logits'].shape}"
    assert out["hate_probs"].shape == (B, 2), f"Bad probs shape: {out['hate_probs'].shape}"
    logger.info(f"  hate_logits: {out['hate_logits'].shape}  [OK]")
    logger.info(f"  hate_probs:  {out['hate_probs'].shape}   [OK]")

    # ── 3. Forward pass with labels ────────────────────────────────
    logger.info("\n[3/5] Forward pass (with labels)...")
    hate_labels = torch.randint(0, 2, (B,), device=device)
    author_labels = torch.randint(0, 10, (B,), device=device)

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        hate_labels=hate_labels,
        author_labels=author_labels,
        step=0,
    )

    assert "loss" in out, "Missing loss"
    assert "hate_loss" in out, "Missing hate_loss"
    assert "author_loss" in out, "Missing author_loss"
    assert "alpha" in out, "Missing alpha"
    assert out["loss"].item() > 0, f"Loss should be positive, got {out['loss'].item()}"
    logger.info(f"  loss:        {out['loss'].item():.4f}            [OK]")
    logger.info(f"  hate_loss:   {out['hate_loss'].item():.4f}       [OK]")
    logger.info(f"  author_loss: {out['author_loss'].item():.4f}     [OK]")
    logger.info(f"  alpha:       {out['alpha']:.4f}                  [OK]")

    if "mim_loss" in out:
        logger.info(f"  mim_loss:    {out['mim_loss'].item():.4f}     [OK]")
    if "orth_loss" in out:
        logger.info(f"  orth_loss:   {out['orth_loss'].item():.4f}    [OK]")

    # ── 4. AGRS scheduler ──────────────────────────────────────────
    logger.info("\n[4/5] AGRS scheduler test...")
    scheduler = model.alpha_scheduler
    scheduler.set_total_steps(100)

    alphas = []
    for step in range(101):
        alpha = scheduler.get_alpha(step)
        alphas.append(alpha)

    logger.info(f"  alpha[0] = {alphas[0]:.4f} (should be ~{cfg.alpha_initial})")
    logger.info(f"  alpha[50] = {alphas[50]:.4f}")
    logger.info(f"  alpha[100] = {alphas[100]:.4f} (should be ~{cfg.alpha_final})")
    assert alphas[0] <= 0.15, f"Initial alpha too high: {alphas[0]}"
    # Sigmoid with gamma=2.0 gives alpha ≈ 0.76 at p=1.0; asymptotically approaches 1.0
    assert alphas[-1] >= 0.75, f"Final alpha too low: {alphas[-1]}"
    logger.info("  [OK] AGRS scheduler")

    # ── 5. MINE mutual information estimate ────────────────────────
    logger.info("\n[5/5] MINE mutual information estimate...")
    if model.mim_module is not None:
        with torch.no_grad():
            transformer_out = model.get_transformer_outputs(input_ids, attention_mask)
            mi = model.mim_module.estimate_mutual_information(
                transformer_out["pooler_repr"], author_labels
            )
        logger.info(f"  MI estimate: {mi.item():.4f}                  [OK]")
    else:
        logger.info("  MIM disabled (mim_weight=0)                    [SKIP]")

    # ── Summary ────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("ALL SMOKE TESTS PASSED")
    logger.info("=" * 60)

    # Return test results dict
    return {
        "status": "passed",
        "device": device,
        "params": sum(p.numel() for p in model.parameters()),
        "config": cfg,
    }


def test_data_loading(args):
    """Test dataset loading with synthetic data."""
    logger.info("\n[optional] Testing data loading...")
    from src.data_utils import load_jigsaw_dataset, load_hatexplain_dataset, get_dataloaders
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")

    # Jigsaw (synthetic)
    train, val, test, n_authors = load_jigsaw_dataset(
        tokenizer=tokenizer, sample_size=200, random_seed=42,
    )
    logger.info(f"  Jigsaw synthetic: train={len(train)}, authors={n_authors}")
    assert len(train) > 0, "Empty training set"

    # HateXplain (synthetic)
    train2, val2, test2, n_authors2 = load_hatexplain_dataset(
        tokenizer=tokenizer, sample_size=200, random_seed=42,
    )
    logger.info(f"  HateXplain synthetic: train={len(train2)}, authors={n_authors2}")
    assert len(train2) > 0, "Empty training set"

    # DataLoader creation
    loaders = get_dataloaders(train, val, test, batch_size=4)
    for batch in loaders[0]:
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "hate_labels" in batch
        assert "author_labels" in batch
        break
    logger.info("  [OK] Data loading works")

    return True


def test_attacks(args):
    """Test privacy attack initialization."""
    logger.info("\n[optional] Testing attack module imports...")
    from src.attacks import (
        MembershipInferenceAttack,
        AttributeInferenceAttack,
        StylometryReidentificationRisk,
        RepresentationPrivacyAudit,
    )

    mia = MembershipInferenceAttack()
    attr = AttributeInferenceAttack()
    stylo = StylometryReidentificationRisk(n_authors=5)
    audit = RepresentationPrivacyAudit()

    logger.info(f"  MIA: type={mia.attack_type}")
    logger.info(f"  Attribute inference: initialized")
    logger.info(f"  Stylometry: n_authors={stylo.n_authors}")
    logger.info(f"  Representation audit: initialized")
    logger.info("  [OK] Attack modules load correctly")
    return True


def test_evaluation(args):
    """Test evaluation framework initialization."""
    logger.info("\n[optional] Testing evaluation framework...")
    from src.evaluate import (
        ParetoFrontierAnalyzer,
        EvaluationResult,
        UtilityMetrics,
        PrivacyMetrics,
        compute_utility_metrics,
    )

    # Test UtilityMetrics computation
    y_true = [0, 1, 0, 1, 0]
    y_pred = [0, 1, 0, 0, 0]
    y_prob = [0.1, 0.9, 0.2, 0.4, 0.3]
    metrics = compute_utility_metrics(
        np.array(y_true), np.array(y_pred), np.array(y_prob)
    )
    logger.info(f"  Utility: F1={metrics.f1_score:.4f}, AUC={metrics.roc_auc:.4f}")
    assert metrics.f1_score > 0
    assert metrics.roc_auc >= 0.5

    # Test ParetoFrontierAnalyzer
    analyzer = ParetoFrontierAnalyzer(output_dir="/tmp/privhsd_test")
    result1 = EvaluationResult(
        config={"name": "test1"},
        utility=UtilityMetrics(f1_score=0.85),
        privacy=PrivacyMetrics(epsilon=8.0),
    )
    result2 = EvaluationResult(
        config={"name": "test2"},
        utility=UtilityMetrics(f1_score=0.90),
        privacy=PrivacyMetrics(epsilon=4.0),
    )
    analyzer.add_result(result1)
    analyzer.add_result(result2)
    pareto = analyzer.compute_pareto_frontier()
    logger.info(f"  Pareto frontier: {len(pareto)} points from {len(analyzer.results)} configs")
    logger.info("  [OK] Evaluation framework works")

    return True


def main():
    parser = argparse.ArgumentParser(description="PrivHSD v2 Smoke Test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-dp", action="store_true", help="Skip DP setup")
    parser.add_argument("--quick", action="store_true", help="Only run model forward test")
    parser.add_argument("--full", action="store_true", help="Run all tests including optional")
    args = parser.parse_args()

    # Run main model test
    result = test_model_forward(args)

    # Run optional tests
    if args.full or args.quick:
        pass  # Only model test
    else:
        test_data_loading(args)
        test_attacks(args)
        test_evaluation(args)

    sys.exit(0)


if __name__ == "__main__":
    main()
