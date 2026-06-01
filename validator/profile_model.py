#!/usr/bin/env python3
"""
PrivHSD v2 — Profiling Script
==============================
Runs torch.profiler on the model to measure memory and compute costs,
including FLOP estimation per the Kaplan et al. scaling laws.

Supports three profiling modes:
  - forward:  inference-only (≈ 2 × params FLOPs)
  - train:    fwd + bwd (≈ 6 × params FLOPs)
  - full:     fwd + bwd + DP clipping (≈ 6-10 × params FLOPs)

Usage:
    python profile_model.py                          # Default: forward pass
    python profile_model.py --mode train             # Train (fwd+bwd)
    python profile_model.py --mode full              # Full DP training step
    python profile_model.py --device cpu             # CPU profiling
    python profile_model.py --steps 20               # More profiling steps
    python profile_model.py --html                   # Generate Chrome trace
    python profile_model.py --layers 2 --heads 4     # Small model for quick test
"""

import argparse
import sys
import math
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "coder"))

from src.model import PrivHSDConfig, PrivHSDModelV2


def build_sample_input(
    config: PrivHSDConfig,
    device: str = "cuda",
    include_labels: bool = True,
) -> Dict[str, torch.Tensor]:
    """Build a synthetic batch matching config dimensions.

    Args:
        config: Model configuration
        device: Target device
        include_labels: Include hate_labels and author_labels for training

    Returns:
        dict of tensors suitable for model.forward()
    """
    B = config.batch_size
    T = config.max_seq_len

    batch = {
        "input_ids": torch.randint(0, min(config.d_model, 30000), (B, T), device=device),
        "attention_mask": torch.ones(B, T, dtype=torch.long, device=device),
    }

    if include_labels:
        batch["hate_labels"] = torch.randint(0, config.num_hate_classes, (B,), device=device)
        batch["author_labels"] = torch.randint(0, config.num_authors, (B,), device=device)
        batch["step"] = 0

    return batch


def estimate_flops(config: PrivHSDConfig, mode: str = "forward") -> float:
    """Estimate FLOPs for one forward (or train) pass.

    Uses the scaling-law approximation from Kaplan et al. (2020):
        forward FLOPs  ≈ 2 × N  (per token, for autoregressive)
        train FLOPs    ≈ 6 × N  (fwd + bwd per token)

    For a transformer classifier (non-autoregressive):
        forward FLOPs  ≈ 2 × N × T × (1 + 1/3)
          where extra factor accounts for non-causal attention

    Args:
        config: Model configuration
        mode: "forward" or "train" or "full"

    Returns:
        Estimated FLOPs (as float; divide by 1e9 for GFLOPs)
    """
    N = 0
    # Embedding
    N += config.vocab_size * config.d_model
    # Per transformer layer
    attn = 4 * config.d_model * config.d_model  # QKV + output projections
    ffn = 2 * config.d_model * config.d_ff       # up + down projections
    per_layer = attn + ffn
    N += config.n_layers * per_layer
    # Classification head
    N += config.classifier_hidden_dim * config.d_model
    N += config.classifier_hidden_dim * config.num_hate_classes

    # Adversary heads
    n_adv = len(config.adversarial_levels)
    for _ in range(n_adv):
        N += config.d_model * config.adversary_hidden_dim
        N += config.adversary_hidden_dim * config.num_authors

    # MINE network
    N += (config.d_model + config.num_authors) * config.mim_hidden_dim
    N += config.mim_hidden_dim * 1

    T = config.max_seq_len

    if mode == "forward":
        flops = 2 * N * T
    elif mode == "train":
        flops = 6 * N * T  # Kaplan scaling: fwd + bwd ≈ 3× fwd-only
    elif mode == "full":
        flops = 8 * N * T  # fwd + bwd + DP clipping overhead
    else:
        flops = 2 * N * T

    return float(flops)


def profile_forward(
    model: PrivHSDModelV2,
    config: PrivHSDConfig,
    device: str = "cuda",
    steps: int = 10,
    mode: str = "forward",
    html: bool = False,
) -> Dict:
    """Profile the model and return timing/memory statistics.

    Args:
        model: PrivHSD model instance
        config: Model configuration
        device: Target device
        steps: Number of profiling steps
        mode: "forward", "train", or "full"
        html: Generate Chrome trace HTML

    Returns:
        dict with profiling results
    """
    from torch.profiler import profile, record_function, ProfilerActivity

    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    sort_key = "cuda_memory_usage" if device == "cuda" else "self_cpu_time_total"

    # Warmup
    sample = build_sample_input(config, device, include_labels=(mode != "forward"))
    model.train() if mode != "forward" else model.eval()

    for _ in range(3):
        outputs = model(**sample)
        if mode != "forward":
            outputs["loss"].backward()
            model.zero_grad()

    # Profiled runs
    profiler_kwargs = dict(
        activities=activities,
        record_shapes=True,
        profile_memory=(device == "cuda"),
        with_stack=True,
    )
    if html:
        profiler_kwargs["with_flops"] = True

    with profile(**profiler_kwargs) as prof:
        for step_idx in range(steps):
            sample = build_sample_input(config, device, include_labels=(mode != "forward"))
            with record_function(f"## {mode}_step_{step_idx}"):
                outputs = model(**sample)
                if mode != "forward":
                    loss = outputs["loss"]
                    loss.backward()
                    model.zero_grad()

    # Format results
    if device == "cuda":
        sort_criteria = ["self_cuda_memory_usage", "self_cuda_time_total"]
    else:
        sort_criteria = ["self_cpu_time_total"]

    print(f"\n{'='*70}")
    print(f"Profiling Results: mode={mode}, device={device}, steps={steps}")
    print(f"{'='*70}")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=20))

    # PARAM / FLOP summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    est_flops = estimate_flops(config, mode)

    print(f"\n{'─'*70}")
    print(f"Model Statistics")
    print(f"{'─'*70}")
    print(f"  Total parameters:     {total_params:>12,}")
    print(f"  Trainable parameters: {trainable_params:>12,}")
    print(f"  Estimated FLOPs:      {est_flops:>12,.0f}  ({est_flops / 1e9:.2f} GFLOPs)")
    print(f"  Estimated TFLOPs:     {est_flops / 1e12:.4f}")

    if device == "cuda":
        # CUDA memory stats
        print(f"\n  CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"  CUDA memory cached:    {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
        max_mem = torch.cuda.max_memory_allocated()
        print(f"  CUDA max allocated:    {max_mem / 1024**2:.1f} MB")

    # Total time
    total_time = sum(
        evt.cpu_time_total for evt in prof.key_averages()
    ) / 1_000_000  # microseconds → seconds
    print(f"\n  Total profile time:    {total_time:.4f} s")
    print(f"  Per-step time:         {total_time / steps:.4f} s")
    print(f"{'─'*70}\n")

    # Export trace
    if html:
        trace_path = Path(f"profile_trace_{mode}.html")
        prof.export_chrome_trace(str(trace_path))
        print(f"Chrome trace exported: {trace_path}")

    return {
        "mode": mode,
        "device": device,
        "steps": steps,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "estimated_flops": est_flops,
        "estimated_gflops": est_flops / 1e9,
        "total_time_s": total_time,
        "per_step_time_s": total_time / steps,
        "cuda_memory_mb": torch.cuda.memory_allocated() / 1024**2 if device == "cuda" else 0,
        "profile_table": str(prof.key_averages().table(sort_by=sort_key, row_limit=20)),
    }


def memory_budget_check(config: PrivHSDConfig, device: str = "cuda") -> Dict:
    """Estimate whether the configuration fits in GPU memory.

    Returns dict with memory estimates and warnings.
    """
    if device != "cuda" or not torch.cuda.is_available():
        return {"status": "unknown", "message": "No CUDA device available for memory check"}

    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    model = PrivHSDModelV2(config)
    model_size = sum(p.numel() for p in model.parameters()) * 4 / 1024**3  # fp32
    del model

    # Estimate: model (fp32) + optimizer states + activations + gradients
    if config.dp_enabled:
        # DP requires ~3-5x memory for per-sample gradients
        dp_mult = 3.0 if config.use_ghost_clipping else 5.0
    else:
        dp_mult = 1.0

    estimated_usage = model_size * (1 + 0.1 + dp_mult) + 0.5  # GB (conservative)
    fits = estimated_usage < total_mem * 0.9  # 90% threshold

    result = {
        "total_gpu_memory_gb": round(total_mem, 2),
        "model_size_fp32_gb": round(model_size, 3),
        "estimated_total_usage_gb": round(estimated_usage, 2),
        "dp_multiplier": dp_mult,
        "fits_on_gpu": fits,
        "warning": None if fits else f"Estimated usage ({estimated_usage:.1f}GB) exceeds "
                                      f"90% of available memory ({total_mem:.1f}GB). "
                                      f"Consider smaller batch_size, fewer authors, or fp16.",
    }

    status = "OK" if fits else "WARNING"
    print(f"[{status}] Memory budget: model={model_size:.2f}GB, "
          f"estimated_total={estimated_usage:.1f}GB / {total_mem:.1f}GB available")
    if result["warning"]:
        print(f"  {result['warning']}")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="PrivHSD v2 Profiling Script")
    parser.add_argument("--mode", type=str, default="forward",
                        choices=["forward", "train", "full"],
                        help="Profiling mode")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for profiling")
    parser.add_argument("--steps", type=int, default=10,
                        help="Number of profiling steps")
    parser.add_argument("--html", action="store_true",
                        help="Export Chrome trace HTML")
    parser.add_argument("--memory-check", action="store_true",
                        help="Run memory budget check and exit")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--max-length", type=int, default=None,
                        help="Override max sequence length")
    parser.add_argument("--layers", type=int, default=None,
                        help="Override number of transformer layers")
    parser.add_argument("--heads", type=int, default=None,
                        help="Override number of attention heads")
    parser.add_argument("--dp", action="store_true", default=True,
                        help="Enable DP-SGD (default)")
    parser.add_argument("--no-dp", action="store_false", dest="dp",
                        help="Disable DP-SGD")
    parser.add_argument("--authors", type=int, default=None,
                        help="Override number of author classes")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    print(f"Device: {device}")

    # Config
    config = PrivHSDConfig(
        model_name="albert-base-v2",
        model_type="albert",
        d_model=768,
        n_layers=args.layers or 12,
        n_heads=args.heads or 12,
        d_ff=3072,
        max_seq_len=args.max_length or 256,
        num_hate_classes=2,
        num_authors=args.authors or 100,
        adversarial_levels=("pooler", "token", "head"),
        adversary_hidden_dim=256,
        adversary_num_layers=3,
        disentanglement_weight=0.3,
        mim_weight=0.1,
        orthogonality_weight=0.05,
        batch_size=args.batch_size or 16,
        dp_enabled=args.dp,
        target_epsilon=8.0,
        use_ghost_clipping=True,
        seed=42,
    )

    # Memory check only
    if args.memory_check:
        memory_budget_check(config, device)
        return

    # Initialize model
    print(f"\nInitializing PrivHSDModelV2...")
    model = PrivHSDModelV2(config)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # FLOP estimation
    flops = estimate_flops(config, args.mode)
    print(f"Estimated {args.mode} FLOPs: {flops:.0f} ({flops / 1e9:.2f} GFLOPs)")

    # Memory budget
    memory_budget_check(config, device)

    # Profile
    results = profile_forward(
        model=model,
        config=config,
        device=device,
        steps=args.steps,
        mode=args.mode,
        html=args.html,
    )

    print(f"\nProfiling complete. Key metrics:")
    print(f"  Mode:            {results['mode']}")
    print(f"  Total params:    {results['total_params']:,}")
    print(f"  Est. FLOPs:      {results['estimated_flops']:.0f}")
    print(f"  Est. GFLOPs:     {results['estimated_gflops']:.2f}")
    print(f"  Per-step time:   {results['per_step_time_s']:.4f}s")
    if results['cuda_memory_mb'] > 0:
        print(f"  CUDA memory:     {results['cuda_memory_mb']:.0f} MB")


if __name__ == "__main__":
    main()
