"""Curriculum-SGD trainer for the LSTM-EML Pi controller — v02 (GPU-optimized).

Changes from v01:
    • Auto-detects MPS (Apple Silicon) / CUDA / CPU — GPU-first strategy.
    • Uses PiDigitCache for O(1) pi digit generation (was O(n·log n) per sample).
    • LRU-cached eml_pi in reward computation eliminates redundant mpmath calls.
    • Lengths tensor stays on CPU to avoid GPU→CPU sync in pack_padded_sequence.
    • Micro-benchmark printed at startup for baseline comparison.

Phases (see §6 of concept_EML_LSTM_pi.md):

    Phase 0  N ∈ [1,   10]    shortcut ON       warm start
    Phase 1  N ∈ [1,  100]    shortcut ON
    Phase 2  N ∈ [1, 1000]    shortcut ON
    Phase 3  N ∈ [1, 9999]    shortcut ON  + REINFORCE refinement
    Phase 4  N ∈ [1, 9999]    shortcut OFF (hard mode)

Run:
    python train.py                       # full curriculum (auto-detects GPU)
    python train.py --quick               # tiny config for smoke-test
    python train.py --device cpu          # force CPU
"""
from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from typing import List

import torch
from torch.optim import Adam

from eml_operator import cached_true_pi, common_prefix_len, eml_pi, true_pi
from lstm_eml_model import LSTM_EML, precision_loss
from pi_generator import sample_batch


# --------------------------------------------------------------------------- #
# Device auto-detection
# --------------------------------------------------------------------------- #
def get_best_device() -> str:
    """Return the fastest available device: mps > cuda > cpu."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# --------------------------------------------------------------------------- #
# Phase configuration
# --------------------------------------------------------------------------- #
@dataclass
class PhaseCfg:
    name: str
    min_n: int
    max_n: int
    steps: int
    use_length_input: bool
    reinforce: bool = False


DEFAULT_PHASES: List[PhaseCfg] = [
    PhaseCfg("phase0",    1,   10,   2_000, True),
    PhaseCfg("phase1",    1,  100,  10_000, True),
    PhaseCfg("phase2",    1, 1000,  20_000, True),
    PhaseCfg("phase3",    1, 9999,  50_000, True, reinforce=True),
    PhaseCfg("phase4",    1, 9999,  20_000, False),
]

QUICK_PHASES: List[PhaseCfg] = [
    PhaseCfg("q0", 1,  10,  200, True),
    PhaseCfg("q1", 1, 100,  400, True),
]


# --------------------------------------------------------------------------- #
# Reward (non-differentiable, used by REINFORCE)
# --------------------------------------------------------------------------- #
def eml_reward(p_hat: torch.Tensor, ns: torch.Tensor) -> torch.Tensor:
    """Digit-agreement reward R ∈ [0,1].

    Uses LRU-cached eml_pi — repeated dps values are O(1) lookups.
    """
    rewards = torch.empty(p_hat.shape[0], dtype=torch.float32)
    for i, (p, n) in enumerate(zip(p_hat.tolist(), ns.tolist())):
        p_int = max(1, int(round(p)))
        n_int = int(n)
        got = eml_pi(p_int)  # LRU cached
        ref = cached_true_pi(n_int)  # O(1) slice from pre-computed cache
        d = common_prefix_len(got, ref)
        rewards[i] = min(1.0, d / float(max(1, n_int)))
    return rewards


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def run_phase(
    model: LSTM_EML,
    opt: torch.optim.Optimizer,
    cfg: PhaseCfg,
    batch_size: int = 32,
    guard: float = 4.0,
    log_every: int = 100,
    device: str = "cpu",
):
    model.train()
    model.use_length_input = cfg.use_length_input

    baseline = 0.5  # running-mean REINFORCE baseline
    baseline_beta = 0.95

    for step in range(1, cfg.steps + 1):
        # Data generation uses cached_true_pi internally (O(1) per sample).
        batch = sample_batch(
            batch_size,
            cfg.min_n,
            cfg.max_n,
            use_length_input=cfg.use_length_input,
        )
        # tokens go to GPU; lengths STAY on CPU (pack_padded_sequence requires it).
        tokens = batch.tokens.to(device, non_blocking=True)
        lengths = batch.lengths  # remains on CPU — no sync point
        ns = batch.n.to(device, non_blocking=True)

        if cfg.reinforce:
            sampled, log_prob, mean = model.sample(tokens, lengths)
            surrogate = precision_loss(mean, ns, guard=guard)

            with torch.no_grad():
                R = eml_reward(sampled.detach(), ns.cpu()).to(device)
                baseline = baseline_beta * baseline + (1 - baseline_beta) * R.mean().item()
            pg_loss = -((R - baseline) * log_prob).mean()
            loss = surrogate + 0.3 * pg_loss
        else:
            p_hat = model(tokens, lengths)
            loss = precision_loss(p_hat, ns, guard=guard)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % log_every == 0 or step == 1:
            with torch.no_grad():
                p_eval = model(tokens, lengths)
                gap = (p_eval - ns.float()).abs().mean().item()
            print(
                f"[{cfg.name}] step {step:6d}/{cfg.steps}  "
                f"loss={loss.item():.4f}  |P'-N|avg={gap:.2f}  "
                f"{'RL_base=' + f'{baseline:.3f}' if cfg.reinforce else ''}"
            )


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(model: LSTM_EML, ns: List[int], use_length_input: bool, device: str = "cpu"):
    """Inference-style digit-agreement check."""
    model.eval()
    model.use_length_input = use_length_input
    hits = 0
    total_digits = 0
    match_digits = 0
    with torch.no_grad():
        for n in ns:
            batch = sample_batch(1, n, n, use_length_input=use_length_input)
            p_hat = model(batch.tokens.to(device), batch.lengths).item()
            p_int = max(1, int(round(p_hat)))
            ref = cached_true_pi(n)
            got = eml_pi(p_int)
            d = common_prefix_len(got, ref)
            hits += int(d >= n)
            total_digits += n
            match_digits += min(d, n)
    print(
        f"  eval  ({'shortcut' if use_length_input else 'hard'})  "
        f"perfect={hits}/{len(ns)}  digit-recall={match_digits/total_digits:.4f}"
    )


# --------------------------------------------------------------------------- #
# Micro-benchmark
# --------------------------------------------------------------------------- #
def micro_benchmark(model: LSTM_EML, device: str, batch_size: int = 32):
    """Time one forward+backward pass for baseline measurement."""
    model.train()
    batch = sample_batch(batch_size, 1, 100, use_length_input=True)
    tokens = batch.tokens.to(device)
    lengths = batch.lengths
    ns = batch.n.to(device)

    # Warmup
    for _ in range(3):
        p = model(tokens, lengths)
        loss = precision_loss(p, ns)
        loss.backward()
        model.zero_grad(set_to_none=True)

    # Sync before timing
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    n_iters = 50
    for _ in range(n_iters):
        p = model(tokens, lengths)
        loss = precision_loss(p, ns)
        loss.backward()
        model.zero_grad(set_to_none=True)

    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - t0
    ms_per_step = (elapsed / n_iters) * 1000
    print(f"  Micro-benchmark ({device}): {ms_per_step:.2f} ms/step  "
          f"({n_iters} iters, batch={batch_size})")
    return ms_per_step


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="tiny curriculum")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--ckpt", type=str, default="lstm_eml_pi.pt")
    parser.add_argument("--device", type=str, default=None,
                        help="force device (default: auto-detect)")
    args = parser.parse_args()

    device = args.device if args.device else get_best_device()
    phases = QUICK_PHASES if args.quick else DEFAULT_PHASES

    print(f"Device: {device}")
    print(f"Phases: {len(phases)} ({'quick' if args.quick else 'full curriculum'})")

    torch.manual_seed(0)
    model = LSTM_EML(hidden=128, use_length_input=True).to(device)
    opt = Adam(model.parameters(), lr=args.lr)

    # Baseline benchmark
    print("\n--- Baseline micro-benchmark ---")
    micro_benchmark(model, device, batch_size=args.batch)

    # Pre-warm the pi cache to max precision needed
    max_n = max(p.max_n for p in phases)
    print(f"\nPre-computing pi digits to {max_n} precision...")
    t0 = time.perf_counter()
    cached_true_pi(max_n)
    print(f"  Done in {time.perf_counter() - t0:.2f}s (one-time cost)\n")

    for cfg in phases:
        print(f"\n=== {cfg.name}  N ∈ [{cfg.min_n}, {cfg.max_n}]  "
              f"shortcut={cfg.use_length_input}  reinforce={cfg.reinforce} ===")
        t_phase = time.perf_counter()
        run_phase(model, opt, cfg, batch_size=args.batch, device=device)
        elapsed = time.perf_counter() - t_phase
        print(f"  [{cfg.name}] completed in {elapsed:.1f}s")

        eval_ns = [1, 5, 10, 50, 200, 1000, 5000, 9999]
        eval_ns = [n for n in eval_ns if cfg.min_n <= n <= cfg.max_n]
        if eval_ns:
            evaluate(model, eval_ns, use_length_input=cfg.use_length_input,
                     device=device)

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {"hidden": 128},
        },
        args.ckpt,
    )
    print(f"\nSaved checkpoint -> {args.ckpt}")


if __name__ == "__main__":
    main()
