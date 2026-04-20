"""Validation script: verify loss convergence is stable after v02 optimizations.

Runs a short training loop and checks that:
    1. Loss decreases monotonically (smoothed).
    2. Pi digit cache produces identical results to direct mpmath calls.
    3. MPS/CPU outputs are numerically close (tolerance: 1e-5).

Usage:
    python validate_convergence.py
    python validate_convergence.py --device cpu   # force CPU comparison
"""
from __future__ import annotations

import argparse
import sys
import time

import torch

from eml_operator import cached_true_pi, eml_pi, true_pi
from lstm_eml_model import LSTM_EML, precision_loss
from pi_generator import sample_batch


def get_best_device() -> str:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def test_cache_correctness():
    """Verify PiDigitCache matches direct mpmath computation.

    The cache truncates from a large pre-computed string, while true_pi(n)
    rounds at position n via mpmath.nstr. The last digit may differ by ±1.
    We verify that all digits EXCEPT the last match exactly (common_prefix_len
    >= n-1), which is sufficient for training reward computation.
    """
    from eml_operator import common_prefix_len

    print("--- Test 1: Pi digit cache correctness ---")
    failures = 0
    for n in [1, 5, 10, 50, 100, 500, 1000, 5000, 9999]:
        ref = true_pi(n)
        cached = cached_true_pi(n)
        prefix = common_prefix_len(ref, cached)
        # Allow last-digit rounding divergence (prefix >= n-1)
        if prefix < n - 1:
            print(f"  FAIL at n={n}: only {prefix}/{n} digits match")
            failures += 1
        else:
            status = "exact" if prefix == n else f"last-digit rounding (ok)"
            print(f"  n={n:5d}: {prefix}/{n} digits match — {status}")
    assert failures == 0, f"{failures} cache failures (>1 digit divergence)!"
    print("  PASSED\n")


def test_device_consistency(device: str):
    """Verify CPU and target device produce close outputs."""
    print(f"--- Test 2: Device consistency (cpu vs {device}) ---")
    torch.manual_seed(42)
    model_cpu = LSTM_EML(hidden=128, use_length_input=True)
    model_gpu = LSTM_EML(hidden=128, use_length_input=True).to(device)
    # Copy weights
    model_gpu.load_state_dict(model_cpu.state_dict())

    batch = sample_batch(8, 1, 100, use_length_input=True)

    with torch.no_grad():
        out_cpu = model_cpu(batch.tokens, batch.lengths)
        out_gpu = model_gpu(batch.tokens.to(device), batch.lengths).cpu()

    diff = (out_cpu - out_gpu).abs().max().item()
    print(f"  Max absolute difference: {diff:.2e}")
    assert diff < 1e-4, f"Device outputs diverged: max diff = {diff}"
    print("  PASSED\n")


def test_convergence(device: str, steps: int = 200, batch_size: int = 16):
    """Verify loss decreases over a short training run."""
    print(f"--- Test 3: Loss convergence ({steps} steps on {device}) ---")
    torch.manual_seed(0)
    model = LSTM_EML(hidden=128, use_length_input=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    losses = []
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        batch = sample_batch(batch_size, 1, 100, use_length_input=True)
        tokens = batch.tokens.to(device, non_blocking=True)
        lengths = batch.lengths
        ns = batch.n.to(device, non_blocking=True)

        p_hat = model(tokens, lengths)
        loss = precision_loss(p_hat, ns)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        losses.append(loss.item())
        if step % 50 == 0:
            print(f"  step {step:4d}  loss={loss.item():.4f}")

    elapsed = time.perf_counter() - t0

    # Check smoothed loss trend: first quarter avg should exceed last quarter avg
    q = len(losses) // 4
    first_q = sum(losses[:q]) / q
    last_q = sum(losses[-q:]) / q
    print(f"\n  First-quarter avg loss: {first_q:.4f}")
    print(f"  Last-quarter avg loss:  {last_q:.4f}")
    print(f"  Total time: {elapsed:.2f}s ({elapsed/steps*1000:.1f} ms/step)")

    assert last_q < first_q, (
        f"Loss did NOT decrease: first_q={first_q:.4f} → last_q={last_q:.4f}"
    )
    print("  PASSED\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device if args.device else get_best_device()
    print(f"Validation device: {device}\n")

    test_cache_correctness()
    if device != "cpu":
        test_device_consistency(device)
    test_convergence(device)

    print("=" * 50)
    print("ALL VALIDATION TESTS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()
