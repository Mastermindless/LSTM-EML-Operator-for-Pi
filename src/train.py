"""Curriculum-SGD trainer for the LSTM-EML Pi controller.

Phases (see §6 of concept_EML_LSTM_pi.md):

    Phase 0  N ∈ [1,   10]    shortcut ON       warm start
    Phase 1  N ∈ [1,  100]    shortcut ON
    Phase 2  N ∈ [1, 1000]    shortcut ON
    Phase 3  N ∈ [1, 9999]    shortcut ON  + REINFORCE refinement
    Phase 4  N ∈ [1, 9999]    shortcut OFF (hard mode)

Run:
    python train.py                       # full curriculum
    python train.py --quick               # tiny config for smoke-test
"""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import List

import torch
from torch.optim import Adam

from eml_operator import common_prefix_len, eml_pi, true_pi
from lstm_eml_model import LSTM_EML, precision_loss
from pi_generator import sample_batch


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
def eml_reward(p_hat: torch.Tensor, ns: torch.Tensor) -> torch.Tensor:
    """Digit-agreement reward R ∈ [0,1]. Non-differentiable, used by REINFORCE."""
    out = []
    for p, n in zip(p_hat.tolist(), ns.tolist()):
        p_int = max(1, int(round(p)))
        got = eml_pi(p_int)
        ref = true_pi(int(n))
        d = common_prefix_len(got, ref)
        out.append(min(1.0, d / float(max(1, int(n)))))
    return torch.tensor(out, dtype=torch.float32)


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
        batch = sample_batch(
            batch_size,
            cfg.min_n,
            cfg.max_n,
            use_length_input=cfg.use_length_input,
        )
        tokens = batch.tokens.to(device)
        lengths = batch.lengths
        ns = batch.n.to(device)

        if cfg.reinforce:
            sampled, log_prob, mean = model.sample(tokens, lengths)
            surrogate = precision_loss(mean, ns, guard=guard)

            with torch.no_grad():
                R = eml_reward(sampled.detach(), ns).to(device)
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
            ref = true_pi(n)
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
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="tiny curriculum")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--ckpt", type=str, default="lstm_eml_pi.pt")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    phases = QUICK_PHASES if args.quick else DEFAULT_PHASES

    torch.manual_seed(0)
    model = LSTM_EML(hidden=128, use_length_input=True).to(args.device)
    opt = Adam(model.parameters(), lr=args.lr)

    for cfg in phases:
        print(f"\n=== {cfg.name}  N ∈ [{cfg.min_n}, {cfg.max_n}]  "
              f"shortcut={cfg.use_length_input}  reinforce={cfg.reinforce} ===")
        run_phase(model, opt, cfg, batch_size=args.batch, device=args.device)

        eval_ns = [1, 5, 10, 50, 200, 1000, 5000, 9999]
        eval_ns = [n for n in eval_ns if cfg.min_n <= n <= cfg.max_n]
        if eval_ns:
            evaluate(model, eval_ns, use_length_input=cfg.use_length_input,
                     device=args.device)

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
