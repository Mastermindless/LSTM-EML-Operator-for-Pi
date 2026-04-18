"""Inference CLI for the LSTM-EML Pi controller.

Usage
-----
Interactive:
    python inference.py --ckpt lstm_eml_pi.pt

Single-shot:
    python inference.py --ckpt lstm_eml_pi.pt --n 100

    python inference.py --ckpt lstm_eml_pi.pt --n 9999 --hard      # hard mode
"""
from __future__ import annotations

import argparse
import time

import torch

from eml_operator import common_prefix_len, eml_pi, true_pi
from lstm_eml_model import LSTM_EML
from pi_generator import sample_batch


def load_model(ckpt_path: str, device: str = "cpu") -> LSTM_EML:
    state = torch.load(ckpt_path, map_location=device)
    model = LSTM_EML(hidden=state["config"].get("hidden", 128)).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


def run_once(model: LSTM_EML, n: int, use_length_input: bool, device: str = "cpu"):
    batch = sample_batch(1, n, n, use_length_input=use_length_input)
    t0 = time.time()
    with torch.no_grad():
        p_float = model(batch.tokens.to(device), batch.lengths).item()
    t_lstm = time.time() - t0
    p_star = max(1, int(round(p_float)))

    t0 = time.time()
    pi_eml = eml_pi(p_star)
    t_eml = time.time() - t0

    pi_ref = true_pi(n)
    match = common_prefix_len(pi_eml, pi_ref)

    print(f"\nRequested precision N            : {n}")
    print(f"Mode                             : "
          f"{'shortcut (N shown to LSTM)' if use_length_input else 'hard (N hidden)'}")
    print(f"LSTM precision (raw)             : {p_float:.3f}")
    print(f"LSTM precision (rounded P*)      : {p_star}")
    print(f"LSTM call time                   : {t_lstm*1000:.2f} ms")
    print(f"EML   call time                  : {t_eml*1000:.2f} ms")
    print(f"Matching digits vs. mpmath π     : {match}/{n}  "
          f"{'✔ perfect' if match >= n else '✘ short by ' + str(n - match)}")

    display_n = min(n, 80)
    print(f"\n  mpmath π@{n} (first {display_n} digits):")
    print("    3." + pi_ref[1:display_n])
    print(f"  EML   π  (first {display_n} digits):")
    print("    3." + pi_eml[1:display_n])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="lstm_eml_pi.pt")
    parser.add_argument("--n", type=int, default=None,
                        help="single-shot precision (else interactive)")
    parser.add_argument("--hard", action="store_true",
                        help="disable length-input shortcut")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    model = load_model(args.ckpt, device=args.device)
    use_length = not args.hard

    if args.n is not None:
        run_once(model, args.n, use_length_input=use_length, device=args.device)
        return

    print("LSTM-EML Pi  —  interactive inference. Enter a precision (1 – 9999).")
    print("Ctrl-D / empty line to quit.\n")
    while True:
        try:
            s = input("precision N > ").strip()
        except EOFError:
            break
        if not s:
            break
        try:
            n = int(s)
        except ValueError:
            print("  not an integer")
            continue
        if n < 1:
            print("  precision must be ≥ 1")
            continue
        run_once(model, n, use_length_input=use_length, device=args.device)


if __name__ == "__main__":
    main()
