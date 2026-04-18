"""Training sample generator for the LSTM-EML Pi controller.

Produces batches of (N, input_tokens, length_mask) where each sample is a random
requested precision N with its Pi digit string (used as the LSTM input in both
shortcut and hard modes).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from eml_operator import true_pi


# Tiny vocabulary: digits 0-9, '.', PAD, LEN_TOK
PAD_ID = 10
DOT_ID = 11
LEN_TOK_ID = 12
VOCAB_SIZE = 13


def encode_pi_string(pi_digits: str) -> List[int]:
    """Encode 'digit string' with an implicit leading '3.' → tokens."""
    # pi_digits = "3141592..." (no decimal). Insert '.' after the first digit.
    tokens = [int(pi_digits[0]), DOT_ID]
    tokens.extend(int(c) for c in pi_digits[1:])
    return tokens


def build_input(
    n: int,
    pi_digits: str,
    use_length_input: bool,
    max_visible: int | None = None,
) -> List[int]:
    """Assemble LSTM input tokens.

    shortcut / use_length_input = True:
        [LEN_TOK, <digits of N in base 10>, LEN_TOK, <pi digits with '.'>]

    hard mode = False:
        [<pi digits with '.'>]
        The model must infer N from the number of tokens.
    """
    pi_tokens = encode_pi_string(pi_digits)
    if max_visible is not None and max_visible < len(pi_tokens):
        pi_tokens = pi_tokens[:max_visible]

    if use_length_input:
        n_digit_tokens = [int(c) for c in str(n)]
        return [LEN_TOK_ID] + n_digit_tokens + [LEN_TOK_ID] + pi_tokens
    return pi_tokens


@dataclass
class Batch:
    n: torch.Tensor          # (B,) long — target precision
    tokens: torch.Tensor     # (B, T) long — padded input ids
    lengths: torch.Tensor    # (B,) long — true sequence length per sample


def sample_batch(
    batch_size: int,
    min_n: int,
    max_n: int,
    use_length_input: bool,
    max_visible: int | None = None,
    rng: torch.Generator | None = None,
) -> Batch:
    """Random batch of (N, encoded input)."""
    if rng is None:
        rng = torch.Generator().manual_seed(torch.seed() & 0xFFFFFFFF)

    ns = torch.randint(min_n, max_n + 1, (batch_size,), generator=rng)
    sequences: List[List[int]] = []
    for n in ns.tolist():
        pi_digits = true_pi(n)
        seq = build_input(n, pi_digits, use_length_input, max_visible=max_visible)
        sequences.append(seq)

    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    T = int(lengths.max().item())
    padded = torch.full((batch_size, T), PAD_ID, dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)

    return Batch(n=ns.long(), tokens=padded, lengths=lengths)


if __name__ == "__main__":
    batch = sample_batch(4, 1, 20, use_length_input=True)
    print("N:", batch.n.tolist())
    print("lens:", batch.lengths.tolist())
    print("tokens[0]:", batch.tokens[0].tolist())
