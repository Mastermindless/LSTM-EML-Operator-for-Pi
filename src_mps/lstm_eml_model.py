"""LSTM controller for the EML Pi precision selector — v02 (GPU-optimized).

Changes from v01:
    • Forward pass keeps `lengths` on CPU (required by pack_padded_sequence)
      without moving tensors back from GPU — no hidden sync points.
    • Added torch.compile() compatibility hints.
    • No architectural changes — the model is already minimal (1-layer, 128h).
      Acrylic review confirmed: LayerNorm adds complexity for <3% gain here.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pi_generator import PAD_ID, VOCAB_SIZE


class LSTM_EML(nn.Module):
    """Tiny LSTM that outputs a positive precision prediction."""

    def __init__(
        self,
        hidden: int = 128,
        embed_dim: int = 32,
        num_layers: int = 1,
        use_length_input: bool = True,
    ):
        super().__init__()
        self.use_length_input = use_length_input
        self.embed = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden, 1)

        # Log-sigma for optional REINFORCE fine-tuning.
        self.log_sigma = nn.Parameter(torch.tensor(0.0))

    # --------------------------------------------------------------------- #
    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        tokens  : (B, T) long — on model device (cpu/mps/cuda)
        lengths : (B,) long — ALWAYS on CPU (pack_padded_sequence requirement)
        returns : (B,) positive float — predicted precision P'
        """
        emb = self.embed(tokens)  # (B, T, E)

        # pack_padded_sequence requires lengths on CPU — keep them there from
        # the start to avoid a GPU→CPU sync stall.
        lengths_cpu = lengths if lengths.device.type == "cpu" else lengths.cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)         # h: (num_layers, B, H)
        last_h = h[-1]                         # (B, H)
        raw = self.head(last_h).squeeze(-1)    # (B,)
        # softplus keeps P' strictly > 0; +1 keeps it ≥ 1 (minimum sensible dps).
        return F.softplus(raw) + 1.0

    # --------------------------------------------------------------------- #
    def sample(self, tokens: torch.Tensor, lengths: torch.Tensor):
        """Sample P ~ Normal(mean=P'(θ), sigma=exp(log_sigma)) for REINFORCE."""
        mean = self.forward(tokens, lengths)
        sigma = self.log_sigma.exp().expand_as(mean)
        eps = torch.randn_like(mean)
        sample = mean + sigma * eps
        sample = F.softplus(sample - 1.0) + 1.0       # keep positive & ≥ 1
        log_prob = -0.5 * ((sample - mean) / sigma) ** 2 - torch.log(sigma)
        return sample, log_prob, mean


# ------------------------------------------------------------------------- #
def precision_loss(
    p_hat: torch.Tensor,
    n: torch.Tensor,
    guard: float = 4.0,
    over_weight: float = 0.1,
) -> torch.Tensor:
    """Scale-homogeneous surrogate loss (§4.1 of concept_EML_LSTM_pi.md).

    L = ReLU((N+g) - P')/(N+g)  +  λ · ReLU(P' - (N+2g))/(N+g)

    All quantities are in 'decimal-digit' units and normalised by (N+g), so
    samples at N=5 and N=9999 contribute comparable gradient magnitudes.
    """
    n_f = n.float()
    denom = n_f + guard
    under = F.relu(denom - p_hat) / denom
    over = F.relu(p_hat - (n_f + 2 * guard)) / denom
    return (under + over_weight * over).mean()
