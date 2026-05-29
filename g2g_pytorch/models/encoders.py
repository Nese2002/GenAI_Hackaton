"""Content and style encoders.

The content encoder mirrors the original Groove2Groove CNN+BiGRU layout but
in PyTorch: a stack of Conv2d/MaxPool2d over the (pitch, time) image,
followed by a flatten and a recurrent net producing a per-time-step memory
tensor used as the cross-attention source by the decoder.

The style encoder runs the same 2-D backbone on Z, then a 1-D conv stack
(matching the paper) over the time axis, then a GRU. We take the final
hidden state as a single global style vector.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Vector Quantization bottleneck
# ---------------------------------------------------------------------------

class VectorQuantizer(nn.Module):
    """Straight-through VQ bottleneck (VQ-VAE, van den Oord et al. 2017).

    Replaces each continuous encoder vector with the nearest codebook entry.
    Gradient flows back through the straight-through estimator.

    Loss returned = codebook_loss + commitment_cost * commitment_loss.
    """

    def __init__(self, num_codes: int, code_dim: int,
                 commitment_cost: float = 0.25) -> None:
        super().__init__()
        self.num_codes      = num_codes
        self.code_dim       = code_dim
        self.commitment_cost = commitment_cost
        self.codebook = nn.Embedding(num_codes, code_dim)
        nn.init.uniform_(self.codebook.weight,
                         -1.0 / num_codes, 1.0 / num_codes)

    def forward(self, z_e: torch.Tensor) -> tuple:
        """
        Args:
            z_e: (B, T, D) continuous encoder output.
        Returns:
            z_q: (B, T, D) quantized output (straight-through in backward).
            loss: scalar — codebook + commitment loss.
        """
        B, T, D = z_e.shape
        flat = z_e.reshape(-1, D)                        # (B*T, D)

        # Squared distances to each codebook entry.
        dists = (
            flat.pow(2).sum(1, keepdim=True)
            + self.codebook.weight.pow(2).sum(1)
            - 2.0 * flat @ self.codebook.weight.t()
        )                                                 # (B*T, K)

        indices = dists.argmin(dim=1)                    # (B*T,)
        z_q_flat = self.codebook(indices)                # (B*T, D)
        z_q = z_q_flat.reshape(B, T, D)

        # Codebook loss: moves codes toward encoder outputs.
        # Commitment loss: moves encoder outputs toward codes.
        loss = (F.mse_loss(z_q, z_e.detach())
                + self.commitment_cost * F.mse_loss(z_e, z_q.detach()))

        # Straight-through: copy gradient of z_q to z_e.
        z_q = z_e + (z_q - z_e).detach()
        return z_q, loss


class Conv2dStack(nn.Module):
    """Sequential Conv2d + ELU + MaxPool2d blocks."""

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        kernels: Sequence[Tuple[int, int]],
        pools: Sequence[Tuple[int, int]],
    ) -> None:
        super().__init__()
        layers = []
        c_in = in_channels
        for c, k, p in zip(channels, kernels, pools):
            layers.append(nn.Conv2d(c_in, c, kernel_size=k, padding=(k[0] // 2, k[1] // 2)))
            layers.append(nn.ELU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=p, stride=p))
            c_in = c
        self.net = nn.Sequential(*layers)
        self.out_channels = c_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, P, T) -> (B, C', P', T')
        return self.net(x)


class Conv1dStack(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        kernels: Sequence[int],
        pools: Sequence[int],
    ) -> None:
        super().__init__()
        layers = []
        c_in = in_channels
        for c, k, p in zip(channels, kernels, pools):
            layers.append(nn.Conv1d(c_in, c, kernel_size=k, padding=k // 2))
            layers.append(nn.ELU(inplace=True))
            layers.append(nn.MaxPool1d(kernel_size=p, stride=p))
            c_in = c
        self.net = nn.Sequential(*layers)
        self.out_channels = c_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        return self.net(x)


# --------------------------------------------------------------------------
# Content encoder: returns a per-time-step memory (B, T', H)
# --------------------------------------------------------------------------


class ContentEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        cnn_channels: Sequence[int],
        cnn_kernels: Sequence[Tuple[int, int]],
        cnn_pools: Sequence[Tuple[int, int]],
        rnn_hidden: int,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.cnn = Conv2dStack(in_channels, cnn_channels, cnn_kernels, cnn_pools)
        self._rnn_hidden = rnn_hidden
        self._bidir      = bidirectional
        self.rnn: nn.Module = nn.Identity()
        self._built = False

    def _build_rnn(self, gru_in_size: int, device: torch.device) -> None:
        self.rnn = nn.GRU(
            input_size=gru_in_size,
            hidden_size=self._rnn_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=self._bidir,
        ).to(device)
        self._built = True

    @property
    def output_dim(self) -> int:
        return self._rnn_hidden * (2 if self._bidir else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.cnn(x)                                # (B, C', P', T')
        B, C, P, T = f.shape
        f = f.permute(0, 3, 1, 2).contiguous()         # (B, T', C', P')
        f = f.view(B, T, C * P)                        # (B, T', C'*P')
        if not self._built:
            self._build_rnn(f.shape[-1], f.device)
        memory, _ = self.rnn(f)                        # (B, T', H)
        return memory


# --------------------------------------------------------------------------
# Style encoder: returns a single global style vector (B, style_dim)
# --------------------------------------------------------------------------


class StyleEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        cnn2d_channels: Sequence[int],
        cnn2d_kernels: Sequence[Tuple[int, int]],
        cnn2d_pools: Sequence[Tuple[int, int]],
        cnn1d_channels: Sequence[int],
        cnn1d_kernels: Sequence[int],
        cnn1d_pools: Sequence[int],
        rnn_hidden: int,
        style_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.cnn2d = Conv2dStack(in_channels, cnn2d_channels, cnn2d_kernels, cnn2d_pools)
        # The 1-D conv stack runs over the time axis after we've folded
        # (channels, pitch) into one feature axis. Lazy-build it on first
        # forward so we don't have to track shapes here.
        self._cnn1d_args = (cnn1d_channels, cnn1d_kernels, cnn1d_pools)
        self.cnn1d: nn.Module = nn.Identity()
        self._rnn_hidden = rnn_hidden
        self.rnn: nn.Module = nn.Identity()
        self.proj = nn.Linear(rnn_hidden, style_dim)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self._built = False

    def _build(self, in_channels_1d: int, device: torch.device) -> None:
        channels, kernels, pools = self._cnn1d_args
        self.cnn1d = Conv1dStack(in_channels_1d, channels, kernels, pools).to(device)
        self.rnn = nn.GRU(
            input_size=self.cnn1d.out_channels,
            hidden_size=self._rnn_hidden,
            num_layers=1,
            batch_first=True,
        ).to(device)
        self._built = True

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, C, P, T)
        f = self.cnn2d(z)                              # (B, C', P', T')
        B, C, P, T = f.shape
        f = f.permute(0, 1, 3, 2).contiguous()         # (B, C', T', P')
        f = f.view(B, C * P, T)                        # (B, C'*P', T')   (for 1-D conv)
        if not self._built:
            self._build(f.shape[1], f.device)
        f = self.cnn1d(f)                              # (B, C1d, T'')
        f = f.transpose(1, 2).contiguous()             # (B, T'', C1d)
        _, h = self.rnn(f)                             # h: (1, B, H)
        h = h.squeeze(0)                               # (B, H)
        s = self.proj(h)                               # (B, style_dim)
        s = self.dropout(s)
        return s


__all__ = ["ContentEncoder", "StyleEncoder"]
