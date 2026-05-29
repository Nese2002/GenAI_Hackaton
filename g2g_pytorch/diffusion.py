"""DDPM noise schedule and DDIM sampler for piano-roll diffusion.

References:
    DDPM:  Ho et al., 2020  — https://arxiv.org/abs/2006.11239
    DDIM:  Song et al., 2020 — https://arxiv.org/abs/2010.02502
    Cosine schedule: Nichol & Dhariwal, 2021
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine beta schedule (Nichol & Dhariwal, 2021). Returns betas (T,)."""
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return betas.clamp(1e-4, 0.9999)


class DiffusionSchedule:
    """Pre-computes all quantities needed for forward/reverse diffusion.

    Piano-roll values are in [0, 1]. We normalise to [-1, 1] before
    adding noise and convert back at the end of sampling.
    """

    def __init__(self, T: int = 1000, device: torch.device = torch.device("cpu")) -> None:
        self.T = T
        betas = cosine_beta_schedule(T).to(device)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar[:-1], (1, 0), value=1.0)

        self.betas           = betas
        self.alphas          = alphas
        self.alphas_bar      = alphas_bar          # ᾱ_t
        self.alphas_bar_prev = alphas_bar_prev     # ᾱ_{t-1}
        self.sqrt_ab         = alphas_bar.sqrt()
        self.sqrt_one_m_ab   = (1.0 - alphas_bar).sqrt()

    def to(self, device: torch.device) -> "DiffusionSchedule":
        for attr in ("betas", "alphas", "alphas_bar", "alphas_bar_prev",
                     "sqrt_ab", "sqrt_one_m_ab"):
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _gather(values: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Index a 1-D tensor by t and broadcast to (B, 1, 1, 1)."""
        return values[t].view(-1, 1, 1, 1)

    # ------------------------------------------------------------------
    # Forward process
    # ------------------------------------------------------------------
    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add noise to x0 at step t.

        Args:
            x0: (B, C, H, W) clean roll, normalised to [-1, 1].
            t:  (B,) integer timesteps.
        Returns:
            x_t: noisy roll at step t.
            noise: the Gaussian noise that was added.
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab  = self._gather(self.sqrt_ab,       t)
        sqrt_mab = self._gather(self.sqrt_one_m_ab, t)
        x_t = sqrt_ab * x0 + sqrt_mab * noise
        return x_t, noise

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------
    def training_loss(
        self,
        model,
        x0: torch.Tensor,
        x_content: torch.Tensor,
        style_vec: torch.Tensor,
        cfg_content_p: float = 0.1,
        cfg_style_p:   float = 0.1,
    ) -> torch.Tensor:
        """Sample t, add noise, predict noise, return MSE loss.

        CFG dropout: randomly zero out x_content or style_vec during training.
        """
        B = x0.shape[0]
        device = x0.device
        t = torch.randint(0, self.T, (B,), device=device)

        # CFG dropout
        if cfg_content_p > 0:
            mask_c = (torch.rand(B, device=device) >= cfg_content_p).float()
            x_content = x_content * mask_c.view(B, 1, 1, 1)
        if cfg_style_p > 0:
            mask_s = (torch.rand(B, device=device) >= cfg_style_p).float()
            style_vec = style_vec * mask_s.view(B, 1)

        x_t, noise = self.q_sample(x0, t)
        noise_pred = model(x_t, x_content, style_vec, t)
        return F.mse_loss(noise_pred, noise)

    # ------------------------------------------------------------------
    # DDIM sampling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        x_content: torch.Tensor,
        style_vec:  torch.Tensor,
        ddim_steps: int = 50,
        cfg_scale:  float = 3.0,
        eta:        float = 0.0,   # 0 = deterministic DDIM
    ) -> torch.Tensor:
        """Generate a piano roll from noise via DDIM.

        Returns:
            x0: (B, C, H, W) float in [0, 1].
        """
        B, _, H, W = x_content.shape
        device = x_content.device

        # Evenly-spaced timesteps from T-1 down to 0
        step_size = self.T // ddim_steps
        timesteps = list(range(self.T - 1, -1, -step_size))[:ddim_steps]

        x = torch.randn(B, 2, H, W, device=device)  # pure noise

        zeros_c = torch.zeros_like(x_content)
        zeros_s = torch.zeros_like(style_vec)

        for i, t_val in enumerate(timesteps):
            t = torch.full((B,), t_val, device=device, dtype=torch.long)
            t_prev_val = timesteps[i + 1] if i + 1 < len(timesteps) else -1

            # CFG: interpolate between unconditional and conditional prediction
            eps_cond   = model(x, x_content, style_vec, t)
            eps_uncond = model(x, zeros_c,   zeros_s,   t)
            eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

            # DDIM update
            ab_t    = self._gather(self.alphas_bar, t)
            ab_prev = (self.alphas_bar[t_prev_val] if t_prev_val >= 0
                       else torch.ones(1, device=device)).view(-1, 1, 1, 1)

            x0_pred = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            sigma = (eta * ((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev)).sqrt()
                     if eta > 0 else torch.zeros(1, device=device))
            noise  = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
            x = ab_prev.sqrt() * x0_pred + (1 - ab_prev - sigma**2).sqrt() * eps + sigma * noise

        # Denormalise from [-1, 1] to [0, 1]
        return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


__all__ = ["DiffusionSchedule", "cosine_beta_schedule"]
