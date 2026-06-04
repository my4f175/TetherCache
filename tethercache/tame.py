from __future__ import annotations

from typing import Optional, Tuple

import torch


def _per_head_per_channel_stat(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    x32 = x.float()
    mu = x32.mean(dim=(0, 1))
    sigma = x32.std(dim=(0, 1), unbiased=False).clamp_min(1e-6)
    return mu, sigma


def adain_toward_trusted(
    x: torch.Tensor,                     # [B, F, H, D] candidate frame
    trusted: Optional[torch.Tensor],     # [B, L_t, H, D] sink ∪ memory K (or V)
    tau: float,
) -> Tuple[torch.Tensor, bool]:

    if tau <= 0.0:
        return x, False
    if x.numel() == 0:
        return x, False
    if trusted is None or trusted.numel() == 0:
        # No anchor available — caller should have gated on this. Defensive
        # fall-through.
        return x, False

    mu_x, sigma_x = _per_head_per_channel_stat(x)            # [H, D]
    mu_t, sigma_t = _per_head_per_channel_stat(trusted)      # [H, D]

    x32 = x.float()
    # Broadcast (H, D) -> (1, 1, H, D).
    mu_x_b = mu_x.unsqueeze(0).unsqueeze(0)
    sigma_x_b = sigma_x.unsqueeze(0).unsqueeze(0)
    mu_t_b = mu_t.unsqueeze(0).unsqueeze(0)
    sigma_t_b = sigma_t.unsqueeze(0).unsqueeze(0)

    x_norm = (x32 - mu_x_b) / sigma_x_b
    x_renorm = sigma_t_b * x_norm + mu_t_b
    out = (1.0 - tau) * x32 + tau * x_renorm
    return out.to(dtype=x.dtype), True
