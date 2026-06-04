from __future__ import annotations

import math
from typing import Tuple

import torch


# ---------------------------------------------------------------------------
# Attention Mass.
# ---------------------------------------------------------------------------

def attention_mass_score(
    q_chunk: torch.Tensor,         # [B, q_len, H, D]
    k_candidates: torch.Tensor,    # [B, n_cand * frame_seqlen, H, D]
    frame_seqlen: int,
    temperature: float = 1.0,
) -> torch.Tensor:

    B, q_len, H, D = q_chunk.shape
    total_k = k_candidates.shape[1]
    n_cand = total_k // frame_seqlen
    if n_cand == 0:
        return torch.zeros(
            B, 0, device=q_chunk.device, dtype=torch.float32
        )

    # QK^T per head in fp32 for stability. q_len is the chunk size, which
    # is small (3-4 in Self-Forcing), so the [B, H, q_len, total_k] tensor
    # stays modest. We collapse heads / queries / tokens-within-frame down
    # to a single scalar per (batch, candidate frame).
    q32 = q_chunk.float().permute(0, 2, 1, 3)        # [B, H, q_len, D]
    k32 = k_candidates.float().permute(0, 2, 1, 3)   # [B, H, total_k, D]
    scale = 1.0 / math.sqrt(D)
    logits = (q32 @ k32.transpose(-1, -2)) * scale   # [B, H, q_len, total_k]
    logits = logits.view(B, H, q_len, n_cand, frame_seqlen)
    frame_logits = logits.mean(dim=(1, 2, 4))        # [B, n_cand]

    # Softmax across candidates with temperature.
    frame_logits = frame_logits / max(temperature, 1e-6)
    return torch.softmax(frame_logits, dim=-1)       # [B, n_cand]


# ---------------------------------------------------------------------------
# Temporal Diversity.
# ---------------------------------------------------------------------------

def temporal_diversity_score(
    candidate_frame_indices: torch.Tensor,   # [n_cand], long
    attention_mass: torch.Tensor,            # [B, n_cand]
    sigma_floor: float = 1.0,
) -> torch.Tensor:

    if candidate_frame_indices.numel() == 0:
        return torch.zeros_like(attention_mass)

    diffs = (candidate_frame_indices.float().unsqueeze(0)
             - candidate_frame_indices.float().unsqueeze(1)).abs()  # [n, n]
    extent = float(
        candidate_frame_indices.max().item()
        - candidate_frame_indices.min().item() + 1
    )
    sigma_t = max(sigma_floor, 0.5 * extent)
    sim = torch.exp(-diffs / sigma_t)                                # [n, n]
    sim.fill_diagonal_(0.0)  # don't let a candidate penalise itself.

    # Per (batch, candidate) max-similarity weighted by neighbour mass.
    redundancy = (sim.unsqueeze(0) * attention_mass.unsqueeze(1)).max(dim=-1).values
    return (1.0 - redundancy).clamp_min(0.0)


# ---------------------------------------------------------------------------
# Combined score → admission decision.
# ---------------------------------------------------------------------------

def grab_select(
    q_chunk: torch.Tensor,                       # [B, q_len, H, D]
    candidate_k: torch.Tensor,                   # [B, n_cand * F, H, D]
    candidate_frame_indices: torch.Tensor,       # [n_cand] long
    memory_size: int,
    frame_seqlen: int,
    alpha: float,
    score_temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor]:

    n_cand = candidate_frame_indices.numel()
    if n_cand <= memory_size:
        # Keep them all.
        keep = torch.ones(
            n_cand, dtype=torch.bool, device=candidate_frame_indices.device,
        )
        scores = torch.zeros(
            n_cand, dtype=torch.float32, device=candidate_frame_indices.device,
        )
        return keep, scores

    s_attn = attention_mass_score(
        q_chunk, candidate_k, frame_seqlen, temperature=score_temperature,
    )                                               # [B, n_cand]
    s_div = temporal_diversity_score(
        candidate_frame_indices, s_attn,
    )                                               # [B, n_cand]
    score = s_attn + alpha * s_div                  # [B, n_cand]

    # Reduce across batch (Self-Forcing inference is batch=1).
    score_1d = score.mean(dim=0)                    # [n_cand]
    topk = torch.topk(score_1d, memory_size, largest=True).indices
    keep = torch.zeros(
        n_cand, dtype=torch.bool, device=score_1d.device,
    )
    keep[topk] = True
    return keep, score_1d
