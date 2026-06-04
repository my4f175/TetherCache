from __future__ import annotations

from typing import Optional

import torch


class TetherCacheState:

    __slots__ = (
        "layer_idx",
        "num_heads",
        "head_dim",
        # --- Region geometry (in latent frames). Tokens per frame is
        # ``frame_seqlen``, set by the attention layer at runtime.
        "sink_size",
        "memory_size",
        "recent_size",
        # --- Lifecycle.
        "cache_filled",          # True once the cache first reaches K frames.
        # --- Memory occupancy bookkeeping.
        "mem_occupancy",         # int in [0, memory_size]
        "mem_global_frame_index",  # LongTensor [memory_size], -1 = unused
        # --- Pending eviction snapshot.
        # Captured on the first noisy pass of a chunk (when ``recent``
        # rolls); consumed by GRAB on the context pass. Each is the raw
        # K/V (unrotated) that fell off the front of ``recent``; their
        # global frame indices are stored too.
        "pending_evicted_k",
        "pending_evicted_v",
        "pending_evicted_global_lo",
        "pending_evicted_chunk_frames",
        # --- Diagnostics.
        "n_admissions",
        "n_rejections",
        "n_edits",
    )

    def __init__(
        self,
        layer_idx: int,
        num_heads: int,
        head_dim: int,
        sink_size: int,
        memory_size: int,
        recent_size: int,
    ):
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.sink_size = sink_size
        self.memory_size = memory_size
        self.recent_size = recent_size

        self.cache_filled = False
        self.mem_occupancy = 0
        self.mem_global_frame_index: Optional[torch.Tensor] = None

        self.pending_evicted_k: Optional[torch.Tensor] = None
        self.pending_evicted_v: Optional[torch.Tensor] = None
        self.pending_evicted_global_lo: int = -1
        self.pending_evicted_chunk_frames: int = 0

        self.n_admissions = 0
        self.n_rejections = 0
        self.n_edits = 0

    # ------------------------------------------------------------------
    # Lazy device-aware allocation for the per-slot frame-index buffer.
    # ------------------------------------------------------------------
    def _ensure_index_buffer(self, device, dtype=torch.long) -> None:
        if self.memory_size <= 0:
            return
        if (self.mem_global_frame_index is None
                or self.mem_global_frame_index.device != device):
            self.mem_global_frame_index = torch.full(
                (self.memory_size,), -1, dtype=dtype, device=device,
            )

    # ------------------------------------------------------------------
    # Lifecycle.
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.cache_filled = False
        self.mem_occupancy = 0
        if self.mem_global_frame_index is not None:
            self.mem_global_frame_index.fill_(-1)
        self.pending_evicted_k = None
        self.pending_evicted_v = None
        self.pending_evicted_global_lo = -1
        self.pending_evicted_chunk_frames = 0
        self.n_admissions = 0
        self.n_rejections = 0
        self.n_edits = 0

    def memory_filled_slots(self) -> int:
        return self.mem_occupancy

    def has_memory_slots_free(self) -> bool:
        return self.mem_occupancy < self.memory_size

    def __repr__(self) -> str:
        return (
            f"TetherCacheState(layer={self.layer_idx}, "
            f"S={self.sink_size}, M={self.memory_size}, R={self.recent_size}, "
            f"filled={self.cache_filled}, "
            f"mem_used={self.mem_occupancy}/{self.memory_size}, "
            f"admits={self.n_admissions}, rejects={self.n_rejections}, "
            f"edits={self.n_edits})"
        )
