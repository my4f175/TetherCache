"""Configuration for TetherCache.

A fixed cache budget K is split into three
contiguous regions inside the underlying KV tensor:

    [ S  sink ][ M  memory ][ R  recent ]
    |<------- K = S + M + R ----------->|

* ``sink`` (S) — the first S latent frames; frozen anchors that never evict.
* ``memory`` (M) — slots managed by **GRAB** (Gated Recall with
  Attention-Diversity Balancing). Each chunk, the about-to-evict frame from
  ``recent`` competes against the current memory members on a combined score
  of *Attention Mass* + ``alpha`` * *Temporal Diversity*; the top-M survive.
* ``recent`` (R) — the most recent R latent frames, FIFO.

When a frame is admitted into memory, **TAME** (Trusted Alignment via Memory
Editing) blends it toward the per-(head, channel) (μ, σ) of the
trusted pool via AdaIN with strength ``tau``::

    x' = (1 - tau) * x + tau * adain(x; trusted)

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TetherCacheConfig:

    # --- Cache geometry ----------------------------------------------------
    budget_size: int = 21
    sink_size: int = 3
    recent_size: int = 4
    memory_size: Optional[int] = None

    # --- GRAB --------------------------------------------------------------
    alpha: float = 0.35
    score_temperature: float = 1.0

    # --- TAME --------------------------------------------------------------
    tau: float = 0.35

    # --- Misc --------------------------------------------------------------
    verbose: bool = False

    def resolved_memory_size(self) -> int:
        """Return M. If ``memory_size`` is None, compute it from K - S - R."""
        if self.memory_size is not None:
            return int(self.memory_size)
        return int(self.budget_size - self.sink_size - self.recent_size)

    def __post_init__(self):
        if self.sink_size < 0:
            raise ValueError(
                f"sink_size must be >= 0, got {self.sink_size}"
            )
        if self.recent_size < 1:
            raise ValueError(
                f"recent_size must be >= 1 (need somewhere to land new "
                f"frames), got {self.recent_size}"
            )
        if self.budget_size <= 0:
            raise ValueError(
                f"budget_size must be > 0, got {self.budget_size}"
            )
        m = self.resolved_memory_size()
        if m < 0:
            raise ValueError(
                f"memory_size = budget_size - sink_size - recent_size "
                f"= {m} < 0; adjust sink_size/recent_size or set "
                f"memory_size explicitly."
            )
        if self.budget_size != self.sink_size + m + self.recent_size:
            raise ValueError(
                f"budget_size ({self.budget_size}) != "
                f"sink_size ({self.sink_size}) + "
                f"memory_size ({m}) + "
                f"recent_size ({self.recent_size})"
            )
        if not (0.0 <= self.tau <= 1.0):
            raise ValueError(
                f"tau must be in [0, 1], got {self.tau}"
            )
        if self.alpha < 0.0:
            raise ValueError(
                f"alpha must be >= 0, got {self.alpha}"
            )
        if self.score_temperature <= 0.0:
            raise ValueError(
                f"score_temperature must be > 0, got "
                f"{self.score_temperature}"
            )
