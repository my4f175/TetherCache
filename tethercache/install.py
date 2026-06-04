from __future__ import annotations

import types
from typing import Iterable

from .config import TetherCacheConfig
from .patched_attention import patched_self_attn_forward
from .state import TetherCacheState


# ---------------------------------------------------------------------------
# Shared per-pipeline state.
# ---------------------------------------------------------------------------

class _TetherGlobalState:

    __slots__ = ("is_context_pass", "context_noise")

    def __init__(self, context_noise: int):
        self.is_context_pass = False
        self.context_noise = context_noise


def _iter_self_attn_modules(pipeline) -> Iterable:
    blocks = pipeline.generator.model.blocks
    for i, blk in enumerate(blocks):
        yield i, blk.self_attn


def _wrap_generator_forward(pipeline, g_state: _TetherGlobalState) -> None:
    """Wrap ``pipeline.generator.forward`` so each call updates the shared
    state based on its ``timestep`` argument. Idempotent.
    """
    if getattr(pipeline.generator, "_tether_forward_wrapped", False):
        # Update the state reference (in case install is called again).
        pipeline.generator._tether_global_state = g_state
        return

    original_forward = pipeline.generator.forward

    def wrapped(self, *args, **kwargs):
        ts = kwargs.get("timestep", None)
        if ts is None and len(args) >= 3:
            ts = args[2]
        if ts is not None:
            try:
                ts_scalar = int(ts.flatten()[0].item())
                self._tether_global_state.is_context_pass = (
                    ts_scalar == self._tether_global_state.context_noise
                )
            except Exception:
                self._tether_global_state.is_context_pass = False
        else:
            self._tether_global_state.is_context_pass = False
        return original_forward(*args, **kwargs)

    pipeline.generator._tether_global_state = g_state
    pipeline.generator._tether_original_forward = original_forward
    pipeline.generator.forward = types.MethodType(wrapped, pipeline.generator)
    pipeline.generator._tether_forward_wrapped = True


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------

def force_local_attn(pipeline, budget_size: int, sink_size: int) -> None:

    model = pipeline.generator.model
    frame_seq_length = pipeline.frame_seq_length
    for blk in model.blocks:
        blk.self_attn.local_attn_size = budget_size
        blk.self_attn.sink_size = sink_size
        blk.self_attn.max_attention_size = budget_size * frame_seq_length
    model.local_attn_size = budget_size
    pipeline.local_attn_size = budget_size


def install_tethercache(pipeline, config: TetherCacheConfig) -> None:

    context_noise = int(getattr(pipeline.args, "context_noise", 0))
    g_state = _TetherGlobalState(context_noise=context_noise)
    _wrap_generator_forward(pipeline, g_state)

    # Sanity: cache budget must equal the attention's local_attn_size.
    first_attn = next(iter(_iter_self_attn_modules(pipeline)))[1]
    local_attn_size = int(first_attn.local_attn_size)
    if local_attn_size != config.budget_size:
        # Accept and align: the user almost certainly meant for budget to
        # equal the layer's window size.
        config_resolved_M = config.resolved_memory_size()
        if local_attn_size != (config.sink_size
                               + config_resolved_M
                               + config.recent_size):
            raise ValueError(
                f"[TetherCache] local_attn_size ({local_attn_size}) does "
                f"not equal budget_size ({config.budget_size}) and "
                f"S+M+R "
                f"({config.sink_size}+{config_resolved_M}+{config.recent_size}) "
                f"does not match either. Call force_local_attn(...) or "
                f"adjust TetherCacheConfig."
            )
        config.budget_size = local_attn_size

    M_resolved = config.resolved_memory_size()
    for layer_idx, attn in _iter_self_attn_modules(pipeline):
        # Each block's attention sees a consistent sink_size.
        attn.sink_size = int(config.sink_size)

        if getattr(attn, "_tether_patched", False):
            # Re-installing — refresh state but keep the patched .forward.
            attn.tether_cfg = config
            attn.tether_state = TetherCacheState(
                layer_idx=layer_idx,
                num_heads=attn.num_heads,
                head_dim=attn.head_dim,
                sink_size=config.sink_size,
                memory_size=M_resolved,
                recent_size=config.recent_size,
            )
            attn.tether_global = g_state
            continue

        attn.tether_cfg = config
        attn.tether_state = TetherCacheState(
            layer_idx=layer_idx,
            num_heads=attn.num_heads,
            head_dim=attn.head_dim,
            sink_size=config.sink_size,
            memory_size=M_resolved,
            recent_size=config.recent_size,
        )
        attn.tether_global = g_state
        attn._tether_original_forward = attn.forward
        attn.forward = types.MethodType(patched_self_attn_forward, attn)
        attn._tether_patched = True

    if config.verbose:
        n_layers = sum(1 for _ in _iter_self_attn_modules(pipeline))
        print(
            f"[TetherCache] installed on {n_layers} blocks; "
            f"K={config.budget_size} = "
            f"S({config.sink_size}) + M({M_resolved}) + R({config.recent_size}); "
            f"alpha={config.alpha} tau={config.tau} "
            f"score_temperature={config.score_temperature} "
            f"context_noise={context_noise}",
            flush=True,
        )


def reset_tethercache_state(pipeline) -> None:
    """Reset every block's :class:`TetherCacheState` (call between videos)."""
    for _, attn in _iter_self_attn_modules(pipeline):
        if getattr(attn, "_tether_patched", False):
            attn.tether_state.reset()
