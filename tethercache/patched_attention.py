"""Patched ``CausalWanSelfAttention.forward`` implementing TetherCache.

Cache layout (frames; tokens = frames * frame_seqlen)::

    [ sink (S) | memory (M) | recent (R) ]
    0          S            S+M          K = S + M + R

The KV cache tensor has the same shape the baseline model expects —
``[B, K * frame_seqlen, H, D]``; TetherCache only re-interprets its slot
ranges. Storage is **unrotated K** (V is never rotated); RoPE is applied
on read with block-relative indices, matching the
MemRoPE / Infinity-RoPE recipe.

Pass ordering for one chunk
---------------------------

Self-Forcing's pipeline calls the generator several times per chunk: one
pass per denoising step plus one final ``context`` pass at timestep
``context_noise``. We split the per-call work into four phases:

A. **Layout mutation** — only on the chunk's first call (``current_end >
   global_end_index``). In WARMUP we append the chunk into the live
   region's tail; in STEADY we capture ``recent``'s head as a pending
   eviction, roll ``recent`` left, and write the chunk into ``recent``'s
   tail.
B. **Slot overwrite** — every call writes the current pass's K/V into the
   chunk's slot, overwriting the previous (noisier) pass.
C. **Post-write GRAB / TAME** — only on the context pass (clean K/V):

   * If we just hit the cache-fill boundary, mark the cache as filled and
     seed the M memory slots with the M frames currently sitting in
     ``[S, S + M)``.
   * If a pending eviction is queued, run :func:`grab_select` against the
     candidate set ``[existing memory members] + [evicted chunk frames]``,
     apply :func:`adain_toward_trusted` to admitted-from-evicted slices,
     and write the new memory layout.

D. **Attention readout** — apply RoPE to cached K with block-relative
   indices, run attention, project out.
"""
from __future__ import annotations

import math

import torch
from torch.nn.attention.flex_attention import flex_attention

from wan.modules.attention import attention
from wan.modules.causal_model import causal_rope_apply, rope_apply

from .config import TetherCacheConfig
from .grab import grab_select
from .state import TetherCacheState
from .tame import adain_toward_trusted


def patched_self_attn_forward(
    self,
    x,
    seq_lens,
    grid_sizes,
    freqs,
    block_mask,
    kv_cache=None,
    current_start=0,
    cache_start=None,
):
    """Drop-in replacement for ``CausalWanSelfAttention.forward``."""
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    if cache_start is None:
        cache_start = current_start

    cfg: TetherCacheConfig = self.tether_cfg
    st: TetherCacheState = self.tether_state
    g_state = self.tether_global

    # ----- qkv --------------------------------------------------------------
    def qkv_fn(x_in):
        q_ = self.norm_q(self.q(x_in)).view(b, s, n, d)
        k_ = self.norm_k(self.k(x_in)).view(b, s, n, d)
        v_ = self.v(x_in).view(b, s, n, d)
        return q_, k_, v_

    q, k, v = qkv_fn(x)

    # ----- teacher-forcing branch (unchanged from baseline) -----------------
    if kv_cache is None:
        is_tf = (s == seq_lens[0].item() * 2)
        if is_tf:
            q_chunks = torch.chunk(q, 2, dim=1)
            k_chunks = torch.chunk(k, 2, dim=1)
            roped_query, roped_key = [], []
            for ii in range(2):
                roped_query.append(
                    rope_apply(q_chunks[ii], grid_sizes, freqs).type_as(v)
                )
                roped_key.append(
                    rope_apply(k_chunks[ii], grid_sizes, freqs).type_as(v)
                )
            roped_query = torch.cat(roped_query, dim=1)
            roped_key = torch.cat(roped_key, dim=1)
        else:
            roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
            roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

        padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
        zeros_q = torch.zeros(
            [q.shape[0], padded_length, q.shape[2], q.shape[3]],
            device=q.device, dtype=v.dtype,
        )
        zeros_k = torch.zeros(
            [k.shape[0], padded_length, k.shape[2], k.shape[3]],
            device=k.device, dtype=v.dtype,
        )
        zeros_v = torch.zeros(
            [v.shape[0], padded_length, v.shape[2], v.shape[3]],
            device=v.device, dtype=v.dtype,
        )
        x = flex_attention(
            query=torch.cat([roped_query, zeros_q], dim=1).transpose(2, 1),
            key=torch.cat([roped_key, zeros_k], dim=1).transpose(2, 1),
            value=torch.cat([v, zeros_v], dim=1).transpose(2, 1),
            block_mask=block_mask,
        )[:, :, :-padded_length].transpose(2, 1)
        x = x.flatten(2)
        x = self.o(x)
        return x

    # ====== AR-inference branch (kv_cache is not None) ======================
    is_context_pass = bool(getattr(g_state, "is_context_pass", False))

    frame_seqlen = math.prod(grid_sizes[0][1:]).item()
    current_end = current_start + s
    chunk_frames = s // frame_seqlen

    sink_frames = st.sink_size
    memory_frames = st.memory_size
    recent_frames = st.recent_size
    K_frames = sink_frames + memory_frames + recent_frames

    sink_tokens = sink_frames * frame_seqlen
    memory_tokens = memory_frames * frame_seqlen
    recent_tokens = recent_frames * frame_seqlen

    kv_cache_size = kv_cache["k"].shape[1]
    expected_cache_tokens = K_frames * frame_seqlen
    if kv_cache_size != expected_cache_tokens:
        raise RuntimeError(
            f"[TetherCache] layer {st.layer_idx}: cache tensor has "
            f"{kv_cache_size} tokens but config implies "
            f"{expected_cache_tokens} (K={K_frames}, "
            f"frame_seqlen={frame_seqlen}). Did you forget to call "
            f"force_local_attn(pipeline, K, S) before install_tethercache?"
        )

    # K is stored unrotated; RoPE is applied on read.
    k_to_store = k

    cur_local_end_before = kv_cache["local_end_index"].item()
    cur_global_end_before = kv_cache["global_end_index"].item()
    is_first_write_of_chunk = current_end > cur_global_end_before

    # ----- Phase A: layout mutation (only on first write) ------------------
    if is_first_write_of_chunk:
        if cur_local_end_before + s <= kv_cache_size:
            # WARMUP: append into the live region's tail.
            local_start_index = cur_local_end_before
            local_end_index = local_start_index + s
        else:
            # STEADY: snapshot recent's head, roll recent left, prepare tail
            # to receive the new chunk.
            recent_start_tok = sink_tokens + memory_tokens
            evicted_k = kv_cache["k"][
                :, recent_start_tok:recent_start_tok + s
            ].clone()
            evicted_v = kv_cache["v"][
                :, recent_start_tok:recent_start_tok + s
            ].clone()
            evicted_global_lo = (
                cur_global_end_before // frame_seqlen
            ) - recent_frames

            st.pending_evicted_k = evicted_k
            st.pending_evicted_v = evicted_v
            st.pending_evicted_global_lo = int(evicted_global_lo)
            st.pending_evicted_chunk_frames = chunk_frames

            # recent[s:] <- recent[:-s]; tail freed for the new chunk.
            if recent_tokens - s > 0:
                kv_cache["k"][
                    :, recent_start_tok:recent_start_tok + recent_tokens - s
                ] = kv_cache["k"][
                    :, recent_start_tok + s:
                       recent_start_tok + recent_tokens
                ].clone()
                kv_cache["v"][
                    :, recent_start_tok:recent_start_tok + recent_tokens - s
                ] = kv_cache["v"][
                    :, recent_start_tok + s:
                       recent_start_tok + recent_tokens
                ].clone()

            local_start_index = recent_start_tok + recent_tokens - s
            local_end_index = kv_cache_size

        kv_cache["global_end_index"].fill_(current_end)
        kv_cache["local_end_index"].fill_(local_end_index)
    else:
        # Subsequent pass for the same chunk — slot lives at the tail of
        # the live region.
        local_end_index = kv_cache["local_end_index"].item()
        local_start_index = local_end_index - s

    # ----- Phase B: write current pass's K, V into the chunk's slot --------
    # Every pass overwrites the slot with its (possibly cleaner) K/V.
    kv_cache["k"][:, local_start_index:local_end_index] = k_to_store
    kv_cache["v"][:, local_start_index:local_end_index] = v

    # ----- Phase C: GRAB / TAME on the context pass only -------------------
    if is_context_pass:
        # C.1 — Cache-fill boundary: seed memory.
        if not st.cache_filled and local_end_index == kv_cache_size:
            st._ensure_index_buffer(kv_cache["k"].device)
            if memory_frames > 0:
                init_indices = torch.arange(
                    sink_frames,
                    sink_frames + memory_frames,
                    dtype=torch.long,
                    device=kv_cache["k"].device,
                )
                st.mem_global_frame_index[:memory_frames] = init_indices
                st.mem_occupancy = memory_frames
            st.cache_filled = True
            if cfg.verbose and st.layer_idx == 0:
                print(
                    f"[TetherCache] layer {st.layer_idx}: cache filled — "
                    f"mem_init={st.mem_occupancy}/{st.memory_size}",
                    flush=True,
                )

        # C.2 — Run admission on any pending eviction.
        if (st.pending_evicted_k is not None
                and st.cache_filled
                and memory_frames > 0):
            _run_grab_admission(
                self,
                kv_cache=kv_cache,
                st=st,
                cfg=cfg,
                q_chunk=q,
                evicted_k=st.pending_evicted_k,
                evicted_v=st.pending_evicted_v,
                evicted_global_frame_lo=st.pending_evicted_global_lo,
                chunk_frames=st.pending_evicted_chunk_frames,
                frame_seqlen=frame_seqlen,
                sink_tokens=sink_tokens,
                memory_tokens=memory_tokens,
            )
            st.pending_evicted_k = None
            st.pending_evicted_v = None
            st.pending_evicted_global_lo = -1
            st.pending_evicted_chunk_frames = 0

    # ----- Phase D: attention readout --------------------------------------
    cache_lo = max(0, local_end_index - self.max_attention_size)
    cache_hi = local_end_index
    cached_K = kv_cache["k"][:, cache_lo:cache_hi]
    cached_V = kv_cache["v"][:, cache_lo:cache_hi]

    cached_frames = (cache_hi - cache_lo) // frame_seqlen
    h_grid = grid_sizes[0][1].item()
    w_grid = grid_sizes[0][2].item()
    cache_grid = torch.tensor(
        [[cached_frames, h_grid, w_grid]],
        dtype=grid_sizes.dtype, device=grid_sizes.device,
    )
    cached_K = causal_rope_apply(
        cached_K, cache_grid, freqs, start_frame=0,
    ).type_as(v)
    # Q corresponds to the trailing ``chunk_frames`` frames in the cached
    # window.
    q_start_frame = cached_frames - chunk_frames
    roped_query = causal_rope_apply(
        q, grid_sizes, freqs, start_frame=q_start_frame,
    ).type_as(v)

    out = attention(roped_query, cached_K, cached_V)
    out = out.flatten(2)
    out = self.o(out)
    return out


# ---------------------------------------------------------------------------
# GRAB admission helper.
# ---------------------------------------------------------------------------

def _run_grab_admission(
    self,
    kv_cache,
    st: TetherCacheState,
    cfg: TetherCacheConfig,
    q_chunk: torch.Tensor,
    evicted_k: torch.Tensor,
    evicted_v: torch.Tensor,
    evicted_global_frame_lo: int,
    chunk_frames: int,
    frame_seqlen: int,
    sink_tokens: int,
    memory_tokens: int,
) -> None:
    """Run GRAB on ``[existing memory members] + [evicted chunk frames]``,
    apply TAME to admitted-from-evicted slices, and write the new memory
    layout in place.

    Mutates ``kv_cache`` and updates ``st``; returns nothing.
    """
    device = evicted_k.device
    st._ensure_index_buffer(device)

    M_max = st.memory_size
    if M_max <= 0:
        return

    mem_used = st.mem_occupancy
    if mem_used > 0:
        existing_k = kv_cache["k"][
            :, sink_tokens:sink_tokens + mem_used * frame_seqlen
        ]
        existing_idx = st.mem_global_frame_index[:mem_used].to(device)
    else:
        existing_k = evicted_k.new_zeros(
            evicted_k.shape[0], 0, evicted_k.shape[2], evicted_k.shape[3]
        )
        existing_idx = torch.zeros(0, dtype=torch.long, device=device)

    candidate_k = torch.cat([existing_k, evicted_k], dim=1)
    new_indices = torch.arange(
        evicted_global_frame_lo,
        evicted_global_frame_lo + chunk_frames,
        device=device, dtype=torch.long,
    )
    candidate_idx = torch.cat([existing_idx, new_indices], dim=0)

    keep_mask, _scores = grab_select(
        q_chunk=q_chunk,
        candidate_k=candidate_k,
        candidate_frame_indices=candidate_idx,
        memory_size=M_max,
        frame_seqlen=frame_seqlen,
        alpha=cfg.alpha,
        score_temperature=cfg.score_temperature,
    )

    selected_positions = torch.nonzero(
        keep_mask, as_tuple=False
    ).flatten().tolist()
    if not selected_positions:
        st.n_rejections += chunk_frames
        return

    # Build the trusted pool used by TAME. Snapshot BEFORE we start mutating
    # cache slices — TAME's target (μ, σ) should reflect the slots admission
    # has agreed to trust this far.
    trusted_k = trusted_v = None
    if cfg.tau > 0.0:
        trusted_k_chunks, trusted_v_chunks = [], []
        if sink_tokens > 0:
            trusted_k_chunks.append(kv_cache["k"][:, :sink_tokens].clone())
            trusted_v_chunks.append(kv_cache["v"][:, :sink_tokens].clone())
        if mem_used > 0:
            mem_hi = sink_tokens + mem_used * frame_seqlen
            trusted_k_chunks.append(
                kv_cache["k"][:, sink_tokens:mem_hi].clone()
            )
            trusted_v_chunks.append(
                kv_cache["v"][:, sink_tokens:mem_hi].clone()
            )
        if trusted_k_chunks:
            trusted_k = torch.cat(trusted_k_chunks, dim=1)
            trusted_v = torch.cat(trusted_v_chunks, dim=1)

    new_k_list, new_v_list, new_indices_list = [], [], []
    n_admitted_from_evicted = 0
    n_kept_from_existing = 0
    for cand_pos in selected_positions:
        if cand_pos < mem_used:
            # Re-selected existing memory member — copy its slot in place.
            tok_lo = sink_tokens + cand_pos * frame_seqlen
            tok_hi = tok_lo + frame_seqlen
            new_k_list.append(kv_cache["k"][:, tok_lo:tok_hi].clone())
            new_v_list.append(kv_cache["v"][:, tok_lo:tok_hi].clone())
            new_indices_list.append(int(candidate_idx[cand_pos].item()))
            n_kept_from_existing += 1
        else:
            # Newly admitted from the evicted chunk — apply TAME.
            ev_off = cand_pos - mem_used
            ev_k_slice = evicted_k[
                :, ev_off * frame_seqlen:(ev_off + 1) * frame_seqlen
            ]
            ev_v_slice = evicted_v[
                :, ev_off * frame_seqlen:(ev_off + 1) * frame_seqlen
            ]
            ev_k_slice, ek_edited = adain_toward_trusted(
                x=ev_k_slice, trusted=trusted_k, tau=cfg.tau,
            )
            ev_v_slice, ev_edited = adain_toward_trusted(
                x=ev_v_slice, trusted=trusted_v, tau=cfg.tau,
            )
            if ek_edited or ev_edited:
                st.n_edits += 1
            new_k_list.append(ev_k_slice)
            new_v_list.append(ev_v_slice)
            new_indices_list.append(int(candidate_idx[cand_pos].item()))
            n_admitted_from_evicted += 1

    n_new_mem = len(new_k_list)
    new_k = torch.cat(new_k_list, dim=1)
    new_v = torch.cat(new_v_list, dim=1)

    kv_cache["k"][
        :, sink_tokens:sink_tokens + n_new_mem * frame_seqlen
    ] = new_k
    kv_cache["v"][
        :, sink_tokens:sink_tokens + n_new_mem * frame_seqlen
    ] = new_v
    if n_new_mem < M_max:
        kv_cache["k"][
            :, sink_tokens + n_new_mem * frame_seqlen:
               sink_tokens + M_max * frame_seqlen
        ].zero_()
        kv_cache["v"][
            :, sink_tokens + n_new_mem * frame_seqlen:
               sink_tokens + M_max * frame_seqlen
        ].zero_()

    st.mem_occupancy = n_new_mem
    if n_new_mem > 0:
        st.mem_global_frame_index[:n_new_mem] = torch.tensor(
            new_indices_list, dtype=torch.long, device=device,
        )
    if n_new_mem < M_max:
        st.mem_global_frame_index[n_new_mem:] = -1

    st.n_admissions += n_admitted_from_evicted
    st.n_rejections += (chunk_frames - n_admitted_from_evicted)

    if cfg.verbose and st.layer_idx == 0:
        print(
            f"[TetherCache] layer {st.layer_idx}: admitted "
            f"{n_admitted_from_evicted}/{chunk_frames} evicted; kept "
            f"{n_kept_from_existing}/{mem_used} existing; "
            f"mem={st.mem_occupancy}/{M_max} "
            f"(admits={st.n_admissions}, edits={st.n_edits})",
            flush=True,
        )
