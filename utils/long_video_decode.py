"""Helpers for chunked VAE decode + CPU offload, used by long-video runs that
exceed a single GPU's memory budget.  All baselines / Self-Forcing share the
same WAN VAE; this monkey-patch is applied uniformly inside each wrapper.

WAN VAE ``cached_decode(z, scale)`` keeps an internal feat-map cache between
calls, so we can decode the latent in chunks of ``chunk_size`` frames and
get the *same* output as a full-tensor ``decode(...)`` call as long as we
clear the cache once at the start of each video.
"""
import torch


def install_chunked_decode(vae_wrapper, chunk_size: int = 30, return_to_cpu: bool = True):
    """Replace ``vae_wrapper.decode_to_pixel`` with a chunked version.

    Behavior:
      * On entry: clear the WAN VAE feat-map cache.
      * For each chunk of ``chunk_size`` latent frames, call the underlying
        ``cached_decode`` (which does NOT clear cache) and immediately move the
        decoded pixel chunk to CPU.
      * Concatenate chunks on CPU and return.

    The output is bit-identical to the original full-tensor ``decode`` for
    chunk_size >= num_latent_frames, and approximately identical (within
    numerical noise from chunked execution) for smaller chunk sizes.
    """
    inner_vae = vae_wrapper  # WanVAEWrapper
    base_model = inner_vae.model  # the actual VAE module with decode/cached_decode

    def chunked_decode_to_pixel(latent: torch.Tensor, use_cache: bool = False, **kwargs):
        device, dtype = latent.device, latent.dtype
        scale = [
            inner_vae.mean.to(device=device, dtype=dtype),
            1.0 / inner_vae.std.to(device=device, dtype=dtype),
        ]
        # latent shape: [B, T, C, H, W]; cached_decode wants [B, C, T, H, W]
        zs = latent.permute(0, 2, 1, 3, 4)

        out_chunks_cpu = []
        for u in zs:
            base_model.clear_cache()
            sub_outs = []
            T = u.shape[1]
            for s in range(0, T, chunk_size):
                e = min(s + chunk_size, T)
                z = u[:, s:e].unsqueeze(0).contiguous()
                piece = base_model.cached_decode(z, scale).float().clamp_(-1, 1).squeeze(0)
                if return_to_cpu:
                    sub_outs.append(piece.cpu())
                    del piece
                    torch.cuda.empty_cache()
                else:
                    sub_outs.append(piece)
            out_chunks_cpu.append(torch.cat(sub_outs, dim=1))
            base_model.clear_cache()

        # back to [B, T, C, H, W]
        out = torch.stack(out_chunks_cpu, dim=0).permute(0, 2, 1, 3, 4)
        return out

    vae_wrapper.decode_to_pixel = chunked_decode_to_pixel
    return vae_wrapper
