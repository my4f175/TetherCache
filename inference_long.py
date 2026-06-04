import argparse
import os
import time

import torch
import torch.distributed as dist
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.io import write_video
from tqdm import tqdm

if "LOCAL_RANK" in os.environ:
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

from pipeline import CausalInferencePipeline  # noqa: E402
from utils.dataset import TextDataset  # noqa: E402
from utils.misc import set_seed  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument(
        "--prompt_path", type=str, default=None,
        help="File of short prompts (one per line). Mutually exclusive "
             "with --prompt.",
    )
    parser.add_argument(
        "--extended_prompt_path", type=str, default=None,
        help="File of LLM-extended prompts aligned with --prompt_path. "
             "If omitted but --prompt_path is given, falls back to using "
             "the short prompts verbatim. Mutually exclusive with --prompt.",
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Inline prompt text (single video). The text encoder will see "
             "--extended_prompt if given, otherwise it sees this same "
             "string. Output filename uses the first 80 chars of this string.",
    )
    parser.add_argument(
        "--extended_prompt", type=str, default=None,
        help="Optional LLM-extended version of --prompt to feed the text "
             "encoder. Defaults to --prompt.",
    )
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument(
        "--num_latents", type=int, default=120,
        help="Number of latent frames to generate (120 ≈ 30s @ 16 fps).",
    )
    parser.add_argument(
        "--local_attn_size", type=int, default=21,
        help="Sliding KV-cache window size in latent frames "
             "(== TetherCache budget K).",
    )
    parser.add_argument(
        "--sink_size", type=int, default=3,
        help="Number of sink frames retained in the cache budget (S).",
    )
    parser.add_argument(
        "--recent_size", type=int, default=4,
        help="Number of recent frames always retained at the cache tail (R).",
    )
    parser.add_argument(
        "--memory_size", type=int, default=None,
        help="Explicit memory-cache size (M). Defaults to "
             "local_attn_size - sink_size - recent_size.",
    )
    parser.add_argument(
        "--cache_backend", choices=["baseline", "tethercache"],
        default="baseline",
        help="Use the baseline FIFO eviction or the TetherCache "
             "Sink + Memory + Recent patch (GRAB admission + TAME edit).",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.35,
        help="GRAB diversity weight: score = attention_mass + alpha * "
             "temporal_diversity. 0.0 disables diversity.",
    )
    parser.add_argument(
        "--tau", type=float, default=0.35,
        help="TAME blend strength: x' = (1 - tau) * x + tau * "
             "adain(x; trusted). 0.0 disables editing.",
    )
    parser.add_argument(
        "--score_temperature", type=float, default=1.0,
        help="Softmax temperature applied to GRAB attention-mass logits "
             "across candidates. Higher = flatter.",
    )
    parser.add_argument("--tether_verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument(
        "--low_memory", action="store_true",
        help="Enable chunked VAE decode + dynamic swap. Strongly "
             "recommended for --num_latents > 480 (otherwise the full RGB "
             "tensor for one video can pin >20 GB and OOM on the next "
             "prompt).",
    )
    return parser.parse_args()


def force_local_attn(pipeline, win, sink):
    """Configure each block's local attention window. Same helper as
    ``tethercache.install.force_local_attn`` — duplicated here so it works
    for the baseline FIFO path too (no TetherCache install required).
    """
    model = pipeline.generator.model
    frame_seq_length = pipeline.frame_seq_length
    for blk in model.blocks:
        blk.self_attn.local_attn_size = win
        blk.self_attn.sink_size = sink
        blk.self_attn.max_attention_size = win * frame_seq_length
    model.local_attn_size = win
    pipeline.local_attn_size = win


class _InlinePromptDataset(torch.utils.data.Dataset):
    """Tiny stand-in for utils.dataset.TextDataset when the user passes a
    single inline prompt via --prompt. Mirrors the field names emitted by
    TextDataset so the rest of the loop stays unchanged.
    """

    def __init__(self, short_prompt: str, extended_prompt: str):
        self.short = short_prompt
        self.ext = extended_prompt

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {
            "idx": idx,
            "prompts": self.short,
            "extended_prompts": self.ext,
        }


def _build_dataset(args):
    if args.prompt is not None:
        if args.prompt_path is not None or args.extended_prompt_path is not None:
            raise SystemExit(
                "[inference_long] --prompt is mutually exclusive with "
                "--prompt_path / --extended_prompt_path."
            )
        ext = args.extended_prompt or args.prompt
        return _InlinePromptDataset(args.prompt[:80], ext)
    if args.prompt_path is None:
        raise SystemExit(
            "[inference_long] either --prompt or --prompt_path is required."
        )
    ext_path = args.extended_prompt_path or args.prompt_path
    return TextDataset(prompt_path=args.prompt_path,
                       extended_prompt_path=ext_path)


def maybe_install_tethercache(args, pipeline):
    """Return a per-video reset closure if TetherCache is requested,
    otherwise None.
    """
    if args.cache_backend != "tethercache":
        return None
    from tethercache import (  # noqa: E402
        TetherCacheConfig,
        install_tethercache,
        reset_tethercache_state,
    )

    cfg = TetherCacheConfig(
        budget_size=args.local_attn_size,
        sink_size=args.sink_size,
        recent_size=args.recent_size,
        memory_size=args.memory_size,
        alpha=args.alpha,
        tau=args.tau,
        score_temperature=args.score_temperature,
        verbose=args.tether_verbose,
    )
    install_tethercache(pipeline, cfg)
    return reset_tethercache_state


def main():
    args = parse_args()

    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        set_seed(args.seed + local_rank)
    else:
        device = torch.device("cuda")
        local_rank = 0
        set_seed(args.seed)

    torch.set_grad_enabled(False)

    cfg = OmegaConf.merge(
        OmegaConf.load("configs/default_config.yaml"),
        OmegaConf.load(args.config_path),
    )

    pipeline = CausalInferencePipeline(cfg, device=device)
    state = torch.load(
        args.checkpoint_path, map_location="cpu", weights_only=False
    )
    pipeline.generator.load_state_dict(
        state["generator_ema" if args.use_ema else "generator"]
    )

    force_local_attn(pipeline, args.local_attn_size, args.sink_size)
    reset_cache_state = maybe_install_tethercache(args, pipeline)

    pipeline = pipeline.to(dtype=torch.bfloat16)
    pipeline.text_encoder.to(device)
    pipeline.generator.to(device)
    pipeline.vae.to(device)

    dataset = _build_dataset(args)
    if dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
    else:
        sampler = SequentialSampler(dataset)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0)

    if local_rank == 0:
        os.makedirs(args.output_folder, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    for batch in tqdm(loader, disable=(local_rank != 0)):
        if isinstance(batch, list):
            batch = batch[0]
        idx = batch["idx"].item()
        if idx >= len(dataset):
            continue
        short_prompt = batch["prompts"][0]
        ext_prompt = batch["extended_prompts"][0]
        if reset_cache_state is not None:
            reset_cache_state(pipeline)
        for s in range(args.num_samples):
            torch.manual_seed(args.seed + idx * 100 + s)
            noise = torch.randn(
                [1, args.num_latents, 16, 60, 104],
                device=device, dtype=torch.bfloat16,
            )
            t0 = time.time()
            video, _ = pipeline.inference(
                noise=noise,
                text_prompts=[ext_prompt],
                return_latents=True,
                low_memory=args.low_memory,
            )
            pipeline.vae.model.clear_cache()
            v = rearrange(video, "b t c h w -> b t h w c").cpu()
            v = (255.0 * v).clamp(0, 255).to(torch.uint8)
            base = short_prompt.replace("/", "_")
            out_path = os.path.join(args.output_folder, f"{base}-{s}.mp4")
            write_video(out_path, v[0], fps=16)
            if local_rank == 0:
                print(
                    f"[rank0] idx={idx} t={time.time()-t0:.1f}s -> {out_path}",
                    flush=True,
                )

            del video, v, noise
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
