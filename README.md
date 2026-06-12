# TetherCache: Stabilizing Autoregressive Long-Form Video Generation with Gated Recall and Trusted Alignment

[![Paper](https://img.shields.io/badge/ArXiv-Paper-brown)](https://arxiv.org/abs/2606.13035)
[![Demo](https://img.shields.io/badge/Project-Page-brightgreen)](https://my4f175.github.io/TetherCache/)
[![Code](https://img.shields.io/badge/GitHub-TetherCache-blue)](https://github.com/my4f175/TetherCache)

Yu Meng<sup>1</sup>, Xiangyang Luo<sup>1</sup>, Letian Li<sup>1</sup>, Wenyuan Jiang<sup>2</sup>, Chen Gao<sup>1</sup>, Xinlei Chen<sup>1</sup>, Yong Li<sup>1</sup> and Xiao-Ping Zhang<sup>1</sup>

<sup>1</sup>Tsinghua University, <sup>2</sup>D-INFK, ETH Zürich

---

## 1. Setup

```bash
conda create -n tethercache python=3.10 -y
conda activate tethercache

git clone https://github.com/my4f175/TetherCache.git TetherCache && cd TetherCache
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
python setup.py develop
```

---

## 2. Download checkpoints

```bash
# Wan2.1 T2V-1.3B base model (text encoder + VAE + transformer)
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
    --local-dir-use-symlinks False \
    --local-dir wan_models/Wan2.1-T2V-1.3B

# Self-Forcing distilled checkpoint
huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt \
    --local-dir .
```


---

## 3. Quickstart

### 3.1 Short (5s, 21 latents, no cache rolling)

```bash
python inference.py \
    --config_path configs/self_forcing_dmd.yaml \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --output_folder videos/short \
    --data_path prompts/MovieGenVideoBench_extended.txt \
    --use_ema
```

### 3.2 Long (with TetherCache)

Single inline prompt:

```bash
python inference_long.py \
    --config_path configs/self_forcing_dmd.yaml \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --output_folder videos/long \
    --prompt "A cat surfing on a wave at sunset, photorealistic." \
    --num_latents 240 \
    --cache_backend tethercache \
    --use_ema --low_memory
```

A file of prompts (one per line):

```bash
python inference_long.py \
    --config_path configs/self_forcing_dmd.yaml \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --output_folder videos/long \
    --prompt_path prompts/short.txt \
    --extended_prompt_path prompts/extended.txt \
    --num_latents 240 \
    --cache_backend tethercache \
    --use_ema --low_memory
```

Multi-GPU (8x):

```bash
torchrun --nproc_per_node=8 inference_long.py \
    --config_path configs/self_forcing_dmd.yaml \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --output_folder videos/long \
    --prompt_path prompts/short.txt \
    --extended_prompt_path prompts/extended.txt \
    --num_latents 240 \
    --cache_backend tethercache \
    --use_ema --low_memory
```

---

## 4. CLI reference (long-form path)

| Flag | Default | Meaning |
|------|---------|---------|
| `--cache_backend` | `baseline` | `baseline` = FIFO eviction (no TetherCache); `tethercache` = enable GRAB + TAME. |
| `--local_attn_size` | `21` | Total cache budget K. Must equal the model's local attention window. |
| `--sink_size` | `3` | S — frozen anchor frames at the cache head. |
| `--recent_size` | `4` | R — FIFO recent frames at the cache tail. |
| `--memory_size` | `K - S - R` | M — GRAB-managed slots. Defaults to whatever's left of K after S and R. |
| `--alpha` | `0.35` | GRAB diversity weight: `score = attention_mass + α · temporal_diversity`. |
| `--tau` | `0.35` | TAME blend strength: `x' = (1 − τ) · x + τ · adain(x; trusted)`. |
| `--score_temperature` | `1.0` | Softmax temperature on GRAB attention-mass logits. |
| `--use_ema` | off | Load `generator_ema` from the checkpoint instead of `generator`. |
| `--low_memory` | off | Chunked VAE decode + dynamic swap. Recommended for `--num_latents > 480`. |
| `--num_samples` | `1` | Samples per prompt (different seeds). |
| `--seed` | `0` | Base RNG seed. |

---

## 5. Programmatic usage

```python
from omegaconf import OmegaConf
from pipeline import CausalInferencePipeline
from tethercache import (
    TetherCacheConfig,
    install_tethercache,
    reset_tethercache_state,
    force_local_attn,
)

cfg = OmegaConf.merge(
    OmegaConf.load("configs/default_config.yaml"),
    OmegaConf.load("configs/self_forcing_dmd.yaml"),
)
pipeline = CausalInferencePipeline(cfg, device="cuda")
# … load checkpoint, .to(bfloat16) …

tc_cfg = TetherCacheConfig(
    budget_size=21, sink_size=3, recent_size=4,   # K = S + M + R, M = 14
    alpha=0.35,
    tau=0.35,
)
force_local_attn(pipeline, tc_cfg.budget_size, tc_cfg.sink_size)
install_tethercache(pipeline, tc_cfg)

# Per video:
reset_tethercache_state(pipeline)
video, _ = pipeline.inference(noise=..., text_prompts=[...], return_latents=True)
```

## Acknowledgements

This repository is built upon several excellent open-source projects. We sincerely thank the contributors of [Self-Forcing](https://github.com/guandeh17/Self-Forcing), [MemRoPE](https://github.com/YoungRaeKimm/MemRoPE) and [DeepForcing](https://github.com/cvlab-kaist/DeepForcing) for their inspiring work and open-source implementation.