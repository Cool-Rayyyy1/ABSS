# Attention-Based Seed Selection (ABSS)

[![Paper: arXiv](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2605.19532)

## News 🚀️

Our paper ***ABSS*** has been accepted by NeurIPS 2026. 🎉🌸🎉

![Good and bad seeds across text-to-image models](assets/motivation.jpg)

## Overview

ABSS is a training-free method that ranks seeds using early attention to prompt core tokens and continues generation for the top-k seeds. This repository supports **FLUX.1-dev** and **HunyuanDiT v1.2**.

## Abstract

Text-to-image diffusion models can synthesize high-quality images, yet the outcome is notoriously sensitive to the random seed: different initial seeds often yield large variations in image quality and prompt–image alignment. We revisit this “seed effect” and show that attention dynamics over prompt core tokens, the content-bearing words, measured during the first few denoising steps, strongly predict final generation quality. Building on this observation, we introduce **Attention-Based Seed Selection (ABSS)**, a training-free, plug-and-play method that ranks seeds for a given prompt by leveraging cross-attention to core tokens during the denoising process. ABSS requires no finetuning and does not alter the initial noise; it scores and ranks all candidate seeds, keeps only the top-k for full generation, and discards the rest, without relying on a fixed accept/reject threshold. Operating purely at inference time, ABSS can serve as a lightweight pre-selection add-on for existing seed-optimization pipelines, enabling additional gains. Across three benchmarks, extensive experiments show that ABSS enables consistent improvements in text–image alignment and visual quality for Stable Diffusion variants, as corroborated by human preference and alignment metrics.

## Installation

Before running, download the [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) and [HunyuanDiT v1.2](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers) weights. Use `--model-path` to select a local checkpoint.

```bash
git clone https://github.com/Cool-Rayyyy1/ABSS.git
cd ABSS
conda create -n abss python=3.10 -y
conda activate abss
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Usage👀️

```bash
python run.py --model flux --dataset initno
python run.py --model hunyuan --dataset initno
```

Both commands generate **3 ABSS images and 3 random images** per prompt. ABSS screens 10 candidate seeds for **10 denoising steps**, selects the top 3, and resumes their saved states for the remaining **40 steps**.

| Argument | Default |
| --- | --- |
| `--dataset` | `initno` (`drawbench` and `pick` also supported) |
| `--seed-pool-size` | `10` |
| `--top-k` | `3` |
| `--screening-steps` | `10` |
| `--num-inference-steps` | `50` |
| `--guidance-scale` | `7.5` |
| `--base-seed` | `11` |

Use `--start-idx 101 --end-idx 101` for a single prompt, or `--no-random-baseline` for ABSS only. Prompts and core-token annotations are in [datasets/](datasets/); override them with `--prompts` and `--core-tokens`. See `python run.py --help` for all options.

### Output🎉️

Each run saves images and scoring manifests under `runs/<model>/<dataset>/test_<base-seed>/`:

```text
abss/prompt_101/img_seed*.png
random/prompt_101/img_seed*.png
metadata/prompt_101/
manifest.csv
manifest.jsonl
```

The manifests match each image to its prompt, seed, model, and method for later evaluation. Use `--output` for a new run directory. FLUX outputs 512×512 images; HunyuanDiT outputs 1024×1024 images by default.

For Slurm, use `sbatch scripts/run.slurm.sh --model flux --dataset initno`. Merge completed run manifests with `python scripts/collect.py --help`.

Upstream credits and licenses: [Third-party notices](THIRD_PARTY_NOTICES.md).
