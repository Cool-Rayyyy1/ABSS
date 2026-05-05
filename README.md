# ABSS

Flux: pick seeds from attention on core tokens, then generate **topK_attn** vs **topK_random** images.

## Run

From this folder (with Flux weights and deps installed):

```bash
python abss.py --start-idx 101 --end-idx 110 --base-seed 11
```

`--start-idx` / `--end-idx` are **required**. Default weights are **`black-forest-labs/FLUX.1-dev`** on Hugging Face (first run downloads; gated models may need `huggingface-cli login` / `HF_TOKEN`). Override with `--model-id /path/to/local` if you already have a snapshot.

Outputs go to `./<run-name>/test_<base-seed>/prompt_XXX/` (`topK_attn/`, `topK_random/`, scores JSON/CSV).

## Data

By default reads **`prompts_initno.json`** and **`initno_core_token_index_dict.json`** next to `abss.py` (word positions → entity → T5 indices). To refresh those files from the paper code repo:

```bash
cd /path/to/initno1
python scripts/export_abss_data.py
```

`prompts_drawbench.json` / `prompts_pick.json` are prompt text only for Drawbench and Pick-a-Pic dataset.

## Parameters

| Option | Default | What it does |
|--------|---------|----------------|
| `--model-id` | `black-forest-labs/FLUX.1-dev` | Flux checkpoint on Hugging Face or a local path. |
| `--torch-dtype` | `float16` | `float16`, `bfloat16`, or `float32`. |
| `--data-dir` | this directory | Where to find the JSON files. |
| `--prompts-json` | `prompts_initno.json` | Prompt id → text. |
| `--token-json` | `initno_core_token_index_dict.json` | Per-prompt word positions (e.g. `entity`) for scoring. |
| `--start-idx` / `--end-idx` | (required) | Inclusive prompt id range. |
| `--base-seed` | `42` | Fixes the global seed pool and random top-K picks. |
| `--seeds-per-prompt` | `50` | How many seeds to score per prompt. |
| `--top-k` | `1` | Keep this many best seeds for `topK_attn`. |
| `--gen-steps` | `50` | Denoising steps when saving images. |
| `--score-step` | `10` | Step index (0-based) where attention is read for scoring; scoring runs `score_step + 1` steps. |
| `--guidance-scale` | `7.5` | CFG scale. |
| `--height` / `--width` | `512` | Output resolution. |
| `--probe-block-name` | `transformer_blocks.12.attn` | Which attention block to hook for scoring. |
| `--run-name` | auto | Output folder name; if empty: `Flux_SmallPool<seeds>_top<k>_run`. |

Full list: `python abss.py --help`.
