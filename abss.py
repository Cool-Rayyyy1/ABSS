#!/usr/bin/env python3
"""
InitNO two-stage run (seed scoring + Attend-and-Excite images) from ABSS data:
  - prompts_initno.json
  - initno_core_token_index_dict.json

See README.md in this folder for setup and commands.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ABSS_DIR = Path(__file__).resolve().parent
if str(ABSS_DIR) not in sys.path:
    sys.path.insert(0, str(ABSS_DIR))

from abss_token_utils import debug_print_prompt_alignment, map_token_groups_bosless_to_77


def _load_int_dict(path: Path) -> dict[int, dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in raw.items()}


def _load_prompts(path: Path) -> dict[int, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): str(v) for k, v in raw.items()}


def process_seed_dataframe_entity_only(df, top_k, root_dir, p_idx):
    df_sorted = df.sort_values("entity_mean", ascending=False)
    top_seeds = df_sorted["seed"].head(top_k).tolist()
    prompt_dir = os.path.join(root_dir, f"prompt_{p_idx:03d}")
    os.makedirs(prompt_dir, exist_ok=True)
    with open(os.path.join(prompt_dir, "top50_entity_seeds.json"), "w", encoding="utf-8") as f:
        json.dump(top_seeds, f, indent=2)
    return top_seeds


def _flatten_token_groups(token_idx_mapped: dict) -> list[int]:
    all_token_indices = []
    for v in token_idx_mapped.values():
        if isinstance(v, list):
            all_token_indices.extend(v)
    seen: set[int] = set()
    out = []
    for x in all_token_indices:
        if isinstance(x, int) and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run InitNO attention experiment using prompts_initno.json + initno_core_token_index_dict.json from ABSS."
    )
    parser.add_argument(
        "--initno_repo",
        default=os.environ.get("INITNO_REPO", str(ABSS_DIR.parent / "initno1")),
        help="Root of the InitNO code repo (must contain the `initno` package). Default: ../initno1 or INITNO_REPO.",
    )
    parser.add_argument(
        "--data_dir",
        default=str(ABSS_DIR),
        help="Directory that contains prompts_initno.json and initno_core_token_index_dict.json (default: directory of this script).",
    )
    parser.add_argument(
        "--sd_path",
        default="",
        help="Path to Stable Diffusion v1.4 weights (local folder). Default: {initno_repo}/stable-diffusion-v1-4",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=11,
        help="Controls reproducibility of seed pools (same as original run_attention9_initno convention).",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        required=True,
        help="First prompt ID to include (must exist in prompts_initno.json).",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        required=True,
        help="Last prompt ID to include (inclusive).",
    )
    parser.add_argument("--debug", action="store_true", help="Print BOS-less vs 77-token alignment for some prompts.")
    parser.add_argument(
        "--debug_every",
        type=int,
        default=10,
        help="With --debug, print alignment every N prompts in the selected range (first prompt is always eligible when N divides).",
    )
    args = parser.parse_args()

    initno_root = Path(args.initno_repo).resolve()
    if str(initno_root) not in sys.path:
        sys.path.insert(0, str(initno_root))

    from initno.pipelines.pipeline_sd import StableDiffusionAttendAndExcitePipeline
    from initno.pipelines.pipeline_sd_initno_c5 import StableDiffusionInitNOPipeline

    sd_path = Path(args.sd_path).resolve() if args.sd_path else (initno_root / "stable-diffusion-v1-4").resolve()
    data_dir = Path(args.data_dir).resolve()

    prompts_path = data_dir / "prompts_initno.json"
    token_path = data_dir / "initno_core_token_index_dict.json"

    if not prompts_path.is_file():
        raise FileNotFoundError(f"Missing {prompts_path}")
    if not token_path.is_file():
        raise FileNotFoundError(f"Missing {token_path}")

    prompts_dict = _load_prompts(prompts_path)
    prompt_token_index_dict = _load_int_dict(token_path)

    guidance_scale_1 = 7.5
    num_infer_steps = 50
    seeds_per_prompt = 10
    top_k = 3
    root_dir = os.path.join(os.getcwd(), f"SmallPool10_top3/abss/test{args.base_seed}")
    os.makedirs(root_dir, exist_ok=True)

    all_pidx = sorted(prompts_dict.keys())
    selected_pidx = [i for i in all_pidx if args.start_idx <= i <= args.end_idx]
    if not selected_pidx:
        print(f"[INFO] Nothing to do: start_idx={args.start_idx}, end_idx={args.end_idx}")
        return

    print(f"[INFO] InitNO | prompts={len(prompts_dict)} token entries={len(prompt_token_index_dict)}")
    print(f"[INFO] sd_path={sd_path} | output root={root_dir}")

    print("Loading pipe_attn...")
    pipe_attn = StableDiffusionInitNOPipeline.from_pretrained(
        str(sd_path), local_files_only=True, torch_dtype=torch.float32
    ).to("cuda")

    top_seeds_dict = {}
    scoring_time_per_prompt = {}
    scoring_plus_topK_times = []
    randomK_times = []

    for j, p_idx in enumerate(selected_pidx):
        if p_idx not in prompt_token_index_dict:
            print(f"[WARN] p{p_idx:03d} missing in initno_core_token_index_dict.json, skip.")
            continue

        prompt = prompts_dict[p_idx]
        raw = prompt_token_index_dict[p_idx]
        token_idx_77 = map_token_groups_bosless_to_77(pipe_attn.tokenizer, prompt, raw)
        entity_indices = token_idx_77.get("entity", [])

        print(f"\n=== Prompt {p_idx:03d}: {prompt}")
        print(f"[INFO] entity indices (77-space, scoring): {entity_indices}")

        do_debug = args.debug and (j % args.debug_every == 0)
        if do_debug:
            debug_print_prompt_alignment(pipe_attn.tokenizer, prompt, raw, p_idx, max_show=40)

        random.seed(args.base_seed)
        seed_pool = random.sample(range(1_000_000), seeds_per_prompt)
        rows = []
        start_scoring = time.time()
        for seed in tqdm(seed_pool, desc=f"Scoring seeds p{p_idx:03d}"):
            gen = torch.Generator("cuda").manual_seed(seed)
            if entity_indices:
                _, mean_dict = pipe_attn(
                    prompt=prompt,
                    token_indices=entity_indices,
                    guidance_scale=guidance_scale_1,
                    generator=gen,
                    num_inference_steps=num_infer_steps,
                    result_root=None,
                    seed=seed,
                )
                ent_mean = (
                    np.mean([mean_dict[i] for i in entity_indices if i in mean_dict]) if mean_dict else 0.0
                )
                if do_debug and len(rows) == 0:
                    hit = [i for i in entity_indices if i in mean_dict]
                    miss = [i for i in entity_indices if i not in mean_dict]
                    print(f"[DEBUG] entity_77={entity_indices} hit={hit} miss={miss}")
            else:
                ent_mean = 0.0
            rows.append({"seed": seed, "entity_mean": ent_mean})

        scoring_time = time.time() - start_scoring
        scoring_time_per_prompt[p_idx] = scoring_time
        df = pd.DataFrame(rows)
        prompt_dir = os.path.join(root_dir, f"prompt_{p_idx:03d}")
        os.makedirs(prompt_dir, exist_ok=True)
        df.to_csv(os.path.join(prompt_dir, "seed_scores_entity.csv"), index=False)
        top_seeds_dict[p_idx] = process_seed_dataframe_entity_only(df, top_k, root_dir, p_idx)
        print(f"  ⏱ Scoring seeds time: {scoring_time:.2f} s")

    del pipe_attn
    gc.collect()
    torch.cuda.empty_cache()

    print("Loading pipe_sd...")
    pipe_sd = StableDiffusionAttendAndExcitePipeline.from_pretrained(str(sd_path), local_files_only=True).to("cuda")

    for p_idx in selected_pidx:
        if p_idx not in prompts_dict or p_idx not in prompt_token_index_dict or p_idx not in top_seeds_dict:
            continue

        prompt = prompts_dict[p_idx]
        raw = prompt_token_index_dict[p_idx]
        token_idx_77 = map_token_groups_bosless_to_77(pipe_sd.tokenizer, prompt, raw)
        all_token_indices = _flatten_token_groups(token_idx_77)

        prompt_dir = os.path.join(root_dir, f"prompt_{p_idx:03d}")
        ae_dir = os.path.join(prompt_dir, "attend_excite")
        os.makedirs(ae_dir, exist_ok=True)
        top_seeds = top_seeds_dict[p_idx]

        print(f"\n=== Generating images for Prompt {p_idx:03d}: {prompt}")
        print(f"[INFO] Attend-and-Excite token_indices: {all_token_indices}")

        start_top = time.time()
        for seed in tqdm(top_seeds, desc="TopK"):
            gen = torch.Generator("cuda").manual_seed(seed)
            img = pipe_sd(
                prompt=prompt,
                token_indices=all_token_indices,
                guidance_scale=guidance_scale_1,
                generator=gen,
                num_inference_steps=num_infer_steps,
                result_root=None,
                seed=seed,
            ).images[0]
            img.save(os.path.join(ae_dir, f"img_seed{seed}.jpg"))
        top_time = time.time() - start_top

        scoring_plus_topK = scoring_time_per_prompt.get(p_idx, 0.0) + top_time
        scoring_plus_topK_times.append(scoring_plus_topK)

        random.seed(args.base_seed)
        rand_seeds = random.sample(range(1_000_000), top_k)
        start_rand = time.time()
        for seed in tqdm(rand_seeds, desc="RandK"):
            gen = torch.Generator("cuda").manual_seed(seed)
            img = pipe_sd(
                prompt=prompt,
                token_indices=all_token_indices,
                guidance_scale=guidance_scale_1,
                generator=gen,
                num_inference_steps=num_infer_steps,
                result_root=None,
                seed=seed,
            ).images[0]
            img.save(os.path.join(ae_dir, f"img_rand{seed}.jpg"))
        rand_time = time.time() - start_rand
        randomK_times.append(rand_time)
        print(f"✅ Prompt {p_idx:03d} finished.")

    if scoring_plus_topK_times:
        print("\n=== Timing ===")
        print(f"Scoring + TopK: {sum(scoring_plus_topK_times) / len(scoring_plus_topK_times):.2f} s/prompt")
    if randomK_times:
        print(f"RandomK:        {sum(randomK_times) / len(randomK_times):.2f} s/prompt")


if __name__ == "__main__":
    main()
