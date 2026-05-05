#!/usr/bin/env python3
"""
Flux (Hugging Face diffusers): attention-based seed scoring, then image generation
(topK_attn vs topK_random), wired through ``attention_map_diffusers`` — same role as
the old standalone ``main.py``. This does **not** use ``initno.*`` Stable Diffusion
pipelines; those live only in the initno1 repo (e.g. ``run_attention9_initno.py``).

Default JSON names still say ``prompts_initno`` / ``initno_core_token_index_dict`` because
they ship the benchmark prompt list and word-position entity labels for that split.

Single-file CLI: Flux only. Which checkpoint to load is ``--model-id`` (HF id or local path).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import types
from json import JSONDecodeError
from pathlib import Path

import attention_map_diffusers as amd
import pandas as pd
import torch
from diffusers import FluxPipeline
from tqdm import tqdm

from attention_map_diffusers import init_pipeline

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def safe_load_json(path: str | os.PathLike[str]):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, JSONDecodeError):
        return None


def atomic_dump_json(obj, path: str | os.PathLike[str]) -> None:
    path = str(path)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Word positions (JSON) -> T5 token indices for Flux attention map
# ---------------------------------------------------------------------------


def score_from_attn_map(attn_map: torch.Tensor | None, token_indices: list[int]) -> float:
    if attn_map is None or not token_indices:
        return 0.0
    text_len = attn_map.shape[-1]
    token_indices = [i for i in token_indices if 0 <= i < text_len]
    if not token_indices:
        return 0.0
    x = attn_map[..., token_indices]
    return float(x.mean().item())


def normalize_word(w: str) -> str:
    w = w.strip().lower()
    return re.sub(r"^[^\w]+|[^\w]+$", "", w)


def split_prompt_words(prompt: str) -> list[str]:
    raw = prompt.strip().split()
    words = [normalize_word(w) for w in raw]
    return [w for w in words if w]


def get_target_words_from_positions(prompt: str, positions_1based: list[int]) -> list[str]:
    words = split_prompt_words(prompt)
    targets = []
    for p in positions_1based:
        p = int(p)
        if 1 <= p <= len(words):
            targets.append(words[p - 1])
    return targets


def get_t5_tokens(tokenizer_2, prompt: str, max_len: int = 512) -> list[str]:
    enc = tokenizer_2(
        [prompt],
        padding="max_length",
        max_length=max_len,
        truncation=True,
        return_tensors="pt",
    )
    ids = enc.input_ids[0].tolist()
    return tokenizer_2.convert_ids_to_tokens(ids)


def find_word_span_in_t5_tokens(
    toks: list[str], word: str, start_from: int = 0
) -> tuple[list[int], int, str]:
    w = word.lower()
    n = len(toks)
    for i in range(start_from, n):
        t = toks[i]
        if t.lower() == ("▁" + w) or t.lstrip("▁").lower() == w:
            return [i], i + 1, t
    stripped = [t.lstrip("▁").lower() for t in toks]
    for i in range(start_from, n):
        acc = ""
        span: list[int] = []
        for j in range(i, min(i + 8, n)):
            acc += stripped[j]
            span.append(j)
            if acc == w:
                matched = "".join([toks[k] for k in span])
                return span, j + 1, matched
            if len(acc) > len(w):
                break
    return [], start_from, ""


def word_positions_to_t5_indices(
    prompt: str, positions_1based: list[int], tokenizer_2, max_len: int = 512
) -> tuple[list[int], list[str], list[str], list[str]]:
    targets = get_target_words_from_positions(prompt, positions_1based)
    toks = get_t5_tokens(tokenizer_2, prompt, max_len=max_len)
    out: list[int] = []
    cursor = 0
    misses: list[str] = []
    for w in targets:
        span, cursor, _matched = find_word_span_in_t5_tokens(toks, w, start_from=cursor)
        if not span:
            misses.append(w)
        out.extend(span)
    seen: set[int] = set()
    out = [i for i in out if not (i in seen or seen.add(i))]
    return out, targets, toks, misses


# ---------------------------------------------------------------------------
# Flux pipeline
# ---------------------------------------------------------------------------


def _torch_dtype(name: str) -> torch.dtype:
    m = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = name.lower().replace(" ", "")
    if key not in m:
        raise ValueError(f"Unknown dtype {name!r}; use one of: {sorted(set(m.keys()))}")
    return m[key]


def build_flux_pipeline(args: argparse.Namespace) -> FluxPipeline:
    """Load Flux weights from ``args.model_id`` and apply attention hooks / device layout."""
    dt = _torch_dtype(args.torch_dtype)
    pipe = FluxPipeline.from_pretrained(args.model_id, torch_dtype=dt)
    pipe = init_pipeline(pipe)
    cuda = torch.device("cuda")
    cpu = torch.device("cpu")

    pipe.transformer.to(cuda)
    if getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder.to(cuda)
    if getattr(pipe, "text_encoder_2", None) is not None:
        pipe.text_encoder_2.to(cpu)
        pipe.text_encoder_2.float()
    if getattr(pipe, "vae", None) is not None:
        pipe.vae.to(cuda)
    pipe.enable_vae_slicing()
    pipe.enable_attention_slicing()

    t5_cache: dict = {}

    def patched_get_t5_prompt_embeds(
        self,
        prompt,
        num_images_per_prompt=1,
        max_sequence_length=512,
        device=None,
        dtype=None,
    ):
        out_device = device or getattr(self, "_execution_device", cuda)
        out_dtype = dtype or getattr(self.transformer, "dtype", torch.float16)
        prompt_list = [prompt] if isinstance(prompt, str) else prompt
        key = (tuple(prompt_list), int(num_images_per_prompt), int(max_sequence_length), str(out_dtype))
        if key in t5_cache:
            return t5_cache[key].to(device=out_device, dtype=out_dtype, non_blocking=True)

        tok2 = self.tokenizer_2
        enc2 = self.text_encoder_2
        enc2_device = next(enc2.parameters()).device
        text_inputs = tok2(
            prompt_list,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(enc2_device)
        with torch.inference_mode():
            prompt_embeds = enc2(input_ids, output_hidden_states=False)[0]
        bsz, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bsz * num_images_per_prompt, seq_len, -1)
        prompt_embeds = prompt_embeds.to(device=cpu, dtype=out_dtype)
        t5_cache[key] = prompt_embeds
        return prompt_embeds.to(device=out_device, dtype=out_dtype, non_blocking=True)

    pipe._get_t5_prompt_embeds = types.MethodType(patched_get_t5_prompt_embeds, pipe)

    clip_cache: dict = {}

    def patched_get_clip_prompt_embeds(self, prompt, device=None, num_images_per_prompt=1):
        out_device = device or getattr(self, "_execution_device", cuda)
        out_dtype = getattr(self.transformer, "dtype", torch.float16)
        prompt_list = [prompt] if isinstance(prompt, str) else prompt
        key = (tuple(prompt_list), int(num_images_per_prompt), str(out_dtype))
        if key in clip_cache:
            return clip_cache[key].to(device=out_device, dtype=out_dtype, non_blocking=True)

        tok = self.tokenizer
        enc = self.text_encoder
        enc_device = next(enc.parameters()).device
        text_inputs = tok(
            prompt_list,
            padding="max_length",
            max_length=tok.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(enc_device)
        with torch.inference_mode():
            enc_out = enc(input_ids, output_hidden_states=False)
        pooled = enc_out[1] if isinstance(enc_out, (tuple, list)) else getattr(enc_out, "pooler_output", None)
        if pooled is None and hasattr(enc_out, "pooler_output"):
            pooled = enc_out.pooler_output
        if pooled is None:
            last = enc_out[0] if isinstance(enc_out, (tuple, list)) else enc_out.last_hidden_state
            pooled = last[:, 0, :]
        pooled = pooled.repeat_interleave(num_images_per_prompt, dim=0)
        pooled = pooled.to(device=cpu, dtype=out_dtype)
        clip_cache[key] = pooled
        return pooled.to(device=out_device, dtype=out_dtype, non_blocking=True)

    pipe._get_clip_prompt_embeds = types.MethodType(patched_get_clip_prompt_embeds, pipe)
    pipe.transformer = amd.replace_call_method_for_flux(pipe.transformer)
    return pipe


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).resolve()
    prompts_path = Path(args.prompts_json)
    if not prompts_path.is_absolute():
        prompts_path = data_dir / prompts_path
    token_path = Path(args.token_json)
    if not token_path.is_absolute():
        token_path = data_dir / token_path

    if not prompts_path.is_file():
        raise FileNotFoundError(f"Prompts JSON not found: {prompts_path}")
    if not token_path.is_file():
        raise FileNotFoundError(f"Token index JSON not found: {token_path}")

    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts_dict = {int(k): str(v) for k, v in json.load(f).items()}
    with open(token_path, "r", encoding="utf-8") as f:
        prompt_token_index_dict = {int(k): v for k, v in json.load(f).items()}

    keys = sorted(prompts_dict.keys())
    prompts = [(int(p), prompts_dict[p]) for p in keys if args.start_idx <= int(p) <= args.end_idx]

    run_name = args.run_name or f"Flux_SmallPool{args.seeds_per_prompt}_top{args.top_k}_run"
    root_dir = os.path.join(os.getcwd(), run_name, f"test_{args.base_seed}")
    os.makedirs(root_dir, exist_ok=True)

    print(
        f"[INFO] model_id={args.model_id} base_seed={args.base_seed} "
        f"range=[{args.start_idx},{args.end_idx}] n_prompts={len(prompts)} -> {root_dir}"
    )
    if not prompts:
        print("[INFO] No prompts in range; exiting.")
        return

    pool_path = os.path.join(root_dir, "global_seed_pool.json")
    rand_path = os.path.join(root_dir, "global_random_top_seeds.json")

    seed_pool = safe_load_json(pool_path)
    if not seed_pool:
        seed_pool = random.Random(args.base_seed).sample(range(1_000_000), args.seeds_per_prompt)
        atomic_dump_json(seed_pool, pool_path)
    seed_pool = [int(x) for x in seed_pool]

    rand_top_seeds_global = safe_load_json(rand_path)
    if not rand_top_seeds_global:
        rand_top_seeds_global = random.Random(args.base_seed).sample(seed_pool, args.top_k)
        atomic_dump_json(rand_top_seeds_global, rand_path)
    rand_top_seeds_global = [int(x) for x in rand_top_seeds_global]

    print(f"[INFO] seed_pool={len(seed_pool)} random_top_k={rand_top_seeds_global}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print("[INFO] Loading Flux pipeline...")
    pipe = build_flux_pipeline(args)

    t5_index_cache: dict[int, dict[str, tuple]] = {}

    def get_field_t5_indices(p_idx: int, prompt: str, field: str):
        if p_idx not in t5_index_cache:
            t5_index_cache[p_idx] = {}
        if field in t5_index_cache[p_idx]:
            return t5_index_cache[p_idx][field]
        pos_1based = (prompt_token_index_dict.get(p_idx, {}) or {}).get(field, []) or []
        t5_idx, targets, toks, misses = word_positions_to_t5_indices(
            prompt, pos_1based, pipe.tokenizer_2, max_len=512
        )
        t5_index_cache[p_idx][field] = (pos_1based, targets, t5_idx, toks, misses)
        return t5_index_cache[p_idx][field]

    top_seeds_dict: dict[int, list[int]] = {}

    for p_idx, prompt in prompts:
        print(f"\n=== [Scoring] prompt {p_idx:03d} ===")
        print("Prompt:", prompt)

        _pos_entity, _entity_words, entity_t5_idx, _t5_toks, misses = get_field_t5_indices(p_idx, prompt, "entity")
        if misses:
            print("[WARN] words not found in T5 tokens:", misses)

        if not entity_t5_idx:
            print("[WARN] entity_t5_idx empty; scores will be 0.")

        rows = []
        for seed in tqdm(seed_pool, desc=f"score p{p_idx:03d}", leave=False):
            amd.attn_maps.clear()
            amd.PROBE_RESULT = {}
            gen = torch.Generator(device="cuda").manual_seed(int(seed))
            score_infer_steps = max(args.score_step + 1, 1)
            _ = pipe(
                prompt,
                num_inference_steps=score_infer_steps,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                generator=gen,
                output_type="latent",
                attention_only=True,
                probe_step_idx=args.score_step,
                probe_block_name=args.probe_block_name,
            )
            probe = getattr(amd, "PROBE_RESULT", {}) or {}
            attn_map = probe.get("attn_map", None)

            ent_mean = score_from_attn_map(attn_map, entity_t5_idx) if attn_map is not None else 0.0
            rows.append({"seed": int(seed), "entity_mean": float(ent_mean)})

        df = pd.DataFrame(rows).sort_values("entity_mean", ascending=False)
        top_seeds = df["seed"].head(args.top_k).tolist()
        top_seeds_dict[p_idx] = top_seeds

        prompt_dir = os.path.join(root_dir, f"prompt_{p_idx:03d}")
        os.makedirs(prompt_dir, exist_ok=True)
        df.to_csv(os.path.join(prompt_dir, "seed_scores.csv"), index=False)
        with open(os.path.join(prompt_dir, "top_entity_seeds.json"), "w", encoding="utf-8") as f:
            json.dump(top_seeds, f, indent=2)
        with open(os.path.join(prompt_dir, "random_top_seeds.json"), "w", encoding="utf-8") as f:
            json.dump(rand_top_seeds_global, f, indent=2)

        print(f"\n  Seed scores (entity attention), prompt {p_idx:03d}:")
        for rank, r in enumerate(df.itertuples(index=False), start=1):
            flag = "  <- TOP" if rank <= args.top_k else ""
            print(f"    {rank:02d}. seed={int(r.seed):6d}  score={float(r.entity_mean):.8f}{flag}")
        print(f"  Top-{args.top_k} seeds: {top_seeds}")

    print("\n=== [Generation] ===")
    amd.CAPTURE_ENABLED = False

    for p_idx, prompt in prompts:
        print(f"\n=== [Generate] prompt {p_idx:03d} ===")
        prompt_dir = os.path.join(root_dir, f"prompt_{p_idx:03d}")

        out_dir_attn = os.path.join(prompt_dir, "topK_attn")
        os.makedirs(out_dir_attn, exist_ok=True)
        for seed in tqdm(top_seeds_dict[p_idx], desc="topK_attn", leave=False):
            out_path = os.path.join(out_dir_attn, f"img_seed{int(seed)}.png")
            if os.path.exists(out_path):
                continue
            gen = torch.Generator(device="cuda").manual_seed(int(seed))
            images = pipe(
                prompt,
                num_inference_steps=args.gen_steps,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                generator=gen,
                output_type="pil",
            ).images
            images[0].save(out_path)

        out_dir_rand = os.path.join(prompt_dir, "topK_random")
        os.makedirs(out_dir_rand, exist_ok=True)
        for seed in tqdm(rand_top_seeds_global, desc="topK_random", leave=False):
            out_path = os.path.join(out_dir_rand, f"img_seed{int(seed)}.png")
            if os.path.exists(out_path):
                continue
            gen = torch.Generator(device="cuda").manual_seed(int(seed))
            images = pipe(
                prompt,
                num_inference_steps=args.gen_steps,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                generator=gen,
                output_type="pil",
            ).images
            images[0].save(out_path)

        print(f"Done prompt {p_idx:03d}.")

    print("\n=== Finished ===")
    print("Output:", root_dir)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Flux attention scoring + generation (single file).")
    p.add_argument(
        "--model-id",
        default="black-forest-labs/FLUX.1-dev",
        help="Flux weights: Hugging Face model id (default: FLUX.1-dev) or a local directory.",
    )
    p.add_argument("--torch-dtype", default="float16", help="float16 | bfloat16 | float32")

    p.add_argument("--data-dir", type=Path, default=here, help="Directory for prompts/token JSON.")
    p.add_argument("--prompts-json", type=Path, default=Path("prompts_initno.json"))
    p.add_argument("--token-json", type=Path, default=Path("initno_core_token_index_dict.json"))

    p.add_argument("--base-seed", type=int, default=11)
    p.add_argument("--start-idx", type=int, required=True)
    p.add_argument("--end-idx", type=int, required=True)

    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument(
        "--gen-steps",
        type=int,
        default=50,
        help="Denoising steps for full image generation (topK_attn / topK_random).",
    )
    p.add_argument(
        "--score-step",
        type=int,
        default=10,
        help="0-based denoise step where attention is read for seed scoring; scoring runs exactly score_step+1 steps.",
    )
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--seeds-per-prompt", type=int, default=50)
    p.add_argument("--top-k", type=int, default=1)

    p.add_argument("--run-name", default="", help="Output folder under cwd; default Flux_SmallPool{sp}_top{k}_run")
    p.add_argument(
        "--probe-block-name",
        default="transformer_blocks.12.attn",
        help="Transformer block submodule name for captured attention.",
    )
    return p.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
