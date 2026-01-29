import os
import json
import random
import re
import types
import argparse
from json import JSONDecodeError

import torch
import pandas as pd
from tqdm import tqdm
from diffusers import FluxPipeline

import attention_map_diffusers as amd
from attention_map_diffusers import init_pipeline


# -------------------------
# IO helpers
# -------------------------
def safe_load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, JSONDecodeError):
        return None


def atomic_dump_json(obj, path: str):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# -------------------------
# Scoring
# -------------------------
def score_from_attn_map(attn_map: torch.Tensor, token_indices):
    """
    attn_map shape: (batch, heads, h, w, text_len)
    token_indices: List[int] in [0, text_len)
    """
    if attn_map is None or not token_indices:
        return 0.0

    text_len = int(attn_map.shape[-1])
    idx = [int(i) for i in token_indices if 0 <= int(i) < text_len]
    if not idx:
        return 0.0

    x = attn_map[..., idx]  # (b, heads, h, w, K)
    return float(x.mean().item())


# -------------------------
# Token mapping (word positions -> T5 indices)
# -------------------------
def normalize_word(w: str) -> str:
    w = w.strip().lower()
    w = re.sub(r"^[^\w]+|[^\w]+$", "", w)
    return w


def split_prompt_words(prompt: str):
    raw = prompt.strip().split()
    words = [normalize_word(w) for w in raw]
    return [w for w in words if w]


def get_target_words_from_positions(prompt: str, positions_1based):
    words = split_prompt_words(prompt)
    targets = []
    for p in positions_1based:
        p = int(p)
        if 1 <= p <= len(words):
            targets.append(words[p - 1])
    return targets


def get_t5_tokens(tokenizer_2, prompt: str, max_len=512):
    enc = tokenizer_2(
        [prompt],
        padding="max_length",
        max_length=max_len,
        truncation=True,
        return_tensors="pt",
    )
    ids = enc.input_ids[0].tolist()
    toks = tokenizer_2.convert_ids_to_tokens(ids)
    return toks  # length = max_len


def find_word_span_in_t5_tokens(toks, word: str, start_from=0):
    """
    Find a token span for a word in T5 SentencePiece tokens.
    Returns (span_indices, next_cursor).
    """
    w = word.lower()
    n = len(toks)

    # 1) Single-token match: ▁word / word
    for i in range(start_from, n):
        t = toks[i]
        if t.lower() == ("▁" + w) or t.lstrip("▁").lower() == w:
            return [i], i + 1

    # 2) Fallback: concatenate multiple tokens (e.g., back + pack -> backpack)
    stripped = [t.lstrip("▁").lower() for t in toks]
    for i in range(start_from, n):
        acc = ""
        span = []
        for j in range(i, min(i + 8, n)):
            acc += stripped[j]
            span.append(j)
            if acc == w:
                return span, j + 1
            if len(acc) > len(w):
                break

    return [], start_from


def word_positions_to_t5_indices(prompt: str, positions_1based, tokenizer_2, max_len=512):
    """
    positions_1based: 1-based positions based on whitespace-splitting of the prompt.
    returns: unique 0-based indices aligned with T5 token sequence of length max_len.
    """
    targets = get_target_words_from_positions(prompt, positions_1based)
    toks = get_t5_tokens(tokenizer_2, prompt, max_len=max_len)

    out = []
    cursor = 0
    for w in targets:
        span, cursor = find_word_span_in_t5_tokens(toks, w, start_from=cursor)
        out.extend(span)

    seen = set()
    out = [i for i in out if not (i in seen or seen.add(i))]
    return out


# -------------------------
# Prompt set (edit as needed)
# -------------------------
prompts_dict = {
    101: "a horse and a yellow clock",
    102: "a dog with glasses",
    # Add more prompts here if desired.
}


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    test_dir = os.path.join(args.output_dir, f"test_{args.base_seed}")
    os.makedirs(test_dir, exist_ok=True)

    # Select prompts
    keys = sorted(int(k) for k in prompts_dict.keys())
    start_idx = args.start_idx if args.start_idx is not None else keys[0]
    end_idx = args.end_idx if args.end_idx is not None else keys[-1]
    prompts = [(p, prompts_dict[p]) for p in keys if start_idx <= p <= end_idx]
    if not prompts:
        return

    # Load token position dict (required)
    with open(args.prompt_token_index_dict, "r", encoding="utf-8") as f:
        prompt_token_index_dict = {int(k): v for k, v in json.load(f).items()}

    # Seed pool and global random topK
    pool_path = os.path.join(test_dir, "global_seed_pool.json")
    rand_path = os.path.join(test_dir, "global_random_top_seeds.json")

    seed_pool = safe_load_json(pool_path)
    if not seed_pool:
        seed_pool = random.Random(args.base_seed).sample(range(1_000_000), args.seeds_per_prompt)
        atomic_dump_json(seed_pool, pool_path)
    seed_pool = [int(x) for x in seed_pool]

    rand_top = safe_load_json(rand_path)
    if not rand_top:
        rand_top = random.Random(args.base_seed).sample(seed_pool, args.top_k)
        atomic_dump_json(rand_top, rand_path)
    rand_top = [int(x) for x in rand_top]

    # Load pipeline
    pipe = FluxPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16)
    pipe = init_pipeline(pipe)

    cuda = torch.device("cuda")
    cpu = torch.device("cpu")

    # Move modules (keep T5 on CPU to save GPU memory)
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

    # Cache T5 embeddings on CPU (same semantics as original logic)
    t5_cache = {}

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
        enc2_device = next(enc2.parameters()).device  # cpu

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

    # Cache CLIP pooled embeds on CPU (same semantics as original logic)
    clip_cache = {}

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

    # Replace Flux attention call for probing
    pipe.transformer = amd.replace_call_method_for_flux(pipe.transformer)

    # Cache t5 indices for each prompt/field
    index_cache = {}  # p_idx -> field -> indices list

    def get_field_t5_indices(p_idx: int, prompt: str, field: str):
        if p_idx not in index_cache:
            index_cache[p_idx] = {}
        if field in index_cache[p_idx]:
            return index_cache[p_idx][field]

        pos_1based = (prompt_token_index_dict.get(p_idx, {}) or {}).get(field, []) or []
        t5_idx = word_positions_to_t5_indices(prompt, pos_1based, pipe.tokenizer_2, max_len=args.max_sequence_length)
        index_cache[p_idx][field] = t5_idx
        return t5_idx

    top_seeds_dict = {}

    # -------------------------
    # Phase 1: Scoring
    # -------------------------
    for p_idx, prompt in prompts:
        entity_t5_idx = get_field_t5_indices(p_idx, prompt, "entity")

        rows = []
        for seed in tqdm(seed_pool, desc=f"Scoring prompt {p_idx}", leave=False):
            amd.attn_maps.clear()
            amd.PROBE_RESULT = {}

            gen = torch.Generator(device="cuda").manual_seed(int(seed))

            _ = pipe(
                prompt,
                num_inference_steps=args.score_num_infer_steps,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                generator=gen,
                output_type="latent",
                attention_only=True,
                probe_step_idx=args.capture_step_idx,
                probe_block_name=args.capture_block_name,
            )

            probe = getattr(amd, "PROBE_RESULT", {}) or {}
            attn_map = probe.get("attn_map", None)
            score = score_from_attn_map(attn_map, entity_t5_idx) if attn_map is not None else 0.0
            rows.append({"seed": int(seed), "entity_mean": float(score)})

        df = pd.DataFrame(rows).sort_values("entity_mean", ascending=False)
        top_seeds = df["seed"].head(args.top_k).tolist()
        top_seeds_dict[p_idx] = top_seeds

        prompt_dir = os.path.join(test_dir, f"prompt_{p_idx:03d}")
        os.makedirs(prompt_dir, exist_ok=True)

        df.to_csv(os.path.join(prompt_dir, "seed_scores.csv"), index=False)
        with open(os.path.join(prompt_dir, "top_entity_seeds.json"), "w", encoding="utf-8") as f:
            json.dump(top_seeds, f, indent=2)
        with open(os.path.join(prompt_dir, "random_top_seeds.json"), "w", encoding="utf-8") as f:
            json.dump(rand_top, f, indent=2)

    # -------------------------
    # Phase 2: Generation
    # -------------------------
    for p_idx, prompt in prompts:
        prompt_dir = os.path.join(test_dir, f"prompt_{p_idx:03d}")

        out_attn = os.path.join(prompt_dir, "topK_attn")
        os.makedirs(out_attn, exist_ok=True)
        for seed in tqdm(top_seeds_dict[p_idx], desc=f"Generate topK(attn) p{p_idx}", leave=False):
            out_path = os.path.join(out_attn, f"img_seed{int(seed)}.png")
            if os.path.exists(out_path):
                continue
            gen = torch.Generator(device="cuda").manual_seed(int(seed))
            img = pipe(
                prompt,
                num_inference_steps=args.gen_steps,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                generator=gen,
                output_type="pil",
            ).images[0]
            img.save(out_path)

        out_rand = os.path.join(prompt_dir, "topK_random")
        os.makedirs(out_rand, exist_ok=True)
        for seed in tqdm(rand_top, desc=f"Generate topK(random) p{p_idx}", leave=False):
            out_path = os.path.join(out_rand, f"img_seed{int(seed)}.png")
            if os.path.exists(out_path):
                continue
            gen = torch.Generator(device="cuda").manual_seed(int(seed))
            img = pipe(
                prompt,
                num_inference_steps=args.gen_steps,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                generator=gen,
                output_type="pil",
            ).images[0]
            img.save(out_path)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, help="Path or HF id for Flux model.")
    parser.add_argument("--prompt_token_index_dict", type=str, default="prompt_token_index_dict.json")
    parser.add_argument("--output_dir", type=str, default="Flux_SmallPool10_top3_attn12_release")

    parser.add_argument("--base_seed", type=int, default=11)
    parser.add_argument("--start_idx", type=int, required=True)
    parser.add_argument("--end_idx", type=int, required=True)

    parser.add_argument("--seeds_per_prompt", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=3)

    parser.add_argument("--capture_step_idx", type=int, default=10)
    parser.add_argument("--capture_block_name", type=str, default="transformer_blocks.12.attn")

    parser.add_argument("--score_num_infer_steps", type=int, default=50)
    parser.add_argument("--gen_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)

    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--max_sequence_length", type=int, default=512)

    args = parser.parse_args()
    main(args)
