import os, json, time, random, types, re
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import torch
from diffusers import FluxPipeline
from json import JSONDecodeError
import attention_map_diffusers as amd
from attention_map_diffusers import init_pipeline

MODEL_ID = "./flux_1_dev"

# ====== 你要抓的“第几步(step idx)” + “哪一层(block name)” ======
# step idx：denoising loop 里的 i（0-based）
CAPTURE_STEP_IDX = 10
CAPTURE_BLOCK_NAME = "transformer_blocks.12.attn"


# ====== Debug 开关 ======
DEBUG_CHECK_MAPPING = True          # 每个 prompt 打印一次映射检查
DEBUG_PRINT_ATTN_ON_FIRST_SEED = True  # 每个 prompt 的第一个 seed 打印 attn_map shape/text_len

def safe_load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, JSONDecodeError):
        return None

def atomic_dump_json(obj, path):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)  # 原子替换


def report_gpu_mem(tag: str):
    if not torch.cuda.is_available():
        print(f"[GPU MEM] {tag}: CUDA not available")
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    free_b, total_b = torch.cuda.mem_get_info()
    alloc_b = torch.cuda.memory_allocated()
    reserv_b = torch.cuda.memory_reserved()
    max_alloc_b = torch.cuda.max_memory_allocated()
    max_reserv_b = torch.cuda.max_memory_reserved()

    def gb(x): return x / 1024**3
    print(f"[GPU MEM] {tag}")
    print(f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"  device=0 name={torch.cuda.get_device_name(0)}")
    print(f"  free/total={gb(free_b):.2f}GB/{gb(total_b):.2f}GB")
    print(f"  allocated={gb(alloc_b):.2f}GB reserved={gb(reserv_b):.2f}GB")
    print(f"  max_alloc={gb(max_alloc_b):.2f}GB max_reserved={gb(max_reserv_b):.2f}GB")
    print()


def score_from_attn_map(attn_map: torch.Tensor, token_indices):
    """
    attn_map: (batch, heads, h, w, text_len)
    token_indices: list[int]
    return: float score
    """
    if attn_map is None:
        return 0.0
    if not token_indices:
        return 0.0

    text_len = attn_map.shape[-1]
    token_indices = [i for i in token_indices if 0 <= i < text_len]
    if not token_indices:
        return 0.0

    x = attn_map[..., token_indices]     # (b, heads, h, w, K)
    score = x.mean().item()              # heads + hw + token 平均
    return float(score)


def normalize_word(w: str) -> str:
    w = w.strip().lower()
    # 去掉首尾标点
    w = re.sub(r"^[^\w]+|[^\w]+$", "", w)
    return w

def split_prompt_words(prompt: str):
    raw = prompt.strip().split()
    words = [normalize_word(w) for w in raw]
    words = [w for w in words if w]
    return words  # 1-based positions: words[pos-1]

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
    return toks  # len=512

def find_word_span_in_t5_tokens(toks, word: str, start_from=0):
    """
    在 T5 SentencePiece token 序列里，从 start_from 开始找 word 对应的 token span。
    返回 (span_indices, next_cursor, matched_str)
    - 优先单 token 命中：▁word / word
    - 兜底：多 token 拼接命中（back + pack -> backpack）
    """
    w = word.lower()
    n = len(toks)

    # 1) 单 token 命中（最常见）
    for i in range(start_from, n):
        t = toks[i]
        if t.lower() == ("▁" + w) or t.lstrip("▁").lower() == w:
            return [i], i + 1, t

    # 2) 多 token 拼接兜底
    stripped = [t.lstrip("▁").lower() for t in toks]
    for i in range(start_from, n):
        acc = ""
        span = []
        for j in range(i, min(i + 8, n)):  # 最多拼 8 段，够用
            acc += stripped[j]
            span.append(j)
            if acc == w:
                matched = "".join([toks[k] for k in span])
                return span, j + 1, matched
            if len(acc) > len(w):
                break

    return [], start_from, ""




def word_positions_to_t5_indices(prompt: str, positions_1based, tokenizer_2, max_len=512):
    """
    输入：positions_1based 是你 json 里那种“按空格分词的 1-based 位置”
    输出：t5_indices 是能直接索引 attn_map[..., idx] 的 0-based indices（对齐 text_len=512）
    同时返回 targets(词) 和 toks(用于 debug)
    """
    targets = get_target_words_from_positions(prompt, positions_1based)
    toks = get_t5_tokens(tokenizer_2, prompt, max_len=max_len)

    out = []
    cursor = 0
    misses = []
    for w in targets:
        span, cursor, matched = find_word_span_in_t5_tokens(toks, w, start_from=cursor)
        if not span:
            misses.append(w)
        out.extend(span)

    # 去重保持顺序
    seen = set()
    out = [i for i in out if not (i in seen or seen.add(i))]
    return out, targets, toks, misses



def fetch_captured_attn():
    """
    从 amd.attn_maps 里拿 map：
    amd.attn_maps: {timestep: {layer_name: attn_map}}
    这里取“任意一个 timestep”（因为你 hook 已经只存目标那一张）
    """
    if not getattr(amd, "attn_maps", None):
        return None
    if len(amd.attn_maps) == 0:
        return None

    any_t = next(iter(amd.attn_maps.keys()))
    layer_dict = amd.attn_maps.get(any_t, {})

    return layer_dict.get(CAPTURE_BLOCK_NAME, None)


def inspect_entity_indices(pipe, prompt, entity_indices, max_t5_len=512):
    print("\nPROMPT:", prompt)
    print("entity_indices:", entity_indices)

    # ---- CLIP tokens ----
    if getattr(pipe, "tokenizer", None) is not None:
        tok = pipe.tokenizer(
            [prompt],
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        clip_ids = tok.input_ids[0].tolist()
        clip_tokens = pipe.tokenizer.convert_ids_to_tokens(clip_ids)
        print("\n[CLIP] len =", len(clip_tokens), " model_max_length =", pipe.tokenizer.model_max_length)
        for i in entity_indices:
            if 0 <= i < len(clip_tokens):
                print(f"  idx {i:3d}: {clip_tokens[i]}")
            else:
                print(f"  idx {i:3d}: (out of range for CLIP)")

    # ---- T5 tokens ----
    if getattr(pipe, "tokenizer_2", None) is not None:
        tok2 = pipe.tokenizer_2(
            [prompt],
            padding="max_length",
            max_length=max_t5_len,
            truncation=True,
            return_tensors="pt",
        )
        t5_ids = tok2.input_ids[0].tolist()
        # 注意：很多 T5 tokenizer 是 SentencePiece，token 形式可能是 ▁word
        t5_tokens = pipe.tokenizer_2.convert_ids_to_tokens(t5_ids)
        print("\n[T5] len =", len(t5_tokens), " max_length =", max_t5_len)
        for i in entity_indices:
            if 0 <= i < len(t5_tokens):
                print(f"  idx {i:3d}: {t5_tokens[i]}")
            else:
                print(f"  idx {i:3d}: (out of range for T5)")

    print()



def main(base_seed=11, start_idx=None, end_idx=None):
    guidance_scale = 7.5
    GEN_STEPS = 50

    SEEDS_PER_PROMPT = 10
    TOP_K = 3

    # scoring 用 50-step schedule，但在 probe_step_idx 早停
    SCORE_NUM_INFER_STEPS = 50

    height = 512
    width = 512

    root_dir = f"Flux_SmallPool{SEEDS_PER_PROMPT}_top{TOP_K}_new_attention12"
    test_dir = os.path.join(root_dir, f"test_{base_seed}")
    os.makedirs(test_dir, exist_ok=True)

    # ====== prompts 选取 ======
    if "prompts_dict" not in globals() or not isinstance(prompts_dict, dict):
        prompts = []
    else:
        keys = sorted(prompts_dict.keys())
        if start_idx is None: start_idx = int(keys[0])
        if end_idx is None: end_idx = int(keys[-1])
        prompts = [(int(p), prompts_dict[p]) for p in keys if start_idx <= int(p) <= end_idx]

    print(f"[INFO] base_seed={base_seed}, start_idx={start_idx}, end_idx={end_idx}, n_prompts={len(prompts)}")
    if len(prompts) == 0:
        print(f"[INFO] No prompts in range {start_idx}~{end_idx}. Nothing to do.")
        print("Saved to:", test_dir)
        return

    # ====== 读你的 word-position json（1-based positions） ======
    with open("prompt_token_index_dict.json", "r") as f:
        prompt_token_index_dict = {int(k): v for k, v in json.load(f).items()}

    # ====== seed_pool & random_top（全局固定） ======
    pool_path = os.path.join(test_dir, "global_seed_pool.json")
    rand_path = os.path.join(test_dir, "global_random_top_seeds.json")

    seed_pool = safe_load_json(pool_path)
    if not seed_pool:
        seed_pool = random.Random(base_seed).sample(range(1_000_000), SEEDS_PER_PROMPT)
        atomic_dump_json(seed_pool, pool_path)
    seed_pool = [int(x) for x in seed_pool]

    rand_top_seeds_global = safe_load_json(rand_path)
    if not rand_top_seeds_global:
        rand_top_seeds_global = random.Random(base_seed).sample(seed_pool, TOP_K)
        atomic_dump_json(rand_top_seeds_global, rand_path)
    rand_top_seeds_global = [int(x) for x in rand_top_seeds_global]

    print(f"[INFO] Global seed_pool size={len(seed_pool)} | Global random_top={rand_top_seeds_global}")

    # ====== Load pipeline ======
    print("Loading Flux pipeline...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    pipe = init_pipeline(pipe)

    cuda = torch.device("cuda")
    cpu = torch.device("cpu")

    print("[INFO] Move transformer + CLIP to CUDA; keep T5(text_encoder_2) on CPU.")
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

    # ====== patch T5/CLIP cache（保持你原逻辑） ======
    t5_cache = {}
    def patched_get_t5_prompt_embeds(self, prompt, num_images_per_prompt=1,
                                     max_sequence_length=512, device=None, dtype=None):
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
            prompt_list, padding="max_length", max_length=max_sequence_length,
            truncation=True, return_tensors="pt"
        )
        input_ids = text_inputs.input_ids.to(enc2_device)

        with torch.inference_mode():
            prompt_embeds = enc2(input_ids, output_hidden_states=False)[0]

        bsz, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bsz * num_images_per_prompt, seq_len, -1)

        # cache on CPU in out_dtype
        prompt_embeds = prompt_embeds.to(device=cpu, dtype=out_dtype)
        t5_cache[key] = prompt_embeds
        return prompt_embeds.to(device=out_device, dtype=out_dtype, non_blocking=True)

    pipe._get_t5_prompt_embeds = types.MethodType(patched_get_t5_prompt_embeds, pipe)

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
            prompt_list, padding="max_length", max_length=tok.model_max_length,
            truncation=True, return_tensors="pt"
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

    # ====== hook forward 替换一次 ======
    pipe.transformer = amd.replace_call_method_for_flux(pipe.transformer)

    # ====== 每个 prompt 的 “word-pos -> T5 indices” 只算一次，缓存起来 ======
    t5_index_cache = {}  # p_idx -> dict(field -> indices)

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

    top_seeds_dict = {}

    # ==========================================================
    # 第一阶段：Scoring
    # ==========================================================
    for p_idx, prompt in prompts:
        print(f"\n=== [Scoring] Prompt {p_idx:03d} ===")
        print("Prompt:", prompt)

        # --- 取你 json 里的 word-positions，然后映射到 T5 token indices ---
        pos_entity, entity_words, entity_t5_idx, t5_toks, misses = get_field_t5_indices(p_idx, prompt, "entity")

        if DEBUG_CHECK_MAPPING:
            print("\n[CHECK] entity word-positions (1-based):", pos_entity)
            print("[CHECK] entity target words:", entity_words)
            print("[CHECK] mapped T5 token indices:", entity_t5_idx)

            # 打印这些 index 对应的 T5 token
            for i in entity_t5_idx:
                if 0 <= i < len(t5_toks):
                    print(f"    [CHECK] t5_idx={i:3d} token={t5_toks[i]}")
            if misses:
                print("  [WARN] These target words were NOT found in T5 tokens:", misses)

        if not entity_t5_idx:
            print("  [WARN] entity_t5_idx empty -> scores will be 0.0 for all seeds (check mapping).")

        rows = []
        printed_attn = False

        for seed in tqdm(seed_pool, desc=f"Scoring p{p_idx:03d}", leave=False):
            amd.attn_maps.clear()
            amd.PROBE_RESULT = {}

            gen = torch.Generator(device="cuda").manual_seed(int(seed))

            _ = pipe(
                prompt,
                num_inference_steps=SCORE_NUM_INFER_STEPS,   # 50-step schedule
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=gen,
                output_type="latent",
                attention_only=True,
                probe_step_idx=CAPTURE_STEP_IDX,             # e.g. 10 (0-based)
                probe_block_name=CAPTURE_BLOCK_NAME,         # e.g. "transformer_blocks.3.attn"
            )

            probe = getattr(amd, "PROBE_RESULT", {}) or {}
            attn_map = probe.get("attn_map", None)

            if DEBUG_PRINT_ATTN_ON_FIRST_SEED and (not printed_attn):
                if attn_map is None:
                    print("[DEBUG] attn_map is None (probe failed?)")
                else:
                    print("[DEBUG] attn_map shape:", tuple(attn_map.shape))
                    print("[DEBUG] attn_map text_len:", attn_map.shape[-1])
                    # 再做一个越界检查
                    if entity_t5_idx:
                        print("[DEBUG] max(entity_t5_idx) =", max(entity_t5_idx),
                              " < text_len?", max(entity_t5_idx) < attn_map.shape[-1])
                printed_attn = True

            ent_mean = score_from_attn_map(attn_map, entity_t5_idx) if attn_map is not None else 0.0
            rows.append({"seed": int(seed), "entity_mean": float(ent_mean)})

        df = pd.DataFrame(rows).sort_values("entity_mean", ascending=False)
        top_seeds = df["seed"].head(TOP_K).tolist()
        top_seeds_dict[p_idx] = top_seeds

        prompt_dir = os.path.join(test_dir, f"prompt_{p_idx:03d}")
        os.makedirs(prompt_dir, exist_ok=True)

        df.to_csv(os.path.join(prompt_dir, "seed_scores.csv"), index=False)
        with open(os.path.join(prompt_dir, "top_entity_seeds.json"), "w") as f:
            json.dump(top_seeds, f, indent=2)
        with open(os.path.join(prompt_dir, "random_top_seeds.json"), "w") as f:
            json.dump(rand_top_seeds_global, f, indent=2)

        # 打印该 prompt 的 seed 分数排序（可选）
        print(f"\n  📊 Seed scores (entity attn) for prompt {p_idx:03d} (sorted):")
        for rank, r in enumerate(df.itertuples(index=False), start=1):
            flag = "  <-- TOP" if rank <= TOP_K else ""
            print(f"    {rank:02d}. seed={int(r.seed):6d}  score={float(r.entity_mean):.8f}{flag}")
        print(f"  ✅ Top{TOP_K} seeds (entity attn): {top_seeds}")

    # ==========================================================
    # 第二阶段：生成图（不抓 attention）
    # ==========================================================
    print("\n=== [Generation] start ===")
    amd.CAPTURE_ENABLED = False  # 保险起见

    for p_idx, prompt in prompts:
        print(f"\n=== [Generate] Prompt {p_idx:03d} ===")
        prompt_dir = os.path.join(test_dir, f"prompt_{p_idx:03d}")

        # 1) topK(attn)
        out_dir_attn = os.path.join(prompt_dir, "topK_attn")
        os.makedirs(out_dir_attn, exist_ok=True)
        for seed in tqdm(top_seeds_dict[p_idx], desc="TopK(attn) gen", leave=False):
            out_path = os.path.join(out_dir_attn, f"img_seed{int(seed)}.png")
            if os.path.exists(out_path):
                continue
            gen = torch.Generator(device="cuda").manual_seed(int(seed))
            images = pipe(
                prompt,
                num_inference_steps=GEN_STEPS,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=gen,
                output_type="pil",
            ).images
            images[0].save(out_path)

        # 2) topK(random)
        out_dir_rand = os.path.join(prompt_dir, "topK_random")
        os.makedirs(out_dir_rand, exist_ok=True)
        for seed in tqdm(rand_top_seeds_global, desc="TopK(random) gen", leave=False):
            out_path = os.path.join(out_dir_rand, f"img_seed{int(seed)}.png")
            if os.path.exists(out_path):
                continue
            gen = torch.Generator(device="cuda").manual_seed(int(seed))
            images = pipe(
                prompt,
                num_inference_steps=GEN_STEPS,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=gen,
                output_type="pil",
            ).images
            images[0].save(out_path)

        print(f"✅ Prompt {p_idx:03d} finished.")

    print("\n=== Done ===")
    print("Saved to:", test_dir)


prompts_dict = {
    # 1:  "a elephant and a rabbit",
    # 2:  "a dog and a frog",
    # 3:  "a bird and a mouse",
    # 4:  "a monkey and a frog",
    # 5:  "a horse and a monkey",
    # 6:  "a bird and a turtle",
    # 7:  "a bird and a lion",
    # 8:  "a lion and a monkey",
    # 9:  "a horse and a turtle",
    # 10: "a bird and a monkey",
    # 11: "a bear and a frog",
    # 12: "a bear and a turtle",
    # 13: "a dog and a elephant",
    # 14: "a dog and a horse",
    # 15: "a turtle and a mouse",
    # 16: "a cat and a turtle",
    # 17: "a dog and a mouse",
    # 18: "a cat and a elephant",
    # 19: "a cat and a bird",
    # 20: "a dog and a monkey",
    # 21: "a lion and a mouse",
    # 22: "a bear and a lion",
    # 23: "a bird and a elephant",
    # 24: "a lion and a turtle",
    # 25: "a dog and a bird",
    # 26: "a bird and a rabbit",
    # 27: "a elephant and a turtle",
    # 28: "a lion and a elephant",
    # 29: "a cat and a rabbit",
    # 30: "a dog and a bear",
    # 31: "a dog and a rabbit",
    # 32: "a cat and a bear",
    # 33: "a bird and a horse",
    # 34: "a rabbit and a mouse",
    # 35: "a bird and a bear",
    # 36: "a bear and a monkey",
    # 37: "a horse and a frog",
    # 38: "a cat and a horse",
    # 39: "a frog and a rabbit",
    # 40: "a bear and a mouse",
    # 41: "a monkey and a rabbit",
    # 42: "a cat and a dog",
    # 43: "a lion and a frog",
    # 44: "a frog and a mouse",
    # 45: "a dog and a lion",
    # 46: "a lion and a rabbit",
    # 47: "a elephant and a frog",
    # 48: "a frog and a turtle",
    # 49: "a cat and a lion",
    # 50: "a horse and a rabbit",
    # 51: "a cat and a monkey",
    # 52: "a bear and a rabbit",
    # 53: "a turtle and a rabbit",
    # 54: "a elephant and a monkey",
    # 55: "a bird and a frog",
    # 56: "a lion and a horse",
    # 57: "a bear and a horse",
    # 58: "a bear and a elephant",
    # 59: "a horse and a mouse",
    # 60: "a dog and a turtle",
    # 61: "a monkey and a mouse",
    # 62: "a cat and a frog",
    # 63: "a monkey and a turtle",
    # 64: "a horse and a elephant",
    # 65: "a cat and a mouse",
    # 66: "a elephant and a mouse",
    # 67: "a horse with a glasses",
    # 68: "a bear with a glasses",
    # 69: "a monkey and a red car",
    # 70: "a elephant with a bow",
    # 71: "a frog and a purple balloon",
    # 72: "a mouse with a bow",
    # 73: "a bird with a crown",
    # 74: "a turtle and a yellow bowl",
    # 75: "a rabbit and a gray chair",
    # 76: "a dog and a black apple",
    # 77: "a rabbit and a white bench",
    # 78: "a lion and a yellow clock",
    # 79: "a turtle and a gray backpack",
    # 80: "a elephant and a green balloon",
    # 81: "a monkey and a orange apple",
    # 82: "a lion and a red car",
    # 83: "a lion with a crown",
    # 84: "a bird and a purple bench",
    # 85: "a rabbit and a orange backpack",
    # 86: "a rabbit and a orange apple",
    # 87: "a monkey and a green bowl",
    # 88: "a frog and a red suitcase",
    # 89: "a monkey and a green balloon",
    # 90: "a cat with a glasses",
    # 91: "a bear and a blue clock",
    # 92: "a cat and a gray bench",
    # 93: "a bear with a crown",
    # 94: "a lion with a bow",
    # 95: "a bear and a red balloon",
    # 96: "a bird and a black backpack",
    # 97: "a horse and a pink balloon",
    # 98: "a turtle and a yellow car",
    # 99: "a lion with a glasses",
    # 100: "a cat and a yellow balloon",
    101: "a horse and a yellow clock",
    102: "a dog with a glasses",
    103: "a horse and a blue backpack",
    104: "a frog with a bow",
    105: "a elephant with a glasses",
    106: "a mouse and a red bench",
    107: "a bird and a brown balloon",
    108: "a monkey and a yellow backpack",
    109: "a turtle and a pink balloon",
    110: "a cat and a red apple",
    111: "a monkey and a brown bench",
    112: "a rabbit with a glasses",
    113: "a bear and a gray bench",
    114: "a turtle and a blue clock",
    115: "a monkey and a blue chair",
    116: "a turtle and a blue chair",
    117: "a dog with a bow",
    118: "a elephant and a black chair",
    119: "a mouse and a purple chair",
    120: "a bear and a white car",
    121: "a lion and a black backpack",
    122: "a dog with a crown",
    123: "a horse and a green apple",
    124: "a dog and a gray clock",
    125: "a dog and a purple car",
    126: "a dog and a gray bowl",
    127: "a monkey with a bow",
    128: "a mouse and a blue clock",
    129: "a bird and a black bowl",
    130: "a horse and a white car",
    131: "a mouse and a pink apple",
    132: "a bear and a orange backpack",
    133: "a elephant and a yellow clock",
    134: "a bird and a green chair",
    135: "a mouse and a black balloon",
    136: "a turtle and a white bench",
    137: "a bird with a bow",
    138: "a turtle with a crown",
    139: "a bird and a yellow car",
    140: "a frog and a orange car",
    141: "a dog and a pink bench",
    142: "a frog with a crown",
    143: "a frog and a green bowl",
    144: "a frog and a pink bench",
    145: "a horse with a bow",
    146: "a bird and a yellow apple",
    147: "a monkey with a crown",
    148: "a cat and a blue backpack",
    149: "a turtle and a pink apple",
    150: "a dog and a orange chair",
    151: "a horse and a green suitcase",
    152: "a elephant with a crown",
    153: "a monkey and a orange suitcase",
    154: "a turtle and a orange suitcase",
    155: "a lion and a gray apple",
    156: "a mouse with a crown",
    157: "a mouse with a glasses",
    158: "a horse and a brown bowl",
    159: "a monkey and a yellow clock",
    160: "a turtle with a bow",
    161: "a dog and a brown backpack",
    162: "a cat and a purple bowl",
    163: "a lion and a white bench",
    164: "a rabbit and a blue bowl",
    165: "a lion and a brown balloon",
    166: "a horse and a pink chair",
    167: "a elephant and a green bench",
    168: "a rabbit and a white balloon",
    169: "a elephant and a pink backpack",
    170: "a lion and a orange suitcase",
    171: "a elephant and a orange apple",
    172: "a elephant and a green suitcase",
    173: "a horse with a crown",
    174: "a bear with a bow",
    175: "a rabbit and a yellow suitcase",
    176: "a horse and a blue bench",
    177: "a dog and a green suitcase",
    178: "a mouse and a red car",
    179: "a cat and a black chair",
    180: "a bear and a red suitcase",
    181: "a rabbit and a gray clock",
    182: "a bear and a pink apple",
    183: "a lion and a white chair",
    184: "a rabbit with a crown",
    185: "a mouse and a purple bowl",
    186: "a frog and a black apple",
    187: "a rabbit with a bow",
    188: "a mouse and a pink suitcase",
    189: "a lion and a pink bowl",
    190: "a frog and a black chair",
    191: "a frog and a green clock",
    192: "a bear and a white chair",
    193: "a elephant and a brown car",
    194: "a turtle with a glasses",
    195: "a cat and a black suitcase",
    196: "a cat and a yellow car",
    197: "a frog and a yellow backpack",
    198: "a bird and a black suitcase",
    199: "a cat with a crown",
    200: "a rabbit and a yellow car",
    201: "a cat with a bow",
    202: "a bird and a white clock",
    203: "a cat and a green clock",
    204: "a bear and a purple bowl",
    205: "a monkey with a glasses",
    206: "a frog with a glasses",
    207: "a elephant and a green bowl",
    208: "a bird with a glasses",
    209: "a dog and a blue balloon",
    210: "a mouse and a brown backpack",
    211: "a pink crown and a purple bow",
    212: "a blue clock and a blue apple",
    213: "a blue balloon and a orange bench",
    214: "a pink crown and a red chair",
    215: "a orange chair and a blue clock",
    216: "a purple bowl and a black bench",
    217: "a green glasses and a black crown",
    218: "a purple chair and a red bow",
    219: "a yellow glasses and a black car",
    220: "a orange backpack and a purple car",
    221: "a white balloon and a white apple",
    222: "a brown suitcase and a black clock",
    223: "a yellow backpack and a purple chair",
    224: "a gray backpack and a green clock",
    225: "a blue crown and a red balloon",
    226: "a gray suitcase and a black bowl",
    227: "a brown balloon and a pink car",
    228: "a black backpack and a green bow",
    229: "a blue balloon and a blue bow",
    230: "a white bow and a white car",
    231: "a orange bowl and a purple apple",
    232: "a brown chair and a white bench",
    233: "a purple crown and a blue suitcase",
    234: "a yellow bow and a orange bench",
    235: "a yellow glasses and a brown bow",
    236: "a red glasses and a red suitcase",
    237: "a pink bow and a gray apple",
    238: "a gray crown and a white clock",
    239: "a black car and a white clock",
    240: "a brown bowl and a green clock",
    241: "a green backpack and a yellow crown",
    242: "a orange glasses and a pink clock",
    243: "a purple chair and a orange bowl",
    244: "a orange suitcase and a brown bench",
    245: "a white glasses and a orange balloon",
    246: "a yellow backpack and a gray apple",
    247: "a green bench and a red apple",
    248: "a gray backpack and a yellow glasses",
    249: "a green glasses and a yellow chair",
    250: "a white glasses and a gray apple",
    251: "a gray suitcase and a brown bow",
    252: "a white car and a black bowl",
    253: "a purple car and a pink apple",
    254: "a gray crown and a purple apple",
    255: "a orange car and a red bench",
    256: "a red suitcase and a blue apple",
    257: "a red backpack and a yellow bowl",
    258: "a red bench and a yellow clock",
    259: "a black backpack and a pink balloon",
    260: "a blue suitcase and a gray balloon",
    261: "a yellow glasses and a gray bowl",
    262: "a white suitcase and a white chair",
    263: "a purple crown and a blue bench",
    264: "a yellow bow and a pink bowl",
    265: "a green backpack and a brown suitcase",
    266: "a green glasses and a black bench",
    267: "a white bow and a black clock",
    268: "a red crown and a black bowl",
    269: "a green chair and a purple car",
    270: "a white chair and a gray balloon",
    271: "a pink chair and a gray apple",
    272: "a yellow suitcase and a yellow car",
    273: "a green backpack and a purple bench",
    274: "a black crown and a red car",
    275: "a green balloon and a pink bowl",
    276: "a purple balloon and a white clock"
}




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_seed", type=int, default=11)
    parser.add_argument("--start_idx", type=int, required=True)
    parser.add_argument("--end_idx", type=int, required=True)
    args = parser.parse_args()

    main(
        base_seed=args.base_seed,
        start_idx=args.start_idx,
        end_idx=args.end_idx
    )
