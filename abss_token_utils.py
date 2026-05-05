"""
BOS-less → CLIP 77-space helpers for InitNO initno_core_token_index_dict.json (abss.py).
"""
from __future__ import annotations


def tokenize_77space_bosless(tokenizer, prompt: str, max_length: int = 77):
    enc = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    ids77 = enc["input_ids"][0].tolist()
    toks77 = tokenizer.convert_ids_to_tokens(ids77)
    special_ids = set(tokenizer.all_special_ids)
    content_pos_1based = []
    content_toks = []
    for j0, tid in enumerate(ids77):
        if tid in special_ids:
            continue
        content_pos_1based.append(j0 + 1)
        content_toks.append(toks77[j0])
    return ids77, toks77, content_pos_1based, content_toks


def bosless_to_77_1based(tokenizer, prompt: str, idxs_bosless_1based):
    _, _, content_pos_1based, content_toks = tokenize_77space_bosless(tokenizer, prompt, max_length=77)
    out = []
    bad = []
    for x in idxs_bosless_1based or []:
        if not isinstance(x, int):
            continue
        if 1 <= x <= len(content_pos_1based):
            out.append(content_pos_1based[x - 1])
        else:
            bad.append(x)
    return out, bad, content_pos_1based, content_toks


def debug_print_prompt_alignment(tokenizer, prompt: str, token_idx_raw: dict, p_idx: int, max_show: int = 40):
    ids77, toks77, content_pos_1based, content_toks = tokenize_77space_bosless(tokenizer, prompt, max_length=77)
    print("\n" + "=" * 110)
    print(f"[DEBUG][p{p_idx:03d}] Prompt: {prompt}")
    print("-" * 110)
    print("[DEBUG] 77-space token table (1-based):")
    for i1, t in enumerate(toks77[:max_show], start=1):
        print(f"  {i1:>3d}: {t}")
    if len(toks77) > max_show:
        print(f"  ... (showing {max_show}/77)")
    print("-" * 110)
    print("[DEBUG] BOS-less content token table:")
    for k1, (pos77, tok) in enumerate(list(zip(content_pos_1based, content_toks))[:max_show], start=1):
        print(f"  bosless_idx={k1:>3d} -> 77_pos={pos77:>3d} token={tok}")
    if len(content_toks) > max_show:
        print(f"  ... (showing {max_show}/{len(content_toks)})")
    print("-" * 110)

    def show_group(name, idxs_bosless):
        idxs_bosless = [x for x in (idxs_bosless or []) if isinstance(x, int)]
        print(f"[DEBUG] group='{name}' JSON(BOS-less 1-based) indices={idxs_bosless}")
        if not idxs_bosless:
            return
        mapped_77, bad, _, _ = bosless_to_77_1based(tokenizer, prompt, idxs_bosless)
        for x in idxs_bosless[:50]:
            if 1 <= x <= len(content_toks):
                tok_bosless = content_toks[x - 1]
            else:
                tok_bosless = "(out of range)"
            if 1 <= x <= len(content_pos_1based):
                pos77 = content_pos_1based[x - 1]
                tok77 = toks77[pos77 - 1]
            else:
                pos77 = None
                tok77 = "(out of range)"
            print(f"    bosless_idx={x:>3d} -> bosless_token={tok_bosless:<18s} | 77_pos={str(pos77):>3s} -> 77_token={tok77}")
        if bad:
            print(f"    [WARN] out-of-range BOS-less indices: {bad}")
        print(f"    mapped 77-space (1-based) indices for '{name}': {mapped_77}")

    print("[DEBUG] RAW (from JSON, BOS-less 1-based):")
    for k in ["entity", "adjective", "verb", "other"]:
        show_group(k, token_idx_raw.get(k, []))
        print("-" * 110)
    print("=" * 110 + "\n")


def map_token_groups_bosless_to_77(tokenizer, prompt: str, token_idx_raw: dict):
    token_idx_77 = {}
    for k, v in token_idx_raw.items():
        if not isinstance(v, list):
            token_idx_77[k] = v
            continue
        mapped, bad, _, _ = bosless_to_77_1based(tokenizer, prompt, v)
        if bad:
            print(f"[WARN] prompt has out-of-range indices in group '{k}': {bad}")
        seen = set()
        dedup = []
        for t in mapped:
            if t not in seen:
                seen.add(t)
                dedup.append(t)
        token_idx_77[k] = dedup
    return token_idx_77
