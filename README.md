# ABSS — InitNO data + `abss.py` runner

This directory is the **InitNO evaluation bundle**: JSON prompts and token labels, plus **`abss.py`** to reproduce the two-stage pipeline (InitNO seed scoring → Attend-and-Excite images) the same way as `run_attention9_initno.py` in the main repo.

**Paths:** Windows `D:\projects\ABSS` · WSL `/mnt/d/projects/ABSS`

---

## 1. What you need before running

| Requirement | Notes |
|-------------|--------|
| **InitNO code repo** | A checkout that contains the `initno` Python package (e.g. clone of this paper repo next to ABSS). |
| **GPU + CUDA** | Same as the original InitNO experiments. |
| **Stable Diffusion v1.4** | Local folder with the SD1.4 checkpoint (default: `{InitNO repo}/stable-diffusion-v1-4`). |
| **Python env** | Same dependencies as InitNO (`diffusers`, `torch`, etc.). |

`abss.py` **imports** `initno.pipelines...`, so it does **not** replace the InitNO codebase — it only supplies **data** (JSON) and a thin driver.

---

## 2. InitNO data files (what `abss.py` reads)

All paths below are under **`--data_dir`** (default: this folder).

| File | Role |
|------|------|
| **`prompts_initno.json`** | Map `prompt_id` → text string. Keys are stringified integers in JSON; IDs in the shipped export are **101–276** (176 prompts in the current `run_attention9_initno.py` snapshot). |
| **`initno_core_token_index_dict.json`** | **InitNO core-token** labels for each benchmark prompt: map `prompt_id` → groups (`entity`, `adjective`, `verb`, `other`, …). “Core” here means the **token-index groups** used for InitNO / Attend-and-Excite (not DrawBench/PICK word indices). Indices are **BOS-less 1-based**: the first *non-special* CLIP token in the padded 77-token sequence is **1**, then **2**, … — same file as `initno_core_token_index_dict.json` in the **initno1** repo root (loaded by `run_attention9_initno.py`). |

**Rule:** Every `prompt_id` you pass in `--start_idx` … `--end_idx` must appear in **both** JSON files. If an ID is missing from `initno_core_token_index_dict.json`, that prompt is skipped with a warning.

---

## 3. Refresh JSON from the InitNO repo

From the **initno1** (paper code) root:

```bash
python scripts/export_abss_data.py
```

This overwrites / merges into `ABSS/`:

- `prompts_drawbench.json`, `prompts_pick.json` — prompt text only (see §6).
- `prompts_initno.json`
- `initno_core_token_index_dict.json` — if ABSS already has a **larger** dict than `initno1/initno_core_token_index_dict.json`, the larger file is kept (so you do not accidentally shrink a full 276-entry file).

---

## 4. How to run `abss.py` (step by step)

**Step 1 — Set the InitNO repo path** so Python can import `initno`:

```bash
export INITNO_REPO=/mnt/d/projects/initno1    # Linux/WSL
# Windows PowerShell: $env:INITNO_REPO = "D:\projects\initno1"
```

If you omit this, the default is **`../initno1`** relative to the ABSS folder (sibling directory).

**Step 2 — Go to ABSS** (or stay anywhere and pass `--data_dir`):

```bash
cd /mnt/d/projects/ABSS
```

**Step 3 — Run** with a **contiguous prompt ID range** that exists in `prompts_initno.json`:

```bash
python abss.py --start_idx 101 --end_idx 110 --base_seed 11
```

**Optional flags:**

| Flag | Meaning |
|------|--------|
| `--initno_repo PATH` | Overrides `INITNO_REPO`. |
| `--data_dir PATH` | Folder containing `prompts_initno.json` and `initno_core_token_index_dict.json` (default: directory of `abss.py`). |
| `--sd_path PATH` | SD1.4 folder if not at `{initno_repo}/stable-diffusion-v1-4`. |
| `--base_seed N` | Same role as in `run_attention9_initno.py` for seed sampling. |
| `--debug` | Print BOS-less ↔ 77-token tables for some prompts. |
| `--debug_every K` | With `--debug`, print every *K* prompts in the loop (0-based index modulo *K*). |

**Step 4 — Outputs**

Results are written under your **current working directory**:

```text
SmallPool10_top3/abss/test{base_seed}/
  prompt_{id:03d}/
    seed_scores_entity.csv
    top50_entity_seeds.json
    attend_excite/
      img_seed*.jpg
      img_rand*.jpg
```

Same layout idea as the original `run_attention9_initno.py` runs, with a top-level folder name **`abss`** matching this script.

---

## 5. Helper module

**`abss_token_utils.py`** — maps BOS-less indices in `initno_core_token_index_dict.json` to **77-space 1-based** token positions for the SD1.x CLIP tokenizer. You normally do not run it alone; `abss.py` imports it.

---

## 6. DrawBench and PICK prompts (reference only)

We also ship the **exact prompt strings** we used for DrawBench-style and PICK-style lists as two standalone JSON files:

- **`prompts_drawbench.json`** — prompt IDs **1–200** and their text.
- **`prompts_pick.json`** — prompt IDs **1–100** in the current export (length matches `run_attention9_pick.py` in the repo).

These two files are **not** read by **`abss.py`**. They are provided so others can reuse the same **wording** in their own code or benchmarks. Full attention pipelines for those splits (with word-level token annotations) remain in the main repo: **`run_attention9_drawbench.py`** and **`run_attention9_pick.py`**.

---

## Summary

- **To run InitNO from this folder:** use **`abss.py`** + **`prompts_initno.json`** + **`initno_core_token_index_dict.json`** (InitNO **core-token** groups), with **`INITNO_REPO`** pointing at the InitNO codebase and SD1.4 available locally.  
- **DrawBench / PICK:** we publish **`prompts_drawbench.json`** and **`prompts_pick.json`** as **prompt-only** artifacts; they are documented here for transparency, not wired into `abss.py`.
