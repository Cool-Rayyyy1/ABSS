# Prompt datasets and core-token annotations

| Dataset | Preserved prompt IDs | Original evaluation subset | Prompt file | Annotation file |
| --- | --- | --- | --- | --- |
| INITNO | 1–276 | 101–276 | `initno.prompts.json` | `initno.core_tokens.json` |
| DrawBench | 1–200 | 101–200 | `drawbench.prompts.json` | `drawbench.core_tokens.json` |
| Pick | 1–100 | 51–100 | `pick.prompts.json` | `pick.core_tokens.json` |

Prompt files map each original numeric ID, serialized as a string, to its prompt text. Annotation files map the same IDs to the original dictionaries. All original fields are preserved, including `entity`, `adjective`, `verb`, `other`, and, where present, `article` and `non-entity`.

Each annotation position is a **1-based position in the normalized words of the prompt**. The original code splits on whitespace, lowercases each word, removes leading and trailing non-word characters, and omits empty words. The runner converts the selected words into the model's tokenizer positions. Annotation positions are not tokenizer IDs. ABSS scores only the `entity` field.

These files were extracted without importing or executing the original model scripts. The source bundles in `attention-map-diffusers_flux_huanyuanDiT_new/prompt_datasets/` exactly match the corresponding bundles in `attention-map-diffusers_flux_huanyuanDiT/prompt_datasets/`. INITNO and Pick prompts match the combined original FLUX script subsets; DrawBench matches except for prompt 86. Source paths, SHA-256 fingerprints, evaluation ranges, and discrepancies are recorded in `metadata.json`.

DrawBench prompt 86 uses “A long curved banana…” in the Hunyuan bundles and “A long curved fruit…” in the original FLUX script. The shared prompt JSON preserves the Hunyuan wording. `metadata.json` supplies the original FLUX wording in `model_prompt_overrides.flux.drawbench.86`, allowing the runner to preserve the original input for each model.

Two original Pick annotations contain out-of-range positions in fields unused by entity scoring: prompt 69 has `article: [10]` for a seven-word prompt, and prompt 73 includes `22` and `23` in `non-entity` for a 21-word prompt. These values are preserved and documented in `metadata.json`; all `entity` positions are valid, and all prompts have entity annotations.
