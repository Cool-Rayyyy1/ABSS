# INITNO prompts and core tokens

The JSON files preserve INITNO prompt IDs **1–276**. The default benchmark runs **101–276** (176 prompts).

| File | Contents |
| --- | --- |
| [initno.prompts.json](initno.prompts.json) | Prompt ID → prompt text |
| [initno.core_tokens.json](initno.core_tokens.json) | Prompt ID → core-token annotations |

JSON prompt IDs are strings. Annotation positions are **1-based word positions**, not tokenizer IDs. Words are split on whitespace, lowercased, stripped of leading/trailing non-word characters, and empty words are omitted. The runner maps these positions to each model's tokenizer. ABSS scores the `entity` field; other annotation fields are preserved.

Use `--prompts /path/to/prompts.json --core-tokens /path/to/core_tokens.json` for custom inputs with the same format, and `--start-idx` / `--end-idx` to select prompt IDs.

Prompt source: [InitNO](https://github.com/xiefan-guo/initno).
