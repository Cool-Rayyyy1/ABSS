# ABSS — 跑 InitNO（SD1.4）

需要：InitNO 代码仓（能 `import initno`）、本目录下的 SD1.4 权重、`prompts_initno.json` 与 `initno_core_token_index_dict.json`。

**导出 JSON（在 initno1 根目录）：**

```bash
python scripts/export_abss_data.py
```

**运行（在 ABSS 目录）：**

```bash
export INITNO_REPO=/path/to/initno1   # 可选；默认 ../initno1
python abss.py --start_idx 101 --end_idx 110 --base_seed 11
```

**参数：**

| 参数 | 作用 |
|------|------|
| `--start_idx` / `--end_idx` | 要跑的 prompt id 区间（须在两份 JSON 里都有） |
| `--base_seed` | 与 `run_attention9_initno.py` 里一致的采样基准 seed |
| `--initno_repo` | InitNO 仓库根目录（覆盖 `INITNO_REPO`） |
| `--data_dir` | 放 JSON 的目录，默认本脚本所在目录 |
| `--sd_path` | SD1.4 目录，默认 `{initno_repo}/stable-diffusion-v1-4` |
| `--debug` / `--debug_every K` | 打印 BOS-less 与 77 token 对齐（调试用） |

结果目录：`当前工作目录/SmallPool10_top3/abss/test{base_seed}/`。

另附 **`prompts_drawbench.json`**、**`prompts_pick.json`**：仅 prompt 文本，给对齐文案用；**`abss.py` 不读这两个文件**。
