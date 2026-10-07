import hashlib
import json
from pathlib import Path

from .tokens import split_prompt_words


DATASET_DIR = Path(__file__).resolve().parents[1] / "datasets"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_dataset(name, model, prompts_path=None, core_tokens_path=None, start_idx=None, end_idx=None):
    if prompts_path is not None and core_tokens_path is None:
        raise ValueError("Custom --prompts requires matching --core-tokens")
    metadata_path = DATASET_DIR / "metadata.json"
    metadata = read_json(metadata_path)
    specification = metadata["datasets"][name]
    default_prompts = prompts_path is None
    prompts_path = Path(prompts_path) if prompts_path else DATASET_DIR / specification["prompts_file"]
    core_tokens_path = Path(core_tokens_path) if core_tokens_path else DATASET_DIR / specification["core_tokens_file"]
    prompts = read_json(prompts_path)
    annotations = read_json(core_tokens_path)
    if not isinstance(prompts, dict) or not isinstance(annotations, dict):
        raise ValueError("Dataset JSON files must be objects keyed by prompt ID")
    overrides = metadata.get("model_prompt_overrides", {}).get(model, {}).get(name, {}) if default_prompts else {}
    prompts = {**prompts, **overrides}
    defaults = specification["evaluation_range"] if default_prompts else [min(map(int, prompts)), max(map(int, prompts))]
    start_idx = defaults[0] if start_idx is None else start_idx
    end_idx = defaults[1] if end_idx is None else end_idx
    if start_idx > end_idx:
        raise ValueError("start_idx must not exceed end_idx")
    selected = []
    for prompt_id in sorted(prompts, key=int):
        if not start_idx <= int(prompt_id) <= end_idx:
            continue
        prompt = prompts[prompt_id]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Empty or invalid prompt {prompt_id}")
        annotation = annotations.get(prompt_id)
        if not isinstance(annotation, dict):
            raise ValueError(f"Missing core-token annotation for prompt {prompt_id}")
        positions = annotation.get("entity")
        length = len(split_prompt_words(prompt))
        if not isinstance(positions, list) or not positions or any(
            type(position) is not int or not 1 <= position <= length for position in positions
        ):
            raise ValueError(f"Invalid entity positions for prompt {prompt_id}: {positions}")
        selected.append((str(prompt_id), prompt, annotation))
    if not selected:
        raise ValueError("No prompts in the selected range")
    description = {
        "name": name,
        "prompts_path": str(prompts_path.resolve()),
        "core_tokens_path": str(core_tokens_path.resolve()),
        "prompts_sha256": sha256(prompts_path),
        "core_tokens_sha256": sha256(core_tokens_path),
        "metadata_sha256": sha256(metadata_path),
        "start_idx": start_idx,
        "end_idx": end_idx,
        "prompt_ids": [row[0] for row in selected],
        "applied_prompt_overrides": {key: value for key, value in overrides.items() if start_idx <= int(key) <= end_idx},
    }
    return selected, description
