import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import time
from collections import defaultdict
from pathlib import Path


METRICS = ("hps", "clip", "imagereward", "pickscore")
FIELDS = ("dataset", "prompt_id", "prompt", "model", "method", "rank", "seed", "image_path", "base_seed")
WEIGHTS = ("hps_checkpoint", "hps_backbone", "clip_model", "pickscore_model", "pickscore_processor",
           "imagereward_checkpoint", "imagereward_config", "bert_tokenizer")
WEIGHT_HELP = {
    "hps_checkpoint": "Local HPS_v2.1_compressed.pt or HPS_v2.1.pt file, or its directory",
    "hps_backbone": "Local LAION ViT-H-14 open_clip_pytorch_model.bin file, or its directory",
    "clip_model": "Local openai/clip-vit-large-patch14 model and processor directory",
    "pickscore_model": "Local yuvalkirstain/PickScore_v1 model directory",
    "pickscore_processor": "Local laion/CLIP-ViT-H-14-laion2B-s32B-b79K processor directory",
    "imagereward_checkpoint": "Local ImageReward.pt file, or its directory",
    "imagereward_config": "Local med_config.json file, or its directory",
    "bert_tokenizer": "Local bert-base-uncased tokenizer directory",
}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path, fields, rows):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path, required):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not set(required).issubset(fields):
            raise ValueError(f"Invalid CSV columns in {path}: {fields}")
        rows = list(reader)
    if not rows or any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Empty or malformed CSV: {path}")
    return fields, rows


def read_manifest(path):
    path = Path(path).resolve()
    fields, rows = read_csv(path, FIELDS)
    if {"score", "metric"}.intersection(fields):
        raise ValueError("The image manifest must not contain score or metric columns")
    seen_keys, seen_paths, groups = set(), set(), defaultdict(list)
    for row in rows:
        if not row["prompt"].strip() or row["method"] not in ("abss", "random"):
            raise ValueError(f"Invalid prompt or method: {row}")
        for field in ("prompt_id", "rank", "seed", "base_seed"):
            if str(int(row[field])) != row[field]:
                raise ValueError(f"Noncanonical integer {field}: {row[field]}")
        key = tuple(row[field] for field in ("model", "dataset", "prompt_id", "method", "rank"))
        relative = Path(row["image_path"])
        if relative.is_absolute():
            raise ValueError(f"image_path must be relative to the manifest: {relative}")
        image = (path.parent / relative).resolve()
        image.relative_to(path.parent)
        if not image.is_file():
            raise FileNotFoundError(image)
        if key in seen_keys or image in seen_paths:
            raise ValueError(f"Duplicate manifest image: {key}, {image}")
        seen_keys.add(key)
        seen_paths.add(image)
        groups[tuple(row[field] for field in ("model", "dataset", "prompt_id"))].append(row)
    for key, group in groups.items():
        if len({(row["prompt"], row["base_seed"]) for row in group}) != 1:
            raise ValueError(f"Inconsistent prompt or base seed: {key}")
        for method in ("abss", "random"):
            subset = [row for row in group if row["method"] == method]
            if len(subset) != 3 or {row["rank"] for row in subset} != {"1", "2", "3"} or len({row["seed"] for row in subset}) != 3:
                raise ValueError(f"Expected exactly three distinct ranked {method} seeds: {key}")
    return fields, rows


def select_groups(rows, model=None, dataset=None, limit=None):
    groups = defaultdict(list)
    for row in rows:
        if (model is None or row["model"] == model) and (dataset is None or row["dataset"] == dataset):
            groups[tuple(row[field] for field in ("model", "dataset", "prompt_id"))].append(row)
    if not groups:
        raise ValueError("No prompt groups match the requested filters")
    strata = defaultdict(list)
    for key in sorted(groups, key=lambda key: (key[0], key[1], int(key[2]))):
        strata[key[:2]].append(sorted(groups[key], key=lambda row: (row["method"], int(row["rank"]))))
    strata_keys = sorted(strata, key=lambda key: (key[1], key[0]))
    selected = [strata[key][index] for index in range(max(map(len, strata.values())))
                for key in strata_keys if index < len(strata[key])]
    return selected if limit is None else selected[:limit]


def prepare_output(path):
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}. Choose a new output directory.")
    path.mkdir(parents=True, exist_ok=True)
    with (path / "run.lock").open("x", encoding="utf-8") as stream:
        stream.write("This directory belongs to one evaluation run.\n")
    return path


def versions():
    result = {"python": platform.python_version()}
    for name in ("torch", "torchvision", "transformers", "hpsv2", "image-reward", "numpy", "Pillow"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def score_manifest(args, scorer_factory=None):
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("batch-size and limit must be positive")
    manifest = args.manifest.resolve()
    fields, rows = read_manifest(manifest)
    groups = select_groups(rows, args.model, args.dataset, args.limit)
    output = prepare_output(args.output)
    started = time.perf_counter()
    configuration = {
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "image_path_base": str(manifest.parent),
        "metric": args.metric,
        "images": sum(map(len, groups)),
        "prompt_groups": len(groups),
        "versions": versions(),
    }
    write_json(output / "config.json", configuration)
    status = {"complete": False, "scored_images": 0, "expected_images": configuration["images"]}
    write_json(output / "status.json", status)
    if scorer_factory is None:
        if __package__:
            from .metrics import create_scorer
        else:
            from metrics import create_scorer
        scorer_factory = create_scorer
    scorer = scorer_factory(args.metric, weights_root=args.weights_root, device=args.device,
                            **{key: getattr(args, key) for key in WEIGHTS})
    configuration["scorer"] = scorer.configuration
    write_json(output / "config.json", configuration)
    scored = []
    for group in groups:
        for offset in range(0, len(group), args.batch_size):
            batch = group[offset:offset + args.batch_size]
            paths = [str((manifest.parent / row["image_path"]).resolve()) for row in batch]
            values = list(scorer.score(paths, group[0]["prompt"]))
            if len(values) != len(batch):
                raise ValueError(f"Scorer returned {len(values)} values for {len(batch)} images")
            values = [float(value) for value in values]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Nonfinite {args.metric} scores for {paths}: {values}")
            scored.extend({**row, "metric": args.metric, "score": value} for row, value in zip(batch, values))
            write_csv(output / "scores.csv", [*fields, "metric", "score"], scored)
            status.update(scored_images=len(scored), elapsed_seconds=time.perf_counter() - started)
            write_json(output / "status.json", status)
        print(f"[{args.metric}] {group[0]['model']}/{group[0]['dataset']}/prompt_{group[0]['prompt_id']} "
              f"{len(scored)}/{configuration['images']} images", flush=True)
    status.update(complete=True, elapsed_seconds=time.perf_counter() - started)
    write_json(output / "status.json", status)
    return status


def parser():
    command = argparse.ArgumentParser(
        description="Score all three ABSS and random images for each prompt using local weights only",
        epilog="Model directories may be local Hugging Face exports, cache repository directories, or explicit "
               "snapshot directories. Explicit weight arguments override --weights-root for that component.",
    )
    command.add_argument("--manifest", type=Path, required=True, help="Generation manifest.csv; image paths are relative to it")
    command.add_argument("--output", type=Path, required=True, help="New or empty directory for this metric's scores and metadata")
    command.add_argument("--metric", choices=METRICS, required=True)
    command.add_argument(
        "--weights-root", type=Path,
        help="Shared directory containing models--xswu--HPSv2, models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K, "
             "models--openai--clip-vit-large-patch14, models--yuvalkirstain--PickScore_v1, "
             "ImageReward (ImageReward.pt and med_config.json), and models--bert-base-uncased",
    )
    for name in WEIGHTS:
        command.add_argument("--" + name.replace("_", "-"), type=Path, help=WEIGHT_HELP[name])
    command.add_argument("--model", choices=("flux", "hunyuan"))
    command.add_argument("--dataset")
    command.add_argument("--device", default="cuda")
    command.add_argument("--batch-size", type=int, default=4)
    command.add_argument("--limit", type=int, help="Preflight prompt groups (six images each), interleaved across datasets and models")
    return command


if __name__ == "__main__":
    print(json.dumps(score_manifest(parser().parse_args()), indent=2))
