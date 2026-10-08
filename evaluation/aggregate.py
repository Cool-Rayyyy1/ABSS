import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean

if __package__:
    from .score import METRICS, prepare_output, read_csv, read_manifest, write_csv, write_json
else:
    from score import METRICS, prepare_output, read_csv, read_manifest, write_csv, write_json


def aggregate(manifest, score_paths, output):
    manifest = Path(manifest).resolve()
    fields, images = read_manifest(manifest)
    expected = {row["image_path"]: row for row in images}
    files = []
    for path in map(Path, score_paths):
        files.extend(sorted(path.rglob("scores.csv")) if path.is_dir() else [path])
    files = [path.resolve() for path in files]
    if not files or len(files) != len(set(files)):
        raise ValueError("No score files or duplicate score file arguments")
    scores = {}
    for path in files:
        _, rows = read_csv(path, [*fields, "metric", "score"])
        for row in rows:
            key = (row["metric"], row["image_path"])
            if key[0] not in METRICS or key[1] not in expected:
                raise ValueError(f"Unexpected score entry in {path}: {key}")
            if key in scores:
                raise ValueError(f"Duplicate score entry: {key}")
            if any(row[field] != expected[key[1]][field] for field in fields):
                raise ValueError(f"Score metadata differs from original manifest: {key}")
            value = float(row["score"])
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite score: {key}")
            scores[key] = {**expected[key[1]], "metric": key[0], "score": value}
    missing = [(metric, image) for metric in METRICS for image in expected if (metric, image) not in scores]
    if missing:
        raise ValueError(f"Missing {len(missing)} of {len(expected) * len(METRICS)} scores; first entries: {missing[:5]}")
    groups = defaultdict(lambda: defaultdict(list))
    for row in scores.values():
        key = tuple(row[field] for field in ("model", "dataset", "metric", "prompt_id"))
        groups[key][row["method"]].append(row)
    per_prompt = []
    for key in sorted(groups, key=lambda key: (*key[:3], int(key[3]))):
        methods = groups[key]
        abss = fmean(row["score"] for row in methods["abss"])
        random = fmean(row["score"] for row in methods["random"])
        per_prompt.append({
            "model": key[0], "dataset": key[1], "metric": key[2], "prompt_id": key[3],
            "prompt": methods["abss"][0]["prompt"], "base_seed": methods["abss"][0]["base_seed"],
            "n_abss": len(methods["abss"]), "n_random": len(methods["random"]),
            "abss_mean": abss, "random_mean": random, "delta": abss - random,
        })
    summaries = defaultdict(list)
    for row in per_prompt:
        summaries[(row["model"], row["dataset"], row["metric"])].append(row)
        summaries[(row["model"], "all", row["metric"])].append(row)
    comparisons = []
    for key, rows in sorted(summaries.items()):
        abss, random = (fmean(row[field] for row in rows) for field in ("abss_mean", "random_mean"))
        comparisons.append({"model": key[0], "dataset": key[1], "metric": key[2], "n_prompts": len(rows),
                            "abss_mean": abss, "random_mean": random, "delta": abss - random})
    output = prepare_output(output)
    ordered = [scores[(metric, row["image_path"])] for metric in METRICS for row in images]
    write_csv(output / "scores.csv", [*fields, "metric", "score"], ordered)
    write_csv(output / "per_prompt.csv", list(per_prompt[0]), per_prompt)
    write_csv(output / "summary.csv", list(comparisons[0]), comparisons)
    report = {
        "complete": True,
        "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "image_path_base": str(manifest.parent),
        "score_files": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        "images": len(images),
        "score_count": len(scores),
        "aggregation": "Mean of all three seeds per prompt and method, then equal-weight mean across prompts. "
                       "Dataset 'all' pools prompts across datasets; it does not average dataset means.",
        "rows": comparisons,
    }
    write_json(output / "summary.json", report)
    return report


if __name__ == "__main__":
    command = argparse.ArgumentParser(description="Validate complete four-metric coverage and compare ABSS with random")
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--scores", type=Path, nargs="+", required=True, help="Score CSV files or directories containing scores.csv")
    command.add_argument("--output", type=Path, required=True)
    args = command.parse_args()
    report = aggregate(args.manifest, args.scores, args.output)
    print(json.dumps(report["rows"], indent=2))
