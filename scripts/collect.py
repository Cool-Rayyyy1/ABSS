import argparse
import csv
import io
import json
from collections import Counter
from pathlib import Path


def collect(root, require_images=None):
    root = Path(root).resolve()
    records = []
    seen = set()
    fields = ["dataset", "prompt_id", "prompt", "model", "method", "rank", "seed", "image_path", "base_seed"]
    for manifest in sorted(root.rglob("manifest.csv")):
        if manifest.parent == root:
            continue
        with manifest.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                image = (manifest.parent / row["image_path"]).resolve()
                image.relative_to(manifest.parent.resolve())
                relative_image = image.relative_to(root).as_posix()
                if not image.is_file():
                    raise FileNotFoundError(image)
                key = tuple(row[field] for field in ("model", "dataset", "prompt_id", "method", "rank"))
                if key in seen:
                    raise ValueError(f"Duplicate evaluation entry: {key}")
                seen.add(key)
                record = {field: row[field] for field in fields}
                record["image_path"] = relative_image
                for field in ("prompt_id", "rank", "seed", "base_seed"):
                    record[field] = int(record[field])
                records.append(record)
    if require_images is not None and len(records) != require_images:
        raise ValueError(f"Expected {require_images} images, found {len(records)}")
    records.sort(key=lambda row: (row["model"], row["dataset"], row["method"], row["prompt_id"], row["rank"]))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(records)
    for name, content in (
        ("manifest.csv", stream.getvalue()),
        ("manifest.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)),
    ):
        temporary = root / (name + ".tmp")
        temporary.write_text(content, encoding="utf-8", newline="")
        temporary.replace(root / name)
    return {"images": len(records), "groups": dict(Counter(
        f"{row['model']}/{row['dataset']}/{row['method']}" for row in records
    ))}


if __name__ == "__main__":
    command = argparse.ArgumentParser(description="Combine shard manifests; image paths are relative to the batch directory")
    command.add_argument("directory", type=Path)
    command.add_argument("--require-images", type=int)
    args = command.parse_args()
    print(json.dumps(collect(args.directory, args.require_images), indent=2))
