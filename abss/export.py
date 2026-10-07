import csv
import json
from pathlib import Path


class ImageExporter:
    fields = ("dataset", "prompt_id", "prompt", "model", "method", "rank", "seed", "image_path", "base_seed")

    def __init__(self, output, dataset, model, base_seed):
        self.output = Path(output)
        self.common = {"dataset": dataset, "model": model, "base_seed": base_seed}
        self.rows = []
        self._write_manifests()

    def save(self, image, prompt_id, prompt, method, rank, seed):
        if method not in ("abss", "random"):
            raise ValueError(f"Unknown generation method: {method}")
        path = self.output / method / f"prompt_{int(prompt_id):03d}" / f"img_seed{seed}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"Image already exists: {path}")
        temporary = path.with_name(f".{path.stem}.tmp.png")
        image.save(temporary)
        temporary.replace(path)
        self.rows.append({
            **self.common,
            "prompt_id": str(prompt_id),
            "prompt": prompt,
            "method": method,
            "rank": rank,
            "seed": seed,
            "image_path": path.relative_to(self.output).as_posix(),
        })
        self._write_manifests()
        return path

    def _write_manifests(self):
        csv_path = self.output / "manifest.csv"
        csv_temporary = csv_path.with_suffix(".csv.tmp")
        with csv_temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.fields)
            writer.writeheader()
            writer.writerows(self.rows)
        csv_temporary.replace(csv_path)
        jsonl_path = self.output / "manifest.jsonl"
        jsonl_temporary = jsonl_path.with_suffix(".jsonl.tmp")
        with jsonl_temporary.open("w", encoding="utf-8") as stream:
            for row in self.rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        jsonl_temporary.replace(jsonl_path)
