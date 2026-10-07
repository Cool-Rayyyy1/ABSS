import csv
import importlib.metadata
import json
import math
import platform
import random
import time
from pathlib import Path

from .data import sha256
from .export import ImageExporter
from .state import load_state, save_state


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def make_seed_pool(base_seed, pool_size, top_k):
    pool = random.Random(base_seed).sample(range(1_000_000), pool_size)
    baseline = random.Random(base_seed).sample(pool, top_k)
    return pool, baseline


def select_top_results(results, top_k):
    import numpy as np

    scores = np.asarray([result.score for result in results], dtype=np.float64)
    order = np.arange(len(results))[::-1][scores[::-1].argsort(kind="quicksort")][::-1]
    return [results[index] for index in order[:top_k]]


def environment():
    import torch

    versions = {}
    for package in ("torch", "diffusers", "transformers", "accelerate", "numpy", "Pillow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": platform.python_version(),
        "packages": versions,
        "cuda": torch.version.cuda,
        "gpu": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    }


def synchronize():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def verify_continuation(backend, state):
    import torch

    continued = backend.resume(state, output_type="latent").detach().cpu()
    uninterrupted = backend.generate(state.seed, output_type="latent", reference=True).detach().cpu()
    difference = (continued.float() - uninterrupted.float()).abs()
    result = {
        "seed": state.seed,
        "next_step": state.next_step,
        "exact_equal": bool(torch.equal(continued, uninterrupted)),
        "max_absolute_error": float(difference.max().item()),
        "mean_absolute_error": float(difference.mean().item()),
        "reference": "uninterrupted trajectory using the same attention-screening arithmetic",
    }
    if not torch.allclose(continued, uninterrupted, rtol=1e-5, atol=1e-5):
        raise RuntimeError(f"Continuation verification failed: {result}")
    return result


def run(config, dataset, description, output, random_baseline=True, save_checkpoints=True, verify_resume=False, backend=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}. Choose a new --output directory.")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("x", encoding="utf-8") as lock:
        lock.write("This output directory belongs to one ABSS run.\n")
    pool, baseline = make_seed_pool(config.base_seed, config.seed_pool_size, config.top_k)
    source_root = Path(__file__).resolve().parent
    configuration = {
        "config": config.to_dict(),
        "dataset": description,
        "random_baseline": random_baseline,
        "save_checkpoints": save_checkpoints,
        "verify_resume": verify_resume,
        "output_layout": {
            "images": "{method}/prompt_{id:03d}/img_seed{seed}.png",
            "metadata": "metadata/prompt_{id:03d}/",
            "manifests": ["manifest.csv", "manifest.jsonl"],
            "image_path_base": "run output directory",
            "rank_base": 1,
        },
        "environment": environment(),
        "source_sha256": {str(path.relative_to(source_root)): sha256(path) for path in sorted(source_root.rglob("*.py"))},
    }
    write_json(output / "run_config.json", configuration)
    write_json(output / "global_seed_pool.json", pool)
    write_json(output / "global_random_top_seeds.json", baseline)
    exporter = ImageExporter(output, description.get("name", "custom"), config.model, config.base_seed)
    if backend is None:
        from .backends import load_backend

        backend = load_backend(config)
    run_started = time.perf_counter()
    summaries = []
    for prompt_id, prompt, annotation in dataset:
        directory = output / "metadata" / f"prompt_{int(prompt_id):03d}"
        directory.mkdir(parents=True)
        mapping = backend.prepare(prompt, annotation)
        write_json(directory / "prompt.json", {"id": prompt_id, "prompt": prompt, "core_tokens": annotation})
        write_json(directory / "token_mapping.json", mapping)
        if mapping.get("misses"):
            print(f"[prompt {prompt_id}] unmapped entity words: {mapping['misses']}", flush=True)
        synchronize()
        started = time.perf_counter()
        results = []
        for seed in pool:
            result = backend.screen(seed)
            if not math.isfinite(result.score):
                raise RuntimeError(f"Non-finite score for prompt {prompt_id}, seed {seed}")
            if result.state.next_step != config.probe_step + 1:
                raise RuntimeError(f"Unexpected continuation step for prompt {prompt_id}, seed {seed}: {result.state.next_step}")
            results.append(result)
            print(f"[prompt {prompt_id}] seed={seed} score={result.score:.9g} next_step={result.state.next_step}", flush=True)
        synchronize()
        screening_seconds = time.perf_counter() - started
        selected = select_top_results(results, config.top_k)
        selected_seeds = [result.state.seed for result in selected]
        with (directory / "seed_scores.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["seed", "entity_mean", "selected", "next_step"])
            writer.writeheader()
            writer.writerows({"seed": result.state.seed, "entity_mean": result.score,
                             "selected": result.state.seed in selected_seeds, "next_step": result.state.next_step} for result in results)
        write_json(directory / "top_entity_seeds.json", selected_seeds)
        write_json(directory / "random_top_seeds.json", baseline)
        del results, result
        if save_checkpoints:
            for result in selected:
                save_state(result.state, directory / "checkpoints" / f"seed_{result.state.seed}.pt")
        if verify_resume:
            state = selected[0].state
            if save_checkpoints:
                state = load_state(directory / "checkpoints" / f"seed_{state.seed}.pt")
            check = verify_continuation(backend, state)
            check["disk_roundtrip"] = save_checkpoints
            write_json(directory / "resume_verification.json", check)
            print(f"[prompt {prompt_id}] continuation check: {check}", flush=True)
            del state
        synchronize()
        started = time.perf_counter()
        for rank, result in enumerate(selected, start=1):
            image = backend.resume(result.state)
            exporter.save(image, prompt_id, prompt, "abss", rank, result.state.seed)
        synchronize()
        continuation_seconds = time.perf_counter() - started
        del selected, result
        baseline_seconds = 0.0
        if random_baseline:
            started = time.perf_counter()
            for rank, seed in enumerate(baseline, start=1):
                image = backend.generate(seed)
                exporter.save(image, prompt_id, prompt, "random", rank, seed)
            synchronize()
            baseline_seconds = time.perf_counter() - started
        summary = {
            "prompt_id": prompt_id,
            "selected_seeds": selected_seeds,
            "random_seeds": baseline if random_baseline else [],
            "screening_seconds": screening_seconds,
            "continuation_seconds": continuation_seconds,
            "abss_seconds": screening_seconds + continuation_seconds,
            "baseline_seconds": baseline_seconds,
            "resume_next_step": config.probe_step + 1,
            "screening_steps_per_seed": config.probe_step + 1,
            "continuation_steps_per_selected_seed": config.num_inference_steps - config.probe_step - 1,
            "total_denoising_steps": config.num_inference_steps,
        }
        write_json(directory / "result.json", summary)
        summaries.append(summary)
        write_json(output / "summary.json", {"prompts": summaries, "wall_seconds": time.perf_counter() - run_started})
        print(f"[prompt {prompt_id}] selected={selected_seeds} saved={directory}", flush=True)
    return summaries
