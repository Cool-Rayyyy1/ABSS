import argparse
import json
from pathlib import Path

from .config import RunConfig
from .data import DATASETS, load_dataset


def parser():
    command = argparse.ArgumentParser(description="ABSS: attention-based seed screening with cached continuation")
    command.add_argument("--model", choices=("flux", "hunyuan"), required=True)
    command.add_argument("--model-path", default=None)
    command.add_argument("--dataset", choices=DATASETS, default="initno")
    command.add_argument("--prompts", type=Path, default=None)
    command.add_argument("--core-tokens", type=Path, default=None)
    command.add_argument("--start-idx", type=int, default=None)
    command.add_argument("--end-idx", type=int, default=None)
    command.add_argument("--seed-pool-size", type=int, default=10)
    command.add_argument("--top-k", type=int, default=3)
    command.add_argument("--base-seed", type=int, default=11)
    screening = command.add_mutually_exclusive_group()
    screening.add_argument("--screening-steps", type=int, default=None, help="Zero-based screening index (default: 10, the 11th forward); completed steps are index + 1")
    screening.add_argument("--probe-step", type=int, default=None, help="Alias for --screening-steps: zero-based screening index")
    command.add_argument("--probe-blocks", default=None)
    command.add_argument("--num-inference-steps", type=int, default=50)
    command.add_argument("--guidance-scale", type=float, default=7.5)
    command.add_argument("--height", type=int, default=512)
    command.add_argument("--width", type=int, default=512)
    command.add_argument("--max-sequence-length", type=int, default=None)
    command.add_argument("--device", default="cuda")
    command.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    command.add_argument("--offload", choices=("original", "none", "cpu"), default="original")
    command.add_argument("--local-files-only", action="store_true")
    command.add_argument("--output", type=Path, default=None)
    command.add_argument("--random-baseline", action=argparse.BooleanOptionalAction, default=True)
    command.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    command.add_argument("--verify-resume", action="store_true")
    command.add_argument("--dry-run", action="store_true")
    return command


def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    probe_step = args.probe_step if args.probe_step is not None else (10 if args.screening_steps is None else args.screening_steps)
    try:
        config = RunConfig(
            model=args.model, model_path=args.model_path, device=args.device, dtype=args.dtype,
            num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale,
            height=args.height, width=args.width, probe_step=probe_step,
            probe_blocks=tuple(block.strip() for block in args.probe_blocks.split(",") if block.strip()) if args.probe_blocks else None,
            max_sequence_length=args.max_sequence_length, offload=args.offload,
            local_files_only=args.local_files_only, seed_pool_size=args.seed_pool_size,
            top_k=args.top_k, base_seed=args.base_seed,
        )
        dataset, description = load_dataset(
            args.dataset, args.model, args.prompts, args.core_tokens, args.start_idx, args.end_idx
        )
    except (ValueError, KeyError, FileNotFoundError) as error:
        command.error(str(error))
    output = args.output or Path("runs") / args.model / args.dataset / f"test_{args.base_seed}"
    print(json.dumps({"config": config.to_dict(), "screening_schedule": config.screening_schedule(), "dataset": description,
                      "output": str(output), "random_baseline": args.random_baseline,
                      "save_checkpoints": args.save_checkpoints}, indent=2), flush=True)
    if args.dry_run:
        return
    from .runner import run

    run(config, dataset, description, output, args.random_baseline, args.save_checkpoints, args.verify_resume)
