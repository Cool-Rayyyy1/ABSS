import argparse
from pathlib import Path

from abss.backends import load_backend
from abss.config import RunConfig
from abss.data import read_json
from abss.state import load_state


def main():
    parser = argparse.ArgumentParser(description="Continue a selected seed from an ABSS checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, default=None)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    run_config = args.run_config or args.checkpoint.resolve().parents[3] / "run_config.json"
    values = read_json(run_config)["config"]
    values["probe_blocks"] = tuple(values["probe_blocks"])
    config = RunConfig(**values)
    prompt_record = read_json(args.checkpoint.resolve().parents[1] / "prompt.json")
    backend = load_backend(config)
    backend.prepare(prompt_record["prompt"], prompt_record["core_tokens"])
    state = load_state(args.checkpoint)
    image = backend.resume(state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"Resumed seed {state.seed} from step index {state.next_step}: {args.output}")


if __name__ == "__main__":
    main()
