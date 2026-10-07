from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ScreeningState:
    seed: int
    next_step: int
    latents: Any
    scheduler: Any
    generator_state: Any
    extra: dict = field(default_factory=dict)


@dataclass
class ScreeningResult:
    score: float
    state: ScreeningState


def save_state(state: ScreeningState, path: Path):
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"format_version": 1, "state": state}, temporary)
    temporary.replace(path)


def load_state(path: Path) -> ScreeningState:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != 1 or not isinstance(payload.get("state"), ScreeningState):
        raise ValueError("Unsupported ABSS checkpoint")
    return payload["state"]
