from dataclasses import asdict, dataclass


MODEL_DEFAULTS = {
    "flux": {
        "model_path": "black-forest-labs/FLUX.1-dev",
        "probe_blocks": ("transformer_blocks.12.attn",),
        "max_sequence_length": 512,
    },
    "hunyuan": {
        "model_path": "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers",
        "probe_blocks": ("blocks.12.attn2",),
        "max_sequence_length": 256,
    },
}


@dataclass(frozen=True)
class RunConfig:
    model: str
    model_path: str | None = None
    device: str = "cuda"
    dtype: str = "float16"
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    height: int = 512
    width: int = 512
    probe_step: int = 9
    probe_blocks: tuple[str, ...] | None = None
    max_sequence_length: int | None = None
    offload: str = "original"
    local_files_only: bool = False
    seed_pool_size: int = 10
    top_k: int = 3
    base_seed: int = 11

    def __post_init__(self):
        if self.model not in MODEL_DEFAULTS:
            raise ValueError(f"Unknown model: {self.model}")
        for key, value in MODEL_DEFAULTS[self.model].items():
            if getattr(self, key) is None:
                object.__setattr__(self, key, value)
        if not 1 <= self.top_k <= self.seed_pool_size <= 1_000_000:
            raise ValueError("Require 1 <= top_k <= seed_pool_size <= 1000000")
        if not 0 <= self.probe_step < self.num_inference_steps:
            raise ValueError("probe_step must be in [0, num_inference_steps)")
        if self.height < 16 or self.width < 16 or self.height % 16 or self.width % 16:
            raise ValueError("height and width must be positive multiples of 16")
        if self.dtype not in ("float16", "bfloat16", "float32"):
            raise ValueError("Unsupported dtype")
        if self.offload not in ("original", "none", "cpu"):
            raise ValueError("Unsupported offload mode")
        if not self.probe_blocks:
            raise ValueError("At least one probe block is required")
        if self.model == "flux" and len(self.probe_blocks) != 1:
            raise ValueError("FLUX uses exactly one probe block")
        if not 1 <= self.max_sequence_length <= MODEL_DEFAULTS[self.model]["max_sequence_length"]:
            raise ValueError("max_sequence_length exceeds the model's supported text length")
        if self.model == "hunyuan" and self.max_sequence_length != 256:
            raise ValueError("HunyuanDiT uses exactly 256 T5 tokens")

    def to_dict(self):
        return asdict(self)
