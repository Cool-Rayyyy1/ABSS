import os

import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage

from diffusers.models.attention_processor import (
    AttnProcessor,
    AttnProcessor2_0,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    JointAttnProcessor2_0,
    FluxAttnProcessor2_0,
)

from .modules import *


# ============================================================
# Global storage for attention maps (compatible with legacy format)
# attn_maps: {timestep: {layer_name: attn_map}}
# ============================================================
attn_maps = {}

# ============================================================
# Global capture/probe controls
# These are set per pipeline call (e.g., in FluxPipeline_call).
# ============================================================
CAPTURE_ENABLED = False
CAPTURE_STEP_IDX = None       # int
CAPTURE_BLOCK_NAME = None     # str, e.g. "transformer_blocks.9.attn"
CURRENT_STEP_IDX = -1         # current denoising step index
PROBE_RESULT = {}             # single captured result for scoring


def hook_function(layer_name: str, detach: bool = True):
    """
    Forward hook to capture a single attention map at a specific (step, block).
    Expects attention processors to expose:
      - processor.attn_map
      - processor.timestep
    """

    def _forward_hook(module, _input, _output):
        # If processor does not provide attention map, nothing to do.
        processor = getattr(module, "processor", None)
        if processor is None or not hasattr(processor, "attn_map"):
            return

        # If capture is disabled, delete cached attention map to avoid memory accumulation.
        if not CAPTURE_ENABLED:
            try:
                del processor.attn_map
            except Exception:
                pass
            return

        step_idx = CURRENT_STEP_IDX
        cap_step = CAPTURE_STEP_IDX
        cap_block = CAPTURE_BLOCK_NAME

        # Step filtering
        if cap_step is not None and step_idx != cap_step:
            try:
                del processor.attn_map
            except Exception:
                pass
            return

        # Block filtering
        if cap_block is not None and layer_name != cap_block:
            try:
                del processor.attn_map
            except Exception:
                pass
            return

        timestep = getattr(processor, "timestep", None)
        attn = processor.attn_map  # (batch, heads, ..., text_len)

        attn_store = attn.detach().cpu() if detach else attn

        # Keep legacy structure: {timestep: {layer_name: attn_map}}
        attn_maps.clear()
        attn_maps[float(timestep) if timestep is not None else -1.0] = {layer_name: attn_store}

        # Single-shot output for scoring
        global PROBE_RESULT
        PROBE_RESULT = {
            "step_idx": int(step_idx),
            "timestep": float(timestep) if timestep is not None else None,
            "block": layer_name,
            "attn_map": attn_store,
        }

        # Free processor cache
        try:
            del processor.attn_map
        except Exception:
            pass

    return _forward_hook


def _enable_store_attn_map(processor) -> bool:
    """
    Mark attention processors to store attention maps.
    Returns True if successful.
    """
    supported = (
        AttnProcessor,
        AttnProcessor2_0,
        LoRAAttnProcessor,
        LoRAAttnProcessor2_0,
        JointAttnProcessor2_0,
        FluxAttnProcessor2_0,
    )
    if isinstance(processor, supported):
        try:
            processor.store_attn_map = True
            return True
        except Exception:
            return False
    return False


def register_cross_attention_hook(model, capture_hook_fn, target_name_suffix: str):
    """
    Register forward hooks for modules whose qualified name ends with `target_name_suffix`.
    Typical suffix examples:
      - "attn2" for SD1/SDXL UNet cross-attn
      - "attn"  for SD3/Flux transformer blocks
    """
    for name, module in model.named_modules():
        if not name.endswith(target_name_suffix):
            continue

        processor = getattr(module, "processor", None)
        if processor is not None:
            _enable_store_attn_map(processor)
            try:
                processor._hook_name = name
            except Exception:
                pass

        module.register_forward_hook(capture_hook_fn(name))

    return model


def replace_call_method_for_unet(model):
    """
    Monkey-patch forward methods for SD1/SDXL UNet blocks to expose attention maps.
    Requires corresponding forward implementations from .modules.
    """
    if model.__class__.__name__ == "UNet2DConditionModel":
        from diffusers.models.unets import UNet2DConditionModel
        model.forward = UNet2DConditionModelForward.__get__(model, UNet2DConditionModel)

    for _, layer in model.named_children():
        if layer.__class__.__name__ == "Transformer2DModel":
            from diffusers.models import Transformer2DModel
            layer.forward = Transformer2DModelForward.__get__(layer, Transformer2DModel)

        elif layer.__class__.__name__ == "BasicTransformerBlock":
            from diffusers.models.attention import BasicTransformerBlock
            layer.forward = BasicTransformerBlockForward.__get__(layer, BasicTransformerBlock)

        replace_call_method_for_unet(layer)

    return model


def replace_call_method_for_sd3(model):
    """
    Monkey-patch forward methods for SD3 transformer blocks to expose attention maps.
    Requires corresponding forward implementations from .modules.
    """
    if model.__class__.__name__ == "SD3Transformer2DModel":
        from diffusers.models.transformers import SD3Transformer2DModel
        model.forward = SD3Transformer2DModelForward.__get__(model, SD3Transformer2DModel)

    for _, layer in model.named_children():
        if layer.__class__.__name__ == "JointTransformerBlock":
            from diffusers.models.attention import JointTransformerBlock
            layer.forward = JointTransformerBlockForward.__get__(layer, JointTransformerBlock)

        replace_call_method_for_sd3(layer)

    return model


def replace_call_method_for_flux(model):
    """
    Monkey-patch forward methods for Flux transformer blocks to expose attention maps.
    Requires corresponding forward implementations from .modules.
    """
    if model.__class__.__name__ == "FluxTransformer2DModel":
        from diffusers.models.transformers import FluxTransformer2DModel
        model.forward = FluxTransformer2DModelForward.__get__(model, FluxTransformer2DModel)

    for _, layer in model.named_children():
        if layer.__class__.__name__ == "FluxTransformerBlock":
            from diffusers.models.transformers.transformer_flux import FluxTransformerBlock
            layer.forward = FluxTransformerBlockForward.__get__(layer, FluxTransformerBlock)

        replace_call_method_for_flux(layer)

    return model


def init_pipeline(pipeline):
    """
    Initialize a diffusers pipeline to support attention probing.

    This function monkey-patches:
      - Several attention processor __call__ implementations (from .modules)
      - FluxPipeline.__call__ for probe-step/block control (from .modules)
      - Model forward methods to route/store attention maps

    Note: This modifies global class methods; call once per process.
    """
    AttnProcessor.__call__ = attn_call
    AttnProcessor2_0.__call__ = attn_call2_0
    LoRAAttnProcessor.__call__ = lora_attn_call
    LoRAAttnProcessor2_0.__call__ = lora_attn_call2_0

    if "transformer" in vars(pipeline).keys():
        # SD3 and Flux use `pipeline.transformer`
        if pipeline.transformer.__class__.__name__ == "SD3Transformer2DModel":
            JointAttnProcessor2_0.__call__ = joint_attn_call2_0
            pipeline.transformer = register_cross_attention_hook(
                pipeline.transformer, hook_function, "attn"
            )
            pipeline.transformer = replace_call_method_for_sd3(pipeline.transformer)

        elif pipeline.transformer.__class__.__name__ == "FluxTransformer2DModel":
            from diffusers import FluxPipeline
            FluxAttnProcessor2_0.__call__ = flux_attn_call2_0
            FluxPipeline.__call__ = FluxPipeline_call

            pipeline.transformer = register_cross_attention_hook(
                pipeline.transformer, hook_function, "attn"
            )
            pipeline.transformer = replace_call_method_for_flux(pipeline.transformer)

    else:
        # SD1/SDXL use `pipeline.unet`
        if pipeline.unet.__class__.__name__ == "UNet2DConditionModel":
            pipeline.unet = register_cross_attention_hook(
                pipeline.unet, hook_function, "attn2"
            )
            pipeline.unet = replace_call_method_for_unet(pipeline.unet)

    return pipeline


# ============================================================
# Visualization helpers
# ============================================================
def process_token(token: str, start_of_word: bool):
    """
    Format BPE tokens into a filename-friendly representation.
    """
    if "</w>" in token:
        token = token.replace("</w>", "")
        if start_of_word:
            token = "<" + token + ">"
        else:
            token = "-" + token + ">"
            start_of_word = True
    elif token not in ["<|startoftext|>", "<|endoftext|>"]:
        if start_of_word:
            token = "<" + token + "-"
            start_of_word = False
        else:
            token = "-" + token + "-"
    return token, start_of_word


def save_attention_image(attn_map, tokens, batch_dir: str, to_pil: ToPILImage):
    """
    Save per-token attention maps as images under batch_dir.
    """
    start_of_word = True
    n = min(len(tokens), attn_map.shape[0])
    for i in range(n):
        token, start_of_word = process_token(tokens[i], start_of_word)
        to_pil(attn_map[i].to(torch.float32)).save(os.path.join(batch_dir, f"{i}-{token}.png"))


def save_attention_maps(attn_maps_dict, tokenizer, prompts, base_dir="attn_maps", unconditional=True):
    """
    Save attention maps to images.
    Assumes `attn_maps_dict` uses legacy format: {timestep: {layer_name: attn_map}}
    """
    to_pil = ToPILImage()
    os.makedirs(base_dir, exist_ok=True)

    token_ids = tokenizer(prompts)["input_ids"]
    token_ids = token_ids if token_ids and isinstance(token_ids[0], list) else [token_ids]
    total_tokens = [tokenizer.convert_ids_to_tokens(token_id) for token_id in token_ids]

    if not attn_maps_dict:
        return

    # Infer target resolution from the first entry
    first_attn = list(list(attn_maps_dict.values())[0].values())[0]
    total_attn_map = first_attn.sum(1)
    if unconditional:
        total_attn_map = total_attn_map.chunk(2)[1]
    total_attn_map = total_attn_map.permute(0, 3, 1, 2)
    total_attn_map = torch.zeros_like(total_attn_map)
    total_attn_map_shape = total_attn_map.shape[-2:]
    total_attn_map_number = 0

    for timestep, layers in attn_maps_dict.items():
        timestep_dir = os.path.join(base_dir, f"{timestep}")
        os.makedirs(timestep_dir, exist_ok=True)

        for layer, attn_map in layers.items():
            layer_dir = os.path.join(timestep_dir, f"{layer}")
            os.makedirs(layer_dir, exist_ok=True)

            attn_map = attn_map.sum(1).squeeze(1).permute(0, 3, 1, 2)
            if unconditional:
                attn_map = attn_map.chunk(2)[1]

            resized = F.interpolate(attn_map, size=total_attn_map_shape, mode="bilinear", align_corners=False)
            total_attn_map += resized
            total_attn_map_number += 1

            for batch, (tokens, attn) in enumerate(zip(total_tokens, attn_map)):
                batch_dir = os.path.join(layer_dir, f"batch-{batch}")
                os.makedirs(batch_dir, exist_ok=True)
                save_attention_image(attn, tokens, batch_dir, to_pil)

    if total_attn_map_number > 0:
        total_attn_map /= float(total_attn_map_number)

    for batch, (attn_map, tokens) in enumerate(zip(total_attn_map, total_tokens)):
        batch_dir = os.path.join(base_dir, f"batch-{batch}")
        os.makedirs(batch_dir, exist_ok=True)
        save_attention_image(attn_map, tokens, batch_dir, to_pil)
