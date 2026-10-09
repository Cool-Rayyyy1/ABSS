"""Copyright 2024 HunyuanDiT Authors and The HuggingFace Team.

Derived from diffusers 0.31.0 (Apache-2.0) and attention-map-diffusers (MIT),
modified for resumable ABSS screening. See LICENSE.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import torch
from diffusers import HunyuanDiTPipeline
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.pipelines.hunyuandit.pipeline_hunyuandit import SUPPORTED_SHAPE, map_to_standard_shapes

from abss.config import RunConfig
from abss.state import ScreeningResult, ScreeningState
from abss.tokens import map_entity_tokens


def _copy_tensors(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device).clone()
    if isinstance(value, dict):
        return {key: _copy_tensors(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_copy_tensors(item, device) for item in value)
    if isinstance(value, list):
        return [_copy_tensors(item, device) for item in value]
    return value


def _map_scheduler_tensors(scheduler, transform):
    scheduler = deepcopy(scheduler)

    def visit(value, path):
        if isinstance(value, torch.Tensor):
            return transform(value, path)
        if isinstance(value, dict):
            return {key: visit(item, path + (key,)) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(visit(item, path + (index,)) for index, item in enumerate(value))
        if isinstance(value, list):
            return [visit(item, path + (index,)) for index, item in enumerate(value)]
        return value

    for name, value in vars(scheduler).items():
        if name != "_internal_dict":
            vars(scheduler)[name] = visit(value, (name,))
    return scheduler


def _snapshot_scheduler(scheduler):
    placements = {}

    def move(tensor, path):
        placements[path] = tensor.device.type
        return tensor.detach().cpu().clone()

    return _map_scheduler_tensors(scheduler, move), placements


def _restore_scheduler(scheduler, placements, device):
    return _map_scheduler_tensors(
        scheduler,
        lambda tensor, path: tensor.to(device="cpu" if placements[path] == "cpu" else device),
    )


@dataclass
class _Capture:
    probe_step: int
    blocks: tuple[str, ...]
    step: int = 0
    enabled: bool = False
    maps: dict[str, torch.Tensor] = field(default_factory=dict)


class HunyuanAttentionProcessor:
    def __init__(self, original, capture: _Capture, name: str):
        self.original = original
        self.capture = capture
        self.name = name

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        image_rotary_emb=None,
    ):
        if encoder_hidden_states is None or self.capture.step > self.capture.probe_step:
            return self.original(
                attn,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = encoder_hidden_states.shape
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        head_dim = key.shape[-1] // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            if not attn.is_cross_attention:
                key = apply_rotary_emb(key, image_rotary_emb)

        weights = query @ key.transpose(-2, -1) * (1 / query.shape[-1] ** 0.5)
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                weights = weights.masked_fill(~attention_mask, float("-inf"))
            else:
                weights = weights + attention_mask
        weights = torch.softmax(weights, dim=-1)
        if (
            self.capture.enabled
            and self.capture.step == self.capture.probe_step
            and self.name in self.capture.blocks
        ):
            self.capture.maps[self.name] = weights.detach().cpu()

        hidden_states = weights @ value
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


class HunyuanBackend:
    def __init__(self, config: RunConfig, pipeline=None):
        if config.max_sequence_length != 256:
            raise ValueError("HunyuanDiT uses exactly 256 T5 tokens and 77 BERT tokens.")
        if not 0 <= config.probe_step < config.num_inference_steps:
            raise ValueError("probe_step must be within the inference schedule.")
        self.config = config
        self.pipe = pipeline or HunyuanDiTPipeline.from_pretrained(
            config.model_path,
            torch_dtype=getattr(torch, config.dtype),
            local_files_only=config.local_files_only,
        )
        self.pipe.set_progress_bar_config(disable=True)
        if pipeline is None:
            if config.offload in ("original", "none"):
                self.pipe.to(config.device)
            elif config.offload == "cpu":
                self.pipe.enable_model_cpu_offload(device=config.device)
            else:
                raise ValueError(f"Unknown offload mode: {config.offload}")
        self.capture = _Capture(config.probe_step, tuple(config.probe_blocks))
        modules = dict(self.pipe.transformer.named_modules())
        missing = [name for name in self.capture.blocks if name not in modules or not name.endswith(".attn2")]
        if missing:
            raise ValueError(f"Unknown HunyuanDiT cross-attention blocks: {missing}")
        for name, module in modules.items():
            if name.endswith(".attn2"):
                module.set_processor(HunyuanAttentionProcessor(module.processor, self.capture, name))
        self.prompt = None
        self.mapping = None
        self.prompt_kwargs = {}

    def _signature(self):
        return {
            "model_path": self.config.model_path,
            "dtype": self.config.dtype,
            "num_inference_steps": self.config.num_inference_steps,
            "guidance_scale": self.config.guidance_scale,
            "height": self.config.height,
            "width": self.config.width,
            "max_sequence_length": self.config.max_sequence_length,
            "offload": self.config.offload,
            "probe_step": self.config.probe_step,
            "probe_blocks": tuple(self.config.probe_blocks),
        }

    @torch.no_grad()
    def prepare(self, prompt: str, annotation: dict) -> dict:
        self.prompt = prompt
        self.mapping = map_entity_tokens(prompt, annotation, self.pipe.tokenizer_2, 256)
        self.mapping["cross_attention_indices"] = [77 + index for index in self.mapping["token_indices"]]
        height, width = self.config.height, self.config.width
        if (height, width) not in SUPPORTED_SHAPE:
            width, height = map_to_standard_shapes(width, height)
        self.mapping["effective_height"] = int(height)
        self.mapping["effective_width"] = int(width)
        self.prompt_kwargs = {}
        for encoder_index, suffix, length in ((0, "", 77), (1, "_2", 256)):
            encoded = self.pipe.encode_prompt(
                prompt=prompt,
                device=self.pipe._execution_device,
                dtype=self.pipe.transformer.dtype,
                num_images_per_prompt=1,
                do_classifier_free_guidance=self.config.guidance_scale > 1,
                negative_prompt=None,
                max_sequence_length=length,
                text_encoder_index=encoder_index,
            )
            for name, value in zip(
                ("prompt_embeds", "negative_prompt_embeds", "prompt_attention_mask", "negative_prompt_attention_mask"),
                encoded,
            ):
                self.prompt_kwargs[name + suffix] = value
        return deepcopy(self.mapping)

    def _generator(self, seed: int):
        return torch.Generator(device=self.pipe._execution_device).manual_seed(int(seed))

    def _call_kwargs(self):
        if self.prompt is None:
            raise RuntimeError("Call prepare(prompt, annotation) before generation.")
        return {
            **self.prompt_kwargs,
            "num_inference_steps": self.config.num_inference_steps,
            "guidance_scale": self.config.guidance_scale,
            "height": self.config.height,
            "width": self.config.width,
        }

    @torch.no_grad()
    def screen(self, seed: int) -> ScreeningResult:
        generator = self._generator(seed)
        self.capture.step = 0
        self.capture.enabled = True
        self.capture.maps.clear()
        conditioning: dict[str, Any] = {}
        saved: dict[str, Any] = {}

        def capture_conditioning(module, positional, kwargs):
            if not conditioning:
                conditioning.update(_copy_tensors(kwargs, "cpu"))

        def stop_at_probe(pipe, step, timestep, callback_kwargs):
            self.capture.step = step + 1
            if step == self.config.probe_step:
                saved["latents"] = callback_kwargs["latents"].detach().cpu().clone()
                saved["scheduler"], saved["scheduler_placements"] = _snapshot_scheduler(pipe.scheduler)
                saved["generator_state"] = generator.get_state().clone()
                pipe._interrupt = True
            return callback_kwargs

        handle = self.pipe.transformer.register_forward_pre_hook(capture_conditioning, with_kwargs=True)
        try:
            self.pipe(
                **self._call_kwargs(),
                generator=generator,
                output_type="latent",
                callback_on_step_end=stop_at_probe,
            )
        finally:
            handle.remove()
            self.capture.enabled = False
            self.pipe._interrupt = False
        if not saved:
            raise RuntimeError("The screening step was not reached.")
        missing = [name for name in self.capture.blocks if name not in self.capture.maps]
        if missing:
            raise RuntimeError(f"Attention was not captured for blocks: {missing}")
        maps = torch.stack([self.capture.maps[name] for name in self.capture.blocks], dim=0).mean(0)
        indices = [index for index in self.mapping["cross_attention_indices"] if 0 <= index < maps.shape[-1]]
        score = float(maps[..., indices].mean().item()) if indices else 0.0
        self.capture.maps.clear()
        state = ScreeningState(
            seed=int(seed),
            next_step=self.config.probe_step + 1,
            latents=saved["latents"],
            scheduler=saved["scheduler"],
            generator_state=saved["generator_state"],
            extra={
                "model": "hunyuan",
                "prompt": self.prompt,
                "conditioning": conditioning,
                "signature": self._signature(),
                "scheduler_placements": saved["scheduler_placements"],
                "generator_device_type": generator.device.type,
                "effective_height": saved["latents"].shape[-2] * self.pipe.vae_scale_factor,
                "effective_width": saved["latents"].shape[-1] * self.pipe.vae_scale_factor,
            },
        )
        return ScreeningResult(score=score, state=state)

    @torch.no_grad()
    def resume(self, state: ScreeningState, output_type: str = "pil"):
        if state.extra.get("model") != "hunyuan":
            raise ValueError("This state does not belong to the HunyuanDiT backend.")
        if state.extra.get("prompt") != self.prompt:
            raise ValueError("Prepare the checkpoint's prompt before resuming.")
        if state.extra.get("signature") != self._signature():
            raise ValueError("Generation settings differ from the screening checkpoint.")
        if state.next_step != self.config.probe_step + 1 or not 0 < state.next_step <= len(state.scheduler.timesteps):
            raise ValueError("The checkpoint has an invalid continuation boundary.")
        device = self.pipe._execution_device
        if state.extra.get("generator_device_type") != torch.device(device).type:
            raise ValueError("The checkpoint's random generator requires the original device type.")
        scheduler = _restore_scheduler(state.scheduler, state.extra["scheduler_placements"], device)
        self.pipe.scheduler = scheduler
        generator = self._generator(state.seed)
        generator.set_state(state.generator_state.cpu())
        latents = state.latents.to(device=device)
        conditioning = _copy_tensors(state.extra["conditioning"], device)
        self.capture.enabled = False
        self.capture.step = state.next_step
        guidance = self.config.guidance_scale
        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator, 0.0)
        for step in range(state.next_step, len(scheduler.timesteps)):
            timestep = scheduler.timesteps[step]
            latent_model_input = torch.cat([latents] * 2) if guidance > 1 else latents
            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)
            t_expand = torch.tensor([timestep] * latent_model_input.shape[0], device=device).to(
                dtype=latent_model_input.dtype
            )
            noise_pred = self.pipe.transformer(latent_model_input, t_expand, **conditioning)[0]
            noise_pred, _ = noise_pred.chunk(2, dim=1)
            if guidance > 1:
                unconditioned, conditioned = noise_pred.chunk(2)
                noise_pred = unconditioned + guidance * (conditioned - unconditioned)
            latents = scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs, return_dict=False)[0]
        return self._decode(latents, output_type)

    @torch.no_grad()
    def generate(self, seed: int, output_type: str = "pil", reference: bool = False):
        self.capture.step = 0 if reference else self.config.probe_step + 1
        self.capture.enabled = False

        def advance(pipe, step, timestep, callback_kwargs):
            if reference:
                self.capture.step = step + 1
            return callback_kwargs

        result = self.pipe(
            **self._call_kwargs(),
            generator=self._generator(seed),
            output_type=output_type,
            callback_on_step_end=advance,
        ).images
        return result if output_type == "latent" else result[0]

    def _decode(self, latents, output_type):
        if output_type == "latent":
            self.pipe.maybe_free_model_hooks()
            return latents
        image = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor, return_dict=False)[0]
        image, flags = self.pipe.run_safety_checker(image, self.pipe._execution_device, self.pipe.transformer.dtype)
        denormalize = [True] * image.shape[0] if flags is None else [not flag for flag in flags]
        images = self.pipe.image_processor.postprocess(image, output_type=output_type, do_denormalize=denormalize)
        self.pipe.maybe_free_model_hooks()
        return images[0]
