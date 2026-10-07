"""Copyright 2024 Black Forest Labs and The HuggingFace Team.

Derived from diffusers 0.31.0 (Apache-2.0) and attention-map-diffusers (MIT).
Modified for ABSS resumable screening. See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
from diffusers import FluxPipeline
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps

from abss.config import RunConfig
from abss.state import ScreeningResult, ScreeningState
from abss.tokens import map_entity_tokens


class FluxJointAttentionProcessor:
    def __init__(self):
        self.capture = False
        self.token_indices = []
        self.score = None

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None
    ):
        if encoder_hidden_states is None:
            raise ValueError("FLUX ABSS joint attention requires text hidden states.")
        if attention_mask is not None:
            raise ValueError("The original FLUX ABSS attention does not use an attention mask.")
        batch_size = hidden_states.shape[0]
        text_length = encoder_hidden_states.shape[1]
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        head_dim = key.shape[-1] // attn.heads

        def split_heads(tensor):
            return tensor.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        query, key, value = map(split_heads, (query, key, value))
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        text_query = split_heads(attn.add_q_proj(encoder_hidden_states))
        text_key = split_heads(attn.add_k_proj(encoder_hidden_states))
        text_value = split_heads(attn.add_v_proj(encoder_hidden_states))
        if attn.norm_added_q is not None:
            text_query = attn.norm_added_q(text_query)
        if attn.norm_added_k is not None:
            text_key = attn.norm_added_k(text_key)
        query = torch.cat([text_query, query], dim=2)
        key = torch.cat([text_key, key], dim=2)
        value = torch.cat([text_value, value], dim=2)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)
        weights = query @ key.transpose(-2, -1) * (1 / math.sqrt(query.shape[-1]))
        weights = torch.softmax(weights, dim=-1)
        hidden_states = weights @ value
        if self.capture:
            indices = [index for index in self.token_indices if 0 <= index < text_length]
            attention = weights[:, :, text_length:, :text_length].detach().cpu()
            self.score = float(attention[..., indices].mean().item()) if indices else 0.0
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        encoder_hidden_states = hidden_states[:, :text_length]
        hidden_states = hidden_states[:, text_length:]
        hidden_states = attn.to_out[1](attn.to_out[0](hidden_states))
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        return hidden_states, encoder_hidden_states


class FluxBackend:
    def __init__(self, config: RunConfig, pipeline=None):
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype)
        if len(config.probe_blocks) != 1:
            raise ValueError("FLUX ABSS uses exactly one joint attention block.")
        if config.offload not in ("original", "cpu", "none"):
            raise ValueError(f"Unsupported offload mode: {config.offload}")
        self.pipe = pipeline or FluxPipeline.from_pretrained(
            config.model_path, torch_dtype=self.dtype, local_files_only=config.local_files_only
        )
        self.pipe.set_progress_bar_config(disable=True)
        for module in (self.pipe.transformer, self.pipe.text_encoder, self.pipe.text_encoder_2, self.pipe.vae):
            module.eval()
        self.pipe.text_encoder.to(self.device, dtype=self.dtype)
        if config.offload == "none":
            self.pipe.text_encoder_2.to(self.device, dtype=self.dtype)
        else:
            self.pipe.text_encoder_2.to("cpu", dtype=torch.float32)
        model_device = "cpu" if config.offload == "cpu" else self.device
        self.pipe.transformer.to(model_device, dtype=self.dtype)
        self.pipe.vae.to(model_device, dtype=self.dtype)
        self.pipe.enable_vae_slicing()
        self.processors = {}
        for index, block in enumerate(self.pipe.transformer.transformer_blocks):
            processor = FluxJointAttentionProcessor()
            block.attn.set_processor(processor)
            self.processors[f"transformer_blocks.{index}.attn"] = processor
        probe_block = config.probe_blocks[0]
        if probe_block not in self.processors:
            raise ValueError(f"Unknown FLUX joint attention block: {probe_block}")
        self.probe_processor = self.processors[probe_block]
        self.prompt = None
        self.conditioning = None
        self.mapping = None

    @torch.inference_mode()
    def prepare(self, prompt: str, annotation: dict):
        self.prompt = prompt
        self.mapping = map_entity_tokens(
            prompt, annotation, self.pipe.tokenizer_2, self.config.max_sequence_length
        )
        self.probe_processor.token_indices = self.mapping["token_indices"]
        if self.config.offload == "cpu":
            self.pipe.transformer.to("cpu")
            self.pipe.vae.to("cpu")
            self.pipe.text_encoder.to(self.device)
        clip_inputs = self.pipe.tokenizer(
            [prompt], padding="max_length", max_length=self.pipe.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        clip_output = self.pipe.text_encoder(
            clip_inputs.input_ids.to(next(self.pipe.text_encoder.parameters()).device),
            output_hidden_states=False,
        )
        pooled = clip_output.pooler_output
        t5_inputs = self.pipe.tokenizer_2(
            [prompt], padding="max_length", max_length=self.config.max_sequence_length,
            truncation=True, return_tensors="pt"
        )
        prompt_embeds = self.pipe.text_encoder_2(
            t5_inputs.input_ids.to(next(self.pipe.text_encoder_2.parameters()).device),
            output_hidden_states=False,
        )[0]
        prompt_embeds = prompt_embeds.to(dtype=self.dtype).to(self.device)
        pooled = pooled.to(self.device, dtype=self.dtype)
        text_ids = torch.zeros(prompt_embeds.shape[1], 3, device=self.device, dtype=self.dtype)
        self.conditioning = (prompt_embeds, pooled, text_ids)
        if self.config.offload == "cpu":
            self.pipe.text_encoder.to("cpu")
        return self.mapping

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

    def _initial_state(self, seed):
        if self.conditioning is None:
            raise RuntimeError("Call prepare(prompt, annotation) before generating images.")
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        latents, image_ids = self.pipe.prepare_latents(
            1, self.pipe.transformer.config.in_channels // 4,
            self.config.height, self.config.width, self.dtype, self.device, generator, None,
        )
        scheduler = copy.deepcopy(self.pipe.scheduler)
        sigmas = np.linspace(1.0, 1 / self.config.num_inference_steps, self.config.num_inference_steps)
        mu = calculate_shift(
            latents.shape[1], scheduler.config.base_image_seq_len,
            scheduler.config.max_image_seq_len, scheduler.config.base_shift, scheduler.config.max_shift,
        )
        timesteps, _ = retrieve_timesteps(
            scheduler, self.config.num_inference_steps, self.device, None, sigmas, mu=mu
        )
        scheduler.sigmas = scheduler.sigmas.cpu()
        scheduler.timesteps = scheduler.timesteps.cpu()
        return latents, image_ids, timesteps, scheduler, generator

    def _denoise(self, latents, image_ids, timesteps, scheduler, start=0, stop=None, capture=False):
        if self.config.offload == "cpu":
            self.pipe.vae.to("cpu")
            self.pipe.transformer.to(self.device)
        prompt_embeds, pooled, text_ids = self.conditioning
        guidance = None
        if self.pipe.transformer.config.guidance_embeds:
            guidance = torch.full([latents.shape[0]], self.config.guidance_scale, device=self.device)
        stop = len(timesteps) if stop is None else stop
        try:
            for index in range(start, stop):
                timestep = timesteps[index]
                self.probe_processor.capture = capture and index == self.config.probe_step
                timestep_model = timestep.to(self.device).expand(latents.shape[0]).to(latents.dtype)
                noise_pred = self.pipe.transformer(
                    hidden_states=latents, timestep=timestep_model / 1000, guidance=guidance,
                    pooled_projections=pooled, encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids, img_ids=image_ids, return_dict=False,
                )[0]
                latents = scheduler.step(
                    noise_pred.cpu(), timestep.cpu(), latents.cpu(), return_dict=False
                )[0].to(self.device, dtype=self.dtype)
        finally:
            self.probe_processor.capture = False
        return latents

    @torch.inference_mode()
    def screen(self, seed: int):
        self.probe_processor.score = None
        latents, image_ids, timesteps, scheduler, generator = self._initial_state(seed)
        next_step = self.config.probe_step + 1
        if not 1 <= next_step <= len(timesteps):
            raise ValueError("probe_step must be a zero-based index in the full denoising schedule.")
        latents = self._denoise(latents, image_ids, timesteps, scheduler, stop=next_step, capture=True)
        if self.probe_processor.score is None:
            raise RuntimeError("FLUX attention was not captured at the requested block and step.")
        extra = {
            "backend": "flux",
            "prompt": self.prompt,
            "signature": self._signature(),
            "timesteps": timesteps.detach().cpu().clone(),
            "image_ids": image_ids.detach().cpu().clone(),
            "cpu_rng_state": torch.random.get_rng_state().clone(),
        }
        if self.device.type == "cuda":
            extra["cuda_rng_state"] = torch.cuda.get_rng_state(self.device).clone()
        state = ScreeningState(
            seed=int(seed), next_step=next_step, latents=latents.detach().cpu().clone(),
            scheduler=copy.deepcopy(scheduler), generator_state=generator.get_state().clone(), extra=extra,
        )
        return ScreeningResult(score=self.probe_processor.score, state=state)

    @torch.inference_mode()
    def resume(self, state: ScreeningState, output_type="pil"):
        if state.extra.get("backend") != "flux":
            raise ValueError("The screening state belongs to a different backend.")
        if self.prompt != state.extra["prompt"] or self.conditioning is None:
            raise ValueError("Prepare the checkpoint's prompt before resuming its denoising state.")
        if self._signature() != state.extra["signature"]:
            raise ValueError("Generation settings differ from the screening checkpoint.")
        torch.random.set_rng_state(state.extra["cpu_rng_state"].cpu())
        if self.device.type == "cuda" and "cuda_rng_state" in state.extra:
            torch.cuda.set_rng_state(state.extra["cuda_rng_state"].cpu(), self.device)
        latents = self._denoise(
            state.latents.to(self.device, dtype=self.dtype), state.extra["image_ids"].to(self.device),
            state.extra["timesteps"].to(self.device), copy.deepcopy(state.scheduler), start=state.next_step,
        )
        return self._decode(latents, output_type)

    @torch.inference_mode()
    def generate(self, seed: int, output_type="pil", reference=False):
        latents, image_ids, timesteps, scheduler, _ = self._initial_state(seed)
        latents = self._denoise(latents, image_ids, timesteps, scheduler)
        return self._decode(latents, output_type)

    def _decode(self, latents, output_type):
        if output_type == "latent":
            return latents.detach().cpu()
        if self.config.offload == "cpu":
            self.pipe.transformer.to("cpu")
            self.pipe.vae.to(self.device)
        latents = self.pipe._unpack_latents(
            latents, self.config.height, self.config.width, self.pipe.vae_scale_factor
        )
        latents = latents / self.pipe.vae.config.scaling_factor + self.pipe.vae.config.shift_factor
        decoded = self.pipe.vae.decode(latents, return_dict=False)[0]
        return self.pipe.image_processor.postprocess(decoded, output_type=output_type)[0]
