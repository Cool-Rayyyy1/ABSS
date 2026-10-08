# Third-party notices and provenance

ABSS contains adaptations of attention capture and diffusion inference code from the projects below. Their copyright and license notices remain applicable to those portions. This file does not assign a new license to upstream code, model checkpoints, prompt datasets, or ABSS-specific contributions.

## Attention Map Diffusers

Upstream: [wooyeolBaek/attention-map-diffusers](https://github.com/wooyeolBaek/attention-map-diffusers).

Copyright (c) 2023 Wooyeol Baek.

The original attention-map-diffusers repositories carry the MIT License. The complete original notice is preserved in [licenses/attention-map-diffusers-MIT.txt](licenses/attention-map-diffusers-MIT.txt). It was copied byte-for-byte from the original `attention-map-diffusers_flux/LICENSE` and has SHA-256 `7883c2e7b1c80b9cfe6b3150d7baa0e4f6ed16d97e30d7ffa621e26b3aecf838`.

The immediate sources for this ABSS consolidation were the existing `attention-map-diffusers_flux`, `attention-map-diffusers_flux_huanyuanDiT`, and `attention-map-diffusers_flux_huanyuanDiT_new` experiment repositories. They share the upstream MIT notice. In particular, the joint-attention computation in `abss/backends/flux.py` is adapted from the FLUX attention processor in `attention_map_diffusers/modules.py`; Hunyuan attention capture follows the local `attention_map_diffusers/hunyuan_attn.py` implementation.

## Hugging Face Diffusers

Upstream: [huggingface/diffusers](https://github.com/huggingface/diffusers). The runtime dependency is pinned to version 0.31.0.

The diffusion pipelines and attention processors used as the basis for the backend adaptations are licensed under the Apache License, Version 2.0. The full license is preserved in [licenses/diffusers-Apache-2.0.txt](licenses/diffusers-Apache-2.0.txt), copied byte-for-byte from the existing `diffusers/LICENSE`, with SHA-256 `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`.

The relevant source notices are:

| Upstream source, version 0.31.0 | Copyright notice |
| --- | --- |
| [FLUX pipeline](https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/flux/pipeline_flux.py) | Copyright 2024 Black Forest Labs and The HuggingFace Team. All rights reserved. |
| [HunyuanDiT pipeline](https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/hunyuandit/pipeline_hunyuandit.py) | Copyright 2024 HunyuanDiT Authors and The HuggingFace Team. All rights reserved. |
| [Attention processors](https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/attention_processor.py) | Copyright 2024 The HuggingFace Team. All rights reserved. |

`abss/backends/flux.py` and `abss/backends/hunyuan.py` are modified implementations. The ABSS changes expose attention scoring, capture the completed screening-step latents and scheduler state, and continue selected samples from that state. They reorganize the original experiment code into model backends with shared configuration and dataset loading. The original upstream code does not provide this ABSS runner or its selection workflow.

## Model checkpoints

Model checkpoints are obtained separately and retain their own terms. The source-code licenses above do not replace checkpoint licenses.

| Checkpoint | License identified in the original checkpoint model card |
| --- | --- |
| [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) | [FLUX.1 [dev] Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md); the local checkpoint license snapshot is version 1.1.1. |
| [HunyuanDiT-v1.2-Diffusers](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers) | [Tencent Hunyuan Community License](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/blob/main/LICENSE.txt), as linked by the checkpoint model card. |

Review the terms supplied with the checkpoint version you obtain. Other installed runtime packages also retain their own license notices.

## Evaluation dependencies

Evaluation uses [HPS v2](https://github.com/tgxs002/HPSv2) with its v2.1 checkpoint, [OpenAI CLIP](https://github.com/openai/CLIP), [ImageReward](https://github.com/THUDM/ImageReward), and [PickScore](https://github.com/yuvalkirstain/PickScore). Their implementations are installed as dependencies; evaluation checkpoints are supplied separately. These projects and checkpoints retain their own license terms.

## Prompt datasets and annotations

The six prompt and annotation JSON files are extracted from the original experiment bundles. Their immediate source paths, exact fingerprints, preserved IDs, and known discrepancies are recorded in [datasets/metadata.json](datasets/metadata.json). The complete preserved collections contain 276 INITNO prompts, 200 DrawBench prompts, and 100 Pick prompts. See [datasets/README.md](datasets/README.md) for indexing and model-specific wording.

The existing InitNO README identifies the work *InitNO: Boosting Text-to-Image Diffusion Models via Initial Noise Optimization* by Xiefan Guo, Jinlin Liu, Miaomiao Cui, Jiankai Li, Hongyu Yang, and Di Huang, CVPR 2024, and links its [project page](https://xiefan-guo.github.io/initno) and [paper](https://arxiv.org/abs/2404.04650). That README does not establish the precise upstream provenance or dataset license for the 276 prompts packaged here.

The inspected original documentation does not identify the exact upstream record or dataset-specific redistribution terms for the embedded DrawBench and Pick collections. The label “Pick” alone does not establish that these prompts are Pick-a-Pic or another named dataset. Core-token dictionaries are preserved from the existing experiments; their authorship and separate license were not stated in the inspected files.

These provenance and dataset-license details remain to be confirmed by the project authors before the public release. The MIT and Apache software notices above are not asserted as licenses for these prompt collections or annotations.
