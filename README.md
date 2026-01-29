# ABSS InitNO (Flux) — How to Run

This repo contains `abss_initno.py`, which scores a seed pool using cross-attention at a specific denoising step/layer, then generates images for Top-K seeds and a random baseline.

## Requirements
- Python 3.10
- CUDA 11.8
- torch 2.2.2 + cu118
- diffusers 0.31.0
- transformers 4.46.3
- accelerate 1.1.1
- numpy, pandas, tqdm, pillow, sentencepiece

## Install (conda + pip)
```bash
conda create -n flux python=3.10 -y
conda activate flux

pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.2.2 torchvision==0.17.2
pip install diffusers==0.31.0 transformers==4.46.3 accelerate==1.1.1 peft==0.17.0
pip install numpy==1.26.4 pandas tqdm pillow sentencepiece regex pyyaml safetensors
Model
Set the model path/id inside abss_initno.py:

MODEL_ID = "./flux_1_dev"
Input file
Put this JSON next to abss_initno.py:

prompt_token_index_dict.json

Format (example):

{
  "101": { "entity": [2, 6] },
  "102": { "entity": [2, 5] }
}
entity uses 1-based word positions from whitespace-split prompt text.

Run
python abss_initno.py --start_idx 101 --end_idx 110 --base_seed 11
Args:

--start_idx / --end_idx: prompt id range (inclusive)

--base_seed: controls the global seed pool and random baseline

Outputs
Results are written under:

Flux_SmallPool{SEEDS_PER_PROMPT}_top{TOP_K}_new_attention12/test_{base_seed}/
Per prompt:

seed_scores.csv (seed, score)

top_entity_seeds.json (Top-K by attention score)

random_top_seeds.json (random Top-K)

topK_attn/*.png (generated images from Top-K attention seeds)

topK_random/*.png (generated images from random Top-K)