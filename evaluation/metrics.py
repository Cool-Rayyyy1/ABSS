import math
from pathlib import Path
from unittest.mock import patch


def _resolve(value, root, child, filenames):
    if value is None:
        if root is None:
            raise ValueError(f"Provide an explicit local path for {child}, or --weights-root.")
        path = Path(root).expanduser() / child
    else:
        path = Path(value).expanduser()
    if path.is_file():
        if not filenames:
            raise NotADirectoryError(path)
        return path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Local evaluation weights not found: {path}")
    reference = path / "refs" / "main"
    if reference.is_file():
        path = path / "snapshots" / reference.read_text(encoding="utf-8").strip()
    elif (path / "snapshots").is_dir():
        snapshots = sorted(item for item in (path / "snapshots").iterdir() if item.is_dir())
        if len(snapshots) != 1:
            raise ValueError(f"Choose an explicit snapshot directory for {path}.")
        path = snapshots[0]
    if not path.is_dir():
        raise FileNotFoundError(f"Local snapshot not found: {path}")
    if filenames:
        for filename in filenames:
            candidate = path / filename
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(f"Expected one of {filenames} in {path}.")
    return path.resolve()


def _images(paths):
    from PIL import Image

    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


class Scorer:
    def __init__(self, function, configuration, model):
        self.function = function
        self.configuration = configuration
        self.model = model

    def score(self, image_paths, prompt):
        import torch

        paths = [str(Path(path)) for path in image_paths]
        if not paths:
            return []
        with torch.inference_mode():
            scores = [float(value) for value in self.function(paths, prompt)]
        if len(scores) != len(paths) or not all(math.isfinite(value) for value in scores):
            raise RuntimeError(f"Invalid {self.configuration['metric']} scores: {scores}")
        return scores


def _clip_scorer(metric, model_path, processor_path, device):
    from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor

    model_class = CLIPModel if metric == "clip" else AutoModel
    processor_class = CLIPProcessor if metric == "clip" else AutoProcessor
    processor = processor_class.from_pretrained(str(processor_path), local_files_only=True)
    model, loading = model_class.from_pretrained(
        str(model_path), local_files_only=True, output_loading_info=True
    )
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Incomplete {metric} checkpoint: {loading}")
    model = model.eval().to(device)

    def score(paths, prompt):
        image_inputs = processor(images=_images(paths), return_tensors="pt", padding=True).to(device)
        text_inputs = processor(
            text=[prompt], return_tensors="pt", padding=True, truncation=True, max_length=77
        ).to(device)
        image_features = model.get_image_features(**image_inputs)
        text_features = model.get_text_features(**text_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        values = text_features @ image_features.T
        if metric == "pickscore":
            values = model.logit_scale.exp() * values
        return values[0].detach().cpu().tolist()

    return Scorer(score, {
        "metric": metric, "model": str(model_path), "processor": str(processor_path),
        "device": str(device), "precision": "float32", "text_max_length": 77,
        "definition": "raw cosine similarity" if metric == "clip" else "exp(logit_scale) * cosine similarity; no softmax",
    }, model)


def _hps_scorer(checkpoint, backbone, device):
    import torch
    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

    model, _, preprocess = create_model_and_transforms(
        "ViT-H-14", str(backbone), precision="amp", device=device, jit=False,
        force_quick_gelu=False, force_custom_text=False, force_patch_dropout=False,
        force_image_size=None, pretrained_image=False, pretrained_hf=False,
        image_mean=None, image_std=None, light_augmentation=True, aug_cfg={}, output_dict=True,
        with_score_predictor=False, with_region_predictor=False,
    )
    state = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(state["state_dict"], strict=True)
    del state
    model = model.eval().to(device)
    tokenizer = get_tokenizer("ViT-H-14")

    def score(paths, prompt):
        values = []
        text = tokenizer([prompt]).to(device)
        for image in _images(paths):
            tensor = preprocess(image).unsqueeze(0).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(tensor, text)
                value = output["image_features"] @ output["text_features"].T
            values.append(value[0, 0].item())
        return values

    return Scorer(score, {
        "metric": "hps", "version": "v2.1", "checkpoint": str(checkpoint), "backbone": str(backbone),
        "device": str(device), "precision": "float16 autocast" if device.type == "cuda" else "float32",
        "definition": "normalized image/text feature dot product; no scaling",
    }, model)


def _imagereward_scorer(checkpoint, med_config, tokenizer_path, device):
    import torch
    from ImageReward.ImageReward import ImageReward
    from ImageReward.models.BLIP import blip_pretrain
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    with patch.object(blip_pretrain, "init_tokenizer", return_value=tokenizer):
        model = ImageReward(med_config=str(med_config), device=device)
    state = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(state, strict=True)
    del state
    model = model.eval().to(device)

    def score(paths, prompt):
        return [model.score(prompt, image) for image in _images(paths)]

    return Scorer(score, {
        "metric": "imagereward", "version": "ImageReward-v1.0", "checkpoint": str(checkpoint),
        "med_config": str(med_config), "tokenizer": str(tokenizer_path), "device": str(device),
        "precision": "float32", "text_max_length": 35,
        "definition": "official model.score: (reward - 0.16717362830052426) / 1.0333394966054072",
    }, model)


def create_scorer(metric, weights_root=None, device="cuda", hps_checkpoint=None, hps_backbone=None,
                  clip_model=None, pickscore_model=None, pickscore_processor=None,
                  imagereward_checkpoint=None, imagereward_config=None, bert_tokenizer=None):
    import torch

    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if metric == "clip":
        path = _resolve(clip_model, weights_root, "models--openai--clip-vit-large-patch14", ())
        return _clip_scorer(metric, path, path, device)
    if metric == "pickscore":
        model = _resolve(pickscore_model, weights_root, "models--yuvalkirstain--PickScore_v1", ())
        processor = _resolve(pickscore_processor, weights_root, "models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K", ())
        return _clip_scorer(metric, model, processor, device)
    if metric == "hps":
        checkpoint = _resolve(hps_checkpoint, weights_root, "models--xswu--HPSv2", ("HPS_v2.1_compressed.pt", "HPS_v2.1.pt"))
        backbone = _resolve(hps_backbone, weights_root, "models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K", ("open_clip_pytorch_model.bin",))
        return _hps_scorer(checkpoint, backbone, device)
    if metric == "imagereward":
        checkpoint = _resolve(imagereward_checkpoint, weights_root, "ImageReward", ("ImageReward.pt",))
        config = _resolve(imagereward_config, weights_root, "ImageReward", ("med_config.json",))
        tokenizer = _resolve(bert_tokenizer, weights_root, "models--bert-base-uncased", ())
        return _imagereward_scorer(checkpoint, config, tokenizer, device)
    raise ValueError(f"Unknown metric: {metric}")
