import torch

from transformers import AutoImageProcessor, AutoModel
from torchvision.models import ViT_B_16_Weights, vit_b_16
from typing import Any

VIT_CACHE: dict[str, Any] = {}
DINO_CACHE: dict[str, Any] = {}


def get_vit() -> tuple[Any, Any]:

    if "model" not in VIT_CACHE:
        weights = ViT_B_16_Weights.DEFAULT
        model = vit_b_16(weights=weights)
        model.eval()
        VIT_CACHE["model"] = model
        VIT_CACHE["preprocess"] = weights.transforms()
        VIT_CACHE["weights"] = str(weights)
    return VIT_CACHE["model"], VIT_CACHE["preprocess"]

def get_dinov2() -> Any:

    if "model" not in DINO_CACHE:
        model_id = "facebook/dinov2-small"
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id, attn_implementation="eager")
        model.eval()
        DINO_CACHE["model"] = model
        DINO_CACHE["processor"] = processor
        DINO_CACHE["model_id"] = model_id
    return DINO_CACHE["model"], DINO_CACHE["processor"]