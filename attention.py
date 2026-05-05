import math
from functools import partial
from typing import Any

import torch
from fastapi import HTTPException
from PIL import Image

from utils import patch_features
from models import get_dinov2, get_vit


def rng_matrix(rows: int, cols: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, cols, generator=generator) / math.sqrt(max(cols, 1))


def simulation_attention(
    tokens: torch.Tensor, layer: int, head: int
) -> tuple[torch.Tensor, torch.Tensor]:
    state = tokens
    dim_in = tokens.shape[1]
    dim_hidden = 24
    layer_summaries = []
    selected_attention = None

    for layer_idx in range(8):
        heads = []
        for head_idx in range(6):
            seed = 1103 + layer_idx * 97 + head_idx * 17
            wq = rng_matrix(dim_hidden, state.shape[1], seed)
            wk = rng_matrix(dim_hidden, state.shape[1], seed + 1)
            q = state @ wq.T
            k = state @ wk.T
            scores = (q @ k.T) / math.sqrt(dim_hidden)

            distance_bias = distance_bias_matrix(
                state.shape[0], strength=0.15 + layer_idx * 0.04
            )
            scores = scores - distance_bias
            heads.append(torch.softmax(scores, dim=-1))

        layer_attention = torch.stack(heads)
        layer_summaries.append(layer_attention[:, 0, 1:].mean(dim=1))

        mix = layer_attention.mean(dim=0) @ state
        proj = rng_matrix(dim_in, state.shape[1], 2501 + layer_idx * 31)
        state = torch.tanh(mix @ proj.T + tokens * 0.35)

        if layer_idx == layer:
            selected_attention = layer_attention[min(head, layer_attention.shape[0] - 1)]

    if selected_attention is None:
        selected_attention = layer_attention[min(head, layer_attention.shape[0] - 1)]
    return selected_attention, torch.stack(layer_summaries)


def distance_bias_matrix(token_count: int, strength: float) -> torch.Tensor:
    patch_count = token_count - 1
    grid = int(math.sqrt(patch_count))
    bias = torch.zeros(token_count, token_count)
    if grid * grid != patch_count:
        return bias

    yy, xx = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1).float()
    dist = torch.cdist(coords, coords)
    dist = dist / (dist.max() + 1e-6)
    bias[1:, 1:] = dist * strength
    return bias


def forward_torchvision_attention(
    module: Any,
    capture: dict[str, list[torch.Tensor]],
    layer_idx: int,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    output, weights = module._original_forward(
        query,
        key,
        value,
        *args,
        need_weights=True,
        average_attn_weights=False,
        **{k: v for k, v in kwargs.items() if k not in {"need_weights", "average_attn_weights"}},
    )
    capture["layers"][layer_idx] = weights.detach().cpu().squeeze(0)
    return output, weights


def vit_attention(image: Image.Image, layer: int, head: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    model, preprocess = get_vit()
    capture: dict[str, list[torch.Tensor | None]] = {"layers": [None] * len(model.encoder.layers)}
    patched = []

    for layer_idx, block in enumerate(model.encoder.layers):
        module = block.self_attention
        if not hasattr(module, "_original_forward"):
            module._original_forward = module.forward
        module.forward = partial(forward_torchvision_attention, module, capture, layer_idx)
        patched.append(module)

    try:
        with torch.inference_mode():
            tensor = preprocess(image).unsqueeze(0)
            _ = model(tensor)
    finally:
        for module in patched:
            module.forward = module._original_forward

    layers = [item for item in capture["layers"] if item is not None]
    if not layers:
        raise HTTPException(status_code=500, detail="Could not capture ViT attention weights")

    layer_idx = min(layer, len(layers) - 1)
    picked_layer = layers[layer_idx]
    head_idx = min(head, picked_layer.shape[0] - 1)
    summaries = torch.stack([item[:, 0, 1:].mean(dim=1) for item in layers])
    meta = vit_feature_preview(image)
    return picked_layer[head_idx], summaries, meta


def vit_feature_preview(image: Image.Image) -> dict[str, Any]:
    tokens, meta = patch_features(image, 16)
    _ = tokens
    return meta


def dinov2_attention(
    image: Image.Image, layer: int, head: int
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    model, processor = get_dinov2()
    inputs = processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**inputs, output_attentions=True)

    layers = [item.detach().cpu().squeeze(0) for item in outputs.attentions or []]
    if not layers:
        raise HTTPException(status_code=500, detail="Could not capture DINOv2 attention weights")

    layer_idx = min(layer, len(layers) - 1)
    picked_layer = layers[layer_idx]
    head_idx = min(head, picked_layer.shape[0] - 1)
    summaries = torch.stack([item[:, 0, 1:].mean(dim=1) for item in layers])
    meta = patch_features(image, 14)[1]
    return picked_layer[head_idx], summaries, meta
