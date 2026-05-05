import base64
import io
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import HTTPException
from PIL import Image, ImageFilter


def decode_image(data_url: str) -> Image.Image:
    try:
        payload = data_url.split(",", 1)[1] if "," in data_url else data_url
        raw = base64.b64decode(payload)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not decode image data") from exc

    image.thumbnail((512, 512), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (224, 224), (18, 24, 31))
    fitted = image.copy()
    fitted.thumbnail((224, 224), Image.Resampling.LANCZOS)
    x = (224 - fitted.width) // 2
    y = (224 - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def patch_features(image: Image.Image, patch_size: int) -> tuple[torch.Tensor, dict[str, Any]]:
    arr = np.asarray(image).astype(np.float32) / 255.0
    gray = image.convert("L").filter(ImageFilter.FIND_EDGES)
    edges = np.asarray(gray).astype(np.float32) / 255.0

    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    patches = F.unfold(x, kernel_size=patch_size, stride=patch_size)
    patch_count = patches.shape[-1]
    patches = patches.transpose(1, 2).reshape(1, patch_count, 3, patch_size, patch_size)

    mean_rgb = patches.mean(dim=(-1, -2)).squeeze(0)
    std_rgb = patches.std(dim=(-1, -2)).squeeze(0)

    edge_tensor = torch.from_numpy(edges).unsqueeze(0).unsqueeze(0)
    edge_patches = F.unfold(edge_tensor, kernel_size=patch_size, stride=patch_size)
    edge_mean = edge_patches.mean(dim=1).squeeze(0).unsqueeze(1)

    grid = 224 // patch_size
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, grid),
        torch.linspace(-1.0, 1.0, grid),
        indexing="ij",
    )
    coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    radial = torch.sqrt((coords**2).sum(dim=1, keepdim=True))

    features = torch.cat([mean_rgb, std_rgb, edge_mean, coords, radial], dim=1)
    features = (features - features.mean(dim=0, keepdim=True)) / (
        features.std(dim=0, keepdim=True) + 1e-6
    )

    cls = torch.zeros(1, features.shape[1])
    tokens = torch.cat([cls, features], dim=0)
    meta = {
        "grid": grid,
        "patchCount": patch_count,
        "featurePreview": [
            {
                "token": i + 1,
                "rgb": [round(float(v), 3) for v in mean_rgb[i].tolist()],
                "edge": round(float(edge_mean[i].item()), 3),
                "x": int(i % grid),
                "y": int(i // grid),
            }
            for i in range(min(patch_count, 196))
        ],
    }
    return tokens, meta
