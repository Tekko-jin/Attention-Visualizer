from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from schemas import AnalyzeRequest
from utils import decode_image, patch_features
from attention import dinov2_attention, simulation_attention, vit_attention

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")

@app.post("/api/analyze")
def analyze(payload: AnalyzeRequest) -> dict[str, Any]:

    image = decode_image(payload.image_data)
    model_mode = payload.model_mode

    if model_mode == "vit_b_16":
        patch_size = 16
        attention, summaries, meta = vit_attention(image, payload.layer, payload.head)
        model_label = "torchvision vit_b_16 ImageNet pretrained"
    elif model_mode == "dinov2_vits14":
        patch_size = 14
        attention, summaries, meta = dinov2_attention(image, payload.layer, payload.head)
        model_label = "DINOv2 ViT-S/14 self-supervised"
    else:
        patch_size = payload.patch_size
        tokens, meta = patch_features(image, patch_size)
        attention, summaries = simulation_attention(
            tokens, min(payload.layer, 7), min(payload.head, 5)
        )
        model_label = "educational ViT-like simulation"

    token_count = attention.shape[0]
    token = min(payload.token, token_count - 1)
    selected = attention[token, 1:]
    incoming = attention[:, token][1:]

    return {
        "modelMode": model_mode,
        "modelLabel": model_label,
        "imageSize": 224,
        "patchSize": patch_size,
        "grid": meta["grid"],
        "patchCount": meta["patchCount"],
        "tokenCount": token_count,
        "selectedToken": token,
        "layer": min(payload.layer, summaries.shape[0] - 1),
        "head": min(payload.head, summaries.shape[1] - 1),
        "layerCount": summaries.shape[0],
        "headCount": summaries.shape[1],
        "attention": [round(float(v), 6) for v in selected.tolist()],
        "incomingAttention": [round(float(v), 6) for v in incoming.tolist()],
        "layerSummaries": [
            [round(float(v), 6) for v in layer_values.tolist()]
            for layer_values in summaries
        ],
        "featurePreview": meta["featurePreview"],
    }