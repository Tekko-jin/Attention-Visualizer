from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    image_data: str = Field(..., alias="imageData")
    model_mode: str = Field("random", alias="modelMode")
    patch_size: int = Field(16, ge=8, le=32, alias="patchSize")
    layer: int = Field(0, ge=0, le=11)
    head: int = Field(0, ge=0, le=11)
    token: int = Field(0, ge=0)
