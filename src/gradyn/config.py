from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field


class ProcessConfig(BaseModel):
    video: Path
    output: Path
    camera: str
    focal_length_px: float | None = None
    target_labels: list[str] = Field(default_factory=list)
    max_auto_objects: int = 8
    exemplars: dict[str, Path] = Field(default_factory=dict)
    anchor_stride: int = 90
    anchor_backend: str = "grounding_dino_sam2_dinov2"
    anchor_device: str = "auto"
    max_inference_side: int = 960
    depth_backend: str = "depth_anything_v2_small_relative"
    depth_input_size: int = 756
    depth_every: int = 1
    resume: bool = True

    def stable_hash(self) -> str:
        payload = self.model_dump(mode="json")
        payload["video"] = str(self.video.resolve())
        payload["output"] = str(self.output.resolve())
        encoded = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()[:16]
