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
    objects: list[str] = Field(default_factory=list)
    prompt_bank: Path | None = None
    discover_objects: bool = False
    max_auto_objects: int = 8
    qwen_max_candidates: int = 24
    exemplars: dict[str, Path] = Field(default_factory=dict)
    qwen_stride: int = 360
    sam3_stride: int = 90
    box_proposer: str = "qwen"
    qwen_box_max_side: int = 672
    qwen_box_max_tokens: int = 384
    max_inference_side: int = 960
    depth_backend: str = "depth_anything_v2_small_relative"
    depth_input_size: int = 756
    depth_every: int = 1
    resume: bool = True

    def stable_hash(self) -> str:
        payload = self.model_dump(mode="json")
        # `discover_objects` was added after existing production jobs had already
        # been checkpointed. Its disabled value describes the historical explicit
        # object workflow, so omit it to keep those checkpoints hash-compatible.
        # The enabled value remains part of the hash because it changes the stages
        # that run and the source of the object vocabulary.
        if not payload["discover_objects"]:
            payload.pop("discover_objects")
        payload["video"] = str(self.video.resolve())
        payload["output"] = str(self.output.resolve())
        encoded = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()[:16]
