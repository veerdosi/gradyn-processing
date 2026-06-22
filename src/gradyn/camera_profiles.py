from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path


def profiles_directory() -> Path:
    return Path(__file__).resolve().parents[2] / "camera_profiles"


def _fps(stream: dict) -> float:
    value = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return 0.0


def resolve_camera_profile(camera: str, stream: dict) -> dict | None:
    camera_name = camera.strip().casefold()
    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))
    fps = _fps(stream)
    for path in sorted(profiles_directory().glob("*.json")):
        profile = json.loads(path.read_text())
        aliases = {
            str(value).strip().casefold()
            for value in profile.get("aliases", [])
        }
        aliases.add(str(profile.get("camera_make_model", "")).strip().casefold())
        video = profile.get("video", {})
        if camera_name not in aliases:
            continue
        if width != int(video.get("width", 0)) or height != int(
            video.get("height", 0)
        ):
            continue
        nominal_fps = float(video.get("nominal_fps", 0))
        if fps and nominal_fps and abs(fps - nominal_fps) > 0.1:
            continue
        return {**profile, "path": str(path)}
    return None
