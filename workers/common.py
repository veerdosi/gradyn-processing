from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

SAM3_NATIVE_RESOLUTION = 1008


def chunk_ranges(frame_count: int, chunk_frames: int, overlap: int) -> list[tuple[int, int]]:
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")
    if overlap < 0 or overlap >= chunk_frames:
        raise ValueError("overlap must be between 0 and chunk_frames - 1")
    step = chunk_frames - overlap
    ranges: list[tuple[int, int]] = []
    for start in range(0, frame_count, step):
        end = min(start + chunk_frames, frame_count)
        ranges.append((start, end))
        if end == frame_count:
            break
    return ranges


def load_prompt_bank(path: str | Path) -> dict:
    bank_path = Path(path).expanduser().resolve()
    if not bank_path.exists():
        raise ValueError(f"Prompt bank does not exist: {bank_path}")
    data = json.loads(bank_path.read_text())
    prompts = data.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"Prompt bank must contain a non-empty prompts list: {bank_path}")
    normalized = [str(prompt).strip() for prompt in prompts if str(prompt).strip()]
    if not normalized:
        raise ValueError(f"Prompt bank contains no usable prompts: {bank_path}")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"Prompt bank contains duplicate prompts: {bank_path}")
    return {
        "path": str(bank_path),
        "schema_version": str(data.get("schema_version", "1.0")),
        "name": str(data.get("name", bank_path.stem)),
        "description": str(data.get("description", "")),
        "task": str(data.get("task", "")),
        "prompts": normalized,
    }


def job_paths(job: str | Path) -> dict[str, Path]:
    root = Path(job).resolve()
    return {
        "root": root,
        "frames": root / "source" / "frames",
        "source": root / "source",
        "work": root / ".work",
        "objects": root / "objects",
        "hands": root / "hands",
        "depth": root / "depth",
    }


def frame_paths(job: str | Path) -> list[Path]:
    return sorted((Path(job) / "source" / "frames").glob("*.jpg"))


def write_parquet(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        pq.write_table(pa.Table.from_pylist(rows), path)
    else:
        path.write_bytes(b"")


def mask_bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def encode_coco_rle(mask: np.ndarray) -> dict:
    pixels = np.asarray(mask, dtype=np.uint8).T.flatten()
    counts: list[int] = []
    previous = 0
    run = 0
    for pixel in pixels:
        value = int(pixel)
        if value == previous:
            run += 1
        else:
            counts.append(run)
            run = 1
            previous = value
    counts.append(run)
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


def decode_coco_rle(rle: dict) -> np.ndarray:
    values: list[int] = []
    value = 0
    for count in rle["counts"]:
        values.extend([value] * int(count))
        value = 1 - value
    return np.asarray(values, dtype=np.uint8).reshape(
        (int(rle["size"][1]), int(rle["size"][0]))
    ).T.astype(bool)


def iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(left, right).sum() / union)


def merge_intervals(frames: Iterable[tuple[int, list[str]]]) -> list[dict]:
    ordered = sorted(frames)
    if not ordered:
        return []
    intervals: list[dict] = []
    start, previous, reasons = ordered[0][0], ordered[0][0], set(ordered[0][1])
    for index, current_reasons in ordered[1:]:
        if index == previous + 1:
            previous = index
            reasons.update(current_reasons)
            continue
        intervals.append(
            {"start_frame": start, "end_frame": previous, "reasons": sorted(reasons)}
        )
        start = previous = index
        reasons = set(current_reasons)
    intervals.append(
        {"start_frame": start, "end_frame": previous, "reasons": sorted(reasons)}
    )
    return intervals


def color_for_id(value: int) -> tuple[int, int, int]:
    palette = [
        (0, 210, 255),
        (255, 80, 110),
        (100, 230, 120),
        (190, 110, 255),
        (255, 190, 50),
        (40, 150, 255),
    ]
    return palette[value % len(palette)]


def render_video(images: Path, output: Path, fps: float = 30.0) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    base = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            f"{fps:.8f}",
            "-i",
            str(images / "%08d.jpg"),
    ]
    hardware = subprocess.run(
        [
            *base,
            "-c:v",
            "h264_videotoolbox",
            "-pix_fmt",
            "yuv420p",
            "-b:v",
            "8M",
            str(output),
        ],
    )
    if hardware.returncode:
        subprocess.run(
            [
                *base,
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                str(output),
            ],
            check=True,
        )


def source_fps(job: Path) -> float:
    table = pq.read_table(job / "source" / "frame_timestamps.parquet").to_pydict()
    timestamps = table["timestamp_s"]
    if len(timestamps) < 2:
        return 30.0
    durations = np.diff(np.asarray(timestamps, dtype=np.float64))
    median = float(np.median(durations[durations > 0]))
    return 1.0 / median if median > 0 else 30.0


def save_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2))
    os.replace(temporary, path)


def save_npz_atomic(path: Path, *, compressed: bool = True, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    if compressed:
        np.savez_compressed(temporary, **arrays)
    else:
        np.savez(temporary, **arrays)
    os.replace(temporary, path)


def resize_for_max_side(image: Image.Image, max_side: int) -> tuple[Image.Image, float]:
    width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    if scale == 1.0:
        return image, scale
    return image.resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS), scale


def overlay_mask(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int],
    label: str,
    bbox: list[int] | None = None,
) -> Image.Image:
    base = image.convert("RGBA")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    alpha = Image.fromarray((mask.astype(np.uint8) * 100), mode="L")
    solid = Image.new("RGBA", image.size, (*color, 0))
    solid.putalpha(alpha)
    base.alpha_composite(solid)
    draw = ImageDraw.Draw(base)
    if bbox:
        x, y, w, h = bbox
        draw.rectangle((x, y, x + w, y + h), outline=(*color, 255), width=3)
        draw.rectangle((x, max(0, y - 20), x + max(70, len(label) * 8), y), fill=(*color, 220))
        draw.text((x + 3, max(0, y - 18)), label, fill=(0, 0, 0, 255))
    return base.convert("RGB")


def clear_torch(device: str) -> None:
    try:
        import gc
        import torch

        gc.collect()
        if device == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
