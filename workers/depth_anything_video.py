from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageOps

try:
    from common import (
        frame_paths,
        job_paths,
        merge_intervals,
        render_video,
        save_json,
        save_npz_atomic,
        source_fps,
        write_parquet,
    )
except ModuleNotFoundError:
    from workers.common import (
        frame_paths,
        job_paths,
        merge_intervals,
        render_video,
        save_json,
        save_npz_atomic,
        source_fps,
        write_parquet,
    )


def choose_device():
    import torch

    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def robust_normalize(
    raw_depth: np.ndarray,
    previous_low: float | None,
    previous_high: float | None,
    *,
    alpha: float = 0.15,
    reset: bool = False,
) -> tuple[np.ndarray, float, float]:
    valid = raw_depth[np.isfinite(raw_depth)]
    if not valid.size:
        return np.zeros_like(raw_depth, dtype=np.float32), 0.0, 1.0
    current_low, current_high = np.percentile(valid, [2, 98]).astype(float)
    if current_high <= current_low:
        current_high = current_low + 1e-6
    if previous_low is None or previous_high is None or reset:
        low, high = current_low, current_high
    else:
        low = (1 - alpha) * previous_low + alpha * current_low
        high = (1 - alpha) * previous_high + alpha * current_high
    normalized = np.clip((raw_depth - low) / max(high - low, 1e-6), 0, 1)
    return np.nan_to_num(normalized).astype(np.float32), low, high


def preview_image(relative_depth: np.ndarray) -> Image.Image:
    image = Image.fromarray((np.clip(relative_depth, 0, 1) * 255).astype(np.uint8))
    image.thumbnail((960, 960), Image.Resampling.BILINEAR)
    return ImageOps.colorize(image, black="#30123b", white="#f9e721")


def edge_preserving_filter(relative_depth: np.ndarray) -> np.ndarray:
    import cv2

    filtered = cv2.bilateralFilter(
        relative_depth.astype(np.float32),
        d=5,
        sigmaColor=0.035,
        sigmaSpace=3.0,
    )
    return np.clip(filtered, 0, 1).astype(np.float32)


def stabilize_relative_depth(
    current_depth: np.ndarray,
    current_gray: np.ndarray,
    previous_depth_small: np.ndarray | None,
    previous_gray_small: np.ndarray | None,
    *,
    reset: bool = False,
    flow_size: tuple[int, int] = (384, 216),
) -> tuple[np.ndarray, dict[str, float], np.ndarray, np.ndarray]:
    import cv2

    width, height = flow_size
    gray_small = cv2.resize(
        current_gray, (width, height), interpolation=cv2.INTER_AREA
    )
    depth_small = cv2.resize(
        current_depth, (width, height), interpolation=cv2.INTER_AREA
    ).astype(np.float32)
    empty_metrics = {
        "flow_valid_fraction": 0.0,
        "stabilization_blend_fraction": 0.0,
        "photometric_residual": 0.0,
    }
    if (
        reset
        or previous_depth_small is None
        or previous_gray_small is None
        or previous_depth_small.shape != depth_small.shape
    ):
        return current_depth, empty_metrics, depth_small, gray_small

    flow = cv2.calcOpticalFlowFarneback(
        gray_small,
        previous_gray_small,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=3,
        poly_n=7,
        poly_sigma=1.5,
        flags=0,
    )
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]
    warped_depth = cv2.remap(
        previous_depth_small,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    )
    warped_gray = cv2.remap(
        previous_gray_small,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    flow_magnitude = np.linalg.norm(flow, axis=2)
    photometric = np.abs(
        gray_small.astype(np.float32) - warped_gray.astype(np.float32)
    )
    depth_delta = np.abs(depth_small - warped_depth)
    rgb_edges = cv2.Canny(gray_small, 70, 150) > 0
    depth_dx = cv2.Sobel(depth_small, cv2.CV_32F, 1, 0, ksize=3)
    depth_dy = cv2.Sobel(depth_small, cv2.CV_32F, 0, 1, ksize=3)
    depth_edges = np.hypot(depth_dx, depth_dy) > 0.16
    edge_guard = cv2.dilate(
        np.logical_or(rgb_edges, depth_edges).astype(np.uint8),
        np.ones((3, 3), np.uint8),
    ) > 0
    valid = (
        np.isfinite(warped_depth)
        & (flow_magnitude < 45)
        & (photometric < 32)
        & (depth_delta < 0.14)
        & ~edge_guard
    )
    weight = np.zeros_like(depth_small, dtype=np.float32)
    weight[valid] = 0.28
    warped_safe = np.where(np.isfinite(warped_depth), warped_depth, depth_small)
    stabilized_small = depth_small * (1 - weight) + warped_safe * weight
    correction = stabilized_small - depth_small
    correction_full = cv2.resize(
        correction,
        (current_depth.shape[1], current_depth.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    full_gray = cv2.resize(
        gray_small,
        (current_depth.shape[1], current_depth.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    full_edges = cv2.Canny(full_gray, 70, 150) > 0
    full_guard = cv2.dilate(
        full_edges.astype(np.uint8), np.ones((5, 5), np.uint8)
    ) > 0
    stabilized = np.clip(
        current_depth + correction_full * (~full_guard),
        0,
        1,
    ).astype(np.float32)
    metrics = {
        "flow_valid_fraction": float(valid.mean()),
        "stabilization_blend_fraction": float((weight > 0).mean()),
        "photometric_residual": float(np.median(photometric[valid]))
        if np.any(valid)
        else float(np.median(photometric)),
    }
    return stabilized, metrics, stabilized_small, gray_small


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--input-size", type=int, default=756)
    args = parser.parse_args()

    import cv2
    import torch

    root = Path(__file__).resolve().parents[1]
    repo = root / "models" / "depth-anything-v2"
    checkpoint = repo / "checkpoints" / "depth_anything_v2_vits.pth"
    sys.path.insert(0, str(repo))
    logging.getLogger("dinov2").setLevel(logging.ERROR)
    warnings.filterwarnings(
        "ignore",
        message="The operator 'aten::upsample_bicubic2d.out'.*",
        category=UserWarning,
    )
    from depth_anything_v2.dpt import DepthAnythingV2

    paths = job_paths(args.job)
    frames = frame_paths(args.job)
    device = choose_device()
    model = DepthAnythingV2(
        encoder="vits",
        features=64,
        out_channels=[48, 96, 192, 384],
    )
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)
    )
    model = model.to(device).eval()
    print(
        f"Depth Anything V2 Small processing relative depth on {device}; "
        f"every {args.every} source frame.",
        flush=True,
    )
    if device.type == "mps":
        print(
            "MPS acceleration active; positional-embedding bicubic resize uses "
            "PyTorch's isolated CPU fallback.",
            flush=True,
        )

    depth_frames = paths["depth"] / "frames"
    invalid_dir = paths["depth"] / "invalid_masks"
    previews = paths["work"] / "depth_preview_frames"
    for directory in (depth_frames, invalid_dir, previews):
        directory.mkdir(parents=True, exist_ok=True)

    progress_path = paths["work"] / "depth_progress.json"
    signature = {
        "model": "Depth-Anything-V2-Small-relative",
        "frame_count": len(frames),
        "every": args.every,
        "input_size": args.input_size,
    }
    rows: list[dict] = []
    qualities: list[dict] = []
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress.get("signature") == signature:
            rows = progress.get("rows", [])
            qualities = progress.get("qualities", [])
        else:
            shutil.rmtree(depth_frames, ignore_errors=True)
            shutil.rmtree(invalid_dir, ignore_errors=True)
            depth_frames.mkdir(parents=True)
            invalid_dir.mkdir(parents=True)
    completed = {int(row["frame_index"]) for row in rows}
    timestamps = pq.read_table(
        paths["source"] / "frame_timestamps.parquet"
    )["timestamp_s"].to_pylist()
    processed_count = (len(frames) + args.every - 1) // args.every
    previous_low = float(rows[-1]["normalization_low"]) if rows else None
    previous_high = float(rows[-1]["normalization_high"]) if rows else None
    previous_gray: np.ndarray | None = None
    previous_small_depth: np.ndarray | None = None
    previous_qa_depth: np.ndarray | None = None
    previous_flow_gray: np.ndarray | None = None
    processed_index = len(completed)

    for frame_index, frame_path in enumerate(frames):
        if frame_index % args.every:
            continue
        output_path = depth_frames / f"{frame_index:08d}.npz"
        invalid_path = invalid_dir / f"{frame_index:08d}.png"
        if frame_index in completed and output_path.exists() and invalid_path.exists():
            print(f"↷ Relative depth reusing source frame {frame_index}", flush=True)
            with np.load(output_path) as payload:
                saved_depth = payload["relative_depth"].astype(np.float32)
                previous_small_depth = cv2.resize(
                    saved_depth,
                    (384, 216),
                    interpolation=cv2.INTER_AREA,
                )
                previous_qa_depth = cv2.resize(
                    saved_depth,
                    (160, 90),
                    interpolation=cv2.INTER_AREA,
                )
            source_gray = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
            previous_gray = cv2.resize(
                source_gray,
                (160, 90),
                interpolation=cv2.INTER_AREA,
            )
            previous_flow_gray = cv2.resize(
                source_gray, (384, 216), interpolation=cv2.INTER_AREA
            )
            continue

        rows = [row for row in rows if int(row["frame_index"]) != frame_index]
        qualities = [
            row for row in qualities if int(row["frame_index"]) != frame_index
        ]
        started = time.perf_counter()
        image = cv2.imread(str(frame_path))
        full_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(
            full_gray,
            (160, 90),
            interpolation=cv2.INTER_AREA,
        )
        scene_change = (
            previous_gray is not None
            and float(np.mean(np.abs(gray.astype(np.float32) - previous_gray))) > 45
        )
        with torch.inference_mode():
            raw_depth = model.infer_image(image, args.input_size)
        relative_depth, low, high = robust_normalize(
            raw_depth,
            previous_low,
            previous_high,
            reset=scene_change,
        )
        relative_depth = edge_preserving_filter(relative_depth)
        (
            relative_depth,
            stabilization,
            stabilized_small,
            flow_gray,
        ) = stabilize_relative_depth(
            relative_depth,
            full_gray,
            previous_small_depth,
            previous_flow_gray,
            reset=scene_change,
        )
        invalid = ~np.isfinite(raw_depth)
        reasons: list[str] = []
        confidence = max(0.0, 1.0 - float(invalid.mean()) * 4.0)
        small_depth = cv2.resize(
            relative_depth, (160, 90), interpolation=cv2.INTER_AREA
        )
        temporal_delta = (
            float(np.median(np.abs(small_depth - previous_qa_depth)))
            if previous_qa_depth is not None and not scene_change
            else 0.0
        )
        if temporal_delta > 0.28:
            confidence *= 0.65
            reasons.append("large_relative_depth_change")
        elif temporal_delta:
            confidence *= max(0.65, 1.0 - temporal_delta / 0.35)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        channel_max = image.max(axis=2)
        channel_mean = image.mean(axis=2)
        glare_fraction = float(
            np.logical_and(channel_max >= 252, channel_mean >= 210).mean()
        )
        if glare_fraction > 0.03:
            confidence *= 0.65
            reasons.append("severe_glare_or_clipping")
        elif glare_fraction > 0.008:
            confidence *= 0.82
            reasons.append("glare_or_clipping")
        if sharpness < 12:
            confidence *= 0.75
            reasons.append("motion_blur_or_low_texture")
        elif sharpness < 35:
            confidence *= 0.9
        if scene_change:
            reasons.append("scene_cut_normalization_reset")
        status = "accepted" if confidence >= 0.5 else "rejected"

        save_npz_atomic(
            output_path,
            relative_depth=relative_depth.astype(np.float16),
        )
        Image.fromarray(invalid.astype(np.uint8) * 255).save(invalid_path)
        elapsed = time.perf_counter() - started
        print(
            f"Relative depth frame {processed_index + 1}/{processed_count} "
            f"(source {frame_index}) completed in {elapsed:.2f}s",
            flush=True,
        )
        rows.append(
            {
                "frame_index": frame_index,
                "timestamp_s": float(timestamps[frame_index]),
                "normalization_low": low,
                "normalization_high": high,
                "temporal_delta": temporal_delta,
                "flow_valid_fraction": stabilization["flow_valid_fraction"],
                "stabilization_blend_fraction": stabilization[
                    "stabilization_blend_fraction"
                ],
                "photometric_residual": stabilization["photometric_residual"],
                "sharpness": sharpness,
                "glare_fraction": glare_fraction,
                "invalid_fraction": float(invalid.mean()),
                "confidence": confidence,
                "status": status,
                "units": "normalized_relative_inverse_depth",
                "near_value": 1.0,
                "far_value": 0.0,
            }
        )
        qualities.append(
            {
                "frame_index": frame_index,
                "confidence": confidence,
                "status": status,
                "reasons": reasons,
            }
        )
        save_json(
            {"signature": signature, "rows": rows, "qualities": qualities},
            progress_path,
        )
        previous_low, previous_high = low, high
        previous_gray = gray
        previous_small_depth = stabilized_small
        previous_qa_depth = small_depth
        previous_flow_gray = flow_gray
        processed_index += 1
        if device.type == "mps":
            torch.mps.empty_cache()

    rows.sort(key=lambda row: int(row["frame_index"]))
    qualities.sort(key=lambda row: int(row["frame_index"]))
    rejected = [
        (
            int(item["frame_index"]),
            item.get("reasons", []) or ["low_relative_depth_confidence"],
        )
        for item in qualities
        if item["status"] == "rejected"
    ]
    write_parquet(rows, paths["depth"] / "depth_metadata.parquet")
    save_json(
        {"frames": qualities, "quarantined_intervals": merge_intervals(rejected)},
        paths["depth"] / "quality.json",
    )
    shutil.rmtree(previews, ignore_errors=True)
    previews.mkdir(parents=True)
    for preview_index, path in enumerate(sorted(depth_frames.glob("*.npz"))):
        with np.load(path) as payload:
            preview_image(payload["relative_depth"].astype(np.float32)).save(
                previews / f"{preview_index:08d}.jpg",
                quality=88,
            )
    save_json(
        {
            "model": "Depth Anything V2 Small",
            "checkpoint": "depth_anything_v2_vits.pth",
            "quantity": "normalized relative inverse depth",
            "near_value": 1.0,
            "far_value": 0.0,
            "near_color": "#f9e721",
            "far_color": "#30123b",
            "metric_scale": False,
            "temporal_normalization": (
                "per-frame 2nd/98th percentiles with exponential smoothing; "
                "reset at detected scene cuts"
            ),
            "input_size": args.input_size,
            "spatial_filter": "5-pixel bilateral edge-preserving filter",
            "temporal_stabilization": (
                "Farneback optical-flow warping at 384x216 with photometric, "
                "motion, depth-disagreement, and RGB/depth-edge rejection"
            ),
            "warning": (
                "Values are ordinal/relative and must not be interpreted as meters."
            ),
        },
        paths["depth"] / "metadata.json",
    )
    shutil.copy2(
        paths["depth"] / "metadata.json",
        paths["depth"] / "preview_metadata.json",
    )
    render_video(
        previews,
        paths["depth"] / "preview.mp4",
        source_fps(paths["root"]) / args.every,
    )
    progress_path.unlink(missing_ok=True)
    if device.type == "mps":
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
