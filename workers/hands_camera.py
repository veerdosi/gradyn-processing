from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw
from scipy.signal import savgol_filter

try:
    from common import (
        color_for_id,
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
        color_for_id,
        frame_paths,
        job_paths,
        merge_intervals,
        render_video,
        save_json,
        save_npz_atomic,
        source_fps,
        write_parquet,
    )

FINGERTIPS = [4, 8, 12, 16, 20]


def enable_legacy_mano_pickle_compatibility() -> None:
    """Provide aliases required while unpickling the licensed MANO v1.2 files."""
    aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def choose_device(requested: str = "cpu"):
    import torch

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable.")
        return torch.device("mps")
    return torch.device("cpu")


def move_to_device_float32(value, device):
    """Move nested WiLoR inputs while avoiding unsupported MPS float64 tensors."""
    import torch

    if isinstance(value, dict):
        return {key: move_to_device_float32(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device_float32(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device_float32(item, device) for item in value)
    if isinstance(value, torch.Tensor):
        if value.is_floating_point():
            value = value.float()
        return value.to(device)
    return value


def interpolate_short_gaps(values: np.ndarray, valid: np.ndarray, max_gap: int = 5) -> tuple[np.ndarray, np.ndarray]:
    output = values.copy()
    inferred = np.zeros_like(valid)
    indices = np.flatnonzero(valid)
    for left, right in zip(indices[:-1], indices[1:]):
        gap = right - left - 1
        if gap <= 0 or gap > max_gap:
            continue
        for offset in range(1, gap + 1):
            alpha = offset / (gap + 1)
            output[left + offset] = (1 - alpha) * output[left] + alpha * output[right]
            inferred[left + offset] = True
    return output, inferred


def box_iou(left: np.ndarray, right: np.ndarray) -> float:
    x0 = max(float(left[0]), float(right[0]))
    y0 = max(float(left[1]), float(right[1]))
    x1 = min(float(left[2]), float(right[2]))
    y1 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, float(left[2] - left[0])) * max(
        0.0, float(left[3] - left[1])
    )
    right_area = max(0.0, float(right[2] - right[0])) * max(
        0.0, float(right[3] - right[1])
    )
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def select_temporal_detection(
    detections: list[tuple[np.ndarray, float, np.ndarray]],
    previous_box: np.ndarray | None,
    image_width: int,
    image_height: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    if previous_box is None or not np.isfinite(previous_box).all():
        return max(detections, key=lambda item: item[1])
    previous_center = np.asarray(
        [
            (previous_box[0] + previous_box[2]) / 2,
            (previous_box[1] + previous_box[3]) / 2,
        ],
        dtype=np.float32,
    )
    diagonal = max(float(np.hypot(image_width, image_height)), 1.0)

    def score(item: tuple[np.ndarray, float, np.ndarray]) -> float:
        box, detection_confidence, _ = item
        center = np.asarray(
            [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
            dtype=np.float32,
        )
        distance_score = max(
            0.0, 1.0 - float(np.linalg.norm(center - previous_center)) / diagonal
        )
        return (
            0.55 * float(detection_confidence)
            + 0.30 * box_iou(box, previous_box)
            + 0.15 * distance_score
        )

    return max(detections, key=score)


def suppress_short_hand_runs(
    state: np.ndarray,
    confidence: np.ndarray,
    *,
    minimum_frames: int = 12,
    preserve_confidence: float = 0.85,
) -> None:
    active = state != "rejected"
    start: int | None = None
    for index in range(len(active) + 1):
        is_active = index < len(active) and bool(active[index])
        if is_active and start is None:
            start = index
        if not is_active and start is not None:
            end = index
            if (
                end - start < minimum_frames
                and float(np.max(confidence[start:end], initial=0.0))
                < preserve_confidence
            ):
                state[start:end] = "rejected"
                confidence[start:end] = 0.0
            start = None


def smooth_valid_runs(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    output = values.copy()
    start = None
    for index in range(len(valid) + 1):
        active = index < len(valid) and valid[index]
        if active and start is None:
            start = index
        if not active and start is not None:
            end = index
            length = end - start
            if length >= 5:
                window = min(9, length if length % 2 else length - 1)
                output[start:end] = savgol_filter(
                    output[start:end],
                    window_length=window,
                    polyorder=min(2, window - 1),
                    axis=0,
                    mode="interp",
                )
            start = None
    return output


def project(points: np.ndarray, focal: float, width: int, height: int) -> np.ndarray:
    z = np.maximum(points[:, 2:3], 1e-4)
    xy = points[:, :2] / z
    xy[:, 0] = xy[:, 0] * focal + width / 2
    xy[:, 1] = xy[:, 1] * focal + height / 2
    return xy


def focal_length_for_job(paths: dict[str, Path], metadata: dict) -> tuple[float, str]:
    """Choose the focal length used for both camera translation and projection."""
    manifest_path = paths["root"] / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        calibrated = manifest.get("config", {}).get("focal_length_px")
        if calibrated is not None:
            return float(calibrated), "user-supplied calibrated effective focal length"
    profile = metadata.get("camera_profile") or {}
    intrinsics = profile.get("intrinsics") or {}
    profile_focal = intrinsics.get("wilor_projection_focal_length_px")
    if profile_focal is not None:
        return (
            float(profile_focal),
            f"camera profile {profile.get('profile_id', 'unknown')} "
            f"({intrinsics.get('wilor_projection_policy', 'profile policy')})",
        )
    focal = max(metadata["decoded_width"], metadata["decoded_height"]) * 0.8
    source = "image-size heuristic"
    return focal, source


def render_mano_mesh(
    image: Image.Image,
    vertices: np.ndarray,
    faces: np.ndarray,
    focal: float,
    color: tuple[int, int, int],
) -> Image.Image:
    """Render the complete MANO triangle surface with simple depth ordering."""
    projected = project(vertices, focal, image.width, image.height)
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    face_depth = vertices[faces, 2].mean(axis=1)
    for face_index in np.argsort(face_depth)[::-1]:
        triangle = projected[faces[face_index]]
        if not np.isfinite(triangle).all():
            continue
        if (
            triangle[:, 0].max() < 0
            or triangle[:, 1].max() < 0
            or triangle[:, 0].min() >= image.width
            or triangle[:, 1].min() >= image.height
        ):
            continue
        points = [(float(x), float(y)) for x, y in triangle]
        draw.polygon(
            points,
            fill=(*color, 34),
            outline=(*color, 95),
        )
    return Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")


def temporal_derivatives(
    values: np.ndarray, timestamps: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Differentiate only within contiguous valid runs; never bridge missing hands."""
    velocity = np.full_like(values, np.nan)
    acceleration = np.full_like(values, np.nan)
    start = None
    for index in range(len(valid) + 1):
        active = index < len(valid) and valid[index]
        if active and start is None:
            start = index
        if not active and start is not None:
            end = index
            if end - start >= 2:
                velocity[start:end] = np.gradient(
                    values[start:end], timestamps[start:end], axis=0
                )
            start = None
    velocity_valid = np.isfinite(velocity).all(axis=1)
    start = None
    for index in range(len(valid) + 1):
        active = index < len(valid) and velocity_valid[index]
        if active and start is None:
            start = index
        if not active and start is not None:
            end = index
            if end - start >= 2:
                acceleration[start:end] = np.gradient(
                    velocity[start:end], timestamps[start:end], axis=0
                )
            start = None
    return velocity, acceleration


def projected_hand_is_consistent(
    projected: np.ndarray,
    detector_box: np.ndarray,
    width: int,
    height: int,
) -> bool:
    """Reject projections that do not land inside their source hand detection."""
    if not np.isfinite(projected).all():
        return False
    x1, y1, x2, y2 = detector_box
    margin_x = max(8.0, (x2 - x1) * 0.2)
    margin_y = max(8.0, (y2 - y1) * 0.2)
    inside = (
        (projected[:, 0] >= x1 - margin_x)
        & (projected[:, 0] <= x2 + margin_x)
        & (projected[:, 1] >= y1 - margin_y)
        & (projected[:, 1] <= y2 + margin_y)
    )
    in_image = (
        (projected[:, 0] >= 0)
        & (projected[:, 0] < width)
        & (projected[:, 1] >= 0)
        & (projected[:, 1] < height)
    )
    span = np.ptp(projected, axis=0)
    return bool(inside.mean() >= 0.7 and in_image.mean() >= 0.8 and span.max() >= 20)


def projected_landmarks_are_consistent(
    projected: np.ndarray,
    detector_keypoints: np.ndarray,
    detector_box: np.ndarray,
) -> bool:
    """Require WiLoR joints to agree with the detector's 21 hand landmarks."""
    valid = (
        np.isfinite(projected).all(axis=1)
        & np.isfinite(detector_keypoints).all(axis=1)
        & (detector_keypoints[:, 2] >= 0.5)
    )
    if valid.sum() < 12:
        return False
    errors = np.linalg.norm(
        projected[valid] - detector_keypoints[valid, :2],
        axis=1,
    )
    x1, y1, x2, y2 = detector_box
    diagonal = max(float(np.hypot(x2 - x1, y2 - y1)), 1.0)
    return bool(
        np.median(errors) / diagonal <= 0.16
        and np.percentile(errors, 90) / diagonal <= 0.25
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument(
        "--device",
        choices=("cpu", "mps"),
        default="cpu",
        help=(
            "WiLoR reconstruction device. CPU is the production default because "
            "the current MPS path produces numerically incorrect hand geometry."
        ),
    )
    args = parser.parse_args()

    import cv2
    import torch

    # WiLoR's official detector and Lightning checkpoint predate PyTorch 2.6,
    # which changed torch.load(weights_only) to True by default. These are pinned,
    # trusted upstream checkpoints downloaded by Gradyn.
    original_torch_load = torch.load

    def trusted_torch_load(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return original_torch_load(*load_args, **load_kwargs)

    torch.load = trusted_torch_load
    enable_legacy_mano_pickle_compatibility()
    from ultralytics import YOLO

    root = Path(__file__).resolve().parents[1]
    repo = root / "models" / "WiLoR"
    sys.path.insert(0, str(repo))
    from wilor.datasets.vitdet_dataset import ViTDetDataset
    from wilor.models import load_wilor

    def cam_crop_to_full(cam_bbox, box_center, box_size, img_size, focal_length=5000.0):
        cam_tx = cam_bbox[:, 1]
        cam_ty = cam_bbox[:, 2]
        bbox_height = box_size
        image_width, image_height = img_size[:, 0], img_size[:, 1]
        cx = 2 * (box_center[:, 0] - image_width / 2.0) / (bbox_height + 1e-9)
        cy = 2 * (box_center[:, 1] - image_height / 2.0) / (bbox_height + 1e-9)
        depth = 2 * focal_length / (bbox_height * cam_bbox[:, 0] + 1e-9)
        return torch.stack(
            [cam_tx + cx / cam_bbox[:, 0], cam_ty + cy / cam_bbox[:, 0], depth],
            dim=-1,
        )

    paths = job_paths(args.job)
    frames = frame_paths(args.job)
    frame_count = len(frames)
    device = choose_device(args.device)
    metadata = json.loads((paths["source"] / "video_metadata.json").read_text())
    focal, focal_source = focal_length_for_job(paths, metadata)

    original_cwd = Path.cwd()
    __import__("os").chdir(repo)
    try:
        model, cfg = load_wilor(
            str(repo / "pretrained_models" / "wilor_final.ckpt"),
            str(repo / "pretrained_models" / "model_config.yaml"),
        )
        detector = YOLO(str(repo / "pretrained_models" / "detector.pt"))
        model = model.to(device).eval()
        # Ultralytics pose detection has a known MPS correctness issue. It is
        # small compared with WiLoR reconstruction, so keep detection on CPU
        # while retaining the expensive reconstruction model on MPS.
        detector_device = torch.device("cpu")
        detector.to(detector_device)

        hand_names = ["left", "right"]
        vertices = np.full((2, frame_count, 778, 3), np.nan, np.float32)
        joints = np.full((2, frame_count, 21, 3), np.nan, np.float32)
        rotations = np.full((2, frame_count, 16, 3, 3), np.nan, np.float32)
        betas = np.full((2, frame_count, 10), np.nan, np.float32)
        translations = np.full((2, frame_count, 3), np.nan, np.float32)
        detector_boxes = np.full((2, frame_count, 4), np.nan, np.float32)
        detector_keypoints = np.full((2, frame_count, 21, 3), np.nan, np.float32)
        confidence = np.zeros((2, frame_count), np.float32)
        mano_faces = model.mano.faces
        if hasattr(mano_faces, "detach"):
            mano_faces = mano_faces.detach().cpu().numpy()
        faces = np.asarray(mano_faces, dtype=np.int32)
        progress_path = paths["work"] / "hands_progress.npz"
        progress_meta_path = paths["work"] / "hands_progress.json"
        progress_signature = {
            "config_hash": json.loads(
                (paths["root"] / "manifest.json").read_text()
            ).get("config_hash"),
            "frame_count": frame_count,
            "device": str(device),
            "focal": focal,
            "edge_fallback": "reflect_pad_v1",
            "short_gap_frames": 5,
            "short_track_suppression": "12_frames_below_0.85",
        }
        completed = np.zeros(frame_count, dtype=bool)
        if progress_path.exists() and progress_meta_path.exists():
            progress_meta = json.loads(progress_meta_path.read_text())
            if progress_meta.get("signature") == progress_signature:
                progress = np.load(progress_path)
                vertices = progress["vertices"]
                joints = progress["joints"]
                rotations = progress["rotations"]
                betas = progress["betas"]
                translations = progress["translations"]
                detector_boxes = progress["detector_boxes"]
                detector_keypoints = progress["detector_keypoints"]
                confidence = progress["confidence"]
                completed = progress["completed"]
            else:
                progress_path.unlink(missing_ok=True)
                progress_meta_path.unlink(missing_ok=True)

        def save_progress() -> None:
            save_npz_atomic(
                progress_path,
                compressed=False,
                vertices=vertices,
                joints=joints,
                rotations=rotations,
                betas=betas,
                translations=translations,
                detector_boxes=detector_boxes,
                detector_keypoints=detector_keypoints,
                confidence=confidence,
                completed=completed,
            )
            save_json({"signature": progress_signature}, progress_meta_path)

        print(
            f"WiLoR processing {frame_count} frames "
            f"(detector={detector_device}, reconstruction={device}, "
            f"focal={focal:.1f}px from {focal_source}).",
            flush=True,
        )

        def collect_detections(
            detector_image: np.ndarray,
            *,
            offset_x: float = 0.0,
            offset_y: float = 0.0,
        ) -> dict[int, list[tuple[np.ndarray, float, np.ndarray]]]:
            result = detector(
                detector_image,
                conf=0.22,
                verbose=False,
                device=str(detector_device),
            )[0]
            found: dict[
                int, list[tuple[np.ndarray, float, np.ndarray]]
            ] = defaultdict(list)
            for detection in result:
                data = detection.boxes.data.detach().cpu().numpy().reshape(-1)
                if len(data) < 6:
                    continue
                box = data[:4].astype(np.float32)
                box[[0, 2]] -= offset_x
                box[[1, 3]] -= offset_y
                box[[0, 2]] = np.clip(box[[0, 2]], 0, image.shape[1] - 1)
                box[[1, 3]] = np.clip(box[[1, 3]], 0, image.shape[0] - 1)
                if box[2] - box[0] < 8 or box[3] - box[1] < 8:
                    continue
                keypoints = (
                    detection.keypoints.data[0].detach().cpu().numpy()
                    if detection.keypoints is not None
                    else np.full((21, 3), np.nan, np.float32)
                ).astype(np.float32)
                keypoints[:, 0] -= offset_x
                keypoints[:, 1] -= offset_y
                found[int(data[5])].append(
                    (box, float(data[4]), keypoints)
                )
            return found

        def previous_box_for(hand_index: int, frame_index: int) -> np.ndarray | None:
            start = max(0, frame_index - 12)
            for previous in range(frame_index - 1, start - 1, -1):
                box = detector_boxes[hand_index, previous]
                if np.isfinite(box).all():
                    return box
            return None

        for frame_index, frame_path in enumerate(frames):
            if completed[frame_index]:
                if frame_index == 0 or frame_index % 100 == 0:
                    print(
                        f"↷ WiLoR reusing through frame {frame_index + 1}",
                        flush=True,
                    )
                continue
            if frame_index == 0 or frame_index % 100 == 0:
                print(
                    f"WiLoR frame {frame_index + 1}/{frame_count}",
                    flush=True,
                )
            image = cv2.imread(str(frame_path))
            candidates = collect_detections(image)
            expected_classes = {
                hand_index
                for hand_index in range(2)
                if previous_box_for(hand_index, frame_index) is not None
            }
            missing_expected = expected_classes - set(candidates)
            if not candidates or missing_expected:
                pad_x = max(96, round(image.shape[1] * 0.16))
                pad_y = max(72, round(image.shape[0] * 0.14))
                padded = cv2.copyMakeBorder(
                    image,
                    pad_y,
                    pad_y,
                    pad_x,
                    pad_x,
                    cv2.BORDER_REFLECT_101,
                )
                fallback = collect_detections(
                    padded,
                    offset_x=float(pad_x),
                    offset_y=float(pad_y),
                )
                for class_id, detections in fallback.items():
                    candidates[class_id].extend(detections)
            for class_id, detections in candidates.items():
                hand_index = 1 if class_id == 1 else 0
                box, detection_confidence, detection_keypoints = (
                    select_temporal_detection(
                        detections,
                        previous_box_for(hand_index, frame_index),
                        image.shape[1],
                        image.shape[0],
                    )
                )
                dataset = ViTDetDataset(
                    cfg,
                    image,
                    np.asarray([box]),
                    np.asarray([class_id], np.float32),
                    rescale_factor=2.0,
                )
                cpu_batch = torch.utils.data.default_collate([dataset[0]])
                batch = move_to_device_float32(
                    cpu_batch,
                    device,
                )
                try:
                    with torch.inference_mode():
                        output = model(batch)
                except RuntimeError as error:
                    if device.type != "mps":
                        raise
                    print(
                        f"WiLoR MPS operation fell back to CPU on frame "
                        f"{frame_index + 1}: {error}",
                        flush=True,
                    )
                    model = model.cpu()
                    batch = move_to_device_float32(
                        cpu_batch,
                        torch.device("cpu"),
                    )
                    with torch.inference_mode():
                        output = model(batch)
                    model = model.to(device)

                multiplier = 2 * batch["right"] - 1
                pred_cam = output["pred_cam"].clone()
                pred_cam[:, 1] = multiplier * pred_cam[:, 1]
                image_size = batch["img_size"].float()
                camera_translation = cam_crop_to_full(
                    pred_cam,
                    batch["box_center"].float(),
                    batch["box_size"].float(),
                    image_size,
                    focal,
                )[0].detach().cpu().numpy()
                current_vertices = output["pred_vertices"][0].detach().cpu().numpy()
                current_joints = output["pred_keypoints_3d"][0].detach().cpu().numpy()
                sign = 1 if class_id == 1 else -1
                current_vertices[:, 0] *= sign
                current_joints[:, 0] *= sign
                current_vertices += camera_translation
                current_joints += camera_translation
                parameters = output["pred_mano_params"]
                current_rotations = np.concatenate(
                    [
                        parameters["global_orient"][0].detach().cpu().numpy(),
                        parameters["hand_pose"][0].detach().cpu().numpy(),
                    ],
                    axis=0,
                )
                vertices[hand_index, frame_index] = current_vertices
                joints[hand_index, frame_index] = current_joints[:21]
                rotations[hand_index, frame_index] = current_rotations
                betas[hand_index, frame_index] = (
                    parameters["betas"][0].detach().cpu().numpy()
                )
                translations[hand_index, frame_index] = camera_translation
                detector_boxes[hand_index, frame_index] = box
                detector_keypoints[hand_index, frame_index] = detection_keypoints
                confidence[hand_index, frame_index] = detection_confidence
            completed[frame_index] = True
            if (frame_index + 1) % 10 == 0 or frame_index == frame_count - 1:
                save_progress()
            if device.type == "mps":
                torch.mps.empty_cache()

        del detector, model
        if device.type == "mps":
            torch.mps.empty_cache()

        state = np.full((2, frame_count), "rejected", dtype="<U16")
        for hand_index in range(2):
            valid = np.isfinite(joints[hand_index]).all(axis=(1, 2))
            vertices[hand_index], inferred = interpolate_short_gaps(
                vertices[hand_index], valid
            )
            joints[hand_index], _ = interpolate_short_gaps(joints[hand_index], valid)
            translations[hand_index], _ = interpolate_short_gaps(
                translations[hand_index], valid
            )
            detector_boxes[hand_index], _ = interpolate_short_gaps(
                detector_boxes[hand_index], valid
            )
            detector_keypoints[hand_index], _ = interpolate_short_gaps(
                detector_keypoints[hand_index], valid
            )
            betas[hand_index], _ = interpolate_short_gaps(betas[hand_index], valid)
            smooth_valid = valid | inferred
            vertices[hand_index] = smooth_valid_runs(
                vertices[hand_index], smooth_valid
            )
            joints[hand_index] = smooth_valid_runs(joints[hand_index], smooth_valid)
            translations[hand_index] = smooth_valid_runs(
                translations[hand_index], smooth_valid
            )
            state[hand_index, valid] = "observed"
            state[hand_index, inferred] = "inferred"
            confidence[hand_index, inferred] = 0.35
            suppress_short_hand_runs(state[hand_index], confidence[hand_index])

        np.savez_compressed(
            paths["hands"] / "mano_parameters.npz",
            rotation_matrices=rotations,
            betas=betas,
            translation_camera_m=translations,
            handedness=np.asarray(hand_names),
            state=state,
        )
        np.savez_compressed(
            paths["hands"] / "meshes.npz",
            vertices_camera_m=vertices,
            faces=faces,
        )
        np.savez_compressed(
            paths["hands"] / "detector_landmarks.npz",
            keypoints_2d=detector_keypoints,
            boxes_xyxy=detector_boxes,
            handedness=np.asarray(hand_names),
        )

        timestamps = np.asarray(
            pq.read_table(paths["source"] / "frame_timestamps.parquet")[
                "timestamp_s"
            ],
            dtype=np.float64,
        )
        if frame_count > 1:
            delta = np.diff(timestamps)
            default_delta = (
                float(np.median(delta[delta > 0])) if np.any(delta > 0) else 1 / 30
            )
            for index in range(1, frame_count):
                if timestamps[index] <= timestamps[index - 1]:
                    timestamps[index] = timestamps[index - 1] + default_delta

        rows: list[dict] = []
        velocities = np.full((2, frame_count, 3), np.nan, np.float32)
        accelerations = np.full((2, frame_count, 3), np.nan, np.float32)
        for hand_index, handedness in enumerate(hand_names):
            wrist = joints[hand_index, :, 0]
            velocity, acceleration = temporal_derivatives(
                wrist,
                timestamps,
                state[hand_index] != "rejected",
            )
            velocities[hand_index] = velocity
            accelerations[hand_index] = acceleration
            for frame_index in range(frame_count):
                rows.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_s": float(timestamps[frame_index]),
                        "handedness": handedness,
                        "state": str(state[hand_index, frame_index]),
                        "confidence": float(confidence[hand_index, frame_index]),
                        "wrist_xyz_m": wrist[frame_index].tolist(),
                        "wrist_velocity_mps": velocity[frame_index].tolist(),
                        "wrist_acceleration_mps2": acceleration[frame_index].tolist(),
                        "fingertips_xyz_m": joints[
                            hand_index, frame_index, FINGERTIPS
                        ]
                        .reshape(-1)
                        .tolist(),
                    }
                )
        write_parquet(rows, paths["hands"] / "frames.parquet")
        write_parquet(rows, paths["hands"] / "trajectories.parquet")

        save_json(
            {
                "coordinate_system": "camera-relative",
                "camera_motion_present": True,
                "projection_focal_length_px": focal,
                "projection_focal_source": focal_source,
                "translation_source": (
                    "WiLoR weak-perspective camera converted using the declared "
                    "projection focal length"
                ),
                "reconstruction_device": str(device),
                "mps_status": (
                    "not used; CPU required for production-correct WiLoR geometry"
                    if device.type == "cpu"
                    else "experimental; known to differ numerically from CPU"
                ),
                "metric_scale_status": (
                    "calibrated"
                    if focal_source
                    == "user-supplied calibrated effective focal length"
                    else "estimated from focal length; validate against the capture setup"
                ),
            },
            paths["hands"] / "metadata.json",
        )
        overlays = paths["work"] / "hand_overlay_frames"
        shutil.rmtree(overlays, ignore_errors=True)
        overlays.mkdir(parents=True)
        frame_quality: list[dict] = []
        rejected: list[tuple[int, list[str]]] = []
        for frame_index, frame_path in enumerate(frames):
            image = Image.open(frame_path).convert("RGB")
            draw = ImageDraw.Draw(image)
            accepted_confidences: list[float] = []
            frame_reasons: list[str] = []
            for hand_index, handedness in enumerate(hand_names):
                if state[hand_index, frame_index] == "rejected":
                    continue
                projected = project(
                    joints[hand_index, frame_index], focal, image.width, image.height
                )
                depth_m = float(joints[hand_index, frame_index, 0, 2])
                speed_mps = float(
                    np.linalg.norm(velocities[hand_index, frame_index])
                )
                acceleration_mps2 = float(
                    np.linalg.norm(accelerations[hand_index, frame_index])
                )
                reasons: list[str] = []
                if not 0.15 <= depth_m <= 3.0:
                    reasons.append(f"{handedness}_invalid_camera_depth")
                if not projected_hand_is_consistent(
                    projected,
                    detector_boxes[hand_index, frame_index],
                    image.width,
                    image.height,
                ):
                    reasons.append(f"{handedness}_reprojection_mismatch")
                if not projected_landmarks_are_consistent(
                    projected,
                    detector_keypoints[hand_index, frame_index],
                    detector_boxes[hand_index, frame_index],
                ):
                    reasons.append(f"{handedness}_landmark_mismatch")
                if np.isfinite(speed_mps) and speed_mps > 5.0:
                    reasons.append(f"{handedness}_implausible_velocity")
                if np.isfinite(acceleration_mps2) and acceleration_mps2 > 100.0:
                    reasons.append(f"{handedness}_implausible_acceleration")
                if reasons:
                    frame_reasons.extend(reasons)
                    continue
                color = color_for_id(hand_index + 4)
                image = render_mano_mesh(
                    image,
                    vertices[hand_index, frame_index],
                    faces,
                    focal,
                    color,
                )
                draw = ImageDraw.Draw(image)
                for x, y in projected:
                    draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
                draw.text(
                    (10, 10 + 20 * hand_index),
                    f"{handedness} {state[hand_index, frame_index]} "
                    f"{confidence[hand_index, frame_index]:.2f}",
                    fill=color,
                )
                accepted_confidences.append(float(confidence[hand_index, frame_index]))
            overall = max(accepted_confidences, default=0.0)
            status_value = "accepted" if overall >= 0.3 else "rejected"
            reasons = (
                []
                if status_value == "accepted"
                else frame_reasons or ["no_reliable_hand"]
            )
            if status_value == "rejected":
                rejected.append((frame_index, reasons))
            frame_quality.append(
                {
                    "frame_index": frame_index,
                    "confidence": overall,
                    "status": status_value,
                    "reasons": reasons,
                }
            )
            image.save(overlays / f"{frame_index:08d}.jpg", quality=90)
        render_video(
            overlays, paths["hands"] / "overlay.mp4", source_fps(paths["root"])
        )
        save_json(
            {
                "frames": frame_quality,
                "quarantined_intervals": merge_intervals(rejected),
            },
            paths["hands"] / "quality.json",
        )
        progress_path.unlink(missing_ok=True)
        progress_meta_path.unlink(missing_ok=True)
    finally:
        __import__("os").chdir(original_cwd)


if __name__ == "__main__":
    main()
