from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from scipy import ndimage

MIN_DIRECTIONAL_AREA_RATIO = 0.10
DISAGREEMENT_IOU_THRESHOLD = 0.30
LIFECYCLE_SIGNAL_RUN = 3
LIFECYCLE_COLLAPSE_RUN = 5
TRUSTED_ANCHOR_PROPAGATION = 60
REPLACEMENT_STABILIZATION_SECONDS = 0.8
MAX_UNCACHED_INTERVAL_FRAMES = 900
CLIP_LABEL_PROB_THRESHOLD = 0.36
PROPAGATED_AREA_COLLAPSE_RATIO = 0.20

try:
    from common import (
        color_for_id,
        decode_coco_rle,
        encode_coco_rle,
        frame_paths,
        job_paths,
        mask_bbox,
        merge_intervals,
        overlay_mask,
        render_video,
        save_json,
        save_npz_atomic,
        source_fps,
        write_parquet,
    )
except ModuleNotFoundError:
    from workers.common import (
        color_for_id,
        decode_coco_rle,
        encode_coco_rle,
        frame_paths,
        job_paths,
        mask_bbox,
        merge_intervals,
        overlay_mask,
        render_video,
        save_json,
        save_npz_atomic,
        source_fps,
        write_parquet,
    )


def anchor_hash(item: dict) -> str:
    payload = json.dumps(
        item["mask_rle"], sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def segment_cache_name(
    object_id: int,
    anchor: dict,
    start_frame: int,
    end_frame: int,
    internal_size: int,
    direction: str,
    mem_every: int,
    max_mem_frames: int,
) -> str:
    return (
        f"object_{object_id}_{start_frame}_{end_frame}_{direction}_"
        f"{anchor_hash(anchor)}_{internal_size}_m{mem_every}_w{max_mem_frames}.npz"
    )


def pack_masks(masks: list[np.ndarray]) -> tuple[np.ndarray, int]:
    stacked = np.stack(masks).astype(np.uint8)
    return np.packbits(stacked, axis=-1), int(stacked.shape[-1])


def unpack_masks(packed: np.ndarray, width: int) -> list[np.ndarray]:
    values = np.unpackbits(packed, axis=-1)[..., :width].astype(bool)
    return [values[index] for index in range(len(values))]


def masked_crop_for_clip(
    image: Image.Image,
    mask: np.ndarray,
    *,
    pad_fraction: float = 0.18,
) -> Image.Image:
    x, y, width, height = mask_bbox(mask)
    if width <= 0 or height <= 0:
        return image
    pad = int(max(width, height) * pad_fraction)
    left = max(0, x - pad)
    top = max(0, y - pad)
    right = min(image.width, x + width + pad)
    bottom = min(image.height, y + height + pad)
    array = np.asarray(image).copy()
    clipped_mask = mask.astype(bool)
    array[~clipped_mask] = 255
    return Image.fromarray(array).crop((left, top, right, bottom))


def clip_label_tracks(
    *,
    root: Path,
    frames: list[Path],
    masks_by_object: dict[int, dict[int, np.ndarray]],
    confidence_by_object: dict[int, dict[int, float]],
    fallback_labels: dict[int, str],
    target_labels: list[str],
    device_name: str,
) -> tuple[dict[int, str], dict[int, dict]]:
    if not target_labels:
        return fallback_labels, {}

    import clip
    import torch

    clip_home = root / "models" / "clip-home"
    clip_checkpoint = clip_home / ".cache" / "clip" / "ViT-B-32.pt"
    if not clip_checkpoint.exists():
        raise SystemExit(
            "CLIP ViT-B/32 checkpoint is missing at "
            f"{clip_checkpoint}. Restore models/clip-home before tracking."
        )

    device = torch.device(device_name)
    previous_home = os.environ.get("HOME")
    os.environ["HOME"] = str(clip_home)
    try:
        model, preprocess = clip.load("ViT-B/32", device=device, download_root=None)
    finally:
        if previous_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = previous_home
    model.eval()
    text = clip.tokenize([f"a photo of a {label}" for label in target_labels]).to(device)

    details: dict[int, dict] = {}
    probability_by_object: dict[int, list[float]] = {}
    with torch.inference_mode():
        for object_id, frame_masks in sorted(masks_by_object.items()):
            candidates = [
                (
                    confidence_by_object[object_id].get(frame_index, 0.0)
                    * max(float(mask.sum()), 1.0) ** 0.5,
                    frame_index,
                    mask,
                )
                for frame_index, mask in frame_masks.items()
                if mask.any()
            ]
            if not candidates:
                continue
            _, frame_index, mask = max(candidates, key=lambda item: item[0])
            image = Image.open(frames[frame_index]).convert("RGB")
            crop = masked_crop_for_clip(image, mask)
            image_tensor = preprocess(crop).unsqueeze(0).to(device)
            logits_per_image, _ = model(image_tensor, text)
            probabilities = (
                logits_per_image.softmax(dim=1)[0].detach().cpu().numpy()
            )
            ranking = sorted(
                [
                    (float(probabilities[index]), target_labels[index])
                    for index in range(len(target_labels))
                ],
                reverse=True,
            )
            score, label = ranking[0]
            probability_by_object[object_id] = [
                float(probabilities[index]) for index in range(len(target_labels))
            ]
            details[object_id] = {
                "object_id": object_id,
                "label": f"unknown object {object_id}",
                "accepted": False,
                "representative_frame": int(frame_index),
                "clip_score": score,
                "scores": [
                    {"label": candidate_label, "score": candidate_score}
                    for candidate_score, candidate_label in ranking
                ],
            }

    final_labels = dict(fallback_labels)
    assignments: dict[int, tuple[str, float]] = {}
    used_objects: set[int] = set()
    used_labels: set[int] = set()
    edges = sorted(
        [
            (scores[label_index], object_id, label_index)
            for object_id, scores in probability_by_object.items()
            for label_index in range(len(scores))
        ],
        reverse=True,
    )
    for score, object_id, label_index in edges:
        if object_id in used_objects or label_index in used_labels:
            continue
        if score < CLIP_LABEL_PROB_THRESHOLD:
            continue
        used_objects.add(object_id)
        used_labels.add(label_index)
        assignments[object_id] = (target_labels[label_index], float(score))
    for object_id in sorted(probability_by_object):
        label, score = assignments.get(
            object_id,
            (f"unknown object {object_id}", max(probability_by_object[object_id])),
        )
        final_labels[object_id] = label
        details[object_id]["label"] = label
        details[object_id]["accepted"] = object_id in assignments
        details[object_id]["clip_score"] = score

    del model
    if device_name == "mps":
        torch.mps.empty_cache()
    return final_labels, details


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.logical_or(first, second).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(first, second).sum() / union)


def primary_component(mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Remove disconnected false regions without inventing filled mask pixels."""
    mask = np.asarray(mask, dtype=bool)
    area = int(mask.sum())
    if area == 0:
        return mask, 1.0

    # A one-pixel bridge prevents tiny raster gaps from splitting one object,
    # while the returned mask still contains only pixels predicted by Cutie.
    bridge = ndimage.binary_dilation(
        mask,
        structure=np.ones((3, 3), dtype=bool),
        iterations=1,
    )
    components, count = ndimage.label(bridge)
    if count <= 1:
        return mask, 1.0
    component_areas = np.bincount(components.ravel())
    component_areas[0] = 0
    primary_label = int(component_areas.argmax())
    cleaned = np.logical_and(mask, components == primary_label)
    return cleaned, float(cleaned.sum() / area)


def propagated_mask_is_plausible(mask: np.ndarray | None, anchor_area: int) -> bool:
    if mask is None or not mask.any():
        return False
    return int(mask.sum()) >= PROPAGATED_AREA_COLLAPSE_RATIO * max(anchor_area, 1)


def fuse_directional_masks(
    forward: np.ndarray | None,
    backward: np.ndarray | None,
    minimum_component_fraction: float = 0.65,
) -> tuple[np.ndarray | None, float, float]:
    """Fuse complementary directional predictions, then require one object."""
    if forward is None and backward is None:
        return None, 0.0, 0.0
    if forward is None:
        cleaned, component_fraction = primary_component(backward)
        return (
            cleaned if cleaned.any() and component_fraction >= minimum_component_fraction else None,
            0.0,
            component_fraction,
        )
    if backward is None:
        cleaned, component_fraction = primary_component(forward)
        return (
            cleaned if cleaned.any() and component_fraction >= minimum_component_fraction else None,
            0.0,
            component_fraction,
        )

    agreement = mask_iou(forward, backward)
    cleaned, component_fraction = primary_component(
        np.logical_or(forward, backward)
    )
    if not cleaned.any() or component_fraction < minimum_component_fraction:
        return None, agreement, component_fraction
    return cleaned, agreement, component_fraction


def first_sustained_true(
    values: list[bool] | np.ndarray,
    run_length: int,
) -> int | None:
    run = 0
    for index, value in enumerate(values):
        run = run + 1 if bool(value) else 0
        if run >= run_length:
            return index - run_length + 1
    return None


def lifecycle_motion_scores(
    frame_files: list[Path],
    masks: dict[int, np.ndarray],
    start_frame: int,
    end_frame: int,
) -> dict[int, float]:
    """Measure object motion after subtracting median global camera flow."""
    import cv2

    width, height = 384, 216
    previous = cv2.resize(
        cv2.imread(
            str(frame_files[start_frame]), cv2.IMREAD_GRAYSCALE
        ),
        (width, height),
    )
    scores: dict[int, float] = {start_frame: 0.0}
    for frame_index in range(start_frame + 1, end_frame + 1):
        current = cv2.resize(
            cv2.imread(
                str(frame_files[frame_index]), cv2.IMREAD_GRAYSCALE
            ),
            (width, height),
        )
        flow = cv2.calcOpticalFlowFarneback(
            previous,
            current,
            None,
            0.5,
            3,
            21,
            3,
            5,
            1.2,
            0,
        )
        global_flow = np.median(flow.reshape(-1, 2), axis=0)
        residual = np.linalg.norm(flow - global_flow, axis=2)
        mask = masks.get(frame_index)
        if mask is None or not mask.any():
            scores[frame_index] = 0.0
        else:
            resized = cv2.resize(
                mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            scores[frame_index] = (
                float(np.median(residual[resized]))
                if resized.any()
                else 0.0
            )
        previous = current
    return scores


def detect_lifecycle_boundary(
    frame_indices: list[int],
    forward_masks: dict[int, np.ndarray],
    backward_masks: dict[int, np.ndarray],
    left_anchor_area: int,
    right_anchor_area: int,
    motion_scores: dict[int, float],
    fps: float = 30.0,
) -> dict | None:
    """Split repeated physical instances only when multiple signals agree."""
    forward_viable: list[bool] = []
    backward_viable: list[bool] = []
    disagreement: list[bool] = []
    for frame_index in frame_indices:
        forward = forward_masks.get(frame_index)
        backward = backward_masks.get(frame_index)
        forward_ok = (
            forward is not None
            and int(forward.sum())
            >= MIN_DIRECTIONAL_AREA_RATIO * left_anchor_area
        )
        backward_ok = (
            backward is not None
            and int(backward.sum())
            >= MIN_DIRECTIONAL_AREA_RATIO * right_anchor_area
        )
        forward_viable.append(forward_ok)
        backward_viable.append(backward_ok)
        disagreement.append(
            forward_ok
            and backward_ok
            and mask_iou(forward, backward)
            < DISAGREEMENT_IOU_THRESHOLD
        )

    collapse_offset = first_sustained_true(
        [not value for value in forward_viable],
        LIFECYCLE_COLLAPSE_RUN,
    )
    disagreement_offset = first_sustained_true(
        disagreement,
        LIFECYCLE_SIGNAL_RUN,
    )
    motion_values = np.asarray(
        [motion_scores.get(frame_index, 0.0) for frame_index in frame_indices],
        dtype=np.float32,
    )
    positive_motion = motion_values[motion_values > 0]
    if len(positive_motion):
        baseline = float(np.median(positive_motion))
        deviation = float(
            np.median(np.abs(positive_motion - baseline))
        )
        motion_threshold = max(1.25, baseline + 1.5 * deviation)
    else:
        motion_threshold = float("inf")
    motion_offset = first_sustained_true(
        motion_values >= motion_threshold,
        LIFECYCLE_SIGNAL_RUN,
    )

    signals = {
        "forward_collapse": collapse_offset is not None,
        "directional_disagreement": disagreement_offset is not None,
        "independent_motion": motion_offset is not None,
    }
    if sum(signals.values()) < 2 or collapse_offset is None:
        return None

    old_end_offset = max(collapse_offset - 1, 0)
    if disagreement_offset is not None:
        # When the two directions already disagree while both masks are
        # substantial, the physical-object swap is proven. Split immediately at
        # the first collapsed forward frame and let the backward track own the
        # new instance from there.
        new_start_offset = old_end_offset + 1
        split_reason = "directional_disagreement"
    elif motion_offset is not None:
        # If the forward object collapses and we see independent object motion,
        # start the new instance shortly after the replacement motion, once the
        # backward track is already stable. This keeps the rule time-based and
        # transferable across videos instead of tied to a specific anchor frame.
        stabilization = max(1, int(round(fps * REPLACEMENT_STABILIZATION_SECONDS)))
        candidate_offset = min(
            len(frame_indices) - 1,
            motion_offset + stabilization,
        )
        viable_offset = first_sustained_true(
            backward_viable[candidate_offset:],
            LIFECYCLE_SIGNAL_RUN,
        )
        if viable_offset is None:
            new_start_offset = max(
                old_end_offset + 1,
                len(frame_indices) - 1 - TRUSTED_ANCHOR_PROPAGATION,
            )
            split_reason = "anchor_proximity_fallback"
        else:
            new_start_offset = max(
                old_end_offset + 1,
                candidate_offset + viable_offset,
            )
            split_reason = "post_motion_backward_stability"
    else:
        new_start_offset = max(
            old_end_offset + 1,
            len(frame_indices) - 1 - TRUSTED_ANCHOR_PROPAGATION,
        )
        split_reason = "anchor_proximity_fallback"
    return {
        "old_end_frame": frame_indices[old_end_offset],
        "new_start_frame": frame_indices[new_start_offset],
        "split_reason": split_reason,
        "signals": signals,
        "signal_frames": {
            "forward_collapse": (
                frame_indices[collapse_offset]
                if collapse_offset is not None
                else None
            ),
            "directional_disagreement": (
                frame_indices[disagreement_offset]
                if disagreement_offset is not None
                else None
            ),
            "independent_motion": (
                frame_indices[motion_offset]
                if motion_offset is not None
                else None
            ),
        },
        "motion_threshold": motion_threshold,
    }


def anchor_directional_consistency_reasons(
    anchor_mask: np.ndarray,
    previous_forward: np.ndarray,
    next_backward: np.ndarray,
    previous_anchor_area: int,
    next_anchor_area: int,
) -> list[str]:
    previous_viable = (
        int(previous_forward.sum())
        >= MIN_DIRECTIONAL_AREA_RATIO * previous_anchor_area
    )
    next_viable = (
        int(next_backward.sum())
        >= MIN_DIRECTIONAL_AREA_RATIO * next_anchor_area
    )
    previous_iou = (
        mask_iou(anchor_mask, previous_forward) if previous_viable else 0.0
    )
    next_iou = (
        mask_iou(anchor_mask, next_backward) if next_viable else 0.0
    )
    if (
        previous_viable
        and next_viable
        and max(previous_iou, next_iou) < 0.20
    ):
        return ["adjacent_tracks_disagree"]
    if (
        not previous_viable
        and next_viable
        and next_iou < 0.20
    ):
        return ["previous_track_collapsed_next_track_disagrees"]
    if (
        previous_viable
        and not next_viable
        and previous_iou < 0.20
    ):
        return ["next_track_collapsed_previous_track_disagrees"]
    return []


def choose_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    return "mps" if torch.backends.mps.is_available() else "cpu"


SEVERE_FRAGMENT_COLLAPSE_RATIO = 0.15
FRAGMENT_COLLAPSE_SCORE_GATE = 0.72
LOW_ANCHOR_CONFIDENCE_GATE = 0.40


def select_trusted_anchors(items: list[dict]) -> tuple[list[dict], list[dict]]:
    """Reject semantic fragments before they reset Cutie's object memory."""
    accepted: list[dict] = []
    rejected: list[dict] = []
    accepted_areas: list[int] = []
    for item in sorted(items, key=lambda value: int(value["frame_index"])):
        area = int(decode_coco_rle(item["mask_rle"]).sum())
        recent_median = (
            float(np.median(accepted_areas[-5:])) if accepted_areas else area
        )
        manual = item.get("source") == "user_visual_initialization"
        label = str(item["label"])
        reasons: list[str] = []
        has_anchor_confidence = "anchor_confidence" in item
        score = float(item.get("anchor_confidence", item.get("score", 1.0)))
        low_score_gate = LOW_ANCHOR_CONFIDENCE_GATE if has_anchor_confidence else 0.50
        if score < low_score_gate and not manual:
            reasons.append(
                "low_anchor_confidence"
                if has_anchor_confidence
                else "low_semantic_score"
            )
        if (
            accepted_areas
            and area < SEVERE_FRAGMENT_COLLAPSE_RATIO * recent_median
            and score < FRAGMENT_COLLAPSE_SCORE_GATE
            and not manual
        ):
            reasons.append("fragment_area_collapse")
        if reasons:
            rejected.append(
                {
                    "frame_index": int(item["frame_index"]),
                    "object_id": int(item["object_id"]),
                    "label": label,
                    "area_px": area,
                    "recent_trusted_area_px": recent_median,
                    "fragment_collapse_ratio": SEVERE_FRAGMENT_COLLAPSE_RATIO,
                    "fragment_collapse_score_gate": FRAGMENT_COLLAPSE_SCORE_GATE,
                    "anchor_confidence_gate": LOW_ANCHOR_CONFIDENCE_GATE,
                    "reasons": reasons,
                }
            )
            continue
        accepted.append(item)
        accepted_areas.append(area)
    return accepted, rejected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--internal-size", type=int, default=480)
    parser.add_argument("--mem-every", type=int)
    parser.add_argument("--max-mem-frames", type=int)
    args = parser.parse_args()

    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict
    from torchvision.transforms.functional import to_tensor

    root = Path(__file__).resolve().parents[1]
    cutie_repo = root / "models" / "Cutie"
    sys.path.insert(0, str(cutie_repo))
    from cutie.inference.inference_core import InferenceCore
    from cutie.inference.utils.args_utils import get_dataset_cfg
    from cutie.model.cutie import CUTIE

    paths = job_paths(args.job)
    fps = source_fps(paths["root"]) or 30.0
    frames = frame_paths(args.job)
    discoveries = json.loads((paths["work"] / "anchor_discoveries.json").read_text())
    discovery_summary_path = paths["objects"] / "discovery.json"
    discovery_summary = (
        json.loads(discovery_summary_path.read_text())
        if discovery_summary_path.exists()
        else {}
    )
    target_labels = [
        str(value)
        for value in discovery_summary.get(
            "target_labels",
            discovery_summary.get("requested_labels", []),
        )
    ]
    labels: dict[int, str] = {
        int(item["object_id"]): str(item["label"])
        for item in discovery_summary.get("selected_objects", [])
    }
    raw_by_object: dict[int, list[dict]] = defaultdict(list)
    for item in discoveries:
        labels[int(item["object_id"])] = str(item["label"])
        if item.get("found") and item.get("mask_rle"):
            mask = decode_coco_rle(item["mask_rle"])
            if mask.any():
                raw_by_object[int(item["object_id"])].append(item)
    if not labels:
        raise SystemExit("No requested objects are recorded for this job.")

    by_object: dict[int, list[dict]] = {}
    rejected_anchors: list[dict] = []
    for object_id, items in raw_by_object.items():
        unique = {
            int(item["frame_index"]): item
            for item in sorted(items, key=lambda value: int(value["frame_index"]))
        }
        trusted, rejected = select_trusted_anchors(
            [unique[key] for key in sorted(unique)]
        )
        by_object[object_id] = trusted
        rejected_anchors.extend(rejected)
    save_json(
        rejected_anchors, paths["work"] / "cutie_rejected_anchors.json"
    )
    if rejected_anchors:
        print(
            "Cutie rejected unsafe object anchors: "
            + ", ".join(
                f"{item['label']}@{item['frame_index']} "
                f"({'+'.join(item['reasons'])})"
                for item in rejected_anchors
            ),
            flush=True,
        )

    config_dir = str((cutie_repo / "cutie" / "config").resolve())
    with initialize_config_dir(
        version_base="1.3.2",
        config_dir=config_dir,
        job_name="gradyn_cutie",
    ):
        cfg = compose(config_name="eval_config")
    weights = cutie_repo / "weights" / "cutie-base-mega.pth"
    with open_dict(cfg):
        cfg.dataset = "generic"
        cfg.weights = str(weights)
        cfg.amp = False
    get_dataset_cfg(cfg)

    device_name = choose_device(args.device)
    mem_every = (
        int(args.mem_every)
        if args.mem_every is not None
        else (10 if device_name == "mps" else int(cfg.mem_every))
    )
    max_mem_frames = (
        int(args.max_mem_frames)
        if args.max_mem_frames is not None
        else (3 if device_name == "mps" else int(cfg.max_mem_frames))
    )
    if mem_every < 1:
        raise SystemExit("--mem-every must be at least 1")
    if max_mem_frames < 2:
        raise SystemExit("--max-mem-frames must be at least 2")
    with open_dict(cfg):
        cfg.mem_every = mem_every
        cfg.max_mem_frames = max_mem_frames
    print(
        f"Cutie tracking on {device_name} at {args.internal_size}px; "
        f"mem_every={mem_every}, max_mem_frames={max_mem_frames}; "
        "one object and one anchor interval at a time.",
        flush=True,
    )
    device = torch.device(device_name)
    model = CUTIE(cfg).to(device).eval()
    model.load_weights(torch.load(weights, map_location="cpu", weights_only=True))

    cache_root = paths["work"] / "cutie_segments"
    cache_root.mkdir(parents=True, exist_ok=True)
    masks_by_object: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    confidence_by_object: dict[int, dict[int, float]] = defaultdict(dict)
    source_by_object: dict[int, dict[int, str]] = defaultdict(dict)
    agreement_by_object: dict[int, dict[int, float]] = defaultdict(dict)
    component_fraction_by_object: dict[int, dict[int, float]] = defaultdict(dict)
    instance_by_object: dict[int, dict[int, int]] = defaultdict(dict)
    lifecycle_boundaries: list[dict] = []

    def run_direction(
        object_id: int,
        label: str,
        anchor: dict,
        start_frame: int,
        end_frame: int,
        direction: str,
        return_frames: set[int] | None = None,
    ) -> tuple[dict[int, np.ndarray], dict[int, float]]:
        cache_file = cache_root / segment_cache_name(
            object_id,
            anchor,
            start_frame,
            end_frame,
            args.internal_size,
            direction,
            int(cfg.mem_every),
            int(cfg.max_mem_frames),
        )
        if cache_file.exists():
            with np.load(cache_file) as cached:
                indices = np.asarray(cached["frame_indices"], dtype=np.int32)
                masks = unpack_masks(
                    np.asarray(cached["masks_packed"], dtype=np.uint8),
                    int(np.asarray(cached["mask_width"]).item()),
                )
                confidences = np.asarray(
                    cached["confidence"], dtype=np.float32
                )
            print(
                f"↷ Cutie reusing {label} {direction} interval "
                f"{start_frame}-{end_frame}",
                flush=True,
            )
        elif end_frame - start_frame + 1 > MAX_UNCACHED_INTERVAL_FRAMES:
            print(
                f"Cutie skipping uncached {label} {direction} interval "
                f"{start_frame}-{end_frame} "
                f"({end_frame - start_frame + 1} frames exceeds "
                f"{MAX_UNCACHED_INTERVAL_FRAMES}); this sparse-anchor gap "
                "needs a closer generated/manual anchor.",
                flush=True,
            )
            anchor_mask = decode_coco_rle(anchor["mask_rle"])
            empty = np.zeros_like(anchor_mask, dtype=bool)
            indices = np.arange(start_frame, end_frame + 1, dtype=np.int32)
            masks = [empty] * len(indices)
            seed_position = 0 if direction == "forward" else -1
            masks[seed_position] = anchor_mask
            confidences = np.zeros(len(indices), dtype=np.float32)
            confidences[seed_position] = float(anchor.get("score", 1.0))
        else:
            print(
                f"Cutie {label} {direction} interval "
                f"{start_frame}-{end_frame}",
                flush=True,
            )
            processor = InferenceCore(model, cfg=model.cfg)
            processor.max_internal_size = args.internal_size
            anchor_mask = decode_coco_rle(anchor["mask_rle"])
            seed = torch.from_numpy(
                anchor_mask.astype(np.int64) * object_id
            ).to(device)
            frame_order = (
                range(start_frame, end_frame + 1)
                if direction == "forward"
                else range(end_frame, start_frame - 1, -1)
            )
            seed_frame = (
                start_frame if direction == "forward" else end_frame
            )
            indices_list: list[int] = []
            masks = []
            confidence_list: list[float] = []
            for progress, frame_index in enumerate(frame_order, start=1):
                image = to_tensor(
                    Image.open(frames[frame_index]).convert("RGB")
                ).to(device).float()
                if frame_index == seed_frame:
                    probability = processor.step(
                        image, seed, objects=[object_id]
                    )
                    mask = anchor_mask
                    confidence = float(anchor.get("score", 1.0))
                else:
                    probability = processor.step(image)
                    class_mask = processor.output_prob_to_mask(probability)
                    mask = (
                        class_mask.detach().cpu().numpy().astype(np.int32)
                        == object_id
                    )
                    # Each processor owns one physical object, so Cutie's
                    # temporary foreground channel is always channel 1.
                    foreground = probability[1]
                    confidence = (
                        float(
                            foreground[
                                torch.from_numpy(mask).to(device)
                            ].mean()
                        )
                        if mask.any()
                        else 0.0
                    )
                indices_list.append(frame_index)
                masks.append(mask)
                confidence_list.append(confidence)
                del image, probability
                if progress == 1 or progress % 30 == 0:
                    print(
                        f"  {direction} frame {frame_index} "
                        f"({progress}/{end_frame - start_frame + 1})",
                        flush=True,
                    )
            order = np.argsort(indices_list)
            indices = np.asarray(indices_list, dtype=np.int32)[order]
            masks = [masks[index] for index in order]
            confidences = np.asarray(
                confidence_list, dtype=np.float32
            )[order]
            del processor
            gc.collect()
            if device_name == "mps":
                torch.mps.synchronize()
                torch.mps.empty_cache()
            packed, width = pack_masks(masks)
            save_npz_atomic(
                cache_file,
                frame_indices=indices,
                masks_packed=packed,
                mask_width=np.asarray(width, np.int32),
                confidence=confidences,
            )
        wanted_frames = return_frames
        if wanted_frames is not None:
            wanted_frames = {int(frame_index) for frame_index in wanted_frames}
        return (
            {
                int(frame_index): mask
                for frame_index, mask in zip(indices, masks, strict=True)
                if wanted_frames is None or int(frame_index) in wanted_frames
            },
            {
                int(frame_index): float(confidence)
                for frame_index, confidence in zip(
                    indices, confidences, strict=True
                )
                if wanted_frames is None or int(frame_index) in wanted_frames
            },
        )

    with torch.inference_mode():
        for object_id, anchors in sorted(by_object.items()):
            if not anchors:
                continue
            label = str(anchors[0]["label"])
            adjacent_directions: dict[
                tuple[int, int], tuple[np.ndarray, np.ndarray]
            ] = {}
            for anchor_number in range(len(anchors) - 1):
                left_anchor = anchors[anchor_number]
                right_anchor = anchors[anchor_number + 1]
                start_frame = int(left_anchor["frame_index"])
                end_frame = int(right_anchor["frame_index"])
                forward = run_direction(
                    object_id,
                    label,
                    left_anchor,
                    start_frame,
                    end_frame,
                    "forward",
                    return_frames={end_frame},
                )
                backward = run_direction(
                    object_id,
                    label,
                    right_anchor,
                    start_frame,
                    end_frame,
                    "backward",
                    return_frames={start_frame},
                )
                adjacent_directions[(start_frame, end_frame)] = (
                    forward[0][end_frame].copy(),
                    backward[0][start_frame].copy(),
                )
                del forward, backward
                gc.collect()

            inconsistent_frames: set[int] = set()
            for anchor_number in range(1, len(anchors) - 1):
                previous_anchor = anchors[anchor_number - 1]
                anchor = anchors[anchor_number]
                next_anchor = anchors[anchor_number + 1]
                previous_frame = int(previous_anchor["frame_index"])
                anchor_frame = int(anchor["frame_index"])
                next_frame = int(next_anchor["frame_index"])
                previous_forward = adjacent_directions[
                    (previous_frame, anchor_frame)
                ][0]
                next_backward = adjacent_directions[
                    (anchor_frame, next_frame)
                ][1]
                anchor_mask = decode_coco_rle(anchor["mask_rle"])
                reasons = anchor_directional_consistency_reasons(
                    anchor_mask,
                    previous_forward,
                    next_backward,
                    int(
                        decode_coco_rle(
                            previous_anchor["mask_rle"]
                        ).sum()
                    ),
                    int(
                        decode_coco_rle(next_anchor["mask_rle"]).sum()
                    ),
                )
                if not reasons:
                    continue
                inconsistent_frames.add(anchor_frame)
                rejected_anchors.append(
                    {
                        "frame_index": anchor_frame,
                        "object_id": object_id,
                        "label": label,
                        "area_px": int(anchor_mask.sum()),
                        "recent_trusted_area_px": None,
                        "reasons": reasons,
                    }
                )
                print(
                    f"Cutie rejected directionally inconsistent anchor "
                    f"{label}@{anchor_frame} ({'+'.join(reasons)})",
                    flush=True,
                )
            if inconsistent_frames:
                anchors = [
                    anchor
                    for anchor in anchors
                    if int(anchor["frame_index"]) not in inconsistent_frames
                ]
                by_object[object_id] = anchors

            current_instance = 1
            first_anchor = anchors[0]
            first_anchor_area = int(decode_coco_rle(first_anchor["mask_rle"]).sum())
            head_end = int(first_anchor["frame_index"])
            if head_end > 0:
                head_masks, head_confidences = run_direction(
                    object_id,
                    label,
                    first_anchor,
                    0,
                    head_end,
                    "backward",
                )
                for frame_index in range(0, head_end + 1):
                    if frame_index == head_end:
                        mask = decode_coco_rle(first_anchor["mask_rle"])
                        component_fraction = 1.0
                        source = "object_anchor"
                    else:
                        mask, component_fraction = primary_component(
                            head_masks[frame_index]
                        )
                        if component_fraction < 0.65 or not propagated_mask_is_plausible(
                            mask, first_anchor_area
                        ):
                            mask = None
                        source = (
                            "cutie_backward_head"
                            if mask is not None
                            else "cutie_incoherent"
                        )
                    if mask is not None and mask.any():
                        masks_by_object[object_id][frame_index] = mask
                    confidence_by_object[object_id][frame_index] = (
                        head_confidences[frame_index]
                    )
                    source_by_object[object_id][frame_index] = source
                    instance_by_object[object_id][frame_index] = current_instance
                    agreement_by_object[object_id][frame_index] = 0.0
                    component_fraction_by_object[object_id][
                        frame_index
                    ] = component_fraction

            for anchor_number in range(len(anchors) - 1):
                left_anchor = anchors[anchor_number]
                right_anchor = anchors[anchor_number + 1]
                start_frame = int(left_anchor["frame_index"])
                end_frame = int(right_anchor["frame_index"])
                if end_frame <= start_frame:
                    continue
                forward_masks, forward_confidences = run_direction(
                    object_id,
                    label,
                    left_anchor,
                    start_frame,
                    end_frame,
                    "forward",
                )
                backward_masks, backward_confidences = run_direction(
                    object_id,
                    label,
                    right_anchor,
                    start_frame,
                    end_frame,
                    "backward",
                )
                interval_frames = list(range(start_frame, end_frame + 1))
                motion_scores = (
                    lifecycle_motion_scores(
                        frames,
                        backward_masks,
                        start_frame,
                        end_frame,
                    )
                    if end_frame - start_frame > TRUSTED_ANCHOR_PROPAGATION * 2
                    else {}
                )
                lifecycle = detect_lifecycle_boundary(
                    interval_frames,
                    forward_masks,
                    backward_masks,
                    int(decode_coco_rle(left_anchor["mask_rle"]).sum()),
                    int(decode_coco_rle(right_anchor["mask_rle"]).sum()),
                    motion_scores,
                    fps=fps,
                )
                if lifecycle is not None:
                    new_instance = current_instance + 1
                    lifecycle_boundaries.append(
                        {
                            "object_id": object_id,
                            "label": label,
                            "old_instance_id": current_instance,
                            "new_instance_id": new_instance,
                            "interval_start_frame": start_frame,
                            "interval_end_frame": end_frame,
                            **lifecycle,
                        }
                    )
                    print(
                        f"Cutie lifecycle split for {label}: "
                        f"instance {current_instance} ends at "
                        f"{lifecycle['old_end_frame']}; frames "
                        f"{lifecycle['old_end_frame'] + 1}-"
                        f"{lifecycle['new_start_frame'] - 1} quarantined; "
                        f"instance {new_instance} starts at "
                        f"{lifecycle['new_start_frame']}.",
                        flush=True,
                    )
                else:
                    new_instance = current_instance

                for frame_index in range(start_frame, end_frame + 1):
                    if lifecycle is not None and frame_index <= lifecycle[
                        "old_end_frame"
                    ]:
                        mask, component_fraction = primary_component(
                            forward_masks[frame_index]
                        )
                        agreement = mask_iou(
                            forward_masks[frame_index],
                            backward_masks[frame_index],
                        )
                        confidence = forward_confidences[frame_index]
                        source = (
                            "object_anchor"
                            if frame_index == start_frame
                            else "cutie_lifecycle_forward"
                        )
                        instance_id = current_instance
                    elif lifecycle is not None and frame_index >= lifecycle[
                        "new_start_frame"
                    ]:
                        mask, component_fraction = primary_component(
                            backward_masks[frame_index]
                        )
                        agreement = mask_iou(
                            forward_masks[frame_index],
                            backward_masks[frame_index],
                        )
                        confidence = backward_confidences[frame_index]
                        source = (
                            "object_anchor"
                            if frame_index == end_frame
                            else "cutie_lifecycle_backward"
                        )
                        instance_id = new_instance
                    elif lifecycle is not None:
                        mask = None
                        agreement = mask_iou(
                            forward_masks[frame_index],
                            backward_masks[frame_index],
                        )
                        component_fraction = 0.0
                        confidence = 0.0
                        source = "lifecycle_quarantine"
                        instance_id = 0
                    elif frame_index == start_frame:
                        mask = decode_coco_rle(left_anchor["mask_rle"])
                        agreement = 1.0
                        component_fraction = 1.0
                        confidence = float(left_anchor.get("score", 1.0))
                        source = "object_anchor"
                        instance_id = current_instance
                    elif frame_index == end_frame:
                        mask = decode_coco_rle(right_anchor["mask_rle"])
                        agreement = 1.0
                        component_fraction = 1.0
                        confidence = float(right_anchor.get("score", 1.0))
                        source = "object_anchor"
                        instance_id = current_instance
                    else:
                        mask, agreement, component_fraction = (
                            fuse_directional_masks(
                                forward_masks.get(frame_index),
                                backward_masks.get(frame_index),
                            )
                        )
                        forward_present = bool(
                            forward_masks.get(frame_index) is not None
                            and forward_masks[frame_index].any()
                        )
                        backward_present = bool(
                            backward_masks.get(frame_index) is not None
                            and backward_masks[frame_index].any()
                        )
                        confidence = float(
                            np.mean(
                                [
                                    forward_confidences.get(frame_index, 0.0),
                                    backward_confidences.get(frame_index, 0.0),
                                ]
                            )
                        )
                        if mask is None:
                            source = "cutie_incoherent"
                        elif forward_present and backward_present:
                            source = "cutie_bidirectional"
                        elif forward_present:
                            source = "cutie_forward_only"
                        else:
                            source = "cutie_backward_only"
                        instance_id = current_instance
                    if mask is not None:
                        masks_by_object[object_id][frame_index] = mask
                    confidence_by_object[object_id][frame_index] = confidence
                    source_by_object[object_id][frame_index] = source
                    instance_by_object[object_id][
                        frame_index
                    ] = instance_id
                    agreement_by_object[object_id][frame_index] = agreement
                    component_fraction_by_object[object_id][
                        frame_index
                    ] = component_fraction
                current_instance = new_instance

            last_anchor = anchors[-1]
            last_anchor_area = int(decode_coco_rle(last_anchor["mask_rle"]).sum())
            tail_start = int(last_anchor["frame_index"])
            tail_end = len(frames) - 1
            if tail_start <= tail_end:
                tail_masks, tail_confidences = run_direction(
                    object_id,
                    label,
                    last_anchor,
                    tail_start,
                    tail_end,
                    "forward",
                )
                for frame_index in range(tail_start, tail_end + 1):
                    if frame_index == tail_start:
                        mask = decode_coco_rle(last_anchor["mask_rle"])
                        component_fraction = 1.0
                        source = "object_anchor"
                    else:
                        mask, component_fraction = primary_component(
                            tail_masks[frame_index]
                        )
                        if component_fraction < 0.65 or not propagated_mask_is_plausible(
                            mask, last_anchor_area
                        ):
                            mask = None
                        source = (
                            "cutie_forward_tail"
                            if mask is not None
                            else "cutie_incoherent"
                        )
                    if mask is not None and mask.any():
                        masks_by_object[object_id][frame_index] = mask
                    confidence_by_object[object_id][frame_index] = (
                        tail_confidences[frame_index]
                    )
                    source_by_object[object_id][frame_index] = source
                    instance_by_object[object_id][
                        frame_index
                    ] = current_instance
                    agreement_by_object[object_id][frame_index] = 0.0
                    component_fraction_by_object[object_id][
                        frame_index
                    ] = component_fraction

    save_json(
        rejected_anchors, paths["work"] / "cutie_rejected_anchors.json"
    )
    del model
    if device_name == "mps":
        torch.mps.empty_cache()

    final_labels, clip_label_details = clip_label_tracks(
        root=Path(__file__).resolve().parents[1],
        frames=frames,
        masks_by_object=masks_by_object,
        confidence_by_object=confidence_by_object,
        fallback_labels=labels,
        target_labels=target_labels,
        device_name=device_name,
    )
    labels = final_labels
    if clip_label_details:
        save_json(
            {
                "backend": "clip-vit-base-patch32",
                "target_labels": target_labels,
                "probability_threshold": CLIP_LABEL_PROB_THRESHOLD,
                "tracks": list(clip_label_details.values()),
            },
            paths["work"] / "clip_track_labels.json",
        )
        selected_objects = discovery_summary.get("selected_objects", [])
        for item in selected_objects:
            object_id = int(item["object_id"])
            if object_id in clip_label_details:
                item["pre_clip_label"] = item.get("pre_clip_label", item.get("label"))
                item["label"] = clip_label_details[object_id]["label"]
                item["clip_score"] = clip_label_details[object_id]["clip_score"]
                item["clip_accepted"] = clip_label_details[object_id]["accepted"]
        discovery_summary["selected_objects"] = selected_objects
        discovery_summary["labeling_backend"] = "clip-vit-base-patch32"
        save_json(discovery_summary, discovery_summary_path)

    timestamp_table = pq.read_table(
        paths["source"] / "frame_timestamps.parquet"
    ).to_pydict()
    timestamps = timestamp_table["timestamp_s"]
    rows: list[dict] = []
    annotations: list[dict] = []
    quality_frames: list[dict] = []
    rejected_frames: list[tuple[int, list[str]]] = []
    overlay_frames = paths["work"] / "object_overlay_frames"
    shutil.rmtree(overlay_frames, ignore_errors=True)
    overlay_frames.mkdir(parents=True)

    for frame_index, frame_path in enumerate(frames):
        image = Image.open(frame_path).convert("RGB")
        rendered = image
        reasons: list[str] = []
        confidences: list[float] = []
        for object_id, label in sorted(labels.items()):
            anchors = by_object.get(object_id, [])
            mask = masks_by_object[object_id].get(frame_index)
            confidence = confidence_by_object[object_id].get(frame_index, 0.0)
            first_anchor = int(anchors[0]["frame_index"]) if anchors else None
            if mask is None:
                visibility = (
                    "out_of_frame"
                    if first_anchor is not None and frame_index < first_anchor
                    else "rejected"
                )
                bbox = [0, 0, 0, 0]
                area = 0
                if visibility == "rejected":
                    reason = (
                        f"object_{object_id}_lifecycle_ambiguous"
                        if source_by_object[object_id].get(frame_index)
                        == "lifecycle_quarantine"
                        else f"object_{object_id}_missing_track"
                    )
                    reasons.append(reason)
            else:
                area = int(mask.sum())
                bbox = mask_bbox(mask)
                if area == 0:
                    visibility = "fully_occluded"
                elif (
                    bbox[0] <= 1
                    or bbox[1] <= 1
                    or bbox[0] + bbox[2] >= image.width - 1
                    or bbox[1] + bbox[3] >= image.height - 1
                ):
                    visibility = "partially_occluded"
                else:
                    visibility = "visible"
                if area:
                    annotations.append(
                        {
                            "frame_index": frame_index,
                            "object_id": object_id,
                            "instance_id": instance_by_object[object_id].get(
                                frame_index
                            ),
                            "rle": encode_coco_rle(mask),
                        }
                    )
                    rendered = overlay_mask(
                        rendered,
                        mask,
                        color_for_id(object_id),
                        f"{label} #{object_id}."
                        f"{instance_by_object[object_id].get(frame_index, 0)} "
                        f"{confidence:.2f}",
                        bbox,
                    )
                if visibility not in {"fully_occluded", "out_of_frame"}:
                    confidences.append(confidence)
            rows.append(
                {
                    "frame_index": frame_index,
                    "timestamp_s": float(timestamps[frame_index]),
                    "object_id": object_id,
                    "instance_id": (
                        instance_by_object[object_id].get(frame_index) or None
                    ),
                    "label": label,
                    "bbox_x": bbox[0],
                    "bbox_y": bbox[1],
                    "bbox_width": bbox[2],
                    "bbox_height": bbox[3],
                    "mask_area_px": area,
                    "visibility": visibility,
                    "confidence": confidence,
                    "source": source_by_object[object_id].get(
                        frame_index, "untracked"
                    ),
                    "directional_iou": agreement_by_object[object_id].get(
                        frame_index, 0.0
                    ),
                    "primary_component_fraction": (
                        component_fraction_by_object[object_id].get(
                            frame_index, 0.0
                        )
                    ),
                }
            )
        overall = min(confidences) if confidences else (0.0 if reasons else 0.75)
        status = "rejected" if reasons else "accepted"
        if reasons:
            rejected_frames.append((frame_index, reasons))
        quality_frames.append(
            {
                "frame_index": frame_index,
                "confidence": overall,
                "status": status,
                "reasons": reasons,
            }
        )
        rendered.save(
            overlay_frames / f"{frame_index:08d}.jpg", quality=90
        )

    write_parquet(rows, paths["objects"] / "tracks.parquet")
    save_json(
        {"annotations": annotations}, paths["objects"] / "masks.json"
    )
    save_json(
        {
            "frames": quality_frames,
            "quarantined_intervals": merge_intervals(rejected_frames),
        },
        paths["objects"] / "quality.json",
    )
    save_json(
        lifecycle_boundaries,
        paths["objects"] / "lifecycle_boundaries.json",
    )
    render_video(
        overlay_frames,
        paths["objects"] / "overlay.mp4",
        fps,
    )
    print(
        f"Cutie interval checkpoints retained at {cache_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
