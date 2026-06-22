from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

try:
    from common import (
        chunk_ranges,
        color_for_id,
        decode_coco_rle,
        encode_coco_rle,
        frame_paths,
        iou,
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
        chunk_ranges,
        color_for_id,
        decode_coco_rle,
        encode_coco_rle,
        frame_paths,
        iou,
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


def prepare_chunk_directory(
    frames: list[Path], directory: Path, start: int, end: int
) -> None:
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True)
    for local_index, source in enumerate(frames[start:end]):
        (directory / f"{local_index:08d}.jpg").symlink_to(source.resolve())


def choose_device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def choose_bidirectional_mask(
    forward_mask: np.ndarray | None,
    reverse_mask: np.ndarray | None,
    frame_index: int,
    anchor_frames: list[int],
) -> tuple[np.ndarray | None, float]:
    """Prefer a healthy directional track instead of blindly taking forward."""
    if forward_mask is None:
        return reverse_mask, 0.75 if reverse_mask is not None else 0.0
    if reverse_mask is None:
        return forward_mask, 0.75
    forward_area = int(forward_mask.sum())
    reverse_area = int(reverse_mask.sum())
    agreement = iou(forward_mask, reverse_mask)
    if forward_area == 0 and reverse_area > 0:
        return reverse_mask, 0.75
    if reverse_area == 0 and forward_area > 0:
        return forward_mask, 0.75
    if agreement >= 0.5 or forward_area == reverse_area:
        return forward_mask, agreement
    previous_anchors = [value for value in anchor_frames if value <= frame_index]
    next_anchors = [value for value in anchor_frames if value >= frame_index]
    forward_distance = (
        frame_index - max(previous_anchors) if previous_anchors else float("inf")
    )
    reverse_distance = (
        min(next_anchors) - frame_index if next_anchors else float("inf")
    )
    return (
        (forward_mask, agreement)
        if forward_distance <= reverse_distance
        else (reverse_mask, agreement)
    )


def confirmed_absence_intervals(
    discoveries: list[dict],
    *,
    minimum_consecutive_misses: int = 2,
) -> list[tuple[int, int]]:
    """Convert repeated SAM 3 misses into conservative object-absence spans."""
    ordered = sorted(discoveries, key=lambda item: int(item["frame_index"]))
    intervals: list[tuple[int, int]] = []
    index = 0
    while index < len(ordered):
        if ordered[index].get("found"):
            index += 1
            continue
        run_start = index
        while index < len(ordered) and not ordered[index].get("found"):
            index += 1
        run_end = index - 1
        if run_end - run_start + 1 < minimum_consecutive_misses:
            continue
        first_missing = int(ordered[run_start]["frame_index"])
        last_missing = int(ordered[run_end]["frame_index"])
        previous_found = next(
            (
                int(ordered[previous]["frame_index"])
                for previous in range(run_start - 1, -1, -1)
                if ordered[previous].get("found")
            ),
            None,
        )
        next_found = next(
            (
                int(ordered[following]["frame_index"])
                for following in range(index, len(ordered))
                if ordered[following].get("found")
            ),
            None,
        )
        start = (
            (previous_found + first_missing + 1) // 2
            if previous_found is not None
            else first_missing
        )
        end = (
            (last_missing + next_found) // 2
            if next_found is not None
            else last_missing
        )
        intervals.append((start, end))
    return intervals


def frame_is_absent(frame_index: int, intervals: list[tuple[int, int]]) -> bool:
    return any(start <= frame_index <= end for start, end in intervals)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--chunk-frames", type=int, default=180)
    parser.add_argument("--overlap", type=int, default=16)
    args = parser.parse_args()

    import torch

    root = Path(__file__).resolve().parents[1]
    sam2_repo = root / "models" / "sam2"
    checkpoint = sam2_repo / "checkpoints" / "sam2.1_hiera_small.pt"
    sys.path.insert(0, str(sam2_repo))
    from sam2.build_sam import build_sam2_video_predictor
    import sam2.utils.misc as sam2_misc

    # Upstream divides a uint8 NumPy image by 255.0 before torch conversion,
    # which NumPy promotes to float64. MPS cannot transfer float64 tensors.
    original_load_img = sam2_misc._load_img_as_tensor

    def load_img_float32(img_path, image_size):
        image, height, width = original_load_img(img_path, image_size)
        return image.float(), height, width

    sam2_misc._load_img_as_tensor = load_img_float32

    paths = job_paths(args.job)
    frames = frame_paths(args.job)
    discoveries_path = paths["work"] / "sam3_discoveries.json"
    discoveries = json.loads(discoveries_path.read_text())
    by_object: dict[int, list[dict]] = defaultdict(list)
    all_by_object: dict[int, list[dict]] = defaultdict(list)
    for item in discoveries:
        all_by_object[int(item["object_id"])].append(item)
        if item.get("found"):
            by_object[int(item["object_id"])].append(item)
    if not by_object:
        raise SystemExit("SAM 3 found none of the requested objects.")

    device = choose_device()
    print(
        f"SAM 2 tracking device: {device}; small backbone at 512px, "
        "one object per state.",
        flush=True,
    )
    predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        str(checkpoint),
        device=device,
        hydra_overrides_extra=[
            "++model.image_size=512",
            "++model.memory_attention.layer.self_attention.feat_sizes=[32,32]",
            "++model.memory_attention.layer.cross_attention.feat_sizes=[32,32]",
        ],
        apply_postprocessing=False,
    )
    labels: dict[int, str] = {}
    effective_chunk_frames = min(args.chunk_frames, 90)
    effective_overlap = min(args.overlap, effective_chunk_frames - 1)
    ranges = chunk_ranges(
        len(frames), effective_chunk_frames, effective_overlap
    )
    chunk_root = paths["work"] / "sam2_chunks"
    shutil.rmtree(chunk_root, ignore_errors=True)
    chunk_root.mkdir(parents=True)
    checkpoint_root = paths["work"] / "sam2_checkpoints"
    checkpoint_meta = checkpoint_root / "metadata.json"
    checkpoint_signature = {
        "frame_count": len(frames),
        "ranges": ranges,
        "objects": {
            str(object_id): items[0]["label"] for object_id, items in by_object.items()
        },
        "checkpoint": checkpoint.name,
        "image_size": 512,
        "discoveries_sha256": hashlib.sha256(
            discoveries_path.read_bytes()
        ).hexdigest(),
        "negative_anchor_policy": "two_misses_midpoint_absence_v1",
    }
    if checkpoint_meta.exists():
        existing_signature = json.loads(checkpoint_meta.read_text()).get(
            "signature", {}
        )
        # `config_hash` existed briefly in development builds. It was too broad:
        # changing an unrelated downstream option such as depth sampling could
        # invalidate valid SAM 2 inference. Ignore it in both old and new
        # metadata and key compatibility only to SAM 2's actual inputs.
        existing_signature.pop("config_hash", None)
        existing_objects = existing_signature.pop("objects", {})
        new_objects = checkpoint_signature["objects"]
        comparable_signature = {
            key: value
            for key, value in checkpoint_signature.items()
            if key != "objects"
        }
        if existing_signature != comparable_signature:
            shutil.rmtree(checkpoint_root)
        else:
            selected_ids = {int(value) for value in new_objects}
            for checkpoint_file in checkpoint_root.glob("object_*_[fr]_*.npz"):
                object_id = int(checkpoint_file.stem.split("_")[1])
                if object_id not in selected_ids:
                    checkpoint_file.unlink()
            fused_root_to_prune = checkpoint_root / "fused"
            if fused_root_to_prune.exists():
                for object_dir in fused_root_to_prune.glob("object_*"):
                    object_id = int(object_dir.name.split("_")[1])
                    if object_id not in selected_ids:
                        shutil.rmtree(object_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    save_json({"signature": checkpoint_signature}, checkpoint_meta)

    def chunk_checkpoint(
        object_id: int, reverse_direction: bool, chunk_number: int
    ) -> Path:
        return checkpoint_root / (
            f"object_{object_id}_{'r' if reverse_direction else 'f'}_"
            f"{chunk_number:05d}.npz"
        )

    def load_packed_checkpoint(
        checkpoint_file: Path,
    ) -> tuple[dict[int, int], np.ndarray, int] | None:
        if not checkpoint_file.exists():
            return None
        with np.load(checkpoint_file) as cached:
            indices = np.asarray(cached["frame_indices"], dtype=np.int32)
            if "masks_packed" in cached:
                packed = np.asarray(cached["masks_packed"], dtype=np.uint8)
                width = int(np.asarray(cached["mask_width"]).item())
            else:
                masks = np.asarray(cached["masks"], dtype=np.uint8)
                width = int(masks.shape[-1])
                packed = np.packbits(masks, axis=-1)
        return (
            {int(frame_index): row for row, frame_index in enumerate(indices)},
            packed,
            width,
        )

    def packed_mask_at(
        cache: tuple[dict[int, int], np.ndarray, int] | None,
        frame_index: int,
    ) -> np.ndarray | None:
        if cache is None:
            return None
        lookup, packed, width = cache
        row = lookup.get(frame_index)
        if row is None:
            return None
        return np.unpackbits(packed[row], axis=-1)[..., :width].astype(bool)

    def propagate_chunks(
        object_id: int,
        items: list[dict],
        reverse_direction: bool,
    ) -> None:
        ordered_ranges = list(reversed(ranges)) if reverse_direction else ranges
        carry_frame: int | None = None
        carry_mask: np.ndarray | None = None
        for chunk_number, (start, end) in enumerate(ordered_ranges):
            checkpoint_file = chunk_checkpoint(
                object_id, reverse_direction, chunk_number
            )
            next_carry_frame = None
            if chunk_number + 1 < len(ordered_ranges):
                next_carry_frame = (
                    start
                    if reverse_direction
                    else ordered_ranges[chunk_number + 1][0]
                )
            if checkpoint_file.exists():
                print(
                    f"↷ SAM 2 reusing {'reverse' if reverse_direction else 'forward'} "
                    f"object {object_id}, chunk {chunk_number + 1}/{len(ordered_ranges)}",
                    flush=True,
                )
                cached = load_packed_checkpoint(checkpoint_file)
                carry_frame = next_carry_frame
                carry_mask = (
                    packed_mask_at(cached, next_carry_frame)
                    if next_carry_frame is not None
                    else None
                )
                del cached
                continue
            local_items = [
                item for item in items if start <= int(item["frame_index"]) < end
            ]
            if not local_items and carry_mask is None:
                carry_frame = next_carry_frame
                carry_mask = None
                continue

            chunk_dir = chunk_root / (
                f"object_{object_id}_{'r' if reverse_direction else 'f'}_{chunk_number}"
            )
            prepare_chunk_directory(frames, chunk_dir, start, end)
            state = predictor.init_state(
                video_path=str(chunk_dir),
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
                async_loading_frames=False,
            )
            local_output: dict[int, np.ndarray] = {}
            conditioning_indices: list[int] = []
            if (
                carry_mask is not None
                and carry_frame is not None
                and start <= carry_frame < end
            ):
                local_index = carry_frame - start
                predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=local_index,
                    obj_id=object_id,
                    mask=torch.from_numpy(carry_mask),
                )
                conditioning_indices.append(local_index)
                local_output[carry_frame] = carry_mask
            for detection in sorted(
                local_items, key=lambda value: value["frame_index"]
            ):
                local_index = int(detection["frame_index"]) - start
                detection_mask = decode_coco_rle(detection["mask_rle"])
                predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=local_index,
                    obj_id=object_id,
                    mask=torch.from_numpy(detection_mask),
                )
                conditioning_indices.append(local_index)
                local_output[int(detection["frame_index"])] = detection_mask

            propagation_start = (
                max(conditioning_indices) if reverse_direction else min(conditioning_indices)
            )
            print(
                f"SAM 2 {'reverse' if reverse_direction else 'forward'} "
                f"object {object_id}, chunk {chunk_number + 1}/{len(ordered_ranges)} "
                f"frames {start}-{end - 1}",
                flush=True,
            )
            for local_index, _, logits in predictor.propagate_in_video(
                state,
                start_frame_idx=propagation_start,
                reverse=reverse_direction,
            ):
                global_index = start + int(local_index)
                local_output[global_index] = (
                    logits[0, 0].detach().cpu().numpy() > 0
                )
            predictor.reset_state(state)
            del state
            shutil.rmtree(chunk_dir, ignore_errors=True)
            cached_frames = sorted(local_output)
            if cached_frames:
                stacked_masks = np.stack(
                    [local_output[frame_index] for frame_index in cached_frames]
                ).astype(np.uint8)
                save_npz_atomic(
                    checkpoint_file,
                    frame_indices=np.asarray(cached_frames, np.int32),
                    masks_packed=np.packbits(stacked_masks, axis=-1),
                    mask_width=np.asarray(stacked_masks.shape[-1], np.int32),
                )
            carry_frame = next_carry_frame
            carry_mask = (
                local_output.get(next_carry_frame)
                if next_carry_frame is not None
                else None
            )
            del local_output
            if device == "mps":
                torch.mps.synchronize()
                torch.mps.empty_cache()

    fused_root = checkpoint_root / "fused"
    fused_root.mkdir(parents=True, exist_ok=True)

    def fuse_object_chunks(object_id: int, anchor_frames: list[int]) -> None:
        object_root = fused_root / f"object_{object_id}"
        object_root.mkdir(parents=True, exist_ok=True)
        for range_index, (start, end) in enumerate(ranges):
            fused_file = object_root / f"{range_index:05d}.npz"
            if fused_file.exists():
                continue
            owned_end = ranges[range_index + 1][0] if range_index + 1 < len(ranges) else end
            forward_cache = load_packed_checkpoint(
                chunk_checkpoint(object_id, False, range_index)
            )
            reverse_chunk_number = len(ranges) - 1 - range_index
            reverse_cache = load_packed_checkpoint(
                chunk_checkpoint(object_id, True, reverse_chunk_number)
            )
            fused_frames: list[int] = []
            fused_masks: list[np.ndarray] = []
            directional_scores: list[float] = []
            for frame_index in range(start, owned_end):
                mask, directional = choose_bidirectional_mask(
                    packed_mask_at(forward_cache, frame_index),
                    packed_mask_at(reverse_cache, frame_index),
                    frame_index,
                    anchor_frames,
                )
                if mask is None:
                    continue
                fused_frames.append(frame_index)
                fused_masks.append(mask)
                directional_scores.append(float(directional))
            if fused_masks:
                stacked = np.stack(fused_masks).astype(np.uint8)
                save_npz_atomic(
                    fused_file,
                    frame_indices=np.asarray(fused_frames, np.int32),
                    masks_packed=np.packbits(stacked, axis=-1),
                    mask_width=np.asarray(stacked.shape[-1], np.int32),
                    directional=np.asarray(directional_scores, np.float32),
                )
            else:
                save_npz_atomic(
                    fused_file,
                    frame_indices=np.empty(0, np.int32),
                    masks_packed=np.empty((0, 0, 0), np.uint8),
                    mask_width=np.asarray(0, np.int32),
                    directional=np.empty(0, np.float32),
                )
            del forward_cache, reverse_cache

    with torch.inference_mode():
        for object_id, items in by_object.items():
            labels[object_id] = items[0]["label"]
            ordered_items = sorted(items, key=lambda value: value["frame_index"])
            propagate_chunks(object_id, ordered_items, False)
            propagate_chunks(object_id, ordered_items, True)
            fuse_object_chunks(
                object_id,
                [int(item["frame_index"]) for item in ordered_items],
            )
            if device == "mps":
                torch.mps.empty_cache()
    shutil.rmtree(chunk_root, ignore_errors=True)
    del predictor
    if device == "mps":
        torch.mps.empty_cache()

    timestamp_table = pq.read_table(
        paths["source"] / "frame_timestamps.parquet"
    ).to_pydict()
    timestamps = timestamp_table["timestamp_s"]

    rows: list[dict] = []
    masks_json: list[dict] = []
    frame_quality: list[dict] = []
    rejected: list[tuple[int, list[str]]] = []
    overlays = paths["work"] / "object_overlay_frames"
    shutil.rmtree(overlays, ignore_errors=True)
    overlays.mkdir(parents=True)
    previous_area: dict[int, int] = {}
    absence_by_object = {
        object_id: confirmed_absence_intervals(all_by_object[object_id])
        for object_id in labels
    }
    empty_mask_by_object = {
        object_id: np.zeros_like(
            decode_coco_rle(by_object[object_id][0]["mask_rle"])
        )
        for object_id in labels
    }
    for object_id, intervals in absence_by_object.items():
        if intervals:
            print(
                f"SAM 2 object {object_id} confirmed absent in spans: "
                + ", ".join(f"{start}-{end}" for start, end in intervals),
                flush=True,
            )
    current_range_index = -1
    current_range_end = -1
    fused_cache: dict[
        int,
        tuple[
            tuple[dict[int, int], np.ndarray, int] | None,
            dict[int, float],
        ],
    ] = {}
    for frame_index, frame_path in enumerate(frames):
        if frame_index >= current_range_end:
            current_range_index += 1
            _, range_end = ranges[current_range_index]
            current_range_end = (
                ranges[current_range_index + 1][0]
                if current_range_index + 1 < len(ranges)
                else range_end
            )
            fused_cache.clear()
            for object_id in labels:
                fused_file = (
                    fused_root
                    / f"object_{object_id}"
                    / f"{current_range_index:05d}.npz"
                )
                packed_cache = load_packed_checkpoint(fused_file)
                with np.load(fused_file) as cached:
                    indices = np.asarray(
                        cached["frame_indices"], dtype=np.int32
                    )
                    directional = np.asarray(
                        cached["directional"], dtype=np.float32
                    )
                fused_cache[object_id] = (
                    packed_cache,
                    {
                        int(source_frame): float(score)
                        for source_frame, score in zip(indices, directional)
                    },
                )
        image = Image.open(frame_path).convert("RGB")
        rendered = image
        frame_confidences: list[float] = []
        frame_reasons: list[str] = []
        for object_id, label in labels.items():
            object_anchors = sorted(
                int(item["frame_index"]) for item in by_object[object_id]
            )
            packed_cache, directional_by_frame = fused_cache[object_id]
            mask = packed_mask_at(packed_cache, frame_index)
            if frame_is_absent(frame_index, absence_by_object[object_id]):
                mask = empty_mask_by_object[object_id]
            directional = directional_by_frame.get(
                frame_index,
                0.0,
            )
            if mask is None:
                outside_anchors = (
                    frame_index < object_anchors[0] or frame_index > object_anchors[-1]
                )
                status = "out_of_frame" if outside_anchors else "uncertain"
                confidence = 0.75 if outside_anchors else 0.0
                if status == "uncertain":
                    frame_reasons.append(f"object_{object_id}_missing")
                bbox = [0, 0, 0, 0]
                area = 0
            else:
                area = int(mask.sum())
                bbox = mask_bbox(mask)
                area_ratio = (
                    min(area, previous_area.get(object_id, area))
                    / max(area, previous_area.get(object_id, area), 1)
                )
                keyframe_detection = next(
                    (
                        item
                        for item in by_object[object_id]
                        if int(item["frame_index"]) == frame_index
                    ),
                    None,
                )
                anchor_agreement = (
                    iou(mask, decode_coco_rle(keyframe_detection["mask_rle"]))
                    if keyframe_detection
                    else directional
                )
                confidence = float(
                    0.5 * directional + 0.25 * area_ratio + 0.25 * anchor_agreement
                )
                if area == 0:
                    outside_anchors = (
                        frame_index < object_anchors[0]
                        or frame_index > object_anchors[-1]
                    )
                    status = "out_of_frame" if outside_anchors else "fully_occluded"
                    # Absence is a valid visibility state, not automatically a
                    # failed frame. Confidence here describes state confidence.
                    confidence = max(confidence, 0.5)
                elif confidence < 0.45:
                    status = "uncertain"
                    frame_reasons.append(f"object_{object_id}_temporal_inconsistency")
                elif bbox[0] <= 1 or bbox[1] <= 1 or bbox[0] + bbox[2] >= image.width - 1 or bbox[1] + bbox[3] >= image.height - 1:
                    status = "partially_occluded"
                else:
                    status = "visible"
                previous_area[object_id] = area
                masks_json.append(
                    {
                        "frame_index": frame_index,
                        "object_id": object_id,
                        "rle": encode_coco_rle(mask),
                    }
                )
                rendered = overlay_mask(
                    rendered,
                    mask,
                    color_for_id(object_id),
                    f"{label} #{object_id} {confidence:.2f}",
                    bbox,
                )
            rows.append(
                {
                    "frame_index": frame_index,
                    "timestamp_s": float(timestamps[frame_index]),
                    "object_id": object_id,
                    "label": label,
                    "bbox_x": bbox[0],
                    "bbox_y": bbox[1],
                    "bbox_width": bbox[2],
                    "bbox_height": bbox[3],
                    "mask_area_px": area,
                    "visibility": status,
                    "confidence": confidence,
                }
            )
            if status not in {"fully_occluded", "out_of_frame"}:
                frame_confidences.append(confidence)
        overall = min(frame_confidences) if frame_confidences else 0.75
        status = "accepted" if overall >= 0.45 else "rejected"
        if status == "rejected":
            rejected.append((frame_index, frame_reasons or ["low_object_confidence"]))
        frame_quality.append(
            {
                "frame_index": frame_index,
                "confidence": overall,
                "status": status,
                "reasons": frame_reasons,
            }
        )
        rendered.save(overlays / f"{frame_index:08d}.jpg", quality=90)

    write_parquet(rows, paths["objects"] / "tracks.parquet")
    save_json({"annotations": masks_json}, paths["objects"] / "masks.json")
    save_json(
        {
            "frames": frame_quality,
            "quarantined_intervals": merge_intervals(rejected),
        },
        paths["objects"] / "quality.json",
    )
    render_video(overlays, paths["objects"] / "overlay.mp4", source_fps(paths["root"]))
    shutil.rmtree(checkpoint_root, ignore_errors=True)


if __name__ == "__main__":
    main()
