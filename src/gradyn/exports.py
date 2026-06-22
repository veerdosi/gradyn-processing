from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .paths import JobPaths


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _read_rows(path: Path) -> list[dict]:
    if not path.exists() or not path.stat().st_size:
        return []
    return pq.read_table(path).to_pylist()


def _frame_payloads(paths: JobPaths) -> list[dict]:
    timestamps = _read_rows(paths.source / "frame_timestamps.parquet")
    objects: dict[int, list[dict]] = {}
    for row in _read_rows(paths.objects / "tracks.parquet"):
        objects.setdefault(int(row["frame_index"]), []).append(row)
    hands: dict[int, dict[str, dict]] = {}
    for row in _read_rows(paths.hands / "trajectories.parquet"):
        hands.setdefault(int(row["frame_index"]), {})[str(row["handedness"])] = row
    depth = {
        int(row["frame_index"]): row
        for row in _read_rows(paths.depth / "depth_metadata.parquet")
    }

    payloads: list[dict] = []
    for timing in timestamps:
        frame_index = int(timing["frame_index"])
        payloads.append(
            {
                "episode_index": 0,
                "frame_index": frame_index,
                "timestamp": float(timing["timestamp_s"]),
                "objects_json": json.dumps(objects.get(frame_index, [])),
                "hands_json": json.dumps(hands.get(frame_index, {})),
                "depth_metadata_json": json.dumps(depth.get(frame_index, {})),
                "rgb_frame": f"source/frames/{frame_index:08d}.jpg",
                "depth_frame": (
                    f"depth/frames/{frame_index:08d}.npz"
                    if frame_index in depth
                    else None
                ),
            }
        )
    return payloads


def _export_coco_video(paths: JobPaths) -> None:
    destination = paths.exports / "coco_video"
    destination.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((paths.source / "video_metadata.json").read_text())
    tracks = _read_rows(paths.objects / "tracks.parquet")
    masks_data = json.loads((paths.objects / "masks.json").read_text())
    masks = {
        (int(item["frame_index"]), int(item["object_id"])): item["rle"]
        for item in masks_data.get("annotations", [])
    }
    labels = {
        int(row["object_id"]): str(row["label"])
        for row in tracks
    }
    images = [
        {
            "id": index + 1,
            "video_id": 1,
            "frame_id": index,
            "file_name": f"{index:08d}.jpg",
            "width": metadata["decoded_width"],
            "height": metadata["decoded_height"],
        }
        for index in range(metadata["frame_count"])
    ]
    annotations = []
    for annotation_id, row in enumerate(tracks, start=1):
        key = (int(row["frame_index"]), int(row["object_id"]))
        rle = masks.get(key)
        if rle is None:
            continue
        annotations.append(
            {
                "id": annotation_id,
                "image_id": int(row["frame_index"]) + 1,
                "video_id": 1,
                "category_id": int(row["object_id"]),
                "instance_id": int(row["object_id"]),
                "bbox": [
                    row["bbox_x"],
                    row["bbox_y"],
                    row["bbox_width"],
                    row["bbox_height"],
                ],
                "area": row["mask_area_px"],
                "segmentation": rle,
                "iscrowd": 0,
                "visibility": row["visibility"],
                "confidence": row["confidence"],
            }
        )
    data = {
        "info": {
            "description": "Gradyn model-derived egocentric object tracks",
            "schema_version": "1.0.0",
        },
        "videos": [{"id": 1, "file_name": Path(metadata["input_video"]).name}],
        "images": images,
        "categories": [
            {"id": object_id, "name": label}
            for object_id, label in sorted(labels.items())
        ],
        "annotations": annotations,
    }
    (destination / "annotations.json").write_text(json.dumps(data, indent=2))


def _export_lerobot(paths: JobPaths, payloads: list[dict]) -> None:
    destination = paths.exports / "lerobot"
    data_dir = destination / "data" / "chunk-000"
    meta_dir = destination / "meta"
    video_dir = destination / "videos" / "observation.images.egocentric" / "chunk-000"
    for directory in (data_dir, meta_dir, video_dir):
        directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(payloads), data_dir / "episode_000000.parquet")
    metadata = json.loads((paths.source / "video_metadata.json").read_text())
    fps = (
        len(payloads) / metadata["duration_s"]
        if metadata["duration_s"] > 0
        else 30.0
    )
    info = {
        "codebase_version": "v3.0",
        "robot_type": "human_egocentric",
        "total_episodes": 1,
        "total_frames": len(payloads),
        "total_tasks": 1,
        "fps": fps,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4",
        "features": {
            "timestamp": {"dtype": "float64", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "objects_json": {"dtype": "string", "shape": [1]},
            "hands_json": {"dtype": "string", "shape": [1]},
            "depth_metadata_json": {"dtype": "string", "shape": [1]},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2))
    pq.write_table(
        pa.Table.from_pylist(
            [{"episode_index": 0, "length": len(payloads), "task_index": 0}]
        ),
        meta_dir / "episodes.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist([{"task_index": 0, "task": "egocentric demonstration"}]),
        meta_dir / "tasks.parquet",
    )
    _link_or_copy(
        Path(metadata["input_video"]),
        video_dir / "episode_000000.mp4",
    )


def _export_rlds(paths: JobPaths, payloads: list[dict]) -> None:
    destination = paths.exports / "rlds"
    destination.mkdir(parents=True, exist_ok=True)
    steps = []
    for index, payload in enumerate(payloads):
        steps.append(
            {
                **payload,
                "is_first": index == 0,
                "is_last": index == len(payloads) - 1,
                "is_terminal": False,
            }
        )
    pq.write_table(pa.Table.from_pylist(steps), destination / "steps.parquet")
    (destination / "episode_metadata.json").write_text(
        json.dumps(
            {
                "episode_id": "episode_000000",
                "num_steps": len(steps),
                "logical_schema": "RLDS episode/steps interchange",
                "note": "Parquet interchange; convert to TFDS TFRecord only when TensorFlow consumption requires it.",
            },
            indent=2,
        )
    )


def build_exports(output: Path) -> None:
    paths = JobPaths(output)
    paths.exports.mkdir(parents=True, exist_ok=True)
    payloads = _frame_payloads(paths)
    _export_coco_video(paths)
    _export_lerobot(paths, payloads)
    _export_rlds(paths, payloads)
