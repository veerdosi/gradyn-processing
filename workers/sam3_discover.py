from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from common import (
        encode_coco_rle,
        frame_paths,
        job_paths,
        load_prompt_bank,
        resize_for_max_side,
        SAM3_NATIVE_RESOLUTION,
        save_json,
    )
except ModuleNotFoundError:
    from workers.common import (
        encode_coco_rle,
        frame_paths,
        job_paths,
        load_prompt_bank,
        resize_for_max_side,
        SAM3_NATIVE_RESOLUTION,
        save_json,
    )

SAM3_MODEL_URL = (
    "https://huggingface.co/mlx-community/sam3-image/resolve/main/model.safetensors"
)
SAM3_MODEL_BYTES = 3_402_867_661


def ensure_sam3_weights(weights_dir: Path) -> Path:
    weights_dir.mkdir(parents=True, exist_ok=True)
    destination = weights_dir / "model.safetensors"
    temporary = weights_dir / "model.safetensors.download"
    if destination.exists() and destination.stat().st_size == SAM3_MODEL_BYTES:
        print(f"MLX SAM 3 weights already present: {destination}", flush=True)
        return destination
    destination.unlink(missing_ok=True)
    temporary.unlink(missing_ok=True)
    print(
        "Downloading MLX SAM 3 model via curl "
        f"({SAM3_MODEL_BYTES / 1_000_000_000:.2f} GB)…",
        flush=True,
    )
    command = [
        "curl",
        "-L",
        "--fail",
        "--show-error",
        "--progress-bar",
        "--retry",
        "20",
        "--retry-all-errors",
        "--retry-delay",
        "5",
        "--connect-timeout",
        "60",
        "--speed-limit",
        "1024",
        "--speed-time",
        "180",
        "-o",
        str(temporary),
        SAM3_MODEL_URL,
    ]
    try:
        subprocess.run(command, check=True)
    except KeyboardInterrupt:
        temporary.unlink(missing_ok=True)
        raise
    except subprocess.CalledProcessError as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "SAM 3 weight download failed. Check the network connection and rerun "
            "the same Gradyn command."
        ) from error
    actual_size = temporary.stat().st_size if temporary.exists() else 0
    if actual_size != SAM3_MODEL_BYTES:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"SAM 3 download had the wrong size: {actual_size:,} bytes; "
            f"expected {SAM3_MODEL_BYTES:,}."
        )
    os.replace(temporary, destination)
    print("MLX SAM 3 download complete and size verified.", flush=True)
    return destination


def mask_iou(left: dict, right: dict) -> float:
    if left["mask_rle"]["size"] != right["mask_rle"]["size"]:
        return 0.0
    from common import decode_coco_rle

    left_mask = decode_coco_rle(left["mask_rle"])
    right_mask = decode_coco_rle(right["mask_rle"])
    union = np.logical_or(left_mask, right_mask).sum()
    return (
        float(np.logical_and(left_mask, right_mask).sum() / union)
        if union
        else 0.0
    )


def select_auto_labels(
    discoveries: list[dict], keyframe_count: int, max_objects: int
) -> tuple[list[dict], list[dict]]:
    by_label: dict[str, list[dict]] = {}
    for item in discoveries:
        if item.get("found"):
            by_label.setdefault(str(item["label"]), []).append(item)

    candidates: list[dict] = []
    for label, items in by_label.items():
        scores = [float(item["score"]) for item in items]
        hit_rate = len({int(item["frame_index"]) for item in items}) / max(
            keyframe_count, 1
        )
        median_score = float(np.median(scores))
        # Semantic labels are part of the product contract. Do not promote a
        # recurring but weak text match into a stable object identity.
        if median_score < 0.50 and max(scores) < 0.65:
            continue
        # One excellent detection can initialize a short-lived manipulated object;
        # otherwise require recurrence across the sampled video.
        if hit_rate < 0.08 and max(scores) < 0.72:
            continue
        candidates.append(
            {
                "label": label,
                "hit_rate": hit_rate,
                "median_score": median_score,
                "max_score": max(scores),
                "rank": 0.65 * hit_rate + 0.35 * median_score,
                "items": items,
            }
        )
    candidates.sort(key=lambda item: item["rank"], reverse=True)

    # These labels were already approved by the user or a task prompt bank.
    # Do not discard a tool merely because it overlaps the workpiece it touches.
    # Lexical synonym cleanup belongs before SAM 3, not in mask-overlap space.
    selected = candidates[:max_objects]

    selected_labels = {item["label"] for item in selected}
    id_by_label = {
        item["label"]: index for index, item in enumerate(selected, start=1)
    }
    filtered = []
    for item in discoveries:
        if item["label"] not in selected_labels:
            continue
        copied = dict(item)
        copied["object_id"] = id_by_label[item["label"]]
        filtered.append(copied)
    summary = [
        {key: value for key, value in item.items() if key != "items"}
        | {"object_id": id_by_label[item["label"]]}
        for item in selected
    ]
    return filtered, summary


def select_with_exemplar(
    image: Image.Image, masks: np.ndarray, exemplar_path: str | None
) -> int:
    if not exemplar_path or len(masks) <= 1:
        return 0
    exemplar = np.asarray(
        Image.open(exemplar_path).convert("RGB").resize((128, 128)),
        dtype=np.float32,
    )
    exemplar_hist = np.concatenate(
        [
            np.histogram(exemplar[..., channel], bins=32, range=(0, 255), density=True)[0]
            for channel in range(3)
        ]
    )
    scores: list[float] = []
    for mask in masks:
        pixels = np.asarray(image, dtype=np.float32)[mask]
        if not len(pixels):
            scores.append(float("-inf"))
            continue
        candidate_hist = np.concatenate(
            [
                np.histogram(pixels[:, channel], bins=32, range=(0, 255), density=True)[0]
                for channel in range(3)
            ]
        )
        denominator = np.linalg.norm(exemplar_hist) * np.linalg.norm(candidate_hist)
        scores.append(
            float(np.dot(exemplar_hist, candidate_hist) / max(denominator, 1e-9))
        )
    return int(np.argmax(scores))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--objects-json", required=True)
    parser.add_argument("--candidate-prompts-json")
    parser.add_argument("--candidate-source", default="external_candidates")
    parser.add_argument("--manual-seeds-json")
    parser.add_argument("--prompt-bank")
    parser.add_argument("--max-auto-objects", type=int, default=8)
    parser.add_argument("--exemplars-json", required=True)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--max-side", type=int, default=960)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    model_repo = root / "models" / "mlx-sam3"
    if not model_repo.exists():
        raise SystemExit("MLX SAM 3 repository is missing. Run `gradyn models setup`.")
    sys.path.insert(0, str(model_repo))

    import mlx.core as mx
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    paths = job_paths(args.job)
    frames = frame_paths(args.job)
    objects: list[str] = json.loads(args.objects_json)
    automatic = not objects
    prompt_bank = None
    candidate_source = "explicit"
    if automatic:
        candidate_path = (
            Path(args.candidate_prompts_json)
            if args.candidate_prompts_json
            else None
        )
        if candidate_path is not None and candidate_path.exists():
            objects = json.loads(candidate_path.read_text())
            candidate_source = args.candidate_source
        elif args.prompt_bank:
            try:
                prompt_bank = load_prompt_bank(args.prompt_bank)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                raise SystemExit(str(error)) from error
            objects = prompt_bank["prompts"]
            candidate_source = "prompt_bank"
        else:
            raise SystemExit(
                "Automatic discovery requires approved candidates or a prompt bank."
            )
        if args.prompt_bank and prompt_bank is None:
            prompt_bank = load_prompt_bank(args.prompt_bank)
    exemplars: dict[str, str] = json.loads(args.exemplars_json)
    manual_seeds: list[dict] = []
    if args.manual_seeds_json:
        seed_path = Path(args.manual_seeds_json)
        if seed_path.exists():
            manual_seeds = json.loads(seed_path.read_text())
    seed_by_frame_label = {
        (int(seed["frame_index"]), str(seed["label"])): [
            float(value) for value in seed["box_xyxy"]
        ]
        for seed in manual_seeds
    }
    keyframes = sorted(
        set(
            [
                0,
                *range(args.stride, len(frames), args.stride),
                *[
                    int(seed["frame_index"])
                    for seed in manual_seeds
                    if 0 <= int(seed["frame_index"]) < len(frames)
                ],
            ]
        )
    )
    checkpoint_dir = paths["work"] / "sam3_progress"
    checkpoint_meta = checkpoint_dir / "metadata.json"
    signature = {
        "config_hash": json.loads(
            (paths["root"] / "manifest.json").read_text()
        ).get("config_hash"),
        "frame_count": len(frames),
        "keyframes": keyframes,
        "objects": objects,
        "max_side": args.max_side,
        "exemplars": exemplars,
        "manual_seeds": manual_seeds,
    }
    discoveries: list[dict] = []
    completed_keyframes: set[int] = set()
    if checkpoint_meta.exists():
        checkpoint = json.loads(checkpoint_meta.read_text())
        if checkpoint.get("signature") != signature:
            shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_json({"signature": signature}, checkpoint_meta)
    for frame_index in keyframes:
        frame_checkpoint = checkpoint_dir / f"{frame_index:08d}.json"
        if frame_checkpoint.exists():
            discoveries.extend(json.loads(frame_checkpoint.read_text()))
            completed_keyframes.add(frame_index)

    weights_dir = root / "models" / "mlx-sam3" / "weights" / "sam3-image"
    checkpoint = ensure_sam3_weights(weights_dir)
    print("MLX SAM 3 weights are ready. Initializing model…", flush=True)
    model = build_sam3_image_model(checkpoint_path=str(checkpoint))
    print("MLX SAM 3 model initialized.", flush=True)
    # The MLX port builds RoPE and position encodings for a fixed 1008x1008
    # vision input. `max_side` only controls the aspect-preserving image used for
    # source-coordinate masks; it must not change the model tensor resolution.
    processor = Sam3Processor(
        model,
        resolution=SAM3_NATIVE_RESOLUTION,
        confidence_threshold=0.35,
    )
    print(
        f"SAM 3 will inspect {len(keyframes)} keyframes with {len(objects)} prompts.",
        flush=True,
    )
    for keyframe_number, frame_index in enumerate(keyframes, start=1):
        if frame_index in completed_keyframes:
            print(f"↷ SAM 3 reusing source frame {frame_index}", flush=True)
            continue
        print(
            f"SAM 3 keyframe {keyframe_number}/{len(keyframes)} "
            f"(source frame {frame_index})",
            flush=True,
        )
        original = Image.open(frames[frame_index]).convert("RGB")
        image, scale = resize_for_max_side(original, args.max_side)
        state = processor.set_image(image)
        frame_discoveries: list[dict] = []
        for object_id, label in enumerate(objects, start=1):
            processor.reset_all_prompts(state)
            state = processor.set_text_prompt(label, state)
            manual_box = seed_by_frame_label.get((frame_index, label))
            if manual_box is not None:
                x0, y0, x1, y1 = manual_box
                state = processor.add_geometric_prompt(
                    box=[
                        ((x0 + x1) / 2) / original.width,
                        ((y0 + y1) / 2) / original.height,
                        (x1 - x0) / original.width,
                        (y1 - y0) / original.height,
                    ],
                    label=True,
                    state=state,
                )
            masks = np.asarray(state["masks"], dtype=bool)
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0]
            if masks.ndim != 3:
                raise RuntimeError(
                    f"Unexpected SAM 3 mask shape {masks.shape}; "
                    "expected [detections, height, width]."
                )
            boxes = np.asarray(state["boxes"], dtype=np.float32)
            scores = np.asarray(state["scores"], dtype=np.float32)
            if len(scores) == 0:
                frame_discoveries.append(
                    {
                        "frame_index": frame_index,
                        "object_id": object_id,
                        "label": label,
                        "found": False,
                    }
                )
                continue
            order = np.argsort(scores)[::-1]
            masks, boxes, scores = masks[order], boxes[order], scores[order]
            chosen = select_with_exemplar(image, masks, exemplars.get(label))
            mask = masks[chosen]
            if scale != 1.0:
                mask = np.asarray(
                    Image.fromarray(mask.astype(np.uint8) * 255).resize(
                        original.size, Image.Resampling.NEAREST
                    )
                ) > 0
                boxes[chosen] /= scale
            frame_discoveries.append(
                {
                    "frame_index": frame_index,
                    "object_id": object_id,
                    "label": label,
                    "found": True,
                    "score": float(scores[chosen]),
                    "box_xyxy": [float(x) for x in boxes[chosen].tolist()],
                    "mask_rle": encode_coco_rle(mask),
                }
            )
        del state
        mx.clear_cache()
        discoveries.extend(frame_discoveries)
        completed_keyframes.add(frame_index)
        save_json(
            frame_discoveries,
            checkpoint_dir / f"{frame_index:08d}.json",
        )
    if automatic:
        discoveries, selected = select_auto_labels(
            discoveries, len(keyframes), args.max_auto_objects
        )
        if not selected:
            raise SystemExit(
                "Automatic object discovery found no persistent candidates. "
                "Retry with explicit --objects names."
            )
        save_json(
            {
                "mode": "automatic",
                "candidate_source": candidate_source,
                "prompt_bank": (
                    {
                        key: value
                        for key, value in prompt_bank.items()
                        if key != "prompts"
                    }
                    if prompt_bank
                    else None
                ),
                "prompts": objects,
                "selected_objects": selected,
            },
            paths["objects"] / "discovery.json",
        )
        print(
            "Selected automatic objects: "
            + ", ".join(item["label"] for item in selected),
            flush=True,
        )
    else:
        save_json(
            {
                "mode": "explicit",
                "selected_objects": [
                    {"object_id": index, "label": label}
                    for index, label in enumerate(objects, start=1)
                ],
            },
            paths["objects"] / "discovery.json",
        )
    save_json(discoveries, paths["work"] / "sam3_discoveries.json")
    shutil.rmtree(checkpoint_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
