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


def _histogram_similarity(
    image: Image.Image, masks: np.ndarray, exemplar_path: str | None
) -> np.ndarray:
    if not exemplar_path:
        return np.zeros(len(masks), dtype=np.float32)
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
    similarities: list[float] = []
    for mask in masks:
        pixels = np.asarray(image, dtype=np.float32)[mask]
        if not len(pixels):
            similarities.append(0.0)
            continue
        candidate_hist = np.concatenate(
            [
                np.histogram(pixels[:, channel], bins=32, range=(0, 255), density=True)[0]
                for channel in range(3)
            ]
        )
        denominator = np.linalg.norm(exemplar_hist) * np.linalg.norm(candidate_hist)
        similarities.append(
            float(np.dot(exemplar_hist, candidate_hist) / max(denominator, 1e-9))
        )
    return np.asarray(similarities, dtype=np.float32)


def _box_iou(left: np.ndarray, right: np.ndarray) -> float:
    x0 = max(float(left[0]), float(right[0]))
    y0 = max(float(left[1]), float(right[1]))
    x1 = min(float(left[2]), float(right[2]))
    y1 = min(float(left[3]), float(right[3]))
    intersection = max(x1 - x0, 0.0) * max(y1 - y0, 0.0)
    left_area = max(float(left[2] - left[0]), 0.0) * max(
        float(left[3] - left[1]), 0.0
    )
    right_area = max(float(right[2] - right[0]), 0.0) * max(
        float(right[3] - right[1]), 0.0
    )
    return intersection / max(left_area + right_area - intersection, 1e-9)


def _box_area(box: np.ndarray) -> float:
    return max(float(box[2] - box[0]), 0.0) * max(float(box[3] - box[1]), 0.0)


def _box_intersection(left: np.ndarray, right: np.ndarray) -> float:
    x0 = max(float(left[0]), float(right[0]))
    y0 = max(float(left[1]), float(right[1]))
    x1 = min(float(left[2]), float(right[2]))
    y1 = min(float(left[3]), float(right[3]))
    return max(x1 - x0, 0.0) * max(y1 - y0, 0.0)


def box_prompt_alignment(
    candidate_box: np.ndarray,
    prompt_box: np.ndarray | None,
    image_size: tuple[int, int],
) -> dict[str, float]:
    if prompt_box is None:
        return {
            "box_prompt_iou": 0.0,
            "box_prompt_center_similarity": 0.0,
            "box_prompt_candidate_inside": 0.0,
            "box_prompt_coverage": 0.0,
            "box_prompt_area_ratio": 0.0,
            "box_prompt_agreement": 0.0,
        }
    width, height = image_size
    diagonal = max(float(np.hypot(width, height)), 1.0)
    candidate = np.asarray(candidate_box, dtype=np.float32)
    prompt = np.asarray(prompt_box, dtype=np.float32)
    candidate_area = _box_area(candidate)
    prompt_area = _box_area(prompt)
    intersection = _box_intersection(candidate, prompt)
    candidate_center = np.asarray(
        [
            (candidate[0] + candidate[2]) / 2.0,
            (candidate[1] + candidate[3]) / 2.0,
        ],
        dtype=np.float32,
    )
    prompt_center = np.asarray(
        [(prompt[0] + prompt[2]) / 2.0, (prompt[1] + prompt[3]) / 2.0],
        dtype=np.float32,
    )
    center_similarity = float(
        np.exp(
            -3.0
            * np.linalg.norm(candidate_center - prompt_center)
            / diagonal
        )
    )
    candidate_inside = float(intersection / max(candidate_area, 1e-9))
    prompt_coverage = float(intersection / max(prompt_area, 1e-9))
    iou = _box_iou(candidate, prompt)
    area_ratio = float(candidate_area / max(prompt_area, 1e-9))
    # A loose proposal is acceptable if SAM returns a tighter object inside it.
    # The reverse is risky: a prompted mask that balloons beyond the proposal is
    # often a background/work-surface leak.
    agreement = float(
        0.35 * iou
        + 0.35 * center_similarity
        + 0.30 * candidate_inside
    )
    return {
        "box_prompt_iou": float(iou),
        "box_prompt_center_similarity": center_similarity,
        "box_prompt_candidate_inside": candidate_inside,
        "box_prompt_coverage": prompt_coverage,
        "box_prompt_area_ratio": area_ratio,
        "box_prompt_agreement": agreement,
    }


def prompt_anchor_rejection_reason(metrics: dict[str, float]) -> str | None:
    if metrics["box_prompt_area_ratio"] >= 3.5 and metrics[
        "box_prompt_candidate_inside"
    ] < 0.55:
        return "box_prompt_mask_expanded_outside_proposal"
    if (
        metrics["box_prompt_center_similarity"] < 0.45
        and metrics["box_prompt_iou"] < 0.05
    ):
        return "box_prompt_mask_landed_elsewhere"
    if (
        metrics["box_prompt_iou"] < 0.03
        and metrics["box_prompt_candidate_inside"] < 0.15
        and metrics["box_prompt_coverage"] < 0.15
    ):
        return "box_prompt_mask_disagrees_with_proposal"
    return None


def select_object_candidate(
    image: Image.Image,
    masks: np.ndarray,
    boxes: np.ndarray,
    semantic_scores: np.ndarray,
    exemplar_path: str | None = None,
    previous_box: np.ndarray | None = None,
    prompt_box: np.ndarray | None = None,
    prompt_trust: float = 1.0,
) -> tuple[int, list[dict[str, float]]]:
    """Choose the foreground task instance, not merely the best text match.

    Egocentric work videos often contain several members of the same class:
    the sheet being handled, a stack at the image edge, and completed sheets in
    the machine. SAM 3's semantic score alone cannot distinguish those roles.
    """
    width, height = image.size
    diagonal = max(float(np.hypot(width, height)), 1.0)
    image_area = max(float(width * height), 1.0)
    exemplar_scores = _histogram_similarity(image, masks, exemplar_path)
    diagnostics: list[dict[str, float]] = []

    for index, (mask, box, semantic_score) in enumerate(
        zip(masks, boxes, semantic_scores, strict=True)
    ):
        x0, y0, x1, y1 = [float(value) for value in box]
        center_x = (x0 + x1) / 2.0
        center_y = (y0 + y1) / 2.0
        # The useful workspace in a head-mounted view is usually around the
        # middle and slightly below center. This is a prior, not a hard crop.
        focus_distance = np.hypot(
            center_x - 0.50 * width, center_y - 0.56 * height
        )
        centrality = float(np.exp(-3.0 * focus_distance / diagonal))
        mask_fraction = float(mask.sum() / image_area)
        area_support = float(np.clip(np.sqrt(mask_fraction / 0.10), 0.0, 1.0))

        continuity = 0.5
        if previous_box is not None:
            previous_pixels = np.asarray(previous_box, dtype=np.float32).copy()
            if float(np.max(previous_pixels)) <= 1.5:
                previous_pixels *= np.asarray(
                    [width, height, width, height], dtype=np.float32
                )
            previous_center = np.asarray(
                [
                    (previous_pixels[0] + previous_pixels[2]) / 2.0,
                    (previous_pixels[1] + previous_pixels[3]) / 2.0,
                ],
                dtype=np.float32,
            )
            center_similarity = float(
                np.exp(
                    -3.0
                    * np.linalg.norm(
                        np.asarray([center_x, center_y], dtype=np.float32)
                        - previous_center
                    )
                    / diagonal
                )
            )
            continuity = 0.55 * center_similarity + 0.45 * _box_iou(
                box, previous_pixels
            )

        edge_penalty = 0.0
        touches_top = y0 <= 0.035 * height
        touches_side = x0 <= 0.02 * width or x1 >= 0.98 * width
        if touches_top:
            edge_penalty += 0.24
        if touches_side:
            edge_penalty += 0.08

        prompt_metrics = box_prompt_alignment(
            box,
            prompt_box,
            (width, height),
        )
        prompt_weight = (
            0.15 * max(0.0, min(1.0, float(prompt_trust)))
            if prompt_box is not None
            else 0.0
        )
        if exemplar_path:
            prompt_fraction = prompt_weight / 0.15 if prompt_weight else 0.0
            semantic_weight = 0.30 - 0.05 * prompt_fraction
            centrality_weight = 0.15 - 0.03 * prompt_fraction
            exemplar_weight = 0.30 - 0.05 * prompt_fraction
            total = (
                semantic_weight * float(semantic_score)
                + centrality_weight * centrality
                + 0.10 * area_support
                + 0.13 * continuity
                + exemplar_weight * float(exemplar_scores[index])
                + prompt_weight * prompt_metrics["box_prompt_agreement"]
                - edge_penalty
            )
        else:
            prompt_fraction = prompt_weight / 0.15 if prompt_weight else 0.0
            semantic_weight = 0.45 - 0.09 * prompt_fraction
            centrality_weight = 0.25 - 0.05 * prompt_fraction
            area_weight = 0.15 - 0.03 * prompt_fraction
            total = (
                semantic_weight * float(semantic_score)
                + centrality_weight * centrality
                + area_weight * area_support
                + 0.12 * continuity
                + prompt_weight * prompt_metrics["box_prompt_agreement"]
                - edge_penalty
            )
        diagnostics.append(
            {
                "total": float(total),
                "semantic": float(semantic_score),
                "centrality": centrality,
                "area_support": area_support,
                "continuity": float(continuity),
                "edge_penalty": float(edge_penalty),
                "exemplar": float(exemplar_scores[index]),
                **prompt_metrics,
            }
        )

    return int(np.argmax([item["total"] for item in diagnostics])), diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--objects-json", required=True)
    parser.add_argument("--candidate-prompts-json")
    parser.add_argument("--candidate-source", default="external_candidates")
    parser.add_argument("--manual-seeds-json")
    parser.add_argument("--box-proposals-json")
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
    box_proposals: dict[tuple[int, str], dict] = {}
    if args.box_proposals_json:
        proposal_path = Path(args.box_proposals_json)
        if proposal_path.exists():
            proposal_data = json.loads(proposal_path.read_text())
            for proposal in proposal_data.get("boxes", []):
                key = (int(proposal["frame_index"]), str(proposal["label"]))
                if proposal.get("found") and proposal.get("box_xyxy"):
                    box_proposals[key] = proposal
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
        "box_proposals": (
            str(Path(args.box_proposals_json).resolve())
            if args.box_proposals_json
            else None
        ),
        "instance_selector": "text_first_optional_box_v3",
    }
    discoveries: list[dict] = []
    completed_keyframes: set[int] = set()
    checkpointed_by_frame: dict[int, list[dict]] = {}
    if checkpoint_meta.exists():
        checkpoint = json.loads(checkpoint_meta.read_text())
        if checkpoint.get("signature") != signature:
            shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_json({"signature": signature}, checkpoint_meta)
    for frame_index in keyframes:
        frame_checkpoint = checkpoint_dir / f"{frame_index:08d}.json"
        if frame_checkpoint.exists():
            cached = json.loads(frame_checkpoint.read_text())
            discoveries.extend(cached)
            checkpointed_by_frame[frame_index] = cached
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
    previous_box_by_label: dict[str, np.ndarray] = {}
    consecutive_misses: dict[str, int] = {}
    source_width, source_height = Image.open(frames[0]).size
    source_box_scale = np.asarray(
        [source_width, source_height, source_width, source_height],
        dtype=np.float32,
    )
    for keyframe_number, frame_index in enumerate(keyframes, start=1):
        if frame_index in completed_keyframes:
            for item in checkpointed_by_frame[frame_index]:
                label = str(item["label"])
                if item.get("found"):
                    previous_box_by_label[label] = np.asarray(
                        item["box_xyxy"], dtype=np.float32
                    ) / source_box_scale
                    consecutive_misses[label] = 0
                else:
                    consecutive_misses[label] = consecutive_misses.get(label, 0) + 1
                    if consecutive_misses[label] >= 2:
                        previous_box_by_label.pop(label, None)
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
            manual_box = seed_by_frame_label.get((frame_index, label))
            proposal_key = (frame_index, label)
            proposal = box_proposals.get(proposal_key)
            proposal_box = (
                [float(value) for value in proposal["box_xyxy"]]
                if proposal is not None
                else None
            )
            prompt_box = manual_box if manual_box is not None else proposal_box
            prompt_box_for_image = (
                np.asarray(prompt_box, dtype=np.float32) * float(scale)
                if prompt_box is not None
                else None
            )
            prompt_trust = (
                1.0
                if manual_box is not None
                else float(proposal.get("confidence", 0.7))
                if proposal is not None
                else 1.0
            )

            def evaluate_anchor(use_prompt_box: bool) -> dict | None:
                processor.reset_all_prompts(state)
                processor.set_text_prompt(label, state)
                if use_prompt_box and prompt_box is not None:
                    x0, y0, x1, y1 = prompt_box
                    processor.add_geometric_prompt(
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
                    return None
                order = np.argsort(scores)[::-1]
                masks, boxes, scores = masks[order], boxes[order], scores[order]
                chosen, candidate_scores = select_object_candidate(
                    image,
                    masks,
                    boxes,
                    scores,
                    exemplars.get(label),
                    previous_box_by_label.get(label),
                    prompt_box_for_image if use_prompt_box else None,
                    prompt_trust,
                )
                rejection_reason = (
                    prompt_anchor_rejection_reason(candidate_scores[chosen])
                    if use_prompt_box and prompt_box is not None
                    else None
                )
                if rejection_reason is not None:
                    return None
                return {
                    "mask": masks[chosen],
                    "box": boxes[chosen].copy(),
                    "score": float(scores[chosen]),
                    "candidate_count": int(len(scores)),
                    "selection_score": candidate_scores[chosen],
                    "source": (
                        "manual_seed"
                        if use_prompt_box and manual_box is not None
                        else (
                            str(proposal.get("source", "box_proposal"))
                            if use_prompt_box and proposal_box is not None
                            else "none"
                        )
                    ),
                }

            text_anchor = evaluate_anchor(False)
            prompted_anchor = (
                evaluate_anchor(True) if prompt_box is not None else None
            )
            anchor = text_anchor
            if prompted_anchor is not None:
                if manual_box is not None:
                    anchor = prompted_anchor
                elif text_anchor is None:
                    anchor = prompted_anchor
                elif (
                    prompt_box_for_image is not None
                    and box_prompt_alignment(
                        text_anchor["box"],
                        prompt_box_for_image,
                        image.size,
                    )["box_prompt_agreement"]
                    < 0.35
                ):
                    anchor = prompted_anchor
                elif (
                    text_anchor["score"] < 0.70
                    and prompted_anchor["score"] >= text_anchor["score"] - 0.05
                ):
                    anchor = prompted_anchor
                elif (
                    prompted_anchor["selection_score"]["total"]
                    > text_anchor["selection_score"]["total"] + 0.12
                ):
                    anchor = prompted_anchor

            if anchor is None:
                consecutive_misses[label] = consecutive_misses.get(label, 0) + 1
                if consecutive_misses[label] >= 2:
                    previous_box_by_label.pop(label, None)
                frame_discoveries.append(
                    {
                        "frame_index": frame_index,
                        "object_id": object_id,
                        "label": label,
                        "found": False,
                    }
                )
                continue
            mask = anchor["mask"]
            chosen_box = anchor["box"]
            previous_box_by_label[label] = chosen_box / np.asarray(
                [image.width, image.height, image.width, image.height],
                dtype=np.float32,
            )
            consecutive_misses[label] = 0
            if scale != 1.0:
                mask = np.asarray(
                    Image.fromarray(mask.astype(np.uint8) * 255).resize(
                        original.size, Image.Resampling.NEAREST
                    )
                ) > 0
                chosen_box /= scale
            frame_discoveries.append(
                {
                    "frame_index": frame_index,
                    "object_id": object_id,
                    "label": label,
                    "found": True,
                    "score": anchor["score"],
                    "box_xyxy": [float(x) for x in chosen_box.tolist()],
                    "mask_rle": encode_coco_rle(mask),
                    "candidate_count": anchor["candidate_count"],
                    "selection_score": anchor["selection_score"],
                    "box_prompt_source": anchor["source"],
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
