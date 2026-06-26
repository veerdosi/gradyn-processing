from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from common import encode_coco_rle, frame_paths, job_paths, resize_for_max_side, save_json
except ModuleNotFoundError:
    from workers.common import encode_coco_rle, frame_paths, job_paths, resize_for_max_side, save_json

SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
BROAD_PROMPTS = [
    "tool",
    "hand tool",
    "object",
    "small object",
    "equipment",
    "instrument",
    "implement",
    "device",
    "part",
    "component",
    "piece",
    "attachment",
    "accessory",
    "material",
    "workpiece",
    "block",
    "bar",
    "rod",
    "tube",
    "plate",
    "sheet",
    "fastener",
    "hardware",
    "fixture",
    "paper",
]


def choose_torch_device(requested: str):
    import torch
    if requested == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested but is not available to PyTorch.")
    return torch.device(requested)


def clamp_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.asarray([
        max(0.0, min(float(box[0]), float(width - 1))),
        max(0.0, min(float(box[1]), float(height - 1))),
        max(1.0, min(float(box[2]), float(width))),
        max(1.0, min(float(box[3]), float(height))),
    ], dtype=np.float32)


def box_iou(left: np.ndarray, right: np.ndarray) -> float:
    x0 = max(float(left[0]), float(right[0])); y0 = max(float(left[1]), float(right[1]))
    x1 = min(float(left[2]), float(right[2])); y1 = min(float(left[3]), float(right[3]))
    inter = max(x1 - x0, 0.0) * max(y1 - y0, 0.0)
    la = max(float(left[2]-left[0]), 0.0) * max(float(left[3]-left[1]), 0.0)
    ra = max(float(right[2]-right[0]), 0.0) * max(float(right[3]-right[1]), 0.0)
    return inter / max(la + ra - inter, 1e-9)


def xyxy_from_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return np.asarray([0, 0, 0, 0], dtype=np.float32)
    return np.asarray([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def mask_passes_geometry(mask: np.ndarray, box: np.ndarray) -> bool:
    height, width = mask.shape
    area_fraction = float(mask.sum() / max(width * height, 1))
    if area_fraction <= 0.0004 or area_fraction > 0.40:
        return False
    x0, y0, x1, y1 = [float(v) for v in box]
    bw = max(x1 - x0, 1.0); bh = max(y1 - y0, 1.0)
    aspect = max(bw / bh, bh / bw)
    if aspect > 6.0:
        return False
    touches = int(x0 <= 1) + int(y0 <= 1) + int(x1 >= width - 1) + int(y1 >= height - 1)
    return touches < 3


def crop_square(image: Image.Image, box: np.ndarray, pad: float = 0.12) -> Image.Image:
    width, height = image.size
    x0, y0, x1, y1 = [float(v) for v in box]
    bw = x1 - x0; bh = y1 - y0
    x0 -= bw * pad; x1 += bw * pad; y0 -= bh * pad; y1 += bh * pad
    return image.crop((max(0, x0), max(0, y0), min(width, x1), min(height, y1)))


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    norm = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
    return norm @ norm.T


def cluster_embeddings(candidates: list[dict], similarity_threshold: float) -> list[list[int]]:
    if not candidates:
        return []
    vectors = np.asarray([item["embedding"] for item in candidates], dtype=np.float32)
    sim = cosine_matrix(vectors)
    parent = list(range(len(candidates)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    best_edges: list[tuple[float, int, int]] = []
    for i in range(len(candidates)):
        best_score = similarity_threshold
        best_index: int | None = None
        for j in range(len(candidates)):
            if i == j or candidates[i]["frame_index"] == candidates[j]["frame_index"]:
                continue
            if sim[i, j] > best_score:
                best_score = float(sim[i, j])
                best_index = j
        if best_index is not None and i < best_index:
            best_edges.append((best_score, i, best_index))
    used: set[int] = set()
    for _, i, j in sorted(best_edges, reverse=True):
        if i in used or j in used:
            continue
        union(i, j)
        used.update({i, j})
    groups: dict[int, list[int]] = {}
    for i in range(len(candidates)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def cluster_score(items: list[dict], keyframe_count: int) -> float:
    frames = {int(item["frame_index"]) for item in items}
    persistence = len(frames) / max(keyframe_count, 1)
    areas = np.asarray([float(item["area_fraction"]) for item in items], dtype=np.float32)
    aspects = np.asarray([float(item["aspect_ratio"]) for item in items], dtype=np.float32)
    size_stability = float(1.0 / (1.0 + np.std(areas) / max(float(np.mean(areas)), 1e-6)))
    shape_stability = float(1.0 / (1.0 + np.std(aspects) / max(float(np.mean(aspects)), 1e-6)))
    detector = float(np.median([float(item["detector_score"]) for item in items]))
    return 0.45 * persistence + 0.25 * size_stability + 0.20 * shape_stability + 0.10 * detector


def representative_index(items: list[dict]) -> int:
    vectors = np.asarray([item["embedding"] for item in items], dtype=np.float32)
    centroid = vectors.mean(axis=0)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-9)
    vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
    return int(np.argmax(vectors @ centroid))


def select_diverse_clusters(
    clusters: list[tuple[float, list[dict], dict]],
    max_objects: int,
    *,
    overlap_threshold: float = 0.70,
) -> list[tuple[float, list[dict], dict]]:
    selected: list[tuple[float, list[dict], dict]] = []
    for score, items, representative in clusters:
        box = np.asarray(representative["box_xyxy"], dtype=np.float32)
        if any(
            box_iou(box, np.asarray(other_rep["box_xyxy"], dtype=np.float32))
            > overlap_threshold
            for _, _, other_rep in selected
        ):
            continue
        selected.append((score, items, representative))
        if len(selected) >= max_objects:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--target-labels-json", required=True)
    parser.add_argument("--manual-seeds-json")
    parser.add_argument("--max-auto-objects", type=int, default=8)
    parser.add_argument("--exemplars-json", required=True)
    parser.add_argument("--stride", type=int, default=90)
    parser.add_argument("--max-side", type=int, default=960)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--box-threshold", type=float, default=0.15)
    parser.add_argument("--text-threshold", type=float, default=0.12)
    parser.add_argument("--cluster-similarity", type=float, default=0.72)
    args = parser.parse_args()

    import torch
    from transformers import AutoImageProcessor, AutoModel, AutoModelForZeroShotObjectDetection, AutoProcessor

    root = Path(__file__).resolve().parents[1]
    sam2_repo = root / "models" / "sam2"
    sam2_checkpoint = sam2_repo / "checkpoints" / "sam2.1_hiera_small.pt"
    grounding_dino_dir = root / "models" / "grounding-dino-base"
    dinov2_dir = root / "models" / "dinov2-small"
    for required, message in [
        (grounding_dino_dir / "config.json", "GroundingDINO base weights are missing."),
        (dinov2_dir / "config.json", "DINOv2 small weights are missing."),
        (sam2_repo / "sam2" / "__init__.py", "SAM2 repository is missing."),
        (sam2_checkpoint, "SAM2.1 small weights are missing."),
    ]:
        if not required.exists():
            raise SystemExit(message + " Run `gradyn models setup`.")
    sys.path.insert(0, str(sam2_repo))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    paths = job_paths(args.job)
    frames = frame_paths(args.job)
    target_labels: list[str] = json.loads(args.target_labels_json)
    max_objects = min(args.max_auto_objects, len(target_labels) or args.max_auto_objects)
    keyframes = [0, *range(args.stride, len(frames), args.stride)]
    keyframes = sorted(set(i for i in keyframes if 0 <= i < len(frames)))

    checkpoint_dir = paths["work"] / "object_cluster_progress"
    checkpoint_meta = checkpoint_dir / "metadata.json"
    signature = {
        "config_hash": json.loads((paths["root"] / "manifest.json").read_text()).get("config_hash"),
        "frame_count": len(frames),
        "keyframes": keyframes,
        "target_labels": target_labels,
        "max_side": args.max_side,
        "device": args.device,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "cluster_similarity": args.cluster_similarity,
        "selector": "broad_grounding_dino_sam2_dinov2_cluster_v2",
    }
    if checkpoint_meta.exists() and json.loads(checkpoint_meta.read_text()).get("signature") != signature:
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_json({"signature": signature}, checkpoint_meta)

    device = choose_torch_device(args.device)
    cpu_device = torch.device("cpu")
    print(
        f"Loading GroundingDINO base, SAM2.1 small, and DINOv2 small on {device.type}.",
        flush=True,
    )
    gd_processor = AutoProcessor.from_pretrained(str(grounding_dino_dir))
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(str(grounding_dino_dir)).to(device).eval()
    sam2_model = build_sam2(SAM2_CONFIG, str(sam2_checkpoint), device=device, mode="eval")
    sam2 = SAM2ImagePredictor(sam2_model)
    dino_processor = AutoImageProcessor.from_pretrained(str(dinov2_dir))
    dino_model = AutoModel.from_pretrained(str(dinov2_dir)).to(device).eval()

    print(f"Object discovery will inspect {len(keyframes)} keyframes with broad prompts.", flush=True)
    candidates: list[dict] = []
    for number, frame_index in enumerate(keyframes, start=1):
        cached_path = checkpoint_dir / f"{frame_index:08d}.json"
        if cached_path.exists():
            cached = json.loads(cached_path.read_text())
            candidates.extend(cached)
            print(f"↷ Object discovery reusing source frame {frame_index}", flush=True)
            continue
        print(f"Object discovery keyframe {number}/{len(keyframes)} (source frame {frame_index})", flush=True)
        original = Image.open(frames[frame_index]).convert("RGB")
        image, scale = resize_for_max_side(original, args.max_side)
        width, height = image.size
        sam2.set_image(np.asarray(image).copy())
        frame_boxes: list[tuple[np.ndarray, float, str]] = []
        for prompt in BROAD_PROMPTS:
            inputs = gd_processor(images=image, text=[[prompt]], return_tensors="pt").to(device)
            with torch.inference_mode():
                outputs = gd_model(**inputs)
            results = gd_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                target_sizes=[(height, width)],
            )[0]
            for box, score in zip(results.get("boxes", []), results.get("scores", []), strict=False):
                box_array = clamp_box(box.detach().to(cpu_device).numpy().astype(np.float32), width, height)
                if box_array[2] <= box_array[0] + 3 or box_array[3] <= box_array[1] + 3:
                    continue
                if any(box_iou(box_array, existing[0]) > 0.88 for existing in frame_boxes):
                    continue
                frame_boxes.append((box_array, float(score.detach().cpu()), prompt))
        frame_candidates: list[dict] = []
        for box, detector_score, prompt in frame_boxes:
            with torch.inference_mode():
                masks, mask_scores, _ = sam2.predict(box=box, multimask_output=True)
            for mask, sam2_score in zip(masks, mask_scores, strict=False):
                binary = np.asarray(mask, dtype=bool)
                if not binary.any():
                    continue
                mask_box = xyxy_from_mask(binary)
                if not mask_passes_geometry(binary, mask_box):
                    continue
                crop = crop_square(image, mask_box)
                dino_inputs = dino_processor(images=crop, return_tensors="pt").to(device)
                with torch.inference_mode():
                    features = dino_model(**dino_inputs).last_hidden_state[:, 0, :]
                embedding = features[0].detach().to(cpu_device).float().numpy()
                embedding /= max(float(np.linalg.norm(embedding)), 1e-9)
                output_mask = binary
                output_box = mask_box.copy()
                if scale != 1.0:
                    output_mask = np.asarray(
                        Image.fromarray(binary.astype(np.uint8) * 255).resize(original.size, Image.Resampling.NEAREST)
                    ) > 0
                    output_box = output_box / scale
                area_fraction = float(binary.sum() / max(width * height, 1))
                aspect_ratio = max(float(mask_box[2] - mask_box[0]), 1.0) / max(float(mask_box[3] - mask_box[1]), 1.0)
                aspect_ratio = max(aspect_ratio, 1.0 / aspect_ratio)
                frame_candidates.append({
                    "frame_index": frame_index,
                    "prompt": prompt,
                    "detector_score": detector_score,
                    "sam2_mask_score": float(sam2_score),
                    "box_xyxy": [float(v) for v in output_box.tolist()],
                    "mask_rle": encode_coco_rle(output_mask),
                    "embedding": embedding.tolist(),
                    "area_fraction": area_fraction,
                    "aspect_ratio": aspect_ratio,
                })
        save_json(frame_candidates, cached_path)
        candidates.extend(frame_candidates)
        if device.type == "mps":
            torch.mps.empty_cache()

    clusters = []
    for indices in cluster_embeddings(candidates, args.cluster_similarity):
        items = [candidates[i] for i in indices]
        frames_seen = {int(item["frame_index"]) for item in items}
        if len(frames_seen) < 2 and len(keyframes) >= 3:
            continue
        score = cluster_score(items, len(keyframes))
        rep = items[representative_index(items)]
        clusters.append((score, items, rep))
    clusters.sort(key=lambda value: value[0], reverse=True)
    selected = select_diverse_clusters(clusters, max_objects)

    discoveries: list[dict] = []
    selected_objects: list[dict] = []
    for object_id, (score, items, rep) in enumerate(selected, start=1):
        label = f"unknown object {object_id}"
        selected_objects.append({
            "object_id": object_id,
            "label": label,
            "pre_clip_label": label,
            "cluster_score": float(score),
            "keyframe_hits": len({int(item["frame_index"]) for item in items}),
            "representative_frame": int(rep["frame_index"]),
        })
        by_frame: dict[int, dict] = {}
        for item in items:
            frame_index = int(item["frame_index"])
            current = by_frame.get(frame_index)
            item_quality = float(item["detector_score"]) * float(item["sam2_mask_score"])
            current_quality = (
                float(current["detector_score"]) * float(current["sam2_mask_score"])
                if current is not None
                else -1.0
            )
            if item_quality > current_quality:
                by_frame[frame_index] = item
        for item in sorted(by_frame.values(), key=lambda value: int(value["frame_index"])):
            copied = {k: v for k, v in item.items() if k != "embedding"}
            copied.update({
                "object_id": object_id,
                "label": label,
                "found": True,
                "score": float(score),
                "anchor_confidence": float(score),
                "cluster_size": len(items),
                "representative_frame": int(rep["frame_index"]),
                "box_prompt_source": "broad_grounding_dino_sam2_dinov2_cluster",
            })
            discoveries.append(copied)

    save_json({
        "mode": "object_agnostic_anchor_clustering",
        "anchor_backend": "broad_grounding_dino_sam2_dinov2_cluster",
        "target_labels": target_labels,
        "selected_objects": selected_objects,
    }, paths["objects"] / "discovery.json")
    save_json(discoveries, paths["work"] / "anchor_discoveries.json")
    shutil.rmtree(checkpoint_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
