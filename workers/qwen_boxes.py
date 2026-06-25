from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from PIL import Image

try:
    from common import frame_paths, job_paths, load_prompt_bank, save_json
except ModuleNotFoundError:
    from workers.common import frame_paths, job_paths, load_prompt_bank, save_json


MODEL_PATH = "models/qwen3-vl-4b"
QWEN_COORDINATE_SIZE = 1000
BOX_PROMPT_TEMPLATE = """You are grounding named physical objects in one egocentric
robotics video frame.

Image size: {width} x {height} pixels.
Requested objects: {objects}

Your job is precise visual grounding, not scene description. For each requested object,
find at most one tight bounding box around the visible physical object instance most
relevant to the hand task. If you cannot draw a tight, object-specific box, mark it
visible=false and box_xyxy=null.

Important rules:
- Return boxes in a 0-1000 normalized coordinate system, where [0, 0] is the
  top-left of the image and [1000, 1000] is the bottom-right of the image.
- Use [x1, y1, x2, y2] normalized coordinates in that 0-1000 system.
- The box must be tight: it should include the object and only a small amount of
  surrounding background. Do not draw broad half-frame or full-workbench boxes.
- Box the object itself, not a hand, arm, foot, clothing, shadow, glare, reflection,
  text label, floor, table, machine surface, or surrounding background.
- Never reuse the same or almost-same box for two different requested labels. If two
  requested labels appear to refer to the same visible region, keep only the more specific
  one and mark the other visible=false.
- Distinguish object scale:
  - "metal sheet", "paper sheet", "panel", or "workpiece" means the bounded flat item,
    not the whole table/floor and not a thin edge unless the whole sheet is only visible
    as that edge.
  - "narrow metal strip", "metal rod", or similar long thin labels mean only the narrow
    elongated part, not the larger sheet/panel it touches.
  - "hammer", "metal punch", "screwdriver", "torch", or other tools mean the complete
    visible tool, not the nearby hand, not another tool, and not the surface beneath it.
- For repeated or similar objects, choose the instance being handled or most central to
  the current task. If this cannot be determined visually, mark visible=false.
- Do not invent hidden or off-frame object regions.
- Do not output duplicate synonyms or additional object names.
- Prefer no box over a loose or ambiguous box.

Return strict JSON only:
{{
  "objects": [
    {{
      "label": "requested label exactly",
      "visible": true,
      "box_xyxy": [x1_0_to_1000, y1_0_to_1000, x2_0_to_1000, y2_0_to_1000],
      "confidence": 0.0
    }}
  ]
}}

Confidence meaning:
- 0.90-1.00: tight box around the correct object
- 0.70-0.89: likely correct but partially occluded or slightly blurred
- below 0.70: uncertain; prefer visible=false instead of a weak box
"""


def extract_json_payload(text: str) -> dict:
    candidates = [text.strip()]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    left, right = text.find("{"), text.rfind("}")
    if left >= 0 and right > left:
        candidates.insert(0, text[left : right + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def normalize_label(value: str, requested: list[str]) -> str | None:
    folded = str(value).strip().casefold()
    for label in requested:
        if folded == label.casefold():
            return label
    return None


def box_area_fraction(box: list[float], width: int, height: int) -> float:
    return ((box[2] - box[0]) * (box[3] - box[1])) / max(float(width * height), 1.0)


def box_iou(left: list[float], right: list[float]) -> float:
    lx0, ly0, lx1, ly1 = left
    rx0, ry0, rx1, ry1 = right
    ix0, iy0 = max(lx0, rx0), max(ly0, ry0)
    ix1, iy1 = min(lx1, rx1), min(ly1, ry1)
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = (
        max(0.0, lx1 - lx0) * max(0.0, ly1 - ly0)
        + max(0.0, rx1 - rx0) * max(0.0, ry1 - ry0)
        - intersection
    )
    return float(intersection / union) if union > 0 else 0.0


def clamp_box(
    box: list[float], width: int, height: int, *, label: str
) -> list[float] | None:
    if len(box) != 4:
        return None
    x0, y0, x1, y1 = [float(value) for value in box]
    x0 = max(0.0, min(float(width - 1), x0))
    y0 = max(0.0, min(float(height - 1), y0))
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    if x1 <= x0 + 4 or y1 <= y0 + 4:
        return None
    return [x0, y0, x1, y1]


def remove_cross_label_duplicate_boxes(
    parsed: dict[str, dict], *, width: int, height: int
) -> None:
    labels = [label for label, item in parsed.items() if item.get("found")]
    ambiguous: set[str] = set()
    for index, left_label in enumerate(labels):
        left_box = parsed[left_label]["box_xyxy"]
        for right_label in labels[index + 1 :]:
            right_box = parsed[right_label]["box_xyxy"]
            iou = box_iou(left_box, right_box)
            left_area = box_area_fraction(left_box, width, height)
            right_area = box_area_fraction(right_box, width, height)
            # If Qwen assigns substantially the same visual region to two
            # different requested labels, the grounding is not discriminative
            # enough to be a safe SAM 3 box prompt.
            if iou >= 0.62 or (
                iou >= 0.45 and min(left_area, right_area) / max(left_area, right_area) > 0.55
            ):
                ambiguous.update([left_label, right_label])
    for label in ambiguous:
        parsed[label] = {
            "label": label,
            "found": False,
            "box_xyxy": None,
            "confidence": 0.0,
            "rejection_reason": "cross_label_duplicate_box",
        }


def parse_box_response(
    text: str,
    *,
    requested: list[str],
    width: int,
    height: int,
) -> dict[str, dict]:
    payload = extract_json_payload(text)
    values = payload.get("objects", []) if isinstance(payload, dict) else []
    if not isinstance(values, list):
        values = []
    parsed: dict[str, dict] = {
        label: {
            "label": label,
            "found": False,
            "box_xyxy": None,
            "confidence": 0.0,
        }
        for label in requested
    }
    for item in values:
        if not isinstance(item, dict):
            continue
        label = normalize_label(str(item.get("label", "")), requested)
        if label is None:
            continue
        visible = bool(item.get("visible", False))
        raw_box = item.get("box_xyxy")
        if not visible or raw_box is None:
            continue
        if not isinstance(raw_box, list):
            continue
        try:
            box = clamp_box(
                [float(value) for value in raw_box],
                width,
                height,
                label=label,
            )
        except (TypeError, ValueError):
            box = None
        if box is None:
            continue
        try:
            confidence = float(item.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))
        # If Qwen repeats a label, keep the higher-confidence valid grounding.
        if confidence >= float(parsed[label]["confidence"]):
            parsed[label] = {
                "label": label,
                "found": True,
                "box_xyxy": box,
                "confidence": confidence,
            }
    remove_cross_label_duplicate_boxes(parsed, width=width, height=height)
    return parsed


def scaled_box(box: list[float], x_scale: float, y_scale: float) -> list[float]:
    return [
        float(box[0] * x_scale),
        float(box[1] * y_scale),
        float(box[2] * x_scale),
        float(box[3] * y_scale),
    ]


def load_requested_objects(args: argparse.Namespace) -> list[str]:
    objects: list[str] = json.loads(args.objects_json)
    if not objects and args.candidate_prompts_json:
        candidate_path = Path(args.candidate_prompts_json)
        if candidate_path.exists():
            objects = json.loads(candidate_path.read_text())
    if not objects and args.prompt_bank:
        objects = load_prompt_bank(args.prompt_bank)["prompts"]
    objects = [str(label).strip() for label in objects if str(label).strip()]
    deduped: list[str] = []
    for label in objects:
        if label.casefold() not in {existing.casefold() for existing in deduped}:
            deduped.append(label)
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--objects-json", required=True)
    parser.add_argument("--candidate-prompts-json")
    parser.add_argument("--prompt-bank")
    parser.add_argument("--stride", type=int, default=90)
    parser.add_argument("--max-side", type=int, default=672)
    parser.add_argument("--max-tokens", type=int, default=384)
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    root = Path(__file__).resolve().parents[1]
    model_path = root / MODEL_PATH
    if not (model_path / "config.json").exists():
        raise SystemExit("Qwen3-VL weights are missing. Run `gradyn models setup`.")

    paths = job_paths(args.job)
    frames = frame_paths(args.job)
    objects = load_requested_objects(args)
    if not objects:
        raise SystemExit("Qwen box proposals require at least one object label.")

    keyframes = [0, *range(args.stride, len(frames), args.stride)]
    keyframes = sorted(set(index for index in keyframes if 0 <= index < len(frames)))
    output_path = paths["work"] / "qwen_box_proposals.json"
    progress_dir = paths["work"] / "qwen_box_progress"
    metadata_path = progress_dir / "metadata.json"
    signature = {
        "frame_count": len(frames),
        "keyframes": keyframes,
        "objects": objects,
        "model": MODEL_PATH,
        "max_side": args.max_side,
        "max_tokens": args.max_tokens,
        "prompt_template": BOX_PROMPT_TEMPLATE,
    }
    if metadata_path.exists():
        try:
            if json.loads(metadata_path.read_text()).get("signature") != signature:
                shutil.rmtree(progress_dir)
        except json.JSONDecodeError:
            shutil.rmtree(progress_dir)
    progress_dir.mkdir(parents=True, exist_ok=True)
    save_json({"signature": signature}, metadata_path)

    print(
        f"Loading Qwen3-VL 4B 4-bit for boxes; {len(keyframes)} keyframes, "
        f"{len(objects)} object prompts.",
        flush=True,
    )
    mx.set_memory_limit(6 * 1024**3)
    mx.reset_peak_memory()
    model, processor = load(str(model_path))

    inference_image = paths["work"] / "qwen_box_current_frame.jpg"
    all_rows: list[dict] = []
    for keyframe_number, frame_index in enumerate(keyframes, start=1):
        checkpoint = progress_dir / f"{frame_index:08d}.json"
        if checkpoint.exists():
            rows = json.loads(checkpoint.read_text())
            all_rows.extend(rows)
            print(f"↷ Qwen boxes reusing source frame {frame_index}", flush=True)
            continue

        print(
            f"Qwen boxes keyframe {keyframe_number}/{len(keyframes)} "
            f"(source frame {frame_index})",
            flush=True,
        )
        with Image.open(frames[frame_index]) as source:
            original = source.convert("RGB")
            original_width, original_height = original.size
            image = original.copy()
            image.thumbnail((args.max_side, args.max_side), Image.Resampling.LANCZOS)
            resized_width, resized_height = image.size
            image.save(inference_image, quality=92)
        prompt_text = BOX_PROMPT_TEMPLATE.format(
            width=resized_width,
            height=resized_height,
            objects=json.dumps(objects, ensure_ascii=False),
        )
        prompt = apply_chat_template(
            processor,
            model.config,
            prompt_text,
            num_images=1,
        )
        result = generate(
            model,
            processor,
            prompt,
            [str(inference_image)],
            max_tokens=args.max_tokens,
            temperature=0.0,
            verbose=False,
        )
        text = result.text if hasattr(result, "text") else str(result)
        parsed = parse_box_response(
            text,
            requested=objects,
            width=QWEN_COORDINATE_SIZE,
            height=QWEN_COORDINATE_SIZE,
        )
        x_scale = original_width / float(QWEN_COORDINATE_SIZE)
        y_scale = original_height / float(QWEN_COORDINATE_SIZE)
        rows: list[dict] = []
        for object_id, label in enumerate(objects, start=1):
            item = parsed[label]
            if not item["found"]:
                rows.append(
                    {
                        "frame_index": frame_index,
                        "object_id": object_id,
                        "label": label,
                        "found": False,
                        "source": "qwen3_vl_box",
                        "raw_response": text,
                    }
                )
                continue
            box = scaled_box(item["box_xyxy"], x_scale, y_scale)
            rows.append(
                {
                    "frame_index": frame_index,
                    "object_id": object_id,
                    "label": label,
                    "found": True,
                    "box_xyxy": box,
                    "confidence": float(item["confidence"]),
                    "source": "qwen3_vl_box",
                    "resized_image_size": [resized_width, resized_height],
                    "box_coordinate_system": "qwen_0_1000",
                    "raw_response": text,
                }
            )
        save_json(rows, checkpoint)
        all_rows.extend(rows)
        del result, image, original
        mx.clear_cache()
        found = [row["label"] for row in rows if row.get("found")]
        print(
            f"  found={found or 'none'}; "
            f"MLX active={mx.get_active_memory() / 1024**3:.2f} GB, "
            f"peak={mx.get_peak_memory() / 1024**3:.2f} GB",
            flush=True,
        )

    save_json(
        {
            "schema_version": "1.0",
            "source": "qwen3_vl_box",
            "model": MODEL_PATH,
            "box_coordinate_system": "qwen_0_1000",
            "objects": objects,
            "keyframes": keyframes,
            "boxes": all_rows,
        },
        output_path,
    )
    inference_image.unlink(missing_ok=True)
    shutil.rmtree(progress_dir, ignore_errors=True)
    print(f"Qwen box proposals saved: {output_path}", flush=True)


if __name__ == "__main__":
    main()
