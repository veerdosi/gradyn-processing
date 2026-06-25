from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

try:
    from common import frame_paths, job_paths, save_json
except ModuleNotFoundError:
    from workers.common import frame_paths, job_paths, save_json


MODEL_PATH = "models/qwen3-vl-4b"
DISCOVERY_PROMPT = """You are building a general physical-object vocabulary for
robotics video tracking.

Inspect this egocentric frame from a hands-on work activity. The activity may involve
fabrication, assembly, maintenance, packaging, printing, lamination, food preparation,
laboratory work, crafts, cleaning, or another practical task. Do not assume a particular
industry or task from this prompt.

Return only distinct, visible physical objects that are useful for understanding hand
interaction and the task.

Include when visible:
- objects being touched, moved, aligned, inserted, removed, operated, or inspected
- tools, utensils, instruments, controls, machines, and fixtures
- bounded task materials such as a paper sheet, printed page, film sheet, panel, part,
  workpiece, package, label, roll, or container
- nearby objects likely to become part of the task, even if not currently held
- safety equipment only when directly manipulated or task-relevant

Name each object by its most specific visually supported category. Prefer "laminator",
"paper sheet", or "roller" over generic words such as "machine", "material", or
"equipment". Do not guess a brand, exact model, hidden object, material composition, or
purpose that cannot be seen.

Exclude:
- people, hands, feet, clothing, footwear and body parts
- actions, verbs, process names, tasks, and abstract concepts
- unbounded substances or generic materials unless they form a distinct trackable item
- text content, logos, colors, states, and duplicate synonyms
- smoke, sparks, glare, reflections, shadows, rooms, floors, walls, and background

Return strict JSON only:
{"objects": ["concise singular object name", "..."]}
"""

IGNORED_LABELS = {
    "action",
    "arm",
    "background",
    "building",
    "ceiling",
    "clothing",
    "equipment",
    "factory",
    "fire",
    "flame",
    "floor",
    "foot",
    "hand",
    "human",
    "light",
    "machine shop",
    "man",
    "metal",
    "person",
    "reflection",
    "repair",
    "room",
    "shadow",
    "smoke",
    "spark",
    "steel",
    "protective gear",
    "welding helmet",
    "welding mask",
    "wall",
    "woman",
    "workshop",
}

SYNONYMS = {
    "angle grinder machine": "angle grinder",
    "grinding machine": "angle grinder",
    "vice": "bench vise",
    "vise": "bench vise",
    "clamps": "clamp",
    "fixturing": "fixture",
    "machines": "machine",
    "laminating machine": "laminator",
    "lamination machine": "laminator",
    "laminating machines": "laminator",
    "papers": "paper sheet",
    "paper": "paper sheet",
    "printed paper": "printed page",
    "printed papers": "printed page",
    "printed sheet": "printed page",
    "plastic film": "laminating film",
    "lamination film": "laminating film",
    "laminating sheet": "laminating film",
    "rollers": "roller",
    "sheets": "sheet",
    "welding clamp": "clamp",
    "welding container": "container",
    "welding fixture": "fixture",
    "welding gun": "welding torch",
    "welder": "welding torch",
    "work piece": "workpiece",
}

COLOR_PREFIXES = {
    "black",
    "blue",
    "brown",
    "gray",
    "green",
    "grey",
    "orange",
    "pink",
    "purple",
    "red",
    "white",
    "yellow",
}


def normalize_object_name(value: str) -> str | None:
    label = re.sub(r"\s+", " ", str(value).strip().lower().replace("_", " "))
    label = label.strip(" .,:;|/\\\"'[]{}()")
    words = label.split()
    if len(words) > 1 and words[0] in COLOR_PREFIXES:
        label = " ".join(words[1:])
    label = SYNONYMS.get(label, label)
    if not label or len(label) < 3 or label in IGNORED_LABELS:
        return None
    if label.isdigit() or len(label.split()) > 5:
        return None
    return label


def parse_object_response(text: str) -> list[str]:
    candidates = [text.strip()]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    left, right = text.find("{"), text.rfind("}")
    if left >= 0 and right > left:
        candidates.insert(0, text[left : right + 1])
    payload = None
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue
    values = payload.get("objects", []) if isinstance(payload, dict) else []
    if not isinstance(values, list):
        return []
    normalized = []
    for value in values:
        label = normalize_object_name(value)
        if label and label not in normalized:
            normalized.append(label)
    return normalized


def aggregate_objects(
    labels_by_frame: dict[int, list[str]],
    *,
    max_candidates: int,
    minimum_hits: int = 2,
) -> list[dict]:
    normalized_by_frame = {
        frame_index: {
            label
            for value in values
            if (label := normalize_object_name(value))
        }
        for frame_index, values in labels_by_frame.items()
    }
    counts = Counter(
        label for labels in normalized_by_frame.values() for label in labels
    )
    total = max(len(normalized_by_frame), 1)
    candidates = [
        {
            "label": label,
            "keyframe_hits": hits,
            "hit_rate": hits / total,
            "source": "qwen3_vl",
        }
        for label, hits in counts.items()
        if hits >= minimum_hits or total == 1
    ]
    labels = {item["label"] for item in candidates}
    candidates = [
        item
        for item in candidates
        if not (
            item["label"] in {"machine", "tool", "container", "fixture"}
            and any(
                other != item["label"] and other.endswith(" " + item["label"])
                for other in labels
            )
        )
    ]
    candidates.sort(
        key=lambda item: (
            item["keyframe_hits"],
            item["hit_rate"],
            len(item["label"]),
        ),
        reverse=True,
    )
    return candidates[:max_candidates]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--stride", type=int, default=360)
    parser.add_argument("--max-candidates", type=int, default=24)
    parser.add_argument("--max-side", type=int, default=672)
    parser.add_argument("--max-tokens", type=int, default=160)
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    root = Path(__file__).resolve().parents[1]
    model_path = root / MODEL_PATH
    if not (model_path / "config.json").exists():
        raise SystemExit(
            "Qwen3-VL weights are missing. Run `gradyn models setup`."
        )

    paths = job_paths(args.job)
    frames = frame_paths(args.job)
    keyframes = sorted(
        set(
            [
                0,
                *range(args.stride, len(frames), args.stride),
                max(len(frames) - 1, 0),
            ]
        )
    )
    checkpoint_dir = paths["work"] / "qwen_progress"
    metadata_path = checkpoint_dir / "metadata.json"
    signature = {
        "frame_count": len(frames),
        "keyframes": keyframes,
        "model": MODEL_PATH,
        "max_side": args.max_side,
        "max_tokens": args.max_tokens,
        "prompt": DISCOVERY_PROMPT,
    }
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("signature") != signature:
            shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_json({"signature": signature}, metadata_path)

    print(
        f"Loading Qwen3-VL 4B 4-bit with MLX; "
        f"{len(keyframes)} frames will be streamed one at a time.",
        flush=True,
    )
    mx.set_memory_limit(6 * 1024**3)
    mx.reset_peak_memory()
    model, processor = load(str(model_path))
    prompt = apply_chat_template(
        processor,
        model.config,
        DISCOVERY_PROMPT,
        num_images=1,
    )
    labels_by_frame: dict[int, list[str]] = {}
    raw_frames: list[dict] = []
    inference_image = paths["work"] / "qwen_current_frame.jpg"
    for keyframe_number, frame_index in enumerate(keyframes, start=1):
        frame_checkpoint = checkpoint_dir / f"{frame_index:08d}.json"
        if frame_checkpoint.exists():
            record = json.loads(frame_checkpoint.read_text())
            labels_by_frame[frame_index] = record.get("objects", [])
            raw_frames.append(record)
            print(f"↷ Qwen reusing source frame {frame_index}", flush=True)
            continue

        print(
            f"Qwen frame {keyframe_number}/{len(keyframes)} "
            f"(source {frame_index})",
            flush=True,
        )
        with Image.open(frames[frame_index]) as source:
            image = source.convert("RGB")
            image.thumbnail((args.max_side, args.max_side), Image.Resampling.LANCZOS)
            image.save(inference_image, quality=90)
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
        objects = parse_object_response(text)
        record = {
            "frame_index": frame_index,
            "objects": objects,
            "raw_response": text,
        }
        labels_by_frame[frame_index] = objects
        raw_frames.append(record)
        save_json(record, frame_checkpoint)
        del image, result
        mx.clear_cache()
        print(
            f"  objects={objects or 'none'}; "
            f"MLX active={mx.get_active_memory() / 1024**3:.2f} GB, "
            f"peak={mx.get_peak_memory() / 1024**3:.2f} GB",
            flush=True,
        )

    candidates = aggregate_objects(
        labels_by_frame,
        max_candidates=args.max_candidates,
        # Human approval follows this stage, so retain brief one-frame tools
        # instead of silently dropping them before review.
        minimum_hits=1,
    )
    save_json(
        {
            "model": "mlx-community/Qwen3-VL-4B-Instruct-4bit",
            "sampling_stride_frames": args.stride,
            "keyframes": keyframes,
            "frames": sorted(raw_frames, key=lambda item: item["frame_index"]),
            "candidates": candidates,
        },
        paths["objects"] / "qwen_discovery.json",
    )
    save_json(
        [item["label"] for item in candidates],
        paths["work"] / "qwen_candidate_prompts.json",
    )
    print(
        "Qwen candidate objects: "
        + (", ".join(item["label"] for item in candidates) or "none"),
        flush=True,
    )
    inference_image.unlink(missing_ok=True)
    shutil.rmtree(checkpoint_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
