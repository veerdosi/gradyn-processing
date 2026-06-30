from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from common import resize_for_max_side, save_json
    from grounded_sam2_discover import clamp_box, box_iou, xyxy_from_mask
except ModuleNotFoundError:
    from workers.common import resize_for_max_side, save_json
    from workers.grounded_sam2_discover import clamp_box, box_iou, xyxy_from_mask

SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
COLORS = [
    (255, 48, 96),
    (0, 186, 255),
    (255, 190, 40),
    (40, 220, 120),
    (190, 110, 255),
    (255, 120, 50),
]


def choose_torch_device(requested: str):
    import torch

    if requested == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested but is not available to PyTorch.")
    return torch.device(requested)


def video_metadata(video: Path) -> dict:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        "-select_streams",
        "v:0",
        str(video),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def frame_count_from_metadata(metadata: dict) -> int:
    stream = metadata["streams"][0]
    value = stream.get("nb_frames")
    if value and str(value).isdigit():
        return int(value)
    duration = float(metadata.get("format", {}).get("duration") or 0.0)
    rate_text = stream.get("avg_frame_rate", "0/1")
    numerator, denominator = [float(part) for part in rate_text.split("/")]
    fps = numerator / denominator if denominator else 0.0
    return max(int(round(duration * fps)), 1)


def default_probe_frames(frame_count: int, count: int) -> list[int]:
    count = max(1, min(count, frame_count))
    if count == 1:
        return [frame_count // 2]
    return sorted(
        {
            int(round(index * (frame_count - 1) / (count - 1)))
            for index in range(count)
        }
    )


def extract_frames(video: Path, frame_indices: list[int], output_dir: Path) -> dict[int, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = sorted(set(frame_indices))
    if not selected:
        raise SystemExit("No frames selected for probing.")
    expression = "+".join(f"eq(n\\,{index})" for index in selected)
    output_pattern = output_dir / "probe_%08d.jpg"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-vf",
        f"select={expression},showinfo",
        "-vsync",
        "0",
        "-q:v",
        "2",
        str(output_pattern),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(result.stderr.strip() or "FFmpeg frame extraction failed.")
    extracted = sorted(output_dir.glob("probe_*.jpg"))
    if len(extracted) != len(selected):
        raise SystemExit(
            f"Expected {len(selected)} probe frames, but FFmpeg wrote {len(extracted)}."
        )
    return dict(zip(selected, extracted, strict=True))


def mask_area_fraction(mask: np.ndarray) -> float:
    return float(mask.sum() / max(mask.shape[0] * mask.shape[1], 1))


def draw_probe(
    image: Image.Image,
    detections: list[dict],
    prompt: str,
    frame_index: int,
    max_side: int,
) -> Image.Image:
    scale = max_side / max(image.size) if max(image.size) > max_side else 1.0
    canvas = image.resize(
        (int(round(image.width * scale)), int(round(image.height * scale))),
        Image.Resampling.BILINEAR,
    ).convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for rank, detection in enumerate(detections, start=1):
        color = COLORS[(rank - 1) % len(COLORS)]
        box = [float(value) * scale for value in detection["box_xyxy"]]
        mask = Image.fromarray(np.asarray(detection["mask"], dtype=np.uint8) * 130)
        mask = mask.resize(canvas.size, Image.Resampling.NEAREST)
        tint = Image.new("RGBA", canvas.size, (*color, 0))
        tint.putalpha(mask)
        overlay.alpha_composite(tint)
        draw.rectangle(box, outline=(*color, 255), width=3)
        label = f"{rank}:{detection['detector_score']:.2f}/{detection['sam2_score']:.2f}"
        text_box = draw.textbbox((box[0], box[1]), label)
        draw.rectangle(
            [text_box[0] - 2, text_box[1] - 2, text_box[2] + 4, text_box[3] + 3],
            fill=(*color, 230),
        )
        draw.text((box[0] + 1, box[1]), label, fill=(0, 0, 0, 255))
    composed = Image.alpha_composite(canvas, overlay).convert("RGB")
    header_height = 42
    out = Image.new("RGB", (composed.width, composed.height + header_height), "white")
    out.paste(composed, (0, header_height))
    header = ImageDraw.Draw(out)
    title = f"frame {frame_index} | {prompt}"
    if len(detections) == 0:
        title += " | no detections"
    header.rectangle([0, 0, out.width, header_height], fill=(24, 24, 28))
    header.text((10, 12), title, fill=(255, 255, 255))
    return out


def make_contact_sheet(images: list[tuple[str, Image.Image]], columns: int) -> Image.Image:
    if not images:
        raise SystemExit("No probe images were generated.")
    columns = max(1, columns)
    cell_width = max(image.width for _, image in images)
    cell_height = max(image.height for _, image in images)
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (cell_width * columns, cell_height * rows), (245, 245, 245))
    for index, (_, image) in enumerate(images):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        sheet.paste(image, (x, y))
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--prompts-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames-json", default="[]")
    parser.add_argument("--frame-count", type=int, default=3)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--max-side", type=int, default=960)
    parser.add_argument("--box-threshold", type=float, default=0.20)
    parser.add_argument("--text-threshold", type=float, default=0.12)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    root = Path(__file__).resolve().parents[1]
    sam2_repo = root / "models" / "sam2"
    sam2_checkpoint = sam2_repo / "checkpoints" / "sam2.1_hiera_small.pt"
    grounding_dino_dir = root / "models" / "grounding-dino-base"
    for required, message in [
        (grounding_dino_dir / "config.json", "GroundingDINO base weights are missing."),
        (sam2_repo / "sam2" / "__init__.py", "SAM2 repository is missing."),
        (sam2_checkpoint, "SAM2.1 small weights are missing."),
    ]:
        if not required.exists():
            raise SystemExit(message + " Run `gradyn models setup`.")
    sys.path.insert(0, str(sam2_repo))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    video = Path(args.video).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    prompts = [str(prompt).strip() for prompt in json.loads(args.prompts_json)]
    prompts = [prompt for prompt in prompts if prompt]
    if not prompts:
        raise SystemExit("At least one prompt is required.")
    metadata = video_metadata(video)
    total_frames = frame_count_from_metadata(metadata)
    explicit_frames = [int(value) for value in json.loads(args.frames_json)]
    frame_indices = explicit_frames or default_probe_frames(total_frames, args.frame_count)
    frame_indices = [index for index in frame_indices if 0 <= index < total_frames]
    if not frame_indices:
        raise SystemExit(f"No selected frames are inside 0..{total_frames - 1}.")
    frame_paths = extract_frames(video, frame_indices, output / "frames")

    device = choose_torch_device(args.device)
    cpu_device = torch.device("cpu")
    print(
        f"Loading GroundingDINO base and SAM2.1 small on {device.type}.",
        flush=True,
    )
    gd_processor = AutoProcessor.from_pretrained(str(grounding_dino_dir))
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        str(grounding_dino_dir)
    ).to(device).eval()
    sam2_model = build_sam2(SAM2_CONFIG, str(sam2_checkpoint), device=device, mode="eval")
    sam2 = SAM2ImagePredictor(sam2_model)

    summary: dict = {
        "video": str(video),
        "frame_count": total_frames,
        "selected_frames": frame_indices,
        "prompts": prompts,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "results": [],
    }
    sheet_items: list[tuple[str, Image.Image]] = []
    for frame_number, frame_index in enumerate(frame_indices, start=1):
        original = Image.open(frame_paths[frame_index]).convert("RGB")
        image, scale = resize_for_max_side(original, args.max_side)
        width, height = image.size
        sam2.set_image(np.asarray(image).copy())
        print(
            f"Probe frame {frame_number}/{len(frame_indices)} "
            f"(source frame {frame_index})",
            flush=True,
        )
        for prompt in prompts:
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
            raw_boxes: list[tuple[np.ndarray, float]] = []
            for box, score in zip(results.get("boxes", []), results.get("scores", []), strict=False):
                box_array = clamp_box(
                    box.detach().to(cpu_device).numpy().astype(np.float32),
                    width,
                    height,
                )
                if box_array[2] <= box_array[0] + 3 or box_array[3] <= box_array[1] + 3:
                    continue
                if any(box_iou(box_array, existing[0]) > 0.88 for existing in raw_boxes):
                    continue
                raw_boxes.append((box_array, float(score.detach().cpu())))
            raw_boxes.sort(key=lambda item: item[1], reverse=True)
            detections: list[dict] = []
            for box, detector_score in raw_boxes[: args.top_k]:
                with torch.inference_mode():
                    masks, mask_scores, _ = sam2.predict(box=box, multimask_output=True)
                best_mask = None
                best_score = -1.0
                for mask, score in zip(masks, mask_scores, strict=False):
                    if float(score) > best_score:
                        best_mask = np.asarray(mask, dtype=bool)
                        best_score = float(score)
                if best_mask is None or not best_mask.any():
                    continue
                mask_box = xyxy_from_mask(best_mask)
                output_mask = best_mask
                output_box = mask_box.copy()
                if scale != 1.0:
                    output_mask = np.asarray(
                        Image.fromarray(best_mask.astype(np.uint8) * 255).resize(
                            original.size,
                            Image.Resampling.NEAREST,
                        )
                    ) > 0
                    output_box = output_box / scale
                detections.append(
                    {
                        "box_xyxy": [float(value) for value in output_box.tolist()],
                        "detector_score": detector_score,
                        "sam2_score": best_score,
                        "area_fraction": mask_area_fraction(best_mask),
                        "mask": output_mask,
                    }
                )
            image_out = draw_probe(original, detections, prompt, frame_index, 820)
            image_path = output / f"frame_{frame_index:08d}__{prompt.replace('/', '_')[:60]}.jpg"
            image_out.save(image_path, quality=92)
            sheet_items.append((f"{frame_index}:{prompt}", image_out))
            summary["results"].append(
                {
                    "frame_index": frame_index,
                    "prompt": prompt,
                    "image": str(image_path),
                    "detections": [
                        {key: value for key, value in detection.items() if key != "mask"}
                        for detection in detections
                    ],
                }
            )
        if device.type == "mps":
            torch.mps.empty_cache()

    sheet = make_contact_sheet(sheet_items, columns=len(prompts))
    sheet_path = output / "contact_sheet.jpg"
    sheet.save(sheet_path, quality=92)
    summary["contact_sheet"] = str(sheet_path)
    save_json(summary, output / "summary.json")
    print(f"Probe saved: {output}", flush=True)


if __name__ == "__main__":
    main()
