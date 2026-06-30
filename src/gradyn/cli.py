from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import typer
from rich.console import Console

from .config import ProcessConfig
from .models import (
    download_public_weights,
    install_mano,
    install_model_runtime_dependencies,
    setup_repositories,
    verify_models,
)
from .pipeline import process as run_pipeline
from .runtime import conda_executable, project_root

app = typer.Typer(no_args_is_help=True)
models_app = typer.Typer(no_args_is_help=True, hidden=True)
app.add_typer(models_app, name="models", hidden=True)
console = Console()


def _checkpoint_status(root: Path) -> dict:
    work = root / ".work"
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    config_hash = str(manifest.get("config_hash", ""))
    completed: dict[str, bool] = {}
    for marker in sorted(work.glob("*.done.json")):
        try:
            payload = json.loads(marker.read_text())
            completed[marker.name.removesuffix(".done.json")] = (
                str(payload.get("config_hash", "")) == config_hash
            )
        except (OSError, json.JSONDecodeError):
            completed[marker.name.removesuffix(".done.json")] = False

    partial: dict[str, dict] = {}
    anchor_progress = work / "object_cluster_progress"
    if anchor_progress.exists():
        metadata_path = anchor_progress / "metadata.json"
        metadata = (
            json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        )
        partial["anchors"] = {
            "completed_keyframes": len(
                [
                    path
                    for path in anchor_progress.glob("*.json")
                    if path.name != "metadata.json"
                ]
            ),
            "total_keyframes": len(
                metadata.get("signature", {}).get("keyframes", [])
            ),
        }

    cutie_progress = work / "cutie_segments"
    if cutie_progress.exists():
        partial["cutie"] = {
            "completed_anchor_intervals": len(
                list(cutie_progress.glob("*.npz"))
            )
        }

    depth_progress = work / "depth_progress.json"
    if depth_progress.exists():
        payload = json.loads(depth_progress.read_text())
        signature = payload.get("signature", {})
        frame_count = int(signature.get("frame_count", 0))
        every = max(int(signature.get("every", 1)), 1)
        partial["depth"] = {
            "completed_frames": len(payload.get("rows", [])),
            "total_frames": (frame_count + every - 1) // every,
            "source_stride": every,
        }

    hands_progress = work / "hands_progress.npz"
    if hands_progress.exists():
        import numpy as np

        with np.load(hands_progress) as payload:
            completed_frames = int(np.asarray(payload["completed"]).sum())
            total_frames = int(len(payload["completed"]))
        partial["hands"] = {
            "completed_frames": completed_frames,
            "total_frames": total_frames,
            "checkpoint_interval_frames": 10,
        }

    return {
        "result": str(root),
        "config_hash": config_hash or None,
        "completed_stages": completed,
        "partial_checkpoints": partial,
        "resume_command": (
            "Rerun the original `gradyn process ... --output "
            f"{root}` command without --no-resume."
        ),
    }


def _environment_check(name: str, code: str) -> dict:
    result = subprocess.run(
        [conda_executable(), "run", "-n", name, "python", "-c", code],
        capture_output=True,
        text=True,
    )
    return {
        "ok": result.returncode == 0,
        "output": (result.stdout or result.stderr).strip(),
    }


def _parse_exemplars(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise typer.BadParameter("Exemplars must use OBJECT=/path/to/image.jpg")
        name, path = value.split("=", 1)
        parsed[name.strip()] = Path(path).expanduser().resolve()
    return parsed


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@app.command()
def process(
    video: Path = typer.Argument(..., exists=True, dir_okay=False),
    camera: str = typer.Option(..., help="Camera make/model."),
    focal_length_px: float | None = typer.Option(
        None,
        "--focal-length-px",
        min=1.0,
        help=(
            "Calibrated effective focal length in pixels for the delivered video "
            "resolution and camera mode."
        ),
    ),
    objects: str = typer.Option(
        "",
        "--objects",
        help=(
            "Comma-separated object names to actively track. Mutually exclusive "
            "with --target-labels."
        ),
    ),
    target_labels: str = typer.Option(
        "",
        "--target-labels",
        help=(
            "Comma-separated label vocabulary for automatic object discovery."
        ),
    ),
    max_auto_objects: int = typer.Option(
        8, min=1, max=20, help="Maximum automatically selected object tracks."
    ),
    anchor_stride: int = typer.Option(
        90,
        "--anchor-stride",
        min=1,
        help="Run GroundingDINO + SAM2.1 anchor generation every N source frames.",
    ),
    anchor_device: str = typer.Option(
        "auto",
        "--anchor-device",
        help="Device for GroundingDINO + SAM2.1 anchors: auto, mps, or cpu.",
    ),
    output: Path = typer.Option(..., "-o", "--output"),
    exemplar: list[str] = typer.Option([], help="Repeat OBJECT=/path/image.jpg."),
    skip_depth: bool = typer.Option(
        False,
        "--skip-depth",
        help="Skip relative-depth generation and produce object/hand outputs only.",
    ),
    depth_every: int = typer.Option(
        1,
        "--depth-every",
        min=1,
        help="Run relative depth every N source frames. Default 1 is full rate.",
    ),
    depth_input_size: int = typer.Option(
        756,
        "--depth-input-size",
        min=280,
        max=1036,
        help="Depth Anything inference size. Default 756 favors finer geometry.",
    ),
    no_resume: bool = typer.Option(False),
) -> None:
    object_names = _split_csv(objects)
    label_names = _split_csv(target_labels)
    if object_names and label_names:
        raise typer.BadParameter("Use --objects or --target-labels, not both.")
    active_labels = object_names or label_names
    if object_names:
        max_auto_objects = min(max_auto_objects, len(object_names))
    if anchor_device not in {"auto", "mps", "cpu"}:
        raise typer.BadParameter("--anchor-device must be auto, mps, or cpu")
    # missing = verify_models()
    # if missing:
    #     console.print("[red]Missing required model assets:[/red]")
    #     for item in missing:
    #         console.print(f"  - {item}")
    #     raise typer.Exit(2)
    config = ProcessConfig(
        video=video.resolve(),
        output=output.resolve(),
        camera=camera,
        focal_length_px=focal_length_px,
        target_labels=active_labels,
        object_mode=bool(object_names),
        max_auto_objects=max_auto_objects,
        anchor_stride=anchor_stride,
        anchor_device=anchor_device,
        exemplars=_parse_exemplars(exemplar),
        depth_every=depth_every,
        depth_input_size=depth_input_size,
        resume=not no_resume,
    )
    run_pipeline(config, skip_depth=skip_depth)
    console.print(f"[bold green]Complete:[/bold green] {config.output}")


@app.command("probe-objects")
def probe_objects(
    video: Path = typer.Argument(..., exists=True, dir_okay=False),
    prompts: str = typer.Option(
        "",
        "--prompts",
        help="Comma-separated GroundingDINO prompts to compare.",
    ),
    prompt: list[str] = typer.Option(
        [],
        "--prompt",
        help="One GroundingDINO prompt. Can be repeated.",
    ),
    frames: str = typer.Option(
        "",
        "--frames",
        help="Optional comma-separated source frame numbers. Defaults to evenly spaced frames.",
    ),
    frame_count: int = typer.Option(
        3,
        "--frame-count",
        min=1,
        max=12,
        help="Number of evenly spaced frames to probe when --frames is omitted.",
    ),
    output: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Probe output directory. Defaults to object-prompt-probes/VIDEO_NAME.",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Device for GroundingDINO + SAM2.1 probe: auto, mps, or cpu.",
    ),
    box_threshold: float = typer.Option(
        0.15,
        "--box-threshold",
        min=0.0,
        max=1.0,
        help="GroundingDINO box threshold.",
    ),
    text_threshold: float = typer.Option(
        0.12,
        "--text-threshold",
        min=0.0,
        max=1.0,
        help="GroundingDINO text threshold.",
    ),
    top_k: int = typer.Option(
        8,
        "--top-k",
        min=1,
        max=30,
        help="Maximum boxes to send to SAM2 per prompt per frame.",
    ),
    max_side: int = typer.Option(
        960,
        "--max-side",
        min=320,
        max=1600,
        help="Maximum inference side for the probe frames.",
    ),
) -> None:
    """Compare object-detection prompts on a few frames without tracking."""
    from .runtime import run_worker

    prompt_values = [*prompt, *_split_csv(prompts)]
    prompt_values = [value.strip() for value in prompt_values if value.strip()]
    if not prompt_values:
        raise typer.BadParameter("Pass at least one --prompt or --prompts value.")
    if len(prompt_values) != len(set(prompt_values)):
        raise typer.BadParameter("Prompt list contains duplicates.")
    if device not in {"auto", "mps", "cpu"}:
        raise typer.BadParameter("--device must be auto, mps, or cpu")
    try:
        frame_values = [int(item) for item in _split_csv(frames)]
    except ValueError as error:
        raise typer.BadParameter("--frames must be comma-separated integers") from error
    if any(value < 0 for value in frame_values):
        raise typer.BadParameter("--frames cannot contain negative values")
    output_dir = (
        output.expanduser().resolve()
        if output is not None
        else (project_root() / "object-prompt-probes" / video.stem).resolve()
    )
    run_worker(
        "gradyn-objects",
        "probe_grounded_sam2.py",
        [
            "--video",
            str(video.expanduser().resolve()),
            "--prompts-json",
            json.dumps(prompt_values),
            "--frames-json",
            json.dumps(frame_values),
            "--frame-count",
            str(frame_count),
            "--output",
            str(output_dir),
            "--device",
            device,
            "--box-threshold",
            str(box_threshold),
            "--text-threshold",
            str(text_threshold),
            "--top-k",
            str(top_k),
            "--max-side",
            str(max_side),
        ],
    )
    console.print(f"[bold green]Prompt probe saved:[/bold green] {output_dir}")
    console.print(f"Open {output_dir / 'contact_sheet.jpg'}")


@app.command()
def doctor() -> None:
    """Check runtime environments, Metal visibility, FFmpeg, and model assets."""
    report = {
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "models": {
            "ok": not verify_models(),
            "problems": verify_models(),
        },
        "core": _environment_check(
            "gradyn-core", "import gradyn; print(gradyn.__version__)"
        ),
        "objects": _environment_check(
            "gradyn-objects",
            "import torch, transformers, sam2; "
            "print({'mps': torch.backends.mps.is_available(), "
            "'transformers': transformers.__version__, 'sam2': True})",
        ),
        "vocab": _environment_check(
            "gradyn-vocab",
            "import mlx.core as mx, mlx_vlm; "
            "print({'metal': mx.metal.is_available(), 'mlx_vlm': True})",
        ),
        "inference": _environment_check(
            "gradyn-inference",
            "import torch, cv2, cutie; "
            "print({'mps': torch.backends.mps.is_available(), "
            "'opencv': cv2.__version__, 'cutie': True})",
        ),
    }
    console.print_json(json.dumps(report))
    if not all(
        [
            report["ffmpeg"],
            report["models"]["ok"],
            report["core"]["ok"],
            report["objects"]["ok"],
            report["inference"]["ok"],
        ]
    ):
        raise typer.Exit(1)


@app.command()
def status(
    output: Path = typer.Argument(..., exists=True, file_okay=False),
) -> None:
    """Show completed stages and in-progress model checkpoints."""
    console.print_json(
        json.dumps(_checkpoint_status(output.expanduser().resolve()))
    )


@app.command("seed-object")
def seed_object(
    output: Path = typer.Argument(..., exists=True, file_okay=False),
    label: str = typer.Option(..., "--label", help="Canonical object label."),
    frame: int = typer.Option(..., "--frame", min=0, help="Source frame index."),
    box: str = typer.Option(
        ...,
        "--box",
        help="Source-pixel box as x1,y1,x2,y2.",
    ),
) -> None:
    """Add one visual object box used to initialize anchor generation and Cutie."""
    root = output.expanduser().resolve()
    metadata = json.loads((root / "source" / "video_metadata.json").read_text())
    if frame >= int(metadata["frame_count"]):
        raise typer.BadParameter(
            f"--frame must be below {metadata['frame_count']}"
        )
    try:
        values = [float(value.strip()) for value in box.split(",")]
    except ValueError as error:
        raise typer.BadParameter("--box must contain four numbers") from error
    if len(values) != 4:
        raise typer.BadParameter("--box must be x1,y1,x2,y2")
    x0, y0, x1, y1 = values
    width = float(metadata["decoded_width"])
    height = float(metadata["decoded_height"])
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise typer.BadParameter(
            f"--box must lie inside the {int(width)}x{int(height)} source frame"
        )
    canonical = label.strip()
    if not canonical:
        raise typer.BadParameter("--label cannot be empty")
    seed_path = root / "objects" / "manual_seeds.json"
    seeds = json.loads(seed_path.read_text()) if seed_path.exists() else []
    seeds = [
        seed
        for seed in seeds
        if not (
            str(seed["label"]).casefold() == canonical.casefold()
            and int(seed["frame_index"]) == frame
        )
    ]
    seeds.append(
        {
            "label": canonical,
            "frame_index": frame,
            "box_xyxy": values,
            "source": "user_visual_initialization",
        }
    )
    temporary_seed = seed_path.with_suffix(".json.tmp")
    temporary_seed.write_text(json.dumps(seeds, indent=2))
    temporary_seed.replace(seed_path)

    console.print(
        f"[bold green]Saved object seed:[/bold green] {canonical} at frame "
        f"{frame}, box {values}"
    )
    console.print("Run `gradyn rebuild-objects RESULT` to apply it.")


@app.command("rebuild-objects", hidden=True)
def rebuild_objects(
    output: Path = typer.Argument(..., exists=True, file_okay=False),
) -> None:
    """Rebuild object anchors/Cutie tracks while preserving preprocessing, hands, and depth."""
    from .exports import build_exports
    from .quality import build_quality_report
    from .runtime import mark_stage, run_worker

    root = output.expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    config = manifest["config"]
    config_hash = str(manifest["config_hash"])
    common = ["--job", str(root)]
    target_labels = [
        str(value)
        for value in config.get("target_labels", config.get("objects", []))
    ]
    anchor_args = [
        *common,
        "--target-labels-json",
        json.dumps(target_labels),
        "--max-auto-objects",
        str(config.get("max_auto_objects", 8)),
        "--exemplars-json",
        json.dumps(config.get("exemplars", {})),
        "--stride",
        str(config.get("anchor_stride", 90)),
        "--max-side",
        str(config.get("max_inference_side", 960)),
        "--device",
        str(config.get("anchor_device", "auto")),
    ]
    if bool(config.get("object_mode", False)):
        anchor_args.append("--object-mode")
    manual_seeds = root / "objects" / "manual_seeds.json"
    if manual_seeds.exists():
        anchor_args.extend(["--manual-seeds-json", str(manual_seeds)])

    run_worker("gradyn-objects", "grounded_sam2_discover.py", anchor_args)
    mark_stage(root / ".work", "anchors", config_hash)
    run_worker(
        "gradyn-inference",
        "cutie_track.py",
        ["--job", str(root)],
    )
    mark_stage(root / ".work", "cutie", config_hash)
    build_quality_report(root)
    build_exports(root)
    console.print(
        f"[bold green]Object tracking rebuilt:[/bold green] {root / 'objects'}"
    )


@app.command("rebuild-tracks", hidden=True)
def rebuild_tracks(
    output: Path = typer.Argument(..., exists=True, file_okay=False),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Cutie device for rebuilding object tracks: auto, mps, or cpu.",
    ),
) -> None:
    """Rebuild only Cutie tracks from the existing object anchors."""
    from .exports import build_exports
    from .quality import build_quality_report
    from .runtime import mark_stage, run_worker

    root = output.expanduser().resolve()
    discoveries = root / ".work" / "anchor_discoveries.json"
    if not discoveries.exists():
        raise typer.BadParameter(
            "Existing object anchors are missing; use `gradyn rebuild-objects`."
        )
    if device not in {"auto", "mps", "cpu"}:
        raise typer.BadParameter("--device must be auto, mps, or cpu")
    manifest = json.loads((root / "manifest.json").read_text())
    config = manifest["config"]
    config_hash = str(manifest["config_hash"])
    run_worker(
        "gradyn-inference",
        "cutie_track.py",
        ["--job", str(root), "--device", device],
    )
    mark_stage(root / ".work", "cutie", config_hash)
    build_quality_report(root)
    build_exports(root)
    console.print(
        f"[bold green]Object tracks rebuilt:[/bold green] {root / 'objects'}"
    )


@app.command("add-depth", hidden=True)
def add_depth(
    output: Path = typer.Argument(..., exists=True, file_okay=False),
    every: int = typer.Option(
        1,
        "--every",
        min=1,
        help="Run relative depth every N source frames. Default 1 is full rate.",
    ),
    input_size: int = typer.Option(
        756,
        "--input-size",
        min=280,
        max=1036,
        help="Depth Anything inference size.",
    ),
) -> None:
    """Add or replace relative-depth output in an existing Gradyn result."""
    from .exports import build_exports
    from .quality import build_quality_report
    from .runtime import mark_stage, run_worker

    root = output.expanduser().resolve()
    run_worker(
        "gradyn-inference",
        "depth_anything_video.py",
        [
            "--job",
            str(root),
            "--every",
            str(every),
            "--input-size",
            str(input_size),
        ],
    )
    manifest = json.loads((root / "manifest.json").read_text())
    mark_stage(root / ".work", "depth", str(manifest["config_hash"]))
    build_quality_report(root)
    build_exports(root)
    console.print(f"[bold green]Depth complete:[/bold green] {root / 'depth'}")


@app.command("rebuild-hands", hidden=True)
def rebuild_hands(
    output: Path = typer.Argument(..., exists=True, file_okay=False),
    device: str = typer.Option(
        "cpu",
        "--device",
        help="WiLoR device. Use CPU for production; MPS is experimental.",
    ),
) -> None:
    """Rebuild WiLoR hands without rerunning objects or relative depth."""
    from .exports import build_exports
    from .quality import build_quality_report
    from .runtime import mark_stage, run_worker

    if device not in {"cpu", "mps"}:
        raise typer.BadParameter("--device must be cpu or mps")
    root = output.expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    config_hash = str(manifest["config_hash"])
    run_worker(
        "gradyn-inference",
        "hands_camera.py",
        ["--job", str(root), "--device", device],
    )
    mark_stage(root / ".work", "hands", config_hash)
    build_quality_report(root)
    build_exports(root)
    console.print(
        f"[bold green]Hand reconstruction rebuilt:[/bold green] "
        f"{root / 'hands'}"
    )


@app.command("calibrate-camera", hidden=True)
def calibrate_camera(
    video: Path = typer.Argument(..., exists=True, dir_okay=False),
    square_size_mm: float = typer.Option(
        ..., "--square-size-mm", min=0.1, help="Physical checker square size."
    ),
    output: Path = typer.Option(..., "-o", "--output"),
    columns: int = typer.Option(9, min=2, help="Checkerboard inner-corner columns."),
    rows: int = typer.Option(6, min=2, help="Checkerboard inner-corner rows."),
    sample_every: int = typer.Option(10, min=1),
) -> None:
    """Calibrate effective intrinsics for one exact camera capture mode."""
    from .runtime import run_worker

    run_worker(
        "gradyn-inference",
        "calibrate_camera.py",
        [
            "--video",
            str(video.resolve()),
            "--output",
            str(output.resolve()),
            "--columns",
            str(columns),
            "--rows",
            str(rows),
            "--square-size-mm",
            str(square_size_mm),
            "--sample-every",
            str(sample_every),
        ],
    )
    console.print(f"[bold green]Calibration written:[/bold green] {output.resolve()}")


@models_app.command("setup")
def models_setup() -> None:
    setup_repositories()
    install_model_runtime_dependencies()
    download_public_weights()


@models_app.command("install-mano")
def models_install_mano(
    right: Path = typer.Option(..., exists=True, dir_okay=False),
    left: Path = typer.Option(..., exists=True, dir_okay=False),
) -> None:
    install_mano(right.resolve(), left.resolve())


@models_app.command("verify")
def models_verify() -> None:
    missing = verify_models()
    if missing:
        console.print(json.dumps({"ready": False, "missing": missing}, indent=2))
        raise typer.Exit(1)
    console.print(json.dumps({"ready": True}, indent=2))
