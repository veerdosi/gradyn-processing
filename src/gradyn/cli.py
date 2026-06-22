from __future__ import annotations

import json
import hashlib
import shutil
import subprocess
from pathlib import Path

import typer
from rich.console import Console

from .config import ProcessConfig
from .models import download_public_weights, install_mano, setup_repositories, verify_models
from .pipeline import process as run_pipeline
from .runtime import conda_executable, project_root

app = typer.Typer(no_args_is_help=True)
models_app = typer.Typer(no_args_is_help=True)
app.add_typer(models_app, name="models")
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
    qwen_progress = work / "qwen_progress"
    if qwen_progress.exists():
        metadata_path = qwen_progress / "metadata.json"
        metadata = (
            json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        )
        partial["qwen"] = {
            "completed_keyframes": len(
                [
                    path
                    for path in qwen_progress.glob("*.json")
                    if path.name != "metadata.json"
                ]
            ),
            "total_keyframes": len(
                metadata.get("signature", {}).get("keyframes", [])
            ),
        }

    sam3_progress = work / "sam3_progress"
    if sam3_progress.exists():
        metadata_path = sam3_progress / "metadata.json"
        metadata = (
            json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        )
        partial["sam3"] = {
            "completed_keyframes": len(
                [
                    path
                    for path in sam3_progress.glob("*.json")
                    if path.name != "metadata.json"
                ]
            ),
            "total_keyframes": len(
                metadata.get("signature", {}).get("keyframes", [])
            ),
        }

    sam2_progress = work / "sam2_checkpoints"
    if sam2_progress.exists():
        metadata_path = sam2_progress / "metadata.json"
        metadata = (
            json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        )
        signature = metadata.get("signature", {})
        total = (
            len(signature.get("ranges", []))
            * len(signature.get("objects", {}))
            * 2
        )
        partial["sam2"] = {
            "completed_object_chunks": len(list(sam2_progress.glob("*.npz"))),
            "total_object_chunks": total,
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
        help=(
            "Comma-separated object names for SAM 3 localization and SAM 2 tracking."
        ),
    ),
    prompt_bank: Path | None = typer.Option(
        None,
        help="Task-specific JSON vocabulary for SAM 3 verification.",
    ),
    discover_objects: bool = typer.Option(
        False,
        "--discover-objects",
        help=(
            "Opt in to Qwen3-VL vocabulary discovery when object names are unknown. "
            "Qwen is disabled by default."
        ),
    ),
    max_auto_objects: int = typer.Option(
        8, min=1, max=20, help="Maximum automatically selected object classes."
    ),
    qwen_max_candidates: int = typer.Option(
        24,
        "--qwen-max-candidates",
        min=1,
        max=100,
        help="Maximum Qwen-proposed objects offered for approval.",
    ),
    qwen_stride: int = typer.Option(
        360,
        "--qwen-stride",
        min=1,
        help="Run Qwen vocabulary discovery every N source frames.",
    ),
    sam3_stride: int = typer.Option(
        90,
        "--sam3-stride",
        min=1,
        help="Run SAM 3 verification/localization every N source frames.",
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
    object_names = [item.strip() for item in objects.split(",") if item.strip()]
    vocabulary_sources = sum(
        [bool(object_names), prompt_bank is not None, discover_objects]
    )
    if vocabulary_sources == 0:
        raise typer.BadParameter(
            "Specify --objects, use --prompt-bank, or explicitly enable "
            "--discover-objects."
        )
    if vocabulary_sources > 1:
        raise typer.BadParameter(
            "Use exactly one object vocabulary source: --objects, --prompt-bank, "
            "or --discover-objects."
        )
    missing = verify_models(include_qwen=discover_objects)
    if missing:
        console.print("[red]Missing required model assets:[/red]")
        for item in missing:
            console.print(f"  - {item}")
        raise typer.Exit(2)
    config = ProcessConfig(
        video=video.resolve(),
        output=output.resolve(),
        camera=camera,
        focal_length_px=focal_length_px,
        objects=object_names,
        prompt_bank=prompt_bank.expanduser().resolve() if prompt_bank else None,
        discover_objects=discover_objects,
        max_auto_objects=max_auto_objects,
        qwen_max_candidates=qwen_max_candidates,
        qwen_stride=qwen_stride,
        sam3_stride=sam3_stride,
        exemplars=_parse_exemplars(exemplar),
        depth_every=depth_every,
        depth_input_size=depth_input_size,
        resume=not no_resume,
    )
    run_pipeline(config, skip_depth=skip_depth)
    console.print(f"[bold green]Complete:[/bold green] {config.output}")


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
            "import torch, mlx.core as mx, sam2, sam3; "
            "print({'mps': torch.backends.mps.is_available(), "
            "'metal': mx.metal.is_available()})",
        ),
        "vocab": _environment_check(
            "gradyn-vocab",
            "import mlx.core as mx, mlx_vlm; "
            "print({'metal': mx.metal.is_available(), 'mlx_vlm': True})",
        ),
        "inference": _environment_check(
            "gradyn-inference",
            "import torch, cv2; "
            "print({'mps': torch.backends.mps.is_available(), "
            "'opencv': cv2.__version__})",
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


@app.command("select-objects")
def select_objects(
    output: Path = typer.Argument(..., exists=True, file_okay=False),
    objects: str = typer.Option(
        ...,
        "--objects",
        help="Comma-separated Qwen-proposed labels to send to SAM 3.",
    ),
) -> None:
    """Approve Qwen-proposed labels before SAM 3 localization."""
    root = output.expanduser().resolve()
    qwen_discovery = root / "objects" / "qwen_discovery.json"
    sam3_discovery = root / "objects" / "discovery.json"
    discovery_path = qwen_discovery if qwen_discovery.exists() else sam3_discovery
    if not discovery_path.exists():
        raise typer.BadParameter(
            "No automatic discovery exists yet. Run `gradyn process` first."
        )
    discovery = json.loads(discovery_path.read_text())
    source_items = (
        discovery.get("candidates", [])
        if discovery_path == qwen_discovery
        else discovery.get("selected_objects", [])
    )
    available = {str(item["label"]).casefold(): item for item in source_items}
    requested = [
        value.strip() for value in objects.split(",") if value.strip()
    ]
    if not requested:
        raise typer.BadParameter("Select at least one object.")
    unknown = [value for value in requested if value.casefold() not in available]
    if unknown:
        raise typer.BadParameter(
            "Unknown or unverified labels: "
            + ", ".join(unknown)
            + ". Available: "
            + ", ".join(item["label"] for item in available.values())
        )
    selected = [available[value.casefold()] for value in requested]
    approval = {
        "discovery_sha256": hashlib.sha256(
            discovery_path.read_bytes()
        ).hexdigest(),
        "source": "qwen3_vl" if discovery_path == qwen_discovery else "sam3",
        "selected_labels": [str(item["label"]) for item in selected],
    }
    approval_path = root / "objects" / "approved_prompts.json"
    temporary = approval_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(approval, indent=2))
    temporary.replace(approval_path)
    candidate_path = root / ".work" / "approved_candidate_prompts.json"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_candidates = candidate_path.with_suffix(".json.tmp")
    temporary_candidates.write_text(
        json.dumps(approval["selected_labels"], indent=2)
    )
    temporary_candidates.replace(candidate_path)
    console.print(
        "[bold green]Approved for SAM 3 verification:[/bold green] "
        + ", ".join(item["label"] for item in selected)
    )
    console.print("Rerun the original `gradyn process ...` command to continue.")


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
    """Add one visual object box used to initialize SAM 3 and SAM 2."""
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

    candidate_path = root / ".work" / "approved_candidate_prompts.json"
    candidates = json.loads(candidate_path.read_text()) if candidate_path.exists() else []
    if canonical not in candidates:
        candidates.append(canonical)
    temporary_candidates = candidate_path.with_suffix(".json.tmp")
    temporary_candidates.write_text(json.dumps(candidates, indent=2))
    temporary_candidates.replace(candidate_path)
    console.print(
        f"[bold green]Saved object seed:[/bold green] {canonical} at frame "
        f"{frame}, box {values}"
    )
    console.print("Run `gradyn rebuild-objects RESULT` to apply it.")


@app.command("rebuild-objects")
def rebuild_objects(
    output: Path = typer.Argument(..., exists=True, file_okay=False),
) -> None:
    """Rebuild SAM 3/SAM 2 objects while preserving preprocessing, hands, and depth."""
    from .exports import build_exports
    from .quality import build_quality_report
    from .runtime import mark_stage, run_worker

    root = output.expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    config = manifest["config"]
    config_hash = str(manifest["config_hash"])
    common = ["--job", str(root)]
    object_names = [str(value) for value in config.get("objects", [])]
    prompt_bank = config.get("prompt_bank")
    sam3_args = [
        *common,
        "--objects-json",
        json.dumps(object_names),
        "--max-auto-objects",
        str(config.get("max_auto_objects", 8)),
        "--exemplars-json",
        json.dumps(config.get("exemplars", {})),
        "--stride",
        str(config.get("sam3_stride", 90)),
        "--max-side",
        str(config.get("max_inference_side", 960)),
    ]
    if not object_names and not prompt_bank:
        candidate_path = root / ".work" / "approved_candidate_prompts.json"
        if not candidate_path.exists():
            raise typer.BadParameter(
                "No approved object list exists. Run `gradyn select-objects` first."
            )
        sam3_args.extend(
            [
                "--candidate-prompts-json",
                str(candidate_path),
                "--candidate-source",
                "qwen3_vl_approved",
            ]
        )
    if prompt_bank:
        sam3_args.extend(["--prompt-bank", str(prompt_bank)])
    manual_seeds = root / "objects" / "manual_seeds.json"
    if manual_seeds.exists():
        sam3_args.extend(["--manual-seeds-json", str(manual_seeds)])

    run_worker("gradyn-objects", "sam3_discover.py", sam3_args)
    mark_stage(root / ".work", "sam3", config_hash)
    run_worker(
        "gradyn-objects",
        "sam2_track.py",
        [
            *common,
            "--chunk-frames",
            str(config.get("sam2_chunk_frames", 180)),
            "--overlap",
            str(config.get("sam2_overlap_frames", 16)),
        ],
    )
    mark_stage(root / ".work", "sam2", config_hash)
    build_quality_report(root)
    build_exports(root)
    console.print(
        f"[bold green]Object tracking rebuilt:[/bold green] {root / 'objects'}"
    )


@app.command("rebuild-tracks")
def rebuild_tracks(
    output: Path = typer.Argument(..., exists=True, file_okay=False),
) -> None:
    """Rebuild only SAM 2 tracks from the existing SAM 3 discoveries."""
    from .exports import build_exports
    from .quality import build_quality_report
    from .runtime import mark_stage, run_worker

    root = output.expanduser().resolve()
    discoveries = root / ".work" / "sam3_discoveries.json"
    if not discoveries.exists():
        raise typer.BadParameter(
            "Existing SAM 3 discoveries are missing; use `gradyn rebuild-objects`."
        )
    manifest = json.loads((root / "manifest.json").read_text())
    config = manifest["config"]
    config_hash = str(manifest["config_hash"])
    run_worker(
        "gradyn-objects",
        "sam2_track.py",
        [
            "--job",
            str(root),
            "--chunk-frames",
            str(config.get("sam2_chunk_frames", 180)),
            "--overlap",
            str(config.get("sam2_overlap_frames", 16)),
        ],
    )
    mark_stage(root / ".work", "sam2", config_hash)
    build_quality_report(root)
    build_exports(root)
    console.print(
        f"[bold green]Object tracks rebuilt:[/bold green] {root / 'objects'}"
    )


@app.command("add-depth")
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


@app.command("rebuild-hands")
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


@app.command("calibrate-camera")
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


@models_app.command("download")
def models_download(
    with_qwen: bool = typer.Option(
        False,
        "--with-qwen",
        help="Also download the optional Qwen3-VL discovery checkpoint.",
    ),
) -> None:
    download_public_weights(include_qwen=with_qwen)


@models_app.command("install-mano")
def models_install_mano(
    right: Path = typer.Option(..., exists=True, dir_okay=False),
    left: Path = typer.Option(..., exists=True, dir_okay=False),
) -> None:
    install_mano(right.resolve(), left.resolve())


@models_app.command("verify")
def models_verify(
    with_qwen: bool = typer.Option(
        False,
        "--with-qwen",
        help="Require the optional Qwen3-VL discovery checkpoint.",
    ),
) -> None:
    missing = verify_models(include_qwen=with_qwen)
    if missing:
        console.print(json.dumps({"ready": False, "missing": missing}, indent=2))
        raise typer.Exit(1)
    console.print(json.dumps({"ready": True}, indent=2))
