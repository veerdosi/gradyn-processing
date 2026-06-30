from __future__ import annotations

import json
import time
from pathlib import Path

from rich.console import Console

from .config import ProcessConfig
from .exports import build_exports
from .paths import JobPaths
from .preprocess import preprocess
from .quality import build_quality_report
from .runtime import mark_stage, run_worker, stage_done
from .models import model_provenance

console = Console()


def reusable_preprocessing(config: ProcessConfig, paths: JobPaths) -> bool:
    metadata_path = paths.source / "video_metadata.json"
    timestamps_path = paths.source / "frame_timestamps.parquet"
    if not metadata_path.exists() or not timestamps_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if Path(str(metadata.get("input_video", ""))).resolve() != config.video.resolve():
        return False
    if str(metadata.get("camera_model", "")) != config.camera:
        return False
    expected = int(metadata.get("frame_count", 0))
    return expected > 0 and len(list(paths.frames.glob("*.jpg"))) == expected


def process(config: ProcessConfig, *, skip_depth: bool = False) -> None:
    started = time.monotonic()
    console.print(
        f"[bold green]Gradyn starting[/bold green] — {config.video.name} → {config.output}"
    )
    paths = JobPaths(config.output)
    paths.create()
    config_hash = config.stable_hash()
    console.print("[cyan]Preparing model provenance…[/cyan]")
    manifest = {
        "schema_version": "1.0.0",
        "pipeline_version": "0.2.0",
        "config_hash": config_hash,
        "config": config.model_dump(mode="json"),
        "claims": {
            "objects": "model-derived video object tracks",
            "hands": "camera-relative model-derived 3D hand reconstruction",
            "depth": "temporally normalized monocular relative inverse depth",
        },
        "models": model_provenance(),
    }
    (paths.root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    console.print("[green]✓ Manifest and model provenance ready[/green]")

    preprocess_done = stage_done(paths.work, "preprocess", config_hash)
    if (
        config.resume
        and not preprocess_done
        and reusable_preprocessing(config, paths)
    ):
        mark_stage(paths.work, "preprocess", config_hash)
        preprocess_done = True
        console.print(
            "[dim]↷ Reusing decoded frames from the unchanged source video[/dim]"
        )
    if not config.resume or not preprocess_done:
        console.rule("[bold]1/6 Preprocessing video")
        preprocess(config.video, config.camera, paths)
        mark_stage(paths.work, "preprocess", config_hash)
        console.print("[green]✓ Preprocessing complete[/green]")
    elif stage_done(paths.work, "preprocess", config_hash):
        console.print("[dim]↷ Reusing completed preprocessing stage[/dim]")

    common = ["--job", str(paths.root)]
    anchors_updated = False
    if not config.resume or not stage_done(paths.work, "anchors", config_hash):
        console.rule("[bold]2/6 Discovering object anchors with GroundingDINO + SAM2.1")
        args = [
            *common,
            "--target-labels-json",
            json.dumps(config.target_labels),
            "--max-auto-objects",
            str(config.max_auto_objects),
            "--exemplars-json",
            json.dumps({k: str(v) for k, v in config.exemplars.items()}),
            "--stride",
            str(config.anchor_stride),
            "--max-side",
            str(config.max_inference_side),
            "--device",
            config.anchor_device,
        ]
        if config.object_mode:
            args.append("--object-mode")
        manual_seeds = paths.objects / "manual_seeds.json"
        if manual_seeds.exists():
            args.extend(["--manual-seeds-json", str(manual_seeds)])
        run_worker("gradyn-objects", "grounded_sam2_discover.py", args)
        mark_stage(paths.work, "anchors", config_hash)
        anchors_updated = True
        console.print("[green]✓ Object anchor discovery complete[/green]")
    else:
        console.print("[dim]↷ Reusing completed object anchor discovery[/dim]")

    if anchors_updated or not config.resume or not stage_done(paths.work, "cutie", config_hash):
        console.rule("[bold]3/6 Tracking objects with Cutie")
        run_worker(
            "gradyn-inference",
            "cutie_track.py",
            ["--job", str(paths.root)],
        )
        mark_stage(paths.work, "cutie", config_hash)
        console.print("[green]✓ Object tracking complete[/green]")
    else:
        console.print("[dim]↷ Reusing completed Cutie tracking[/dim]")

    if skip_depth:
        console.print(
            "[yellow]↷ Relative depth skipped by --skip-depth; "
            "object and hand outputs will still be produced.[/yellow]"
        )
    elif not config.resume or not stage_done(paths.work, "depth", config_hash):
        console.rule("[bold]4/6 Estimating relative depth")
        run_worker(
            "gradyn-inference",
            "depth_anything_video.py",
            [
                *common,
                "--every",
                str(config.depth_every),
                "--input-size",
                str(config.depth_input_size),
            ],
        )
        mark_stage(paths.work, "depth", config_hash)
        console.print("[green]✓ Depth complete[/green]")
    else:
        console.print("[dim]↷ Reusing completed depth stage[/dim]")

    if not config.resume or not stage_done(paths.work, "hands", config_hash):
        console.rule("[bold]5/6 Reconstructing hands with WiLoR")
        run_worker("gradyn-inference", "hands_camera.py", common)
        mark_stage(paths.work, "hands", config_hash)
        console.print("[green]✓ Hand reconstruction complete[/green]")
    else:
        console.print("[dim]↷ Reusing completed hand stage[/dim]")

    console.rule("[bold]6/6 Validating and exporting")
    build_quality_report(paths.root)
    mark_stage(paths.work, "quality", config_hash)
    build_exports(paths.root)
    mark_stage(paths.work, "exports", config_hash)
    elapsed = time.monotonic() - started
    console.print(f"[bold green]✓ Gradyn complete in {elapsed / 60:.1f} minutes[/bold green]")
