from __future__ import annotations

import json
import hashlib
import sys
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


def qwen_box_stage_done(paths: JobPaths, config_hash: str) -> bool:
    proposals = paths.work / "qwen_box_proposals.json"
    if not stage_done(paths.work, "boxes", config_hash) or not proposals.exists():
        return False
    try:
        payload = json.loads(proposals.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("box_coordinate_system") == "qwen_0_1000"


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
    qwen_mode = config.discover_objects
    if qwen_mode:
        if not config.resume or not stage_done(paths.work, "qwen", config_hash):
            console.rule("[bold]1.5/6 Proposing physical objects with Qwen3-VL")
            qwen_args = [
                *common,
                "--stride",
                str(config.qwen_stride),
                "--max-candidates",
                str(config.qwen_max_candidates),
            ]
            run_worker("gradyn-vocab", "qwen_discover.py", qwen_args)
            mark_stage(paths.work, "qwen", config_hash)
            console.print("[green]✓ Qwen vocabulary discovery complete[/green]")
        else:
            console.print("[dim]↷ Reusing completed Qwen discovery[/dim]")

        discovery_path = paths.objects / "qwen_discovery.json"
        approval_path = paths.objects / "approved_prompts.json"
        discovery = json.loads(discovery_path.read_text())
        discovery_sha256 = hashlib.sha256(discovery_path.read_bytes()).hexdigest()
        approved_labels: list[str] = []
        if approval_path.exists():
            approval = json.loads(approval_path.read_text())
            if (
                approval.get("source") == "qwen3_vl"
                and approval.get("discovery_sha256") == discovery_sha256
            ):
                approved_labels = [
                    str(value) for value in approval.get("selected_labels", [])
                ]
        if not approved_labels:
            console.rule("[bold yellow]Object vocabulary approval required")
            console.print(
                "Qwen proposed these physical objects. Approve only the names "
                "you want SAM 3 to verify and Cutie to track:"
            )
            for item in discovery.get("candidates", []):
                console.print(
                    f"  - {item['label']} "
                    f"({item['keyframe_hits']} sampled frames, "
                    f"{item['hit_rate']:.1%})"
                )
            if sys.stdin.isatty():
                available = {
                    str(item["label"]).casefold(): str(item["label"])
                    for item in discovery.get("candidates", [])
                }
                while not approved_labels:
                    response = console.input(
                        "\n[bold]Enter comma-separated labels to approve:[/bold] "
                    )
                    requested = [
                        value.strip() for value in response.split(",") if value.strip()
                    ]
                    unknown = [
                        value for value in requested if value.casefold() not in available
                    ]
                    if unknown:
                        console.print(
                            "[red]Unknown labels:[/red] " + ", ".join(unknown)
                        )
                        continue
                    approved_labels = [
                        available[value.casefold()] for value in requested
                    ]
                    if not approved_labels:
                        console.print("[red]Approve at least one label.[/red]")
                temporary = approval_path.with_suffix(".json.tmp")
                temporary.write_text(
                    json.dumps(
                        {
                            "discovery_sha256": discovery_sha256,
                            "source": "qwen3_vl",
                            "selected_labels": approved_labels,
                        },
                        indent=2,
                    )
                )
                temporary.replace(approval_path)
                console.print(
                    "[green]✓ Vocabulary approval saved; continuing[/green]"
                )
            else:
                console.print(
                    "\nThis non-interactive run cannot request approval. Approve "
                    "labels and rerun:\n"
                    f'  gradyn select-objects "{paths.root}" '
                    '--objects "angle grinder,metal sheet,bench vise"\n'
                )
                raise SystemExit(3)
        candidate_prompts_path = paths.work / "approved_candidate_prompts.json"
        temporary = candidate_prompts_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(approved_labels, indent=2))
        temporary.replace(candidate_prompts_path)
        console.print(
            "[green]✓ Approved vocabulary:[/green] "
            + ", ".join(approved_labels)
        )

    boxes_updated = False
    if config.box_proposer == "qwen":
        boxes_done = qwen_box_stage_done(paths, config_hash)
        if not config.resume or not boxes_done:
            console.rule("[bold]1.75/6 Proposing object boxes with Qwen3-VL")
            box_args = [
                *common,
                "--objects-json",
                json.dumps(config.objects),
                "--stride",
                str(config.sam3_stride),
                "--max-side",
                str(config.qwen_box_max_side),
                "--max-tokens",
                str(config.qwen_box_max_tokens),
            ]
            if qwen_mode:
                box_args.extend(
                    [
                        "--candidate-prompts-json",
                        str(paths.work / "approved_candidate_prompts.json"),
                    ]
                )
            if config.prompt_bank:
                box_args.extend(["--prompt-bank", str(config.prompt_bank)])
            run_worker("gradyn-vocab", "qwen_boxes.py", box_args)
            mark_stage(paths.work, "boxes", config_hash)
            boxes_updated = True
            console.print("[green]✓ Qwen box proposals complete[/green]")
        else:
            console.print("[dim]↷ Reusing completed Qwen box proposals[/dim]")

    sam3_updated = False
    if boxes_updated or not config.resume or not stage_done(paths.work, "sam3", config_hash):
        console.rule("[bold]2/6 Discovering objects with MLX SAM 3")
        args = [
            *common,
            "--objects-json",
            json.dumps(config.objects),
            "--max-auto-objects",
            str(config.max_auto_objects),
            "--exemplars-json",
            json.dumps({k: str(v) for k, v in config.exemplars.items()}),
            "--stride",
            str(config.sam3_stride),
            "--max-side",
            str(config.max_inference_side),
        ]
        if qwen_mode:
            args.extend(
                [
                    "--candidate-prompts-json",
                    str(paths.work / "approved_candidate_prompts.json"),
                    "--candidate-source",
                    "qwen3_vl_approved",
                ]
            )
        if config.prompt_bank:
            args.extend(["--prompt-bank", str(config.prompt_bank)])
        manual_seeds = paths.objects / "manual_seeds.json"
        if manual_seeds.exists():
            args.extend(["--manual-seeds-json", str(manual_seeds)])
        box_proposals = paths.work / "qwen_box_proposals.json"
        if config.box_proposer == "qwen" and box_proposals.exists():
            args.extend(["--box-proposals-json", str(box_proposals)])
        run_worker("gradyn-objects", "sam3_discover.py", args)
        mark_stage(paths.work, "sam3", config_hash)
        sam3_updated = True
        console.print("[green]✓ Object discovery complete[/green]")
    else:
        console.print("[dim]↷ Reusing completed SAM 3 discovery[/dim]")

    if sam3_updated or not config.resume or not stage_done(paths.work, "cutie", config_hash):
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
