from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pyarrow.parquet as pq
from typer.testing import CliRunner

from gradyn.camera_profiles import resolve_camera_profile
from gradyn import cli as cli_module
from gradyn.cli import _checkpoint_status
from gradyn.config import ProcessConfig
from gradyn.paths import JobPaths
from gradyn.pipeline import reusable_preprocessing
from gradyn.preprocess import preprocess
from gradyn.runtime import mark_stage

runner = CliRunner()


def test_preprocess_preserves_frame_mapping(tmp_path: Path) -> None:
    video = tmp_path / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x120:rate=10:duration=1",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    paths = JobPaths(tmp_path / "result")
    paths.create()
    preprocess(video, "Test Camera", paths)
    metadata = json.loads((paths.source / "video_metadata.json").read_text())
    timestamps = pq.read_table(paths.source / "frame_timestamps.parquet")
    assert metadata["frame_count"] == 10
    assert timestamps.num_rows == 10
    assert len(list(paths.frames.glob("*.jpg"))) == 10


def test_checkpoint_status_reports_completed_and_partial_work(
    tmp_path: Path,
) -> None:
    root = tmp_path / "result"
    work = root / ".work"
    work.mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({"config_hash": "same"}))
    mark_stage(work, "preprocess", "same")
    (work / "depth_progress.json").write_text(
        json.dumps(
            {
                "signature": {"frame_count": 10, "every": 3},
                "rows": [{"frame_index": 0}, {"frame_index": 3}],
            }
        )
    )
    report = _checkpoint_status(root)
    assert report["completed_stages"]["preprocess"]
    assert report["partial_checkpoints"]["depth"] == {
        "completed_frames": 2,
        "total_frames": 4,
        "source_stride": 3,
    }


def test_default_object_discovery_strides(
    tmp_path: Path,
) -> None:
    config = ProcessConfig(
        video=tmp_path / "video.mp4",
        output=tmp_path / "result",
        camera="Test Camera",
    )
    assert config.anchor_stride == 90
    assert config.anchor_backend == "grounding_dino_sam2_dinov2"
    assert config.anchor_device == "auto"
    assert config.depth_every == 1
    assert config.depth_input_size == 756
    assert config.depth_backend == "depth_anything_v2_small_relative"


def test_config_hash_uses_stable_json_payload(
    tmp_path: Path,
) -> None:
    config = ProcessConfig(
        video=tmp_path / "video.mp4",
        output=tmp_path / "result",
        camera="Test Camera",
        target_labels=["paper sheet"],
    )
    payload = config.model_dump(mode="json")
    payload["video"] = str(config.video.resolve())
    payload["output"] = str(config.output.resolve())
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()[:16]
    assert config.stable_hash() == expected


def test_cli_objects_are_active_tracking_labels(
    tmp_path: Path,
    monkeypatch,
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    captured: dict[str, ProcessConfig] = {}

    def fake_pipeline(config: ProcessConfig, *, skip_depth: bool = False) -> None:
        captured["config"] = config

    monkeypatch.setattr(cli_module, "run_pipeline", fake_pipeline)
    result = runner.invoke(
        cli_module.app,
        [
            "process",
            str(video),
            "--camera",
            "Test Camera",
            "--objects",
            "paper sheet",
            "--max-auto-objects",
            "8",
            "--output",
            str(tmp_path / "result"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["config"].target_labels == ["paper sheet"]
    assert captured["config"].max_auto_objects == 1


def test_cli_rejects_objects_and_target_labels_together(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    result = runner.invoke(
        cli_module.app,
        [
            "process",
            str(video),
            "--camera",
            "Test Camera",
            "--objects",
            "paper sheet",
            "--target-labels",
            "paper sheet,phone",
            "--output",
            str(tmp_path / "result"),
        ],
    )
    assert result.exit_code != 0
    assert "Use --objects or --target-labels, not both" in result.output


def test_dji_osmo_nano_profile_matches_sample_capture_mode() -> None:
    profile = resolve_camera_profile(
        "DJI Osmo Nano",
        {
            "width": 1920,
            "height": 1080,
            "avg_frame_rate": "30000/1001",
        },
    )
    assert profile is not None
    assert profile["profile_id"] == "dji-osmo-nano-rswide-1080p30"
    assert profile["capture_mode"]["camera_setting"] == "RSWIDE1080P30FPS"
    assert profile["capture_mode"]["lens_fov_mode"] == "wide"
    assert profile["capture_mode"]["stabilization_mode"] == "rocksteady_on"
    assert profile["intrinsics"]["distortion_coefficients"] is None


def test_preprocessing_can_be_reused_when_only_downstream_config_changes(
    tmp_path: Path,
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    paths = JobPaths(tmp_path / "result")
    paths.create()
    (paths.frames / "00000000.jpg").write_bytes(b"frame")
    (paths.source / "frame_timestamps.parquet").write_bytes(b"timestamps")
    (paths.source / "video_metadata.json").write_text(
        json.dumps(
            {
                "input_video": str(video.resolve()),
                "camera_model": "DJI Osmo Nano",
                "frame_count": 1,
            }
        )
    )
    config = ProcessConfig(
        video=video,
        output=paths.root,
        camera="DJI Osmo Nano",
        target_labels=["paper sheet", "laminator"],
    )
    assert reusable_preprocessing(config, paths)
