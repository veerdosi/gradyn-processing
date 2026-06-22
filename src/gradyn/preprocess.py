from __future__ import annotations

import json
import re
import subprocess
import time
from fractions import Fraction
from math import gcd
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from rich.console import Console

from .camera_profiles import resolve_camera_profile
from .paths import JobPaths
from .runtime import StageError

console = Console()


def _probe(video: Path) -> dict:
    console.print("[cyan]Reading video metadata…[/cyan]")
    started = time.monotonic()
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
    console.print(
        f"[green]✓ Video scan complete in {time.monotonic() - started:.1f}s[/green]"
    )
    return json.loads(result.stdout)


def preprocess(video: Path, camera: str, paths: JobPaths) -> None:
    probe = _probe(video)
    stream = probe["streams"][0]
    camera_profile = resolve_camera_profile(camera, stream)
    try:
        source_fps = float(Fraction(stream.get("avg_frame_rate", "0/1")))
    except (ValueError, ZeroDivisionError):
        source_fps = 0.0
    aspect_divisor = gcd(int(stream["width"]), int(stream["height"]))
    if camera_profile:
        console.print(
            "[green]✓ Camera profile matched:[/green] "
            f"{camera_profile['profile_id']}"
        )
    else:
        console.print(
            "[yellow]No exact camera profile matched; hand projection will use "
            "the documented image-size heuristic.[/yellow]"
        )

    for old_frame in paths.frames.glob("*.jpg"):
        old_frame.unlink()
    console.print("[cyan]Decoding source-resolution frames…[/cyan]")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "info",
        "-nostats",
        "-y",
        "-i",
        str(video),
        "-map",
        "0:v:0",
        "-vf",
        "showinfo",
        "-fps_mode",
        "passthrough",
        "-q:v",
        "2",
        "-start_number",
        "0",
        "-progress",
        "pipe:1",
        "-stats_period",
        "5",
        str(paths.frames / "%08d.jpg"),
    ]
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    last_reported = -1
    timestamps: list[float] = []
    timestamp_pattern = re.compile(r"pts_time:([^\s]+)")
    assert process.stdout is not None
    for line in process.stdout:
        timestamp_match = timestamp_pattern.search(line)
        if timestamp_match:
            try:
                timestamps.append(float(timestamp_match.group(1)))
            except ValueError:
                pass
        progress_line = line.strip()
        if progress_line.startswith("frame=") and progress_line[6:].isdigit():
            frame = int(progress_line[6:])
            if frame >= last_reported + 100:
                console.print(
                    f"[dim]  decoded {frame:,} frames "
                    f"({time.monotonic() - started:.0f}s elapsed)[/dim]"
                )
                last_reported = frame
    return_code = process.wait()
    if return_code:
        raise StageError(f"FFmpeg frame extraction failed with exit code {return_code}")
    images = sorted(paths.frames.glob("*.jpg"))
    if not images:
        raise StageError("FFmpeg extracted no frames from the input video.")

    if len(timestamps) != len(images):
        console.print(
            "[yellow]Timestamp extraction count differed from decoded frames; "
            "using the source average frame rate.[/yellow]"
        )
        fps_text = stream.get("avg_frame_rate", "30/1")
        numerator, denominator = [float(x) for x in fps_text.split("/")]
        fps = numerator / denominator if denominator else 30.0
        timestamps = [i / fps for i in range(len(images))]
    if len(timestamps) > 1:
        durations: list[float | None] = [
            timestamps[index + 1] - timestamps[index]
            for index in range(len(timestamps) - 1)
        ]
        durations.append(durations[-1])
    else:
        fps_text = stream.get("avg_frame_rate", "30/1")
        numerator, denominator = [float(x) for x in fps_text.split("/")]
        fps = numerator / denominator if denominator else 30.0
        durations = [1.0 / fps] * len(images)

    with Image.open(images[0]) as first:
        output_width, output_height = first.size
    table = pa.table(
        {
            "frame_index": pa.array(range(len(images)), pa.int64()),
            "timestamp_s": pa.array(timestamps, pa.float64()),
            "duration_s": pa.array(durations, pa.float64()),
            "file": pa.array([image.name for image in images]),
        }
    )
    pq.write_table(table, paths.source / "frame_timestamps.parquet")
    metadata = {
        "input_video": str(video.resolve()),
        "camera_model": camera,
        "source_width": int(stream["width"]),
        "source_height": int(stream["height"]),
        "source_fps": source_fps,
        "source_aspect_ratio": (
            f"{int(stream['width']) // aspect_divisor}:"
            f"{int(stream['height']) // aspect_divisor}"
        ),
        "decoded_width": output_width,
        "decoded_height": output_height,
        "frame_count": len(images),
        "codec": stream.get("codec_name"),
        "pixel_format": stream.get("pix_fmt"),
        "duration_s": float(probe.get("format", {}).get("duration", timestamps[-1])),
        "rotation_applied_by_ffmpeg": True,
        "trajectory_coordinate_system": "camera-relative; camera motion remains present",
        "camera_profile": camera_profile,
    }
    (paths.source / "video_metadata.json").write_text(json.dumps(metadata, indent=2))
