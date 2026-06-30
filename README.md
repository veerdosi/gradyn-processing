# Gradyn Processing

Quality-first egocentric RGB video processing on Apple Silicon. The pipeline produces:

1. Object masks, boxes, stable IDs, and visibility flags.
2. Camera-relative MANO hands and wrist/fingertip trajectories.
3. Depth Anything V2 Small temporally normalized relative-depth estimates.

It intentionally does **not** estimate camera trajectories, SLAM, scene geometry, or
world-space hand motion.

See [architecture](docs/ARCHITECTURE.md), [output schema](docs/OUTPUT_SCHEMA.md), and
[camera calibration](docs/CAMERA_CALIBRATION.md).

## Requirements

- Apple Silicon Mac with macOS 13 or newer
- Conda/Miniconda
- FFmpeg
- Licensed MANO `MANO_RIGHT.pkl` and `MANO_LEFT.pkl`
- Sufficient free disk space for model weights and per-frame outputs

All Python dependencies are installed in Conda environments. System Python is not modified.

## First-Time Setup

```bash
git clone https://github.com/YOUR_ORG/gradyn-processing.git
cd gradyn-processing
./setup.sh
```

The setup command creates isolated Conda environments, clones pinned upstream model
repositories, installs runtime dependencies, downloads public weights, installs MANO
assets, verifies model files, and runs the test suite.

The active object-anchor stack is GroundingDINO base plus SAM2.1 small. DINOv2 and CLIP
are used only for automatic object discovery/label assignment when explicit object names
are not supplied. Cutie handles temporal object tracking, WiLoR handles hands, and Depth
Anything V2 Small handles relative depth.

MANO remains subject to its own license. To use an existing licensed MANO directory:

```bash
./setup.sh --mano-dir /path/to/mano_v1_2/models
```

## Process a Video

For production runs, pass the objects you actually want to track:

```bash
./gradyn process /path/video.mp4 \
  --camera "Camera make/model" \
  --objects "cup,screwdriver" \
  --anchor-stride 15 \
  --depth-every 1 \
  --output /path/result
```

`--objects` is strict object mode:

- GroundingDINO uses exactly those object names as detector prompts.
- Broad generic prompts are not used.
- CLIP and DINOv2 are not used for object selection.
- SAM2.1 creates object masks from the GroundingDINO detections.
- Cutie tracks those object anchors through the video.

If you omit `--objects`, Gradyn runs automatic object discovery: GroundingDINO proposes
broad candidate masks, SAM2.1 segments them, DINOv2 clusters physically consistent masks
across keyframes, and CLIP assigns labels when `--target-labels` is provided. Use this
mode when you want a small set of discovered physical-object tracks rather than one named
target.

```bash
./gradyn process /path/video.mp4 \
  --camera "Camera make/model" \
  --target-labels "hammer,chisel,metal block" \
  --anchor-stride 15 \
  --output /path/result
```

`--objects` and `--target-labels` are mutually exclusive.

The `./gradyn` wrapper locates Conda and runs the installed CLI in `gradyn-core`, so no
manual environment activation is required.

## Prompt Probe

Before running a full job, compare candidate object prompts on a few frames:

```bash
./gradyn probe-objects test-sample-2.MP4 \
  --prompts "paper sheet,printed document,printed paper" \
  --frames "0,114,228" \
  --output object-prompt-probes/test-sample-2
```

This runs only GroundingDINO + SAM2.1 and writes a contact sheet plus JSON summary. It
does not run Cutie, hands, depth, CLIP, or DINOv2.

## Checkpoints And Rebuilds

Stages and heavy model workers resume automatically. If a process is interrupted, rerun
the same command with the same output directory. Do not pass `--no-resume` unless you want
to recompute the whole job.

```bash
./gradyn status /path/result
```

To rebuild object anchors and Cutie tracks while preserving preprocessing, hands, and depth:

```bash
./gradyn rebuild-objects result
```

To rebuild only Cutie tracks from existing object anchors:

```bash
./gradyn rebuild-tracks result
```

To add or replace relative depth in an existing result:

```bash
./gradyn add-depth result
```

To rebuild hand outputs without rerunning objects or depth:

```bash
./gradyn rebuild-hands result
```

## Manual Object Seeds

Small or visually ambiguous objects can be initialized with one source-pixel box:

```bash
./gradyn seed-object result \
  --label "welding torch" \
  --frame 1800 \
  --box "1150,650,1380,1079"

./gradyn rebuild-objects result
```

The box is stored in `objects/manual_seeds.json` as provenance. It is a one-time
initialization, not frame-by-frame annotation.

## Depth And Hands

Depth is optional:

```bash
./gradyn process video.mp4 \
  --camera "DJI Osmo Nano" \
  --objects "cup,screwdriver" \
  --skip-depth \
  --output result
```

Relative depth runs at full source frame rate by default. Use `--depth-every N` only when
a lower temporal rate is acceptable. Depth Anything V2 Small uses a 756 px inference size
by default; `--depth-input-size 518` is faster but lower detail.

Camera intrinsics are not required for relative depth. They remain useful for WiLoR
camera-relative hand scale. For highest-quality hand scale, calibrate the exact capture
mode and pass its effective focal length:

```bash
./gradyn calibrate-camera osmo-calibration.mp4 \
  --columns 9 \
  --rows 6 \
  --square-size-mm 25 \
  --output camera-calibration.json

./gradyn process video.mp4 \
  --camera "DJI Osmo Nano" \
  --focal-length-px VALUE_FROM_CALIBRATION \
  --objects "paper sheet" \
  --output result
```

Without explicit calibration, Gradyn marks hand metric scale as estimated.

## Output

```text
result/
  manifest.json
  source/
    video_metadata.json
    frame_timestamps.parquet
  objects/
    discovery.json
    tracks.parquet
    masks.json
    overlay.mp4
  hands/
    frames.parquet
    mano_parameters.npz
    meshes.npz
    trajectories.parquet
    overlay.mp4
  depth/
    frames/*.npz
    depth_metadata.parquet
    metadata.json
    invalid_masks/*.png
    preview.mp4
  quality/
    frame_scores.parquet
    quarantined_intervals.json
    report.html
  exports/
    coco_video/annotations.json
    lerobot/
    rlds/
```

Outputs are model-derived predictions. Depth is ordinal/relative, not metric sensor depth,
and camera-relative hand trajectories contain apparent motion caused by the moving camera.

## Local Performance Notes

Large models run in separate processes and Conda environments:

```text
FFmpeg preprocessing
→ GroundingDINO + SAM2.1 object anchors
→ Cutie object tracking
→ Depth Anything V2 Small
→ WiLoR hands
→ exports and quality reports
```

Current device placement:

- FFmpeg preprocessing, validation, encoding, optical flow, filtering, and exports: CPU
- GroundingDINO + SAM2.1 anchors: PyTorch MPS when available, CPU fallback
- DINOv2/CLIP auto-discovery helpers: PyTorch MPS when used, CPU fallback
- Cutie object tracking: PyTorch MPS when available, CPU mode available
- Depth Anything V2 Small inference: PyTorch MPS, with CPU postprocessing
- WiLoR hand detector/reconstruction: CPU by default for numerical stability

Cutie uses the official `cutie-base-mega` checkpoint at a 480 px internal short-side
resolution and tracks one object per state. Positive object anchors reset its object
memory at each anchor interval.

Depth Anything V2 Small normalizes each prediction to relative inverse depth, applies
edge-preserving filtering, and uses optical-flow-guided temporal stabilization away from
RGB/depth boundaries. Values near 1 are nearer and values near 0 are farther; they must
not be interpreted as meters.
