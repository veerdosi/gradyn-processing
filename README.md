# Gradyn Processing

Quality-first egocentric RGB video processing on Apple Silicon. The pipeline produces:

1. Object masks, boxes, stable IDs, and visibility flags.
2. Camera-relative MANO hands and wrist/fingertip trajectories.
3. Depth Anything V2 Small temporally normalized relative-depth estimates.

It intentionally does **not** estimate camera trajectories, SLAM, scene geometry, or
world-space hand motion.

## Requirements

- Apple Silicon Mac with macOS 13 or newer
- Conda/Miniconda
- FFmpeg (the Conda core environment also installs it)
- Licensed MANO `MANO_RIGHT.pkl` and `MANO_LEFT.pkl`
- Sufficient free disk space for model weights and per-frame outputs

All Python dependencies are installed in Conda environments. System Python is not modified.

See [output schema](docs/OUTPUT_SCHEMA.md) and [architecture](docs/ARCHITECTURE.md).

## First-time setup

Clone the repository and run one command:

```bash
git clone https://github.com/YOUR_ORG/gradyn-processing.git
cd gradyn-processing
./setup.sh
```

The setup command:

- clones pinned versions of MLX SAM 3 Image, Cutie, WiLoR, and Depth Anything V2;
- creates four isolated Conda environments;
- installs each upstream project in its compatible environment;
- downloads and validates all weights needed by the normal explicit-object pipeline;
- downloads the configured MANO left/right files into ignored local storage, verifies their
  known hashes, and installs them for WiLoR;
- runs model verification and the test suite.

It finds Conda from the active shell, common Miniconda/Anaconda locations, or
`GRADYN_CONDA=/path/to/conda`. Re-running `./setup.sh` updates and repairs an existing
installation instead of requiring a clean machine.

MANO remains subject to its own license. The configured shared source must only be used by
people authorized to use those model files. To use an existing licensed MANO directory
instead:

```bash
./setup.sh --mano-dir /path/to/mano_v1_2/models
```

Qwen is installed by default because Gradyn uses it for semantic keyframe box proposals
before SAM 3, even when object names are supplied explicitly.

Gradyn uses `curl` with visible progress, retries, resumable partial files, and exact
SAM 3 file-size validation.

MLX SAM 3 always runs at its native 1008×1008 model resolution. Gradyn's reduced
`max_inference_side` controls source-coordinate mask handling only and does not alter the
model's fixed positional-encoding grid.

`./gradyn doctor` must be run from a normal macOS terminal rather than a headless or
virtualized session, because MLX requires direct access to the Metal device.

## Process a video

```bash
./gradyn process /path/video.mp4 \
  --camera "Camera make/model" \
  --objects "cup,screwdriver" \
  --output /path/result
```

The `./gradyn` wrapper locates Conda and runs the installed CLI in `gradyn-core`, so no
manual environment activation is required.

Qwen is disabled by default. For production jobs, specify the small set of objects that
actually matters using `--objects`, or use a reviewed prompt bank. When object names are
unknown, explicitly enable Qwen3-VL:

```bash
./gradyn process /path/video.mp4 \
  --camera "Camera make/model" \
  --discover-objects \
  --output /path/result
```

In discovery mode Qwen3-VL streams every 360th source frame one at a time and proposes a
task-agnostic physical-object vocabulary. Its prompt covers general hands-on work including
assembly, maintenance, packaging, printing, lamination, food preparation, laboratory work,
crafts, and fabrication. The pipeline then pauses for approval before loading SAM 3:

In an interactive terminal, the same `./gradyn process` command pauses, asks for a
comma-separated selection, saves it, and continues automatically. For non-interactive
jobs, approval can be supplied separately:

```bash
./gradyn select-objects result --objects "angle grinder,metal sheet,bench vise"
./gradyn process video.mp4 --camera "Camera make/model" --discover-objects --output result
```

The approval is stored in `objects/approved_prompts.json`. For each approved or explicit
object name, Qwen3-VL first proposes a tight semantic box on the same keyframes. SAM 3
then uses those boxes as geometric prompts and refines them into masks. If Qwen is
uncertain or returns no valid box for a frame, SAM 3 falls back to text prompting for
that frame. Cutie propagates each positive SAM 3 mask as a persistent physical-object
track.

The normal product workflow only requires `process`; rerunning the same command resumes
from checkpoints. The commands below are hidden maintenance tools for development and
targeted recovery, not commands an operator needs to learn.

To reduce or change the tracked object set after a completed run without rebuilding
hands or depth:

```bash
./gradyn select-objects result --objects "metal sheet,welding torch,clamp"
./gradyn rebuild-objects result
```

Qwen box proposals are enabled by default because text-only SAM 3 can confuse visually
similar task objects, such as a hammer handle versus a metal rod or a background sheet
versus the manipulated sheet. To disable the box proposer for debugging:

```bash
./gradyn process /path/video.mp4 \
  --camera "Camera make/model" \
  --objects "metal sheet,hammer" \
  --box-proposer none \
  --output /path/result
```

Small manipulated tools may still be too small or visually ambiguous for text and
open-vocabulary boxes. Initialize such an object once with a tight source-pixel box; SAM 3
uses that box as the strongest prompt and Cutie then tracks it:

```bash
./gradyn seed-object result \
  --label "welding torch" \
  --frame 1800 \
  --box "1150,650,1380,1079"

./gradyn rebuild-objects result
```

The box is stored in `objects/manual_seeds.json` as provenance. This is a one-time
initialization, not frame-by-frame annotation.

SAM 3 detections are positive localization anchors only. A SAM 3 miss is never treated as
evidence that the physical object disappeared.

Cutie tracks each physical object between positive SAM 3 mask anchors. Every bounded
interval is propagated in both directions: forward from the left anchor and backward from
the right anchor. The two masks are fused, disconnected false regions are removed, and the
bounding box is recomputed from that final source-resolution mask. This recovers object
regions lost in one direction while preventing unrelated machine fragments from expanding
the delivered box.

Each direction of each object/anchor interval is saved as an atomic checkpoint. An
interrupted job therefore restarts only the unfinished direction. Changing an anchor
invalidates only the adjacent directional checkpoints that depend on it. After the final
anchor, Cutie uses cleaned forward propagation because no future anchor is available.

SAM 3 is used for semantic localization; Cutie is used for temporal mask propagation.
SAM 3 misses never create absence spans. Frames before the first positive anchor are
`out_of_frame`; a directionally incoherent or missing propagated result is rejected rather
than fabricated. Directional IoU and retained-primary-component fraction are exported for
QA.

Cutie does not run unbounded sparse-anchor gaps. If an uncached anchor interval is longer
than the configured safe cap, Gradyn skips that propagation and marks the unsupported span
as low-confidence rather than risking MPS out-of-memory or inventing a long track. In
practice, this means ambiguous objects should be renamed, visually seeded, or excluded
instead of forced through the tracker.

To reuse existing SAM 3 detections and rebuild only Cutie tracks:

```bash
./gradyn rebuild-tracks result
```

Prompt banks bypass Qwen when a task vocabulary is already known. The current
sample-specific bank lives at
`prompt_banks/sample-metalwork.json`:

```bash
./gradyn process /path/video.mp4 \
  --camera "DJI Osmo Nano" \
  --prompt-bank prompt_banks/another-task.json \
  --output /path/result
```

A prompt bank is JSON with metadata and a non-empty `prompts` string array. It is treated
as a manually approved vocabulary and sent directly to SAM 3.

For known objects, explicit names remain the highest-quality option:

```bash
./gradyn process /path/video.mp4 \
  --camera "Camera make/model" \
  --objects "cup,screwdriver" \
  --output /path/result
```

Optional exemplar images use `OBJECT=PATH` and can be repeated:

```bash
./gradyn process /path/video.mp4 \
  --camera "Camera make/model" \
  --objects "cup,screwdriver" \
  --exemplar "cup=/path/cup.jpg" \
  --output /path/result
```

Stages and heavy model workers resume automatically. If the process is interrupted,
macOS kills it for memory pressure, or a worker fails, rerun the **same command with the
same output directory**. Do not pass `--no-resume`.

Changing only downstream options such as explicit object names or sampling strides reuses
the already-decoded source frames when the input video and camera name are unchanged.

```bash
./gradyn status /path/result
```

The status command reports both fully completed stages and partial checkpoints. Current
checkpoint granularity is:

- Qwen3-VL: every inspected frame
- MLX SAM 3: every inspected keyframe
- Cutie: every object and SAM 3 anchor interval
- Depth Anything V2: every processed depth frame
- WiLoR: every 10 source frames

Checkpoint files are written atomically under `result/.work/`; a crash cannot leave a
half-written checkpoint looking complete. Partial checkpoints are retained until the worker's
final outputs and preview video succeed, then removed; the parent immediately writes the
stage-level `*.done.json` marker and the canonical output becomes the resume source.

Preprocessing, quality reporting, and export generation are inexpensive
relative to model inference and currently restart their own stage if interrupted. Completed
upstream model stages are still reused. Pass `--no-resume` only when you deliberately want to
recompute the entire job.

Depth is optional. To produce only object tracking and camera-relative hands:

```bash
./gradyn process video.mp4 \
  --camera "DJI Osmo Nano" \
  --prompt-bank prompt_banks/sample-metalwork.json \
  --skip-depth \
  --output result
```

`--skip-depth` is a runtime switch and does not invalidate completed preprocessing or object
tracking checkpoints. A later run without the flag can add depth to the same result folder.

To add or replace relative depth in an existing result without rerunning object or hand models:

```bash
./gradyn add-depth result
```

WiLoR reconstruction runs on CPU by default. Although the model executes on MPS,
representative Apple Silicon comparisons showed materially incorrect projected geometry
without an exception. To rebuild hand outputs without rerunning objects or relative depth:

```bash
./gradyn rebuild-hands result
```

Relative depth runs at full source frame rate by default. Use `--depth-every N` or
`./gradyn add-depth result --every N` only when a lower temporal rate is acceptable.
Every record retains its exact source frame and timestamp; skipped RGB frames are
not assigned fabricated depth.

Depth Anything V2 Small uses a 756 px inference size by default for finer geometry.
Gradyn applies a bilateral edge-preserving filter, then stabilizes only motion-aligned
non-edge regions using optical flow. Scene cuts reset the temporal state, and regions
with strong motion, RGB/depth boundaries, photometric disagreement, glare, or blur are
not blindly smoothed. Use `--depth-input-size 518` for the faster legacy-quality mode.

Camera intrinsics are not required for relative depth. They remain useful for the
WiLoR camera-relative hand conversion. For highest-quality hand scale, calibrate the
exact capture mode and pass its effective focal length:

Gradyn automatically matches known camera profiles from MP4 resolution, frame rate,
aspect ratio, and the declared `--camera` name. The included standard profile matches
`DJI Osmo Nano`, 1920×1080, 30000/1001 fps, and 16:9. The default declared capture
setting is `RSWIDE1080P30FPS`: Wide FOV with RockSteady enabled. It records the advertised 143°
diagonal FOV and f/2.8 aperture. Lens/FOV mode, stabilization, dewarping, and distortion
cannot be recovered reliably from ordinary MP4 metadata. The current user-declared
capture settings are Wide FOV and RockSteady enabled; dewarping and distortion
remain unknown. The profile therefore retains WiLoR's validated image-size projection
heuristic rather than inventing distortion coefficients.

```bash
./gradyn calibrate-camera osmo-calibration.mp4 \
  --columns 9 \
  --rows 6 \
  --square-size-mm 25 \
  --output camera-calibration.json

./gradyn process video.mp4 \
  --camera "DJI Osmo Nano" \
  --focal-length-px VALUE_FROM_CALIBRATION \
  --prompt-bank prompt_banks/sample-metalwork.json \
  --output result
```

See [camera calibration](docs/CAMERA_CALIBRATION.md). Without explicit calibration,
Gradyn marks hand metric scale as estimated.

## Output

```text
result/
  manifest.json
  source/
    video_metadata.json
    frame_timestamps.parquet
  objects/
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

The COCO export contains video-aware instance annotations. The LeRobot export follows the
v3 chunk/meta layout with Gradyn fields stored as JSON payload columns. The RLDS export is a
dependency-light episode/steps Parquet interchange; convert it to TFDS TFRecord only when a
consumer specifically requires TensorFlow serialization.

## 8 GB M2 behavior

Large models run in separate processes and Conda environments in this order:

`[optional Qwen3-VL vocabulary → approval] → Qwen3-VL boxes → MLX SAM 3 → Cutie → Depth Anything V2 Small → WiLoR CPU`

Qwen3-VL vocabulary discovery is opt-in through `--discover-objects`. Qwen3-VL box
grounding is enabled by default for explicit-object and prompt-bank runs. It uses a 4-bit
MLX checkpoint, processes one resized image at a time, checkpoints each response, and is
unloaded before SAM 3 starts.

On the target 8 GB M2, the three-frame production smoke test used 2.88 GB active MLX
memory, peaked at 3.82 GB in MLX and approximately 5.3 GB process footprint, completed in
25.6 seconds, and used no swap. At the every-360-frame default, the 3,608-frame sample
requires 12 Qwen frames including the first and final frames.

Only one model family is resident at a time. Cutie processes one object and one SAM 3
anchor interval at a time, and WiLoR processes frames sequentially on CPU for numerical
correctness. MPS cache cleanup occurs between accelerated stages. Unsupported MPS
operations use PyTorch's explicitly enabled CPU fallback and are recorded in terminal logs.

Current device placement:

- FFmpeg preprocessing, validation, encoding, optical flow, filtering, and exports: CPU
- Qwen3-VL vocabulary and box proposals: MLX/Metal
- SAM 3 Image: MLX/Metal
- Cutie object tracking: PyTorch MPS when supported, with an explicit CPU mode available
- Depth Anything V2 Small inference: PyTorch MPS, with isolated CPU fallback and CPU postprocessing
- WiLoR hand detector: CPU
- WiLoR reconstruction: CPU by default because the current MPS path fails numerical geometry checks

The 8 GB M2 is the supported local quality-validation machine, not a practical single-machine
production fleet for hundreds of hour-long videos. Explicit object names remove Qwen cost but
do not remove the dominant full-frame depth, object tracking, and CPU WiLoR costs.
Quality-preserving production scaling should run independent videos in parallel and use
CUDA-backed workers for Cutie, depth, and WiLoR after CUDA/MPS numerical parity tests. Local
speed knobs such as a larger SAM 3 stride, `--depth-every 2`, lower depth input size, or
temporally subsampled hand reconstruction reduce compute by reducing validation frequency,
temporal density, or spatial detail and therefore require dataset-specific acceptance tests.

Cutie uses the official `cutie-base-mega` checkpoint at a 480 px internal short-side
resolution and tracks one object per state. Positive SAM 3 masks reset its object memory
at each anchor interval, which prevents semantic fragments and repeated background
instances from silently becoming the tracked identity.

Depth Anything V2 Small runs one frame at a time on MPS at 756 px by default. Gradyn
normalizes each prediction to relative inverse depth, applies edge-preserving filtering,
and uses optical-flow-guided temporal stabilization away from RGB/depth boundaries.
Normalization and temporal state reset at detected scene cuts. Values near 1 are nearer
and values near 0 are farther; they must not be interpreted as meters. Relative depth is
not used to rescale WiLoR.

HaWoR is intentionally excluded. The v1 hand output uses WiLoR plus temporal filtering and
short-gap interpolation, all in camera-relative coordinates.
