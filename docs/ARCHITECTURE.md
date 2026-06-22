# Architecture

```text
MP4
 ├─ FFmpeg timestamp-preserving frame extraction
 ├─ Streamed Qwen3-VL physical-object vocabulary
 │    └─ user approval before localization
 ├─ MLX SAM 3 keyframe localization and semantic verification
 │    └─ SAM 2.1 per-object bidirectional tracking
 ├─ Depth Anything V2 Small relative inverse depth
 │    └─ 756 px inference, bilateral filtering, optical-flow stabilization,
 │       glare/blur QA, and scene-cut resets
 └─ WiLoR per-frame MANO reconstruction
      └─ short-gap interpolation and temporal smoothing
```

The stages run as separate processes so only one large model family occupies unified
memory at a time.

## Conda environments

- `gradyn-core`: CLI, FFmpeg orchestration, schemas, quality reports
- `gradyn-vocab`: Qwen3-VL 4-bit MLX vocabulary discovery every 360th frame,
  plus the first and final frames
- `gradyn-objects`: MLX SAM 3 and SAM 2
- `gradyn-inference`: WiLoR and Depth Anything V2 Small

The pipeline intentionally excludes HaWoR, HaMeR, SLAM, camera trajectories, scene
reconstruction, and world-space hand motion.

## Prompt banks

For unfamiliar videos, Qwen3-VL uses a task-agnostic prompt to propose trackable
physical-object nouns from streamed frames across general hands-on activities. The user
approves names before SAM 3 spends compute localizing them. Task-specific JSON files under
`prompt_banks/` bypass Qwen, and explicit `--objects` bypasses both Qwen and the bank.

## Camera metadata

The camera make/model is stored as provenance. Object tracking and relative depth do not
require intrinsics. WiLoR retains its trained weak-perspective camera conversion and uses
a supplied calibrated focal length when available, otherwise an explicitly documented
image-size heuristic. Depth and hands remain independent outputs.
