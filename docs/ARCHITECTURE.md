# Architecture

```text
MP4
 ├─ FFmpeg timestamp-preserving frame extraction
 ├─ Streamed Qwen3-VL physical-object vocabulary
 │    └─ user approval before localization
 ├─ Qwen3-VL keyframe box proposals for requested objects
 ├─ MLX SAM 3 keyframe mask refinement and semantic verification
 │    └─ Cutie per-object mask tracking between trusted anchors
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
- `gradyn-vocab`: Qwen3-VL 4-bit MLX vocabulary discovery and keyframe box
  proposals streamed one frame at a time
- `gradyn-objects`: MLX SAM 3 localization/refinement
- `gradyn-inference`: Cutie, WiLoR, and Depth Anything V2 Small

The pipeline intentionally excludes HaWoR, HaMeR, SLAM, camera trajectories, scene
reconstruction, and world-space hand motion.

## Prompt banks

For unfamiliar videos, Qwen3-VL uses a task-agnostic prompt to propose trackable
physical-object nouns from streamed frames across general hands-on activities. The user
approves names before SAM 3 spends compute localizing them. Task-specific JSON files under
`prompt_banks/` bypass Qwen, and explicit `--objects` bypasses both Qwen and the bank.

## Object tracking

SAM 3 supplies semantic masks every 90 source frames. Low-score or severe area-collapse
fragments are recorded in `.work/cutie_rejected_anchors.json` and cannot reset the
tracker. Cutie propagates each physical object forward from the left trusted anchor and
backward from the right trusted anchor. The directional masks are unioned to recover
complementary regions, then reduced to one coherent physical component. Bounding boxes are
always derived from this final mask. Every interval direction is an atomic checkpoint
under `.work/cutie_segments/`, so interrupted work resumes without repeating completed
directions.

Cutie refuses to start a new uncached propagation interval longer than 900 frames. Such
sparse-anchor gaps need a closer semantic anchor, a clearer object name, or exclusion.

## Camera metadata

The camera make/model is stored as provenance. Object tracking and relative depth do not
require intrinsics. WiLoR retains its trained weak-perspective camera conversion and uses
a supplied calibrated focal length when available, otherwise an explicitly documented
image-size heuristic. Depth and hands remain independent outputs.
