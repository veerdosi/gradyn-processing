# Architecture

```text
MP4
 ├─ FFmpeg timestamp-preserving frame extraction
 ├─ GroundingDINO keyframe object detection
 │    ├─ explicit --objects mode: requested names only
 │    └─ auto-discovery mode: broad candidate prompts
 ├─ SAM2.1 keyframe mask generation
 │    ├─ explicit --objects mode: direct object anchors
 │    └─ auto-discovery mode: DINOv2 clustering + optional CLIP label assignment
 ├─ Cutie per-object mask tracking between trusted anchors
 ├─ Depth Anything V2 Small relative inverse depth
 │    └─ 756 px inference, bilateral filtering, optical-flow stabilization,
 │       glare/blur QA, and scene-cut resets
 └─ WiLoR per-frame MANO reconstruction
      └─ short-gap interpolation and temporal smoothing
```

The stages run as separate processes so only one large model family occupies unified
memory at a time.

## Conda Environments

- `gradyn-core`: CLI, FFmpeg orchestration, schemas, quality reports, exports
- `gradyn-objects`: GroundingDINO, SAM2.1, DINOv2, and CLIP anchor discovery
- `gradyn-inference`: Cutie, WiLoR, and Depth Anything V2 Small

Some older model assets and environments may remain on disk for compatibility with
existing installations, but the active object-anchor path is GroundingDINO + SAM2.1.

The pipeline intentionally excludes HaWoR, HaMeR, SLAM, camera trajectories, scene
reconstruction, and world-space hand motion.

## Object Anchors

In explicit object mode, `--objects` is treated as the source of truth. GroundingDINO uses
only the requested object names as detector prompts. Broad prompts, CLIP label assignment,
and DINOv2 clustering are not used. SAM2.1 produces masks from the object detections, and
those masks become positive anchors for Cutie.

In auto-discovery mode, GroundingDINO uses a broad prompt bank to propose candidate
regions. SAM2.1 turns those regions into masks. DINOv2 embeddings link visually consistent
physical candidates across keyframes, and CLIP can rank/assign labels when
`--target-labels` is supplied.

`gradyn probe-objects` runs the GroundingDINO + SAM2.1 anchor stage on a handful of frames
without Cutie, depth, hands, DINOv2, or CLIP. It is meant for prompt selection before a
full run.

## Object Tracking

SAM2.1 anchors are positive localization anchors only. A miss is never treated as evidence
that the physical object disappeared.

Cutie propagates each physical object forward from the left trusted anchor and backward
from the right trusted anchor. The directional masks are unioned to recover complementary
regions, then reduced to one coherent physical component. Bounding boxes are always
derived from this final mask. Every interval direction is an atomic checkpoint under
`.work/cutie_segments/`, so interrupted work resumes without repeating completed
directions.

Cutie refuses to start a new uncached propagation interval longer than the configured safe
cap. Such sparse-anchor gaps need closer anchors, a clearer object name, or exclusion.

## Camera Metadata

The camera make/model is stored as provenance. Object tracking and relative depth do not
require intrinsics. WiLoR retains its trained weak-perspective camera conversion and uses
a supplied calibrated focal length when available, otherwise an explicitly documented
image-size heuristic. Depth and hands remain independent outputs.
