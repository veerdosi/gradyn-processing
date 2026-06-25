# Output Schema

Schema version: `1.0.0`

Every row references the original video through `frame_index` and `timestamp_s`.
All predictions are model-derived and include confidence/status information.

## Object tracking

### `objects/qwen_discovery.json`

Records per-frame Qwen3-VL responses, normalized physical-object candidates, persistence,
device/model provenance, and whether a prompt-bank fallback was used.

### `objects/discovery.json`

Records whether object classes were supplied explicitly or selected automatically. Automatic
entries include Qwen/prompt-bank provenance, keyframe hit rate, and SAM 3 confidence statistics.

### `objects/manual_seeds.json`

Optional user-supplied source-frame boxes used to initialize small or visually ambiguous
objects that text-only SAM 3 cannot localize reliably. Each record contains the canonical
label, source frame, source-pixel `box_xyxy`, and provenance.

### `objects/tracks.parquet`

One row per requested object per frame:

- `frame_index`
- `timestamp_s`
- `object_id`: stable integer within the video
- `label`: requested canonical object name
- `bbox_x`, `bbox_y`, `bbox_width`, `bbox_height`: source-resolution pixels
- `mask_area_px`
- `visibility`: `visible`, `partially_occluded`, `fully_occluded`, `out_of_frame`,
  or `rejected`
- `confidence`: `[0, 1]`
- `source`: `sam3_anchor`, `cutie_bidirectional`, `cutie_forward_tail`,
  `cutie_incoherent`, `unverified_anchor_gap`, or `untracked`
- `directional_iou`: forward/backward mask agreement for bounded Cutie intervals
- `primary_component_fraction`: fraction of the fused prediction retained as the
  coherent physical object

### `objects/masks.json`

COCO-style uncompressed RLE masks. RLE shape is source-video height and width.

## Camera-relative hands

Coordinate units are meters. Coordinates follow the image-camera frame:

- positive X: image right
- positive Y: image down
- positive Z: forward from the camera

Camera motion remains present. These are not world-space trajectories.

### `hands/mano_parameters.npz`

- `rotation_matrices`: `[2, frames, 16, 3, 3]`
- `betas`: `[2, frames, 10]`
- `translation_camera_m`: `[2, frames, 3]`
- `handedness`: `["left", "right"]`
- `state`: `observed`, `inferred`, or `rejected`

### `hands/meshes.npz`

- `vertices_camera_m`: `[2, frames, 778, 3]`
- `faces`: MANO triangle topology

### `hands/trajectories.parquet`

- camera-relative wrist position, velocity, and acceleration
- five fingertip XYZ positions in thumb-to-little-finger order
- handedness, status, confidence, frame index, and timestamp

### `hands/detector_landmarks.npz`

- `keypoints_2d`: detector wrist/finger landmarks used for independent reprojection QA
- `boxes_xyxy`: source-resolution hand detector boxes
- `handedness`

`observed` means WiLoR reconstructed that frame. `inferred` means Gradyn filled a
gap of at most three frames between observations. Longer gaps remain `rejected`.

## Depth

### `depth/frames/########.npz`

- `relative_depth`: source-resolution float16 normalized relative inverse depth
  (`0=far`, `1=near`)

### `depth/depth_metadata.parquet`

- `frame_index`
- `timestamp_s`
- `normalization_low`
- `normalization_high`
- `temporal_delta`
- `flow_valid_fraction`
- `stabilization_blend_fraction`
- `photometric_residual`
- `sharpness`
- `glare_fraction`
- `invalid_fraction`
- `confidence`
- `status`
- `units`

Depth is monocular ordinal/relative depth, not metric depth. Values cannot be compared
as meters. `depth/metadata.json` records the model, direction convention, normalization,
input size, spatial/temporal filtering policy, and metric-scale warning.

## Quality

### `quality/frame_scores.parquet`

One row per stage and frame:

- `stage`: `objects`, `hands`, or `depth`
- `frame_index`
- `timestamp_s`
- `confidence`
- `status`: `accepted` or `rejected`
- `reasons`: JSON list of machine-readable reason strings

### `quality/quarantined_intervals.json`

Consecutive rejected frames grouped into intervals with stage and reason codes.

## Dataset adapters

- `exports/coco_video/annotations.json`: video-aware COCO instance annotations.
- `exports/lerobot/`: LeRobot v3-style chunk/meta layout plus the source MP4.
- `exports/rlds/steps.parquet`: RLDS episode/steps logical fields in Parquet.

The RLDS adapter deliberately avoids adding TensorFlow to the 8 GB inference environment.
It is an interchange representation, not a TFDS-generated TFRecord directory.
