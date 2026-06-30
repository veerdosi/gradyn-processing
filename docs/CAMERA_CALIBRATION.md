# Camera calibration

The camera name and DJI's advertised 143° field of view are not enough to
recover the effective intrinsics of delivered video. Stabilization, dewarping,
aspect-ratio cropping, and resolution modes alter the effective focal length.

Calibration is optional and affects only the scale/projection quality of the WiLoR hand
output. Depth Anything V2 produces relative depth and does not use camera intrinsics.

## Automatic standard profile

Gradyn includes `camera_profiles/dji-osmo-nano-1080p30-16x9.json`. It is selected
automatically when `--camera "DJI Osmo Nano"` is paired with 1920×1080,
30000/1001-fps video. Its default declared camera setting is `RSWIDE1080P30FPS`:
Wide FOV with RockSteady enabled. Resolution, frame rate, and aspect ratio are read
directly from the MP4.

The profile stores DJI's advertised 143° diagonal FOV and f/2.8 aperture. These values
do not determine a unique pinhole focal length or distortion model: the same advertised
FOV can be delivered as raw fisheye, dewarped, cropped, or stabilized imagery. For
provenance, the profile records both rectilinear and equidistant nominal focal estimates,
but neither is used to alter WiLoR. The current declared capture mode is Wide FOV
with RockSteady enabled. Distortion coefficients remain unset.

Record a checkerboard clip using the exact Osmo Nano resolution, field-of-view
mode, stabilization mode, orientation, and dewarping settings used for dataset
collection. Move the board across the full image at varied angles and distances.
Use at least 20 clear views.

Run calibration inside the installed Conda workflow:

```bash
conda activate gradyn-core

./gradyn calibrate-camera osmo-calibration.mp4 \
  --columns 9 \
  --rows 6 \
  --square-size-mm 25 \
  --output camera-calibration.json
```

Then read `effective_focal_length_px` from the generated JSON and process videos
from the same capture mode with:

```bash
./gradyn process video.mp4 \
  --camera "DJI Osmo Nano" \
  --focal-length-px VALUE_FROM_CALIBRATION \
  --objects "paper sheet" \
  --output result
```

Recalibrate whenever resolution, stabilization, lens/FOV mode, dewarping,
orientation, or digital crop changes. The calibration JSON also records
principal point, distortion coefficients, and reprojection error for provenance.
