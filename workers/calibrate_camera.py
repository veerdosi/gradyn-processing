from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--columns", type=int, default=9)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--square-size-mm", type=float, required=True)
    parser.add_argument("--sample-every", type=int, default=10)
    args = parser.parse_args()

    import cv2

    video = Path(args.video).resolve()
    output = Path(args.output).resolve()
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f"Could not open calibration video: {video}")

    pattern = (args.columns, args.rows)
    object_template = np.zeros((args.rows * args.columns, 3), np.float32)
    object_template[:, :2] = np.mgrid[
        0 : args.columns, 0 : args.rows
    ].T.reshape(-1, 2)
    object_template *= args.square_size_mm / 1000.0
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    frame_index = 0
    sampled = 0
    image_size = None

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % max(args.sample_every, 1):
            frame_index += 1
            continue
        sampled += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])
        found, corners = cv2.findChessboardCornersSB(
            gray,
            pattern,
            flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
        )
        if found:
            object_points.append(object_template.copy())
            image_points.append(corners.astype(np.float32))
            print(
                f"Calibration board detected in frame {frame_index} "
                f"({len(image_points)} usable views)",
                flush=True,
            )
        frame_index += 1
    capture.release()

    if image_size is None or len(image_points) < 10:
        raise SystemExit(
            "Calibration needs at least 10 clear checkerboard views spanning the "
            "image. Record a longer clip with varied distance and orientation."
        )

    rms, camera_matrix, distortion, _, _ = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    result = {
        "schema_version": "1.0",
        "source_video": str(video),
        "projection_model": "opencv_pinhole_with_radial_tangential_distortion",
        "image_width": image_size[0],
        "image_height": image_size[1],
        "fx_px": fx,
        "fy_px": fy,
        "effective_focal_length_px": (fx + fy) / 2.0,
        "cx_px": float(camera_matrix[0, 2]),
        "cy_px": float(camera_matrix[1, 2]),
        "distortion_coefficients": distortion.reshape(-1).astype(float).tolist(),
        "rms_reprojection_error_px": float(rms),
        "checkerboard_inner_columns": args.columns,
        "checkerboard_inner_rows": args.rows,
        "square_size_mm": args.square_size_mm,
        "sampled_frames": sampled,
        "usable_views": len(image_points),
        "usage": (
            "Pass effective_focal_length_px to gradyn process with "
            "--focal-length-px for videos captured in this exact resolution, "
            "lens mode, stabilization mode, and dewarping mode."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
