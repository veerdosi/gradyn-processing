import numpy as np
import pytest
from pathlib import Path

from workers.common import (
    SAM3_NATIVE_RESOLUTION,
    chunk_ranges,
    decode_coco_rle,
    encode_coco_rle,
    load_prompt_bank,
    merge_intervals,
    save_json,
    save_npz_atomic,
)
from workers.hands_camera import (
    select_temporal_detection,
    suppress_short_hand_runs,
    projected_hand_is_consistent,
    projected_landmarks_are_consistent,
    temporal_derivatives,
)
from workers.depth_anything_video import (
    edge_preserving_filter,
    robust_normalize,
    stabilize_relative_depth,
)
from workers.sam2_track import (
    choose_bidirectional_mask,
    confirmed_absence_intervals,
    frame_is_absent,
)
from workers.sam3_discover import select_auto_labels
from workers.qwen_discover import (
    DISCOVERY_PROMPT,
    aggregate_objects,
    normalize_object_name,
    parse_object_response,
)


def test_coco_rle_round_trip() -> None:
    mask = np.zeros((5, 7), dtype=bool)
    mask[1:4, 2:6] = True
    assert np.array_equal(mask, decode_coco_rle(encode_coco_rle(mask)))


def test_quarantine_interval_merge() -> None:
    assert merge_intervals([(2, ["a"]), (3, ["b"]), (7, ["c"])]) == [
        {"start_frame": 2, "end_frame": 3, "reasons": ["a", "b"]},
        {"start_frame": 7, "end_frame": 7, "reasons": ["c"]},
    ]


def test_chunk_ranges_overlap_without_loading_whole_video() -> None:
    assert chunk_ranges(500, 180, 16) == [
        (0, 180),
        (164, 344),
        (328, 500),
    ]


def test_sample_prompt_bank_is_valid() -> None:
    root = Path(__file__).resolve().parents[1]
    bank = load_prompt_bank(root / "prompt_banks" / "sample-metalwork.json")
    assert bank["name"] == "sample-metalwork"
    assert "welding torch" in bank["prompts"]


def test_sam3_uses_native_position_encoding_resolution() -> None:
    assert SAM3_NATIVE_RESOLUTION == 1008


def test_hand_projection_must_overlap_detector_box() -> None:
    projected = np.asarray([[100, 100], [120, 125], [145, 150]], dtype=np.float32)
    assert projected_hand_is_consistent(
        projected, np.asarray([80, 80, 170, 180]), 1920, 1080
    )
    assert not projected_hand_is_consistent(
        projected + 500, np.asarray([80, 80, 170, 180]), 1920, 1080
    )


def test_hand_derivatives_do_not_bridge_missing_runs() -> None:
    values = np.asarray(
        [[0, 0, 0], [1, 0, 0], [np.nan, np.nan, np.nan], [10, 0, 0], [11, 0, 0]],
        dtype=np.float32,
    )
    timestamps = np.arange(5, dtype=np.float64)
    velocity, _ = temporal_derivatives(
        values, timestamps, np.asarray([True, True, False, True, True])
    )
    assert np.isnan(velocity[2]).all()
    assert np.allclose(velocity[[0, 1, 3, 4], 0], 1.0)


def test_bidirectional_tracking_uses_nonempty_reverse_mask() -> None:
    empty = np.zeros((8, 8), dtype=bool)
    reverse = empty.copy()
    reverse[2:6, 2:6] = True
    chosen, confidence = choose_bidirectional_mask(
        empty, reverse, frame_index=20, anchor_frames=[0, 90]
    )
    assert np.array_equal(chosen, reverse)
    assert confidence == 0.75


def test_repeated_negative_anchors_create_absence_barrier() -> None:
    discoveries = [
        {"frame_index": 630, "found": True},
        {"frame_index": 720, "found": False},
        {"frame_index": 810, "found": False},
        {"frame_index": 900, "found": False},
        {"frame_index": 990, "found": False},
        {"frame_index": 1080, "found": False},
        {"frame_index": 1170, "found": True},
    ]
    intervals = confirmed_absence_intervals(discoveries)
    assert intervals == [(675, 1125)]
    assert not frame_is_absent(674, intervals)
    assert frame_is_absent(900, intervals)
    assert not frame_is_absent(1126, intervals)


def test_single_sam3_miss_does_not_force_absence() -> None:
    discoveries = [
        {"frame_index": 0, "found": True},
        {"frame_index": 90, "found": False},
        {"frame_index": 180, "found": True},
    ]
    assert confirmed_absence_intervals(discoveries) == []


def test_automatic_labels_reject_weak_semantic_matches() -> None:
    discoveries = [
        {
            "frame_index": frame,
            "label": "wrong tool label",
            "found": True,
            "score": 0.38,
        }
        for frame in [0, 90, 180]
    ]
    selected, summary = select_auto_labels(discoveries, 4, 8)
    assert selected == []
    assert summary == []


def test_approved_overlapping_objects_are_not_deduplicated() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[4:16, 4:16] = True
    discoveries = []
    for label in ("welding torch", "metal sheet"):
        for frame_index in (0, 60, 120):
            discoveries.append(
                {
                    "frame_index": frame_index,
                    "label": label,
                    "found": True,
                    "score": 0.9,
                    "mask_rle": encode_coco_rle(mask),
                }
            )
    _, selected = select_auto_labels(discoveries, 3, 4)
    assert {item["label"] for item in selected} == {
        "welding torch",
        "metal sheet",
    }


def test_temporal_hand_selection_prefers_continuous_candidate() -> None:
    keypoints = np.zeros((21, 3), dtype=np.float32)
    previous = np.asarray([100, 100, 200, 220], dtype=np.float32)
    continuous = (
        np.asarray([105, 102, 205, 222], dtype=np.float32),
        0.72,
        keypoints,
    )
    distant = (
        np.asarray([1400, 700, 1550, 950], dtype=np.float32),
        0.82,
        keypoints,
    )
    chosen = select_temporal_detection(
        [continuous, distant], previous, 1920, 1080
    )
    assert np.array_equal(chosen[0], continuous[0])


def test_short_low_confidence_hand_track_is_suppressed() -> None:
    state = np.asarray(
        ["rejected", *["observed"] * 11, "rejected", *["observed"] * 12]
    )
    confidence = np.asarray(
        [0.0, *([0.72] * 11), 0.0, *([0.72] * 12)],
        dtype=np.float32,
    )
    suppress_short_hand_runs(state, confidence)
    assert set(state[1:12]) == {"rejected"}
    assert set(state[13:]) == {"observed"}


def test_relative_depth_normalization_is_bounded_and_stabilized() -> None:
    raw = np.linspace(1, 10, 100, dtype=np.float32).reshape(10, 10)
    normalized, low, high = robust_normalize(raw, None, None)
    assert normalized.dtype == np.float32
    assert float(normalized.min()) == 0.0
    assert float(normalized.max()) == 1.0

    _, next_low, next_high = robust_normalize(raw * 2, low, high)
    current_low, current_high = np.percentile(raw * 2, [2, 98])
    assert low < next_low < current_low
    assert high < next_high < current_high


def test_depth_temporal_stabilization_preserves_edges() -> None:
    pytest.importorskip("cv2")
    depth = np.zeros((120, 160), dtype=np.float32)
    depth[:, 80:] = 1.0
    gray = np.zeros((120, 160), dtype=np.uint8)
    gray[:, 80:] = 255
    previous = np.zeros((216, 384), dtype=np.float32)
    previous[:, 192:] = 0.9
    previous_gray = np.zeros((216, 384), dtype=np.uint8)
    previous_gray[:, 192:] = 255
    stabilized, metrics, _, _ = stabilize_relative_depth(
        depth,
        gray,
        previous,
        previous_gray,
    )
    assert stabilized[:, 75].mean() < 0.1
    assert stabilized[:, 85].mean() > 0.9
    assert metrics["flow_valid_fraction"] > 0


def test_depth_edge_filter_remains_bounded() -> None:
    pytest.importorskip("cv2")
    rng = np.random.default_rng(3)
    depth = rng.random((64, 96), dtype=np.float32)
    filtered = edge_preserving_filter(depth)
    assert filtered.dtype == np.float32
    assert 0 <= float(filtered.min()) <= float(filtered.max()) <= 1


def test_projected_landmark_gate_rejects_displaced_mesh() -> None:
    detector = np.column_stack(
        [
            np.linspace(100, 200, 21),
            np.linspace(120, 260, 21),
            np.ones(21),
        ]
    )
    box = np.asarray([80, 100, 220, 280], dtype=np.float32)
    assert projected_landmarks_are_consistent(detector[:, :2], detector, box)
    assert not projected_landmarks_are_consistent(
        detector[:, :2] + np.asarray([0, 150]),
        detector,
        box,
    )


def test_qwen_object_parsing_and_persistence_filtering() -> None:
    assert parse_object_response(
        '```json\n{"objects":["Welding Gun","spark","Metal Sheet","Black Cable","Clamps"]}\n```'
    ) == ["welding torch", "metal sheet", "cable", "clamp"]
    assert normalize_object_name("repair") is None
    assert normalize_object_name("yellow protective gear") is None
    ranked = aggregate_objects(
        {
            0: ["welding gun", "spark", "clamp"],
            90: ["welder", "clamp"],
            180: ["metal sheet"],
        },
        max_candidates=8,
        minimum_hits=2,
    )
    assert [item["label"] for item in ranked] == ["welding torch", "clamp"]


def test_qwen_candidates_collapse_color_and_generic_duplicates() -> None:
    ranked = aggregate_objects(
        {
            0: ["white machine", "welding machine", "pink container"],
            30: ["blue machine", "welding machine", "container"],
        },
        max_candidates=8,
        minimum_hits=1,
    )
    assert {item["label"] for item in ranked} == {
        "container",
        "welding machine",
    }


def test_qwen_prompt_is_task_agnostic_and_supports_lamination() -> None:
    prompt = DISCOVERY_PROMPT.casefold()
    assert "do not assume a particular" in prompt
    assert "industry or task" in prompt
    assert "lamination" in prompt
    assert "paper sheet" in prompt
    assert "industrial-work frame" not in prompt


def test_qwen_normalizes_lamination_objects() -> None:
    assert parse_object_response(
        '{"objects":["Laminating Machine","Printed Paper","Plastic Film","Rollers"]}'
    ) == ["laminator", "printed page", "laminating film", "roller"]


def test_atomic_checkpoint_writers(tmp_path: Path) -> None:
    json_path = tmp_path / "progress.json"
    npz_path = tmp_path / "progress.npz"
    save_json({"completed": [1, 2]}, json_path)
    save_npz_atomic(npz_path, completed=np.asarray([True, False]))
    assert json_path.read_text().strip().startswith("{")
    with np.load(npz_path) as payload:
        assert payload["completed"].tolist() == [True, False]
    assert not list(tmp_path.glob("*.tmp*"))
