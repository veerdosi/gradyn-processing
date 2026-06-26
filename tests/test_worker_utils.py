import numpy as np
import pytest
from pathlib import Path
from PIL import Image

from workers.common import (
    chunk_ranges,
    decode_coco_rle,
    encode_coco_rle,
    merge_intervals,
    overlay_mask,
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
    robust_normalize,
    stabilize_relative_depth,
)
from workers.cutie_track import (
    anchor_directional_consistency_reasons,
    detect_lifecycle_boundary,
    fuse_directional_masks,
    primary_component,
    segment_cache_name,
    select_trusted_anchors,
)
from workers.grounded_sam2_discover import (
    cluster_embeddings,
    cluster_score,
    mask_passes_geometry,
    select_diverse_clusters,
)


def test_coco_rle_round_trip() -> None:
    mask = np.zeros((5, 7), dtype=bool)
    mask[1:4, 2:6] = True
    assert np.array_equal(mask, decode_coco_rle(encode_coco_rle(mask)))


def test_overlay_draws_mask_contour_and_exact_bbox_extent() -> None:
    image = Image.new("RGB", (12, 10), "black")
    mask = np.zeros((10, 12), dtype=bool)
    mask[2:7, 3:9] = True
    rendered = np.asarray(
        overlay_mask(image, mask, (255, 80, 110), "object", [3, 2, 6, 5])
    )
    assert rendered[2, 3, 0] > 200
    assert rendered[6, 8, 0] > 200
    assert rendered[7, 9].sum() == 0


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


def test_cutie_segment_cache_changes_with_anchor_or_interval() -> None:
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    anchor = {"frame_index": 0, "mask_rle": encode_coco_rle(mask)}
    first = segment_cache_name(1, anchor, 0, 90, 480, "forward", 10, 3)
    second = segment_cache_name(1, anchor, 0, 90, 480, "forward", 10, 3)
    changed = segment_cache_name(1, anchor, 0, 45, 480, "forward", 10, 3)
    reversed_name = segment_cache_name(
        1, anchor, 0, 90, 480, "backward", 10, 3
    )
    changed_memory = segment_cache_name(1, anchor, 0, 90, 480, "forward", 5, 5)
    assert first == second
    assert changed != first
    assert reversed_name != first
    assert changed_memory != first


def test_cutie_primary_component_removes_disconnected_machine_leak() -> None:
    mask = np.zeros((30, 40), dtype=bool)
    mask[10:25, 5:30] = True
    mask[2:5, 8:20] = True
    cleaned, retained = primary_component(mask)
    assert cleaned[10:25, 5:30].all()
    assert not cleaned[2:5, 8:20].any()
    assert retained == pytest.approx(375 / 411)


def test_cutie_bidirectional_union_recovers_complementary_object_halves() -> None:
    forward = np.zeros((20, 30), dtype=bool)
    backward = np.zeros((20, 30), dtype=bool)
    forward[5:15, 4:17] = True
    backward[5:15, 13:26] = True
    fused, agreement, retained = fuse_directional_masks(forward, backward)
    assert fused is not None
    assert fused[5:15, 4:26].all()
    assert 0.0 < agreement < 1.0
    assert retained == 1.0


def test_cutie_lifecycle_split_quarantines_between_instances() -> None:
    indices = list(range(20))
    left = np.zeros((20, 20), dtype=bool)
    left[4:16, 3:10] = True
    right = np.zeros((20, 20), dtype=bool)
    right[4:16, 11:18] = True
    empty = np.zeros((20, 20), dtype=bool)
    forward = {
        index: left.copy() if index < 7 else empty.copy()
        for index in indices
    }
    backward = {index: right.copy() for index in indices}
    motion = {index: (2.5 if 10 <= index <= 14 else 0.3) for index in indices}
    boundary = detect_lifecycle_boundary(
        indices,
        forward,
        backward,
        int(left.sum()),
        int(right.sum()),
        motion,
    )
    assert boundary is not None
    assert boundary["old_end_frame"] == 6
    assert boundary["new_start_frame"] == 7
    assert boundary["split_reason"] == "directional_disagreement"
    assert boundary["signals"]["forward_collapse"]
    assert boundary["signals"]["directional_disagreement"]


def test_cutie_lifecycle_motion_starts_new_instance_after_stability() -> None:
    indices = list(range(120))
    left = np.zeros((24, 24), dtype=bool)
    left[5:18, 3:11] = True
    right = np.zeros((24, 24), dtype=bool)
    right[5:18, 12:21] = True
    empty = np.zeros((24, 24), dtype=bool)
    forward = {
        index: left.copy() if index < 40 else empty.copy()
        for index in indices
    }
    backward = {
        index: right.copy() if index >= 84 else empty.copy()
        for index in indices
    }
    # Motion starts well after forward collapse. With 30 fps and the 0.8s
    # stabilization window, the new instance should begin around frame 84,
    # not at the old anchor-proximity fallback of frame 59.
    motion = {index: (2.5 if 60 <= index <= 64 else 0.3) for index in indices}
    boundary = detect_lifecycle_boundary(
        indices,
        forward,
        backward,
        int(left.sum()),
        int(right.sum()),
        motion,
        fps=30.0,
    )
    assert boundary is not None
    assert boundary["old_end_frame"] == 39
    assert boundary["new_start_frame"] == 84
    assert boundary["split_reason"] == "post_motion_backward_stability"
    assert boundary["signals"]["forward_collapse"]
    assert boundary["signals"]["independent_motion"]


def test_bad_semantic_anchor_is_rejected_by_adjacent_tracks() -> None:
    anchor = np.zeros((30, 30), dtype=bool)
    anchor[2:12, 2:12] = True
    previous = np.zeros_like(anchor)
    next_backward = np.zeros_like(anchor)
    next_backward[16:28, 16:28] = True
    assert anchor_directional_consistency_reasons(
        anchor,
        previous,
        next_backward,
        previous_anchor_area=100,
        next_anchor_area=144,
    ) == ["previous_track_collapsed_next_track_disagrees"]


def test_cutie_rejects_only_weak_severe_fragment_anchors() -> None:
    def anchor(frame: int, height: int, score: float) -> dict:
        mask = np.zeros((100, 100), dtype=bool)
        mask[:height, :80] = True
        return {
            "frame_index": frame,
            "object_id": 1,
            "label": "paper sheet",
            "score": score,
            "mask_rle": encode_coco_rle(mask),
        }

    accepted, rejected = select_trusted_anchors(
        [
            anchor(0, 80, 0.8),
            anchor(90, 55, 0.7),
            anchor(180, 8, 0.6),
            anchor(270, 60, 0.4),
        ]
    )
    assert [item["frame_index"] for item in accepted] == [0, 90]
    assert [item["frame_index"] for item in rejected] == [180, 270]


def test_cutie_keeps_confident_small_anchor_without_label_special_case() -> None:
    def anchor(frame: int, height: int, score: float, label: str) -> dict:
        mask = np.zeros((100, 100), dtype=bool)
        mask[:height, :80] = True
        return {
            "frame_index": frame,
            "object_id": 1,
            "label": label,
            "score": score,
            "mask_rle": encode_coco_rle(mask),
        }

    accepted, rejected = select_trusted_anchors(
        [
            anchor(0, 80, 0.8, "object"),
            anchor(90, 10, 0.9, "object"),
        ]
    )
    assert [item["frame_index"] for item in accepted] == [0, 90]
    assert rejected == []


def test_object_mask_geometry_rejects_background_like_masks() -> None:
    good = np.zeros((100, 100), dtype=bool)
    good[30:55, 35:65] = True
    assert mask_passes_geometry(good, np.asarray([35, 30, 65, 55], dtype=np.float32))

    huge = np.ones((100, 100), dtype=bool)
    assert not mask_passes_geometry(huge, np.asarray([0, 0, 100, 100], dtype=np.float32))

    strip = np.zeros((100, 100), dtype=bool)
    strip[45:52, 0:100] = True
    assert not mask_passes_geometry(strip, np.asarray([0, 45, 100, 52], dtype=np.float32))


def test_dinov2_cluster_links_similar_objects_across_keyframes_only() -> None:
    candidates = [
        {"frame_index": 0, "embedding": [1.0, 0.0], "area_fraction": 0.02, "aspect_ratio": 2.0, "detector_score": 0.4},
        {"frame_index": 90, "embedding": [0.98, 0.02], "area_fraction": 0.021, "aspect_ratio": 2.1, "detector_score": 0.35},
        {"frame_index": 0, "embedding": [0.97, 0.03], "area_fraction": 0.02, "aspect_ratio": 2.0, "detector_score": 0.4},
        {"frame_index": 90, "embedding": [0.0, 1.0], "area_fraction": 0.08, "aspect_ratio": 1.1, "detector_score": 0.5},
    ]
    clusters = cluster_embeddings(candidates, 0.95)
    assert sorted(len(cluster) for cluster in clusters) == [1, 1, 2]
    linked = max(clusters, key=len)
    assert {candidates[index]["frame_index"] for index in linked} == {0, 90}
    assert cluster_score([candidates[index] for index in linked], 2) > 0.7


def test_object_cluster_selection_skips_duplicate_boxes() -> None:
    clusters = [
        (0.9, [], {"box_xyxy": [10, 10, 50, 50]}),
        (0.8, [], {"box_xyxy": [12, 12, 48, 48]}),
        (0.7, [], {"box_xyxy": [80, 80, 120, 120]}),
    ]
    selected = select_diverse_clusters(clusters, 2)
    assert [item[2]["box_xyxy"] for item in selected] == [
        [10, 10, 50, 50],
        [80, 80, 120, 120],
    ]


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


def test_atomic_checkpoint_writers(tmp_path: Path) -> None:
    json_path = tmp_path / "progress.json"
    npz_path = tmp_path / "progress.npz"
    save_json({"completed": [1, 2]}, json_path)
    save_npz_atomic(npz_path, completed=np.asarray([True, False]))
    assert json_path.read_text().strip().startswith("{")
    with np.load(npz_path) as payload:
        assert payload["completed"].tolist() == [True, False]
    assert not list(tmp_path.glob("*.tmp*"))
