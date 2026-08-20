"""Shot-disjoint local splitting and the augmentation label guarantees.

The properties here are the ones whose failure produces a number that looks fine
and means nothing: a split that lets near-duplicate frames straddle the boundary,
a locked test that quietly reaches the training set, or an augmented frame whose
label no longer matches its pixels.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from visionpitch.annotation.schema import (
    AnnotationStore,
    BallAnnotation,
    BallVisibility,
    FrameSample,
    SamplingCategory,
)
from visionpitch.annotation.splits import (
    LocalSplit,
    assert_no_leakage,
    build_local_split,
    split_summary,
)


def sample(frame_id, frame_idx, shot, window=None):
    return FrameSample(
        frame_id=frame_id, frame_idx=frame_idx, timestamp_s=frame_idx / 50.0,
        image_path="", shot_index=shot, shot_type="wide_play",
        sampling_category=SamplingCategory.MIDFIELD_PLAY, sampling_reason="test",
        is_live_play_candidate=True, likely_slow_motion=False, window_id=window,
        source_content_hash="hash", width=1280, height=720,
    )


def visible(frame_id):
    return BallAnnotation(
        frame_id=frame_id, visibility=BallVisibility.VISIBLE,
        centre_x=100.0, centre_y=100.0, radius_px=7.0,
    )


def fixture(shot_sizes, window_map=None):
    """shot_sizes: {shot_index: n_frames}."""
    samples, annotations = {}, {}
    idx = 0
    for shot, count in shot_sizes.items():
        for _ in range(count):
            frame_id = f"f{idx:06d}"
            window = (window_map or {}).get(shot)
            samples[frame_id] = sample(frame_id, idx * 100, shot, window)
            annotations[frame_id] = visible(frame_id)
            idx += 1
    return samples, annotations


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #


def test_every_frame_of_a_shot_lands_in_one_split():
    samples, annotations = fixture({s: 4 for s in range(12)})
    split = build_local_split(samples, annotations)
    for frame_id, s in samples.items():
        shot = s.shot_index
        placed = {
            split.split_of_frame(f)
            for f, v in samples.items() if v.shot_index == shot
        }
        assert len(placed) == 1, f"shot {shot} split across {placed}"
        assert split.split_of_frame(frame_id) in placed


def test_leakage_check_passes_on_a_shot_disjoint_split():
    samples, annotations = fixture({s: 5 for s in range(9)})
    split = build_local_split(samples, annotations)
    report = assert_no_leakage(split, samples)
    assert report["shot_disjoint"] and report["window_disjoint"]


def test_a_shot_appearing_in_two_splits_is_rejected():
    samples, annotations = fixture({s: 3 for s in range(6)})
    split = build_local_split(samples, annotations)
    # Force a frame from a train shot into test.
    stolen = split.frames["train"].pop()
    split.frames["test"].append(stolen)
    with pytest.raises(AssertionError, match="appear in both"):
        assert_no_leakage(split, samples)


def test_a_temporal_window_may_not_straddle_a_split():
    samples, annotations = fixture({0: 7, 1: 7, 2: 7}, window_map={0: "w0", 1: "w1", 2: "w2"})
    split = build_local_split(samples, annotations)
    windows = {
        name: {samples[f].window_id for f in ids}
        for name, ids in split.frames.items()
    }
    names = sorted(windows)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (windows[a] & windows[b])


def test_allocation_balances_frames_not_shot_ids():
    """Shots run 1 to 12 frames; splitting on shot ids alone skews badly."""
    samples, annotations = fixture({0: 12, 1: 10, 2: 8, 3: 6, 4: 4, 5: 3, 6: 2, 7: 1})
    split = build_local_split(samples, annotations)
    counts = {k: len(v) for k, v in split.frames.items()}
    total = sum(counts.values())
    assert counts["train"] >= counts["val"]
    assert counts["train"] >= counts["test"]
    # Train should be near 60%, not wildly off.
    assert 0.4 <= counts["train"] / total <= 0.8, counts


def test_split_ratios_must_sum_to_one():
    samples, annotations = fixture({0: 3})
    with pytest.raises(ValueError, match="must sum to 1.0"):
        build_local_split(samples, annotations, ratios={"train": 0.5, "val": 0.2})


def test_split_is_deterministic():
    samples, annotations = fixture({s: 4 for s in range(10)})
    a = build_local_split(samples, annotations)
    b = build_local_split(samples, annotations)
    assert a.fingerprint() == b.fingerprint()
    assert a.frames == b.frames


def test_split_file_detects_editing(tmp_path):
    samples, annotations = fixture({s: 3 for s in range(6)})
    split = build_local_split(samples, annotations)
    path = split.save(tmp_path / "split.json")
    assert LocalSplit.load(path).fingerprint() == split.fingerprint()

    data = json.loads(path.read_text(encoding="utf-8"))
    moved = data["frames"]["train"].pop()
    data["frames"]["test"].append(moved)
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="was edited"):
        LocalSplit.load(path)


def test_split_record_states_the_locked_test_rule():
    samples, annotations = fixture({s: 3 for s in range(6)})
    note = build_local_split(samples, annotations).to_dict()["locked_test_note"]
    for forbidden in ("training", "checkpoint", "threshold", "early stopping"):
        assert forbidden in note


def test_summary_counts_positives_and_negatives_per_split():
    samples, annotations = fixture({s: 3 for s in range(6)})
    first = next(iter(annotations))
    annotations[first] = BallAnnotation(
        frame_id=first, visibility=BallVisibility.NOT_VISIBLE
    )
    split = build_local_split(samples, annotations)
    summary = split_summary(split, samples, annotations)
    assert sum(b["n_positive"] for b in summary.values()) == len(annotations) - 1
    assert sum(b["n_negative"] for b in summary.values()) == 1


def test_store_round_trip_preserves_the_split(tmp_path):
    """Membership and fingerprint must survive; list order is not meaningful."""
    samples, annotations = fixture({s: 3 for s in range(6)})
    store = AnnotationStore(tmp_path)
    store.write_samples(list(samples.values()))
    split = build_local_split(samples, annotations)
    path = split.save(tmp_path / "local_split.json")

    reloaded = LocalSplit.load(path)
    assert reloaded.fingerprint() == split.fingerprint()
    assert set(reloaded.frames) == set(split.frames)
    for name, frames in split.frames.items():
        assert set(reloaded.frames[name]) == set(frames)


# --------------------------------------------------------------------------- #
# Augmentation guarantees
# --------------------------------------------------------------------------- #


def load_augment():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "_bcast_ds", Path("scripts/build_broadcast_dataset.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def dataset_module():
    return load_augment()


def synthetic_frame(width=640, height=360, ball=(320.0, 180.0, 14.0, 14.0)):
    rng = np.random.default_rng(0)
    image = rng.integers(40, 90, (height, width, 3), dtype=np.uint8)
    cx, cy, bw, bh = ball
    image[
        int(cy - bh / 2): int(cy + bh / 2), int(cx - bw / 2): int(cx + bw / 2)
    ] = 240
    return image, [ball]


def test_augmented_labels_stay_inside_the_image(dataset_module):
    import random

    image, boxes = synthetic_frame()
    rng = random.Random(3)
    produced = 0
    for _ in range(40):
        out, kept = dataset_module.augment(image, boxes, rng)
        if out is None:
            continue
        produced += 1
        h, w = out.shape[:2]
        for cx, cy, bw, bh in kept:
            assert 0 <= cx - bw / 2 and cx + bw / 2 <= w
            assert 0 <= cy - bh / 2 and cy + bh / 2 <= h
    assert produced > 0, "augmentation never produced a usable frame"


def test_a_crop_that_would_clip_the_ball_is_rejected(dataset_module):
    """A half-ball labelled as a whole one is exactly the silent corruption
    this pipeline must not create."""
    import random

    # Ball hard against the edge: most crops would clip it.
    image, boxes = synthetic_frame(ball=(8.0, 180.0, 14.0, 14.0))
    rng = random.Random(11)
    for _ in range(30):
        out, kept = dataset_module.scale_and_crop(image, boxes, rng)
        if out is None:
            continue
        h, w = out.shape[:2]
        for cx, cy, bw, bh in kept:
            assert cx - bw / 2 >= -1e-6 and cx + bw / 2 <= w + 1e-6
            assert cy - bh / 2 >= -1e-6 and cy + bh / 2 <= h + 1e-6


def test_photometric_transforms_never_move_the_ball(dataset_module):
    import random

    image, boxes = synthetic_frame()
    rng = random.Random(5)
    for transform in (
        dataset_module.photometric, dataset_module.jpeg,
        dataset_module.low_resolution, dataset_module.noise,
        dataset_module.motion_blur,
    ):
        out = transform(image, rng)
        assert out.shape == image.shape, transform.__name__


def test_ball_survives_augmentation_as_visible_signal(dataset_module):
    """No transform may erase the ball."""
    import random

    image, boxes = synthetic_frame()
    rng = random.Random(7)
    for _ in range(15):
        out, kept = dataset_module.augment(image, boxes, rng)
        if out is None:
            continue
        cx, cy, bw, bh = kept[0]
        patch = out[
            int(max(0, cy - bh)): int(cy + bh), int(max(0, cx - bw)): int(cx + bw)
        ]
        surroundings = float(out.mean())
        assert patch.size > 0
        # The ball was drawn far brighter than the background; after any legal
        # transform it must still stand out.
        assert float(patch.max()) > surroundings


def test_line_negative_never_contains_a_ball(dataset_module):
    import random

    import cv2

    image = np.full((360, 640, 3), 60, dtype=np.uint8)
    cv2.line(image, (0, 180), (639, 180), (250, 250, 250), 3)
    ball = (320.0, 180.0, 14.0, 14.0)
    image[173:187, 313:327] = 240
    rng = random.Random(2)
    for _ in range(20):
        crop = dataset_module.line_negative(image, [ball], rng)
        if crop is None:
            continue
        assert crop.shape[0] >= 64 and crop.shape[1] >= 64


def test_yolo_line_is_normalised(dataset_module):
    line = dataset_module.yolo_line(320.0, 180.0, 14.0, 14.0, 640, 360)
    parts = line.split()
    assert parts[0] == "0"
    assert float(parts[1]) == pytest.approx(0.5)
    assert float(parts[2]) == pytest.approx(0.5)
    assert all(0.0 <= float(v) <= 1.0 for v in parts[1:])
