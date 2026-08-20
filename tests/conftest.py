"""Shared fixtures.

Tests are split by what they need:

* plain tests run anywhere, on CPU, with no downloads
* ``needs_models`` requires the checkpoints in ``models/``
* ``needs_clip`` requires a validation clip in ``data/raw/``

That split matters: the fast tests must stay runnable in CI without a 200 MB
model download or a GPU, otherwise they stop being run.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from visionpitch.common.config import Config, load_config
from visionpitch.common.types import BBox, Detection, ObjectClass, Track, TrackObservation
from visionpitch.pitch.geometry import PitchConfiguration

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def config() -> Config:
    return load_config(config_root=REPO_ROOT)


@pytest.fixture
def pitch() -> PitchConfiguration:
    return PitchConfiguration()


@pytest.fixture(scope="session")
def models_available() -> bool:
    return all(
        (REPO_ROOT / "models" / name).exists()
        for name in (
            "yolo-football-player-detection.pt",
            "yolo-football-ball-detection.pt",
            "yolo-football-pitch-detection.pt",
        )
    )


@pytest.fixture(scope="session")
def validation_clip() -> Path | None:
    clip = REPO_ROOT / "data" / "raw" / "nz_canada_u17.mp4"
    return clip if clip.exists() else None


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory) -> Path:
    """A tiny synthetic video, so ingestion tests need no download.

    Deliberately generated with ffmpeg rather than OpenCV's writer: this is the
    decode path we care about testing, and generating it with the same library
    that reads it would hide container-level problems.
    """
    out_dir = tmp_path_factory.mktemp("video")
    path = out_dir / "synthetic.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )
    return path


# --------------------------------------------------------------------------- #
# Synthetic geometry
# --------------------------------------------------------------------------- #


@pytest.fixture
def synthetic_homography(pitch: PitchConfiguration) -> np.ndarray:
    """A plausible broadcast-camera homography, image (1280x720) -> pitch metres.

    Built by projecting four known pitch points to four plausible image points,
    so tests can compute exact expected answers instead of asserting on
    tolerances pulled out of the air.
    """
    import cv2

    pitch_points = np.array(
        [[20.0, 5.0], [85.0, 5.0], [85.0, 63.0], [20.0, 63.0]], dtype=np.float64
    )
    image_points = np.array(
        [[180.0, 690.0], [1120.0, 690.0], [880.0, 300.0], [400.0, 300.0]], dtype=np.float64
    )
    H, _ = cv2.findHomography(image_points, pitch_points, method=0)
    return H


@pytest.fixture
def sample_tracks() -> dict[int, Track]:
    """Two straight-line tracks over 20 frames, for tracking-metric tests."""
    tracks: dict[int, Track] = {}
    for track_id, x0 in ((1, 100.0), (2, 500.0)):
        observations = [
            TrackObservation(
                frame_idx=f,
                timestamp_s=f / 25.0,
                bbox=BBox(x0 + 4 * f, 300.0, x0 + 4 * f + 40, 400.0),
                det_confidence=0.9,
                track_confidence=0.9,
                interpolated=False,
            )
            for f in range(20)
        ]
        tracks[track_id] = Track(
            track_id=track_id, object_class=ObjectClass.PLAYER, observations=observations
        )
    return tracks


@pytest.fixture
def sample_detections() -> dict[int, list[Detection]]:
    detections: dict[int, list[Detection]] = {}
    for f in range(20):
        detections[f] = [
            Detection(f, ObjectClass.PLAYER, BBox(100 + 4 * f, 300, 140 + 4 * f, 400), 0.9),
            Detection(f, ObjectClass.PLAYER, BBox(500 + 4 * f, 300, 540 + 4 * f, 400), 0.85),
        ]
    return detections
