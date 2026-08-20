"""Shot-level audit of a broadcast video.

Broadcast ball annotation workflow, step 1.

An edited highlights package is not a match. It is a sequence of short live
segments interleaved with replays, slow motion, close-ups, crowd cutaways and
full-screen graphics. Sampling frames uniformly from one would produce a dataset
that is mostly not the thing the detector needs to get right, so the video is
first cut into shots and each shot classified.

What is measured and what is inferred
-------------------------------------
**Cuts are measured.** A hard cut produces a large, isolated jump in the colour
histogram between consecutive frames; that is a direct observation and the
threshold is calibrated from the video's own distribution rather than fixed.

**Shot type is inferred** from evidence: how many people the detector finds, how
tall they are relative to the frame, and how many pitch keypoints are visible. A
wide tactical shot has many small people and visible pitch lines; a close-up has
one or two large people and no usable pitch geometry.

**Replay and slow motion are guessed**, and the guess is labelled as a guess.
Without broadcaster-specific transition templates there is no reliable signal,
so the heuristic here (a pitch shot whose inter-frame motion is far below the
live-play norm) is a *sampling aid only*. The annotation interface exposes
``IGNORE_REPLAY`` precisely because the human, not this module, decides.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from visionpitch.common.logging import get_logger

log = get_logger("annotation.broadcast_audit")

AUDIT_SCHEMA_VERSION = "1.0.0"


class ShotType(str, Enum):
    """What kind of footage a shot contains."""

    WIDE_PLAY = "wide_play"
    MEDIUM_PLAY = "medium_play"
    CLOSE_UP = "close_up"
    CROWD_OR_BENCH = "crowd_or_bench"
    GRAPHIC = "graphic"
    UNKNOWN = "unknown"

    @property
    def is_live_play_candidate(self) -> bool:
        """Whether frames from this shot are worth annotating for ball position.

        Close-ups and crowd shots are excluded from the *primary* sample but a
        deliberate minority is still drawn from them, because the detector must
        not hallucinate a ball in them either.
        """
        return self in (ShotType.WIDE_PLAY, ShotType.MEDIUM_PLAY)


@dataclass
class FrameFeatures:
    """Cheap per-frame measurements, computed over every frame."""

    frame_idx: int
    histogram_distance: float
    motion_px: float
    blur_variance: float
    edge_density: float
    mean_saturation: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Shot:
    """A contiguous run of frames between two cuts."""

    index: int
    start_frame: int
    end_frame: int
    fps: float
    shot_type: ShotType = ShotType.UNKNOWN
    #: evidence behind the classification
    n_people_median: float = 0.0
    person_height_median_px: float = 0.0
    pitch_keypoints_median: float = 0.0
    motion_median_px: float = 0.0
    blur_median: float = 0.0
    saturation_median: float = 0.0
    likely_slow_motion: bool = False
    classification_confidence: float = 0.0
    notes: str = ""

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame + 1

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["shot_type"] = self.shot_type.value
        payload["n_frames"] = self.n_frames
        payload["duration_s"] = round(self.duration_s, 3)
        return payload


@dataclass
class AuditResult:
    video_path: str
    content_hash: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    codec: str
    shots: list[Shot] = field(default_factory=list)
    cut_threshold: float = 0.0
    schema_version: str = AUDIT_SCHEMA_VERSION

    def by_type(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for shot in self.shots:
            entry = out.setdefault(
                shot.shot_type.value, {"n_shots": 0, "n_frames": 0, "seconds": 0.0}
            )
            entry["n_shots"] += 1
            entry["n_frames"] += shot.n_frames
            entry["seconds"] += shot.duration_s
        for entry in out.values():
            entry["seconds"] = round(entry["seconds"], 2)
            entry["share_of_video"] = (
                round(entry["n_frames"] / self.frame_count, 4) if self.frame_count else 0.0
            )
        return dict(sorted(out.items()))

    @property
    def live_play_share(self) -> float:
        live = sum(
            s.n_frames for s in self.shots if s.shot_type.is_live_play_candidate
        )
        return live / self.frame_count if self.frame_count else 0.0

    @property
    def slow_motion_share(self) -> float:
        slow = sum(s.n_frames for s in self.shots if s.likely_slow_motion)
        return slow / self.frame_count if self.frame_count else 0.0

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "video_path": self.video_path,
            "content_hash": self.content_hash,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration_s": round(self.duration_s, 3),
            "codec": self.codec,
            "n_shots": len(self.shots),
            "cut_threshold": round(self.cut_threshold, 5),
            "shot_type_breakdown": self.by_type(),
            "live_play_share": round(self.live_play_share, 4),
            "likely_slow_motion_share": round(self.slow_motion_share, 4),
            "caveats": {
                "cuts": "measured from consecutive-frame histogram distance",
                "shot_type": (
                    "inferred from person count, person height and pitch keypoint "
                    "visibility"
                ),
                "replay_and_slow_motion": (
                    "HEURISTIC ONLY -- a pitch shot whose motion is far below the "
                    "live-play norm. Used to steer sampling; the human reviewer "
                    "decides via IGNORE_REPLAY."
                ),
            },
            "shots": [s.to_dict() for s in self.shots],
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return path

    @staticmethod
    def load(path: str | Path) -> AuditResult:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        result = AuditResult(
            video_path=data["video_path"], content_hash=data["content_hash"],
            width=data["width"], height=data["height"], fps=data["fps"],
            frame_count=data["frame_count"], duration_s=data["duration_s"],
            codec=data["codec"], cut_threshold=data.get("cut_threshold", 0.0),
            schema_version=data.get("schema_version", AUDIT_SCHEMA_VERSION),
        )
        for entry in data["shots"]:
            payload = dict(entry)
            payload.pop("n_frames", None)
            payload.pop("duration_s", None)
            payload["shot_type"] = ShotType(payload["shot_type"])
            result.shots.append(Shot(**payload))
        return result


# --------------------------------------------------------------------------- #
# Pass 1: cheap per-frame features and cut detection
# --------------------------------------------------------------------------- #


def _histogram(frame: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def scan_frames(video_path: Path, progress_every: int = 2000) -> list[FrameFeatures]:
    """One decode pass computing everything cheap."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video_path}")

    features: list[FrameFeatures] = []
    previous_hist: np.ndarray | None = None
    previous_grey: np.ndarray | None = None
    frame_idx = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        hist = _histogram(frame)
        distance = (
            float(cv2.compareHist(previous_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
            if previous_hist is not None else 0.0
        )

        grey = cv2.cvtColor(
            cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY,
        )
        motion = 0.0
        if previous_grey is not None:
            shift = cv2.phaseCorrelate(
                previous_grey.astype(np.float32), grey.astype(np.float32)
            )[0]
            motion = float(np.hypot(shift[0], shift[1]) * (frame.shape[1] / 320.0))

        edges = cv2.Canny(grey, 60, 180)
        saturation = float(
            cv2.cvtColor(
                cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2HSV,
            )[:, :, 1].mean()
        )

        features.append(
            FrameFeatures(
                frame_idx=frame_idx,
                histogram_distance=distance,
                motion_px=motion,
                blur_variance=float(cv2.Laplacian(grey, cv2.CV_64F).var()),
                edge_density=float(edges.mean() / 255.0),
                mean_saturation=saturation,
            )
        )

        previous_hist, previous_grey = hist, grey
        frame_idx += 1
        if progress_every and frame_idx % progress_every == 0:
            log.info("  scanned %d frames", frame_idx)

    capture.release()
    return features


def detect_cuts(
    features: list[FrameFeatures], min_shot_frames: int = 12
) -> tuple[list[int], float]:
    """Cut frame indices, with a threshold taken from this video's own statistics.

    A fixed threshold does not transfer between broadcasters, encoders or
    resolutions. The distribution of consecutive-frame histogram distance is
    heavily concentrated near zero within a shot, so a high percentile plus a
    robust spread term separates real cuts from within-shot variation.
    """
    distances = np.array([f.histogram_distance for f in features])
    if distances.size < 3:
        return [], 0.0

    median = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median))) or 1e-6
    threshold = max(median + 8.0 * mad, float(np.percentile(distances, 99.0)))

    cuts: list[int] = []
    last = -min_shot_frames
    for index, distance in enumerate(distances):
        if distance >= threshold and index - last >= min_shot_frames:
            cuts.append(index)
            last = index
    return cuts, threshold


def build_shots(
    cuts: list[int], features: list[FrameFeatures], fps: float
) -> list[Shot]:
    boundaries = [0, *cuts, len(features)]
    shots: list[Shot] = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:], strict=False)):
        if end - start <= 0:
            continue
        window = features[start:end]
        shots.append(
            Shot(
                index=index,
                start_frame=start,
                end_frame=end - 1,
                fps=fps,
                motion_median_px=float(np.median([f.motion_px for f in window])),
                blur_median=float(np.median([f.blur_variance for f in window])),
                saturation_median=float(np.median([f.mean_saturation for f in window])),
            )
        )
    return shots
