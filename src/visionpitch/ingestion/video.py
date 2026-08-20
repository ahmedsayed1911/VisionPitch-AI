"""Video decoding and frame sampling.

Everything downstream addresses frames by their **absolute source frame index**,
never by a sequential counter. That distinction matters: with ``frame_stride=3``
the 10th processed frame is source frame 27, and an event written against
sequential index 10 would be 0.57 s out of place at 30 fps. Timestamps are
derived from the absolute index, so a stride change never moves an event.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from visionpitch.common.config import IngestionConfig
from visionpitch.common.logging import StageCounters, get_logger

log = get_logger("ingestion")

_SUPPORTED_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".m4v", ".webm", ".mpg", ".mpeg", ".ts"}


# --------------------------------------------------------------------------- #


@dataclass(slots=True, frozen=True)
class Frame:
    """A decoded frame plus its identity in the source video."""

    #: absolute index in the source video
    idx: int
    #: seconds from the start of the source video
    timestamp_s: float
    image: np.ndarray  # BGR, HxWx3, uint8

    @property
    def shape(self) -> tuple[int, int]:
        return (self.image.shape[1], self.image.shape[0])  # (w, h)


@dataclass(slots=True, frozen=True)
class VideoMetadata:
    """Everything about the source needed for reproducible processing."""

    path: str
    video_id: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    codec: str
    file_size_bytes: int
    content_hash: str

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "video_id": self.video_id,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 6),
            "frame_count": self.frame_count,
            "duration_s": round(self.duration_s, 3),
            "codec": self.codec,
            "file_size_bytes": self.file_size_bytes,
            "content_hash": self.content_hash,
        }


# --------------------------------------------------------------------------- #
# Probing
# --------------------------------------------------------------------------- #


def _partial_content_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    """Hash of the head, middle and tail of the file plus its size.

    A full hash of a 4 GB match video costs many seconds for no extra safety in
    this context; three well-separated megabyte windows plus the exact byte
    length is more than enough to key a cache and to detect that the input
    changed underneath a resumed run.
    """
    size = path.stat().st_size
    hasher = hashlib.sha256()
    hasher.update(str(size).encode())
    with path.open("rb") as fh:
        for offset in (0, max(0, size // 2 - chunk_size // 2), max(0, size - chunk_size)):
            fh.seek(offset)
            hasher.update(fh.read(chunk_size))
    return hasher.hexdigest()


def _ffprobe(path: Path) -> dict | None:
    """Container metadata via ffprobe. Returns ``None`` if ffprobe is absent."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,codec_name,duration",
                "-show_entries",
                "format=duration,size",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def _parse_rational(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    if "/" in value:
        num, _, den = value.partition("/")
        try:
            n, d = float(num), float(den)
        except ValueError:
            return None
        return n / d if d else None
    try:
        return float(value)
    except ValueError:
        return None


def probe_video(path: str | Path) -> VideoMetadata:
    """Read video metadata, cross-checking ffprobe against OpenCV.

    OpenCV's ``CAP_PROP_FRAME_COUNT`` is derived from duration x fps and is
    frequently wrong for variable-frame-rate broadcast files; ffprobe's stream
    metadata is preferred when available and the two are reconciled.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"video not found: {path}")
    if path.suffix.lower() not in _SUPPORTED_SUFFIXES:
        log.warning(
            "unusual video extension %r - attempting to decode anyway", path.suffix
        )

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video (unsupported codec or corrupt file): {path}")
    try:
        cv_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cv_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cv_fps = float(cap.get(cv2.CAP_PROP_FPS))
        cv_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ")
    finally:
        cap.release()

    width, height, fps, frame_count = cv_width, cv_height, cv_fps, cv_count
    duration_s = 0.0

    probe = _ffprobe(path)
    if probe and probe.get("streams"):
        stream = probe["streams"][0]
        width = int(stream.get("width") or width)
        height = int(stream.get("height") or height)
        probe_fps = _parse_rational(stream.get("avg_frame_rate")) or _parse_rational(
            stream.get("r_frame_rate")
        )
        if probe_fps and probe_fps > 0:
            fps = probe_fps
        nb_frames = stream.get("nb_frames")
        if nb_frames and str(nb_frames).isdigit() and int(nb_frames) > 0:
            frame_count = int(nb_frames)
        fmt_duration = (probe.get("format") or {}).get("duration")
        if fmt_duration:
            try:
                duration_s = float(fmt_duration)
            except ValueError:
                duration_s = 0.0

    if not fps or fps <= 0 or not np.isfinite(fps):
        raise RuntimeError(
            f"could not determine frame rate for {path}; the file may be corrupt"
        )
    if duration_s <= 0 and frame_count > 0:
        duration_s = frame_count / fps
    if frame_count <= 0 and duration_s > 0:
        frame_count = int(round(duration_s * fps))
    if width <= 0 or height <= 0:
        raise RuntimeError(f"could not determine frame size for {path}")

    content_hash = _partial_content_hash(path)
    video_id = f"{path.stem}_{content_hash[:12]}"

    meta = VideoMetadata(
        path=str(path.resolve()),
        video_id=video_id,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_s=duration_s,
        codec=codec or "unknown",
        file_size_bytes=path.stat().st_size,
        content_hash=content_hash,
    )
    log.info(
        "%s: %dx%d @ %.3f fps, %d frames, %.1fs, codec=%s",
        path.name,
        width,
        height,
        fps,
        frame_count,
        duration_s,
        meta.codec,
    )
    return meta


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


class VideoReader:
    """Sequential frame reader honouring stride, time range and resume point.

    Sequential decoding with skip-on-read is used rather than per-frame seeking:
    seeking in long-GOP broadcast H.264 forces a keyframe search and is both
    slower and, for some containers, inaccurate by several frames.
    """

    def __init__(
        self,
        metadata: VideoMetadata,
        config: IngestionConfig,
        resume_from_frame: int | None = None,
    ) -> None:
        self.metadata = metadata
        self.config = config
        self.counters = StageCounters("ingestion")

        fps = metadata.fps
        start_frame = int(round((config.start_time_s or 0.0) * fps))
        if config.end_time_s is not None:
            end_frame = int(round(config.end_time_s * fps))
        else:
            end_frame = metadata.frame_count if metadata.frame_count > 0 else 1 << 30

        if resume_from_frame is not None:
            # Snap the resume point onto the sampling lattice so absolute frame
            # indices stay identical to a run that was never interrupted.
            base_frame = int(round((config.start_time_s or 0.0) * fps))
            start_frame = max(start_frame, resume_from_frame)
            offset = (start_frame - base_frame) % config.frame_stride
            if offset:
                start_frame += config.frame_stride - offset

        self.start_frame = start_frame
        self.end_frame = end_frame
        self._cap: cv2.VideoCapture | None = None

    # -- planning ----------------------------------------------------------- #

    @property
    def expected_frames(self) -> int:
        """How many frames this reader intends to yield. Used for progress."""
        span = max(0, self.end_frame - self.start_frame)
        planned = (span + self.config.frame_stride - 1) // self.config.frame_stride
        if self.config.max_frames is not None:
            planned = min(planned, self.config.max_frames)
        return planned

    def timestamp_of(self, frame_idx: int) -> float:
        return frame_idx / self.metadata.fps

    # -- context management ------------------------------------------------- #

    def __enter__(self) -> VideoReader:
        self._cap = cv2.VideoCapture(self.metadata.path)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open video: {self.metadata.path}")
        if self.start_frame > 0:
            # One coarse seek to the start is acceptable; within the processing
            # range we never seek again.
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        return self

    def __exit__(self, *exc_info) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # -- iteration ---------------------------------------------------------- #

    def __iter__(self) -> Iterator[Frame]:
        if self._cap is None:
            raise RuntimeError("VideoReader must be used as a context manager")

        cap = self._cap
        stride = self.config.frame_stride
        frame_idx = self.start_frame
        yielded = 0
        consecutive_failures = 0
        #: a run of failures this long is treated as end-of-stream, not corruption
        eof_threshold = 12

        while frame_idx < self.end_frame:
            if self.config.max_frames is not None and yielded >= self.config.max_frames:
                break

            ok, image = cap.read()
            if not ok or image is None:
                consecutive_failures += 1
                if consecutive_failures >= eof_threshold:
                    if frame_idx < self.end_frame - eof_threshold:
                        # We stopped well before the expected end: report it.
                        log.warning(
                            "decoding stopped at frame %d of an expected %d",
                            frame_idx,
                            self.end_frame,
                        )
                        self.counters.warn("early_end_of_stream")
                    break
                self.counters.fail("corrupt_frame")
                frame_idx += 1
                self._check_corruption_budget()
                continue

            consecutive_failures = 0

            if (frame_idx - self.start_frame) % stride == 0:
                self.counters.ok()
                yielded += 1
                yield Frame(
                    idx=frame_idx,
                    timestamp_s=self.timestamp_of(frame_idx),
                    image=image,
                )

            frame_idx += 1

        if self.counters.total_failures:
            log.warning(
                "ingestion finished with %d corrupt frames out of %d decoded (%.2f%%)",
                self.counters.total_failures,
                self.counters.processed,
                100 * self.counters.failure_ratio,
            )

    def _check_corruption_budget(self) -> None:
        seen = self.counters.processed + self.counters.total_failures
        # Only enforce once there is enough evidence to be meaningful.
        if seen < 100:
            return
        if self.counters.failure_ratio > self.config.max_corrupt_frame_ratio:
            raise RuntimeError(
                f"corrupt frame ratio {self.counters.failure_ratio:.3f} exceeds the "
                f"configured limit {self.config.max_corrupt_frame_ratio:.3f} after "
                f"{seen} frames; the input video is likely damaged"
            )
