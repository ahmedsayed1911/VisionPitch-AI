"""Video output.

Uses OpenCV's writer with mp4v, falling back to AVI/MJPG if the codec is not
available in the local build -- a silent write failure that produces a 0-byte
file is a genuinely confusing way to lose a run's output.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from visionpitch.common.logging import get_logger

log = get_logger("visualization.writer")


class VideoWriter:
    """Frame sink with codec fallback and size validation."""

    def __init__(self, path: str | Path, fps: float, size: tuple[int, int]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.size = size
        self.n_frames = 0

        self._writer = cv2.VideoWriter(
            str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, size
        )
        if not self._writer.isOpened():
            fallback = self.path.with_suffix(".avi")
            log.warning("mp4v encoder unavailable; falling back to MJPG at %s", fallback.name)
            self._writer = cv2.VideoWriter(
                str(fallback), cv2.VideoWriter_fourcc(*"MJPG"), self.fps, size
            )
            self.path = fallback
        if not self._writer.isOpened():
            raise RuntimeError(f"could not open any video encoder for {self.path}")

    def write(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if (w, h) != self.size:
            # Silently writing a mismatched frame produces a corrupt file, so
            # resize explicitly and say so once.
            if self.n_frames == 0:
                log.warning(
                    "frame size %dx%d does not match writer size %dx%d; resizing",
                    w,
                    h,
                    *self.size,
                )
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        self._writer.write(frame)
        self.n_frames += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            log.info("wrote %d frames -> %s", self.n_frames, self.path.name)

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
