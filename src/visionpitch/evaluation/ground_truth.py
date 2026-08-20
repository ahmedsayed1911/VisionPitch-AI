"""Ground-truth annotation format and loading.

Format
------
A single JSON file per validation clip::

    {
      "video_id": "clip_a1b2c3",
      "fps": 25.0,
      "annotator": "who made this",
      "notes": "which segment and why it was chosen",
      "frames": {
        "120": [
          {"object_class": "player", "track_id": 3, "bbox": [x1, y1, x2, y2]},
          {"object_class": "ball",   "track_id": 0, "bbox": [x1, y1, x2, y2]}
        ]
      },
      "calibration": {
        "120": {"points": [[u, v], ...], "pitch_indices": [0, 5, 13, ...]}
      }
    }

Only annotated frames are evaluated. That matters: a sparsely annotated clip is
perfectly valid ground truth, but treating unannotated frames as "no objects"
would report a catastrophic false-positive rate. The loaders below therefore
carry the annotated frame set explicitly and every metric restricts itself to it.

``calibration`` is optional and holds manually marked pitch landmarks used to
measure homography error in metres.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from visionpitch.common.logging import get_logger
from visionpitch.common.types import BBox, ObjectClass

log = get_logger("evaluation.ground_truth")


@dataclass(slots=True)
class GTObject:
    object_class: ObjectClass
    track_id: int
    bbox: BBox


@dataclass
class GroundTruth:
    video_id: str
    fps: float
    frames: dict[int, list[GTObject]] = field(default_factory=dict)
    calibration: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    annotator: str = ""
    notes: str = ""

    @property
    def annotated_frames(self) -> list[int]:
        return sorted(self.frames)

    @property
    def n_objects(self) -> int:
        return sum(len(v) for v in self.frames.values())

    def of_class(self, object_class: ObjectClass) -> dict[int, list[GTObject]]:
        return {
            f: [o for o in objs if o.object_class is object_class]
            for f, objs in self.frames.items()
        }

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for objs in self.frames.values():
            for obj in objs:
                counts[obj.object_class.value] = counts.get(obj.object_class.value, 0) + 1
        track_ids = {
            (o.object_class.value, o.track_id) for objs in self.frames.values() for o in objs
        }
        return {
            "video_id": self.video_id,
            "annotated_frames": len(self.frames),
            "frame_range": (
                [min(self.frames), max(self.frames)] if self.frames else None
            ),
            "objects": self.n_objects,
            "objects_by_class": counts,
            "distinct_tracks": len(track_ids),
            "calibrated_frames": len(self.calibration),
            "annotator": self.annotator,
            "notes": self.notes,
        }


def load_ground_truth(path: str | Path) -> GroundTruth:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ground truth not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    gt = GroundTruth(
        video_id=data["video_id"],
        fps=float(data.get("fps", 25.0)),
        annotator=data.get("annotator", ""),
        notes=data.get("notes", ""),
    )

    for raw_frame, objects in data.get("frames", {}).items():
        frame_idx = int(raw_frame)
        parsed: list[GTObject] = []
        for obj in objects:
            box = obj["bbox"]
            if len(box) != 4:
                raise ValueError(f"frame {frame_idx}: bbox must have 4 values, got {box}")
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError(f"frame {frame_idx}: degenerate bbox {box} (expected xyxy)")
            parsed.append(
                GTObject(
                    object_class=ObjectClass(obj["object_class"]),
                    track_id=int(obj.get("track_id", -1)),
                    bbox=BBox.from_xyxy(box),
                )
            )
        gt.frames[frame_idx] = parsed

    for raw_frame, calib in data.get("calibration", {}).items():
        points = np.asarray(calib["points"], dtype=np.float64).reshape(-1, 2)
        indices = np.asarray(calib["pitch_indices"], dtype=int).ravel()
        if points.shape[0] != indices.shape[0]:
            raise ValueError(
                f"frame {raw_frame}: {points.shape[0]} points but {indices.shape[0]} indices"
            )
        gt.calibration[int(raw_frame)] = (points, indices)

    log.info(
        "loaded ground truth %s: %d frames, %d objects, %d calibrated frames",
        path.name,
        len(gt.frames),
        gt.n_objects,
        len(gt.calibration),
    )
    return gt


def save_ground_truth(gt: GroundTruth, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "video_id": gt.video_id,
        "fps": gt.fps,
        "annotator": gt.annotator,
        "notes": gt.notes,
        "frames": {
            str(frame): [
                {
                    "object_class": o.object_class.value,
                    "track_id": o.track_id,
                    "bbox": list(o.bbox.to_xyxy()),
                }
                for o in objs
            ]
            for frame, objs in sorted(gt.frames.items())
        },
        "calibration": {
            str(frame): {
                "points": points.tolist(),
                "pitch_indices": indices.tolist(),
            }
            for frame, (points, indices) in sorted(gt.calibration.items())
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
