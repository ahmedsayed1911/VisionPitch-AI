"""Adapters turning public annotated datasets into :class:`GroundTruth`.

Each adapter records whether the corpus is **in-distribution** for the shipped
checkpoints (their own held-out test split) or **out-of-distribution** (a
different corpus). The evaluation report keeps the two apart and never averages
them: an in-distribution score measures the detector on its own domain, and only
an out-of-distribution score says anything about generalisation to arbitrary
broadcast footage.

Supported formats
-----------------
``yolo``
    Roboflow/YOLO layout -- ``images/*.jpg`` beside ``labels/*.txt``, one line
    per object as ``class cx cy w h`` with all four geometry values normalised
    to the image size. Single frames, so **detection only**: there are no track
    identities and any tracking metric computed from them would be meaningless.

``gsr``
    SoccerNet Game State Reconstruction -- image sequences with per-frame
    annotations carrying persistent ``track_id``, role, team and pitch position.
    This is the same task Phase 1 performs, and the only corpus here that
    supports HOTA, IDF1 and MOTA.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from visionpitch.common.logging import get_logger
from visionpitch.common.types import BBox, ObjectClass
from visionpitch.evaluation.ground_truth import GroundTruth, GTObject

log = get_logger("evaluation.datasets")


@dataclass
class DatasetInfo:
    """Provenance for a benchmark corpus."""

    key: str
    name: str
    kind: str  # "in_distribution" | "out_of_distribution"
    task: str  # "detection" | "detection_ball" | "tracking" | "calibration"
    n_frames: int
    n_objects: int
    supports_identity: bool
    licence: str = ""
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "name": self.name,
            "kind": self.kind,
            "task": self.task,
            "n_frames": self.n_frames,
            "n_objects": self.n_objects,
            "supports_identity": self.supports_identity,
            "licence": self.licence,
            "note": self.note,
        }


# --------------------------------------------------------------------------- #
# YOLO / Roboflow
# --------------------------------------------------------------------------- #


def _read_yaml_names(data_yaml: Path) -> dict[int, str]:
    import yaml

    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = data.get("names")
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return dict(enumerate(str(n) for n in names))
    raise ValueError(f"could not read class names from {data_yaml}")


_ALIASES = {
    "ball": ObjectClass.BALL,
    "football": ObjectClass.BALL,
    "goalkeeper": ObjectClass.GOALKEEPER,
    "player": ObjectClass.PLAYER,
    "referee": ObjectClass.REFEREE,
}


@dataclass
class YoloFrame:
    """One annotated still, with the path needed to run a detector on it."""

    frame_idx: int
    image_path: Path
    width: int
    height: int
    objects: list[GTObject]


class YoloDetectionDataset:
    """A directory of images plus YOLO label files."""

    def __init__(self, root: Path, split: str = "test") -> None:
        self.root = Path(root)
        data_dir = self.root / "data"
        if not data_dir.exists():
            data_dir = self.root

        self.images_dir = data_dir / split / "images"
        self.labels_dir = data_dir / split / "labels"
        if not self.images_dir.exists():
            raise FileNotFoundError(f"no images at {self.images_dir}")

        yaml_path = data_dir / "data.yaml"
        self.class_names = _read_yaml_names(yaml_path) if yaml_path.exists() else {0: "ball"}
        self.class_map = {
            idx: _ALIASES[name.strip().lower()]
            for idx, name in self.class_names.items()
            if name.strip().lower() in _ALIASES
        }
        if not self.class_map:
            raise ValueError(f"none of {list(self.class_names.values())} are known classes")

        source = self.root / "SOURCE.json"
        self.source = json.loads(source.read_text(encoding="utf-8")) if source.exists() else {}
        self.frames = self._load()

    def _load(self) -> list[YoloFrame]:
        import cv2

        frames: list[YoloFrame] = []
        for idx, image_path in enumerate(sorted(self.images_dir.glob("*.*"))):
            if image_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            label_path = self.labels_dir / f"{image_path.stem}.txt"

            image = cv2.imread(str(image_path))
            if image is None:
                log.warning("could not read %s; skipping", image_path.name)
                continue
            h, w = image.shape[:2]

            objects: list[GTObject] = []
            if label_path.exists():
                for line_no, line in enumerate(
                    label_path.read_text(encoding="utf-8").splitlines()
                ):
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cls_idx = int(float(parts[0]))
                    object_class = self.class_map.get(cls_idx)
                    if object_class is None:
                        continue
                    cx, cy, bw, bh = (float(v) for v in parts[1:5])
                    x1 = (cx - bw / 2) * w
                    y1 = (cy - bh / 2) * h
                    x2 = (cx + bw / 2) * w
                    y2 = (cy + bh / 2) * h
                    if x2 <= x1 or y2 <= y1:
                        log.warning(
                            "%s line %d: degenerate box, skipped", label_path.name, line_no
                        )
                        continue
                    # No identity in a still-image corpus. -1 marks that
                    # explicitly so nothing downstream mistakes it for a track.
                    objects.append(GTObject(object_class, -1, BBox(x1, y1, x2, y2)))

            frames.append(YoloFrame(idx, image_path, w, h, objects))
        return frames

    # -- interface ---------------------------------------------------------- #

    def to_ground_truth(self) -> GroundTruth:
        gt = GroundTruth(
            video_id=self.root.name,
            fps=25.0,
            annotator=self.source.get("repo_id", "public dataset"),
            notes=(
                f"{self.source.get('note', '')} Still images: detection only, "
                f"no track identities."
            ),
        )
        for frame in self.frames:
            gt.frames[frame.frame_idx] = frame.objects
        return gt

    def info(self) -> DatasetInfo:
        return DatasetInfo(
            key=self.root.name,
            name=self.source.get("repo_id", self.root.name),
            kind=self.source.get("kind", "unknown"),
            task=self.source.get("task", "detection"),
            n_frames=len(self.frames),
            n_objects=sum(len(f.objects) for f in self.frames),
            supports_identity=False,
            licence="CC BY 4.0",
            note=self.source.get("note", ""),
        )


# --------------------------------------------------------------------------- #
# SoccerNet Game State Reconstruction
# --------------------------------------------------------------------------- #

#: SN-GSR role strings mapped onto our classes.
_GSR_ROLES = {
    "player": ObjectClass.PLAYER,
    "goalkeeper": ObjectClass.GOALKEEPER,
    "referee": ObjectClass.REFEREE,
    "ball": ObjectClass.BALL,
}


@dataclass
class GSRSequence:
    """One SN-GSR clip: ordered image files plus identity-consistent labels."""

    name: str
    image_paths: dict[int, Path]
    ground_truth: GroundTruth
    #: frames whose annotation is incomplete or explicitly ignored
    ignored_frames: set[int]

    @property
    def n_frames(self) -> int:
        return len(self.image_paths)


class GSRDataset:
    """SoccerNet Game State Reconstruction, read from its extracted layout."""

    def __init__(self, root: Path, max_sequences: int | None = None) -> None:
        self.root = Path(root)
        source = self.root / "SOURCE.json"
        self.source = json.loads(source.read_text(encoding="utf-8")) if source.exists() else {}
        self.sequences = self._discover(max_sequences)

    def _sequence_dirs(self) -> list[Path]:
        # The archive nests differently between releases; find any directory
        # holding a Labels-GameState.json rather than assuming a fixed depth.
        found = sorted({p.parent for p in self.root.rglob("Labels-GameState.json")})
        return found

    def _discover(self, max_sequences: int | None) -> list[GSRSequence]:
        dirs = self._sequence_dirs()
        if not dirs:
            raise FileNotFoundError(
                f"no Labels-GameState.json found under {self.root}. "
                f"Run: python scripts/download_eval_data.py gsr"
            )
        if max_sequences is not None:
            dirs = dirs[:max_sequences]

        sequences: list[GSRSequence] = []
        for seq_dir in dirs:
            try:
                sequences.append(self._load_sequence(seq_dir))
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                log.warning("skipping sequence %s: %s", seq_dir.name, exc)
        log.info("loaded %d GSR sequence(s) from %s", len(sequences), self.root)
        return sequences

    def _load_sequence(self, seq_dir: Path) -> GSRSequence:
        labels = json.loads((seq_dir / "Labels-GameState.json").read_text(encoding="utf-8"))

        # image_id -> file
        image_dir = seq_dir / "img1"
        by_image_id: dict[str, Path] = {}
        frame_of_image: dict[str, int] = {}
        for entry in labels.get("images", []):
            image_id = str(entry["image_id"])
            file_name = entry.get("file_name")
            if file_name is None:
                continue
            path = image_dir / file_name
            if path.exists():
                by_image_id[image_id] = path
                # SN-GSR file names are 1-based zero-padded frame numbers.
                frame_of_image[image_id] = int(Path(file_name).stem)

        gt = GroundTruth(
            video_id=seq_dir.name,
            fps=float(labels.get("info", {}).get("frame_rate", 25.0) or 25.0),
            annotator="SoccerNet SN-GSR-2025",
            notes=(
                "Identity-consistent tracking annotations with role and team. "
                "Out-of-distribution relative to the shipped checkpoints."
            ),
        )

        ignored: set[int] = set()
        for annotation in labels.get("annotations", []):
            image_id = str(annotation.get("image_id"))
            if image_id not in frame_of_image:
                continue
            frame_idx = frame_of_image[image_id]

            attributes = annotation.get("attributes") or {}
            role = str(attributes.get("role", "")).strip().lower()
            object_class = _GSR_ROLES.get(role)
            if object_class is None:
                # Non-participants (staff, spectators marked in some releases)
                # are annotated but are not part of our class set. Marking the
                # frame ignored is safer than treating our detection of them as
                # a false positive.
                if role:
                    ignored.add(frame_idx)
                continue

            bbox = annotation.get("bbox_image") or {}
            x = bbox.get("x")
            y = bbox.get("y")
            w = bbox.get("w")
            h = bbox.get("h")
            if None in (x, y, w, h) or w <= 0 or h <= 0:
                continue

            track_id = annotation.get("track_id")
            if track_id is None:
                # No identity means this object cannot participate in identity
                # metrics. Rather than invent one, drop the whole frame from the
                # tracking subset.
                ignored.add(frame_idx)
                continue

            gt.frames.setdefault(frame_idx, []).append(
                GTObject(object_class, int(track_id), BBox(float(x), float(y),
                                                           float(x) + float(w),
                                                           float(y) + float(h)))
            )

        image_paths = {frame_of_image[i]: p for i, p in by_image_id.items()}
        return GSRSequence(seq_dir.name, image_paths, gt, ignored)

    def info(self) -> DatasetInfo:
        return DatasetInfo(
            key="gsr",
            name=self.source.get("repo_id", "SoccerNet/SN-GSR-2025"),
            kind="out_of_distribution",
            task="tracking",
            n_frames=sum(s.n_frames for s in self.sequences),
            n_objects=sum(s.ground_truth.n_objects for s in self.sequences),
            supports_identity=True,
            licence="SoccerNet terms",
            note=self.source.get("note", ""),
        )


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def validate_ground_truth(gt: GroundTruth, require_identity: bool = False) -> dict:
    """Structural checks on an annotation set.

    Catches the mistakes that quietly invalidate metrics: duplicate identities
    inside one frame, degenerate boxes, frames annotated with nothing at all,
    and identity labels that are absent when the metric needs them.
    """
    issues: dict[str, list] = {
        "duplicate_track_ids": [],
        "degenerate_boxes": [],
        "empty_frames": [],
        "missing_identity": [],
        "suspicious_aspect": [],
    }

    for frame_idx, objects in sorted(gt.frames.items()):
        if not objects:
            issues["empty_frames"].append(frame_idx)
            continue

        seen: dict[tuple[str, int], int] = {}
        for obj in objects:
            if obj.bbox.width <= 0 or obj.bbox.height <= 0:
                issues["degenerate_boxes"].append(frame_idx)
            # A person 4x wider than tall, or a ball 10x wider than tall, is
            # almost always a transcription error.
            aspect = obj.bbox.width / max(1e-6, obj.bbox.height)
            if obj.object_class.is_person and not 0.15 <= aspect <= 3.0:
                issues["suspicious_aspect"].append((frame_idx, round(aspect, 2)))

            if require_identity and obj.track_id < 0:
                issues["missing_identity"].append(frame_idx)

            if obj.track_id >= 0:
                key = (obj.object_class.value, obj.track_id)
                seen[key] = seen.get(key, 0) + 1
        for key, count in seen.items():
            if count > 1:
                issues["duplicate_track_ids"].append((frame_idx, key, count))

    counts: dict[str, int] = {}
    for objects in gt.frames.values():
        for obj in objects:
            counts[obj.object_class.value] = counts.get(obj.object_class.value, 0) + 1

    return {
        "n_frames": len(gt.frames),
        "n_objects": gt.n_objects,
        "objects_by_class": counts,
        "missing_classes": sorted(
            {c.value for c in ObjectClass} - set(counts)
        ),
        "issues": {k: v[:20] for k, v in issues.items() if v},
        "issue_counts": {k: len(v) for k, v in issues.items() if v},
        "usable_for_identity_metrics": not issues["missing_identity"],
    }


def bootstrap_interval(
    values: list[float], n_resamples: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float] | None:
    """Percentile bootstrap CI over per-frame values.

    Resampling is over *frames*, not objects: objects within a frame are highly
    correlated (same lighting, same camera pose, same crowd), so resampling
    objects independently would report an interval several times too narrow.
    """
    array = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if array.size < 5:
        return None
    rng = np.random.default_rng(seed)
    means = np.array(
        [rng.choice(array, size=array.size, replace=True).mean() for _ in range(n_resamples)]
    )
    return (
        round(float(np.percentile(means, 100 * alpha / 2)), 4),
        round(float(np.percentile(means, 100 * (1 - alpha / 2))), 4),
    )
