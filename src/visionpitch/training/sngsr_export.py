"""Canonical, Phase-2D-native SN-GSR to YOLO detection exporter."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

from visionpitch.training.data_policy import CANONICAL_NAMES, POLICY_SCHEMA, TrainingDataPolicy

CATEGORY_TO_CLASS = {1: 0, 2: 1, 3: 2, 4: 3}
ROLE_TO_CLASS = {name: class_id for class_id, name in CANONICAL_NAMES.items()}


def _phase(sequence: str, modulus: int) -> int:
    return int(hashlib.sha256(sequence.encode()).hexdigest()[:8], 16) % modulus


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _annotations_by_image(data: dict) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = defaultdict(list)
    for annotation in data.get("annotations", []):
        if int(annotation.get("category_id", -1)) in CATEGORY_TO_CLASS:
            result[str(annotation["image_id"])].append(annotation)
    return result


def _canonical_objects(annotations: list[dict], width: int, height: int) -> list[dict]:
    objects = []
    for annotation in annotations:
        category_id = int(annotation["category_id"])
        class_id = CATEGORY_TO_CLASS[category_id]
        role = str(annotation.get("attributes", {}).get("role", "")).casefold()
        if role and ROLE_TO_CLASS.get(role) != class_id:
            raise ValueError(
                f"Annotation semantic mismatch: category {category_id}, role {role!r}, "
                f"annotation {annotation.get('id')}"
            )
        box = annotation.get("bbox_image", {})
        x, y, w, h = (float(box[key]) for key in ("x", "y", "w", "h"))
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
            raise ValueError(f"Illegal source box in annotation {annotation.get('id')}: {box}")
        objects.append(
            {
                "class_id": class_id,
                "cx": (x + w / 2) / width,
                "cy": (y + h / 2) / height,
                "w": w / width,
                "h": h / height,
                "area_px": w * h,
            }
        )
    return objects


def _sampling_reasons(
    sequence: str,
    index: int,
    total: int,
    objects: list[dict],
    previous_signature: tuple[int, ...] | None,
) -> list[str]:
    reasons = []
    if index % 3 == _phase(sequence, 3):
        reasons.append("uniform_stride_3")
    if index in (0, total - 1):
        reasons.append("sequence_boundary")
    signature = tuple(sorted({obj["class_id"] for obj in objects}))
    if previous_signature is not None and signature != previous_signature:
        reasons.append("class_presence_transition")
    people = [obj for obj in objects if obj["class_id"] < 3]
    hard = (
        len(people) >= 20
        or any(obj["class_id"] == 3 and obj["area_px"] <= 256 for obj in objects)
        or any(obj["class_id"] < 3 and obj["area_px"] <= 1500 for obj in objects)
    )
    if hard and index % 6 == (_phase(sequence, 3) + 1) % 6:
        reasons.append("hard_frame_stride_6")
    if not any(obj["class_id"] == 3 for obj in objects) and index % 9 == _phase(sequence, 9):
        reasons.append("ball_negative_stride_9")
    return reasons


def _link(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        try:
            os.symlink(source, destination)
            return "symlink"
        except OSError as exc:
            raise RuntimeError(
                f"Could not link {source}; exporter refuses to silently duplicate images"
            ) from exc


def export_dataset(project_root: Path, output: Path) -> dict:
    project_root = project_root.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to merge/overwrite an existing export: {output}")
    policy = TrainingDataPolicy(project_root)
    policy.assert_artifact_path_allowed(output)
    frames_path = output / "frames.jsonl"
    for subset in ("train", "val"):
        (output / "images" / subset).mkdir(parents=True)
        (output / "labels" / subset).mkdir(parents=True)

    class_counts = {split: Counter() for split in ("train", "valid")}
    frame_counts = Counter()
    negative_counts = Counter()
    reason_counts = {split: Counter() for split in ("train", "valid")}
    link_counts = Counter()
    manifest_rows = []

    for source_split, yolo_split in (("train", "train"), ("valid", "val")):
        split_root = policy.official[source_split]
        sequences = sorted(path for path in split_root.iterdir() if path.is_dir())
        for sequence_dir in sequences:
            sequence = sequence_dir.name
            policy.assert_sequence_allowed(sequence)
            label_json = sequence_dir / "Labels-GameState.json"
            data = json.loads(label_json.read_text(encoding="utf-8"))
            categories = {
                int(row["id"]): row["name"]
                for row in data.get("categories", [])
                if int(row["id"]) <= 4
            }
            expected_categories = {1: "player", 2: "goalkeeper", 3: "referee", 4: "ball"}
            if categories != expected_categories:
                raise ValueError(f"Unexpected annotation semantics in {label_json}: {categories}")
            images = sorted(data.get("images", []), key=lambda row: row["file_name"])
            annotations = _annotations_by_image(data)
            previous_signature = None
            for index, image_row in enumerate(images):
                width, height = int(image_row["width"]), int(image_row["height"])
                image_id = str(image_row["image_id"])
                objects = _canonical_objects(annotations.get(image_id, []), width, height)
                reasons = _sampling_reasons(
                    sequence, index, len(images), objects, previous_signature
                )
                previous_signature = tuple(sorted({obj["class_id"] for obj in objects}))
                if not reasons:
                    continue
                source = sequence_dir / "img1" / image_row["file_name"]
                policy.assert_source_allowed(source, source_split, sequence)
                if not source.is_file():
                    raise FileNotFoundError(source)
                frame_stem = Path(image_row["file_name"]).stem
                stem = f"{source_split}__{sequence}__{frame_stem}"
                exported_image = output / "images" / yolo_split / f"{stem}{source.suffix.lower()}"
                exported_label = output / "labels" / yolo_split / f"{stem}.txt"
                link_method = _link(source, exported_image)
                lines = [
                    f"{obj['class_id']} {obj['cx']:.8f} {obj['cy']:.8f} "
                    f"{obj['w']:.8f} {obj['h']:.8f}"
                    for obj in objects
                ]
                exported_label.write_text("\n".join(lines), encoding="utf-8")
                counts = Counter(obj["class_id"] for obj in objects)
                class_counts[source_split].update(counts)
                frame_counts[source_split] += 1
                negative_counts[f"{source_split}_empty"] += not objects
                negative_counts[f"{source_split}_ball_negative"] += counts[3] == 0
                reason_counts[source_split].update(reasons)
                link_counts[link_method] += 1
                manifest_rows.append(
                    {
                        "source_split": source_split,
                        "source_sequence": sequence,
                        "source_frame": frame_stem,
                        "source_image_id": image_id,
                        "source_path": _relative(source, project_root),
                        "exported_image": _relative(exported_image, output),
                        "exported_label": _relative(exported_label, output),
                        "width": width,
                        "height": height,
                        "objects": len(objects),
                        "class_counts": {str(key): counts[key] for key in CANONICAL_NAMES},
                        "ball_negative": counts[3] == 0,
                        "sampling_reasons": reasons,
                        "link_method": link_method,
                    }
                )

    with frames_path.open("w", encoding="utf-8") as stream:
        for row in manifest_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "schema": POLICY_SCHEMA,
        "dataset": "SoccerNet Game State Reconstruction",
        "task": "four-class object detection",
        "source_splits": ["train", "valid"],
        "split_policy": "official split boundaries preserved; no internal re-split",
        "sampling_policy": {
            "uniform": "deterministic sequence-phased stride 3",
            "hard_frames": "additional stride 6 for crowded, tiny-ball, or distant-person frames",
            "transitions": "all changes in canonical class-presence signature",
            "ball_negatives": "additional deterministic stride 9 where no ball is annotated",
            "oversampling": False,
        },
        "class_names": {str(key): value for key, value in CANONICAL_NAMES.items()},
        "frames_file": frames_path.name,
        "exported_frames": len(manifest_rows),
        "frames_by_split": dict(frame_counts),
        "objects_by_split_and_class": {
            split: {CANONICAL_NAMES[key]: class_counts[split][key] for key in CANONICAL_NAMES}
            for split in ("train", "valid")
        },
        "negative_frames": dict(negative_counts),
        "sampling_reason_counts": {
            split: dict(reason_counts[split]) for split in ("train", "valid")
        },
        "link_methods": dict(link_counts),
        "legacy_test_status": "LEGACY_CONTAMINATED_TEST; excluded",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    dataset_yaml = {
        "path": output.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "nc": 4,
        "names": CANONICAL_NAMES,
    }
    (output / "dataset.yaml").write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8"
    )
    return manifest


def validate_export(project_root: Path, output: Path) -> dict:
    output = output.resolve()
    policy = TrainingDataPolicy(project_root)
    policy.assert_artifact_path_allowed(output)
    manifest = policy.validate_dataset_yaml(output / "dataset.yaml")
    class_counts = {split: Counter() for split in ("train", "valid")}
    frames = Counter()
    ball_negative = Counter()
    with (output / manifest["frames_file"]).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            image_path = output / row["exported_image"]
            label_path = output / row["exported_label"]
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as exc:
                raise ValueError(f"Unreadable image at row {line_number}: {image_path}") from exc
            seen_objects = 0
            for label_line in label_path.read_text(encoding="utf-8").splitlines():
                fields = label_line.split()
                if len(fields) != 5:
                    raise ValueError(f"Malformed label {label_path}: {label_line!r}")
                class_id = int(fields[0])
                cx, cy, width, height = map(float, fields[1:])
                if class_id not in CANONICAL_NAMES:
                    raise ValueError(f"Invalid class {class_id} in {label_path}")
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
                    raise ValueError(f"Invalid normalized box in {label_path}: {label_line}")
                if cx - width / 2 < -1e-6 or cx + width / 2 > 1 + 1e-6:
                    raise ValueError(f"Box crosses horizontal boundary in {label_path}")
                if cy - height / 2 < -1e-6 or cy + height / 2 > 1 + 1e-6:
                    raise ValueError(f"Box crosses vertical boundary in {label_path}")
                class_counts[row["source_split"]][class_id] += 1
                seen_objects += 1
            if seen_objects != int(row["objects"]):
                raise ValueError(f"Object-count mismatch at manifest row {line_number}")
            frames[row["source_split"]] += 1
            ball_negative[row["source_split"]] += bool(row["ball_negative"])
    expected = manifest["objects_by_split_and_class"]
    actual = {
        split: {CANONICAL_NAMES[key]: class_counts[split][key] for key in CANONICAL_NAMES}
        for split in ("train", "valid")
    }
    if actual != expected or dict(frames) != manifest["frames_by_split"]:
        raise ValueError("Validated distribution does not match manifest summary")
    return {
        "status": "passed",
        "readable_images": sum(frames.values()),
        "frames_by_split": dict(frames),
        "objects_by_split_and_class": actual,
        "ball_negative_by_split": dict(ball_negative),
    }


def make_qa_montage(output: Path, source_split: str, sample_count: int = 12) -> Path:
    output = output.resolve()
    rows = []
    for line in (output / "frames.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["source_split"] == source_split:
            rows.append(row)
    ranked = sorted(rows, key=lambda row: hashlib.sha256(json.dumps(row).encode()).hexdigest())
    samples = ranked[:sample_count]
    tiles = []
    colours = {0: "#20d060", 1: "#ffd020", 2: "#ff7040", 3: "#30a0ff"}
    for row in samples:
        image = Image.open(output / row["exported_image"]).convert("RGB")
        scale = 420 / image.width
        image = image.resize((420, int(image.height * scale)))
        draw = ImageDraw.Draw(image)
        for line in (output / row["exported_label"]).read_text().splitlines():
            class_id, cx, cy, width, height = map(float, line.split())
            x1 = (cx - width / 2) * image.width
            y1 = (cy - height / 2) * image.height
            x2 = (cx + width / 2) * image.width
            y2 = (cy + height / 2) * image.height
            draw.rectangle((x1, y1, x2, y2), outline=colours[int(class_id)], width=2)
        draw.rectangle((0, 0, image.width, 20), fill="black")
        draw.text((4, 3), f"{row['source_sequence']} / {row['source_frame']}", fill="white")
        tiles.append(image)
    tile_height = max(tile.height for tile in tiles)
    sheet = Image.new("RGB", (420 * 3, tile_height * 4), "#202020")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % 3) * 420, (index // 3) * tile_height))
    qa_dir = output / "qa"
    qa_dir.mkdir(exist_ok=True)
    destination = qa_dir / f"{source_split}_sample.jpg"
    sheet.save(destination, quality=92)
    return destination
