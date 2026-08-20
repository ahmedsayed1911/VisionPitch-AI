"""Fetch the football checkpoints and record their provenance.

All three checkpoints are AGPL-3.0. Their hashes are written to
``models/manifest.json`` so any result set can be traced back to exact weights.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

MODELS = [
    {
        "key": "multiclass",
        "repo_id": "martinjolif/yolo-football-player-detection",
        "filename": "yolo-football-player-detection.pt",
        "purpose": "player / goalkeeper / referee / ball detection",
        "architecture": "YOLO11m",
        "license": "AGPL-3.0",
        "reported_metrics": {
            "player": {"mAP50": 0.9937, "mAP50_95": 0.8737},
            "goalkeeper": {"mAP50": 0.9413, "mAP50_95": 0.8024},
            "referee": {"mAP50": 0.9888, "mAP50_95": 0.7741},
            "ball": {"mAP50": 0.6799, "mAP50_95": 0.3380},
        },
    },
    {
        "key": "ball",
        "repo_id": "martinjolif/yolo-football-ball-detection",
        "filename": "yolo-football-ball-detection.pt",
        "purpose": "dedicated high-resolution ball detection",
        "architecture": "YOLO11n",
        "license": "AGPL-3.0",
        "reported_metrics": {
            "ball": {"precision": 0.8879, "recall": 0.8000, "mAP50": 0.8910, "mAP50_95": 0.5510}
        },
    },
    {
        "key": "pitch",
        "repo_id": "martinjolif/yolo-football-pitch-detection",
        "filename": "yolo-football-pitch-detection.pt",
        "purpose": "32-point pitch landmark regression for homography",
        "architecture": "YOLOv8x-pose",
        "license": "AGPL-3.0",
        "reported_metrics": None,
    },
]


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> int:
    from huggingface_hub import hf_hub_download

    out_dir = Path(__file__).resolve().parent.parent / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for spec in MODELS:
        target = out_dir / spec["filename"]
        if target.exists():
            print(f"[skip] {spec['filename']} already present")
        else:
            print(f"[get ] {spec['repo_id']}/{spec['filename']}")
            downloaded = hf_hub_download(repo_id=spec["repo_id"], filename=spec["filename"])
            target.write_bytes(Path(downloaded).read_bytes())

        entry = dict(spec)
        entry["path"] = str(target.relative_to(out_dir.parent)).replace("\\", "/")
        entry["sha256"] = sha256(target)
        entry["size_bytes"] = target.stat().st_size
        manifest.append(entry)
        print(f"       {entry['size_bytes'] / 1e6:.1f} MB  sha256={entry['sha256'][:16]}...")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nwrote {out_dir / 'manifest.json'}")
    print("\nAll three checkpoints are AGPL-3.0. See README.md for what that implies.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
