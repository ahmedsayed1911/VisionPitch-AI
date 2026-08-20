"""Fetch public, expert-annotated football datasets for evaluation.

Why public data rather than hand annotation
-------------------------------------------
Phase 1's open acceptance item was real detection and tracking metrics. Hand
annotating a hundred broadcast frames is slow, and -- more importantly -- the
identity labels on the crowded side of a frame are genuinely ambiguous, so the
resulting "ground truth" would carry an unknown error that silently propagates
into every metric computed from it.

These datasets are annotated by their authors, are downloadable without
credentials, and let anyone reproduce the numbers exactly.

The distinction that matters
----------------------------
``in_distribution``
    The held-out test split of the dataset the shipped checkpoints were
    fine-tuned on. Measures the detector on its own domain. Legitimate and
    reproducible, but it is **not** evidence of generalisation.

``out_of_distribution``
    A different corpus entirely. This is the honest generalisation number and
    the one to quote.

Both are reported, never merged.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

DATASETS = {
    "player_det": {
        "repo_id": "martinjolif/football-player-detection",
        "repo_type": "dataset",
        "kind": "in_distribution",
        "task": "detection",
        "classes": ["ball", "goalkeeper", "player", "referee"],
        "allow_patterns": ["data/test/*", "data/data.yaml", "README.md"],
        "note": "Held-out test split of the multiclass checkpoint's own training corpus.",
        "approx_size_mb": 20,
    },
    "ball_det": {
        "repo_id": "martinjolif/football-ball-detection",
        "repo_type": "dataset",
        "kind": "in_distribution",
        "task": "detection_ball",
        "classes": ["ball"],
        "allow_patterns": ["data/test/*", "data/data.yaml", "README.md"],
        "note": "Held-out test split for the dedicated ball checkpoint.",
        "approx_size_mb": 60,
    },
    "field_keypoints": {
        "repo_id": "nreHieW/SoccerNet_Field_Keypoints",
        "repo_type": "dataset",
        "kind": "out_of_distribution",
        "task": "calibration",
        "allow_patterns": ["data/test-*.parquet", "README.md"],
        "note": "SoccerNet pitch keypoint annotations; independent of our keypoint model.",
        "approx_size_mb": 494,
    },
    "gsr": {
        "repo_id": "SoccerNet/SN-GSR-2025",
        "repo_type": "dataset",
        "kind": "out_of_distribution",
        "task": "tracking",
        "allow_patterns": ["test.zip"],
        "unzip": "test.zip",
        "note": (
            "SoccerNet Game State Reconstruction: identity-consistent tracking with "
            "team, role and pitch coordinates. The same task Phase 1 performs."
        ),
        "approx_size_mb": 8850,
    },
}


def stream_download(url: str, destination: Path, chunk_mb: int = 8) -> Path:
    """Resumable streaming download.

    Used instead of ``snapshot_download`` for the multi-gigabyte archives:
    the hub client stalled indefinitely at zero bytes on the 8.85 GB GSR
    archive, while a plain ranged GET against the same URL streamed fine. A
    direct stream also resumes cleanly after an interruption, which matters at
    this size.
    """
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0

    headers = {"Range": f"bytes={existing}-"} if existing else {}
    response = requests.get(url, stream=True, timeout=60, headers=headers)
    if response.status_code not in (200, 206):
        raise RuntimeError(f"download failed: HTTP {response.status_code} for {url}")

    total = int(response.headers.get("content-length", 0)) + existing
    mode = "ab" if existing and response.status_code == 206 else "wb"
    if mode == "wb":
        existing = 0

    downloaded = existing
    last_report = 0.0
    chunk = chunk_mb << 20
    with partial.open(mode) as fh:
        for block in response.iter_content(chunk):
            fh.write(block)
            downloaded += len(block)
            pct = 100 * downloaded / total if total else 0
            if pct - last_report >= 2.0:
                print(f"       {downloaded / 1e9:.2f} / {total / 1e9:.2f} GB ({pct:.0f}%)",
                      flush=True)
                last_report = pct

    partial.replace(destination)
    return destination


def fetch(key: str, out_root: Path) -> Path:
    from huggingface_hub import snapshot_download

    spec = DATASETS[key]
    target = out_root / key
    target.mkdir(parents=True, exist_ok=True)

    print(f"[get ] {spec['repo_id']}  (~{spec['approx_size_mb']} MB)  -> {target}")

    archive_name = spec.get("unzip")
    if archive_name:
        archive_path = target / archive_name
        marker = target / f".{archive_name}.extracted"
        if not archive_path.exists() and not marker.exists():
            url = (
                f"https://huggingface.co/datasets/{spec['repo_id']}"
                f"/resolve/main/{archive_name}"
            )
            stream_download(url, archive_path)
    else:
        snapshot_download(
            repo_id=spec["repo_id"],
            repo_type=spec["repo_type"],
            allow_patterns=spec["allow_patterns"],
            local_dir=str(target),
            max_workers=8,
        )

    archive = spec.get("unzip")
    if archive:
        zip_path = target / archive
        marker = target / f".{archive}.extracted"
        if zip_path.exists() and not marker.exists():
            print(f"[unzip] {archive} -> {target}")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(target)
            marker.write_text("ok", encoding="utf-8")
            # The zip is the bulk of the disk cost and is no longer needed.
            zip_path.unlink()
            print("       removed the archive after extraction")

    (target / "SOURCE.json").write_text(
        json.dumps({k: v for k, v in spec.items() if k != "allow_patterns"}, indent=2),
        encoding="utf-8",
    )
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", default=None,
                        help=f"subset of {list(DATASETS)}; default all")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent.parent / "data" / "eval")
    args = parser.parse_args()

    keys = args.datasets or list(DATASETS)
    for key in keys:
        if key not in DATASETS:
            print(f"unknown dataset {key!r}; choose from {list(DATASETS)}")
            return 1

    args.out.mkdir(parents=True, exist_ok=True)
    for key in keys:
        fetch(key, args.out)

    print("\nDone. These datasets keep their original licences; see each SOURCE.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
