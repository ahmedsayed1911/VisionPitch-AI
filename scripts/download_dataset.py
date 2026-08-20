"""Download and extract the SoccerNet Game State Reconstruction dataset.

    python scripts/download_dataset.py --splits train valid

Source
------
``SoccerNet/SN-GSR-2025`` on HuggingFace Hub. Public and ungated; no NDA or
credentials required. Licensed GPL-3.0 (see docs/models_and_licenses.md).

Contents per clip: 750 frames at 25 fps (30 s) with bounding boxes, roles
(player / goalkeeper / referee / other), team affiliation, jersey numbers and
pitch coordinates. That combination is why this dataset was chosen: it
supplies training data *and* ground truth for detection, tracking, team
classification and calibration in one place.

Sizes (compressed): train 9.76 GB, valid 11.17 GB, test 8.85 GB,
challenge 5.31 GB. Extraction roughly doubles disk usage.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

REPO_ID = "SoccerNet/SN-GSR-2025"
VALID_SPLITS = ("train", "valid", "test", "challenge")
APPROX_GB = {"train": 9.76, "valid": 11.17, "test": 8.85, "challenge": 5.31}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid"],
        choices=VALID_SPLITS,
        help="Which splits to fetch (default: train valid).",
    )
    p.add_argument(
        "--root",
        type=Path,
        default=Path("data/SoccerNetGS"),
        help="Destination directory.",
    )
    p.add_argument(
        "--extract",
        action="store_true",
        default=True,
        help="Extract archives after download (default: on).",
    )
    p.add_argument(
        "--no-extract", dest="extract", action="store_false"
    )
    p.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep the .zip files after extraction (uses roughly 2x disk).",
    )
    return p.parse_args()


def check_disk_space(root: Path, splits: list[str], extract: bool) -> None:
    """Refuse to start a download that cannot possibly complete."""
    needed = sum(APPROX_GB[s] for s in splits)
    if extract:
        needed *= 2.1  # archive + extracted tree, briefly coexisting
    root.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(root).free / 1e9
    print(f"Estimated requirement: {needed:.1f} GB   Free: {free_gb:.1f} GB")
    if free_gb < needed:
        sys.exit(
            f"Insufficient disk space: need ~{needed:.1f} GB, have {free_gb:.1f} GB."
        )


def download(splits: list[str], root: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit(
            "huggingface_hub is required. Install with:\n"
            "  pip install 'visionpitch[data]'"
        )

    patterns = [f"{s}.zip" for s in splits]
    print(f"Downloading {patterns} from {REPO_ID} -> {root}")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision="main",
        local_dir=str(root),
        allow_patterns=patterns,
        max_workers=4,
    )


def extract(splits: list[str], root: Path, keep_archives: bool) -> None:
    for split in splits:
        archive = root / f"{split}.zip"
        if not archive.exists():
            print(f"  ! {archive} missing, skipping extraction")
            continue
        target = root / split
        if target.exists() and any(target.iterdir()):
            print(f"  = {target} already populated, skipping")
            continue
        print(f"  + extracting {archive.name} -> {target}")
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
        if not keep_archives:
            archive.unlink()
            print(f"  - removed {archive.name}")


def main() -> None:
    args = parse_args()
    check_disk_space(args.root, args.splits, args.extract)
    download(args.splits, args.root)
    if args.extract:
        extract(args.splits, args.root, args.keep_archives)
    print(f"\nDone. Dataset root: {args.root.resolve()}")


if __name__ == "__main__":
    main()
