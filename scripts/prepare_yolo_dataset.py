"""Export and validate the canonical official SN-GSR TRAIN/VALID detector set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionpitch.training.data_policy import find_project_root  # noqa: E402
from visionpitch.training.sngsr_export import (  # noqa: E402
    export_dataset,
    make_qa_montage,
    validate_export,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/yolo_gsr_detect"))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    project_root = find_project_root(Path(__file__).parent)
    output = args.out if args.out.is_absolute() else project_root / args.out
    if not args.validate_only:
        print(json.dumps(export_dataset(project_root, output), indent=2))
    report = validate_export(project_root, output)
    report["qa"] = {
        split: str(make_qa_montage(output, split)) for split in ("train", "valid")
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
