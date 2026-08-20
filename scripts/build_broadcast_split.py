"""Build and verify the shot-disjoint local broadcast split.

Usage::

    python scripts/build_broadcast_split.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionpitch.annotation.splits import (  # noqa: E402
    assert_no_leakage,
    build_local_split,
    load_local,
    split_summary,
)
from visionpitch.common.logging import configure_logging, get_logger  # noqa: E402

log = get_logger("broadcast.split")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path("data/annotation/package"))
    parser.add_argument("--out", type=Path,
                        default=Path("data/annotation/local_split.json"))
    args = parser.parse_args()

    configure_logging("INFO")
    samples, annotations = load_local(args.package)
    if not annotations:
        log.error("no annotations in %s", args.package)
        return 1

    split = build_local_split(samples, annotations)
    leakage = assert_no_leakage(split, samples)
    summary = split_summary(split, samples, annotations)

    payload = {**split.to_dict(), "leakage_checks": leakage, "summary": summary}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nlocal split fingerprint: {split.fingerprint()}")
    print(f"\n{'split':<8}{'frames':>8}{'shots':>7}{'pos':>6}{'neg':>6}")
    for name, block in summary.items():
        print(f"{name:<8}{block['n_frames']:>8}{block['n_shots']:>7}"
              f"{block['n_positive']:>6}{block['n_negative']:>6}")
    print("\nleakage checks:")
    print(f"  shot-disjoint            : {leakage['shot_disjoint']}")
    print(f"  temporal-window-disjoint : {leakage['window_disjoint']}")
    print(f"  cross-split pairs < {leakage['adjacency_gap_frames']} frames apart: "
          f"{leakage['n_close_pairs_across_splits']}")
    for pair in leakage["close_pairs"]:
        print(f"    frames {pair['a']} / {pair['b']} (gap {pair['gap']}) "
              f"{pair['splits']}")
    print("\ncategories per split:")
    for name, block in summary.items():
        print(f"  {name}: {block['categories']}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
