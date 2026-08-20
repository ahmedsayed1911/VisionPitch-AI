"""One comparison table across every ball representation measured.

Cross-domain tiny-ball study, Part 10.

Reads the per-representation artefacts and assembles them into a single table on
identical held-out clips. Any metric a representation did not produce is printed
as ``--`` rather than as a zero: a blank means "not measured", and conflating
that with "measured as bad" is how a study talks itself into a conclusion.

Usage::

    python scripts/tinyball_compare.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionpitch.common.logging import configure_logging, get_logger  # noqa: E402
from visionpitch.evaluation.tinyball import CENTRE_TOLERANCES_PX  # noqa: E402

log = get_logger("tinyball.compare")

#: label -> (artefact path, representation name)
SOURCES = {
    "box_baseline": ("data/eval/tinyball/box_baseline_test.json", "bounding_box"),
    "box_multicorpus": (
        "data/eval/tinyball/box_multicorpus_test.json", "bounding_box"
    ),
    "heatmap": ("data/eval/tinyball/heatmap_test.json", "centre_heatmap"),
    "heatmap_temporal": (
        "data/eval/tinyball/heatmap_temporal_test.json", "temporal_centre_heatmap"
    ),
}

#: Downstream evidence, keyed by the representation it belongs to. Populated by
#: the pipeline runs; absent entries stay blank rather than defaulting.
DOWNSTREAM = Path("data/eval/tinyball/downstream.json")


def cell(value, width: int = 9, digits: int = 4) -> str:
    if value is None:
        return "--".rjust(width)
    if isinstance(value, float):
        return f"{value:.{digits}f}".rjust(width)
    return str(value).rjust(width)


def extract(path: Path) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    centre = payload.get("centre") or payload.get("test")
    if centre is None:
        return None
    per_domain = centre.get("per_domain", {})
    precision = [
        block["precision_at_px"]["25.0"] for block in per_domain.values()
    ]
    return {
        "label": payload.get("label"),
        "representation": payload.get("representation"),
        "iou50_macro_recall": (payload.get("iou50_macro") or {}).get("recall"),
        "iou50_macro_precision": (payload.get("iou50_macro") or {}).get("precision"),
        "centre_recall": centre.get("macro_recall_at_px", {}),
        "worst_centre_recall": centre.get("worst_domain_recall_at_px", {}),
        "macro_precision_25px": (
            round(sum(precision) / len(precision), 4) if precision else None
        ),
        "median_error_px": centre.get("median_error_px"),
        "macro_direct_coverage": centre.get("macro_direct_coverage"),
        "worst_domain_direct_coverage": centre.get("worst_domain_direct_coverage"),
        "false_positives_per_frame": centre.get("macro_false_positives_per_frame"),
        "runtime_ms_per_frame": (
            centre.get("runtime_ms_per_frame") or payload.get("runtime_ms_per_frame")
        ),
        "n_parameters": payload.get("n_parameters"),
        "model_fingerprint": (
            payload.get("model_fingerprint") or payload.get("checkpoint_fingerprint")
        ),
        "per_domain": {
            d: block["recall_at_px"]["25.0"] for d, block in per_domain.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/eval/tinyball"))
    args = parser.parse_args()
    configure_logging("INFO")

    downstream = (
        json.loads(DOWNSTREAM.read_text(encoding="utf-8"))
        if DOWNSTREAM.exists() else {}
    )

    rows = []
    for label, (path, _representation) in SOURCES.items():
        block = extract(Path(path))
        if block is None:
            log.info("%s: no artefact at %s (not measured)", label, path)
            continue
        block["downstream"] = downstream.get(label, {})
        rows.append(block)

    if not rows:
        log.error("no representation artefacts found")
        return 1

    payload = {
        "comparison": rows,
        "centre_tolerances_px": list(CENTRE_TOLERANCES_PX),
        "note": (
            "identical held-out test clips for every row. '--' means the metric "
            "was not measured for that representation, which is not the same as "
            "measuring it as zero."
        ),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / "representation_comparison.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # -- detection table -------------------------------------------------------- #
    header = (
        f"{'representation':<22}{'IoU50 R':>9}{'C@5':>9}{'C@25':>9}{'worst@25':>10}"
        f"{'P@25':>9}{'medErr':>9}{'cover':>9}{'wCover':>9}{'FP/fr':>8}{'ms':>8}"
    )
    print("\nDETECTION, held-out test partition")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['label']:<22}"
            f"{cell(row['iou50_macro_recall'])}"
            f"{cell(row['centre_recall'].get('5.0'))}"
            f"{cell(row['centre_recall'].get('25.0'))}"
            f"{cell(row['worst_centre_recall'].get('25.0'), 10)}"
            f"{cell(row['macro_precision_25px'])}"
            f"{cell(row['median_error_px'], 9, 2)}"
            f"{cell(row['macro_direct_coverage'])}"
            f"{cell(row['worst_domain_direct_coverage'])}"
            f"{cell(row['false_positives_per_frame'], 8, 3)}"
            f"{cell(row['runtime_ms_per_frame'], 8, 1)}"
        )

    print("\nPER-DOMAIN centre recall at 25 px")
    domains = sorted({d for row in rows for d in row["per_domain"]})
    print(f"{'representation':<22}" + "".join(f"{d:>16}" for d in domains))
    for row in rows:
        print(f"{row['label']:<22}"
              + "".join(cell(row["per_domain"].get(d), 16) for d in domains))

    # -- downstream table ------------------------------------------------------- #
    print("\nDOWNSTREAM, SN-BAS segment, unchanged event engine")
    down_header = (
        f"{'representation':<22}{'coverage':>10}{'determ.':>10}"
        f"{'passR':>9}{'passF1':>9}{'carryF1':>9}"
    )
    print(down_header)
    print("-" * len(down_header))
    for row in rows:
        d = row["downstream"]
        print(
            f"{row['label']:<22}"
            f"{cell(d.get('ball_coverage_direct'), 10)}"
            f"{cell(d.get('determinability'), 10)}"
            f"{cell(d.get('pass_recall'))}"
            f"{cell(d.get('pass_f1'))}"
            f"{cell(d.get('carry_f1'))}"
        )
    if not any(row["downstream"] for row in rows):
        print("  (no downstream measurements recorded yet)")

    print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
