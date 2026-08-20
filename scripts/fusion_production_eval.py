"""Four-way locked downstream evaluation of ball temporal fusion.

Rows A/B/C/D and the Part 7 ablation, each run through the **real** pipeline and
the **unchanged** possession and event engines. Nothing downstream of fusion is
touched; the only thing that varies between rows is the detector checkpoint and
the ``ball_fusion`` block, both frozen in
:mod:`visionpitch.evaluation.fusion_rows`.

Why full pipeline runs rather than a cached offline replay
----------------------------------------------------------
The pipeline is deterministic for a fixed video and config, so re-running it
reproduces the detector candidates exactly -- which is what Part 3 requires. A
replay harness would be faster but would introduce a second code path whose
divergence from production is exactly the kind of thing that silently corrupts
a decisive measurement. The runtime cost is paid once.

The offline reuse path Part 1 asks for does exist: ``ball_fusion`` consumes the
candidate map, not pixels, so an ablation row costs no detector inference beyond
the first run of its checkpoint. That is measured and reported as
``fusion_ms_per_frame``.

Usage::

    python scripts/fusion_production_eval.py --rows A B C D
    python scripts/fusion_production_eval.py --ablation
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionpitch.common.logging import configure_logging, get_logger  # noqa: E402
from visionpitch.evaluation.fusion_metrics import (  # noqa: E402
    system_metrics,
    trajectory_metrics,
)
from visionpitch.evaluation.fusion_rows import (  # noqa: E402
    ABLATION,
    ROWS,
    overrides,
)
from visionpitch.evaluation.fusion_rows import fingerprint as rows_fingerprint  # noqa: E402
from visionpitch.evaluation.fusion_thresholds import (  # noqa: E402
    THRESHOLD_ARTIFACT,
    spec_fingerprint,
)

log = get_logger("fusion.production")

# -- the locked evaluation data, unchanged from the broadcast stress test ----- #
LOCAL_VIDEO = Path(
    "ملخص مباراة نيوزيلندا ومصر _ دور المجموعات - كأس العالم FIFA 2026™.mp4"
)
BAS_VIDEO = Path("data/eval/bas/mid_pre_720p.mp4")
BAS_GT = Path("data/eval/bas/event_gt_half1.json")
LOCAL_SEGMENT = (0.0, 120.0)
BAS_SEGMENT = (600.0, 780.0)

RECORD = Path("data/eval/fusion/production_eval.json")
ABLATION_RECORD = Path("data/eval/fusion/production_ablation.json")

#: Reference values from the completed broadcast comparison. Row A and row C
#: must reproduce these; a mismatch means the legacy path was disturbed.
LEGACY_BASELINE = {
    "A": {
        "local": {"ball_coverage_direct": 0.3225, "determinability": 0.0841},
        "bas": {"ball_coverage_direct": 0.4229, "determinability": 0.1214,
                "pass_f1": 0.323, "carry_f1": 0.304},
    },
    "C": {
        "local": {"ball_coverage_direct": 0.3462, "determinability": 0.0758},
        "bas": {"ball_coverage_direct": 0.3982, "determinability": 0.1110,
                "pass_f1": 0.345, "carry_f1": 0.311},
    },
}
#: Reproduction tolerance. The pipeline is deterministic, so this is not a
#: statistical allowance -- it is rounding in the stored baseline.
REPRODUCTION_TOLERANCE = 0.001


def shell(command: list[str]) -> str:
    log.info("$ %s", " ".join(str(c) for c in command))
    result = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        log.error(
            "failed:\n%s", (result.stdout or "")[-3000:] + (result.stderr or "")[-3000:]
        )
        raise SystemExit(result.returncode)
    return result.stdout or ""


def newest_run(output_root: Path) -> Path:
    candidates = sorted(
        (p.parent for p in output_root.glob("*/*/game_state.parquet")),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise SystemExit(f"no completed run under {output_root}")
    return candidates[-1]


def analyse(video: Path, output_root: Path, settings: list[str],
            start: float, end: float) -> tuple[Path, float]:
    command = [
        sys.executable, "-m", "visionpitch.cli", "analyse", str(video),
        "--output", str(output_root), "--start", str(start), "--end", str(end),
        "--no-render",
    ]
    for setting in settings:
        command += ["--set", setting]
    started = time.perf_counter()
    shell(command)
    return newest_run(output_root), time.perf_counter() - started


def cut_frames_of(run_dir: Path) -> set[int]:
    """Shot boundaries the calibrator recorded, for cut-continuity scoring."""
    manifest = run_dir / "manifest.json"
    if not manifest.exists():
        return set()
    stages = json.loads(manifest.read_text(encoding="utf-8")).get("stages", {}) or {}
    return set(stages.get("calibration", {}).get("shot_boundary_frames", []) or [])


def measure(video: Path, output_root: Path, settings: list[str],
            segment: tuple[float, float], label: str,
            event_gt: Path | None) -> dict:
    """One video under one configuration, through every downstream stage."""
    from visionpitch.analytics.runner import run_analytics

    run_dir, wall_s = analyse(video, output_root, settings, *segment)
    run_analytics(run_dir)

    shell([
        sys.executable, "scripts/possession_determinability.py",
        "--run", str(run_dir), "--label", label,
    ])
    determinability = json.loads(
        Path(f"data/eval/determinability/determinability_{label}.json")
        .read_text(encoding="utf-8")
    )

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    stages = manifest.get("stages", {}) or {}

    block: dict = {
        "run_dir": str(run_dir),
        "config_fingerprint": manifest.get("config_fingerprint"),
        "wall_clock_s": round(wall_s, 1),
        "ball_coverage_direct": determinability["ball_coverage_direct"],
        "determinability": determinability["determinability"],
        "unknown_ratio": determinability["unknown_ratio"],
        "state_committed_fraction": determinability["state_committed_fraction"],
        "controlled_fraction": determinability["controlled_fraction"],
        "observable_fraction": determinability["observable_fraction"],
        "by_ball_evidence": determinability["by_ball_evidence"],
        "trajectory": trajectory_metrics(run_dir, cut_frames=cut_frames_of(run_dir)),
        "system": system_metrics(run_dir),
        "ball_fusion_report": stages.get("ball_fusion", {}),
        "ball_tracking_report": {
            k: v for k, v in (stages.get("ball_tracking", {}) or {}).items()
            if not isinstance(v, dict) or k == "error_analysis"
        },
    }
    block["system"]["wall_clock_s"] = round(wall_s, 1)

    if event_gt is not None and event_gt.exists():
        shell([
            sys.executable, "scripts/evaluate_events.py", "--run", str(run_dir),
            "--gt", str(event_gt), "--offset", "0", "--label", label,
        ])
        report = json.loads(
            Path(f"data/eval/bas/benchmarks/events_{label}.json")
            .read_text(encoding="utf-8")
        )
        block["events"] = summarise_events(report)
        block.update(block["events"]["headline"])
    return block


def summarise_events(report: dict) -> dict:
    """Pull the Part 6 event numbers out of the stored report.

    The headline tolerance is 0.40 s, matching every event figure previously
    published for this project. Anything the corpus cannot measure -- player
    identity, and therefore sender and receiver accuracy -- is reported as
    ``None`` with the reason, never as zero.
    """
    tolerance = 0.40
    per_type: dict[str, dict] = {}
    for event_type, entries in (report.get("per_event_type") or {}).items():
        row = next((e for e in entries if e.get("tolerance_s") == tolerance), None)
        if row is None:
            continue
        per_type[event_type] = {
            "precision": row["precision"]["value"],
            "recall": row["recall"]["value"],
            "f1": row["f1"],
            "n_ground_truth": row["n_ground_truth"],
            "n_predicted": row["n_predicted"],
            "median_temporal_error_s": row["temporal_error_s"]["median"],
        }

    def f1_of(name: str) -> float | None:
        return per_type.get(name, {}).get("f1")

    return {
        "tolerance_s": tolerance,
        "per_event_type": per_type,
        "headline": {
            "pass_precision": per_type.get("pass_start", {}).get("precision"),
            "pass_recall": per_type.get("pass_start", {}).get("recall"),
            "pass_f1": f1_of("pass_start"),
            "carry_precision": per_type.get("carry_start", {}).get("precision"),
            "carry_recall": per_type.get("carry_start", {}).get("recall"),
            "carry_f1": f1_of("carry_start"),
            "interception_f1": f1_of("interception"),
            "turnover_f1": f1_of("turnover"),
        },
        "event_coverage": {
            "n_predicted_total": sum(
                v["n_predicted"] for v in per_type.values()
            ),
            "n_ground_truth_total": sum(
                v["n_ground_truth"] for v in per_type.values()
            ),
        },
        "player_attribution": {
            "sender_accuracy": None,
            "receiver_accuracy": None,
            "measurable": False,
            "reason": (
                "SN-BAS carries no player identity labels, so sender and "
                "receiver accuracy have no ground truth on this corpus"
            ),
        },
        "ball_quality": report.get("ball_quality"),
    }


def check_reproduction(row: str, block: dict) -> dict:
    """Row A and row C must reproduce the stored legacy baseline exactly."""
    expected = LEGACY_BASELINE.get(row)
    if not expected:
        return {"checked": False}
    diffs: dict[str, float] = {}
    ok = True
    for video, metrics in expected.items():
        for key, value in metrics.items():
            actual = block.get(video, {}).get(key)
            if actual is None:
                ok = False
                diffs[f"{video}.{key}"] = None
                continue
            delta = round(actual - value, 6)
            diffs[f"{video}.{key}"] = delta
            if abs(delta) > REPRODUCTION_TOLERANCE:
                ok = False
    return {"checked": True, "reproduced": ok, "deltas": diffs,
            "tolerance": REPRODUCTION_TOLERANCE}


def run_one(name: str, settings: list[str], skip_local: bool,
            skip_bas: bool) -> dict:
    block: dict = {"overrides": settings}
    if not skip_local and LOCAL_VIDEO.exists():
        block["local"] = measure(
            LOCAL_VIDEO, Path("outputs_fusion_local"), settings, LOCAL_SEGMENT,
            f"fusion_local_{name}", None,
        )
    if not skip_bas and BAS_VIDEO.exists():
        block["bas"] = measure(
            BAS_VIDEO, Path("outputs_fusion_bas"), settings, BAS_SEGMENT,
            f"fusion_bas_{name}", BAS_GT,
        )
    return block


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="*", default=[], choices=list(ROWS))
    parser.add_argument("--ablation", nargs="*", default=None,
                        help="ablation row names, or empty for all")
    parser.add_argument("--skip-local", action="store_true")
    parser.add_argument("--skip-bas", action="store_true")
    args = parser.parse_args()

    configure_logging("INFO")

    if not THRESHOLD_ARTIFACT.exists():
        raise SystemExit(
            f"{THRESHOLD_ARTIFACT} does not exist. Thresholds are pinned before "
            "evaluation; run visionpitch.evaluation.fusion_thresholds."
            "write_artifact() first."
        )
    pinned = json.loads(THRESHOLD_ARTIFACT.read_text(encoding="utf-8"))
    if pinned.get("fingerprint") != spec_fingerprint():
        raise SystemExit(
            "the pinned threshold artifact no longer matches the specification "
            "in code -- refusing to score against moved goalposts"
        )
    log.info("thresholds pinned at %s; rows frozen at %s",
             pinned["fingerprint"], rows_fingerprint())

    if args.rows:
        record = load(RECORD)
        record.setdefault("meta", {})
        record["meta"].update({
            "threshold_fingerprint": pinned["fingerprint"],
            "rows_fingerprint": rows_fingerprint(),
            "local_segment_s": list(LOCAL_SEGMENT),
            "bas_segment_s": list(BAS_SEGMENT),
        })
        for row in args.rows:
            spec = ROWS[row]
            log.info("=== row %s: %s ===", row, spec["description"])
            block = run_one(spec["label"], overrides(spec),
                            args.skip_local, args.skip_bas)
            block["description"] = spec["description"]
            block["detector"] = spec["detector"]
            block["fusion"] = spec["fusion"]
            block["reference"] = spec["reference"]
            block["reproduction"] = check_reproduction(row, block)
            record[row] = block
            save(RECORD, record)
            log.info("row %s written", row)

    if args.ablation is not None:
        names = args.ablation or [
            k for k, v in ABLATION.items() if "same_as_row" not in v
        ]
        record = load(ABLATION_RECORD)
        record.setdefault("meta", {
            "detector": "candidate_c",
            "threshold_fingerprint": pinned["fingerprint"],
            "rows_fingerprint": rows_fingerprint(),
            "note": "rows 1 and 7 are byte-identical to four-way C and D and "
                    "are reused from production_eval.json rather than re-run",
        })
        from visionpitch.evaluation.fusion_rows import DETECTORS

        for name in names:
            spec = ABLATION[name]
            if "same_as_row" in spec:
                record[name] = {"same_as_row": spec["same_as_row"]}
                continue
            settings = [
                f"ball_detection.model_path={DETECTORS['candidate_c']['weights']}",
                "ball_detection.conf_threshold=0.12",
            ] + [f"ball_fusion.{k}={v}" for k, v in spec["fusion"].items()]
            log.info("=== ablation %s ===", name)
            block = run_one(f"abl_{name}", settings, args.skip_local, args.skip_bas)
            block["fusion"] = spec["fusion"]
            if "note" in spec:
                block["note"] = spec["note"]
            record[name] = block
            save(ABLATION_RECORD, record)
            log.info("ablation %s written", name)

    print(f"wrote {RECORD} / {ABLATION_RECORD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
