"""Run every ball candidate through the real pipeline on identical segments.

Requirement 10 of the broadcast-adaptation comparison. Benchmark wins have three
times in this project failed to survive contact with the pipeline, so no
candidate is judged on detection metrics alone.

Two videos, because they answer different questions:

* **the local broadcast clip** carries no event ground truth, so it gives direct
  ball coverage and possession determinability -- and nothing else honest
* **the SN-BAS segment** has expert action labels, so it is the only source for
  unchanged-engine pass and carry F1

Every candidate sees byte-identical segments and configuration; only the ball
checkpoint changes.

Usage::

    python scripts/broadcast_pipeline_test.py --candidate C_adapt \
        --weights models/finetune/bcast_adapt/weights/best.pt --conf 0.08
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionpitch.common.logging import configure_logging, get_logger  # noqa: E402

log = get_logger("broadcast.pipeline")

LOCAL_VIDEO = Path(
    "ملخص مباراة نيوزيلندا ومصر _ دور المجموعات - كأس العالم FIFA 2026™.mp4"
)
BAS_VIDEO = Path("data/eval/bas/mid_pre_720p.mp4")
BAS_GT = Path("data/eval/bas/event_gt_half1.json")
RECORD = Path("data/eval/broadcast/pipeline.json")

#: Fixed for every candidate. Chosen before any result was seen.
LOCAL_SEGMENT = (0.0, 120.0)
BAS_SEGMENT = (600.0, 780.0)


def run(command: list[str]) -> str:
    log.info("$ %s", " ".join(str(c) for c in command))
    result = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        log.error("failed:\n%s", (result.stdout or "")[-3000:] + (result.stderr or "")[-3000:])
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


def analyse(video: Path, output_root: Path, weights: str, conf: float,
            start: float, end: float) -> Path:
    command = [
        sys.executable, "-m", "visionpitch.cli", "analyse", str(video),
        "--output", str(output_root), "--start", str(start), "--end", str(end),
        "--no-render",
        "--set", f"ball_detection.model_path={weights}",
        "--set", f"ball_detection.conf_threshold={conf}",
    ]
    run(command)
    return newest_run(output_root)


def parse_events(text: str) -> dict:
    out: dict = {}
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("pass_start"):
            parts = s.split()
            if len(parts) >= 4:
                out["pass_precision"] = float(parts[1])
                out["pass_recall"] = float(parts[2])
                out["pass_f1"] = float(parts[3])
        elif s.startswith("carry_start"):
            parts = s.split()
            if len(parts) >= 4:
                out["carry_f1"] = float(parts[3])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--conf", type=float, required=True)
    parser.add_argument("--skip-bas", action="store_true")
    args = parser.parse_args()

    configure_logging("INFO")
    from visionpitch.analytics.runner import run_analytics

    block: dict = {
        "candidate": args.candidate,
        "weights": args.weights,
        "conf": args.conf,
        "local_segment_s": list(LOCAL_SEGMENT),
        "bas_segment_s": list(BAS_SEGMENT),
    }

    # -- local broadcast clip: coverage and determinability ------------------- #
    if LOCAL_VIDEO.exists():
        local_run = analyse(
            LOCAL_VIDEO, Path("outputs_local"), args.weights, args.conf, *LOCAL_SEGMENT
        )
        run_analytics(local_run)
        label = f"local_{args.candidate}"
        run([
            sys.executable, "scripts/possession_determinability.py",
            "--run", str(local_run), "--label", label,
        ])
        payload = json.loads(
            Path(f"data/eval/determinability/determinability_{label}.json")
            .read_text(encoding="utf-8")
        )
        block["local"] = {
            "run_dir": str(local_run),
            "ball_coverage_direct": payload["ball_coverage_direct"],
            "determinability": payload["determinability"],
            "unknown_ratio": payload["unknown_ratio"],
            "observable_fraction": payload["observable_fraction"],
            "note": (
                "the local clip has no event ground truth, so coverage and "
                "determinability are the only honest measurements here"
            ),
        }
    else:
        log.warning("local video not found at %s", LOCAL_VIDEO)

    # -- SN-BAS: unchanged event engine --------------------------------------- #
    if not args.skip_bas and BAS_VIDEO.exists():
        bas_run = analyse(
            BAS_VIDEO, Path("outputs_bas"), args.weights, args.conf, *BAS_SEGMENT
        )
        run_analytics(bas_run)
        label = f"bas_{args.candidate}"
        run([
            sys.executable, "scripts/possession_determinability.py",
            "--run", str(bas_run), "--label", label,
        ])
        determinability = json.loads(
            Path(f"data/eval/determinability/determinability_{label}.json")
            .read_text(encoding="utf-8")
        )
        events = parse_events(run([
            sys.executable, "scripts/evaluate_events.py", "--run", str(bas_run),
            "--gt", str(BAS_GT), "--offset", "0", "--label", label,
        ]))
        block["bas"] = {
            "run_dir": str(bas_run),
            "ball_coverage_direct": determinability["ball_coverage_direct"],
            "determinability": determinability["determinability"],
            **events,
        }

    RECORD.parent.mkdir(parents=True, exist_ok=True)
    record = json.loads(RECORD.read_text(encoding="utf-8")) if RECORD.exists() else {}
    record[args.candidate] = block
    RECORD.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(json.dumps(block, indent=2, ensure_ascii=False))
    print(f"\nwrote {RECORD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
