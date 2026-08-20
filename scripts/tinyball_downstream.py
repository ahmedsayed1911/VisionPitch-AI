"""Measure a ball representation through the real pipeline, end to end.

Cross-domain tiny-ball study, Part 9.

Phase 2D's central lesson was that a detector can win a benchmark and lose the
product: the multi-corpus checkpoint gained +0.232 cross-domain recall while
effective coverage, possession determinability and pass recall all fell on
unseen broadcast footage. So no representation is promoted here on detection
metrics. This runs the SN-BAS segment through the unchanged pipeline, unchanged
analytics and unchanged event engine, and records what actually came out.

SN-BAS has no ball annotations, so this measures **coverage and downstream
effect**, never ball recall. That limitation is inherent to the only broadcast
corpus available and is restated in the output.

Usage::

    python scripts/tinyball_downstream.py --representation heatmap \
        --checkpoint models/finetune/heatmap/best.pt --label heatmap
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionpitch.common.logging import configure_logging, get_logger  # noqa: E402

log = get_logger("tinyball.downstream")

VIDEO = Path("data/eval/bas/mid_pre_720p.mp4")
GROUND_TRUTH = Path("data/eval/bas/event_gt_half1.json")
OUTPUT_ROOT = Path("outputs_bas")
RECORD = Path("data/eval/tinyball/downstream.json")


def run(command: list[str]) -> str:
    """Run a child script and return its stdout.

    UTF-8 is forced explicitly. The CLI prints box-drawing characters in its
    summary tables, and on Windows the default console codepage is cp1252, which
    cannot decode them -- the reader thread dies, ``stdout`` comes back ``None``,
    and the failure surfaces far from its cause as an AttributeError.
    """
    log.info("$ %s", " ".join(command))
    result = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        log.error(
            "command failed:\n%s", (result.stdout or "")[-4000:] + (result.stderr or "")[-4000:]
        )
        raise SystemExit(result.returncode)
    return result.stdout or ""


def parse_events(text: str) -> dict:
    """Pull the metrics the study reports out of the event harness output."""
    out: dict = {}
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("pass_start"):
            parts = stripped.split()
            if len(parts) >= 4:
                out["pass_precision"] = float(parts[1])
                out["pass_recall"] = float(parts[2])
                out["pass_f1"] = float(parts[3])
        elif stripped.startswith("carry_start"):
            parts = stripped.split()
            if len(parts) >= 4:
                out["carry_f1"] = float(parts[3])
        elif "ball observed" in stripped:
            for token, key in (
                ("ball observed", "ball_observed_pct"),
                ("determinable", "possession_determinable_pct"),
            ):
                if token in stripped:
                    segment = stripped.split(token, 1)[1].strip()
                    value = segment.split("%")[0].split()[-1]
                    try:
                        out[key] = float(value)
                    except ValueError:
                        pass
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation", default="heatmap", choices=["box", "heatmap"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--label", required=True)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--start", type=float, default=600.0)
    parser.add_argument("--end", type=float, default=780.0)
    parser.add_argument("--skip-run", action="store_true",
                        help="reuse an existing run directory instead of re-running")
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    configure_logging("INFO")
    if not VIDEO.exists():
        log.error("no SN-BAS clip at %s", VIDEO)
        return 1

    run_dir = args.run_dir
    if not args.skip_run:
        overrides = [
            f"ball_detection.representation={args.representation}",
        ]
        if args.checkpoint:
            overrides.append(f"ball_detection.model_path={args.checkpoint}")
        if args.conf is not None:
            overrides.append(f"ball_detection.conf_threshold={args.conf}")

        command = [
            sys.executable, "-m", "visionpitch.cli", "analyse", str(VIDEO),
            "--output", str(OUTPUT_ROOT), "--start", str(args.start),
            "--end", str(args.end), "--no-render",
        ]
        for override in overrides:
            command += ["--set", override]
        run(command)

        # The run directory is found on disk, not parsed out of stdout. The
        # project path contains a space ("VisionPitch AI") and the CLI wraps its
        # summary table, so splitting a printed line on whitespace silently
        # produces a truncated path that exists nowhere.
        candidates = sorted(
            (p.parent for p in OUTPUT_ROOT.glob("*/*/game_state.parquet")),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            log.error("the run produced no game_state.parquet under %s", OUTPUT_ROOT)
            return 1
        run_dir = candidates[-1]

    if run_dir is None:
        log.error("--skip-run needs --run-dir")
        return 1
    log.info("run directory: %s", run_dir)

    # -- analytics, unchanged -------------------------------------------------- #
    from visionpitch.analytics.runner import run_analytics

    run_analytics(run_dir)

    # -- determinability ------------------------------------------------------- #
    run([
        sys.executable, "scripts/possession_determinability.py",
        "--run", str(run_dir), "--label", args.label,
    ])
    determinability = json.loads(
        Path(f"data/eval/determinability/determinability_{args.label}.json")
        .read_text(encoding="utf-8")
    )

    # -- events, engine unchanged ---------------------------------------------- #
    events_text = run([
        sys.executable, "scripts/evaluate_events.py", "--run", str(run_dir),
        "--gt", str(GROUND_TRUTH), "--offset", "0", "--label", args.label,
    ])
    events = parse_events(events_text)

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    block = {
        "label": args.label,
        "representation": args.representation,
        "run_dir": str(run_dir),
        "config_fingerprint": manifest.get("config_fingerprint"),
        "ball_model": (manifest.get("models") or {}).get("ball_detector", {}),
        "ball_coverage_direct": determinability["ball_coverage_direct"],
        "determinability": determinability["determinability"],
        "unknown_ratio": determinability["unknown_ratio"],
        "observable_fraction": determinability["observable_fraction"],
        **events,
        "note": (
            "SN-BAS carries no ball annotations, so coverage here is the share of "
            "frames with a ball position, not the share that are correct. Event "
            "metrics come from the unchanged event engine against SN-BAS action "
            "labels."
        ),
    }

    RECORD.parent.mkdir(parents=True, exist_ok=True)
    record = json.loads(RECORD.read_text(encoding="utf-8")) if RECORD.exists() else {}
    record[args.label] = block
    RECORD.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(json.dumps(block, indent=2))
    print(f"\nwrote {RECORD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
