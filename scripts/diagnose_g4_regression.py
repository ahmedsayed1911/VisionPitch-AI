"""Why controlled-possession determinability fell when the ball got better.

Phase 2E, Part 7. On the held-out SN-BAS broadcast the candidate ball detector
improved every ball metric and both ground-truthed event metrics, yet gate G4 --
``controlled_s / total_s`` on smoothed spans -- regressed 0.1196 -> 0.1026. A
gate is not something to reinterpret after seeing the result, so this measures
the mechanism instead of arguing about it.

Method
------
Both completed runs are replayed through the unchanged possession engine, and
every frame's decision is attributed to the specific branch of ``_decide`` that
produced it. The branch is recovered from the fields the decision already
carries -- state, nearest distance, ball speed -- rather than by reimplementing
the rule, so this cannot drift from the engine it is describing:

``no_ball``            ball missing or not trustworthy -> UNKNOWN
``out_of_play``        sustained out-of-play call
``no_players``         ball located, nobody tracked in frame
``travelling``         ball speed above the travelling threshold -> LOOSE
``beyond_loose``       nearest player past the loose radius -> LOOSE
``between_radii``      inside loose radius, outside control radius -> LOOSE
``contested``          two opponents both in contact range
``controlled``         nearest player inside the control radius

Then the two runs are cross-tabulated **frame by frame** over the frames they
share, which localises exactly where controlled time went, and the span layer is
compared separately because ``spans()`` keys on ``(state, team, track_id)`` and
``_smooth`` deletes spans shorter than ``min_span_s`` -- so a detector that
switches holder more often can lose controlled time without a single frame
changing its state.

Usage::

    python scripts/diagnose_g4_regression.py --a outputs_phase2e/prod/... \\
        --b outputs_phase2e/cand/...
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionpitch.analytics.context import load_context  # noqa: E402
from visionpitch.analytics.possession import (  # noqa: E402
    PossessionConfig,
    PossessionEngine,
)
from visionpitch.analytics.types import PossessionState  # noqa: E402
from visionpitch.common.logging import configure_logging, get_logger  # noqa: E402

log = get_logger("phase2e.g4")

RECORD = Path("data/eval/fusion/g4_regression_diagnosis.json")


def branch_of(decision, config: PossessionConfig) -> str:
    """Which rule produced this frame's state."""
    state = decision.state
    if state is PossessionState.UNKNOWN:
        return "no_ball"
    if state is PossessionState.OUT_OF_PLAY:
        return "out_of_play"
    if state is PossessionState.CONTESTED:
        return "contested"
    if state is PossessionState.CONTROLLED:
        return "controlled"
    # Everything left is LOOSE_BALL; separate the three ways to get there.
    nearest = decision.nearest_distance_heights
    if nearest is None:
        return "no_players"
    speed = decision.ball_speed_heights_s
    if speed is not None and speed > config.travelling_speed_heights_s:
        return "travelling"
    if nearest > config.loose_radius_heights:
        return "beyond_loose"
    return "between_radii"


def analyse(run_dir: Path, config: PossessionConfig) -> dict:
    context = load_context(run_dir)
    engine = PossessionEngine(context, config)
    decisions = engine.per_frame()
    spans = engine.spans(decisions)

    branches = Counter()
    per_frame: dict[int, dict] = {}
    nearest_values: list[float] = []
    speed_values: list[float] = []
    for decision in decisions:
        branch = branch_of(decision, config)
        branches[branch] += 1
        per_frame[decision.frame_idx] = {
            "branch": branch,
            "state": decision.state.value,
            "team": decision.team_id,
            "track": decision.track_id,
            "nearest": decision.nearest_distance_heights,
            "speed": decision.ball_speed_heights_s,
            "ball_known": decision.ball_state.is_known,
        }
        if decision.nearest_distance_heights is not None:
            nearest_values.append(decision.nearest_distance_heights)
        if decision.ball_speed_heights_s is not None:
            speed_values.append(decision.ball_speed_heights_s)

    controlled_spans = [s for s in spans if s.state is PossessionState.CONTROLLED]
    durations = [s.end_time_s - s.start_time_s for s in controlled_spans]

    # Raw (pre-smoothing) controlled runs, to separate "the engine changed its
    # mind" from "the smoother deleted it".
    raw_runs: list[int] = []
    run = 0
    previous_key = None
    for decision in decisions:
        key = (decision.state, decision.team_id, decision.track_id)
        if decision.state is PossessionState.CONTROLLED:
            if key == previous_key:
                run += 1
            else:
                if run:
                    raw_runs.append(run)
                run = 1
        else:
            if run:
                raw_runs.append(run)
            run = 0
        previous_key = key
    if run:
        raw_runs.append(run)

    fps = context.fps
    total_s = len(decisions) / fps if fps else 0.0
    return {
        "run_dir": str(run_dir),
        "n_frames": len(decisions),
        "fps": fps,
        "branches": dict(sorted(branches.items())),
        "branch_share": {
            k: round(v / max(1, len(decisions)), 4) for k, v in sorted(branches.items())
        },
        "controlled_frames": branches["controlled"],
        "controlled_frame_share": round(
            branches["controlled"] / max(1, len(decisions)), 4
        ),
        "smoothed_controlled_spans": len(controlled_spans),
        "smoothed_controlled_s": round(float(np.sum(durations)) if durations else 0.0, 2),
        "smoothed_controlled_share": round(
            (float(np.sum(durations)) if durations else 0.0) / max(1e-9, total_s), 4
        ),
        "median_controlled_span_s": round(
            float(np.median(durations)) if durations else 0.0, 3
        ),
        "raw_controlled_runs": len(raw_runs),
        "median_raw_run_frames": float(np.median(raw_runs)) if raw_runs else 0.0,
        "raw_runs_under_min_span": sum(
            1 for r in raw_runs if r / max(1e-9, fps) < config.min_span_s
        ),
        "nearest_distance_heights": {
            "median": round(float(np.median(nearest_values)), 3) if nearest_values else None,
            "p25": round(float(np.percentile(nearest_values, 25)), 3) if nearest_values else None,
            "share_within_control_radius": round(
                float(np.mean([v <= config.control_radius_heights for v in nearest_values])), 4
            ) if nearest_values else None,
            "n": len(nearest_values),
        },
        "ball_speed_heights_s": {
            "median": round(float(np.median(speed_values)), 3) if speed_values else None,
            "p90": round(float(np.percentile(speed_values, 90)), 3) if speed_values else None,
            "share_above_travelling": round(
                float(np.mean([v > config.travelling_speed_heights_s for v in speed_values])), 4
            ) if speed_values else None,
            "n": len(speed_values),
        },
        "_per_frame": per_frame,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True, help="baseline run dir")
    parser.add_argument("--b", type=Path, required=True, help="candidate run dir")
    parser.add_argument("--label-a", default="production")
    parser.add_argument("--label-b", default="candidate")
    parser.add_argument("--out", type=Path, default=RECORD)
    args = parser.parse_args()

    configure_logging("WARNING")
    config = PossessionConfig()

    a = analyse(args.a, config)
    b = analyse(args.b, config)
    frames_a = a.pop("_per_frame")
    frames_b = b.pop("_per_frame")

    shared = sorted(set(frames_a) & set(frames_b))
    transitions = Counter()
    controlled_lost_to = Counter()
    controlled_gained_from = Counter()
    holder_changed = 0
    controlled_both = 0
    for frame_idx in shared:
        left, right = frames_a[frame_idx], frames_b[frame_idx]
        transitions[(left["branch"], right["branch"])] += 1
        if left["branch"] == "controlled" and right["branch"] != "controlled":
            controlled_lost_to[right["branch"]] += 1
        if right["branch"] == "controlled" and left["branch"] != "controlled":
            controlled_gained_from[left["branch"]] += 1
        if left["branch"] == "controlled" and right["branch"] == "controlled":
            controlled_both += 1
            if left["track"] != right["track"]:
                holder_changed += 1

    net = a["controlled_frames"] - b["controlled_frames"]
    payload = {
        "schema_version": "1.0.0",
        "what_this_is": (
            "frame-level attribution of the G4 controlled-possession regression "
            "between two completed pipeline runs on the same footage"
        ),
        "gate": {
            "name": "G4_determinability_ratio",
            "definition": "controlled_s / total_s on smoothed spans",
            "min_ratio_vs_baseline": 0.95,
        },
        args.label_a: a,
        args.label_b: b,
        "frames_compared": len(shared),
        "controlled_frames_lost_net": net,
        "controlled_frames_in_both": controlled_both,
        "holder_id_changed_while_both_controlled": holder_changed,
        "controlled_lost_to_branch": dict(controlled_lost_to.most_common()),
        "controlled_gained_from_branch": dict(controlled_gained_from.most_common()),
        "top_branch_transitions": {
            f"{k[0]} -> {k[1]}": v
            for k, v in transitions.most_common(15)
            if k[0] != k[1]
        },
        "unchanged_frames": sum(v for k, v in transitions.items() if k[0] == k[1]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"frames compared: {len(shared)}")
    print(f"\n{'branch':<16}{args.label_a:>14}{args.label_b:>14}{'delta':>10}")
    for branch in sorted(set(a["branches"]) | set(b["branches"])):
        left = a["branches"].get(branch, 0)
        right = b["branches"].get(branch, 0)
        print(f"{branch:<16}{left:>14}{right:>14}{right - left:>+10}")
    print(f"\ncontrolled frames lost (net): {net}")
    print(f"controlled -> other:  {dict(controlled_lost_to.most_common())}")
    print(f"other -> controlled:  {dict(controlled_gained_from.most_common())}")
    print(f"holder id changed while both controlled: {holder_changed}"
          f" of {controlled_both}")
    print(f"\nsmoothed controlled_s: {a['smoothed_controlled_s']} -> "
          f"{b['smoothed_controlled_s']}")
    print(f"smoothed controlled spans: {a['smoothed_controlled_spans']} -> "
          f"{b['smoothed_controlled_spans']}")
    print(f"raw controlled runs: {a['raw_controlled_runs']} -> "
          f"{b['raw_controlled_runs']}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
