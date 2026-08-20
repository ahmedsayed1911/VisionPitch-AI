"""Audit completed human annotations for coverage and diversity.

Broadcast ball annotation workflow, post-review.

Reports three kinds of category, and keeps them separate because they have
different evidential status:

**Sampling categories** were assigned before review from frame-level signals
(shot type, camera motion, blur, player density). They describe why a frame was
chosen.

**Annotation outcomes** are what the reviewer decided: visible, not visible,
outside frame, ambiguous, ignored.

**Derived ball-relative categories** could not exist before annotation, because
they depend on where the ball actually is. Occlusion and aerial position are
computed here by running the person detector on the reviewed frames and
comparing the annotated ball against the player boxes. They are *derived*, not
labelled, and are reported as such.

The diversity verdict is a coverage question, not a taste question: a category
with too few frames cannot be split three ways, cannot be trained on, and cannot
carry a confidence interval narrow enough to mean anything. Those floors are
stated in the output rather than left implicit.

Usage::

    python scripts/audit_annotations.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionpitch.annotation.quality import build_queue, summarise  # noqa: E402
from visionpitch.annotation.schema import (  # noqa: E402
    AnnotationStore,
    BallVisibility,
)
from visionpitch.common.logging import configure_logging, get_logger  # noqa: E402

log = get_logger("annotation.audit")

#: Frames a category needs before it can be split 60/20/20 and still leave
#: enough in test to compute a rate. Below this the category exists but cannot
#: be *measured* separately.
MIN_FOR_SPLIT = 15
#: Frames needed in a category before a per-category test rate has a 95%
#: interval narrower than roughly +-0.20 -- the point at which a number starts
#: informing a decision rather than decorating a report.
MIN_FOR_EVALUATION = 25

#: A ball centre inside a player box, or within this many pixels of one, is
#: occluded or body-adjacent.
OCCLUSION_MARGIN_PX = 6.0


def derive_ball_relative(store: AnnotationStore, samples, annotations) -> dict:
    """Occlusion and aerial position, from player boxes on the reviewed frames.

    Requires the detector, but only for *characterising* annotated frames -- no
    model output becomes ground truth here.
    """
    import cv2

    from visionpitch.common.config import load_config
    from visionpitch.common.types import ObjectClass
    from visionpitch.detection.yolo import build_detector

    config = load_config()
    detector = build_detector(config)

    out: dict[str, dict] = {}
    visible = [
        frame_id for frame_id, a in annotations.items()
        if a.visibility is BallVisibility.VISIBLE and a.centre_x is not None
    ]
    log.info("deriving ball-relative categories on %d visible frame(s)", len(visible))

    for frame_id in visible:
        sample = samples[frame_id]
        annotation = annotations[frame_id]
        image = cv2.imread(sample.image_path)
        if image is None:
            continue
        detections = detector.detect_batch([image], [sample.frame_idx])[0]
        people = [
            d for d in detections
            if d.object_class in (
                ObjectClass.PLAYER, ObjectClass.GOALKEEPER, ObjectClass.REFEREE
            )
        ]
        bx, by = annotation.centre_x, annotation.centre_y

        inside = False
        nearest = float("inf")
        for person in people:
            box = person.bbox
            if box.x1 <= bx <= box.x2 and box.y1 <= by <= box.y2:
                inside = True
                nearest = 0.0
                break
            dx = max(box.x1 - bx, 0.0, bx - box.x2)
            dy = max(box.y1 - by, 0.0, by - box.y2)
            nearest = min(nearest, float(np.hypot(dx, dy)))

        # Aerial: the ball sits above the heads of the players around it. Uses
        # the median box top of nearby players so a distant far-side player does
        # not set the bar.
        nearby = [
            p for p in people
            if abs((p.bbox.x1 + p.bbox.x2) / 2 - bx) < 300
        ] or people
        aerial = False
        head_line = None
        if nearby:
            head_line = float(np.median([p.bbox.y1 for p in nearby]))
            aerial = by < head_line

        out[frame_id] = {
            "n_people": len(people),
            "inside_player_box": inside,
            "nearest_player_px": None if nearest == float("inf") else round(nearest, 1),
            "occluded_or_body_adjacent": bool(
                inside or nearest <= OCCLUSION_MARGIN_PX
            ),
            "aerial": bool(aerial),
            "head_line_y": head_line,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path("data/annotation/package"))
    parser.add_argument("--out", type=Path, default=Path("data/annotation/audit_report.json"))
    parser.add_argument("--no-derive", action="store_true",
                        help="skip the detector pass for ball-relative categories")
    args = parser.parse_args()

    configure_logging("INFO")
    store = AnnotationStore(args.package)
    samples = store.load_samples()
    annotations = store.load_annotations()
    predictions = store.load_predictions()
    manifest = store.manifest()

    if not annotations:
        log.error("no annotations in %s", args.package)
        return 1

    reviewed = {k: v for k, v in annotations.items() if k in samples}
    n = len(reviewed)

    # -- shots ----------------------------------------------------------------- #
    shots = Counter(samples[f].shot_index for f in reviewed)
    shot_types = Counter(samples[f].shot_type for f in reviewed)

    # -- outcomes -------------------------------------------------------------- #
    visibility = Counter(a.visibility.value for a in reviewed.values())
    ignored = Counter(
        a.ignore_reason.value for a in reviewed.values()
        if a.ignore_reason.value != "none"
    )
    scorable = [f for f, a in reviewed.items() if a.is_scorable]
    accepted = Counter(
        a.accepted_proposal_from for a in reviewed.values()
        if a.accepted_proposal_from
    )

    # -- sampling categories --------------------------------------------------- #
    categories = Counter(samples[f].sampling_category.value for f in reviewed)

    # -- ball size ------------------------------------------------------------- #
    radii = [
        a.radius_px for a in reviewed.values()
        if a.visibility is BallVisibility.VISIBLE and a.radius_px is not None
    ]

    # -- temporal windows ------------------------------------------------------ #
    windows: dict[str, int] = defaultdict(int)
    for frame_id in reviewed:
        if samples[frame_id].window_id:
            windows[samples[frame_id].window_id] += 1

    derived = {} if args.no_derive else derive_ball_relative(store, samples, reviewed)

    # -- assemble the reportable category table -------------------------------- #
    def count_where(predicate) -> int:
        return sum(1 for f in reviewed if predicate(f))

    table = {
        "wide_play": count_where(lambda f: samples[f].shot_type == "wide_play"),
        "medium_play": count_where(lambda f: samples[f].shot_type == "medium_play"),
        "midfield": categories.get("midfield_play", 0),
        "near_goal": categories.get("near_goal", 0),
        "motion_blur": categories.get("motion_blur", 0),
        "camera_pan": categories.get("camera_pan", 0),
        "fast_transition": categories.get("fast_transition", 0),
        "crowded_scene": categories.get("crowded_scene", 0),
        "low_contrast": categories.get("low_contrast", 0),
        "graphics": categories.get("broadcast_graphic", 0),
        "crowd_negative": categories.get("crowd_negative", 0),
        "temporal_window": categories.get("temporal_window", 0),
        "ball_not_visible": visibility.get("not_visible", 0),
        "ball_outside_frame": visibility.get("outside_frame", 0),
        "ambiguous": visibility.get("ambiguous", 0),
    }
    if derived:
        table["occlusion_derived"] = sum(
            1 for v in derived.values() if v["occluded_or_body_adjacent"]
        )
        table["aerial_derived"] = sum(1 for v in derived.values() if v["aerial"])

    # -- quality queue --------------------------------------------------------- #
    queue = build_queue(samples, reviewed, predictions)
    quality = summarise(queue, n)

    # -- gaps ------------------------------------------------------------------ #
    gaps = []
    for name, count in sorted(table.items()):
        if count >= MIN_FOR_EVALUATION:
            status = "sufficient"
        elif count >= MIN_FOR_SPLIT:
            status = "splittable_but_not_separately_measurable"
        elif count > 0:
            status = "present_but_too_few"
        else:
            status = "absent"
        gaps.append({
            "category": name, "n": count, "status": status,
            "needed_for_split": max(0, MIN_FOR_SPLIT - count),
            "needed_for_evaluation": max(0, MIN_FOR_EVALUATION - count),
        })

    payload = {
        "package": str(args.package),
        "source_content_hash": manifest.get("source_content_hash"),
        "sampling_fingerprint": (manifest.get("sampling") or {}).get(
            "sampling_fingerprint"
        ),
        "annotation_fingerprint": store.fingerprint(),
        "n_sampled": len(samples),
        "n_reviewed": n,
        "review_completion": round(n / len(samples), 4) if samples else 0.0,
        "n_unique_shots": len(shots),
        "n_shots_in_video": (manifest.get("audit_summary") or {}).get("n_shots"),
        "frames_per_shot": {
            "min": min(shots.values()), "median": int(np.median(list(shots.values()))),
            "max": max(shots.values()),
        },
        "shot_type_breakdown": dict(sorted(shot_types.items())),
        "visibility": dict(sorted(visibility.items())),
        "ignored": dict(sorted(ignored.items())),
        "n_scorable": len(scorable),
        "accepted_proposals": dict(sorted(accepted.items())),
        "accepted_proposal_share": round(sum(accepted.values()) / n, 4) if n else 0.0,
        "sampling_categories": dict(sorted(categories.items())),
        "category_table": dict(sorted(table.items())),
        "ball_radius_px": {
            "n": len(radii),
            "min": round(float(np.min(radii)), 2) if radii else None,
            "p25": round(float(np.percentile(radii, 25)), 2) if radii else None,
            "median": round(float(np.median(radii)), 2) if radii else None,
            "p75": round(float(np.percentile(radii, 75)), 2) if radii else None,
            "max": round(float(np.max(radii)), 2) if radii else None,
            "distinct_values": len(set(radii)),
        } if radii else {"n": 0},
        "temporal_windows": {
            "n_windows_touched": len(windows),
            "frames_per_window": dict(sorted(windows.items())),
            "n_complete_windows": sum(1 for v in windows.values() if v >= 7),
        },
        "derived_ball_relative": {
            "computed": bool(derived),
            "n_frames": len(derived),
            "occluded_or_body_adjacent": sum(
                1 for v in derived.values() if v["occluded_or_body_adjacent"]
            ),
            "aerial": sum(1 for v in derived.values() if v["aerial"]),
            "median_nearest_player_px": round(float(np.median([
                v["nearest_player_px"] for v in derived.values()
                if v["nearest_player_px"] is not None
            ])), 1) if derived else None,
            "note": (
                "derived by running the person detector on reviewed frames and "
                "comparing against the human ball position; no model output is "
                "treated as ground truth"
            ),
        },
        "quality_queue": {
            k: v for k, v in quality.items() if k != "queue"
        },
        "coverage_floors": {
            "min_for_split": MIN_FOR_SPLIT,
            "min_for_evaluation": MIN_FOR_EVALUATION,
            "rationale": (
                "below min_for_split a category cannot be divided 60/20/20 and "
                "still leave test frames; below min_for_evaluation a per-category "
                "test rate has a 95% interval wider than about +-0.20"
            ),
        },
        "gaps": gaps,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # -- print ----------------------------------------------------------------- #
    print(f"\nreviewed        : {n} of {len(samples)} sampled "
          f"({payload['review_completion']:.1%})")
    print(f"unique shots    : {len(shots)} of {payload['n_shots_in_video']} in the video")
    print(f"frames per shot : min {payload['frames_per_shot']['min']}, "
          f"median {payload['frames_per_shot']['median']}, "
          f"max {payload['frames_per_shot']['max']}")
    print(f"scorable        : {len(scorable)}")
    print(f"annotation fp   : {payload['annotation_fingerprint']}")

    print(f"\n{'outcome':<22}{'n':>6}")
    for k, v in payload["visibility"].items():
        print(f"  {k:<20}{v:>6}")
    for k, v in payload["ignored"].items():
        print(f"  {k:<20}{v:>6}")

    print(f"\n{'category':<34}{'n':>6}  status")
    for row in payload["gaps"]:
        print(f"  {row['category']:<32}{row['n']:>6}  {row['status']}")

    if radii:
        r = payload["ball_radius_px"]
        print(f"\nball radius px  : min {r['min']}, p25 {r['p25']}, median "
              f"{r['median']}, p75 {r['p75']}, max {r['max']} "
              f"({r['distinct_values']} distinct)")
    if derived:
        d = payload["derived_ball_relative"]
        print(f"derived         : {d['occluded_or_body_adjacent']} occluded/body-adjacent, "
              f"{d['aerial']} aerial, median nearest player "
              f"{d['median_nearest_player_px']} px")
    print(f"\nquality queue   : {quality['n_flagged']} flagged "
          f"({quality['share_flagged']:.1%}) {quality['by_flag']}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
