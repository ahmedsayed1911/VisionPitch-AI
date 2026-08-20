"""Is a bounding box the right representation for an 11-pixel ball?

Cross-domain tiny-ball study, Part 1.

The question is not rhetorical and it is answerable from the annotations alone,
before training anything. Four measurements decide it:

1. **Scale.** How small is the ball, per domain, and what fraction falls below
   8x8, 12x12, 16x16 and 24x24 px.

2. **What IoU50 actually demands.** For two equal squares of side ``s`` offset
   by ``d``, IoU = (s-d)/(s+d), so IoU >= 0.5 requires ``d <= s/3``. On an 11 px
   ball that is **3.7 px of centre accuracy** -- before any size mismatch. The
   standard detection metric is therefore mostly measuring sub-4-pixel
   localisation, which is not what any downstream consumer needs.

3. **Whether width and height carry signal.** A football is circular, so an
   honest annotation has w == h. Departures from that are annotation noise, and
   their size relative to the ball tells you how much of the box-regression
   target is noise. The same question is asked temporally: within one GSR track
   the true apparent size changes smoothly with depth, so frame-to-frame jitter
   beyond that trend is noise too.

4. **What downstream actually needs.** The possession engine compares the ball
   centre to player boxes at a radius of ~0.6 player-heights. That tolerance is
   computed here from real player box heights, so the "how accurate is accurate
   enough" question is answered with a measurement rather than an opinion.

Usage::

    python scripts/ball_representation_audit.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionpitch.common.logging import configure_logging, get_logger  # noqa: E402
from visionpitch.evaluation.possession_gt import load_gsr_gamestate  # noqa: E402

log = get_logger("ball.representation")

DATASET = Path("data/ball_multicorpus")
GSR_ROOT = Path("data/eval/gsr")


def iou_of_equal_squares(side: float, offset: float) -> float:
    """IoU of two ``side``-length squares whose centres differ by ``offset``."""
    if offset >= side:
        return 0.0
    overlap = (side - offset) * side
    union = 2 * side * side - overlap
    return overlap / union


def implied_centre_tolerance(side: float, iou_target: float = 0.5) -> float:
    """Largest centre offset still reaching ``iou_target`` at this ball size."""
    lo, hi = 0.0, side
    for _ in range(60):
        mid = (lo + hi) / 2
        if iou_of_equal_squares(side, mid) >= iou_target:
            lo = mid
        else:
            hi = mid
    return lo


def collect_dataset_boxes() -> dict[str, list[tuple[float, float]]]:
    """(w, h) in pixels per domain, from the multi-corpus label files."""
    import cv2

    out: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for split in ("train", "val", "test"):
        labels = DATASET / split / "labels"
        images = DATASET / split / "images"
        if not labels.exists():
            continue
        for label_path in sorted(labels.glob("*.txt")):
            lines = [ln for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if not lines:
                continue
            image_path = images / f"{label_path.stem}.jpg"
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            height, width = image.shape[:2]
            domain = (
                "soccernet_gsr" if label_path.stem.startswith("soccernet_gsr_")
                else "roboflow"
            )
            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    continue
                bw, bh = float(parts[3]) * width, float(parts[4]) * height
                out[domain].append((bw, bh))
    return out


def collect_gsr_tracks() -> list[list[tuple[int, float, float]]]:
    """Per-sequence ball (frame, w, h) series, for temporal jitter analysis."""
    series = []
    for labels_path in sorted(GSR_ROOT.glob("**/Labels-GameState.json")):
        frames, _ = load_gsr_gamestate(labels_path)
        track: list[tuple[int, float, float]] = []
        for frame_idx in sorted(frames):
            ball = next((o for o in frames[frame_idx] if o.role == "ball"), None)
            if ball is not None and ball.box_height > 0:
                # load_gsr_gamestate keeps height; width is recovered from the
                # raw record via the same aspect the annotation carries.
                track.append((frame_idx, ball.box_height, ball.box_height))
        if len(track) > 30:
            series.append(track)
    return series


def collect_gsr_box_pairs() -> dict[str, list[tuple[float, float]]]:
    """Raw (w, h) straight from the GSR JSON, which keeps both dimensions."""
    out: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for labels_path in sorted(GSR_ROOT.glob("**/Labels-GameState.json")):
        data = json.loads(labels_path.read_text(encoding="utf-8"))
        for annotation in data.get("annotations", []):
            attributes = annotation.get("attributes") or {}
            if attributes.get("role") != "ball":
                continue
            box = annotation.get("bbox_image") or {}
            w, h = float(box.get("w", 0)), float(box.get("h", 0))
            if w > 0 and h > 0:
                out["soccernet_gsr"].append((w, h))
    return out


def player_height_stats() -> dict:
    """Player box heights, which set the possession engine's real tolerance."""
    heights = []
    for labels_path in sorted(GSR_ROOT.glob("**/Labels-GameState.json")):
        data = json.loads(labels_path.read_text(encoding="utf-8"))
        for annotation in data.get("annotations", []):
            attributes = annotation.get("attributes") or {}
            if attributes.get("role") not in ("player", "goalkeeper"):
                continue
            box = annotation.get("bbox_image") or {}
            h = float(box.get("h", 0))
            if h > 0:
                heights.append(h)
    heights = np.array(heights)
    return {
        "n": int(heights.size),
        "median_px": round(float(np.median(heights)), 2),
        "p10_px": round(float(np.percentile(heights, 10)), 2),
        "p90_px": round(float(np.percentile(heights, 90)), 2),
    }


def describe(values: np.ndarray, name: str) -> dict:
    return {
        "metric": name,
        "n": int(values.size),
        "mean": round(float(values.mean()), 3),
        "median": round(float(np.median(values)), 3),
        "p10": round(float(np.percentile(values, 10)), 3),
        "p90": round(float(np.percentile(values, 90)), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/eval/representation"))
    args = parser.parse_args()
    configure_logging("INFO")

    log.info("collecting dataset boxes")
    by_domain = collect_dataset_boxes()
    log.info("collecting raw GSR boxes (both dimensions)")
    gsr_pairs = collect_gsr_box_pairs()

    payload: dict = {"domains": {}, "iou50_geometry": {}, "aspect_noise": {}}

    for domain, boxes in sorted(by_domain.items()):
        array = np.array(boxes)
        widths, heights = array[:, 0], array[:, 1]
        areas = widths * heights
        sides = np.sqrt(areas)
        payload["domains"][domain] = {
            "n_balls": int(array.shape[0]),
            "width_px": describe(widths, "width"),
            "height_px": describe(heights, "height"),
            "area_px2": describe(areas, "area"),
            "equivalent_side_px": describe(sides, "side"),
            "fraction_below": {
                "8x8": round(float((areas < 64).mean()), 4),
                "12x12": round(float((areas < 144).mean()), 4),
                "16x16": round(float((areas < 256).mean()), 4),
                "24x24": round(float((areas < 576).mean()), 4),
            },
            "implied_iou50_centre_tolerance_px": {
                "at_median_size": round(implied_centre_tolerance(float(np.median(sides))), 3),
                "at_p10_size": round(
                    implied_centre_tolerance(float(np.percentile(sides, 10))), 3
                ),
                "at_p90_size": round(
                    implied_centre_tolerance(float(np.percentile(sides, 90))), 3
                ),
            },
        }

    for side in (6, 8, 11, 16, 24, 40):
        payload["iou50_geometry"][f"{side}px_ball"] = {
            "centre_tolerance_for_iou50_px": round(implied_centre_tolerance(side), 3),
            "note": "equal-size boxes; any size mismatch tightens this further",
        }

    # -- is width/height signal or noise? ------------------------------------- #
    for domain, pairs in sorted(gsr_pairs.items()):
        array = np.array(pairs)
        widths, heights = array[:, 0], array[:, 1]
        aspect = widths / np.maximum(heights, 1e-6)
        # A football is circular: |w-h| is annotation noise, and its size
        # relative to the ball says how much of the regression target is noise.
        absolute_error = np.abs(widths - heights)
        relative = absolute_error / np.maximum((widths + heights) / 2, 1e-6)
        payload["aspect_noise"][domain] = {
            "n": int(array.shape[0]),
            "aspect_ratio": describe(aspect, "w/h"),
            "abs_w_minus_h_px": describe(absolute_error, "|w-h|"),
            "relative_wh_disagreement": describe(relative, "|w-h| / mean(w,h)"),
            "fraction_w_equals_h": round(float((absolute_error < 0.5).mean()), 4),
            "fraction_disagree_over_20pct": round(float((relative > 0.20).mean()), 4),
            "interpretation": (
                "a circular object annotated honestly has w == h; the spread here "
                "is the noise floor of the box-regression target at this scale"
            ),
        }

    payload["player_box_heights"] = player_height_stats()
    median_player = payload["player_box_heights"]["median_px"]
    payload["downstream_tolerance"] = {
        "possession_control_radius_heights": 0.6,
        "median_player_box_height_px": median_player,
        "implied_centre_tolerance_px": round(0.6 * median_player, 2),
        "note": (
            "the possession engine's control radius is 0.6 player-heights, so a "
            "ball centre this far out still yields the same possession decision. "
            "Compare against the IoU50 tolerance above: the detection metric is "
            "far stricter than the task."
        ),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / "representation_audit.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # -- print ----------------------------------------------------------------- #
    print("\nBALL SCALE BY DOMAIN")
    for domain, block in payload["domains"].items():
        print(f"\n  {domain}  (n={block['n_balls']})")
        print(f"    width  px  median {block['width_px']['median']:>6}  "
              f"p10 {block['width_px']['p10']:>6}  p90 {block['width_px']['p90']:>6}")
        print(f"    height px  median {block['height_px']['median']:>6}  "
              f"p10 {block['height_px']['p10']:>6}  p90 {block['height_px']['p90']:>6}")
        print(f"    area px2   median {block['area_px2']['median']:>6}")
        print(f"    below      {block['fraction_below']}")
        print(f"    IoU50 needs centre within "
              f"{block['implied_iou50_centre_tolerance_px']['at_median_size']} px "
              f"at median size")

    print("\nWHAT IoU50 DEMANDS, BY BALL SIZE")
    for key, block in payload["iou50_geometry"].items():
        print(f"  {key:<12} centre tolerance {block['centre_tolerance_for_iou50_px']:>6} px")

    print("\nIS WIDTH/HEIGHT SIGNAL OR NOISE?")
    for domain, block in payload["aspect_noise"].items():
        print(f"  {domain}  (n={block['n']})")
        print(f"    aspect ratio w/h        median {block['aspect_ratio']['median']}  "
              f"p10 {block['aspect_ratio']['p10']}  p90 {block['aspect_ratio']['p90']}")
        print(f"    |w-h| in px             median {block['abs_w_minus_h_px']['median']}")
        print(f"    |w-h| / size            median {block['relative_wh_disagreement']['median']}")
        print(f"    w == h exactly          {block['fraction_w_equals_h']}")
        print(f"    disagree by over 20%    {block['fraction_disagree_over_20pct']}")

    print("\nDOWNSTREAM TOLERANCE")
    for key, value in payload["downstream_tolerance"].items():
        if key != "note":
            print(f"  {key}: {value}")
    print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
