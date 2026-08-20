"""Audit the available corpora for the properties broadcast adaptation needs.

Reports, per dataset and split: positives, explicit negatives, ball scale, blur
and contrast at the ball, and occlusion. Everything is measured from the files
rather than assumed from the corpus description, because two of the properties
that matter most here -- explicit no-ball negatives and genuine occlusion -- are
not mentioned in any dataset's documentation.

Usage::

    python scripts/audit_training_corpora.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionpitch.common.logging import configure_logging, get_logger  # noqa: E402

log = get_logger("corpora.audit")

MULTICORPUS = Path("data/ball_multicorpus")
GSR_ROOT = Path("data/eval/gsr")
LOCAL_PACKAGE = Path("data/annotation/package")

TINY_AREA = 150.0
SMALL_AREA = 400.0
#: Laplacian variance over the ball patch, below which it reads as blurred.
BLUR_CUT = 40.0
#: Michelson contrast between ball and surrounding ring.
CONTRAST_CUT = 0.15


def domain_of(path: Path) -> str:
    return "soccernet_gsr" if path.name.startswith("soccernet_gsr_") else "roboflow"


def patch_stats(image, cx, cy, side):
    """Blur variance and local contrast at a ball location."""
    h, w = image.shape[:2]
    r = max(4, int(round(side)))
    x1, y1 = int(max(0, cx - r)), int(max(0, cy - r))
    x2, y2 = int(min(w, cx + r)), int(min(h, cy + r))
    patch = image[y1:y2, x1:x2]
    if patch.size == 0:
        return None, None
    grey = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(grey, cv2.CV_64F).var())

    inner = max(1, int(round(side / 2)))
    gy, gx = grey.shape[0] // 2, grey.shape[1] // 2
    core = grey[max(0, gy - inner): gy + inner + 1, max(0, gx - inner): gx + inner + 1]
    ring = np.ones(grey.shape, dtype=bool)
    ring[max(0, gy - inner): gy + inner + 1, max(0, gx - inner): gx + inner + 1] = False
    cm = float(core.mean()) if core.size else 0.0
    rm = float(grey[ring].mean()) if ring.any() else 0.0
    contrast = abs(cm - rm) / (cm + rm) if (cm + rm) > 0 else 0.0
    return blur, contrast


def audit_multicorpus(max_per_split: int | None) -> dict:
    out: dict = {}
    for split in ("train", "val", "test"):
        images = sorted((MULTICORPUS / split / "images").glob("*.jpg"))
        if max_per_split:
            images = images[:max_per_split]
        stats: dict[str, dict] = defaultdict(lambda: {
            "n_images": 0, "n_positive": 0, "n_negative": 0, "n_balls": 0,
            "areas": [], "blurs": [], "contrasts": [],
        })
        for path in images:
            domain = domain_of(path)
            entry = stats[domain]
            entry["n_images"] += 1
            label = path.parent.parent / "labels" / f"{path.stem}.txt"
            lines = [
                ln for ln in label.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ] if label.exists() else []
            if not lines:
                entry["n_negative"] += 1
                continue
            entry["n_positive"] += 1
            image = cv2.imread(str(path))
            if image is None:
                continue
            h, w = image.shape[:2]
            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    continue
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
                cx, cy, bw, bh = cx * w, cy * h, bw * w, bh * h
                entry["n_balls"] += 1
                entry["areas"].append(bw * bh)
                blur, contrast = patch_stats(image, cx, cy, max(bw, bh))
                if blur is not None:
                    entry["blurs"].append(blur)
                    entry["contrasts"].append(contrast)
        out[split] = {k: dict(v) for k, v in stats.items()}
    return out


def audit_gsr_occlusion(max_sequences: int) -> dict:
    """How often the annotated ball sits inside an annotated player box."""
    from visionpitch.evaluation.possession_gt import load_gsr_gamestate

    total = occluded = near = 0
    for labels_path in sorted(GSR_ROOT.glob("**/Labels-GameState.json"))[:max_sequences]:
        frames, _ = load_gsr_gamestate(labels_path)
        for objects in frames.values():
            ball = next((o for o in objects if o.role == "ball"), None)
            if ball is None:
                continue
            bx = ball.image_x
            by = ball.image_y - ball.box_height / 2
            people = [o for o in objects if o.role in ("player", "goalkeeper")]
            if not people:
                continue
            total += 1
            inside = False
            nearest = float("inf")
            for person in people:
                x1 = person.image_x - 0  # image_x is the box centre
                # reconstruct the box from the stored centre/height pair
                half_w = person.box_height * 0.35
                px1, px2 = person.image_x - half_w, person.image_x + half_w
                py1, py2 = person.image_y - person.box_height, person.image_y
                if px1 <= bx <= px2 and py1 <= by <= py2:
                    inside = True
                    nearest = 0.0
                    break
                dx = max(px1 - bx, 0.0, bx - px2)
                dy = max(py1 - by, 0.0, by - py2)
                nearest = min(nearest, float(np.hypot(dx, dy)))
                _ = x1
            occluded += int(inside)
            near += int(nearest <= 10.0)
    return {
        "n_frames_with_ball_and_players": total,
        "ball_inside_a_player_box": occluded,
        "ball_within_10px_of_a_player_box": near,
        "occlusion_rate": round(occluded / total, 4) if total else 0.0,
        "body_adjacent_rate": round(near / total, 4) if total else 0.0,
        "note": (
            "player boxes are reconstructed from the stored centre and height, "
            "so this is an approximation of the annotated box"
        ),
    }


def audit_local() -> dict:
    from visionpitch.annotation.schema import AnnotationStore, BallVisibility

    store = AnnotationStore(LOCAL_PACKAGE)
    samples = store.load_samples()
    annotations = store.load_annotations()
    radii = [
        a.radius_px for a in annotations.values()
        if a.visibility is BallVisibility.VISIBLE and a.radius_px is not None
    ]
    areas = [(2 * r) ** 2 for r in radii]
    shots = {samples[f].shot_index for f in annotations if f in samples}
    return {
        "n_reviewed": len(annotations),
        "n_positive": sum(
            1 for a in annotations.values() if a.visibility is BallVisibility.VISIBLE
        ),
        "n_negative": sum(
            1 for a in annotations.values()
            if a.visibility in (
                BallVisibility.NOT_VISIBLE, BallVisibility.OUTSIDE_FRAME
            ) or a.ignore_reason.excludes_from_scoring
        ),
        "n_shots": len(shots),
        "ball_area_px2": {
            "median": round(float(np.median(areas)), 1) if areas else None,
            "p10": round(float(np.percentile(areas, 10)), 1) if areas else None,
            "p90": round(float(np.percentile(areas, 90)), 1) if areas else None,
            "share_tiny_lt150": round(
                float(np.mean([a < TINY_AREA for a in areas])), 4
            ) if areas else None,
        },
    }


def summarise(entry: dict) -> dict:
    areas = np.array(entry["areas"]) if entry["areas"] else np.array([])
    blurs = np.array(entry["blurs"]) if entry["blurs"] else np.array([])
    contrasts = np.array(entry["contrasts"]) if entry["contrasts"] else np.array([])
    return {
        "n_images": entry["n_images"],
        "n_positive": entry["n_positive"],
        "n_negative_no_ball": entry["n_negative"],
        "n_balls": entry["n_balls"],
        "negative_share": round(
            entry["n_negative"] / max(1, entry["n_images"]), 4
        ),
        "ball_area_px2": {
            "median": round(float(np.median(areas)), 1) if areas.size else None,
            "share_tiny_lt150": round(float((areas < TINY_AREA).mean()), 4)
            if areas.size else None,
            "share_small_lt400": round(float((areas < SMALL_AREA).mean()), 4)
            if areas.size else None,
        },
        "share_blurred": round(float((blurs < BLUR_CUT).mean()), 4)
        if blurs.size else None,
        "share_low_contrast": round(float((contrasts < CONTRAST_CUT).mean()), 4)
        if contrasts.size else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-per-split", type=int, default=None)
    parser.add_argument("--gsr-sequences", type=int, default=12)
    parser.add_argument("--out", type=Path,
                        default=Path("data/eval/corpora_audit.json"))
    args = parser.parse_args()

    configure_logging("INFO")
    log.info("auditing %s", MULTICORPUS)
    multicorpus = audit_multicorpus(args.max_per_split)
    log.info("auditing GSR occlusion over %d sequence(s)", args.gsr_sequences)
    occlusion = audit_gsr_occlusion(args.gsr_sequences)
    local = audit_local()

    payload = {
        "multicorpus": {
            split: {domain: summarise(entry) for domain, entry in domains.items()}
            for split, domains in multicorpus.items()
        },
        "gsr_occlusion": occlusion,
        "local_broadcast": local,
        "thresholds": {
            "tiny_area_px2": TINY_AREA, "small_area_px2": SMALL_AREA,
            "blur_variance_cut": BLUR_CUT, "contrast_cut": CONTRAST_CUT,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n{'split':<8}{'domain':<16}{'images':>8}{'pos':>7}{'neg':>7}"
          f"{'tiny':>8}{'blur':>8}{'lowcon':>8}")
    for split, domains in payload["multicorpus"].items():
        for domain, s in sorted(domains.items()):
            print(f"{split:<8}{domain:<16}{s['n_images']:>8}{s['n_positive']:>7}"
                  f"{s['n_negative_no_ball']:>7}"
                  f"{str(s['ball_area_px2']['share_tiny_lt150']):>8}"
                  f"{str(s['share_blurred']):>8}{str(s['share_low_contrast']):>8}")

    print(f"\nGSR occlusion: {occlusion['ball_inside_a_player_box']} of "
          f"{occlusion['n_frames_with_ball_and_players']} frames "
          f"({occlusion['occlusion_rate']:.1%} inside a player box, "
          f"{occlusion['body_adjacent_rate']:.1%} within 10 px)")
    print(f"\nlocal broadcast: {local['n_reviewed']} reviewed, "
          f"{local['n_positive']} positive, {local['n_negative']} negative, "
          f"{local['n_shots']} shots")
    print(f"  ball area px2 median {local['ball_area_px2']['median']}, "
          f"tiny share {local['ball_area_px2']['share_tiny_lt150']}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
