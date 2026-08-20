"""Per-domain ball detector comparison and the Phase 2C promotion rule.

Phase 2C, Part 2.

Every candidate is scored on the **held-out test split of every domain
separately**, never on a pooled average. A pooled number hides the failure this
phase exists to fix: Phase 2B's SN-GSR fine-tune improved the pooled figure
while regressing on a third corpus.

The promotion rule, applied mechanically:

    A candidate replaces the default only if it improves cross-domain recall,
    precision and mAP50, **and** regresses on no single domain by more than
    ``--max-regression``.

Usage::

    python scripts/evaluate_ball_domains.py --out data/eval/ball_domains
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionpitch.common.logging import configure_logging, get_logger  # noqa: E402

log = get_logger("ball.domains")

DATASET = Path("data/ball_multicorpus")

CANDIDATES = {
    "baseline": Path("models/yolo-football-ball-detection.pt"),
    "gsr_finetune": Path("models/yolo-football-ball-detection-gsr.pt"),
    "multicorpus": Path("models/finetune/ball_multicorpus/weights/best.pt"),
}

#: Test images are named ``<domain>_<clip>_<stem>.jpg`` by the dataset builder.
DOMAINS = ("roboflow", "soccernet_gsr")


def domain_of(path: Path) -> str:
    for domain in DOMAINS:
        if path.name.startswith(domain + "_"):
            return domain
    return "unknown"


def build_domain_yaml(root: Path, domain: str, out_dir: Path) -> Path | None:
    """A YOLO data.yaml whose val set is one domain's test images."""
    images = sorted(
        p for p in (root / "test" / "images").glob("*.jpg") if domain_of(p) == domain
    )
    if not images:
        return None

    listing = out_dir / f"{domain}_test.txt"
    listing.write_text(
        "\n".join(str(p.resolve()) for p in images) + "\n", encoding="utf-8"
    )
    yaml_path = out_dir / f"{domain}_test.yaml"
    yaml_path.write_text(
        f"path: {root.resolve().as_posix()}\n"
        f"train: {listing.resolve().as_posix()}\n"
        f"val: {listing.resolve().as_posix()}\n\n"
        f"nc: 1\nnames: ['ball']\n",
        encoding="utf-8",
    )
    return yaml_path


def evaluate(weights: Path, data_yaml: Path, imgsz: int) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml), imgsz=imgsz, split="val", verbose=False, plots=False
    )
    box = metrics.box
    return {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/eval/ball_domains"))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument(
        "--max-regression", type=float, default=0.02,
        help="largest per-domain drop in recall or mAP50 a candidate may cause",
    )
    args = parser.parse_args()

    configure_logging("INFO")
    if not DATASET.exists():
        log.error("no dataset at %s; run scripts/build_ball_dataset.py", DATASET)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = defaultdict(int)
    for path in (DATASET / "test" / "images").glob("*.jpg"):
        counts[domain_of(path)] += 1
    log.info("test images per domain: %s", dict(counts))

    yamls = {
        domain: build_domain_yaml(DATASET, domain, args.out) for domain in DOMAINS
    }
    results: dict[str, dict[str, dict]] = {}
    for name, weights in CANDIDATES.items():
        if not weights.exists():
            log.warning("%s missing at %s; skipped", name, weights)
            continue
        results[name] = {}
        for domain, yaml_path in yamls.items():
            if yaml_path is None:
                continue
            results[name][domain] = evaluate(weights, yaml_path, args.imgsz)
            log.info(
                "%-12s %-14s P %.3f R %.3f mAP50 %.3f",
                name, domain,
                results[name][domain]["precision"],
                results[name][domain]["recall"],
                results[name][domain]["map50"],
            )

    # -- promotion rule ------------------------------------------------------- #
    verdicts = {}
    if "baseline" in results:
        for name, per_domain in results.items():
            if name == "baseline":
                continue
            regressions, improvements = [], []
            for domain, scores in per_domain.items():
                base = results["baseline"].get(domain)
                if base is None:
                    continue
                for metric in ("recall", "precision", "map50"):
                    delta = scores[metric] - base[metric]
                    if delta < -args.max_regression:
                        regressions.append(
                            {"domain": domain, "metric": metric, "delta": round(delta, 4)}
                        )
                    elif delta > 0:
                        improvements.append(
                            {"domain": domain, "metric": metric, "delta": round(delta, 4)}
                        )

            mean = {
                metric: sum(d[metric] for d in per_domain.values()) / len(per_domain)
                for metric in ("recall", "precision", "map50")
            }
            base_mean = {
                metric: sum(
                    d[metric] for k, d in results["baseline"].items() if k in per_domain
                ) / len(per_domain)
                for metric in ("recall", "precision", "map50")
            }
            improves_all = all(mean[m] > base_mean[m] for m in mean)
            promote = improves_all and not regressions
            verdicts[name] = {
                "cross_domain_mean": {k: round(v, 4) for k, v in mean.items()},
                "baseline_mean": {k: round(v, 4) for k, v in base_mean.items()},
                "improves_every_cross_domain_metric": improves_all,
                "material_regressions": regressions,
                "n_improvements": len(improvements),
                "promote_to_default": promote,
                "reason": (
                    "improves every cross-domain metric with no material per-domain "
                    "regression"
                    if promote
                    else "; ".join(
                        filter(None, [
                            "" if improves_all
                            else "does not improve every cross-domain mean",
                            f"{len(regressions)} material per-domain regression(s)"
                            if regressions else "",
                        ])
                    )
                ),
            }
            log.info("%s -> promote=%s (%s)", name, promote, verdicts[name]["reason"])

    payload = {
        "dataset": str(DATASET),
        "split": "test (clip-disjoint)",
        "imgsz": args.imgsz,
        "max_regression": args.max_regression,
        "test_images_per_domain": dict(counts),
        "results": results,
        "promotion": verdicts,
        "note": (
            "Scored per domain on clip-disjoint test images. Roboflow numbers are "
            "NOT comparable to the 0.912 recall reported before Phase 2C: that "
            "figure came from the published frame split, whose test clips all "
            "appear in training."
        ),
    }
    destination = args.out / "ball_domain_comparison.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"results": results, "promotion": verdicts}, indent=2))
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
