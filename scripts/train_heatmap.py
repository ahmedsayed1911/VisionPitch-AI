"""Train and evaluate the centre-heatmap ball detector.

Cross-domain tiny-ball study, Parts 4 and 7.

Trains on the frozen protocol's train partition, selects the checkpoint on
in-domain validation by **worst-domain** centre recall, and scores the test
partition exactly once at the end.

Domain balancing is applied by sampling weight rather than by discarding frames:
the two corpora differ in size, and Phase 2C measured that an unbalanced
multi-corpus run behaves like a single-corpus run wearing a multi-corpus label.

Augmentation targets the measured failure modes rather than being applied out of
habit -- broadcast compression, motion blur, scale and photometric shift, all of
which appear in the Phase 2D failure taxonomy.

Usage::

    python scripts/train_heatmap.py --epochs 40
    python scripts/train_heatmap.py --evaluate-only --checkpoint models/finetune/heatmap/best.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionpitch.common.logging import configure_logging, get_logger  # noqa: E402
from visionpitch.detection.heatmap import (  # noqa: E402
    BallHeatmapNet,
    HeatmapConfig,
    decode,
    focal_loss,
    render_target,
)
from visionpitch.evaluation.tinyball import (  # noqa: E402
    CENTRE_TOLERANCES_PX,
    domain_of,
    pool,
    score_centres,
)
from visionpitch.training.data_policy import (  # noqa: E402
    DataBoundaryError,
    TrainingDataPolicy,
    find_project_root,
)

log = get_logger("tinyball.heatmap")

DATASET = Path("data/ball_multicorpus")
OUTPUT = Path("models/finetune/heatmap")


def load_labels(image_path: Path, width: int, height: int):
    """(centres, sizes) in pixels for one image."""
    label = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    centres: list[tuple[float, float]] = []
    sizes: list[float] = []
    if not label.exists():
        return centres, sizes
    for line in label.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        centres.append((cx * width, cy * height))
        sizes.append(max(bw * width, bh * height))
    return centres, sizes


class BallDataset(Dataset):
    """Letterboxed frames with Gaussian centre targets."""

    def __init__(self, images: list[Path], config: HeatmapConfig, train: bool) -> None:
        self.images = images
        self.cfg = config
        self.train = train

    def __len__(self) -> int:
        return len(self.images)

    def _augment(self, image, centres, sizes):
        rng = random.Random()

        # Scale jitter: the measured failure is concentrated on the smallest
        # balls, so the model must see the ball at a range of apparent sizes.
        if rng.random() < 0.8:
            scale = rng.uniform(0.65, 1.5)
            h, w = image.shape[:2]
            nh, nw = max(8, int(h * scale)), max(8, int(w * scale))
            image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
            centres = [(x * scale, y * scale) for x, y in centres]
            sizes = [s * scale for s in sizes]

        if rng.random() < 0.5:
            image = cv2.flip(image, 1)
            w = image.shape[1]
            centres = [(w - x, y) for x, y in centres]

        # Broadcast compression: SN-BAS is 720p re-encoded broadcast, and JPEG
        # ringing around an 11 px object is a large fraction of its signal.
        if rng.random() < 0.4:
            quality = rng.randint(30, 75)
            ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ok:
                image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

        if rng.random() < 0.3:
            k = rng.choice([3, 5])
            image = cv2.GaussianBlur(image, (k, k), 0)

        if rng.random() < 0.5:
            alpha = rng.uniform(0.7, 1.3)
            beta = rng.uniform(-25, 25)
            image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        return image, centres, sizes

    def _letterbox(self, image, centres, sizes):
        size = self.cfg.input_size
        h, w = image.shape[:2]
        scale = min(size / w, size / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        ox, oy = (size - nw) // 2, (size - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = resized
        centres = [(x * scale + ox, y * scale + oy) for x, y in centres]
        sizes = [s * scale for s in sizes]
        return canvas, centres, sizes

    def __getitem__(self, index: int):
        path = self.images[index]
        image = cv2.imread(str(path))
        if image is None:
            image = np.zeros((64, 64, 3), dtype=np.uint8)
        h, w = image.shape[:2]
        centres, sizes = load_labels(path, w, h)

        if self.train:
            image, centres, sizes = self._augment(image, centres, sizes)
        image, centres, sizes = self._letterbox(image, centres, sizes)

        stride = self.cfg.output_stride
        out_hw = (self.cfg.input_size // stride, self.cfg.input_size // stride)
        target = render_target(centres, sizes, out_hw, stride, self.cfg)

        tensor = torch.from_numpy(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).copy()
        ).float() / 255.0
        return tensor, torch.from_numpy(target)[None], str(path)


def collate(batch):
    tensors = torch.stack([b[0] for b in batch])
    targets = torch.stack([b[1] for b in batch])
    return tensors, targets, [b[2] for b in batch]


def collect_predictions(model, images: list[Path], config: HeatmapConfig, device):
    """Decode once at a permissive threshold; score any operating point offline.

    Re-running the network for every candidate threshold would cost an hour and
    change nothing: the heatmap is identical, only the peak cut-off moves.
    """
    model.eval()
    permissive = HeatmapConfig(**{**config.to_dict_kwargs(), "peak_threshold": 0.02})
    records = []
    started = time.perf_counter()

    with torch.no_grad():
        for path in images:
            image = cv2.imread(str(path))
            if image is None:
                continue
            h, w = image.shape[:2]
            truth, _ = load_labels(path, w, h)

            size = config.input_size
            scale = min(size / w, size / h)
            nw, nh = int(round(w * scale)), int(round(h * scale))
            canvas = np.zeros((size, size, 3), dtype=np.uint8)
            ox, oy = (size - nw) // 2, (size - nh) // 2
            canvas[oy:oy + nh, ox:ox + nw] = cv2.resize(image, (nw, nh))

            tensor = torch.from_numpy(
                cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).copy()
            ).float().div(255.0)[None].to(device)
            heatmap = model(tensor)[0, 0].detach().cpu().numpy()

            detections = [
                # Undo the letterbox back into the original image frame.
                (
                    (d.x - ox) / scale, (d.y - oy) / scale,
                    d.confidence, d.uncertainty_px / scale,
                )
                for d in decode(heatmap, permissive)
            ]
            records.append((domain_of(path), truth, detections))

    elapsed = time.perf_counter() - started
    return records, 1000 * elapsed / max(1, len(images))


def score_at(records, threshold: float, label: str, ms_per_frame: float | None = None):
    per_domain: dict[str, list] = defaultdict(list)
    for domain, truth, detections in records:
        per_domain[domain].append((
            truth, [(x, y) for x, y, c, _ in detections if c >= threshold]
        ))
    results = [
        score_centres(label, domain, frames)
        for domain, frames in sorted(per_domain.items())
    ]
    summary = pool(results, label)
    summary["peak_threshold"] = threshold
    if ms_per_frame is not None:
        summary["runtime_ms_per_frame"] = round(ms_per_frame, 2)
    return summary


def evaluate(model, images: list[Path], config: HeatmapConfig, device, label: str):
    records, ms = collect_predictions(model, images, config, device)
    return score_at(records, config.peak_threshold, label, ms)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--label", default="heatmap")
    parser.add_argument("--checkpoint", type=Path, default=OUTPUT / "best.pt")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument(
        "--eval-sizes", type=int, nargs="*", default=None,
        help=(
            "inference sizes to sweep on validation. The network is fully "
            "convolutional, so it can run larger than it trained; the ball is "
            "~11 px at source and letterboxing 1920x1080 into 640 shrinks it to "
            "under 4 px, which may cost more than the resolution mismatch does."
        ),
    )
    parser.add_argument("--out", type=Path, default=Path("data/eval/tinyball"))
    args = parser.parse_args()

    if not args.evaluate_only:
        project_root = find_project_root(Path(__file__).parent)
        try:
            TrainingDataPolicy(project_root).validate_dataset_yaml(DATASET / "dataset.yaml")
        except (DataBoundaryError, OSError, ValueError) as exc:
            print(f"TRAINING DATA BOUNDARY ABORT: {exc}")
            return 2

    configure_logging("INFO")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = HeatmapConfig(input_size=args.input_size)
    model = BallHeatmapNet(config).to(device)
    log.info(
        "heatmap model: %d parameters, device %s, stride %d",
        sum(p.numel() for p in model.parameters()), device, config.output_stride,
    )

    train_images = sorted((DATASET / "train" / "images").glob("*.jpg"))
    val_images = sorted((DATASET / "val" / "images").glob("*.jpg"))
    test_images = sorted((DATASET / "test" / "images").glob("*.jpg"))
    args.out.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    if not args.evaluate_only:
        # Domain-balanced sampling by weight, so the smaller corpus is seen as
        # often as the larger one without discarding any frame.
        domains = [domain_of(p) for p in train_images]
        counts = {d: domains.count(d) for d in set(domains)}
        weights = [1.0 / counts[d] for d in domains]
        log.info("train frames per domain: %s (balanced by sampling weight)", counts)

        loader = DataLoader(
            BallDataset(train_images, config, train=True),
            batch_size=args.batch,
            sampler=WeightedRandomSampler(weights, len(train_images), replacement=True),
            num_workers=4, collate_fn=collate, drop_last=True, persistent_workers=True,
        )
        optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        schedule = torch.optim.lr_scheduler.OneCycleLR(
            optimiser, max_lr=args.lr, total_steps=args.epochs * len(loader),
        )
        scaler = torch.amp.GradScaler(device, enabled=device == "cuda")

        best_worst = -1.0
        history = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            total = 0.0
            for tensors, targets, _ in loader:
                tensors = tensors.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimiser.zero_grad(set_to_none=True)
                with torch.amp.autocast(device, enabled=device == "cuda"):
                    prediction = model(tensors)
                    loss = focal_loss(prediction.float(), targets, config)
                scaler.scale(loss).backward()
                previous_scale = scaler.get_scale()
                scaler.step(optimiser)
                scaler.update()
                # Only advance the schedule when the optimiser actually stepped.
                # AMP skips steps on overflow, and stepping the scheduler anyway
                # desynchronises the learning-rate curve from training progress.
                if scaler.get_scale() >= previous_scale:
                    schedule.step()
                total += float(loss.detach())

            mean_loss = total / max(1, len(loader))
            if epoch % 4 == 0 or epoch == args.epochs:
                summary = evaluate(model, val_images, config, device, args.label)
                worst = summary["worst_domain_recall_at_px"]["25.0"]
                macro = summary["macro_recall_at_px"]["25.0"]
                history.append({
                    "epoch": epoch, "loss": round(mean_loss, 4),
                    "val_macro_recall_25px": macro, "val_worst_recall_25px": worst,
                })
                log.info(
                    "epoch %2d  loss %.4f  val macro@25 %.4f  worst-domain@25 %.4f%s",
                    epoch, mean_loss, macro, worst,
                    "  <- best" if worst > best_worst else "",
                )
                # Checkpoint selection on WORST-domain recall, never the mean.
                # The whole question is cross-domain transfer, and a mean lets a
                # model win by being excellent on one corpus.
                if worst > best_worst:
                    best_worst = worst
                    torch.save(
                        {"model": model.state_dict(), "config": config.to_dict(),
                         "epoch": epoch, "val_worst_recall_25px": worst},
                        args.checkpoint,
                    )
            else:
                log.info("epoch %2d  loss %.4f", epoch, mean_loss)

        (OUTPUT / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

    # -- final: score the test partition exactly once -------------------------- #
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    log.info(
        "scoring test partition with checkpoint from epoch %s (val worst@25 %.4f)",
        checkpoint.get("epoch"), checkpoint.get("val_worst_recall_25px", float("nan")),
    )

    # -- inference resolution, chosen on validation only ----------------------- #
    size_sweep = []
    if args.eval_sizes:
        for size in args.eval_sizes:
            probe = HeatmapConfig(**{**config.to_dict_kwargs(), "input_size": size})
            summary = evaluate(model, val_images, probe, device, args.label)
            size_sweep.append({
                "input_size": size,
                "macro_recall_25px": summary["macro_recall_at_px"]["25.0"],
                "worst_domain_recall_25px": summary["worst_domain_recall_at_px"]["25.0"],
                "median_error_px": summary["median_error_px"],
                "runtime_ms_per_frame": summary["runtime_ms_per_frame"],
            })
            log.info(
                "val input %d  R@25 %.4f  worst %.4f  median err %s  %.1f ms/frame",
                size, size_sweep[-1]["macro_recall_25px"],
                size_sweep[-1]["worst_domain_recall_25px"],
                size_sweep[-1]["median_error_px"],
                size_sweep[-1]["runtime_ms_per_frame"],
            )
        best_size = max(size_sweep, key=lambda r: r["worst_domain_recall_25px"])
        config.input_size = best_size["input_size"]
        log.info("chosen inference size %d", config.input_size)

    # -- operating point, chosen on validation only ---------------------------- #
    # The box baseline runs at a tuned confidence of 0.08. Comparing an untuned
    # heatmap threshold against it would be a rigged comparison, so the peak
    # threshold gets the same treatment: swept on val_in_domain, fixed, then the
    # test partition is scored once.
    val_records, val_ms = collect_predictions(model, val_images, config, device)
    grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]
    sweep = []
    for threshold in grid:
        summary = score_at(val_records, threshold, args.label)
        precision = sum(
            block["precision_at_px"]["25.0"] for block in summary["per_domain"].values()
        ) / max(1, len(summary["per_domain"]))
        sweep.append({
            "threshold": threshold,
            "macro_recall_25px": summary["macro_recall_at_px"]["25.0"],
            "worst_domain_recall_25px": summary["worst_domain_recall_at_px"]["25.0"],
            "macro_precision_25px": round(precision, 4),
            "false_positives_per_frame": summary["macro_false_positives_per_frame"],
        })
        log.info(
            "val threshold %.2f  R@25 %.4f  worst %.4f  P@25 %.4f  FP/frame %.3f",
            threshold, sweep[-1]["macro_recall_25px"],
            sweep[-1]["worst_domain_recall_25px"],
            sweep[-1]["macro_precision_25px"],
            sweep[-1]["false_positives_per_frame"],
        )

    # Declared rule, matching the study's success criteria: maximise
    # worst-domain centre recall subject to macro precision >= 0.55.
    eligible = [row for row in sweep if row["macro_precision_25px"] >= 0.55]
    chosen = (
        max(eligible, key=lambda r: r["worst_domain_recall_25px"])
        if eligible else max(sweep, key=lambda r: r["macro_precision_25px"])
    )
    config.peak_threshold = chosen["threshold"]
    log.info(
        "chosen peak threshold %.2f (%s precision floor)",
        config.peak_threshold, "meets" if eligible else "CANNOT MEET",
    )

    val_summary = score_at(val_records, config.peak_threshold, args.label, val_ms)
    test_records, test_ms = collect_predictions(model, test_images, config, device)
    test_summary = score_at(test_records, config.peak_threshold, args.label, test_ms)

    payload = {
        "label": args.label,
        "representation": "centre_heatmap",
        "config": config.to_dict(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_fingerprint": hashlib.sha256(
            args.checkpoint.read_bytes()
        ).hexdigest()[:16],
        "selected_epoch": checkpoint.get("epoch"),
        "selection_rule": "highest worst-domain centre recall at 25 px on val_in_domain",
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "threshold_sweep_on_val": sweep,
        "threshold_selection_rule": (
            "maximise worst-domain centre recall at 25 px subject to macro "
            "precision at 25 px >= 0.55; declared before the sweep"
        ),
        "chosen_peak_threshold": config.peak_threshold,
        "precision_floor_reachable": bool(eligible),
        "inference_size_sweep_on_val": size_sweep,
        "chosen_inference_size": config.input_size,
        "trained_input_size": args.input_size,
        "val_in_domain": val_summary,
        "test": test_summary,
        "centre_tolerances_px": list(CENTRE_TOLERANCES_PX),
    }
    destination = args.out / f"{args.label}_test.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n{args.label}: {payload['n_parameters']} parameters, "
          f"epoch {payload['selected_epoch']}")
    print("\ncentre recall by tolerance (macro / worst-domain), TEST:")
    for tolerance in CENTRE_TOLERANCES_PX:
        macro = test_summary["macro_recall_at_px"][str(tolerance)]
        worst = test_summary["worst_domain_recall_at_px"][str(tolerance)]
        print(f"  <= {tolerance:>5.1f} px   {macro:.4f} / {worst:.4f}")
    print(f"\nmedian centre error  : {test_summary['median_error_px']} px")
    print(f"macro direct coverage: {test_summary['macro_direct_coverage']:.4f}")
    print(f"worst-domain coverage: {test_summary['worst_domain_direct_coverage']:.4f}")
    print(f"false positives/frame: {test_summary['macro_false_positives_per_frame']:.4f}")
    print(f"runtime              : {test_summary['runtime_ms_per_frame']} ms/frame")
    for domain, block in test_summary["per_domain"].items():
        print(f"  {domain:<15} R@25 {block['recall_at_px']['25.0']:.4f}  "
              f"P@25 {block['precision_at_px']['25.0']:.4f}  "
              f"coverage {block['direct_coverage']:.4f}")
    print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
