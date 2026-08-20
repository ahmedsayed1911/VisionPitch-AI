"""Fine-tune the detector or the pitch keypoint model.

    python scripts/train.py --task detect --data data/yolo_det/dataset.yaml
    python scripts/train.py --task pose   --data data/yolo_pose/dataset.yaml

Defaults are chosen for football rather than for generic COCO training:

* **1280 px input.** Distant players are the binding constraint; at 640 px they
  are a handful of pixels and recall collapses. This is the single most
  important setting here.
* **Conservative augmentation.** Horizontal flip is safe. Large scale/mosaic
  augmentation is *not* left at COCO defaults for the keypoint task, because
  mosaic composites four images and destroys the single global pitch geometry
  the model must learn.
* **Cosine LR with a long warmup**, since fine-tuning from COCO on a small,
  highly correlated dataset otherwise destabilises early.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionpitch.training.data_policy import (  # noqa: E402
    DataBoundaryError,
    TrainingDataPolicy,
    find_project_root,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=("detect", "pose"), required=True)
    p.add_argument("--data", type=Path, required=True, help="dataset.yaml path")
    p.add_argument("--weights", default=None, help="Starting checkpoint.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--name", default=None)
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


class RunTelemetry:
    """Persist continuous GPU samples and per-epoch wall times."""

    def __init__(self, name: str) -> None:
        self.started = time.time()
        self.stop_event = threading.Event()
        self.peak_used_mib = 0
        self.samples_path = Path("runs") / f"{name}_gpu_telemetry.csv"
        self.epochs_path = Path("runs") / f"{name}_epoch_telemetry.jsonl"
        self.summary_path = Path("runs") / f"{name}_telemetry_summary.json"
        self.samples_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._monitor, daemon=True)

    def _monitor(self) -> None:
        with self.samples_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["unix_time", "gpu_util_percent", "memory_used_mib", "memory_total_mib"]
            )
            while not self.stop_event.is_set():
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                        "--id=0",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                try:
                    utilization, used, total = [
                        int(value.strip()) for value in result.stdout.strip().split(",")
                    ]
                except (ValueError, IndexError):
                    self.stop_event.wait(5.0)
                    continue
                self.peak_used_mib = max(self.peak_used_mib, used)
                writer.writerow([round(time.time(), 3), utilization, used, total])
                stream.flush()
                self.stop_event.wait(5.0)

    def start(self) -> None:
        self._thread.start()

    def record_epoch(self, trainer) -> None:
        payload = {
            "epoch": int(trainer.epoch) + 1,
            "epoch_seconds": round(float(trainer.epoch_time), 3),
            "elapsed_seconds": round(time.time() - self.started, 3),
            "peak_total_gpu_used_mib_so_far": self.peak_used_mib,
            "training_losses": {
                key: float(value)
                for key, value in trainer.label_loss_items(trainer.tloss).items()
            },
            "validation_metrics": {
                key: float(value)
                for key, value in trainer.metrics.items()
                if isinstance(value, (int, float))
            },
        }
        with self.epochs_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")

    def finish(self, status: str, error: str | None = None) -> None:
        self.stop_event.set()
        self._thread.join(timeout=10)
        self.summary_path.write_text(
            json.dumps(
                {
                    "status": status,
                    "total_elapsed_seconds": round(time.time() - self.started, 3),
                    "peak_total_gpu_used_mib": self.peak_used_mib,
                    "gpu_samples": str(self.samples_path),
                    "epoch_telemetry": str(self.epochs_path),
                    "error": error,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def main() -> None:
    args = parse_args()
    project_root = find_project_root(Path(__file__).parent)
    try:
        TrainingDataPolicy(project_root).validate_dataset_yaml(args.data)
    except (DataBoundaryError, OSError, ValueError) as exc:
        sys.exit(f"TRAINING DATA BOUNDARY ABORT: {exc}")
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics is required: pip install ultralytics")

    if not args.data.exists():
        sys.exit(
            f"Dataset config not found: {args.data}\n"
            "Run scripts/prepare_yolo_dataset.py first."
        )

    weights = args.weights or (
        "models/yolo11x.pt" if args.task == "detect" else "models/yolo11x-pose.pt"
    )
    epochs = args.epochs or (60 if args.task == "detect" else 80)
    name = args.name or f"vp_{args.task}_v1"
    telemetry = RunTelemetry(name)

    common = dict(
        data=str(args.data),
        epochs=epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        name=name,
        project="runs",
        cos_lr=True,
        warmup_epochs=5.0,
        patience=20,
        fliplr=0.5,
        # No vertical flip: a football image is never upside down, and the
        # augmentation would teach an impossible orientation.
        flipud=0.0,
        deterministic=True,
        plots=True,
        save_period=1,
    )

    if args.task == "detect":
        common.update(
            # Mosaic helps small-object detection, but is turned off for the
            # final epochs so training ends on real, uncomposited frames.
            mosaic=1.0,
            close_mosaic=10,
            scale=0.5,
            translate=0.1,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
        )
    else:
        common.update(
            # Mosaic composites four images into one and destroys the single
            # coherent pitch geometry the keypoint model has to learn.
            mosaic=0.0,
            scale=0.3,
            translate=0.05,
            degrees=0.0,
            shear=0.0,
            perspective=0.0,
        )

    print(f"Fine-tuning {weights} for '{args.task}' at {args.imgsz}px, {epochs} epochs")
    model = YOLO(weights)
    model.add_callback("on_fit_epoch_end", telemetry.record_epoch)
    telemetry.start()
    try:
        results = model.train(**common)
    except Exception as exc:
        telemetry.finish("crashed", repr(exc))
        raise
    telemetry.finish("completed")

    best = Path("runs") / name / "weights" / "best.pt"
    if best.exists():
        target = Path("models") / f"{name}.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(best.read_bytes())
        print(f"\nBest checkpoint -> {target}")
        print("Record it in models/REGISTRY.md and point the config at it.")
    else:
        print(f"\nTraining finished but {best} was not produced.")
    return results


if __name__ == "__main__":
    main()
