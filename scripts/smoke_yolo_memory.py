"""Bounded CUDA memory smoke test; this is not a full training run."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionpitch.training.data_policy import (  # noqa: E402
    TrainingDataPolicy,
    find_project_root,
)


def _gpu_used_mib() -> int | None:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "--id=0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return int(completed.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, choices=(3, 4), required=True)
    parser.add_argument("--train-images", type=int, default=32)
    parser.add_argument("--val-images", type=int, default=8)
    args = parser.parse_args()
    project_root = find_project_root(Path(__file__).parent)
    data = project_root / "data/yolo_gsr_detect/dataset.yaml"
    weights = project_root / "models/yolo11x.pt"
    TrainingDataPolicy(project_root).validate_dataset_yaml(data)
    if not weights.is_file():
        raise FileNotFoundError(weights)
    export_root = data.parent
    provenance = [
        json.loads(line)
        for line in (export_root / "frames.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    train_images = [
        export_root / row["exported_image"]
        for row in provenance
        if row["source_split"] == "train"
    ][: args.train_images]
    val_images = [
        export_root / row["exported_image"]
        for row in provenance
        if row["source_split"] == "valid"
    ][: args.val_images]
    qa = export_root / "qa"
    train_list, val_list = qa / "smoke_train.txt", qa / "smoke_val.txt"
    train_list.write_text("\n".join(path.as_posix() for path in train_images), encoding="utf-8")
    val_list.write_text("\n".join(path.as_posix() for path in val_images), encoding="utf-8")
    smoke_data = qa / "smoke_dataset.yaml"
    smoke_data.write_text(
        yaml.safe_dump(
            {
                "path": export_root.as_posix(),
                "train": train_list.as_posix(),
                "val": val_list.as_posix(),
                "nc": 4,
                "names": {0: "player", 1: "goalkeeper", 2: "referee", 3: "ball"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = _gpu_used_mib()
    samples: list[int] = []
    stop = threading.Event()

    def monitor() -> None:
        while not stop.wait(0.25):
            used = _gpu_used_mib()
            if used is not None:
                samples.append(used)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    started = time.perf_counter()
    status = "passed"
    error = None
    try:
        model = YOLO(str(weights))
        model.train(
            data=str(smoke_data),
            epochs=1,
            imgsz=1280,
            batch=args.batch,
            device=0,
            workers=0,
            seed=1234,
            deterministic=True,
            project=str(project_root / "runs/memory_smoke"),
            name=f"yolo11x_1280_batch{args.batch}",
            exist_ok=True,
            val=False,
            save=False,
            plots=False,
            verbose=True,
            cos_lr=True,
            warmup_epochs=0.0,
            mosaic=1.0,
            close_mosaic=0,
            scale=0.5,
            translate=0.1,
            fliplr=0.5,
            flipud=0.0,
        )
    except torch.cuda.OutOfMemoryError as exc:
        status = "cuda_oom"
        error = str(exc)
    finally:
        stop.set()
        thread.join(timeout=2)
    report = {
        "schema": "VISIONPITCH_CUDA_SMOKE_V1",
        "bounded_not_full_training": True,
        "status": status,
        "model": "models/yolo11x.pt",
        "dataset": "data/yolo_gsr_detect/dataset.yaml",
        "imgsz": 1280,
        "batch": args.batch,
        "epochs": 1,
        "train_images": len(train_images),
        "val_images": len(val_images),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "baseline_total_gpu_used_mib": baseline,
        "peak_total_gpu_used_mib": max(samples) if samples else None,
        "peak_torch_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
        "peak_torch_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20),
        "error": error,
    }
    destination = project_root / f"data/yolo_gsr_detect/qa/memory_smoke_batch{args.batch}.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if status != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
