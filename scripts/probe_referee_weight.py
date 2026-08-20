"""3-epoch probe: does per-class classification weighting fix referee precision?

Hypothesis under test
---------------------
Referee AP50 (0.2623 on VALID) is limited by class-prior imbalance rather than
appearance, labels or augmentation - all three of which were measured and
rejected in the referee audit.

Mechanism (no library modification)
-----------------------------------
Ultralytics 8.4.114 already supports per-class classification weighting:

    utils/loss.py:438   bce_loss = self.bce(pred_scores, target_scores)  # (bs, anchors, nc)
    utils/loss.py:439   if self.class_weights is not None:
    utils/loss.py:440       bce_loss *= self.class_weights
    utils/loss.py:441   loss[1] = bce_loss.sum() / target_scores_sum

`class_weights` is read from the model via `getattr(model, "class_weights", None)`
at criterion construction. So the entire "patch" is: set that attribute.

We deliberately do NOT use the built-in `cls_pw` path
(models/yolo/detect/train.py:162-186), because it derives *inverse-frequency*
weights for every class and normalises their mean to 1.0. At the measured
300614:12375:30338:19802 distribution that mostly **down-weights player**
(0.574 at cls_pw=0.25) while leaving referee at ~1.019 - it would not test the
hypothesis. `cls_pw=0.0` disables it so our explicit weights survive.

What this touches: the classification (BCE) term only. Box loss, DFL loss,
architecture, resolution, augmentation, optimizer, LR schedule, dataset and
sampling are all untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

CLASS_NAMES = ["player", "goalkeeper", "referee", "ball"]
REFEREE_IDX = 2


def build_weights(referee_weight: float, device: str) -> torch.Tensor:
    """Explicit per-class weights: only referee deviates from 1.0."""
    w = torch.ones(len(CLASS_NAMES), dtype=torch.float32)
    w[REFEREE_IDX] = referee_weight
    return w.to(device)


def smoke_test(referee_weight: float) -> dict:
    """Verify the weighting maths and broadcasting before spending GPU hours."""
    import torch.nn as nn

    bs, anchors, nc = 2, 7, len(CLASS_NAMES)
    torch.manual_seed(0)
    pred = torch.randn(bs, anchors, nc, requires_grad=True)
    target = (torch.rand(bs, anchors, nc) > 0.7).float()

    bce = nn.BCEWithLogitsLoss(reduction="none")
    raw = bce(pred, target)                      # (bs, anchors, nc)
    w = build_weights(referee_weight, "cpu")     # (nc,)
    weighted = raw * w                           # broadcast over last dim

    # 1. Shape is preserved (no accidental reduction/transpose).
    assert weighted.shape == raw.shape, (weighted.shape, raw.shape)

    # 2. Exactly the referee column is scaled; all others are bit-identical.
    per_class_ratio = []
    for c in range(nc):
        num, den = weighted[..., c].sum().item(), raw[..., c].sum().item()
        per_class_ratio.append(num / den if den else float("nan"))
    for c in range(nc):
        expected = referee_weight if c == REFEREE_IDX else 1.0
        assert abs(per_class_ratio[c] - expected) < 1e-6, (c, per_class_ratio[c])
    assert torch.equal(weighted[..., 0], raw[..., 0])
    assert torch.equal(weighted[..., 1], raw[..., 1])
    assert torch.equal(weighted[..., 3], raw[..., 3])

    # 3. Gradients flow and only the referee column's gradient is rescaled.
    g_plain = torch.autograd.grad(raw.sum(), pred, retain_graph=True)[0]
    g_weighted = torch.autograd.grad(weighted.sum(), pred)[0]
    ratio_ref = (g_weighted[..., REFEREE_IDX] / g_plain[..., REFEREE_IDX]).mean().item()
    ratio_player = (g_weighted[..., 0] / g_plain[..., 0]).mean().item()

    return {
        "shape_preserved": True,
        "per_class_loss_ratio": {
            CLASS_NAMES[c]: round(per_class_ratio[c], 6) for c in range(nc)
        },
        "grad_ratio_referee": round(ratio_ref, 6),
        "grad_ratio_player": round(ratio_player, 6),
        "passed": True,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path,
                   default=Path("runs/detect/runs/vp_yolo11x_gsr_1280/weights/best.pt"))
    p.add_argument("--data", type=Path, default=Path("data/yolo_gsr_detect/dataset.yaml"))
    p.add_argument("--referee-weight", type=float, default=2.0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="0")
    p.add_argument("--name", default="probe_refw2_from_best")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--smoke-only", action="store_true")
    args = p.parse_args()

    # --- guards ----------------------------------------------------------
    cfg = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    if "test" in cfg:
        sys.exit("REFUSING: dataset.yaml declares a 'test' split.")
    if cfg.get("nc") != len(CLASS_NAMES) or list(cfg["names"].values()) != CLASS_NAMES:
        sys.exit(f"REFUSING: unexpected class layout {cfg.get('names')}")
    if not args.weights.exists():
        sys.exit(f"REFUSING: missing init weights {args.weights}")
    if "last.pt" in str(args.weights):
        sys.exit("REFUSING: probe must initialise from best.pt, not last.pt")

    out_dir = Path("runs") / args.name
    if out_dir.exists():
        sys.exit(f"REFUSING: {out_dir} exists; will not overwrite an existing run.")

    print("=== smoke test ===")
    smoke = smoke_test(args.referee_weight)
    print(json.dumps(smoke, indent=2))
    if args.smoke_only:
        return

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        print(f"GPU free {free/1e9:.1f} GB / {total/1e9:.1f} GB")
        if free < 8e9:
            sys.exit(f"REFUSING: only {free/1e9:.1f} GB free; another job may be running.")

    from ultralytics import YOLO
    from ultralytics.utils import LOGGER

    model = YOLO(str(args.weights))
    applied: dict = {}

    def inject_class_weights(trainer):
        """Set explicit per-class weights and force the criterion to rebuild."""
        target = trainer.model
        target = getattr(target, "module", target)
        w = build_weights(args.referee_weight, str(next(target.parameters()).device))
        target.class_weights = w
        # The criterion caches class_weights at construction; drop it so the
        # next forward rebuilds with our weights in place.
        target.criterion = None
        applied["class_weights"] = [round(float(v), 4) for v in w.cpu()]
        LOGGER.info(f"[probe] explicit class_weights = {applied['class_weights']}")

    model.add_callback("on_train_start", inject_class_weights)

    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,                 # multiprocessing loaders crash on this machine
        seed=args.seed,
        name=args.name,
        project="runs",
        exist_ok=False,
        # --- identical to the original run ---
        cos_lr=True, warmup_epochs=5.0, patience=20,
        fliplr=0.5, flipud=0.0,
        deterministic=True, plots=True, save_period=1,
        mosaic=1.0, close_mosaic=10, scale=0.5, translate=0.1,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        # --- the single changed variable ---
        cls_pw=0.0,                # disable built-in inverse-frequency weighting
    )

    record = {
        "init_weights": str(args.weights),
        "referee_weight": args.referee_weight,
        "class_weights_applied": applied.get("class_weights"),
        "epochs": args.epochs,
        "smoke_test": smoke,
        "save_dir": str(results.save_dir) if hasattr(results, "save_dir") else None,
    }
    Path("runs").mkdir(exist_ok=True)
    (Path("runs") / f"{args.name}_probe_config.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
