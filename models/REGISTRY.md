# Model registry

Every checkpoint the pipeline can load, with provenance. Weights are **not**
committed to Git; this file is the record of what is used and where it came
from. The machine-readable version, including SHA-256 of every downloaded file,
is `models/manifest.json`.

Fetch the public checkpoints:

```bash
python scripts/download_models.py
```

## Production checkpoints (public, downloadable)

All three come from the HuggingFace Hub and are verified by SHA-256 against
`models/manifest.json` on download.

| File | Purpose | Architecture | Source | Size | Licence | Config key |
|---|---|---|---|---|---|---|
| `yolo-football-player-detection.pt` | **production person detector** — player / goalkeeper / referee / ball | YOLO11m | `martinjolif/yolo-football-player-detection` | 40.6 MB | AGPL-3.0 | `detection.model_path` |
| `yolo-football-ball-detection.pt` | baseline ball specialist; the documented rollback target | YOLO11n | `martinjolif/yolo-football-ball-detection` | 5.5 MB | AGPL-3.0 | `ball_detection.model_path` (rollback) |
| `yolo-football-pitch-detection.pt` | 32-point pitch landmark regression for homography | YOLOv8x-pose | `martinjolif/yolo-football-pitch-detection` | 140.1 MB | AGPL-3.0 | `calibration.model_path` |

Reported metrics as published by the upstream author, recorded in
`models/manifest.json`:

| Checkpoint | Class | mAP50 | mAP50-95 |
|---|---|---|---|
| player-detection | player | 0.9937 | 0.8737 |
| player-detection | goalkeeper | 0.9413 | 0.8024 |
| player-detection | referee | 0.9888 | 0.7741 |
| player-detection | ball | 0.6799 | 0.3380 |
| ball-detection | ball | 0.8910 | 0.5510 |

The ball row is the entire reason the pipeline runs two detectors: 0.551 against
0.338 on the same object.

## Locally fine-tuned checkpoints (not distributed)

These are produced by training scripts in this repository and are **not**
available for download. They are derivatives of SN-GSR-2025 training data.

| Path | Purpose | Provenance | Status |
|---|---|---|---|
| `models/finetune/ball_gsrtrain_v2c/weights/best.pt` | **production ball detector** | SN-GSR canonical `train` only; zero validation/test/challenge sequences, asserted at build time | **Promoted.** Selected in Phase 2E |
| `models/finetune/ball_gsrtrain_v2/weights/best.pt` | undertrained predecessor of v2c | same | Superseded |
| `models/finetune/ball_gsrtrain_v1/weights/best.pt` | first clean-provenance attempt | same | Superseded |
| `models/finetune/ball_gsr/`, `ball_multicorpus/`, `bcast_*/`, `heatmap/` | historical ball experiments | **fine-tuned on SN-GSR *test* frames** | Retired; test-burned, retained only for historical comparison |
| `models/yolo-football-ball-detection-gsr.pt` | early SN-GSR ball fine-tune | **fine-tuned on SN-GSR *test* frames** | Retired; test-burned |

### Reproducing the production ball checkpoint

```bash
python scripts/download_dataset.py --splits train
```

```bash
python scripts/build_ball_gsrtrain_dataset.py
```

```bash
python scripts/train_ball_gsrtrain.py
```

Dataset provenance, including the leakage guard, is committed at
`docs/provenance/ball_gsrtrain_v1_dataset.json`. Selection method, sweep design
and the promotion decision are in `docs/PHASE2E_BALL_OPERATING_POINT.md`.

### Running without it

The pipeline works on the public baseline alone. Roll back with three file
copies:

```bash
cp configs/default.pre_phase2e_ball.yaml configs/default.yaml
```

```bash
cp configs/modes/balanced.pre_phase2e_ball.yaml configs/modes/balanced.yaml
```

```bash
cp configs/modes/max_accuracy.pre_phase2e_ball.yaml configs/modes/max_accuracy.yaml
```

Measured cost on canonical SN-GSR validation: ball F1 0.8517 → 0.5900, centre
recall @25px 0.8056 → 0.4584, possession team F1 0.5131 → 0.3669. Everything
that does not depend on the ball is unaffected.

## COCO-pretrained backbones

Present on disk from earlier work; **not used by any shipping config**.

| File | Size | Licence | Why it is not used |
|---|---|---|---|
| `yolo11x.pt` | 114.6 MB | AGPL-3.0 | COCO classes only. Knows `person` and `sports ball`; cannot distinguish goalkeeper or referee, and under-recalls a broadcast-scale football badly. Selecting `detection.backend: coco` builds `CocoFallbackDetector`, which logs a warning that roles are unavailable rather than pretending otherwise. |
| `yolo11x-pose.pt` | 86.1 MB | AGPL-3.0 | Predicts 17 **human** COCO keypoints, not pitch landmarks. `PitchKeypointDetector` raises on the shape mismatch instead of silently misinterpreting the channels. |

## Optional appearance encoders

| Model | Purpose | Size | Licence | Status |
|---|---|---|---|---|
| `vit_small_patch14_dinov2.lvd142m` (timm) | ReID + team embeddings | ~90 MB | Apache-2.0 | **Not downloaded** — not approved |

Without it the pipeline uses `KitColourEncoder`, a torso-band CIELAB histogram
with grey-world colour constancy and grass masking. That is a considered
descriptor rather than naive RGB averaging — and it measured *better* than a
learned embedding at broadcast crop scale (0.501 silhouette vs 0.237 for SigLIP),
which is why it is the default rather than a fallback. See `docs/EVALUATION.md`.

## Licence consequence

`ultralytics` is AGPL-3.0, so this project is distributed under
AGPL-3.0-or-later. Swapping the detector for an Apache-2.0 alternative
(RF-DETR, RT-DETRv2) would remove that obligation; the `YoloDetector` interface
is deliberately narrow to keep that swap cheap. See
`docs/models_and_licenses.md`.
