# Setup

## Environment

Verified on: Windows 11, Python 3.14.6, PyTorch 2.12.1+cu126, CUDA 12.6,
RTX 3090 Ti (24 GB), OpenCV 4.13.

Python 3.11+ is required. PyTorch must be a CUDA build for GPU inference.

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # Linux/macOS
pip install -e ".[eval,data,dev]"
```

If PyTorch is already installed system-wide with a working CUDA build, create
the venv with `--system-site-packages` to avoid a ~2.5 GB re-download:

```bash
python -m venv --system-site-packages .venv
```

Verify:

```bash
visionpitch info
```

## Models

```bash
python scripts/download_models.py
```

Downloads `yolo11x.pt` (114.6 MB) and `yolo11x-pose.pt` (86.1 MB) into
`models/`. Weights are never committed; see
[`models/REGISTRY.md`](../models/REGISTRY.md) for provenance.

**These are backbones, not finished models.** Both require fine-tuning on
football data before the pipeline produces meaningful roles, ball detections or
calibration. Until then:

- every person is reported as `role=player`,
- ball state is unreliable,
- calibration is disabled and `pitch_xy` is `None` everywhere.

The pipeline reports each of these as an explicit warning at startup.

## Dataset

```bash
python scripts/download_dataset.py --splits train valid
```

SoccerNet-GSR (SN-GSR-2025) from HuggingFace: public, ungated, GPL-3.0.
`train` 9.76 GB + `valid` 11.17 GB, roughly doubling on extraction. The script
checks free disk space before starting and refuses if there is not enough.

Layout after extraction:

```
data/SoccerNetGS/<split>/SNGS-<nnn>/
    img1/000001.jpg ...
    Labels-GameState.json
```

## Fine-tuning

### Detector

```bash
python scripts/prepare_yolo_dataset.py --split train --task detect --out data/yolo_det
python scripts/train.py --task detect --data data/yolo_det/dataset.yaml --imgsz 1280
```

Classes: `player`, `goalkeeper`, `referee`, `ball`. Train at 1280 px — distant
players are the binding constraint and lower resolution loses them.

### Pitch keypoints

```bash
python scripts/prepare_yolo_dataset.py --split train --task pose --out data/yolo_pose
python scripts/train.py --task pose --data data/yolo_pose/dataset.yaml --imgsz 1280
```

Pitch-landmark labels are **derived** by projecting the 37 template landmarks
through the per-frame camera parameters in the dataset. If SN-GSR-2025 does not
expose usable camera parameters, `prepare_yolo_dataset.py --task pose` reports
how many frames it skipped and the keypoint model cannot be trained this way;
`SoccerNet/SN-Calibration` would then be the fallback source.

The keypoint model must output exactly the 37 landmarks defined in
`visionpitch.geometry.pitch`, **in that order** — it is a stable contract, and
`PitchKeypointDetector` raises on a mismatch rather than misinterpreting
channels.

Point the config at the results:

```yaml
detection:
  weights: models/vp_detector_v1.pt
calibration:
  weights: models/vp_pitch_v1.pt
```

## Optional: learned appearance encoder

The default `colour` encoder needs no download. To use DINOv2 (~90 MB, fetched
by `timm` on first use):

```yaml
reid:
  encoder: dinov2
```

Expected to improve team classification and ReID materially over the colour
histogram. Benchmark both before claiming it does.

## Configuration

```bash
visionpitch config --mode quality --out configs/quality.yaml
```

Precedence: quality-mode preset → YAML file → CLI overrides. A file need only
contain the keys it changes. Every run records its config hash in the game
state so results are traceable.

`QUALITY` (default) uses 1280 px inference, TTA, tiled ball search and a larger
ReID gallery. `BALANCED` uses 960 px and disables TTA and tiling.

## Running

```bash
visionpitch run match.mp4 \
    --mode quality \
    --output-video outputs/annotated.mp4 \
    --output-state outputs/state.jsonl
```

Useful flags: `--max-frames` and `--stride` for quick iteration, `--device cpu`
where no GPU is available.

## Troubleshooting

**`Detector weights not found`** — run `scripts/download_models.py`, or point
`detection.weights` at an existing checkpoint.

**`Keypoint model outputs N landmarks but the pitch template defines 37`** —
expected with the stock COCO pose model. Fine-tune a pitch keypoint model, or
set `calibration.enabled: false` to run without metric coordinates.

**Minimap shows `NO CALIBRATION`** — working as intended: no validated
homography for those frames. Stale positions are never shown instead.

**`CUDA out of memory`** — lower `detection.image_size`, reduce
`detection.batch_size`, or set `detection.augment: false`.

**All players `team=UNKNOWN`** — the classifier refused to fit. The reason is
in `metadata.extra.team_refusal_reason` (typically `kits_not_separable` or
`too_few_samples`).
