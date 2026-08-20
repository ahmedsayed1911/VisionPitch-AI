# Models, datasets and licences

## Project licence

**AGPL-3.0-or-later**, forced by the `ultralytics` dependency.

`ultralytics` is AGPL-3.0. Any project that links it and is conveyed to others
(including over a network) inherits that obligation. This is acceptable for a
research/portfolio project but blocks closed commercial use.

### Escaping AGPL, if ever needed

The detector interface (`visionpitch.detection.detector.YoloDetector`) is
deliberately narrow — it takes frames and returns `Detection` objects, nothing
else. Replacing it with an Apache-2.0 detector removes the obligation:

| Alternative | Licence | Notes |
|---|---|---|
| RF-DETR | Apache-2.0 | Strong on small objects, relevant for the ball |
| RT-DETRv2 | Apache-2.0 | Transformer detector, no NMS |
| D-FINE | Apache-2.0 | Competitive accuracy/latency |

The pitch keypoint model would also need replacing (currently Ultralytics
pose). Nothing else in the codebase depends on Ultralytics.

## Runtime dependencies

| Package | Licence | Used for |
|---|---|---|
| ultralytics | **AGPL-3.0** | Detection and pose backbones |
| torch / torchvision | BSD-3-Clause | Inference |
| opencv-python | Apache-2.0 | Video I/O, geometry, rendering |
| numpy | BSD-3-Clause | Numerics |
| scipy | BSD-3-Clause | Hungarian assignment |
| scikit-learn | BSD-3-Clause | Clustering utilities |
| pydantic | MIT | Typed configuration |
| timm | Apache-2.0 | Optional DINOv2 encoder |
| supervision | MIT | Utilities |
| lap | BSD-2-Clause | Fast linear assignment |
| PyYAML | MIT | Config files |
| typer / rich | MIT | CLI |

### Evaluation-only

| Package | Licence | Used for |
|---|---|---|
| trackeval | MIT | Cross-validating our HOTA/IDF1/CLEAR |
| motmetrics | MIT | Optional secondary check |
| pycocotools | BSD-2-Clause | Optional COCO cross-check |

Our tracking metrics are implemented in-repo and pinned to `trackeval` by
`tests/test_metrics_reference.py`. `trackeval` is not needed at inference time.

## Datasets

### SoccerNet Game State Reconstruction (SN-GSR-2025)

* **Source**: `SoccerNet/SN-GSR-2025` on HuggingFace Hub — public, ungated,
  no NDA or credentials required.
* **Licence**: GPL-3.0.
* **Size**: train 9.76 GB, valid 11.17 GB, test 8.85 GB, challenge 5.31 GB.
* **Contents**: 200 clips of 30 s at 25 fps, with bounding boxes, roles
  (player / goalkeeper / referee / other), team affiliation, jersey numbers,
  track ids and metric pitch coordinates.
* **Downloaded here**: `train` + `valid` (~21 GB).
* **Used for**: detector and keypoint fine-tuning (`train`); all reported
  metrics (`valid`, held out).
* **Citation**: Somers et al., *SoccerNet Game State Reconstruction:
  End-to-End Athlete Tracking and Identification on a Minimap*, CVPRW 2024.

This dataset was chosen because it supplies training data **and** ground truth
for detection, tracking, team classification and calibration in one place —
which is what makes the mandatory evaluation possible at all.

Datasets are never committed to Git. See `scripts/download_dataset.py`.

## Ideas and prior art referenced

Concepts were drawn from the following; no code was copied.

| Source | Contribution |
|---|---|
| ByteTrack (Zhang et al., ECCV 2022) | Two-stage association using low-confidence detections |
| BoT-SORT (Aharon et al., 2022) | Camera motion compensation, appearance-motion fusion |
| DIoU (Zheng et al., AAAI 2020) | Distance-aware overlap cost for small objects |
| HOTA (Luiten et al., IJCV 2021) | Detection/association-separable tracking metric |
| IDF1 (Ristani et al., ECCVW 2016) | Identity-level tracking metric |
| CLEAR-MOT (Bernardin & Stiefelhagen, 2008) | MOTA / MOTP / ID switches |
| COCO | Interpolated average-precision protocol |
| SoccerNet GSR baseline / TrackLab | Task decomposition for game-state reconstruction |
| Hartley & Zisserman, *MVG* §4.4 | Normalised DLT for homography estimation |

## Attribution requirements

* Redistributing this project requires AGPL-3.0 compliance, including source
  availability for network users.
* SoccerNet data is GPL-3.0 and subject to its own terms; it is not
  redistributed here.
* Ultralytics weights are AGPL-3.0 and are downloaded at setup, not vendored.
