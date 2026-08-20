# Evaluation

## The one thing to read first

**Measured results now live in [PHASE1B_REPORT.md](PHASE1B_REPORT.md).** This
document covers metric *definitions* and *method*; that one carries the numbers,
the before/after comparison and the Phase 2 readiness verdict.

Detection and tracking accuracy are measured against expert-annotated public
corpora — `martinjolif/*` test splits (in-distribution) and `SoccerNet/SN-GSR-2025`
(out-of-distribution). Headline: player recall 0.988, HOTA 0.607, IDF1 0.735.

Everything below still applies, in particular the separation of metric kinds
that must never be mixed.

## Two kinds of metric

**Reference-free.** Calibration coverage and temporal stability, ball coverage,
track fragmentation, team cluster separation, throughput. No annotation needed,
fully rigorous, available on every run. They cannot prove the system is right,
but they reliably prove when it is wrong — a calibration jittering 30 m between
frames is broken whatever ground truth would say.

**Ground-truthed.** Detection P/R/mAP, HOTA/IDF1/MOTA, pitch-position error in
metres. These need annotation. When none is supplied the report marks them
`"status": "not measured"` rather than omitting them, so their absence is loud.

## Validation clips

Both CC BY-SA 4.0 from Wikimedia Commons, fetched by `scripts/download_clip.py`,
attribution written alongside each file.

| Key | Content | Size | Why it is useful |
|---|---|---|---|
| `nz_canada_u17` | 2018 FIFA U-17 Women's World Cup, NZ v Canada | 1280×720, 45 s, 30 fps | live play, elevated wide view, centre circle and both penalty areas seen, legible jersey numbers, white vs red kit |
| `versailles_nancy` | Championnat National, FC Versailles v AS Nancy | 1920×1080, 18.5 s, 30 fps | handheld shake, a goal celebration and a period with almost no pitch lines — a deliberate stress case |

## Measured results — `nz_canada_u17`, balanced mode

Full report: `outputs/nz_canada_u17_*/<fingerprint>/evaluation/report.json`.
1350 frames, RTX 3090 Ti.

### Calibration (reference-free)

| Metric | Value |
|---|---|
| frames with a valid homography | **75.4%** |
| frames above the confidence floor | 60.5% |
| self-reported reprojection error, mean | 0.36 m |
| frame-to-frame stability, **median** | **0.42 m** |
| frame-to-frame stability, mean | 0.90 m |
| frame-to-frame stability, p90 / p95 | 1.92 m / 3.25 m |
| frames moving > 5 m between neighbours | 17 of 925 |
| rejected as temporal outliers | 332 |

Quote the **median**, not the mean. The distribution is heavy-tailed: most
frames are stable to well under half a metre while a minority fit badly, and a
mean silently reports the minority. Before temporal outlier rejection existed,
the same clip measured median 1.86 m / mean 29.2 m — the mean alone would have
suggested the calibration was unusable when 863 of 1350 frames sat in one tight
cluster.

The 24.6% of frames without a homography are reported as uncalibrated. They are
not filled in.

### Ball (reference-free)

| Metric | Value |
|---|---|
| directly observed | **48.0%** of frames |
| observed or bounded-gap interpolated | 88.1% |
| left explicitly unknown | 161 frames (11.9%) |

The 11.9% is a feature. Those are gaps longer than the interpolation limit, and
the estimator declines to invent a position rather than handing Phase 2's
possession engine a fabricated one.

### Tracking (reference-free — *not* accuracy)

| Metric | Value |
|---|---|
| raw tracks | 262 |
| after stitching and short-track removal | 120 |
| fragments rejoined offline | 96 |
| dropped as too short | 46 |
| mean / median track length | 93.7 / 46.5 frames |

96 rejoins from 262 raw tracks says fragmentation is the dominant tracking
failure on this clip. It does **not** say identities are correct — only ground
truth can.

### Team discovery (reference-free)

| Metric | Value |
|---|---|
| cluster separation (silhouette) | **0.498** |
| discovered kit colour A (BGR) | `[204, 198, 196]` — white |
| discovered kit colour B (BGR) | `[88, 78, 167]` — red |
| tracks assigned | A 50, B 21, officials 7 |
| tracks left `unknown` | 42 |

The discovered colours match the actual kits (New Zealand white, Canada red).
The 42 unknown tracks are mostly short tracks that never gathered enough votes.

### Throughput

| Stage | Seconds |
|---|---|
| pass 1 (detect + calibrate + track) | 77.6 |
| all offline stages combined | 1.3 |
| rendering three videos | 28.2 |
| **total** | **107** for 45 s of 720p |

≈ 17 fps analysis, ≈ 0.58× real time in balanced mode. The offline stages cost
1.3 s of 107 — the accuracy-over-speed choices are nearly free.

## Measured model-selection findings

These changed the shipped configuration.

### Inference resolution must not exceed training resolution

The multiclass checkpoint was fine-tuned at `imgsz=1280` (read from the
checkpoint's own `train_args`). Raising inference resolution is off-distribution,
not "more accurate":

| Setting | Detections over 5 U-17 frames |
|---|---|
| 960 | 34 players, 2 referees |
| **1280 (training resolution)** | 35 players, 1 referee |
| 1280 + test-time augmentation | **45 players, 4 referees** |
| 1920 | 38 players, 1 referee |

On the Versailles clip, 1920 lost referee detections entirely on a frame where
1280 found two. `max_accuracy` mode therefore keeps `imgsz=1280` and enables
TTA instead. A test pins this so it cannot regress.

### Colour beats a learned embedding at broadcast crop scale

Fitting two team clusters over 600 harvested crops from `nz_canada_u17`:

| Backend | Silhouette | Tracks left unknown |
|---|---|---|
| **hue-saturation histogram** | **0.501** | 39 of 117 |
| SigLIP ViT (`siglip-base-patch16-224`) | 0.237 | 45 of 117 |

At broadcast distance a torso crop is ~20×35 px; upscaled to the 224×224 a ViT
expects, it is mostly interpolation blur, so its features track pose and motion
blur rather than kit. A hue histogram over the same pixels keeps exactly the
discriminating signal. This contradicted the initial design assumption, and the
default was changed to `color`. SigLIP is retained for kits differing in pattern
rather than hue, and for larger crops.

### Two ball detectors, not one

Published on the checkpoints' own test splits:

| Model | ball mAP50 | ball mAP50-95 |
|---|---|---|
| multiclass (player/GK/ref/ball) | 0.680 | 0.338 |
| dedicated ball detector | 0.891 | **0.551** |

The specialist runs on a motion-predicted ROI, so it recovers most of that gap
for a fraction of a second full-frame pass.

## Metric implementations

Implemented in-repo rather than delegated, so the definitions are auditable, and
validated in `tests/unit/test_evaluation.py` against cases with known analytic
answers — a perfect tracker scores HOTA ≈ 1.0, an injected ID switch is counted
exactly once, a swapped pair lowers AssA, coasted boxes do not inflate recall.

- **Detection** — greedy score-ordered matching (the COCO convention), 101-point
  interpolated AP, per-class, with the ball always reported separately and a
  size-stratified small-object recall.
- **Tracking** — HOTA with the reference global-alignment-weighted matching and a
  0.05→0.95 α sweep; IDF1 via one global one-to-one ID assignment; CLEAR MOTA
  with sticky matching so ID switches are counted only when the tracker actually
  switched.
- **Calibration** — coverage, confidence, temporal stability, and (with
  annotation) reprojection error against independently marked landmarks.

Interpolated boxes are excluded from tracking metrics by definition: including
them would let a tracker inflate recall by coasting through occlusions.

## Producing the missing numbers

```bash
python scripts/annotate.py boxes data/raw/nz_canada_u17.mp4 --frames 100,160,220,280,340
```

```bash
python scripts/annotate.py pitch data/raw/nz_canada_u17.mp4 --frames 100,400,700,1000
```

```bash
visionpitch evaluate outputs/<video_id>/<fingerprint> --annotations data/annotations/nz_canada_u17.json
```

Annotation guidance that materially affects the result:

- **Label every object in an annotated frame.** A partial annotation turns real
  players into false positives and destroys precision.
- **Keep track ids consistent across frames.** That consistency is exactly what
  IDF1 and HOTA measure; renumbering each frame scores every tracker at zero.
- **Mark pitch landmarks in frames where the camera has moved**, not four
  consecutive frames. Independent landmarks are the only way to detect a
  systematic homography bias — a homography fitted to the model's own keypoints
  always reproduces those keypoints well, right or wrong.
- Aim for 100+ annotated frames before quoting a number.
