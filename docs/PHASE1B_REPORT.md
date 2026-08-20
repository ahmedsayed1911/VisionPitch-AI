# Phase 1B — Measurement and Accuracy Hardening

## Verdict

**READY FOR PHASE 2, with two named constraints.** Reasoning and thresholds are
in [the verdict section](#verdict-in-detail) at the end.

---

## 1. What was measured, and against what

The Phase 1 acceptance gap was that detection and tracking accuracy had never
been measured. That is now closed, using **expert-annotated public corpora**
rather than self-drawn annotations. The reasoning: hand-labelling a hundred
broadcast frames is slow, and identity on the crowded side of a frame is
genuinely ambiguous, so the resulting ground truth would carry an unknown error
that propagates silently into every metric derived from it.

| Corpus | Task | Distribution | Size used |
|---|---|---|---|
| `martinjolif/football-player-detection` (test) | detection, 4 classes | **in-distribution** | 25 frames, 599 objects |
| `martinjolif/football-ball-detection` (test) | ball detection | **in-distribution** | 125 frames |
| `SoccerNet/SN-GSR-2025` (test) | tracking with identities | **out-of-distribution** | 6 sequences × 250 frames |
| Wikimedia CC BY-SA clips | end-to-end pipeline | — | 1350 frames |

**In-distribution** means the checkpoint's own held-out split: it measures the
detector on its training domain and is *not* evidence of generalisation.
**Out-of-distribution** is a different corpus entirely, and is the number to
quote. The two are never averaged.

---

## 2. Detection — measured

In-distribution, IoU 0.5, 95% bootstrap CI resampled over frames.

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| player | 0.976 | **0.988** [0.976, 0.998] | 0.979 | 0.822 |
| goalkeeper | 0.941 | 0.842 [0.684, 1.000] | 0.796 | 0.656 |
| referee | 0.964 | 0.964 [0.900, 1.000] | 0.960 | 0.719 |
| ball (multiclass corpus) | 0.528 | 0.792 [0.630, 0.957] | 0.701 | 0.361 |
| **ball (dedicated corpus)** | **0.663** | **0.912** [0.856, 0.960] | **0.879** | 0.526 |

All-class mAP50 **0.859**. Small-object recall is tracked per class; the ball is
always reported separately because averaging it with three person classes above
0.95 hides it entirely.

Out-of-distribution detection quality is captured by DetA below (0.598), which
is markedly lower than the in-distribution figures — as expected, and exactly
why the split is reported.

---

## 3. Tracking — measured

Out-of-distribution, SN-GSR-2025, 6 sequences, players + goalkeepers + referees.

| Metric | Value | 95% CI |
|---|---|---|
| **HOTA** | **0.607** | [0.552, 0.657] |
| DetA | 0.598 | |
| AssA | 0.623 | |
| **IDF1** | **0.735** | [0.665, 0.789] |
| **MOTA** | **0.671** | [0.558, 0.770] |
| ID switches | 113 | |
| Fragmentations | 673 | |
| Mostly tracked / mostly lost | 67 / 11 | |

### Association A/B — a negative result, reported as measured

| Association | HOTA | AssA | IDF1 | MOTA | ID sw | Median track (frames) |
|---|---|---|---|---|---|---|
| none | 0.6061 | 0.6196 | 0.7305 | 0.6760 | 107 | 84.7 |
| greedy (Phase 1) | 0.6068 | 0.6232 | 0.7351 | 0.6712 | 115 | 92.0 |
| **global (Phase 1B)** | 0.6068 | 0.6233 | **0.7354** | 0.6714 | 113 | **94.8** |

Global association beats *no association at all* by **+0.0007 HOTA and +0.0049
IDF1** — comfortably inside the confidence interval. **It is not a demonstrated
improvement on this corpus.**

Why, specifically: association's strongest cue is agreement in pitch
coordinates, and this benchmark scores in image space with calibration disabled,
so **zero** merges had pitch evidence available. On the validation clip, where
calibration does run, the same solver merged 262 raw tracklets to 92 (vs
greedy's 120) and raised pitch-based merges from 6 to 43 — but that gain is
*unverified for correctness*, because the validation clip has no identity
ground truth.

Global is retained as the default because it is no worse, produces marginally
fewer ID switches than greedy (113 vs 115), gates every join explicitly, and
reports why each refusal happened. It is **not** claimed as an accuracy win.

CLEAR "fragmentation" counts interruptions in *ground-truth* coverage, so
association cannot move it by construction — merging predicted tracklets changes
identity assignment, not per-frame coverage. Median predicted-track length is
the metric that does respond, and it rose 84.7 → 94.8 frames (+12%).

---

## 4. Ball — measured and materially improved

Direct ball observation on the validation clip:

| Configuration | Frames with a candidate | Directly observed |
|---|---|---|
| Phase 1 baseline | 64.8% | **48.0%** |
| + trajectory-search fix | 64.8% | 53.4% |
| + sweep every frame | 91%¹ | **60.2%** |
| (+ lowered thresholds — **rejected**) | — | 63.2% |

¹ detector still-image ceiling; pipeline candidate rate rises correspondingly.

**+12.2 percentage points, a 25% relative improvement, with no precision cost.**

Two distinct root causes, both found by instrumenting rather than guessing:

1. **The trajectory search discarded real detections.** It peeled off the
   best-scoring path repeatedly and *stopped* the moment the best remaining path
   fell below the length floor. Because a short high-confidence pair can outscore
   a long chain of weak detections, that exit fired early and threw away every
   remaining candidate in the clip — 26% of all frames that had a real ball
   detection. Replaced with score-ordered peeling and per-node claiming.

2. **The tiled sweep was rate-limited to every 3rd frame.** Two thirds of the
   frames where the ROI crop missed therefore got no second look at all. At
   every frame, candidate coverage rose to match the detector's actual ability.

Lowering the confidence floors was measured and **rejected**: +3pp of pipeline
observations for −11.3pp of detector precision (0.663 → 0.550), and the
trajectory search accepts ~73% of candidates so it is not a strong enough filter
to absorb that. The option is documented for anyone who wants the trade.

Interpolation remains bounded and separately flagged: 88.1% of frames have a
position, 60.2% of them observed, and 11.9% are left explicitly unknown.

---

## 5. Calibration — measured and materially improved

| Metric | Before | After |
|---|---|---|
| Frames with a valid homography | 75.4% | **97.0%** |
| Person rows with pitch coordinates | 73.8% | **94.8%** |
| Frame-to-frame stability, median | 0.421 m | 0.441 m |
| Frame-to-frame stability, p95 | 3.25 m | 5.19 m |
| Mean reprojection error | 0.364 m | 0.364 m |

### Far-side coverage — the headline bottleneck

Pitch-coordinate coverage for person rows, by vertical image position:

| Image band | Before | After | Δ |
|---|---|---|---|
| **0–300 px (far side / horizon)** | **53.4%** | **87.9%** | **+34.5 pp** |
| 300–400 px | 60.7% | 93.1% | +32.4 pp |
| 400–500 px | 87.7% | 98.1% | +10.4 pp |
| 500–600 px (near camera) | 95.3% | 97.6% | +2.3 pp |

Two mechanisms:

**Motion-propagated camera pose.** Most unsolved frames were unsolved because
the camera happened to be pointing at featureless grass, not because its pose
was unknown. The tracker already estimates frame-to-frame background motion; the
calibrator now chains it from solved anchors (`H_{t+1} = H_t · W⁻¹`), bounded in
length with decaying confidence. 817 anchors filled 292 frames.

**Extrapolation is flagged, not hidden.** Previously, far-side projections that
fell outside a hard containment margin were *silently discarded* — which is why
coverage read 53% there with no explanation. They are now retained and marked
`validation_status = extrapolated`, with the risk derived from how far the point
lies outside the image region the landmarks actually constrained.

Note the honest consequence: rows labelled `valid` **decreased** from 5120 to
4188, because far-side rows previously counted as `valid` are now correctly
identified as extrapolated. Total usable rows rose; the *confidently* usable
subset is now labelled accurately rather than optimistically.

The p95 stability regression (3.25 → 5.19 m) is the cost of propagation: the
tail is longer because propagated frames compound motion-estimation error. The
median is unchanged, and propagated frames are individually flagged.

---

## 6. Full-match chunked processing

Implemented and tested. Peak memory is a function of chunk length, not match
length.

- Ownership **tiles the frame range exactly once** — property-tested, so the
  merge cannot duplicate or drop a frame.
- Chunks overlap for tracker/calibrator warm-up and to give the merger shared
  frames on which to re-link identities.
- Identity linking across a seam uses a one-to-one assignment over the shared
  frames, requiring several frames of agreement so two players crossing near a
  boundary cannot be merged.
- Atomic per-chunk checkpoints; an interrupted run resumes at the last completed
  chunk.

Measured on the validation clip at 400-frame chunks with 60-frame overlap:
4 chunks, **19 identities linked across seams**, 6 unlinked, **360 duplicate
rows correctly dropped**, full 1350-frame coverage.

Integration tests assert the properties that matter: chunked frame coverage
equals single-pass coverage exactly, no duplicate `(frame, track_id)` rows, no
duplicate ball rows, and a second invocation skips completed chunks.

---

## 7. Data integrity

- **`frames.parquet` added.** Previously `game_state.parquet` contained 1283 of
  1350 processed frames: 67 frames produced no rows, and a consumer could not
  distinguish "processed, nothing in it" from "never processed". Every per-frame
  rate computed from the game state alone had the wrong denominator. There is
  now one row per processed frame carrying object counts, calibration state and
  chunk provenance.
- **Config-mutation provenance bug fixed.** The pipeline was writing an
  effective frame rate into the live config after the manifest had been written,
  so a run's stored `config_fingerprint` disagreed with the config that produced
  it. Caught by an integration test.
- Parquet round-trips re-verified for detections, tracks, calibration nulls,
  frames, ball observed/interpolated flags and team assignments.
- Backward compatible: no existing column changed type, meaning or nullability;
  `frames.parquet` and the `extrapolated` status are additive.

---

## 8. Tests

**184 passing** (was 144). New coverage: chunk-plan tiling properties, seam
identity linking and its refusals, merge de-duplication, global association
gates (teleport, temporal overlap, size, class, long-gap-without-pitch),
calibration propagation (gap filling, no-overwrite, decay, boundedness, panning
camera), extrapolation risk, frame-presence table, chunked-vs-single-pass
equivalence, and chunked resume.

---

## 9. Reproducing every number

```bash
python scripts/download_eval_data.py player_det ball_det gsr
```

```bash
visionpitch benchmark data/eval/player_det --label baseline
```

```bash
visionpitch benchmark-tracking data/eval/gsr --label global --sequences 6 --max-frames 250
```

```bash
visionpitch analyse data/raw/nz_canada_u17.mp4 --mode balanced
```

```bash
visionpitch analyse-match data/raw/match.mp4 --chunk-frames 9000 --overlap-frames 150
```

---

## Verdict in detail

### READY FOR PHASE 2 — for team-level and zonal analytics

The thresholds that support this:

| Requirement | Threshold | Measured |
|---|---|---|
| Player detection recall | > 0.95 | **0.988** (in-dist) |
| Tracking identity quality | IDF1 > 0.70 | **0.735** (OOD) |
| Tracking overall | HOTA > 0.55 | **0.607** (OOD) |
| Pitch-coordinate coverage | > 90% | **94.8%** |
| Calibration stability (median) | < 1 m | **0.44 m** |
| Full-match capability | required | implemented, tested |
| Reproducible measurement | required | public corpora, one command |

### The two constraints

**1. Ball-dependent analytics are capped at ~60% direct observation.** Possession
and pass detection inherit this. A pass beginning and ending inside an unknown
window cannot be recovered at all, and the possession state machine will spend
real time in `unknown`. Phase 2 must treat `unknown` as a first-class state, not
an error, and must not report possession percentages without also reporting
coverage. Raising this further needs detector work — temporal frame stacking or
fine-tuning — not another downstream fix.

**2. Per-player physical statistics are not yet safe; team-level ones are.**
IDF1 0.735 means roughly a quarter of identity assignments are wrong somewhere
in a track's life. Aggregate team metrics absorb that; per-player distance and
sprint counts do not. Additionally, 2496 rows are flagged `extrapolated` and
must be filtered (`validation_status == 'valid'`) before any distance or speed
computation, or far-side error will inflate every total.

### What would move the verdict to unconditional

Ranked by measured leverage:

1. **Ball recall to >80%** — detector-side work; the downstream gains are spent.
2. **IDF1 to >0.85** — needs identity evidence that survives occlusion, most
   plausibly jersey-number recognition as an anchor, since appearance cannot
   separate players in identical kit.
3. **Calibration confident-coverage up**, which would in turn let association
   use pitch evidence on most joins rather than 27 of 170 — the single change
   most likely to make global association a real improvement rather than a
   measured non-result.
