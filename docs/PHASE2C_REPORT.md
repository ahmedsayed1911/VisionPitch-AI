# Phase 2C — Robust Ball and Event Understanding

**Verdict: NOT READY FOR PHASE 3.** Six of eight readiness thresholds are not
met. The full table is in §9.

Phase 2C set out to fix the three blockers Phase 2B named: multi-corpus ball
training, possession ground truth, and pass detection above ~0.5 F1. The first
two were delivered and both produced real gains. The third was not reached, and
this report explains precisely what bounds it.

Two results matter more than the rest, and both come from measurement rather
than from new modelling:

* **The published Roboflow train/test split leaks.** All 14 of its test clips
  also appear in training. The in-distribution ball recall of 0.912 carried
  through Phases 1B and 2B was measured on frames from matches the model trained
  on.
* **One uncalibrated constant was costing half the pass F1.** The possession
  control radius had never been fitted to anything. Calibrating it against
  annotated geometry took pass F1 from 0.158 to 0.312 with no change to any
  detector.

---

## 1. Corpus audit — the leak

An audit of the corpora on disk, before any training, found:

| Finding | Evidence |
|---|---|
| `ball_det` and `player_det` share source clips | 17 distinct clips across both |
| The published split is a random **frame** split | `train ∩ test = 14 of 14` clips |
| They are therefore **one domain**, not two | recorded in `evaluation/registry.py` |

Consequences, stated rather than buried:

* **The 0.912 in-distribution ball recall is not a held-out number.** It is not
  quoted anywhere in this report except here, to retire it.
* Any "two-corpus" claim in earlier phases was really one corpus.

Every corpus is now re-split **by source clip** in
[`registry.py`](src/visionpitch/evaluation/registry.py), with a stable hash so
adding sequences later cannot move an existing clip out of a test set. The
registry also records what each corpus may be used for: SN-BAS has no ball
boxes, so it is marked ineligible for ball training and validation and is used
only as the cross-domain coverage and event test set.

## 2. Multi-corpus ball dataset

Built by [`build_ball_dataset.py`](scripts/build_ball_dataset.py) into
`data/ball_multicorpus`.

| | train | val | test |
|---|---|---|---|
| SoccerNet-GSR frames | 1528 | 378 | 567 |
| Roboflow frames | 937 | 85 | 587 |
| ball instances | 2327 | 435 | 1110 |

Split fingerprint `d36028408932eddc`. Domain balancing caps any one domain at
62% of the training set — without it, GSR's order-of-magnitude larger frame
count makes a "multi-corpus" run a GSR run wearing a different label, which is
exactly what Phase 2B showed does not transfer. Validation and test keep every
frame, because capping them would change what the metric measures.

An unlabelled image is treated as **unknown, not as a negative**. Treating it as
"no ball here" teaches the model to suppress real balls.

## 3. Ball detection — per domain, never pooled

40 epochs, YOLO11n, imgsz 960. Scored on the **clip-disjoint test split of each
domain separately**.

| model | domain | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| baseline | roboflow | 0.804 | 0.545 | 0.646 | 0.348 |
| baseline | soccernet_gsr | 0.435 | 0.212 | 0.121 | 0.031 |
| gsr fine-tune | roboflow | 0.546 | 0.356 | 0.346 | 0.098 |
| gsr fine-tune | soccernet_gsr | 0.576 | 0.451 | 0.422 | 0.166 |
| **multi-corpus** | roboflow | 0.783 | **0.662** | **0.724** | **0.367** |
| **multi-corpus** | soccernet_gsr | 0.489 | 0.374 | 0.313 | 0.092 |

Cross-domain means:

| model | recall | precision | mAP50 |
|---|---|---|---|
| baseline | 0.379 | 0.620 | 0.384 |
| gsr fine-tune | 0.403 | 0.561 | 0.384 |
| **multi-corpus** | **0.518** | **0.636** | **0.518** |

Multi-corpus training works: it improves every cross-domain mean, and unlike the
Phase 2B fine-tune it does not trade one domain for another — it beats the
baseline on roboflow recall (+0.117) *and* on GSR recall (+0.162).

### The promotion rule, applied

The rule was fixed before the numbers were seen: promote only if the candidate
improves cross-domain recall, precision and mAP50 **and** regresses on no single
domain by more than 0.02.

| candidate | improves every cross-domain mean | material regressions | promoted |
|---|---|---|---|
| gsr fine-tune | no | 3 (roboflow R −0.189, P −0.259, mAP50 −0.300) | **no** |
| multi-corpus | **yes** | 1 (roboflow precision **−0.0211**) | **no** |

The multi-corpus checkpoint fails on a single metric by 0.0011 — one tenth of
one percent past the tolerance. **The threshold has not been moved.** It was
chosen before the results existed, and moving it now to reach a nicer conclusion
is the exact failure mode this project's rules forbid. The checkpoint ships as
`models/finetune/ball_multicorpus/weights/best.pt`, available and documented,
not default.

That verdict is also supported independently by the downstream test in §5,
where the baseline still gives the best effective coverage on a third corpus.

## 4. Possession ground truth — the Phase 2B blocker, resolved

Phase 2B could not measure possession because SN-BAS has no possession labels.
SN-GSR does have what is needed, and Phase 2B did not use it: every annotation
carries a **team label**, a **track id** and a **pitch position in metres**.

[`possession_gt.py`](src/visionpitch/evaluation/possession_gt.py) derives
possession intervals from those annotations. Across all 49 sequences (36,750
frames, 1,079 intervals):

| label | share of time |
|---|---|
| loose | 42.8% |
| unknown | 26.9% |
| left | 15.4% |
| right | 13.7% |
| contested | 1.2% |

Mean coverage 73.1% (min 0.360, max 0.985).

**What this is.** A derived reference. It is independent of the engine's
perception (annotated boxes, not detections) and of the engine's geometry
(metric pitch space, not image space). It is **not** independent of the
proximity assumption: both assume the nearest player owns the ball. If that
premise is wrong — a player screening the ball, a defender closer than the
carrier — both are wrong together. Validating the premise needs a human watching
video, which no corpus here provides. This is stated in the module and repeated
here because it bounds every number in §5.

### The airborne-ball problem, measured

`bbox_pitch` is a ground-plane projection, so an airborne ball's coordinate
slides toward the horizon. Measured over 13,126 frames:

* 9.8% put the ball outside the pitch plus a 5 m margin
* 9.4% imply a step over 1.6 m per frame — above 40 m/s, with an observed
  extreme of **374 m in one frame** (9,350 m/s)
* 13.3% fail at least one check

Those frames are `UNKNOWN`, never guessed. `LOOSE` and `UNKNOWN` are kept
distinct: loose means the ball was located and nobody was near it; unknown means
its position is not trustworthy. Collapsing them would let a detector that loses
the ball score as if it had proven the ball was free.

### Threshold justification

The control radius is measured, not chosen. Over 11,385 usable frames the
ball-to-nearest-player distance is bimodal: a peak at 0.5–1.0 m, a trough at
1.5–2.0 m, and a second population beyond. **1.75 m** sits in the trough.

## 5. Possession engine — validated, and it revealed two defects

Run in two configurations so perception error and logic error are separable.
Held-out **test** sequences of the clip-disjoint split; the radius was swept on
the **train** sequences only.

### Configuration A — perfect perception (annotated boxes)

| | value |
|---|---|
| scorable frames | 5,230 of 6,750 |
| reference coverage | 0.775 |
| prediction coverage | 0.928 |
| **team F1 (macro)** | **0.685** |
| holder accuracy | **0.978** [0.970, 0.984], n = 1559 |

Per label: loose 0.738, right 0.710, left 0.661, contested 0.356.

This is the **logic ceiling**. Detection, tracking and team classification are
all perfect, and the engine still scores 0.685 — below the 0.75 threshold before
a single detector error is introduced. That is the single most important number
in this report.

Holder accuracy of 0.978 says the opposite about attribution: **when the engine
knows a team has the ball, it names the right player 98% of the time.** Player
attribution is not the problem; knowing *whether* anyone has the ball is.

### Two defects the measurement exposed

**1. The control radius was never calibrated.** Swept against the reference on
34 training sequences:

| radius (player-heights) | 0.2 | 0.35 | 0.5 | **0.6** | 0.8 | 1.2 | **1.6** | 2.0 |
|---|---|---|---|---|---|---|---|---|
| team F1 | 0.350 | 0.651 | 0.745 | **0.751** | 0.731 | 0.668 | 0.620 | 0.589 |

The old default of 1.6 (≈2.9 m) sat far down the right-hand slope. At 1.6 the
engine produced 1,179 false negatives on loose balls and **zero** false
positives — it almost never admitted the ball was free. Default now **0.6**,
with the sweep recorded in the config docstring.

**2. The contest rule hard-coded team ids.** It tested
`nearest_team in ("A", "B")`. The pipeline's classifier emits `A`/`B`, so this
worked in production, but the same engine run against any corpus labelling teams
`left`/`right` had contest detection **silently disabled** — no error, no
warning, contested F1 exactly 0.000. The same latent bug was present in the
event engine's pass-vs-turnover split, the heatmap and passing-network filters,
the quality counters, and team-profile construction, where it would have
returned **no team profiles at all**. Replaced everywhere with a shared
`is_team()` predicate over a sentinel set. Contested F1 is now 0.356.

## 6. Events — measured before and after

Same clip, same ground truth, same tolerance as Phase 2B. Full record in
[`phase2c_before_after.json`](data/eval/bas/benchmarks/phase2c_before_after.json).

| | ball observed | pass P | pass R | **pass F1** | carry F1 | determinable |
|---|---|---|---|---|---|---|
| Phase 2B baseline | 43.4% | 0.188 | 0.136 | **0.158** | 0.235 | 12.4% |
| **Phase 2C baseline** | 43.4% | **0.500** | 0.227 | **0.312** | 0.333 | 12.0% |
| Phase 2C gsr fine-tune | 37.2% | 0.800 | 0.182 | 0.296 | **0.425** | 10.2% |
| Phase 2C multi-corpus | 40.0% | 0.667 | 0.182 | 0.286 | 0.340 | 10.5% |

**Pass F1 roughly doubled on identical detections.** Precision went 0.188 →
0.500. Nothing about the detector changed; one constant was calibrated.

**Pass recall did not move** — 10 predictions against 22 ground-truth passes.
This is the bound worth naming precisely: possession is determinable on 12% of
frames, because the ball is effectively observed on 43% of them and calibration
covers 62% of this clip. A pass needs two consecutive determinable possessions
by different players of the same team. No change to event logic can lift recall
through that; only ball coverage can.

**ball_out and restart remain 0.000 F1** in every configuration.
**Header remains a false-positive generator**: 62 predictions against 2 real
headers. It is emitted as a candidate and must not be shown as a detection.

## 7. Event review workspace — the Phase 2B gap, closed

[`reviews.py`](src/visionpitch/api/reviews.py) plus five endpoints on the API.

* Corrections are **append-only** and stored separately from predictions.
  `events.parquet` is byte-identical before and after a review round-trip —
  asserted in the smoke test by SHA-256.
* The corrected view is computed at read time and keeps the model's original
  values under `raw`, so prediction and human judgement stay independently
  inspectable.
* Seven actions including `add_missed` and `mark_unknown`. A review workflow
  that can only delete and retype teaches the model nothing about its blind
  spots. `UNKNOWN_PLAYER` lets a reviewer assert an event happened without
  fabricating who did it.
* Superseding writes a new record; history is reconstructable.
* Each correction stores run fingerprint, model weight hashes and the analytics
  schema version.
* The active-learning queue ranks by **uncertainty**, not by predicted
  wrongness — a reviewer cannot know which is which in advance, and confident
  errors are found by sampling, not by ranking on confidence.
* Export is an explicit, fingerprinted step, never automatic. A pipeline that
  silently retrains on review clicks has no reproducible training set.

### A provenance gap found while building it

The first correction recorded `models: {}`. The chunked runner **never wrote
model fingerprints into its manifest** — only the single-pass runner did. Every
chunked run, which is how full matches are processed, had no record of which
weights produced it. Fixed, including a guard that fails loudly if two chunks
used different weights, and one that does not erase provenance on a full resume.

## 8. Feature validation status

| Feature | Status |
|---|---|
| Multi-corpus registry, clip-disjoint splits | **Validated** — leak detected, stability and tamper-detection pinned |
| Multi-corpus ball dataset and training | **Validated** — per-domain held-out gains measured |
| Possession ground truth (derived) | **Validated as a derived reference** — cannot validate the proximity premise |
| Possession engine, logic | **Measured and insufficient** — team F1 0.685 with perfect perception |
| Player attribution (holder) | **Validated under perfect perception** — 0.978 [0.970, 0.984]; not measurable end-to-end |
| Pass / carry detection | **Improved and still poor** — pass F1 0.312, carry F1 0.333 |
| Ball-out / restart detection | **Unsupported** — 0.000 F1 in every configuration |
| Header detection | **Unsupported** — 62 predictions against 2 real headers |
| Restart classification | **Unsupported** — type is not classified at all |
| Goal / save detection | **Unsupported** — no ground truth |
| Goalkeeper analytics | **Experimental** — GSR contains 3,206 goalkeeper annotations, so this is now *measurable*; it has not been measured |
| Event review workspace | **Validated** — round-trip and immutability tested |
| Active-learning export | **Implemented, not validated** — no retraining loop has been run on exported corrections |

## 9. Verdict

# NOT READY FOR PHASE 3

| Criterion | Target | Measured | Met |
|---|---|---|---|
| Cross-domain ball recall | ≥ 0.60 | 0.518 (roboflow 0.662, GSR 0.374) | **No** |
| Cross-domain ball precision | ≥ 0.55 | **0.636** | **Yes** |
| Effective ball coverage | ≥ 0.60 | 0.400–0.434 on SN-BAS | **No** |
| Team possession F1 | ≥ 0.75 | **0.685 with perfect perception** | **No** |
| Pass-attempt F1 @ 0.4 s | ≥ 0.50 | 0.312 | **No** |
| Sender accuracy | ≥ 0.75 | 0.978 under perfect perception; not measurable end-to-end | **Partial** |
| Receiver accuracy | ≥ 0.65 | not measurable — no corpus labels pass receivers | **No** |
| Carry F1 | ≥ 0.50 | 0.333 (0.425 with the GSR checkpoint) | **No** |

No threshold was changed after seeing a result.

### Why Phase 3 must still wait

Phase 3 is xPass and xT, both trained on detected passes. At pass F1 0.312 with
recall 0.227, roughly three in four real passes are never seen and half of those
detected are wrong. An xPass model trained on that learns the engine's blind
spots, and because those blind spots correlate with camera distance and
occlusion, it would produce a model that looks plausible everywhere and is
wrong systematically in the same places — unfalsifiable without exactly the
ground truth we do not have.

### The blockers, in order, with the reasoning now available

1. **Ball coverage, and nothing else, bounds events.** Pass recall is capped by
   12% possession determinability, itself capped by 43% effective ball coverage.
   Every event improvement in this phase came from precision; recall did not
   move at all. Multi-corpus training is the right method — it produced the
   first ball gain that did not cost another domain — and it needs more corpora,
   not more epochs.
2. **The possession logic is genuinely insufficient, independently of
   perception.** 0.685 with perfect boxes is now a measured fact, not a
   suspicion. The remaining error is concentrated in loose-vs-controlled
   (F1 0.738) and contested (F1 0.356). Contest handling has never been tuned
   and, until this phase, never even executed outside production.
3. **Receiver attribution is still unmeasurable.** No corpus here labels who
   received a pass. Sender attribution is strong (0.978 under perfect
   perception), so the gap is corpus availability, not method.

### What would change the verdict

Cross-domain ball recall from 0.518 to above 0.60 is the single highest-leverage
change; it lifts coverage, determinability, pass recall and possession F1
together. Combined with contest tuning against the now-existing reference, pass
F1 above 0.50 is a reachable target. It was not reached here, and saying so is
the point.

## Reproduction

```bash
python scripts/build_ball_dataset.py --out data/ball_multicorpus --clean
```

```bash
python scripts/finetune_ball.py train --data data/ball_multicorpus/data.yaml --epochs 40 --name ball_multicorpus
```

```bash
python scripts/evaluate_ball_domains.py --out data/eval/ball_domains
```

```bash
python scripts/evaluate_possession.py --split train --sweep
```

```bash
python scripts/evaluate_possession.py --split test
```

```bash
python scripts/evaluate_events.py --run outputs_bas/mid_pre_720p_5505c698690b/6ed3a5e25bcefdd2 --gt data/eval/bas/event_gt_half1.json --offset 0 --label baseline
```
