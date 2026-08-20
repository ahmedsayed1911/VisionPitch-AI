# Ball perception: failure taxonomy, observability, and recovery

Phase 2D methodology. Every number here cites the split, model, matcher and
tolerance that produced it, because several of them contradict figures published
in earlier phases and the reason is always a difference in measurement, not in
the pipeline.

## Provenance for everything in this document

| | |
|---|---|
| Dataset | `data/ball_multicorpus`, clip-disjoint, split fingerprint `d36028408932eddc` |
| Sequence corpus | SN-GSR test split, split fingerprint `c17a32a5bf06dca5` |
| Candidate model | `models/finetune/ball_multicorpus/weights/best.pt` |
| Incumbent model | `models/yolo-football-ball-detection.pt` |
| Operating confidence | 0.08 (`BallDetectionConfig.conf_threshold`) |
| Inference size | 960 |
| Code revision | this repository has **no git commits**, so no revision hash can be cited; results are reproducible from the scripts named in each section |

Leakage was re-verified from scratch at the start of Phase 2D, independently of
`registry.py`: 0 shared source clips between any pair of splits, and 0
byte-identical images across splits out of 4,082 distinct images.

---

## 1. The matcher problem

A broadcast football is about 11 px across. On the multi-corpus test split the
median ball area is **120 px²** for Roboflow and **221 px²** for SN-GSR;
63.7% of Roboflow test balls are under 150 px².

At that size, IoU ≥ 0.5 requires the predicted box to sit within roughly two
pixels of truth. It is a *localisation* criterion far more than a detection one.
Nothing downstream needs that precision: the possession engine compares the ball
centre against player boxes at a radius of tens of pixels.

Measured on the same model, same split, same confidence:

| matcher | macro recall | macro precision |
|---|---|---|
| IoU ≥ 0.5 | 0.6251 | 0.3266 |
| centre distance ≤ 25 px | **0.7575** | 0.4071 |

Both are reported throughout. **Neither replaces the other**, and the Phase 2D
readiness threshold is judged against the criterion it was declared with. The
25 px tolerance is not new to this phase — it was measured in Phase 2B from the
bimodal distribution of ball-to-nearest-prediction distance.

> A caution that applies to every earlier figure: Phase 2C's headline
> "cross-domain recall 0.518" came from `ultralytics.val`, which uses its own
> confidence sweep and assignment. The 0.6251 above is a greedy IoU-0.5 match at
> the fixed operating threshold. They measure different things and should not be
> subtracted from one another.

## 2. Failure taxonomy

`src/visionpitch/evaluation/ball_failures.py`, run by
`scripts/ball_failure_audit.py`.

Each ground-truth ball is placed in exactly one category by a fixed priority
order running from *the ball was not available to be seen* down to *the ball was
clean and the detector missed it*. Priority rather than multi-label, because a
40 px² ball behind a defender satisfies both "tiny" and "occluded", and a better
small-object model cannot reveal a ball that is not visible.

Each image is scored **twice** — at the 0.08 operating threshold and at a floor
of 0.001 — so a ball that had a candidate at the floor is recorded as
*threshold-rejected* rather than as blindness. Phase 2B could not separate those
two and concluded "detector blindness" without being able to size it.

Player boxes come from the shipped multiclass detector. They are evidence used
to characterise a miss, never ground truth and never used to score one.

### Result: multi-corpus model, test split, 1,110 ground-truth balls

Recall 0.7595 (centre 25 px). Of the 267 misses:

| category | count | % of misses | roboflow | gsr |
|---|---|---|---|---|
| player occlusion | 173 | **64.8%** | 83 | 90 |
| detector threshold rejection | 65 | **24.3%** | 15 | 50 |
| low contrast | 21 | 7.9% | 5 | 16 |
| detector miss, unexplained | 3 | 1.1% | 1 | 2 |
| ball outside visible frame | 3 | 1.1% | 0 | 3 |
| pitch-line confusion | 1 | 0.4% | 0 | 1 |
| genuinely unobservable | 1 | 0.4% | 0 | 1 |

**89.1% of misses are occlusion or thresholding.** Only 1.1% is a visible,
in-focus, well-contrasted ball that the detector simply failed on.

This is the central finding of Phase 2D and it redirects the milestone. The
detector is not mostly blind; it is mostly looking at a ball that is behind
somebody, or reporting one and having the threshold discard it. A larger or more
temporal detector architecture addresses the 1.1%.

### By size and domain

| | recall |
|---|---|
| tiny (< 150 px²) | 0.7176 |
| small (150–400 px²) | 0.8200 |
| medium (400–2000 px²) | 0.6700 |
| roboflow | 0.8185 |
| soccernet_gsr | 0.6965 |

The tiny-ball collapse Phase 2B reported (recall 0.00 under 150 px²) is gone:
multi-corpus training with scale augmentation lifted it to 0.72.

### Model comparison, identical matcher and threshold

| model | recall | roboflow | gsr | tiny | occlusion | threshold |
|---|---|---|---|---|---|---|
| multi-corpus | **0.7595** | 0.8185 | 0.6965 | 0.7176 | 173 | 65 |
| baseline | 0.6423 | 0.4695 | 0.8268 | 0.4725 | 256 | 98 |
| GSR fine-tune | 0.5279 | 0.6091 | 0.4413 | 0.4373 | 246 | 106 |

## 3. Operating threshold

`scripts/ball_threshold_sweep.py`. Chosen on train+val, measured once on test.
The selection rule was fixed before running: maximise worst-domain centre-25
recall subject to macro centre-25 precision at or above the declared 0.55 floor.

Selection set behaviour:

| conf | IoU50 R | IoU50 P | selected |
|---|---|---|---|
| 0.03 | 0.8005 | 0.2732 | |
| 0.05 | 0.7848 | 0.3509 | |
| 0.08 | 0.7700 | 0.4408 | previous default |
| 0.12 | 0.7540 | 0.5300 | **chosen** |
| 0.25 | 0.7204 | 0.7004 | |
| 0.50 | 0.6143 | 0.8731 | |

Held-out test, at both thresholds:

| | IoU50 R | IoU50 P | centre25 R | centre25 P | worst-domain centre25 R |
|---|---|---|---|---|---|
| 0.08 | 0.6251 | 0.3266 | **0.7575** | 0.4071 | 0.6965 |
| 0.12 | 0.6033 | 0.3878 | 0.7366 | 0.4884 | 0.6704 |

**The threshold was not changed.** The sweep's own rule picked 0.12 on the
selection set, but on test that configuration loses 2.1 points of recall and
still does not reach the 0.55 precision floor (0.4884). Recall is the binding
constraint for this milestone — every downstream ceiling traces back to it — so
trading recall for precision that still fails its floor buys nothing. `0.08`
remains the default, and this paragraph is the record of a change that was
measured and declined.

> A defect found here and worth recording: the first version of this sweep
> initialised its best-match score to `-1` while the centre criterion scores
> candidates as *negative* distance. Only sub-pixel matches were accepted, and
> centre-25 recall came out *below* IoU-50 recall — which is arithmetically
> impossible and is what exposed the bug. Any matcher whose permissive criterion
> scores lower than its strict one is broken, not surprising.

## 4. Observability

`src/visionpitch/ball_tracking/observability.py`.

Every ball metric divides by "frames", and that denominator lumps together a
detector failure, a ball behind a player, and a replay of the crowd. The
observability model labels each frame with whether the ball *could* have been
seen, so rates can be reported over frames where a miss is actually the
detector's fault.

Labels: `likely_visible`, `likely_occluded`, `likely_outside_frame`,
`likely_hidden_by_players`, `likely_motion_blurred`, `not_on_pitch`,
`uncertain`. Only the visible, blurred and uncertain classes count against the
detector.

**The model never produces a ball position.** It builds an internal expectation
purely to decide *which part of the image to reason about*; that expectation is
never returned, never stored, and never reaches the trajectory estimator. A
unit test asserts that no coordinate appears anywhere in its serialised output.
Without that rule it would become the fabrication the trajectory estimator
refuses, laundered through a different module.

Two expiry rules keep it honest:

* a camera cut resets the expectation entirely — the previous shot says nothing
  about the new one
* an expectation older than `max_extrapolation_frames`, or drifting further than
  `max_expected_drift_px`, is abandoned and the frame becomes `uncertain`
  rather than confidently labelled from a stale position

Measured on the SN-GSR test sequences (9 sequences, 6,750 frames): observable
fraction **0.9422**. Detector recall rises from 0.674 raw to **0.714** when
scored only over frames where the ball was plausibly visible. On this corpus the
gap is small because GSR is clean continuous footage with no cuts; on broadcast
footage with replays it is expected to be much larger, and that is exactly where
the raw denominator misleads.

## 5. Track-before-detect recovery

`src/visionpitch/ball_tracking/recovery.py`.

Given that 24.3% of misses are threshold rejections, sensitivity is available —
but lowering the global threshold pays a precision cost on every frame including
the easy ones. This stage spends extra sensitivity only where the trajectory
says the ball should be: a few hundred pixels out of two million.

Method: three-frame differencing inside a 48 px window around the interpolated
position. Three frames rather than two because a two-frame difference lights up
both where the object was and where it is, and on a small fast object those
blobs are indistinguishable.

The rule that keeps it safe: **one frame of weak evidence is never an
observation.** A recovery needs `min_supporting_frames` consecutive candidates
whose step sizes are mutually consistent with a plausible trajectory. Gaps
longer than `max_gap_frames` are refused without even being searched. Frames the
observability model calls out-of-frame or off-pitch are skipped.

Recovered positions are labelled `BallStateKind.RECOVERED` — known, carrying
image evidence, but **not direct**, and never counted as a sighting.

### Held-out result, and what it means

9 SN-GSR test sequences, default parameters:

| | |
|---|---|
| gap frames | 1,435 |
| recovered | 248 (yield 17.3%) |
| **recovery accuracy** | **0.5735** |
| recoveries in the wrong place | 87 |
| recoveries on frames with no annotated ball | 44 |
| coverage, direct | 0.7874 |
| coverage, direct + recovered | 0.8241 |

Recovery adds 3.7 points of coverage and is **wrong 43% of the time**. That is
not a coverage improvement; it is 87 confident wrong ball positions handed to
the possession engine, which is worse than 87 frames of honest `UNKNOWN`.

The parameter sweep (`scripts/sweep_recovery.py`, train sequences, accuracy bar
0.85) found **no usable configuration**. Tightening the deviation gate to 12 px
yields zero recoveries; loosening it to 36 px yields recoveries that are wrong
two thirds of the time. The failure is structural rather than a tuning gap:
there is no setting at which the weak evidence is both plentiful and correct.

Consequently the stage **is not wired into the pipeline**. It exists as an
evaluated, tested module; no configuration flag turns it on. Enabling it would
require a code change and a fresh measurement.

## 6. What this changes about where the effort should go

The taxonomy says the ball is mostly missed because a player is in front of it.
That is not a detector problem and no detector architecture solves it. It is a
temporal problem, and the honest options are:

1. better trajectory continuation through short occlusions, with the position
   marked inferred — bounded, already partly implemented, and safe because it is
   labelled
2. recovery from weak evidence — attempted here, and **measured at 57% accuracy,
   which is not good enough to enable**
3. accepting that occluded frames are unobservable and reporting coverage over
   observable frames instead, which is what the observability model makes
   possible

Option 3 is the only one that produced a defensible improvement in this
milestone, and it improves *honesty of measurement* rather than perception
itself.
