# Broadcast ball adaptation — A/B/C/D comparison

**Decision: KEEP CURRENT DEFAULT.** No candidate passes all eight predeclared
criteria. Every one fails on false positives, and two also regress possession
determinability. Details in §5.

The headline is worth stating up front because it is not the obvious reading of
the numbers: **all three candidates massively improve ball detection, on both the
public benchmark and the local broadcast test, and all three still fail.** The
gain is real and it is bought with false positives the pipeline cannot absorb.

## 1. Training data composition

| candidate | training set | images | composition |
|---|---|---|---|
| A | *none — current default* | — | shipped checkpoint `fb37942448e7de08` |
| B | `ball_broadcast_public` | 2,465 | public multi-corpus train only |
| C | `ball_broadcast_adapt` | 2,534 | public + 69 local broadcast train frames |
| D | `ball_broadcast_adapt_aug` | 4,102 | above + 1,448 augmented copies + 120 pitch-line hard negatives |

Identical recipe for all three: YOLO11n, imgsz 960, batch 8, 40 epochs,
patience 12, seed fixed, same validation set for checkpoint selection and early
stopping. Ultralytics' built-in augmentation is held constant across B, C and D —
**D differs only by the offline augmented copies and hard negatives in its
training set**, so the comparison isolates that and nothing else.

| candidate | checkpoint fingerprint | selected threshold |
|---|---|---|
| A | `fb37942448e7de08` | 0.12 |
| B | `a8ec3ab1b1056487` | 0.12 |
| C | `e1e373009e4a8c96` | 0.12 |
| D | `4f65fd4c8d3fd2ab` | 0.12 |

Thresholds were swept on validation only, under a rule declared before the sweep
(maximise centre-25 recall subject to centre-25 precision ≥ 0.55), then frozen
before either locked test was touched. All four selected 0.12 independently.

## 2. Public locked test (1,154 images, 1,108 balls, 46 negatives)

| | A | B | C | D |
|---|---|---|---|---|
| IoU50 recall | 0.3910 | 0.5811 | **0.6036** | 0.5847 |
| IoU50 precision | **0.5622** | 0.4718 | 0.3953 | 0.4364 |
| IoU50 F1 | 0.4612 | **0.5208** | 0.4777 | 0.4998 |
| centre recall @5 px | 0.424 | 0.606 | **0.631** | 0.615 |
| centre recall @25 px | 0.490 | 0.704 | **0.726** | 0.705 |
| roboflow @25 | 0.5585 | 0.7574 | **0.7941** | 0.7644 |
| soccernet_gsr @25 | 0.4171 | 0.6462 | **0.6536** | 0.6425 |

### By ball size, blur and occlusion (centre recall @25 px)

| | A | B | C | D |
|---|---|---|---|---|
| tiny < 150 px² (n=510) | 0.386 | 0.635 | **0.680** | 0.649 |
| small 150–400 px² (n=500) | 0.606 | 0.776 | **0.782** | 0.766 |
| medium+ > 400 px² (n=100) | 0.440 | **0.690** | 0.680 | **0.690** |
| blurred (n=7) | 0.571 | 0.429 | 0.571 | 0.571 |
| sharp (n=1103) | 0.490 | 0.705 | **0.727** | 0.706 |
| occluded (n=443) | 0.352 | 0.476 | **0.524** | 0.499 |
| clear (n=667) | 0.582 | 0.855 | **0.861** | 0.843 |

The blur row has **n=7** and carries no weight — the public corpora contain
almost no motion-blurred balls, which was the measured gap augmentation was
meant to fill. With seven examples it cannot be shown either way.

### False positives — cross-domain evidence

| | A | B | C | D |
|---|---|---|---|---|
| per negative frame (46 frames) | 0.3913 | **0.3261** | 0.5217 | 0.7174 |
| per frame, all 1,154 | **0.2929** | 0.6256 | 0.8882 | 0.7262 |

## 3. Local locked test (23 frames, 23 balls, **0 negatives**)

| | A | B | C | D |
|---|---|---|---|---|
| centre recall @25 px | 0.5652 | 0.6957 | **0.8696** | 0.8261 |
| 95% CI (Wilson, n=23) | [0.37, 0.74] | [0.49, 0.84] | [0.68, 0.95] | [0.63, 0.93] |
| median centre error | 2.612 px | 2.115 px | **1.904 px** | 1.940 px |
| **precision** | **not computable** | **not computable** | **not computable** | **not computable** |

**No local precision is reported, in any form.** The locked local test has zero
negative frames, so there is no denominator for a false-positive rate. The
public-negative figures in §2 are *cross-domain* evidence and must not be read
as local precision.

The intervals overlap heavily. C's [0.68, 0.95] and A's [0.37, 0.74] are
separated, but C against D ([0.63, 0.93]) is not remotely distinguishable at
n=23. Every local comparison here is directional, not precise.

## 4. Pipeline, identical segments

Local clip 0–120 s; SN-BAS 600–780 s. Same configuration, only the checkpoint
changes.

| | A | B | C | D |
|---|---|---|---|---|
| local direct ball coverage | 0.3225 | 0.2827 | **0.3462** | 0.3273 |
| local possession determinability | **0.0841** | 0.0532 | 0.0758 | 0.0483 |
| SN-BAS coverage | 0.4229 | 0.3822 | 0.3982 | **0.4320** |
| SN-BAS determinability | **0.1214** | 0.1112 | 0.1110 | 0.1079 |
| unchanged pass recall | 0.227 | 0.227 | 0.227 | 0.227 |
| unchanged pass F1 | 0.323 | 0.294 | **0.345** | **0.345** |
| unchanged carry F1 | 0.304 | **0.375** | 0.311 | 0.360 |
| runtime ms/frame | 20.7 | 21.2 | 21.8 | 21.2 |
| peak GPU MB | 110.8 | 121.4 | 129.2 | 134.9 |

## 5. The eight criteria, applied mechanically

| criterion | B | C | D |
|---|---|---|---|
| 1. local recall ≥ +0.05 | **pass** +0.131 | **pass** +0.304 | **pass** +0.261 |
| 2. local precision ≥ 0.55 | **not evaluable** | **not evaluable** | **not evaluable** |
| 3. no public domain regression > 0.05 | **pass** | **pass** | **pass** |
| 4. FP growth ≤ +25% (negatives) | **pass** −16.7% | **FAIL** +33.3% | **FAIL** +83.3% |
| 4. FP growth ≤ +25% (all frames) | **FAIL** +113.6% | **FAIL** +203.2% | **FAIL** +147.9% |
| 5. local coverage ≥ 0 | **FAIL** −0.040 | **pass** +0.024 | **pass** +0.005 |
| 6. local determinability ≥ 0 | **FAIL** −0.031 | **FAIL** −0.008 | **FAIL** −0.036 |
| 7. pass F1 ≥ 0 | **FAIL** −0.029 | **pass** +0.022 | **pass** +0.022 |
| 7. carry F1 ≥ 0 | **pass** +0.071 | **pass** +0.007 | **pass** +0.056 |
| 8. runtime ≤ 1.5× | **pass** 1.02× | **pass** 1.06× | **pass** 1.03× |

### Two disclosures about my own criteria

**Criterion 2 is unevaluable and I knew it might be.** I declared "local test
precision ≥ 0.55" while already knowing the local test had no negatives, and
flagged that at the time. It has no denominator. I have not substituted a
different number in its place and called the criterion met — it is reported as
N/A, and under a rule requiring all criteria to hold, an unverifiable criterion
cannot be treated as passed.

**Criterion 4 was ambiguous as written.** "False positives per frame" admits two
readings: predictions on frames that contain no ball, and predictions per frame
overall. They disagree for candidate B — it passes the first (−16.7%) and fails
the second (+113.6%). I did not specify which when I declared it, so I applied
**both**, and B fails on one. Choosing the reading that promotes B after seeing
the numbers is exactly the move these predeclared rules exist to prevent.

The all-frames figure is partly an artefact of IoU50 strictness on an 11 px ball
— a correct centre with a slightly wrong box counts as both a miss and a false
positive — which is why the per-negative-frame column is the more meaningful
hallucination measure. That is an argument for measuring differently next time,
not for waiving the criterion now.

# DECISION: KEEP CURRENT DEFAULT

## What each candidate failed

**B (public only)** — failed 4 criteria: false positives over all frames
(+113.6%), local coverage (−0.040), local determinability (−0.031), pass F1
(−0.029). B is the only candidate whose false-positive rate on true negatives
*improved*, but it is also the only one that made local coverage worse.

**C (public + local adaptation)** — failed 3: false positives on both readings
(+33.3%, +203.2%) and local determinability (−0.008). **C is the strongest
candidate.** It has the best local recall (0.870), best public recall (0.726),
best tiny-ball recall (0.680), best occluded recall (0.524), improves local
coverage and both event metrics, and its determinability shortfall is −0.008 —
well inside the noise of a 6,000-frame segment. Only the false-positive growth
is a clear, large failure.

**D (public + local + augmentation)** — failed 3: false positives on both
readings (+83.3%, +147.9%) and local determinability (−0.036).

## What this measured that is worth keeping

**Local adaptation works.** 69 broadcast frames took local recall from 0.565 to
0.870 and *also* improved every public domain. That is the first intervention in
this project to improve the target domain and the source domains together.

**Augmentation did not help here.** D is worse than C on local recall
(0.826 vs 0.870), public recall (0.705 vs 0.726) and false positives
(+83% vs +33%). On 4,102 images against 2,534, the augmented copies added noise
rather than robustness — and the one thing they were designed to fix, motion
blur, has n=7 in the public test and cannot be evaluated.

**The binding constraint has moved.** For three phases it was recall. It is now
precision: every candidate finds far more balls and hands the trajectory search
more false candidates than it can reject, which is why determinability falls
even as coverage rises.

## Preserved

Current default `fb37942448e7de08` unchanged and remains the fallback. All 115
annotations, the 285 unreviewed samples, the QC queue, every model fingerprint,
and all dataset provenance are intact. Candidate checkpoints are kept for the
next iteration.

## Reproduction

```bash
python scripts/build_broadcast_split.py
```

```bash
python scripts/build_broadcast_dataset.py --variant adapt
```

```bash
python scripts/finetune_ball.py train --data data/ball_broadcast_adapt/data.yaml --epochs 40 --imgsz 960 --batch 8 --patience 12 --name bcast_adapt
```

```bash
python scripts/evaluate_broadcast_candidates.py
```

```bash
python scripts/broadcast_promotion.py
```
