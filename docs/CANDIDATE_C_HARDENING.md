# Candidate C precision hardening

**Decision: KEEP CURRENT DEFAULT.** C-Hardened fails 5 of 11 predeclared
criteria. Hardening did exactly what it was designed to do — false positives on
negative frames fell 37.5% — and the recall it cost was worth more than the
precision it bought. Details in §7.

Candidate C remains available as an optional high-recall mode.

## 1. False-positive audit

`scripts/audit_false_positives.py`, Candidate C (`e1e373009e4a8c96`) at conf
0.12 over all four locked splits: **1,188 false positives in 1,663 frames
(0.714 per frame)**.

| cause | n | % | per frame | median conf | minable |
|---|---|---|---|---|---|
| player socks / boots | 390 | **32.8%** | 0.235 | 0.231 | yes |
| duplicate candidate | 312 | **26.3%** | 0.188 | 0.182 | **no** |
| penalty spot | 122 | 10.3% | 0.073 | 0.359 | yes |
| pitch line | 101 | 8.5% | 0.061 | 0.311 | yes |
| jersey marking | 83 | 7.0% | 0.050 | 0.200 | yes |
| unknown | 54 | 4.5% | 0.033 | 0.272 | no |
| advertising board | 46 | 3.9% | 0.028 | 0.239 | yes |
| motion-blur artifact | 21 | 1.8% | 0.013 | 0.306 | yes |
| outside pitch | 15 | 1.3% | 0.009 | 0.217 | yes |
| crowd highlight | 14 | 1.2% | 0.008 | 0.155 | yes |
| white seat | 11 | 0.9% | 0.007 | 0.212 | yes |
| compression artifact | 7 | 0.6% | 0.004 | 0.242 | yes |
| broadcast graphic | 7 | 0.6% | 0.004 | 0.230 | yes |
| goal net | 5 | 0.4% | 0.003 | 0.224 | yes |

Per split: public val 0.577/frame, public test 0.770, local val 0.783, local
test 0.652.

**Players account for 39.8%** (socks/boots plus jersey markings) and
**duplicates for 26.3%**. Every category's median confidence sits between 0.15
and 0.36, which is why a threshold move is effective at all.

Duplicates and trajectory-inconsistent candidates are marked **not minable**:
they are fusion failures, not appearance failures, and cropping them would teach
the detector to suppress real balls.

### Temporal persistence

Local clip, 0–120 s, stride 5: **325 candidate tracks over 1,201 sampled
frames. 56% last a single frame; only 15.7% persist five frames or more.** That
is the gap a temporal filter can act on. (No ball ground truth on this clip, so
these are candidate tracks, not confirmed false tracks.)

## 2. Hard-negative mining

`scripts/mine_hard_negatives.py`. **858 crops, fingerprint `9a60241c2b0fd517`**,
from 3,020 training and validation frames. The locked tests are audited but
never mined.

| kind | crops | | kind | crops |
|---|---|---|---|---|
| player socks / boots | 400 | | goal net | 16 |
| jersey marking | 173 | | compression artifact | 15 |
| pitch line | 106 | | white seat | 10 |
| advertising board | 37 | | broadcast graphic | 6 |
| penalty spot | 29 | | crowd highlight | 26 |
| motion-blur artifact | 22 | | outside pitch | 18 |

Rejected during mining: 886 duplicates, **207 crops that contained a real ball**
(rejected rather than emitted with an empty label — that would have trained
suppression of the target object), 191 over the per-kind quota, 86 unknown.

## 3. Confidence calibration — a useful negative result

`scripts/calibrate_candidate_c.py`, fitted on validation only, 5-fold
cross-validated so the reported curves are out-of-fold. Selection rule declared
before fitting: among operating points holding validation recall at ≥95% of
uncalibrated, minimise false positives per frame.

Candidate C, uncalibrated at 0.12: validation recall 0.6856, 0.7119 FP/frame.

| method | point | recall | precision | FP/frame | FP cut |
|---|---|---|---|---|---|
| **fixed threshold** | 0.178 | 0.6528 | 0.5578 | **0.4877** | **31.5%** |
| Platt | 0.146 | 0.6528 | 0.5466 | 0.5103 | 28.3% |
| isotonic | 0.162 | 0.6572 | 0.5385 | 0.5309 | 25.4% |
| size-aware | 0.130 | 0.6616 | 0.5270 | 0.5597 | 21.4% |

**A plain threshold beat all three learned calibrators.** In hindsight this is
structural rather than surprising: Platt scaling and isotonic regression are
*monotone* functions of the confidence they consume, so they cannot reorder
predictions and can only relabel the same operating points. Their apparent
differences here are grid discretisation. Only the size-aware model can reorder,
and it did worse.

For C-Hardened the same procedure selected size-aware at 0.225 (39.6% FP cut on
validation). **That threshold was not applied to the locked test** — the test
had already been scored at the protocol threshold of 0.12, and re-scoring under
a second operating point would be a second look at locked data.

## 4. Temporal false-positive filter

`src/visionpitch/ball_tracking/fp_filter.py`. Implemented and unit-tested; **not
wired into the production pipeline**, so it contributes nothing to the numbers
below. Stated plainly rather than implied.

Its central property is that it **only removes or downgrades**. It never
proposes a position, never fills a gap, and rejected candidates carry no
coordinates — asserted in the tests.

The camera-motion test is the substantive part. Three things behave differently
in image space during a pan: a broadcast overlay stays fixed, a static pitch
feature moves by exactly the camera displacement, and a ball moves independently
of both. So a candidate with near-zero displacement during a pan is painted on
the frame, and one matching the camera's displacement is glued to the world.
Neither is a ball, and both are physical tests rather than appearance heuristics.

States kept separate: `direct`, `temporally_verified`, `rejected`, `unknown`.

## 5. Negative-aware fine-tuning

Candidate C as initialisation, its exact training data plus the 858 mined
crops as empty-label images, 25 epochs, no new augmentation. Candidate D already
measured what heavy augmentation does here, so it was not repeated.

Dataset: 3,392 train (2,534 base + 858 hard negatives), 486 val. Hard negatives
went to **train only** — a validation set stuffed with negatives mined against
this exact model would make every threshold look good.

C-Hardened fingerprint: **`60a138f35a404583`**.

## 6. Locked evaluation

Same scripts, clips, splits and threshold protocol as the A/B/C/D comparison.

| | A default | C | C-Hardened |
|---|---|---|---|
| public centre recall @25 | 0.4901 | **0.7261** | 0.7027 |
| public IoU50 precision | **0.5622** | 0.3953 | 0.4850 |
| public IoU50 F1 | 0.4612 | 0.4777 | **0.5295** |
| tiny-ball recall | 0.3863 | **0.6804** | 0.6510 |
| occluded recall | 0.3521 | **0.5237** | 0.4989 |
| **FP per negative frame** | 0.3913 | 0.5217 | **0.3261** |
| **FP per all frames** | **0.2929** | 0.8882 | 0.5953 |
| local centre recall @25 | 0.5652 | **0.8696** | 0.7826 |
| local 95% CI (n=23) | [0.37,0.74] | [0.68,0.95] | [0.58,0.90] |
| local median centre error | 2.612 px | 1.904 px | **1.852 px** |
| local direct coverage | 0.3225 | **0.3462** | 0.2800 |
| local determinability | **0.0841** | 0.0758 | 0.0471 |
| pass F1 | 0.323 | **0.345** | 0.323 |
| carry F1 | 0.304 | 0.311 | **0.360** |
| runtime | **15.9 ms** | 16.2 | 16.2 |
| peak GPU | **110.8 MB** | 127.7 | 137.9 |

## 7. The eleven criteria

| criterion | result |
|---|---|
| 1. local recall ≥ 0.82 | **FAIL** 0.7826 |
| 2. public recall ≥ 0.68 | pass 0.7027 |
| 3. tiny-ball ≥ 0.62 | pass 0.6510 |
| 4. occluded ≥ 0.47 | pass 0.4989 |
| 5. FP/negative improves materially over C | pass **−37.5%** |
| 6. FP/all ≤ 1.25× default | **FAIL** 2.03× |
| 7. local coverage ≥ 0.34 | **FAIL** 0.2800 |
| 8. determinability ≥ default within uncertainty | **FAIL** −0.0370 (tolerance ±0.0070) |
| 9. pass F1 ≥ C | **FAIL** −0.0220 |
| 10. carry F1 no material regression | pass **+15.8%** |
| 11. runtime ≤ 1.5× | pass 1.01× |

Two criteria said "materially" without a number. Both were pinned before the
results were read: criterion 5 at a ≥10% relative reduction, criterion 10 at a
≤5% relative drop. Both are stated in the artefact so the interpretation is
visible rather than inferred.

# DECISION: KEEP CURRENT DEFAULT

## What failed, and why

**Criterion 1 (local recall 0.7826 vs 0.82).** Two frames of 23. The interval
[0.58, 0.90] overlaps Candidate C's [0.68, 0.95] almost entirely, so this is not
a distinguishable difference — but the criterion was set at 0.82 and 0.7826 is
below it. A 23-frame test cannot support a 0.04 threshold, which is a flaw in
the criterion I wrote, not grounds to waive it now.

**Criterion 6 (FP/all 2.03× default).** Hardening cut FP on true negatives by
37.5% and cut FP over all frames from 0.888 to 0.595 — a 33% improvement on
Candidate C — but the default remains far lower at 0.293.

**Criteria 7, 8 and 9 are the real result.** Local coverage fell to 0.280 (below
even the default's 0.322), determinability to 0.047, and pass F1 gave back
Candidate C's gain. Suppressing false positives suppressed true detections in
the pipeline too: coverage and determinability moved together and downward.

## What was learned

**Hard-negative mining works on the metric it targets.** FP per negative frame
0.5217 → 0.3261 is the single cleanest intervention in this project, and it came
from 858 crops of the model's own mistakes.

**It does not survive the pipeline.** The same suppression that removes a false
sock removes a genuine ball at the edge of detectability, and the trajectory
search has fewer candidates to work with. This is the third distinct instance in
this project of a benchmark gain reversing downstream.

**Learned calibration is the wrong tool for a single monotone score.** Platt and
isotonic cannot beat a threshold sweep on the score they are calibrating. Worth
knowing before reaching for them again.

**The precision/recall trade here is not favourable.** Candidate C buys +0.304
local recall for +33% FP; C-Hardened gives back 0.087 recall for −37.5% FP and
loses more downstream than it gains. Neither sits where the pipeline wants it.

## Preserved

- Current default `fb37942448e7de08` unchanged and still the fallback
- **Candidate C `e1e373009e4a8c96` kept as an optional high-recall mode** —
  best local recall (0.870), best tiny-ball (0.680) and occluded (0.524) recall,
  best pass F1 (0.345)
- C-Hardened `60a138f35a404583` retained
- All 115 annotations, 285 unreviewed samples, QC queue, splits, dataset
  fingerprints and provenance intact

## Reproduction

```bash
python scripts/audit_false_positives.py
```

```bash
python scripts/mine_hard_negatives.py
```

```bash
python scripts/calibrate_candidate_c.py
```

```bash
python scripts/build_hardened_dataset.py
```

```bash
python scripts/finetune_ball.py train --data data/ball_hardened/data.yaml --weights models/finetune/bcast_adapt/weights/best.pt --epochs 25 --imgsz 960 --batch 8 --patience 10 --name bcast_hardened
```

```bash
python scripts/evaluate_broadcast_candidates.py && python scripts/hardening_promotion.py
```
