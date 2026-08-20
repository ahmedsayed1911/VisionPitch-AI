# Phase 2B — Event Validation and Ball Generalization Hardening

**Verdict: NOT READY FOR PHASE 3.** Measured pass F1 is 0.118–0.158. The
detailed basis is at the end of this document.

Phase 2B set out to make the possession and event layer measurable, and to
improve ball generalization. It achieved the first completely and the second
only within one domain. Both results are reported as measured, including the one
that did not work.

---

## 1. Verification of the pre-existing numbers

Every figure carried into this phase was re-derived from stored artefacts before
being used.

| Claim | Verified | Source |
|---|---|---|
| OOD HOTA 0.607, IDF1 0.735 | **Confirmed** | `tracking_global.json`, 6 seqs / 1500 frames, HOTA 0.6068 [0.552, 0.658] |
| OOD ball recall ≈ 0.300 | **Confirmed** at 0.294 | independent re-measurement on the held-out split |
| In-distribution ball recall 0.912 | **Confirmed as reported — and later retired.** Phase 2C found the published Roboflow split shares all 14 test clips with training, so this measured memorisation, not generalisation. Do not quote it. | `ball_det/benchmarks/baseline.json` |

One discrepancy was resolved rather than assumed away: an earlier run reported
HOTA 0.568 / IDF1 0.711 on 3 sequences and 600 frames. Both numbers are correct;
the README quotes the larger, better-powered run. Their confidence intervals
overlap almost completely, so the two are statistically indistinguishable.

A previously documented negative result was also reconfirmed: global tracklet
association (HOTA 0.6068) is **indistinguishable from no association at all**
(0.6061) on out-of-distribution footage.

---

## 2. Event ground truth — acquired, not hand-annotated

The brief allowed manual annotation if no compatible public corpus existed. One
does: **SoccerNet SN-BAS-2025 (Ball Action Spotting)**, ungated, expert-labelled,
with millisecond-precision timestamps on real broadcast video.

| | |
|---|---|
| Clip | Middlesbrough v Preston North End, 2019-10-01, 1280×720 @ 25 fps, 97.9 min |
| Events | **1,603 scorable** |
| Fingerprint | `5241106058a93271` |
| Licence | SoccerNet terms, non-commercial research |

Composition: 700 pass, 554 carry (DRIVE), 127 header, 75 ball-out, 58 restart,
28 interception, 25 shot, 24 cross, 12 turnover.

**What this corpus cannot measure, stated up front:** SN-BAS carries a single
timestamp per action and **no player identity**. Player-attribution accuracy is
therefore reported as `null`, never as a number. A schema-level validation rule
rejects any annotation claiming a player id on a corpus that has none.

The annotation schema (`evaluation/event_gt.py`, version 1.0.0) supports the
full required vocabulary plus `UNKNOWN`, `AMBIGUOUS` and `IGNORE` as
first-class labels, ignore intervals, per-event confidence, and a content
fingerprint that detects out-of-tool editing.

---

## 3. Event evaluation — the first real numbers

3-minute segment (600–780 s), 4500 frames, 49 ground-truth events in window.
Tolerance 0.40 s.

| Event | P | R | **F1** | nGT | nPred | median error |
|---|---|---|---|---|---|---|
| pass_start | 0.188 | 0.136 | **0.158** | 22 | 16 | 0.12 s |
| carry_start | 0.194 | 0.300 | **0.235** | 20 | 31 | 0.20 s |
| header | 0.031 | 1.000 | 0.060 | 2 | 65 | 0.30 s |
| ball_out | 0.000 | 0.000 | 0.000 | 2 | 4 | — |
| restart | 0.000 | 0.000 | 0.000 | 2 | 4 | — |
| turnover | 0.000 | — | — | 1 | 20 | — |

Context: ball observed (analytics-effective) **43.4%**, unknown 49.4%,
possession determinable **12.4%**.

Two measurement bugs were found and fixed while producing this table. Both would
have produced confidently wrong numbers:

1. **Window bug.** Predictions from a 3-minute segment were scored against
   ground truth from the whole 98-minute file, inflating every recall
   denominator ~37×. Now `evaluate_events` takes an explicit window, echoes it,
   and a regression test pins it.
2. **Offset bug.** The run's timestamps are already absolute source-video
   seconds, so an offset of 600 shifted the comparison onto the wrong three
   minutes. The harness now prints the window and the ground-truth events inside
   it before scoring anything, which is how this was caught.

### A defect the measurement exposed

Baseline predicted **36 ball-out and 35 restart events against a ground truth of
2 each**. Root cause was not event logic: 19.5% of projected ball positions land
between the touchline and the implausibility margin, because calibration on this
clip covers only 62.4%. Out-of-play now requires trustworthy calibration *and*
sustained duration.

| | ball_out predicted | restart predicted | pass F1 | carry F1 | determinable |
|---|---|---|---|---|---|
| before | 36 | 35 | 0.118 | 0.227 | 9.5% |
| after | **4** | **4** | **0.158** | **0.235** | **12.4%** |

---

## 4. Ball error analysis — done before any fine-tuning

Part 5 forbids fine-tuning before the failure distribution is known. It was
obeyed, and it changed the conclusion.

The initial taxonomy suggested a large `near_miss_localisation` population.
Direct measurement of the distance from each ground-truth ball to the nearest
prediction showed the distribution is **sharply bimodal**: 48.3% of frames have
a prediction within 12 px, 49.3% within 60 px — a single percentage point in
between.

There is no "found but mislocalised" population. The 60 px threshold was
mislabelling unrelated false positives as near misses. Corrected to 25 px with
the measurement recorded in the code.

**Conclusion: the dominant failure is genuine detector blindness**, not
threshold calibration and not localisation. That is what justified fine-tuning.

Supporting distribution: recall 0.00 for balls under 150 px², 0.84 at
150–400 px², falling again at larger sizes; false positives 1.6 per frame,
concentrated on players and line markings.

---

## 5. Ball fine-tuning — a large in-domain gain

SN-GSR's 49 sequences were split **by sequence** (never by frame — consecutive
frames are near-duplicates and a frame split would leak test data into
training): 26 train / 12 val / **11 held-out test**. Split fingerprint
`9b8525a07da99662`, stable under a hash so adding sequences later cannot move an
existing one out of test.

Held-out test split, never used for training or threshold selection:

| Model | P | **R** | F1 | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| baseline | 0.504 | **0.294** | 0.371 | 0.223 | 0.060 |
| fine-tuned | 0.585 | **0.471** | 0.522 | 0.400 | 0.139 |
| **delta** | **+0.082** | **+0.177** | +0.151 | +0.177 | +0.079 |

Recall rose 60% relative **and precision rose too**, so this is not recall
bought by flooding the trajectory engine with candidates — the constraint the
brief set explicitly.

### The gain did not transfer — a negative result

Applied to SN-BAS, a *third* corpus unrelated to both training and validation:

| | baseline ball model | fine-tuned ball model |
|---|---|---|
| ball observed (analytics) | 43.4% | **37.2%** |
| pass F1 | 0.158 | **0.118** |
| carry F1 | 0.235 | **0.286** |
| possession determinable | 12.4% | 11.7% |

Fine-tuning on SN-GSR produced **domain specialization, not generalization**.
Carry detection improved; pass detection and effective ball coverage regressed.

The fine-tuned checkpoint is therefore **not** promoted to the default. It ships
as `models/yolo-football-ball-detection-gsr.pt`, labelled as a SoccerNet-GSR
domain model. The real lesson is that ball generalization needs *multi-corpus*
training data, not more data from one corpus.

---

## 6. Feature validation status

| Feature | Status |
|---|---|
| Event annotation schema, validation, fingerprinting | **Validated** — 31 unit tests |
| Event matching, tolerances, Wilson intervals | **Validated** — known-answer tests |
| SN-BAS adapter | **Validated** — loads, validates, fingerprints |
| Dataset splits, leak-freedom | **Validated** — tampering detected, stability pinned |
| Ball fine-tuning pipeline | **Validated** — held-out gain measured |
| Pass / carry / ball-out / restart detection | **Implemented, measured, and poor** — F1 0.118–0.286 |
| Possession state machine | **Implemented, not validated** — superseded by Phase 2C, which built a GSR-derived reference and measured team F1 0.685 under perfect perception |
| Player attribution | **Not measurable** — superseded by Phase 2C: holder accuracy 0.978 [0.970, 0.984] under perfect perception |
| Goalkeeper analytics | **Experimental** — superseded by Phase 2C: SN-GSR contains 3,206 goalkeeper annotations, so this is measurable, though still unmeasured |
| Goal / save / restart subtype | **Unsupported** — no ground truth, emitted as candidates only |
| Dashboard event-review workflow | **Not implemented** — see below; **delivered in Phase 2C** |

> **Superseded in places.** Three statements in this document were corrected by
> Phase 2C measurements: SN-GSR *does* carry the team labels, track ids and
> pitch coordinates needed for a possession reference; the in-distribution ball
> recall of 0.912 in §1 came from a leaking frame split and is retired; and the
> possession numbers here were produced with an uncalibrated control radius that
> Phase 2C measured and corrected. See [PHASE2C_REPORT.md](PHASE2C_REPORT.md).

---

## 7. What was not delivered

Stated plainly rather than buried:

- **The dashboard event-review workspace (Part 9) was not built.** The
  correction API for *teams* exists from Phase 1; the event-level review
  workflow does not.
- **Possession was not measured against ground truth** (Part 3). SN-BAS has no
  possession intervals, and the planned GSR-derived reference was not built. The
  possession numbers in this document are coverage, not accuracy.
- **Temporal fusion improvements (Part 7) were not attempted** beyond the
  out-of-play fix, because the ball change did not transfer and re-tuning fusion
  against a non-transferring detector would be premature.

---

## 8. Verdict

# NOT READY FOR PHASE 3

Against the recommended readiness thresholds:

| Criterion | Target | Measured | Met |
|---|---|---|---|
| OOD ball recall above 0.300 | materially above | **0.471** on held-out GSR | **Yes** (in-domain only) |
| Ball recall on an unseen third corpus | improved | **regressed** (43.4% → 37.2% coverage) | **No** |
| Pass F1 on held-out footage | measured and usable | **0.158** | measured, not usable |
| Possession F1 on held-out footage | measured | **not measured** | **No** |
| Player attribution accuracy | measured | **not measurable** with available corpora | **No** |
| UNKNOWN coverage reported | yes | yes, 49.4% | **Yes** |
| No severe unmeasured failure mode | — | ball generalization fails across corpora | **No** |

Phase 3 is xPass and xT. Both are trained on detected passes. A pass F1 of 0.158
means roughly five in six detected passes are wrong or misplaced, so an xPass
model would learn the engine's error distribution rather than football. Training
it now would produce a model that looks plausible and is unfalsifiable.

### The three blockers, in order

1. **Multi-corpus ball training.** The single-corpus fine-tune proved the method
   works (+0.177 recall in domain) and proved one corpus is not enough. Combine
   SN-GSR, the martinjolif corpus and SN-BAS frames into one training set and
   re-measure on all three.
2. **Possession ground truth.** Build the GSR-derived reference to separate
   perception error from logic error, then annotate a possession segment
   properly. Without this, no possession number means anything.
3. **Pass detection F1 above ~0.5** on held-out footage before any xPass work.
   Current 0.158 is not a tuning gap; it is bounded by 43% effective ball
   coverage and 62% calibration on that clip.

## Reproduction

```bash
python scripts/download_eval_data.py gsr
```

```bash
python scripts/ball_error_analysis.py --sequences 6 --max-frames 120
```

```bash
python scripts/finetune_ball.py export --clean && python scripts/finetune_ball.py train --epochs 25 && python scripts/finetune_ball.py evaluate
```

```bash
python scripts/evaluate_events.py --run outputs_bas/<video>/<fingerprint> --gt data/eval/bas/event_gt_half1.json --offset 0 --label baseline
```
