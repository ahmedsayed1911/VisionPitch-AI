# Phase 2D — Ball Perception Ceiling Removal

**Verdict: NOT READY FOR PHASE 3.** The single smallest remaining blocker is
named in §9.

Phase 2D set out to lift the ball-perception ceiling that Phase 2C identified as
the binding constraint on every event metric. It did not lift it. What it did
instead was establish, with measurement, *why* the obvious routes do not work —
and it found that two of the numbers the milestone was scoped against were
measurement artefacts rather than facts about the pipeline.

Three results carry the phase:

1. **89% of missed balls are not detector blindness.** 64.8% are behind a
   player, 24.3% were found and discarded by the confidence threshold. Only 1.1%
   is a clean, visible ball the detector failed on.
2. **The multi-corpus checkpoint detects the ball far better and makes the
   product worse.** Cross-domain recall +0.232, and effective coverage,
   possession determinability and pass recall all *fall* on unseen broadcast
   footage. The promotion rule rejects it on 6 criteria under both matchers.
3. **Track-before-detect recovery is accurate 57% of the time.** It adds 3.7
   points of coverage by putting 87 confidently wrong ball positions into the
   possession engine. It ships disabled.

---

## 0. Entry audit and reproduction

Performed before any change, as required.

| Check | Result |
|---|---|
| Test suite | 315 tests, all passing |
| Phase 2C cross-domain recall 0.518 / precision 0.636 | **Reproduced exactly** from stored artefacts (0.5182 / 0.6363) |
| Clip leakage, all split pairs | **0 shared clips** — re-derived from filenames, independent of `registry.py` |
| Byte-identical images across splits | **0** of 4,082 distinct images |
| Possession determinability 12% | **Reproduced** (0.1196) |
| Pass F1 0.312, carry F1 0.333 | **Reproduced** bit-identically |

**Code revision cannot be cited**: this repository has no git commits. Every
result below names the script that produces it instead. Initialising a repo and
committing would fix this and is a one-line change, but it is not mine to make
unasked.

## 1. Failure stratification (Part 1)

`scripts/ball_failure_audit.py`, `src/visionpitch/evaluation/ball_failures.py`.
Full method in [BALL_PERCEPTION.md](BALL_PERCEPTION.md).

Each ball is scored twice — at the 0.08 operating threshold and at a 0.001 floor
— so *found and discarded* is separated from *never seen*. Phase 2B could not
make that split and concluded "detector blindness" without sizing it.

Multi-corpus model, held-out test split, 1,110 balls, recall 0.7595. Of 267
misses:

| category | count | % of misses |
|---|---|---|
| player occlusion | 173 | **64.8%** |
| detector threshold rejection | 65 | **24.3%** |
| low contrast | 21 | 7.9% |
| detector miss, unexplained | 3 | 1.1% |
| ball outside visible frame | 3 | 1.1% |
| pitch-line confusion | 1 | 0.4% |
| genuinely unobservable | 1 | 0.4% |

By size: tiny (<150 px²) 0.7176, small 0.8200, medium 0.6700. The tiny-ball
collapse Phase 2B reported (recall 0.00) is gone.

**This redirects the milestone.** A temporal detector architecture, a bigger
backbone, or more training data all address the 1.1%. The 64.8% is a ball that
is not visible, and the 24.3% is a threshold decision.

## 2. The matcher discovery

A broadcast football is ~11 px across; median ball area on the test split is
120 px² (Roboflow) and 221 px² (SN-GSR). At that size IoU ≥ 0.5 demands roughly
two-pixel box accuracy — a *localisation* criterion. The possession engine
compares ball centres against player boxes at radii of tens of pixels and never
uses box extent.

Same model, same split, same threshold:

| matcher | macro recall | macro precision |
|---|---|---|
| IoU ≥ 0.5 | 0.6251 | 0.3266 |
| centre ≤ 25 px | **0.7575** | 0.4071 |

Both are reported everywhere in this phase. **The readiness threshold is judged
against the criterion it was declared with**, and the improvement from 0.518 to
0.6251 is a change of measurement, not of perception — the model is identical.
Presenting it as progress would be exactly the goalpost-moving the brief forbids.

> A bug found and fixed here: the first matcher initialised its best-score to
> `-1` while the centre criterion scores candidates as *negative distance*, so
> only sub-pixel matches were accepted. It surfaced because centre-25 recall
> came out *below* IoU-50 recall, which is arithmetically impossible.

## 3. Observability model (Part 2)

`src/visionpitch/ball_tracking/observability.py`.

Seven labels — visible, occluded, outside frame, hidden by players, motion
blurred, not on pitch, uncertain — of which only visible, blurred and uncertain
count against the detector. **It never produces a ball position**; a unit test
asserts no coordinate appears anywhere in its output. A camera cut resets its
expectation; an expectation that outlives its frame budget or drift cap becomes
`uncertain` rather than a confident label from a stale position.

| corpus | observable fraction | raw detector recall | recall on observable frames |
|---|---|---|---|
| SN-GSR test sequences (6,750 frames) | 0.9422 | 0.674 | **0.714** |
| SN-BAS segment (4,500 frames) | 0.9567 | — | — |

On SN-BAS the breakdown is the useful one: 85.3% of frames are `likely_visible`,
and **49.4% of frames have no ball position at all**. The balls being missed are
overwhelmingly ones the model believes were there to see.

## 4. Track-before-detect recovery (Part 4)

`src/visionpitch/ball_tracking/recovery.py`. Three-frame differencing in a 48 px
window around the interpolated path; a recovery requires 3+ consecutive
mutually-consistent candidates; gaps beyond 15 frames are refused without being
searched; recovered positions are labelled `BallStateKind.RECOVERED`, never
counted as sightings.

Held-out SN-GSR test sequences:

| model | gap frames | recovered | yield | **accuracy** | wrong | unverifiable |
|---|---|---|---|---|---|---|
| multi-corpus | 1,435 | 248 | 17.3% | **0.5735** | 87 | 44 |
| baseline | — | 321 | — | **0.3965** | — | — |

Coverage rises 0.7874 → 0.8241. **It is wrong 43% of the time.** Eighty-seven
confident wrong ball positions handed to the possession engine is worse than 87
frames of honest `UNKNOWN`, because the engine cannot tell the difference and
will build possession spans on them.

`scripts/sweep_recovery.py` sweeps the gating parameters on *training*
sequences against an accuracy-first rule (accuracy ≥ 0.85, then maximise yield)
with an explicit "keep it disabled" outcome declared in advance:

| min frames | min ratio | max deviation | recovered | yield | accuracy |
|---|---|---|---|---|---|
| 3 | 3.0 | 36 px | 12 | 0.141 | 0.333 |
| 3 | 8.0 | 36 px | 12 | 0.141 | 0.333 |
| 5 | 3.0 | 36 px | 5 | 0.059 | 0.200 |
| 5 | 8.0 | 36 px | 5 | 0.059 | 0.200 |
| 3 or 5 | 3.0 or 8.0 | 12 px | **0** | 0.000 | — |

The failure is structural, not a matter of tuning. Tightening the deviation gate
to 12 px produces **zero** recoveries — the evidence peaks simply are not on the
predicted path. Loosening it to 36 px produces recoveries that are wrong two
thirds of the time. There is no setting where the evidence is both plentiful and
correct. (Train-split counts are small — 12 and 5 recoveries — so those accuracy
figures are weak on their own; the 0.5735 over 248 recoveries on the held-out
set is the reliable number, and it points the same way.)

**The stage is not wired into the pipeline.** It exists as an evaluated module
with tests; no config flag enables it, which is the strongest form of disabled.
Enabling it would require a code change and a fresh measurement.

## 5. Promotion decision (Part 10)

`src/visionpitch/evaluation/promotion.py` — the rule as executable, unit-tested
code, so it cannot be reinterpreted while writing up. `scripts/ball_promotion_report.py`
applies it.

Candidate `multicorpus` (`f1ddaf5a465390cf`) against incumbent `baseline`
(`fb37942448e7de08`), config `conf=0.08, imgsz=960`:

| criterion | incumbent | candidate | verdict |
|---|---|---|---|
| cross-domain recall | 0.5252 | **0.7575** | pass (+0.232) |
| worst-domain recall | 0.4413 | **0.6965** | pass (+0.255) |
| roboflow recall | 0.6091 | 0.8185 | pass |
| soccernet_gsr recall | 0.4413 | 0.6965 | pass |
| long-gap fills | 0 | 0 | pass |
| **cross-domain precision** | 0.6339 | 0.4071 | **fail** (floor 0.55) |
| **roboflow precision** | 0.7570 | 0.3807 | **fail** (−0.376) |
| **soccernet_gsr precision** | 0.5108 | 0.4334 | **fail** (−0.077) |
| **effective ball coverage** | 0.4342 | 0.4002 | **fail** (−0.034) |
| **possession determinability** | 0.1196 | 0.1050 | **fail** (−0.015) |
| **downstream pass recall** | 0.2270 | 0.1820 | **fail** (−0.045) |

**Not promoted.** Identical verdict under both matchers.

The lesson is worth stating plainly: **a large cross-domain recall gain made the
product worse.** The extra recall came with a precision collapse, and the
trajectory search — which is the pipeline's precision mechanism — does not
benefit from more candidates when a larger share of them are wrong. Detection
benchmark gains are not pipeline gains, and this phase now has the measurement
to prove it rather than the intuition.

The `baseline` checkpoint remains the default. The multi-corpus checkpoint
remains available and documented at
`models/finetune/ball_multicorpus/weights/best.pt`.

## 6. Operating threshold

Swept on train+val, measured once on test, selection rule declared beforehand.
The rule chose 0.12; on test that loses 2.1 points of centre-25 recall and still
misses the 0.55 precision floor (0.4884). Recall is the binding constraint, so
the change was **declined and 0.08 kept**. Recorded as a measured non-change
rather than omitted.

## 7. Possession determinability (Part 8)

`scripts/possession_determinability.py`, on the SN-BAS segment.

| | baseline | multi-corpus |
|---|---|---|
| ball coverage, direct | 0.4342 | 0.4002 |
| **determinability** (controlled share, unchanged definition) | **0.1196** | 0.1050 |
| unknown ratio | 0.5017 | 0.5281 |
| observable fraction | 0.9567 | 0.9504 |

By ball evidence, baseline:

| evidence | frames | share | determinability |
|---|---|---|---|
| observed | 1,954 | 0.434 | 1.000 |
| interpolated | 325 | 0.072 | 1.000 |
| **unknown** | **2,221** | **0.494** | **0.000** |

This is the whole story in one row. Where the ball position is known,
possession is always determinable. Where it is not, it is never determinable.
There is no possession logic to improve here — the input is absent.

> Two definitions of "determinability" exist in this codebase and they differ by
> a factor of four. The primary one, used above and in every previous phase, is
> `controlled_s / total_s` on smoothed spans. A per-frame "engine committed to
> some state" figure is 0.5064 for the same run. Both are reported in the
> artefact under separate names; quoting the larger one against a threshold set
> for the smaller would be dishonest.

## 8. Downstream ceiling, event engine unchanged (Part 9)

The event engine was re-run with no modifications after the `BallStateKind`
addition. Results are **bit-identical to Phase 2C**, confirming the schema
extension changed nothing:

| | ball observed | pass P | pass R | pass F1 | carry F1 | determinable |
|---|---|---|---|---|---|---|
| baseline | 43.42% | 0.500 | 0.227 | **0.312** | 0.333 | 12.0% |
| multi-corpus | 40.02% | 0.667 | 0.182 | 0.286 | 0.340 | 10.5% |

No perception change in this phase improved any downstream number. That is the
honest headline.

## 9. Verdict

# NOT READY FOR PHASE 3

Thresholds as declared, none moved:

| Criterion | Target | Measured (shipping config) | Met |
|---|---|---|---|
| Cross-domain ball recall | ≥ 0.60 | 0.5252 centre25 / 0.4089 IoU50 | **No** |
| Cross-domain ball precision | ≥ 0.55 | 0.6339 centre25 / 0.4980 IoU50 | **Yes** on centre25 |
| Effective reliable ball coverage | ≥ 0.60 | 0.4342 | **No** |
| Possession determinability | ≥ 0.40 | 0.1196 | **No** |
| Team possession F1 | ≥ 0.75 | 0.685 (perfect perception) | **No** |
| Pass F1, unchanged engine | ≥ 0.45 | 0.312 | **No** |
| Carry F1 | ≥ 0.45 | 0.333 | **No** |
| No severe domain regression | — | default unchanged | **Yes** |
| No hallucination increase | — | recovery disabled; 0 long-gap fills | **Yes** |

Acceptance criteria: ball failures quantified by category and domain (yes);
observability explicitly modelled (yes); at least one temporal/recovery approach
evaluated (yes — track-before-detect, and rejected on measurement); effective
coverage re-measured (yes); determinability re-measured (yes); event metrics
re-run unchanged (yes); promotion followed predeclared rules (yes); no long-gap
hallucinations introduced (yes); all tests pass (yes, 364); no Phase 3 feature
implemented (yes).

### The single smallest remaining blocker

**The ball detector misses roughly half of all frames on unseen broadcast
footage, in frames where the ball was visible.**

Specifically: on SN-BAS, 49.4% of frames have no ball position, while the
observability model rates 85.3% of frames as `likely_visible`. Determinability
is exactly 0.000 on every frame without a ball position and 1.000 on every frame
with one. Nothing downstream — possession logic, event logic, attribution — can
move until that 49.4% falls.

It is a *cross-domain detection* problem specifically, not a detection problem
in general: the same checkpoint reaches 0.787 direct coverage on SN-GSR
sequences and 0.434 on SN-BAS broadcast footage.

### What this phase ruled out, with evidence

Stated so the next attempt does not repeat them:

- **More recall from a stronger detector** — measured. +0.232 cross-domain
  recall produced −0.034 coverage, −0.015 determinability and −0.045 pass recall
  on unseen footage.
- **Lowering the confidence threshold** — measured. Trades recall for precision
  in the wrong direction at every grid point.
- **Recovering sub-threshold evidence along the trajectory** — measured at 57%
  accuracy. Not usable.

### What was not attempted, and why

Parts 3 (temporal detection architectures), 5 (adaptive inference modes) and 6
(domain-robust training curriculum) were **not implemented**. The Part 1
stratification says a detector-capacity intervention addresses the 1.1% of
misses that are unexplained detector failures, and Part 5's promotion evidence
says the last detector improvement made the product worse. Building three more
detector variants before understanding the SN-GSR-to-SN-BAS domain gap would be
spending GPU time on the category the measurement says is smallest.

That is my judgement on sequencing, not a decision that was mine to make alone —
the work remains outstanding and is listed here rather than quietly dropped.

## Reproduction

```bash
python scripts/ball_failure_audit.py --model models/finetune/ball_multicorpus/weights/best.pt --split test
```

```bash
python scripts/ball_threshold_sweep.py --model models/yolo-football-ball-detection.pt
```

```bash
python scripts/evaluate_ball_temporal.py --label baseline --model models/yolo-football-ball-detection.pt
```

```bash
python scripts/possession_determinability.py --run outputs_bas/mid_pre_720p_5505c698690b/6ed3a5e25bcefdd2 --label bas_baseline
```

```bash
python scripts/ball_promotion_report.py --matcher centre25
```
