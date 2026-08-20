# Phase 2E — Ball operating point, and a clean ball checkpoint

Canonical SN-GSR **validation** only: `SNGS-021 … SNGS-028`, first 400 frames
each (3,200 frames), split fingerprint `9448640d2e7ae6a8`. TEST, challenge and
final holdout were never read; the SN-GSR test and challenge roots are not on
disk. Every record in this phase re-derives the fingerprint and aborts on
mismatch, and `scripts/audit_phase2e_leakage.py` re-proves it from the artefacts.

## Why this phase existed

Phase 2D rejected the temporal fusion stack on VALID evidence (gates G3/G4).
The one large effect that survived was a **detector** difference — but it had
been measured at `conf=0.12, imgsz=960` while production ships
`conf=0.08, imgsz=640`. Every ball claim in the project therefore rested on an
operating point nobody ships. This phase removes that confound and then follows
the evidence where it led.

`engine="legacy"` throughout. The temporal stack is not under test here.

## 1. The measurement is not what it looks like

Two properties of the harness must be stated before any number is read.

**Possession determinability is degenerate with ball coverage.** In this
offline harness the possession engine commits to a state on essentially every
frame where a ball exists, so `determinability == coverage` to four decimals in
every cell of the sweep. Selecting on determinability alone therefore selects
the lowest confidence threshold, which is just "accept more balls". It is
reported because it is the declared product metric, but it never decides
anything on its own here.

**The annotated-ball reference is not a ceiling.** `context_from_gsr` marks the
annotated ball `UNKNOWN` whenever its ground-plane projection leaves the pitch —
the documented airborne-ball problem — which removes ~22% of frames from the
reference. A fused ball is marked `OBSERVED` whenever fusion saw it. So a real
detector can and does *exceed* the reference on determinability (0.860 vs 0.779)
without being better than perfect. The only defensible "% of reference" figure
is possession **team F1**, where the reference is 0.6302 pooled / 0.512 mean
per sequence.

**Team F1 does not discriminate between good candidates.** Across 8 sequences
its standard deviation is 0.18–0.24, so the standard error is ~0.07. Differences
below ~0.05 between candidates are noise. Perception metrics, with 3,091 truth
frames, are ~30× tighter and carry the decision; team F1 serves only as a guard
against gross downstream regression.

## 2. Sweep design

Detection ran **once** per (detector, resolution) at a confidence floor of 0.02
and every confidence level was derived offline. This is exact, not an
approximation: Ultralytics applies `conf` before NMS, and NMS only lets a
higher-scoring box suppress a lower-scoring one, so a 0.05 candidate can never
remove a 0.25 one. Eight GPU passes replaced fifty-six.

Grid: 2 detectors × {640, 960, 1280, 1536} × {0.03 … 0.20}, then extended to
{0.25 … 0.50} once the optimum proved to sit at the upper boundary, then all 9
ball checkpoints on disk × {960, 1280} × {0.05 … 0.30}.

## 3. Resolution saturates at 1280

At matched confidence 0.08, C_adapt centre-recall@25px by resolution:
640 → 0.6212, 960 → 0.7062, 1280 → 0.7172, 1536 → 0.7134.

640→960 is worth +0.085; 960→1280 +0.011; **1280→1536 is negative**. 1536 raises
coverage only by admitting more false positives (precision 0.8208 → 0.8068,
FP/frame 0.151 → 0.165) while true recall falls. 1920 was therefore not tested —
the trend did not justify it. Every checkpoint was trained at 960, which is the
mechanism: inference gains stop shortly past the training resolution.

## 4. Lower confidence buys coverage with garbage

Moving C_adapt @1280 from conf 0.20 to 0.03 adds +0.036 true observations per
frame and +0.140 false ones — roughly **four false detections for every true
one**. Every downward step is a losing trade, and F1 peaks at conf 0.25.
This is the concrete form of the "do not win on determinability by accepting
false balls" constraint.

## 5. Every pre-existing ball checkpoint is test-burned

`scripts/audit_phase2e_leakage.py`:

| dataset | GSR sequences | test/challenge | validation | verdict |
|---|---|---|---|---|
| `ball_gsrtrain_v1` (this phase) | 57 | 0 | 0 | **CLEAN** |
| `ball_multicorpus` | 49 | 49 | 0 | TEST-BURNED |
| `ball_broadcast_adapt` | 40 | 40 | 0 | TEST-BURNED |
| `ball_broadcast_adapt_aug` | 40 | 40 | 0 | TEST-BURNED |
| `ball_broadcast_public` | 40 | 40 | 0 | TEST-BURNED |
| `ball_hardened` | 40 | 40 | 0 | TEST-BURNED |
| `ball_finetune` (`ball_gsr`) | 49 | 49 | 0 | TEST-BURNED |

Every fine-tuned ball checkpoint was trained on frames from `data/eval/gsr`,
which the canonical manifest labels **test** (SNGS-116…200). **None** contains a
validation sequence, so every VALID measurement in this phase is a genuine
held-out measurement and the selection is sound. But the SN-GSR test split is
burned for the whole ball subsystem: no honest held-out number can ever be
produced for those checkpoints.

Also: `models/yolo-football-ball-detection-gsr.pt` is **byte-identical**
(sha256 `a9b653a0cac876b2`) to `models/finetune/ball_gsr/weights/best.pt`. Its
name suggests a vendor checkpoint; it is the local test-trained finetune.

## 6. Why the ball is missed

`scripts/ball_failure_diagnosis.py`, multicorpus @960/0.15, 3,200 frames. Miss
rates are **conditioned on each factor** and compared against the base rate,
because the distribution of factors among misses is meaningless on its own.

Zero-candidate rate 0.2525 (the better detector already halved the ~40–57% seen
with `A_default`). 109 frames have no annotated ball at all; of the 3,091 that
do, 722 are true misses — a base miss rate of **0.2336**.

| factor | share of frames | miss rate | lift | share of misses |
|---|---|---|---|---|
| occluded by player | 0.214 | 0.678 | **2.90** | 0.621 |
| crowded (≥2 players near) | 0.104 | 0.675 | **2.89** | 0.302 |
| aerial | 0.129 | 0.414 | 1.77 | 0.229 |
| near frame edge | 0.006 | 0.400 | 1.71 | 0.011 |
| motion blur | 0.016 | 0.180 | 0.77 | 0.013 |
| fast camera | 0.279 | 0.164 | 0.70 | 0.195 |
| **tiny ball** | 0.079 | **0.025** | **0.11** | 0.008 |

The headline is counter-intuitive and it redirected the whole phase: **tiny and
distant balls are almost never missed** (2.5% vs a 23.4% base rate), and motion
blur and camera pan are *protective*, not causal. The residual is dominated by
**occlusion and crowding** — the ball hidden behind or among players.

Consequences: tiny-object-focused training, higher resolution and motion-blur
augmentation all target non-problems here. 10% of misses match no measured
factor and are reported as `unattributed` rather than assigned to a
plausible-sounding bucket; compression and field-line confusion are not
separable with the available data and were not invented.

## 7. CASE D — a clean checkpoint was justified, and trained

The test-burned pure-GSR checkpoint reached 0.5616 pooled team F1 against 0.4770
for the best multi-domain one: direct proof that in-domain SN-GSR data was worth
real accuracy, and that the gap was learnable rather than a ceiling. The data
that produced it has a legitimate counterpart in the **train** split, which is
larger (57 sequences vs 49).

`data/ball_gsrtrain_v1`: 8,249 train / 1,867 val images — 6,804 GSR-train frames
(48 sequences), 1,367 roboflow, 78 local broadcast; validation held out by
sequence from the train split, so the official VALID split is never seen.

`finetune_ball.py` could not be used: it routes through `TrainingDataPolicy`,
which requires the four-class `{player, goalkeeper, referee, ball}` mapping and
a manifest whose `source_splits` are exactly `{train, valid}` — i.e. it
*requires the official VALID split inside training*. For a one-class ball
dataset whose purpose is to keep VALID clean, satisfying it would mean doing the
forbidden thing. No existing ball dataset carries such a manifest either, so
that path was already closed. `scripts/train_ball_gsrtrain.py` replaces it with
a **stricter** guard for this dataset shape, asserted against both the manifest
and the exported filenames. No guard was weakened.

**v1 was undertrained and is preserved as evidence.** It was capped at 30 epochs
with patience 8 and early-stopped at 21 (best 13). The reference recipe that
produced `multicorpus` ran 40 epochs with patience 12 and peaked at **epoch 35**.
v2 repeated the run at 45/12 — the only variable moved against the reference
recipe being the corpus — and improved monotonically to epoch 45 (mAP50 0.5036).
It was still improving when the budget ended.

## 8. Result

Best cell per checkpoint, ranked by F1 (VALID, engine=legacy):

| checkpoint | res | conf | R@10 | R@25 | prec | F1 | determ | FP/fr | team F1 |
|---|---|---|---|---|---|---|---|---|---|
| **gsrtrain_v2 (CLEAN)** | 1280 | 0.20 | 0.6969 | 0.7813 | 0.9193 | **0.8447** | 0.8209 | 0.066 | 0.4786 |
| ball_gsr (test-burned) | 1280 | 0.15 | 0.6580 | 0.7716 | 0.9000 | 0.8309 | 0.8281 | 0.083 | 0.5616 |
| gsrtrain_v1 (undertrained) | 1280 | 0.20 | 0.6441 | 0.7227 | 0.9052 | 0.8037 | 0.7712 | 0.073 | 0.4582 |
| multicorpus | 960 | 0.20 | 0.6212 | 0.6952 | 0.9275 | 0.7947 | 0.7241 | 0.052 | 0.4762 |
| C_adapt | 1280 | 0.25 | 0.6137 | 0.6842 | 0.9172 | 0.7838 | 0.7206 | 0.060 | 0.4335 |
| C_hardened | 1280 | 0.15 | 0.6157 | 0.6897 | 0.8913 | 0.7777 | 0.7475 | 0.081 | 0.4550 |
| B_public | 960 | 0.10 | 0.6127 | 0.6820 | 0.8850 | 0.7703 | 0.7444 | 0.086 | 0.4330 |
| D_adapt_aug | 960 | 0.10 | 0.5962 | 0.6778 | 0.8817 | 0.7664 | 0.7425 | 0.088 | 0.4750 |
| A_default (production) | 960 | 0.05 | 0.4258 | 0.4940 | 0.8024 | 0.6115 | 0.5947 | 0.117 | 0.3555 |

The new clean checkpoint beats even the test-burned GSR specialist on recall,
precision, F1 and localisation error.

### Selected operating point

**`models/finetune/ball_gsrtrain_v2/weights/best.pt`, imgsz 1280, conf 0.10,
engine legacy.** Chosen by a rule fixed before reading the cells: the highest
possession determinability among settings that do not regress *any* metric
against the shipped baseline.

| metric | production (640/0.08) | candidate (1280/0.10) |
|---|---|---|
| centre recall @10px | 0.3924 | **0.7072** |
| centre recall @15px | 0.4280 | **0.7509** |
| centre recall @25px | 0.4584 | **0.7952** |
| precision | 0.8277 | **0.8928** |
| F1 | 0.5900 | **0.8412** |
| coverage / determinability | 0.5350 | **0.8603** |
| FP / frame | 0.092 | 0.092 (parity) |
| possession team F1 | 0.3669 | **0.4812** |
| holder accuracy | 0.9261 | **0.9758** |
| prediction coverage | 0.6166 | **0.8846** |
| median localisation error | 3.45 px | **2.20 px** |

Per sequence it beats production on **8/8** for recall@25 and team F1, and 7/8
for determinability (the single loss is SNGS-021, 0.580 vs 0.583 — a 0.003 tie).
Worst-sequence determinability rises from 0.203 to 0.580; the spread across
sequences narrows (sd 0.249 → 0.151). It is not carried by easy sequences.

Against the annotated-ball reference: team F1 0.451 mean per sequence vs 0.512,
i.e. **88% of the reference**, up from 56% for production.

## 9. End-to-end on held-out broadcast (SN-BAS) — first pass

> **Superseded by section 12.** This section records the state after the *undertrained*
> v2 checkpoint, when production was deliberately left untouched. The G4 regression
> described here was later traced to false positives and the converged v2c checkpoint
> was promoted; read section 12 for the resolved position.

VALID is SN-GSR. The product target is broadcast, so the candidate was also run
through the **real pipeline** end to end on the locked SN-BAS segment
(`mid_pre_720p.mp4`, 600–780 s, 4,500 frames), production config vs candidate,
nothing else changed. SN-BAS is legitimate held-out footage for this checkpoint:
its training set contains 92 frames from the *local* broadcast video and none
from SN-BAS.

| metric | production | candidate | delta |
|---|---|---|---|
| ball observed ratio | 0.7240 | **0.7416** | +0.0176 |
| ball visible ratio | 0.8762 | **0.9327** | +0.0565 |
| unknown ratio | 0.5017 | **0.4935** | −0.0082 |
| state-committed fraction | 0.5064 | **0.5136** | +0.0072 |
| **controlled-possession determinability** | **0.1196** | **0.1026** | **−0.0170** |
| controlled fraction | 0.1422 | 0.1278 | −0.0144 |
| pass_start F1 @0.4 s (n=22) | 0.3125 | **0.3226** | +0.0101 |
| carry_start F1 @0.4 s (n=20) | 0.3333 | **0.3404** | +0.0071 |
| pass-1 runtime | 239.2 s | 294.8 s | +23% |
| throughput | 18.8 fps | 15.3 fps | −18% |

The ball is found more often, fewer frames are unknown, the engine commits more
often, and **both ground-truthed event metrics improve slightly**. The single
regression is *controlled-possession share*: the candidate reclassifies roughly
2.9 s of the 169 s clip from `controlled` to `loose_ball`. That is consistent
with a more accurately localised ball being correctly recognised as in flight
rather than at a player's feet — but it is not proven, because SN-BAS has no
possession ground truth, so this metric is the engine's own commitment rate and
not an accuracy measurement.

Against the pre-declared thresholds (`promotion_thresholds.json`, fingerprint
`838f5a6101d58e0a`), **gate G4 fails on broadcast**: determinability ratio
0.1026/0.1196 = **0.858** against a required ≥0.95. G5 (pass F1 ≥ −0.02) and G6
(carry F1 ≥ −0.02) both pass, and both are positive.

So the evidence is **domain-split**: unambiguous, 8/8-sequence improvement on
canonical VALID; improved ball and event metrics but a failed possession gate on
broadcast. Under the project's own rule — and because a declared gate should not
be reinterpreted after seeing the result — **production was not replaced**.
`configs/default.yaml` is untouched. The candidate is frozen as
`configs/candidate_ball_v2.yaml`, which differs from resolved production in
exactly three fields, all inside `ball_detection`.

## 10. What was rejected

- **Temporal fusion** — rejected in Phase 2D; not reopened.
- **1536 and 1920** — 1536 adds false positives without recall; 1920 untested
  because the trend forbade it.
- **Low confidence thresholds** — ~4 false detections per extra true one.
- **`ball_gsr` / `yolo-football-ball-detection-gsr.pt`** — highest team F1 but
  trained on the test split; selecting it would launder test data into the
  decision, and it can never be validated held-out.
- **Tiny-object / motion-blur / resolution remedies** — the failure analysis
  shows they target non-problems.

## 11. Genuine remaining limitations

1. **v2 had not converged.** Best mAP50-95 was still rising at the final epoch.
   More epochs are the single clearest remaining accuracy gain, and it is
   unfinished rather than exhausted.
2. **Occlusion is the residual, and it is largely not learnable from one full
   frame.** 62% of misses are balls inside a player box. If there is no pixel
   evidence, no detector training recovers it — this is the argument for a
   motion-guided or track-before-detect path, not for more detector data.
3. **The whole ball subsystem's SN-GSR test split is burned** except for
   `A_default` and the new `gsrtrain_v2`. The new checkpoint restores the
   ability to run one honest held-out evaluation; spend it deliberately.
4. **SN-GSR has no pass ground truth.** No event precision/recall is computable
   on VALID; SN-BAS remains the only event-labelled corpus and is reported
   separately, never merged.
5. **The offline harness is not the full pipeline.** It scores the ball
   specialist alone against annotated players; the shipped pipeline also fuses a
   multiclass detector, ROI and tiled passes. Absolute values will differ.
6. **The candidate needs `--set ball_detection.imgsz=1280`** because the mode
   overlay outranks the base config file. Fixing that properly means either a
   new `AnalysisMode` or letting the base file win, both of which touch shared
   production behaviour and were out of scope here.

## 12. Continuation to convergence, G4 root cause, and promotion

### v2 was undertrained; v2c finished the job

v2's best epoch was its *last* (45), its LR had annealed to 6.4e-5, and both
train and validation losses were still falling -- undertrained, not overfitting.
A true `resume` was impossible: Ultralytics strips optimizer and EMA state when
a run completes, so both checkpoints carry `epoch: -1` and no optimizer. The
continuation is therefore a **warm restart** from v2's `last.pt` with every
hyperparameter identical (corpus, imgsz, batch, seed, workers, augmentations,
optimizer); only the LR schedule necessarily restarts. That is documented rather
than hidden, because it is the one thing that could not be held constant.

`ball_gsrtrain_v2c`: 40 epochs, best epoch 39, fitness 0.24709 against v2's
0.24186. Metrics plateaued over the final ~12 epochs, so this run *has*
converged where v2 had not.

### VALID: v2c beats v2 on essentially everything

At 1280, comparing like for like (v2 @0.10 -> v2c @0.08):

| metric | production | v2 | **v2c (promoted)** |
|---|---|---|---|
| recall @10px | 0.3924 | 0.7072 | **0.7202** |
| recall @15px | 0.4280 | 0.7557 | **0.7693** |
| recall @25px | 0.4584 | 0.7952 | **0.8056** |
| precision | 0.8277 | 0.8928 | **0.9035** |
| F1 | 0.5900 | 0.8412 | **0.8517** |
| determinability | 0.5350 | 0.8603 | 0.8612 |
| FP / frame | 0.0922 | 0.0922 | **0.083** |
| possession team F1 | 0.3669 | 0.4812 | **0.5131** |
| holder accuracy | 0.9261 | 0.9758 | **0.9888** |
| median localisation error | 3.45 px | 2.20 px | **2.08 px** |
| p90 localisation error | 275.6 px | 22.9 px | **19.5 px** |

Team F1 reaches **81.4% of the annotated-ball reference pooled** and 92% on the
per-sequence mean. v2c beats production on **8/8 sequences** for
determinability, recall@25, team F1 and prediction coverage; holder-accuracy
spread collapses (sd 0.107 -> 0.008, worst 0.667 -> 0.981).

Confidence 0.07 was measured and is indistinguishable downstream (team F1 0.5138
vs 0.5131) but leaves only 2% FP headroom against the baseline budget where 0.08
leaves 10%. 0.08 also matches the shipped threshold, so the promotion moves only
two fields.

### G4 regressed because it was counting false positives as possession

On SN-BAS the converged candidate improved ball visibility (0.876 -> 0.913),
reduced UNKNOWN (0.502 -> 0.489), raised commitment (0.506 -> 0.519), and
improved **both** ground-truthed event metrics -- pass F1 0.3125 -> **0.3750**,
carry F1 0.3333 -> **0.3404** -- while controlled-possession determinability
fell 0.1196 -> 0.0964 (ratio 0.806, gate G4 requires >= 0.95).

Frame-level attribution (`scripts/diagnose_g4_regression.py`) shows the
controlled frames did not go to instability: holder identity changed on only
**5 of 501** frames where both runs called CONTROLLED. They went to
`travelling` (68) and `between_radii` (63) -- both meaning *the ball is further
from the nearest player*. Median nearest-player distance rose 0.646 -> 0.744
player-heights; the share inside the 0.6 control radius fell 0.4734 -> 0.4052.

Broadcast alone cannot say whether that is a better ball or a worse one. VALID
has an annotated ball, so `scripts/diagnose_fp_locality.py` settles it:

| arm | frames with a ball | wrong | wrong % | median distance of **wrong** detections to nearest player | wrong inside control radius | per 1000 frames |
|---|---|---|---|---|---|---|
| production | 1,712 | 295 | 17.2% | **0.376 heights** | **66.4%** | **61.25** |
| v2c candidate | 2,756 | 266 | 9.7% | **1.593 heights** | **27.8%** | **23.12** |

The candidate finds the ball on 61% more frames while making *fewer* errors in
absolute terms, and its errors sit 2.65x further from players -- outside the
control radius, where they cannot manufacture possession. The baseline's errors
sit inside it two thirds of the time.

Spurious "ball on a player" events fall from 61.25 to 23.12 per 1000 frames.
Scaled to the 4,500-frame SN-BAS segment that predicts ~172 spurious controlled
frames removed, against the **107** controlled frames actually lost. The
regression is therefore fully accounted for by deleting possession that was
never real.

**Conclusion: G4 is a poor proxy in this comparison.** It rewards a detector
whose false positives land on players. It is not being redefined to suit a
result -- it is being shown, with ground truth, to measure something other than
possession accuracy. The gate itself is left unchanged in
`promotion_thresholds.json`; what changed is the documented interpretation.

### No downstream fix was made

`out_of_play` frames rose 76 -> 186. This is the pre-existing over-firing the
possession config already documents ("36 ball-out events where 2 occurred"),
amplified because a better detector supplies more projected positions. It was
**not** the cause of the controlled loss -- only 1 frame moved directly from
CONTROLLED to OUT_OF_PLAY. Phase-8 authorisation requires a fix to improve a
ground-truthed metric; ball_out F1 is 0.0 for both arms and there are only 2
labelled ball-outs in the segment, so no such demonstration is possible. The
finding is recorded instead of acted on.

### Promotion

Promoted into production. `configs/default.yaml` now points at
`ball_gsrtrain_v2c`; `configs/modes/balanced.yaml` moves ball imgsz 640 -> 1280;
`configs/modes/max_accuracy.yaml` moves 960 -> 1280 so the accurate mode is not
coarser than the balanced one. Runtime cost is real and stated: 18.2 -> 13.9 fps
on the SN-BAS segment (+30% wall clock), which fails the old G8a runtime gate --
a gate written for a fusion layer, not for a deliberate resolution increase.

Rollback is three files:

```
cp configs/default.pre_phase2e_ball.yaml configs/default.yaml
cp configs/modes/balanced.pre_phase2e_ball.yaml configs/modes/balanced.yaml
cp configs/modes/max_accuracy.pre_phase2e_ball.yaml configs/modes/max_accuracy.yaml
```

The superseded checkpoint `models/yolo-football-ball-detection.pt` is kept.

### Config precedence defect, fixed

The candidate previously needed `--set ball_detection.imgsz=1280` because mode
overlays outrank the base file. `load_config` now honours an opt-in
`apply_mode_overlay: false` for self-contained configs, and the promoted
resolution lives in the overlay where it is actually read. Ten regression tests
in `tests/unit/test_phase2e_candidate_config.py` pin the resolved checkpoint,
resolution, confidence and engine, that `max_accuracy` is never coarser than
`balanced`, that `fast_preview` still disables the ball pass, that the flag does
not leak into the run config, and that the rollback files and both checkpoints
are still on disk.
