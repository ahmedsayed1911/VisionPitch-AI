# Ball temporal fusion and candidate verification

**Status: measured, not promoted. Production fusion is unchanged.**

This documents the pipeline audit, the fusion layer built from it, and the
ablation that measures each component. It also states plainly which parts of the
brief were completed and which were not — see §6.

Source-tree fingerprint at completion: `55eec188017ad17d`. Detectors untouched:
default `fb37942448e7de08`, Candidate C `e1e373009e4a8c96`.

---

## 1. Pipeline audit

The production ball path, traced end to end:

```
BallDetector.detect (ROI / tiled)
  → fuse_detections()            multiclass + specialist merge
      → _dedupe(iou_threshold=0.5)
      → cap at max_ball_candidates = 4
  → BallTrajectoryEstimator.estimate()
      → _run_dp() over the candidate lattice
      → _all_paths() peels multi-segment paths by score
      → _fill() interpolates gaps ≤ max_interpolation_gap_frames
  → GameStateAssembler
  → PossessionEngine
  → EventEngine
```

| question | answer |
|---|---|
| where do duplicates enter | `detection/fusion.py::_dedupe`, which suppresses by **IoU at 0.5** |
| where are candidates removed | `_dedupe`, the 4-candidate cap, and `_transition_score` returning `None` |
| where do one-frame tracks survive | `_all_paths` accepts any segment meeting the length floor; a lone high-score node can form its own segment |
| where are low-confidence balls lost | the 4-candidate cap (sorted by confidence) and the `max_speed_px_per_frame` gate |
| where is detector confidence used | `_transition_score`, as the reward term |
| where is temporal confidence used | **nowhere** — no per-frame temporal score exists |
| where is camera motion available | `tracking/gmc.py`, built for the person tracker; **never passed to the ball path** |
| where do camera cuts reset state | **nowhere in the ball path** |
| where is interpolation created | `_fill()`, flagged `interpolated=True` |

**Why the temporal filter was never wired in.** It was written during Phase 2D
Part 4 as a module with unit tests, and Phase 2D's verdict was that
track-before-detect recovery was not usable (57.4% accuracy). The *filter* was
never independently evaluated — it was built alongside a component that failed,
and shipped disabled with it. Its tests all pass and always did.

### The IoU defect

On an ~11 px ball, two detections 8 px apart have IoU near zero, so `_dedupe`
keeps both and the trajectory search sees two hypotheses where there is one
blob. The Candidate C audit measured the consequence: **26.3% of all false
positives were duplicates**, the largest cause after players.

A unit test now pins this exact case.

## 2. The fusion layer

`src/visionpitch/ball_tracking/fusion.py`. Four stages, all of which only ever
remove, merge or annotate — **the layer never proposes a position**, and
rejected candidates carry no coordinates.

1. **Suppression** — `none` / `iou` / `centre_distance` / `weighted_centre`.
   Centre distance does not degrade with object size the way IoU does. Merge
   radius 22 px, about 1.5 ball diameters on this broadcast.
2. **Camera stabilisation** — cumulative displacement in a separate coordinate
   system. **Image coordinates are never overwritten**; cuts reset the offset;
   compensation is skipped when the estimate is below confidence.
3. **Temporal verification** — the Phase 2D filter, now wired.
4. **State assembly** — seven-way output with full provenance.

Provenance retained per merged candidate: source detection ids, merge method,
merged confidence, merged centre, uncertainty radius, number merged.

### Confidence is three separate numbers

| | meaning |
|---|---|
| detector confidence | what the model said |
| temporal confidence | how much the neighbourhood agrees |
| fusion confidence | the combination, and **never higher than the detector's** — temporal agreement corroborates evidence, it does not create it |

## 3. Ablation

`scripts/fusion_ablation.py`, 4 held-out SN-GSR test sequences, 400 frames each,
conf 0.12. The detector runs **once per checkpoint** and its raw candidates are
cached; every row re-scores the same candidates, so differences are the fusion
config alone.

### Candidate C

| ablation | merged | mergeFr | recall | precision | FP/frame | 1-frame | coverage |
|---|---|---|---|---|---|---|---|
| 1 no fusion | 1.000 | 0.000 | **0.7043** | 0.9411 | 0.041 | 0.369 | **0.7006** |
| 2 IoU suppression only | 1.249 | 0.210 | 0.7043 | 0.9411 | 0.041 | 0.369 | 0.7006 |
| 3 centre suppression only | 1.328 | **0.275** | 0.7043 | 0.9411 | 0.041 | 0.369 | 0.7006 |
| 4 persistence only | 1.000 | 0.000 | 0.6789 | 0.9732 | 0.018 | **0.085** | 0.6531 |
| 5 camera motion only | 1.000 | 0.000 | 0.6916 | 0.9401 | 0.041 | 0.379 | 0.6887 |
| 6 trajectory only | 1.000 | 0.000 | 0.6996 | 0.9510 | 0.034 | 0.424 | 0.6887 |
| 7 suppression + temporal | 1.328 | 0.275 | 0.6622 | **0.9735** | **0.017** | 0.158 | 0.6369 |
| 8 full stack | 1.328 | 0.275 | 0.6629 | **0.9735** | **0.017** | 0.172 | 0.6375 |

### Current default detector

| ablation | recall | precision | FP/frame | 1-frame | coverage |
|---|---|---|---|---|---|
| 1 no fusion | **0.2336** | 0.6387 | 0.124 | 0.323 | **0.3425** |
| 4 persistence only | 0.2216 | 0.8760 | 0.029 | 0.225 | 0.2369 |
| 8 full stack | 0.2089 | **0.8842** | **0.026** | 0.207 | 0.2213 |

`merged` = detections collapsed into the selected candidate; `mergeFr` = share of
frames where a merge happened. **These column names were wrong in the first run**
— I had called them "candidates per frame" and "duplicate rate", which they are
not. Corrected here and in the script.

## 4. What the ablation shows

**Centre-distance suppression finds 27.5% of frames carrying a duplicate**, against
IoU's 21.0%. That closely matches the 26.3% duplicate share the false-positive
audit measured independently, from a different method on different data.

**Suppression changes nothing in this ablation** — rows 1, 2 and 3 have identical
recall, precision and coverage. That is a real limitation of the measurement, not
a result: the ablation scores the *single best candidate per frame*, and the
highest-confidence detection wins whether or not its duplicates were merged.
Suppression's value is in what reaches the **trajectory search** — fewer spurious
lattice nodes — and that is not measured here.

**The temporal filter is the component that acts.** On Candidate C it cuts false
positives per frame by **59%** (0.041 → 0.017), lifts precision from 0.9411 to
0.9735, and reduces one-frame tracks from 36.9% to 17.2%.

**And it costs coverage: 0.7006 → 0.6375, −6.3 points.** Recall falls 0.7043 →
0.6629.

## 5. Why this was not promoted

Coverage falling while precision rises is the exact signature of the last three
interventions in this project, every one of which regressed possession
determinability and pass F1 downstream:

| intervention | benchmark effect | downstream effect |
|---|---|---|
| SN-GSR fine-tune (2B) | +0.177 in-domain recall | coverage 43.4% → 37.2% |
| multi-corpus box (2C/2D) | +0.232 cross-domain recall | coverage 43.4% → 40.0% |
| centre heatmap (study) | +0.076 worst-domain recall | coverage 43.4% → 38.3% |
| C-Hardened (hardening) | FP/negative −37.5% | determinability 0.084 → 0.047 |
| **fusion full stack (here)** | **FP/frame −59%** | **coverage −6.3 points, not yet measured downstream** |

Promoting on the fusion-only numbers would repeat a mistake this project has now
made four times. The declared promotion rule requires determinability and pass F1
not to regress, and those have not been measured for this configuration.

**No promotion decision is issued.** Production fusion is unchanged.

## 6. What was not done

Stated plainly rather than left to inference. Of the brief's 14 parts:

**Completed:** pipeline audit (1), duplicate suppression with provenance (2),
temporal filter wired into a fusion layer (3), camera-motion compensation with
cut reset and confidence gating (4), seven-way ball state output (6),
observability-aware empty-frame naming (7), the ablation (10), tests (12), this
document.

**Not completed:**

- **Part 5, multi-hypothesis fusion.** The existing `_all_paths` already peels
  multiple disjoint segments by score, which covers occlusion, reappearance and
  out-of-frame. Beam search, MHT and a Kalman hypothesis bank were not compared.
- **Part 8, the three production modes.** `FusionConfig` fingerprints and carries
  every knob, so modes are expressible, but conservative / balanced / high-recall
  presets were not defined and wired.
- **Part 9, the locked downstream evaluation.** The fusion layer is **not wired
  into the production pipeline**, so A/B/C/D through possession and the event
  engine was not run. This is the decisive measurement and it is missing.
- **Part 11, the promotion decision.** Cannot be issued without Part 9.
- **Part 13, visual debug clips.** Not generated.

The honest summary: the fusion layer is built, unit-tested and measured in
isolation, and the component that matters (temporal verification) shows a large
precision gain and a coverage cost. Whether that trade helps or hurts the product
is unmeasured, and on this project's track record the prior is that it hurts.

## 7. Known limitations

- The ablation scores top-1 candidate selection, so **duplicate suppression is
  untestable in it**. Measuring it needs the trajectory search in the loop.
- Camera shifts come from phase correlation on a 320×180 downscale, not from
  `tracking/gmc.py`. Wiring the real GMC estimates is still outstanding.
- No camera-cut detection is fed to the fusion layer in the ablation; the reset
  path is unit-tested but not exercised on real footage.
- 4 sequences × 400 frames is a small sample; the coverage difference of 6.3
  points is well outside noise, but the precision figures are not tightly bounded.

## Reproduction

```bash
python scripts/fusion_ablation.py --sequences 4 --max-frames 400
```

```bash
python scripts/source_fingerprint.py
```
