# Accuracy bottlenecks and known limitations

Ordered by how much they constrain Phase 2. Each is something observed on real
footage, not a theoretical concern.

> **Updated after Phase 1B.** Limitations 1–4 below describe the *pre-Phase-1B*
> state and are retained for context; the measured position after Phase 1B is in
> [PHASE1B_REPORT.md](PHASE1B_REPORT.md). Current status in one line each:
>
> | Was | Now |
> |---|---|
> | far-side projection coverage 53.4% | **87.9%**, and extrapolated rows are flagged rather than dropped |
> | direct ball observation 48.0% | **60.2%**; remaining ceiling is detector-side |
> | median track 1.55 s, association unverified | **1.87 s**; association measured as **no better than none** out-of-distribution |
> | memory scaled with clip length | chunked full-match processing, tested |
> | detection/tracking accuracy unmeasured | **measured** on public expert corpora |

## After the tiny-ball representation study — the blocker is data, not modelling

The study replaced box detection with a centre-heatmap representation and
measured it end to end. It improved the weaker labelled domain (SN-GSR centre
recall 0.4413 → 0.5177) and **degraded the broadcast corpus** (coverage 0.4342 →
0.3833, carry F1 0.333 → 0.302). Rejected by the declared promotion rule on 3 of
8 criteria.

That is the third consecutive intervention to show the same pattern:

| change | benchmark | SN-BAS coverage |
|---|---|---|
| SN-GSR fine-tune (2B) | +0.177 recall in domain | 43.4% → 37.2% |
| multi-corpus box (2C/2D) | +0.232 cross-domain recall | 43.4% → 40.0% |
| centre heatmap (this study) | +0.076 worst-domain recall | 43.4% → 38.3% |

**The single limitation now dominating everything: there is no labelled ball
data on broadcast footage.** Only two corpora carry ball annotations, and
SN-BAS — where coverage collapses — has none. Improvements can be measured
everywhere except where they matter, and three different techniques have now
improved the measurable and degraded the unmeasurable.

The smallest unblocking action is a bounded annotation task: a few hundred ball
positions on SN-BAS-like footage, enough to measure recall on the target domain
and to validate pseudo-labelling or fine-tuning against something verifiable.

## The ranking after Phase 2D

Phase 2D measured the ball layer directly and narrowed the ranking below to a
single item. Full detail in [PHASE2D_REPORT.md](PHASE2D_REPORT.md) and
[BALL_PERCEPTION.md](BALL_PERCEPTION.md).

1. **Cross-domain ball detection on broadcast footage — the only blocker that
   matters.** On SN-BAS, 49.4% of frames carry no ball position while the
   observability model rates 85.3% of frames as ones where the ball was
   visible. Possession determinability is exactly 0.000 on frames without a ball
   position and 1.000 on frames with one. The same checkpoint reaches 0.787
   direct coverage on SN-GSR sequences and 0.434 on SN-BAS, so this is a domain
   gap, not a capacity gap.
2. **Occlusion, which is not a detector problem at all.** 64.8% of missed balls
   are behind or inside a player box. No detector architecture recovers those;
   only bounded, clearly-labelled temporal inference can, and the attempt at it
   is item 3.
3. **Track-before-detect recovery does not work well enough.** Measured at 57.4%
   accuracy on held-out sequences — 87 confidently wrong positions out of 248.
   Shipped disabled.

Three approaches were **measured and ruled out** in Phase 2D. Recorded so they
are not retried by default:

- a stronger detector: +0.232 cross-domain recall produced −0.034 effective
  coverage, −0.015 determinability and −0.045 pass recall on unseen footage
- a lower confidence threshold: trades recall away at every grid point
- sub-threshold trajectory recovery: 57% accurate

## The ranking after Phase 2C

Phase 2C measured the analytics layer end to end and changed which bottleneck
matters most. In order:

1. **Ball coverage bounds every event number, and nothing else does.**
   Effective ball observation is 40–43% on unseen broadcast footage. Possession
   is therefore determinable on 12% of frames, which caps pass recall at 0.227.
   Every event gain in Phase 2C came from precision; recall did not move at all.
   Cross-domain ball recall is **0.518** against a 0.60 target.
2. **The possession logic is insufficient on its own terms.** Given perfectly
   annotated boxes — no detection, tracking or team error whatsoever — team
   possession F1 is **0.685**, below the 0.75 target. The error concentrates in
   loose-versus-controlled (F1 0.738) and contested (F1 0.356).
3. **Two whole event classes do not work.** `ball_out` and `restart` score
   0.000 F1 in every configuration measured. `header` emits 62 predictions
   against 2 real headers. These are candidates, never detections.
4. **Receiver attribution cannot be measured at all.** No available corpus
   labels who received a pass. Sender attribution, by contrast, is strong:
   holder accuracy **0.978** [0.970, 0.984] under perfect perception.
5. **Projection error** (limitation 1 below) remains, and now has a named
   downstream cost: calibration covers 62% of the SN-BAS clip, which is part of
   why possession is determinable on only 12% of its frames.

A defect class worth recording separately, because it was invisible rather than
inaccurate: several decision rules tested `team_id in ("A", "B")` directly.
Against any corpus using a different team vocabulary they returned False
everywhere, with no error — contested possession was never detected, passes were
never separated from turnovers, and team profiles came back empty. Replaced with
a shared `is_team()` predicate. Hard-coding a value vocabulary inside a decision
rule fails silently and should be treated as a bug pattern, not a style
preference.

## 1. Projection error grows sharply toward the horizon

**The single largest limit on everything physical.**

A monocular ground-plane homography is only as good as its constraints, and on
broadcast football the pitch landmarks visible in any one frame are typically
4–9 points clustered in one small region — a penalty area, or the centre circle.
The homography is well determined there and extrapolates poorly elsewhere,
especially upward in the frame, where the projected position of a point is
extremely sensitive to a fraction of a degree of camera tilt.

Observed: on U-17 frame 700 the radar plotted 2 of 10 detected players. The
other 8 were near the far touchline, high in the frame, and their projections
fell outside plausible bounds and were rejected. That is the system behaving
correctly — refusing to publish a position it cannot trust — but it means
**73% of game-state rows carry pitch coordinates, and the missing 27% are
biased toward far-side players**.

Consequences for Phase 2: distance-covered and speed statistics will
systematically under-sample players on the far side of the pitch. Do not treat
per-player totals as complete without checking each player's coverage.

Mitigations worth trying, in order: a stronger keypoint model fine-tuned on more
data; incorporating pitch *lines* as constraints, not just point landmarks;
propagating a camera model temporally instead of solving each frame
independently; explicit uncertainty propagation so each row carries a positional
error estimate rather than a binary accept/reject.

## 2. Ball recall is the ceiling on all ball-dependent analytics

48% of frames on the validation clip have a directly observed ball; 88% have a
position after bounded interpolation; 12% are explicitly unknown.

Every Phase 2 possession, pass and event metric inherits this. A pass that
begins and ends inside an unknown window cannot be detected at all, and the
possession state machine will spend real time in `unknown`.

The specialist detector and the whole-clip path search both help substantially,
but the underlying problem is physical: a 6–10 px motion-blurred object that is
occluded by players for large stretches.

Worth trying: a temporal detection model that consumes a stack of frames rather
than one; the ball's own trajectory as a detection prior fed back into the
detector, not only into the ROI crop; higher-resolution source footage.

## 3. Team assignment leaves a third of tracks unknown

42 of 120 tracks on the validation clip. Almost all are short tracks that never
accumulated the minimum votes.

This is the intended failure mode — a track with three ambiguous crops should be
`unknown`, not assigned by coin flip — but it does mean team-level aggregates are
incomplete. Better track continuity (limitation 4) is the most effective fix,
since longer tracks gather more votes.

Colour clustering also degrades when: both kits share a hue; the pitch is half
sunlit and half shadowed; or a goalkeeper's kit happens to resemble an outfield
kit. The reported silhouette score is the signal to check — below ~0.15 the
run's team labels should not be trusted, and the pipeline says so.

## 4. Fragmentation is the dominant tracking failure

262 raw tracks became 120 after rejoining 96 fragments. Even after stitching,
median track length is 46 frames — about 1.5 seconds.

Broadcast football is adversarial for appearance-based re-identification: ten
outfield players per side wear identical kit, so appearance can separate *teams*
but not *individuals*. The greedy stitcher rejoins tracks on motion and size
consistency, which fails when a player is occluded for longer than the gap
threshold or changes direction while hidden.

Worth trying: a global (not greedy) association over all track fragments;
short-horizon motion prediction to bridge occlusions; jersey-number recognition
as an identity anchor (see limitation 6).

## 5. Goalkeeper-to-team attribution is a heuristic

Appearance cannot answer it — a keeper's kit is deliberately unlike both
outfield kits, so the two-cluster model has no useful opinion. The current
method attributes a keeper to whichever team is on average nearer to them over
the clip, which works because defenders sit between their keeper and the ball.

It needs calibrated positions, so it fails entirely when calibration coverage is
low, and it reports `unknown` when the margin between teams is under 5%. On
short clips there is often not enough evidence and the answer is `unknown`.

## 6. Jersey-number recognition is experimental and off by default

The data model, the temporal voting and the track-level assignment are complete
and exercised. The recogniser itself is not accurate enough on
broadcast-resolution numbers (~12×16 px, on a curved deforming surface, visible
only when the player faces away) to enable without review, and the brief forbids
inventing a number when confidence is insufficient. Enable with
`reid.jersey_ocr_enabled=true` and the `ocr` extra; treat the output as a
suggestion.

## 7. Match segmentation is minimal

`segment_kind` is inferred from shot-change detection and landmark count only.
It distinguishes "no pitch visible" from "pitch visible" reasonably, and does
**not** reliably identify replays, which in broadcast football look much like
live play. Half boundaries and match start/end are not detected on short clips
and `direction_switch_frames` correctly returns empty rather than guessing.

The full segmentation described in the brief belongs with the Phase 4 product
stage, where whole matches are the input.

## 8. Memory scales with clip length

Detections are sharded to disk during pass 1, but tracks, ball states,
calibration and the assembled rows are held in memory for the whole clip. Fine
for the validation clips; a 90-minute match at 25 fps would need chunked
processing with per-chunk flushing and cross-chunk track stitching. The run
directory and checkpoint layout were designed to accommodate that; the chunking
itself is not implemented.

## 9. Environment-specific findings worth knowing

- **`USAC_MAGSAC` returns `None`** on small, clean correspondence sets whose
  destination coordinates are in metres (OpenCV 5). Verified against an 8-point
  exact-fit case where plain `RANSAC` recovers the camera to 0.0000 m. Since
  broadcast frames routinely offer only 5–9 landmarks, this is the common case
  here. The solver uses `RANSAC` with a least-squares fallback.
- **pyarrow fixed-size lists do not round-trip nulls** in this version: a null
  is written back as a zero-length list and the file then fails to read with
  "Expected all lists to be of size=9". The calibration homography column is
  therefore variable-length with an explicit length check on read.
- **Ultralytics `half=` is deprecated** in favour of `quantize="fp16"`.

## 10. What is validated, and what is not

| Component | State |
|---|---|
| pitch model, geometry, projection | implemented and unit-tested |
| detection (two-model design) | implemented; **accuracy not measured on a VisionPitch clip** |
| tracking | implemented; **accuracy not measured**; fragmentation statistics measured |
| ball trajectory | implemented and unit-tested; coverage measured; **accuracy not measured** |
| team discovery | implemented; separation measured; **assignment accuracy not measured** |
| calibration | implemented; coverage and stability measured; **absolute error not measured** |
| goalkeeper attribution | implemented, experimental |
| jersey numbers | implemented, experimental, disabled |
| evaluation harness | implemented and validated against known-answer synthetic cases |
| storage, resume, manifest | implemented and tested |
| match segmentation | minimal, Phase 1 scope only |

## 11. What Phase 1B did not fix

**Association is not a demonstrated improvement.** The global solver measured
+0.0007 HOTA and +0.0049 IDF1 over doing no association at all — inside the
confidence interval. It is retained because it is no worse and is explicitly
gated and instrumented, not because it was shown to help. Its strongest cue is
pitch-coordinate agreement, and only 27 of ~170 merges on the validation clip
had that evidence available.

**Team assignment still leaves tracks unknown**, though fewer: 25–38 of ~116
depending on configuration, down from 42, as a knock-on from longer tracks
gathering more votes.

**Goalkeeper recall is the weakest class at 0.842**, with a wide interval
[0.684, 1.000] on 25 frames. Goalkeepers are rare in the corpus and rare in
wide broadcast framing.

**Calibration's stability tail widened** (p95 3.25 → 5.19 m) as the cost of
motion propagation. The median is unchanged at 0.44 m and propagated frames are
individually flagged, but a Phase 2 consumer computing per-frame velocity should
filter on `calibration_propagated` where precision matters.

**Match segmentation remains minimal** — see limitation 7. Unchanged by
Phase 1B.
