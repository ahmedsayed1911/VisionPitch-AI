# Phase 2 — Football Intelligence Engine

Turns the Phase 1 game-state reconstruction into possession, events, player and
team analytics, heatmaps, passing networks, a synchronized timeline, a REST API
and a web dashboard.

## The one thing to read first

Phase 2 inherits Phase 1's measurement limits and does not paper over them.
Three numbers set the ceiling on everything below:

| Constraint | Measured | What it caps |
|---|---|---|
| Ball directly observed | 60.2% of frames | every possession and event metric |
| Player rows usable for physics | 33.7% | distance, speed, sprints |
| Possession determinable | 26.0% of analysed time | possession shares |

A possession share of 70% means *70% of the 26% that could be determined*, and
the dashboard says so on the overview. It does not mean 70% of the match.

## Architecture

Analytics is a **separate re-runnable stage**, not a pipeline extension:

```
game_state.parquet ──▶ analytics engine ──▶ analytics/*.{parquet,json} ──▶ API ──▶ dashboard
     (Phase 1, 93s)         (~1.5s)               (artefacts)            (reads only)
```

Three consequences that were the point of the design:

1. **Tuning is a one-second experiment.** Coupling analytics to vision would
   make every threshold change a 93-second round trip.
2. **The dashboard and the exports cannot disagree.** Both read the same
   artefacts; nothing is recomputed at request time.
3. **Phase 1 output is untouched.** Every Phase 1 artefact and test is
   unchanged, so Phase 2 is additive and backward compatible.

```bash
visionpitch analyse match.mp4 --mode balanced
```

```bash
visionpitch analytics outputs/<video_id>/<fingerprint>
```

```bash
visionpitch serve
```

```bash
cd web && npm install && npm run dev
```

## Coverage is a type, not a convention

Every analytics number is a `Metric`, which cannot be serialised without its
coverage:

```json
{
  "value": 46.9, "coverage": 0.675, "confidence": 0.675,
  "n_samples": 405, "basis": "valid_only", "unit": "m", "reportable": true
}
```

`basis` records which rows the number came from:

| Basis | Meaning |
|---|---|
| `valid_only` | only `validation_status == "valid"` rows — required for physical stats |
| `includes_extrapolated` | team aggregates; permitted, and labelled in the UI |
| `event_derived` | counted from events, needs no pitch coordinates |
| `image_space` | counted from tracking without projection |

`Metric.unavailable()` returns `value: null`, never `0`. A player who was never
tracked has *no* distance; reporting zero would be a different claim.

Every player carries the four coverages required by the Phase 1B constraints —
tracking, pitch, ball, identity — and the dashboard shows all four.

## Ball state is explicit everywhere

`BallStateKind` is `observed`, `interpolated` or `unknown`, and travels with
every ball-dependent record. Possession on a frame with an unknown ball is
`PossessionState.UNKNOWN`; it is never carried forward from the last holder.
A test asserts this over the whole real run.

## Possession is measured in image space, not metres

The most consequential decision in Phase 2, and it was forced by measurement.

Projecting the ball onto the pitch assumes it is on the ground. It frequently is
not, and a lofted ball's ground projection races across the pitch in proportion
to its height. Measured on the validation clip, the ball's projected **pitch
speed has a median of 34 m/s** — 123 km/h, sustained — and 83% of frames exceed
any sane "the ball is travelling" threshold. Possession decided in that space
reported the ball loose for 30 of 42 seconds and controlled for 0.07.

Proximity is therefore measured in image pixels **normalised by the player's
bounding-box height**. A footballer is about 1.8 m tall, so one box-height is
about 1.8 m wherever they stand — a calibration-free depth scale immune to both
the homography error and the ball-height error.

Effect on the validation clip:

| | Pitch space | Image-normalised |
|---|---|---|
| Controlled possession | 0.07 s | **13.75 s** |
| Determinable | 0.2% | **26.0%** |

Pitch coordinates are still used for event *geometry* — pass length, progression,
zone entries — where the ball is on the ground.

## Physical statistics use a Kalman/RTS smoother

Differencing per-frame positions does not measure speed, it measures noise.
Measured: the median frame-to-frame step of a *valid* row is 0.717 m, implying
21.5 m/s, when a footballer covers ~0.2 m per frame at 30 fps. Essentially all
of that is measurement error.

`kinematics._rts_smooth` runs a constant-velocity Kalman filter with a
Rauch-Tung-Striebel backward pass, estimating velocity **as part of the state**
rather than differentiating position. Distance is integrated from the smoothed
path. Samples above 12 m/s are discarded rather than clipped, because clipping a
teleport still adds its metres to the total. A unit test asserts the smoother
recovers 3 m/s from a signal whose naive estimate exceeds 8 m/s.

Segments are never bridged: a player untracked for two seconds did not travel in
a straight line during them.

## Events

Derived from **transitions between possession spans**, not from frames. A pass
is the structure "A controlled, the ball travelled, B controlled" — a frame-level
rule fires every time a position jitters.

All 22 required types are implemented. Each event carries timestamp, frame,
track ids, team, confidence, ball coverage, ball state, evidence and a clip
reference. On the 45-second validation clip: 29 touches, 24 carries, 7 passes,
7 turnovers, 4 interceptions, 4 failed passes, 4 long passes, 10 ball-outs.

Things deliberately **not** claimed:

- **Goals** are `goal_candidate`. Confirming a goal needs goal-line evidence
  Phase 1 does not determine.
- **Saves** are `save_candidates`. A save cannot be distinguished from a missed
  shot without the same evidence.
- **Progressive / back passes and zone entries** are withheld entirely when the
  attack direction is unknown, rather than guessed.
- **Restart type** (throw-in, corner, goal kick) is not classified.

## Artefacts

```
<run_dir>/analytics/
    events.parquet          one row per event, with evidence and clip range
    possession.parquet      one row per possession span
    player_stats.json       per-player metrics with four coverages
    goalkeeper_stats.json   goalkeeper analytics
    team_stats.json         team aggregates
    heatmaps.json           precomputed surfaces
    networks.json           passing networks, full match and per half
    timeline.json           filterable, seekable event timeline
    summary.json            overview plus the data-quality header
    manifest.json           analytics schema version and provenance
```

Analytics schema version **2.0.0**, independent of the Phase 1 schema version so
either can evolve alone.

## API

`visionpitch serve` exposes projects, jobs, players, teams, goalkeepers, events,
timeline, heatmaps, networks, reports, CSV/Parquet downloads and video streaming.
Full OpenAPI docs at `/docs`.

Uploads are guarded by extension and an 8 GB cap. Deleting a project removes its
run directories and uploads from disk, not just its rows.

## Measured results

### Analytics on the validation clip

| | Value |
|---|---|
| Events detected | 88 |
| Possession spans | 268 |
| Player profiles | 116 |
| Goalkeepers | 0 (none in this footage) |
| Analytics runtime | 1.5 s |

### Detection and tracking, in-distribution vs out-of-distribution

The generalisation number is the one to quote.

| Metric | In-distribution | **Out-of-distribution (SN-GSR)** |
|---|---|---|
| player precision / recall | 0.976 / 0.988 | **0.688 / 0.727** |
| player mAP50 | 0.979 | **0.620** |
| **ball recall** | ~~0.912~~ **retired, see note** | **0.300** |
| HOTA | — | **0.568** |
| DetA / AssA | — | 0.526 / 0.624 |
| IDF1 | — | **0.711** |
| MOTA | — | 0.574 |
| ID switches / fragmentations | — | 41 / 381 |

In-distribution figures come from the checkpoints' own published test splits
(`martinjolif/*`, CC BY 4.0). Out-of-distribution comes from SoccerNet
SN-GSR-2025, 3 sequences, 600 frames.

> **The in-distribution ball recall of 0.912 is retired.** Phase 2C audited that
> published split and found it is a random *frame* split whose 14 test clips all
> appear in training, so the figure measured memorisation. On a clip-disjoint
> split the same checkpoint scores **0.545** on Roboflow and **0.212** on
> SN-GSR. The player figures on the same row come from the same leaking split
> and should be read with the same caution. See
> [PHASE2C_REPORT.md](PHASE2C_REPORT.md) §1.

**The ball generalisation gap is the headline risk**, and it is wider than this
table suggested. Since possession, passes, turnovers and interceptions all rest
on ball position, Phase 2 output quality on arbitrary broadcast footage is
materially worse than on the validation clip. Phase 2C measured the downstream
consequence directly: possession is determinable on 12% of frames, which caps
pass recall at 0.227.

Reproduce:

```bash
python scripts/benchmark_tracking.py --sequences 3 --max-frames 200
```

## Validation status of every feature

Three levels, applied strictly:

- **Validated** — exercised against real data and checked against ground truth
  or a known-answer test. The number can be quoted with its coverage.
- **Implemented, not yet validated** — the code runs on real data and its
  output is structurally correct, but no ground truth exists to say whether it
  is *right*.
- **Experimental** — known to be incomplete or to rest on evidence the pipeline
  does not have. Do not build on it.

### Vision (Phase 1 / 1B)

| Feature | Status | Evidence |
|---|---|---|
| Player/GK/referee detection | **Validated** | P/R 0.688/0.727, mAP50 0.620 on SN-GSR (OOD) |
| Ball detection | **Validated** — and known weak | clip-disjoint: **0.545** Roboflow, **0.212** SN-GSR. The 0.912 figure is retired (leaking split). |
| Multi-object tracking | **Validated** | HOTA 0.568, IDF1 0.711, MOTA 0.574 on SN-GSR |
| Global tracklet association | **Validated** | 262→92 tracks, median 46.5→73.5 frames |
| Pitch calibration | **Validated** (reference-free) | 97.0% frames, median stability 0.44 m |
| Calibration vs marked landmarks | **Not validated** | needs manual landmark annotation |
| Chunked full-match processing | **Validated** | chunked output matches single-pass at 33.7% usable rows, 0 duplicate rows |
| Team discovery | **Implemented, not yet validated** | silhouette 0.501; assignment accuracy unmeasured |
| Goalkeeper identification | **Experimental** | measurable since Phase 2C (SN-GSR has 3,206 GK annotations) but still unmeasured |
| Jersey number OCR | **Experimental**, disabled by default | ~12×16 px targets |

### Analytics (Phase 2)

| Feature | Status | Evidence |
|---|---|---|
| Kinematics (distance, speed, sprints) | **Validated** | smoother recovers 3 m/s from a signal whose naive estimate exceeds 8 m/s; impossible samples discarded |
| Coverage / `Metric` discipline | **Validated** | enforced by type and asserted over the whole real run |
| Possession state machine | **Implemented, not yet validated** | 26% determinable; no possession ground truth exists |
| Ball touch, carry, pass, turnover | **Implemented, not yet validated** | structurally checked (evidence + clip on every event); no event ground truth |
| Interception, recovery, clearance | **Implemented, not yet validated** | derived from handover geometry |
| Progressive / back / long pass, cross | **Implemented, not yet validated** | withheld entirely when attack direction is unknown |
| **Shot / shot on target** | **Experimental** | inferred from ball direction toward goal; no shot ground truth |
| **Goal events** | **Experimental** | emitted only as `goal_candidate`. A goal cannot be confirmed without goal-line or scoreboard evidence, which the pipeline does not have. **Never treat as a scoreline.** |
| **Save detection** | **Experimental** | reported as `save_candidates` = shots faced. A save is *not* distinguished from a shot that missed, because that also requires goal-line evidence. The count is an upper bound. |
| **Restart classification** | **Experimental — type not classified at all** | `restart` marks first control after the ball left play. Throw-in, corner, goal kick and free kick are **not** distinguished. |
| **Goalkeeper analytics** | **Experimental — never run on real data** | no goalkeeper track exists in any validation clip, so every code path is untested against real input. Distribution accuracy, sweeper actions, cross claims and punches are unimplemented or unexercised. |
| Heatmaps | **Validated** structurally | fixed grid, sample counts carried; `reportable` false under 10 samples |
| Passing network | **Implemented, not yet validated** | depends on pass detection above |
| Timeline | **Validated** | every event carries frame, timestamp and clip range; asserted in tests |
| REST API | **Validated** | integration tests over every endpoint against a real run |
| Dashboard | **Implemented, not yet validated** | builds and typechecks; no browser tests |

## Post-Phase-2 validation review

Two runs were investigated for producing zero usable player rows. They had
different causes.

### `524e5d9707b041fc` — pipeline bug (fixed)

A chunked run emitted 11,304 person rows of which **none** were usable, with no
error anywhere. Calibration had in fact succeeded: 1,310 of 1,350 frames solved.

Two defects compounded:

1. `PipelineResult` had no `support_regions` field, and the chunked merge read
   it via `getattr(result, "support_regions", {})`. The default silently
   returned an empty dict on every chunked run.
2. `extrapolation_risk` returned a middling value for an unknown support
   region, which exceeded the caller's threshold — so *absence of information
   about the support* was treated as *evidence of extrapolation*.

Every row was therefore downgraded to `EXTRAPOLATED`, and every physical metric
became unavailable. The `getattr` default is what made it silent.

**Fixed**: `support_regions` is now a field on `PipelineResult`, the merge uses
direct attribute access so a future regression fails loudly, an unknown region
carries zero risk, and `correct-teams` preserves the original extrapolation
marking instead of promoting every row to `VALID`.

**Verified**: the same run now yields 4,226 `valid` rows and 33.7% usable person
rows — identical to the single-pass run — with zero duplicate rows across chunk
seams. Pinned by `tests/unit/test_extrapolation_regression.py`.

### `ec1d4f1232de0feb` — expected behaviour

A 7-second tail segment (frames 1140–1350) containing 55 player detections
across 191 frames — 0.36 people per frame. The tracker produced 9 raw tracks and
all were dropped by the 5-observation minimum, so the game state legitimately
contains only ball rows. The pipeline correctly reported zero people rather than
inventing them.

Worth knowing: a chunk's own working directory looks like a complete run
directory. When auditing, check `stages.chunking` in the manifest to tell a
merged run from a chunk.

## Known limitations

1. **Ball detection does not generalise** (0.91 → 0.30). Fine-tuning the ball
   detector on more diverse footage is the highest-value next step for Phase 2
   quality.
2. **Possession is determinable for 26% of time** on the validation clip.
3. **Physical statistics rest on 34% of rows** and are lower bounds.
4. **No goalkeeper in the validation footage**, so the goalkeeper engine is
   implemented but unvalidated on real data.
5. **44 of 116 tracks have no confident team**, and are excluded from team
   aggregates.
6. **Halves are not detected** on clips too short to contain a switch, so
   half-filtered views fall back to a single half. This is correct behaviour,
   not a bug, but it means the half filter is inert on short clips.
7. **Saves, goals and restart types are candidates**, not verified outcomes.
