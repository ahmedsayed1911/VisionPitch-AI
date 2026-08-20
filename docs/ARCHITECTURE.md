# Architecture

## Shape of the system

Phase 1 is a two-pass pipeline. Everything that needs pixels happens in pass 1;
everything that can be decided with hindsight happens offline afterwards; video
is only decoded a second time if you asked for rendered output.

```
                    ┌─────────────────────────────────────────┐
   match.mp4  ──▶   │  PASS 1  (one decode, needs pixels)     │
                    │                                         │
                    │  ingestion ─┬─▶ multiclass detector      │
                    │             ├─▶ ball detector (ROI/tiled)│
                    │             ├─▶ pitch keypoints ─▶ homography
                    │             ├─▶ tracker (+ GMC, appearance)
                    │             └─▶ jersey crop harvest      │
                    └────────────────────┬────────────────────┘
                                         │  detections.parquet (sharded)
                    ┌────────────────────▼────────────────────┐
                    │  OFFLINE  (no video access at all)      │
                    │                                         │
                    │  ball trajectory search (whole-clip DP) │
                    │  track stitching + short-track removal  │
                    │  team discovery (cluster + vote)        │
                    │  goalkeeper attribution                 │
                    │  calibration outlier rejection + smooth │
                    │  match-setup inference                  │
                    │  game-state assembly                    │
                    └────────────────────┬────────────────────┘
                                         │  game_state.parquet ── Phase 2+
                    ┌────────────────────▼────────────────────┐
                    │  PASS 2  (optional, rendering only)     │
                    │  annotated.mp4 · radar.mp4 · combined   │
                    └─────────────────────────────────────────┘
```

## Why two passes and not one

The offline stages are offline on purpose. Rule 20 of the brief prefers accuracy
over real-time during initial development, and several decisions are strictly
better with hindsight:

- **Ball trajectory.** A causal tracker must commit at frame *t* to whichever
  blob looks best. Given the whole clip, a false positive can only capture the
  trajectory if doing so improves the score of the entire sequence — which an
  isolated blob never does.
- **Track stitching.** Rejoining track 47 to track 63 requires knowing that 63
  exists, which is information from the future.
- **Team discovery.** Two clusters fitted over crops sampled across the whole
  clip are far more stable than clusters bootstrapped from the first few seconds.
- **Calibration.** A frame's homography can be judged against its neighbours on
  both sides, which is what makes "the camera cannot teleport" enforceable.

## Why crops, not frames, are harvested

Team discovery needs many frames spread across the clip. Caching those frames
costs ~2.7 MB each at 720p and is untenable on a full match. Harvesting the
torso crop at the instant the frame is decoded costs a few KB per player and
carries exactly the same information. The harvester is memory-capped and reports
how many crops it had to drop.

## Module layout

| Module | Responsibility | Key decision recorded in-module |
|---|---|---|
| `common/` | config, canonical schema, types, geometry, logging | four confidences stay separate; nothing collapses them |
| `ingestion/` | decode, sample, time-range, resume | frames are addressed by **absolute source index**, never a sequential counter |
| `detection/` | multiclass + specialist ball detector, fusion | two detectors, because the specialist scores 0.551 ball mAP50-95 vs the multiclass model's 0.338 |
| `tracking/` | Kalman, GMC, appearance, association, offline cleaning | GMC masks detections before fitting, or 22 players moving together *are* the estimated camera motion |
| `ball_tracking/` | constant-acceleration filter, whole-clip path search | multiple disjoint segments, because a ball leaves frame and comes back |
| `team_classification/` | crops, embeddings, clustering, temporal vote, GK attribution | colour beats a learned embedding at broadcast crop scale — measured, see below |
| `calibration/` | keypoints, homography, validation, temporal handling | validation probes the region the landmarks cover, not the image corners |
| `reid/` | jersey number recognition (experimental, off by default) | number is a property of the **track**, never of a frame |
| `game_state/` | join everything; infer attack direction and squad size | an uncalibrated frame stores NULL, never a guessed coordinate |
| `storage/` | run directories, checkpoints, Parquet, manifest | every result traces to a config fingerprint and weight hashes |
| `visualization/` | annotated video, 2D radar | objects with untrustworthy projections are *reported as omitted*, not silently dropped |
| `evaluation/` | detection, tracking, calibration metrics; annotation format | reference-free and ground-truthed metrics are never mixed |
| `pipeline/` | orchestration, correction workflow | corrections rebuild the game state without re-running detection |

## Stage contracts

Every stage consumes and produces plain dataclasses from `common/types.py`, so
any stage can be replaced without touching its neighbours. The two contracts
that are load-bearing:

**Detector** (`detection/base.py`)
```python
detect_batch(images: list[np.ndarray], frame_indices: list[int]) -> list[list[Detection]]
```
This is what keeps the AGPL checkpoints swappable.

**Pitch landmark ordering** (`pitch/geometry.py`)
The 32 vertices are a contract between the keypoint model and the homography
solver. It was verified empirically, not assumed: on frames showing the centre
circle the model returns indices 13–16 and 30–31 (halfway line + circle poles),
and on frames showing a goal it returns 17–26 (penalty area). Those match
`PitchConfiguration.vertices` exactly. `tests/unit/test_geometry_and_pitch.py`
pins the semantics so a future edit cannot quietly break it.

## Failure handling

Rule 18 of the brief: nothing fails silently. Every stage owns a
`StageCounters` recording successes and each distinct failure reason; those
totals land in `manifest.json`. A run that "succeeded" while dropping 12% of its
frames says so, in the manifest and in the CLI summary.

Uncertainty is expressed in the data, not in prose:

| Situation | What is stored |
|---|---|
| no usable homography | `pitch_x = pitch_y = NULL`, `validation_status = no_calibration` |
| homography below confidence floor | coordinates stored, `validation_status = low_calibration` |
| tracker predicted rather than observed | `interpolated = true`, lower `tracking_confidence` |
| ball missing longer than the gap limit | no ball row at all for that frame |
| team vote split | `team_id = unknown`, `team_confidence` = the actual vote share |
| jersey number not resolved | `jersey_number = NULL`, fallback identity `Team A - Player A03` |

## Resume and reuse

A run is one `(video, resolved config)` pair, keyed by a config fingerprint:

```
outputs/<video_id>/<config_fingerprint>/
    config.yaml  manifest.json  summary.json
    detections.parquet  tracks.parquet  calibration.parquet  game_state.parquet
    checkpoints/<stage>/   video/   evaluation/
```

Changing any setting produces a new fingerprint and a new directory, so results
can never be mixed across configurations. Stage state is written with an atomic
replace, so a crash cannot leave a half-written checkpoint that a later run
would trust.

## Known architectural limits

See [LIMITATIONS.md](LIMITATIONS.md). The two that matter most for Phase 2:
projection error grows sharply toward the horizon, and detections/tracks are
held in memory for the whole clip, which is fine for validation clips and will
need chunking for a full 90-minute match.
