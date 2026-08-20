# Output schema

Schema version **1.0.0**, recorded in every Parquet file's Arrow metadata under
`visionpitch_schema_version`.

This is the contract between Phase 1 and everything after it. Phase 2 analytics,
Phase 3 tactical models and the Phase 4 product read these tables and never
re-decode the video.

## `game_state.parquet` — the deliverable

One row per object per frame. Written grouped by frame, so per-frame scans are
sequential.

### Identity

| Column | Type | Null | Meaning |
|---|---|---|---|
| `video_id` | string | no | `<stem>_<content hash>`; changes if the file changes |
| `frame_idx` | int32 | no | **absolute** index in the source video |
| `timestamp_s` | float64 | no | `frame_idx / fps`, seconds from video start |
| `match_clock_s` | float64 | **yes** | match time; always NULL in Phase 1 |

`frame_idx` is absolute, not a sequential counter. With `frame_stride=3` the
10th processed frame is source frame 27; writing 10 would put every downstream
event 0.57 s out of place at 30 fps.

### What was seen

| Column | Type | Null | Meaning |
|---|---|---|---|
| `object_class` | string | no | `player` · `goalkeeper` · `referee` · `ball` |
| `track_id` | int32 | **yes** | NULL for the ball, which has a trajectory not a track |
| `team_id` | string | no | `A` · `B` · `none` (officials, ball) · `unknown` |
| `role` | string | no | `outfield` · `goalkeeper` · `referee` · `ball` · `unknown` |
| `jersey_number` | int32 | **yes** | NULL unless resolved confidently; never invented |
| `jersey_confidence` | float32 | no | 0.0 when unresolved |

`A` and `B` are discovered labels, not clubs. Rename them downstream without
re-running anything.

### Where — image space

| Column | Type | Meaning |
|---|---|---|
| `bbox_x1` `bbox_y1` `bbox_x2` `bbox_y2` | float32 | xyxy pixels |
| `image_x` `image_y` | float32 | the point that was projected |

`image_x, image_y` is the **ground contact point** — bottom-centre for a person,
centre for the ball. Not the box centre: a player's box centre sits ~90 cm above
the pitch, and projecting it through a ground-plane homography introduces a
depth-dependent error of several metres.

### Where — world space

| Column | Type | Null | Meaning |
|---|---|---|---|
| `pitch_x` | float32 | **yes** | metres along pitch length, 0 → `length_m` |
| `pitch_y` | float32 | **yes** | metres across pitch width, 0 → `width_m` |
| `pitch_x_norm` `pitch_y_norm` | float32 | **yes** | same, normalised to [0, 1] |

NULL when the frame has no valid homography, or when the projection landed
implausibly far off the pitch. **A NULL here means "unknown", never "zero".**
Filter on it before computing any physical statistic.

Origin is the bottom-left corner as drawn; `x` runs along the length, `y` across
the width, both in metres, independent of camera and resolution.

### How much to trust it

| Column | Type | Meaning |
|---|---|---|
| `detection_confidence` | float32 | detector score; 0.0 for an interpolated position |
| `tracking_confidence` | float32 | detector score blended with accumulated track support |
| `team_confidence` | float32 | share of the track's team vote; 1.0 if human-corrected |
| `calibration_confidence` | float32 | homography confidence for this frame |

Four numbers, deliberately not collapsed into one. "A confident detection under
a bad homography" and "a weak detection under a good one" are different failures
needing different responses, and a single score cannot express both.

### Provenance

| Column | Type | Meaning |
|---|---|---|
| `interpolated` | bool | position was predicted, not observed |
| `validation_status` | string | see below |
| `segment_kind` | string | `live` · `replay` · `close_up` · `unknown` |
| `source` | string | `tracker` · `ball_trajectory` |

`validation_status`:

| Value | Meaning |
|---|---|
| `valid` | observed, calibrated above the confidence floor |
| `low_calibration` | coordinates present but the homography was weak |
| `no_calibration` | no homography; pitch coordinates are NULL |
| `interpolated` | position inferred, not observed |
| `implausible` | flagged by a temporal consistency check |
| `non_live` | frame classified as replay or close-up |

**Minimum filter for physical analytics:**
```python
df = df[(df.validation_status == "valid") & df.pitch_x.notna()]
```

## `detections.parquet`

Raw detector output before any association: `video_id`, `frame_idx`,
`timestamp_s`, `object_class`, `bbox_*`, `confidence`, `source`.

Kept separate so tracking parameters can be re-tuned and re-run in seconds
without a second detection pass — detection is the expensive stage.

`source` distinguishes `football_multiclass`, `ball_roi`, `ball_tiled`,
`ball_consensus` (both detectors agreed) and `coco_fallback`.

## `tracks.parquet`

One row per track: identity, lifespan, `n_observations`, team/role/jersey with
confidences, and `display_name`.

`display_name` never invents a number. With a number: `Team A - Player #10`.
Without: `Team A - Player A03` — stable across the run and unambiguous to a
reviewer.

This is the table a human edits to correct team assignments; see
`visionpitch correct-teams`.

## `calibration.parquet`

One row per frame: `homography` (row-major 3×3, or NULL), `confidence`,
`reprojection_error_m`, `n_keypoints`, `n_inliers`, `smoothed`, `segment_kind`.

`homography` is a **variable-length** list even though every non-null value has
exactly 9 entries. pyarrow's fixed-size-list encoding does not round-trip nulls
here — a null is written back as a zero-length list and the file then fails to
read entirely. Since uncalibrated frames are normal, the fixed form produced
tables that wrote cleanly and were unreadable afterwards. Use
`storage.tables.homography_from_row` to read the column; it validates length.

## `manifest.json`

Provenance and data quality: video metadata and content hash, config
fingerprint, schema version, library and CUDA versions, every model's path and
**sha256**, per-stage counters and timings, a data-quality block, and a
`requires_manual_review` list.

Read `data_quality` before anything else. It is what tells you whether the rest
of the run is worth reading.

## Reading it

```python
import pandas as pd
df = pd.read_parquet("outputs/<video_id>/<fingerprint>/game_state.parquet")

usable = df[(df.validation_status == "valid") & df.pitch_x.notna()]
ball = df[(df.object_class == "ball") & ~df.interpolated]
```

## Compatibility

Additive changes bump the minor version; any change to the meaning or nullability
of an existing column bumps the major version. Readers should check
`visionpitch_schema_version` and refuse a major version they do not know.
