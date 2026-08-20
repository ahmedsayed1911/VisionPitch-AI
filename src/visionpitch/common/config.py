"""Typed configuration, loaded from YAML with mode overlays.

Every threshold, model path, resolution and runtime switch in VisionPitch lives
here. Stage code never contains a magic number it could have read from config.

Layering, lowest precedence first:

1. the field defaults in this module
2. ``configs/default.yaml``
3. ``configs/modes/<mode>.yaml``   (fast_preview | balanced | max_accuracy)
4. explicit ``--set key.path=value`` overrides from the CLI

The fully resolved config is hashed and written into every run manifest, so a
result set can always be traced back to the exact settings that produced it.
"""

from __future__ import annotations

import copy
import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

# --------------------------------------------------------------------------- #


class AnalysisMode(str, Enum):
    FAST_PREVIEW = "fast_preview"
    BALANCED = "balanced"
    MAX_ACCURACY = "max_accuracy"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# --------------------------------------------------------------------------- #
# Stage configs
# --------------------------------------------------------------------------- #


class RuntimeConfig(_Base):
    device: str = Field("auto", description="'auto' | 'cuda' | 'cuda:0' | 'cpu'")
    half_precision: bool = Field(True, description="fp16 inference; ignored on CPU")
    batch_size: int = Field(16, ge=1)
    num_workers: int = Field(4, ge=0)
    seed: int = 1234


class IngestionConfig(_Base):
    #: process every Nth frame. 1 = every frame.
    frame_stride: int = Field(1, ge=1)
    start_time_s: float | None = Field(None, ge=0)
    end_time_s: float | None = Field(None, ge=0)
    #: hard cap for smoke tests; None = whole video
    max_frames: int | None = Field(None, ge=1)
    #: write a checkpoint every N processed frames
    checkpoint_every: int = Field(500, ge=1)
    #: abort if more than this fraction of frames fail to decode
    max_corrupt_frame_ratio: float = Field(0.02, ge=0, le=1)

    @field_validator("end_time_s")
    @classmethod
    def _end_after_start(cls, v: float | None, info) -> float | None:
        start = info.data.get("start_time_s")
        if v is not None and start is not None and v <= start:
            raise ValueError("end_time_s must be greater than start_time_s")
        return v


class DetectionConfig(_Base):
    #: 'football' uses the fine-tuned multiclass model; 'coco' is the offline fallback
    backend: str = Field("football", pattern="^(football|coco)$")
    model_path: str = "models/yolo-football-player-detection.pt"
    #: The shipped checkpoint was fine-tuned at 1280. Inferring above its training
    #: resolution is off-distribution, not "more accurate": measured on the
    #: validation clips, raising this to 1920 *lost* referee detections because the
    #: apparent object scale no longer matches what the model was trained on.
    #: Use ``augment`` for extra accuracy instead. See docs/EVALUATION.md.
    imgsz: int = Field(1280, ge=320)
    #: test-time augmentation - multi-scale + flipped inference, merged. Roughly
    #: 3x the cost, and the correct accuracy lever at native resolution.
    augment: bool = False
    conf_threshold: float = Field(0.25, ge=0, le=1)
    iou_threshold: float = Field(0.5, ge=0, le=1)
    max_detections: int = Field(60, ge=1)
    #: classes below this box area (px) are treated as small-object regime
    small_object_area_px: float = Field(1024.0, gt=0)
    #: per-class confidence floors override conf_threshold when higher
    class_conf_overrides: dict[str, float] = Field(
        default_factory=lambda: {"ball": 0.10, "goalkeeper": 0.30, "referee": 0.30}
    )


class BallFusionConfig(_Base):
    """Candidate fusion between detection and trajectory estimation.

    ``legacy`` is the shipped behaviour and remains the default: IoU
    de-duplication inside ``fuse_detections`` and no temporal verification.
    ``temporal`` routes candidates through ``ball_tracking.fusion`` instead.

    Nothing here changes an existing run. A run recorded before this option
    existed has no ``ball_fusion`` block, loads with these defaults, and
    reproduces exactly as before.
    """

    engine: Literal["legacy", "temporal"] = "legacy"
    suppression: Literal["iou", "centre_distance", "weighted_centre"] = "centre_distance"
    #: Candidates whose centres are within this are one hypothesis. 22 px is
    #: about 1.5 ball diameters on a 1280x720 broadcast, where the measured
    #: median ball is 14 px across.
    merge_radius_px: float = Field(22.0, gt=0)
    temporal_filter_enabled: bool = True
    camera_motion_enabled: bool = True
    camera_cut_reset_enabled: bool = True
    observability_enabled: bool = True
    #: frames of neighbouring support before a candidate is trusted alone
    min_support_frames: int = Field(2, ge=0)
    #: a candidate above this confidence bypasses the persistence requirement,
    #: so a ball reappearing from occlusion is not discarded for having no
    #: neighbours yet
    trust_confidence: float = Field(0.75, ge=0, le=1)
    #: camera displacement below this means the shot is effectively still and
    #: the overlay / background tests say nothing
    camera_motion_px: float = Field(6.0, gt=0)
    #: GMC estimates below this confidence are not used for compensation
    min_camera_confidence: float = Field(0.35, ge=0, le=1)

    def fingerprint(self) -> str:
        import hashlib
        import json

        return hashlib.sha256(
            json.dumps(self.model_dump(), sort_keys=True, default=str).encode()
        ).hexdigest()[:16]


class BallDetectionConfig(_Base):
    """Dedicated high-resolution ball pass.

    Justification: the multiclass checkpoint reports ball mAP50-95 = 0.338 while
    the dedicated ball checkpoint reports 0.551 on its own test split. Running
    the specialist model on a motion-predicted ROI recovers most of that gap for
    a small fraction of a full-frame second pass.
    """

    enabled: bool = True
    model_path: str = "models/yolo-football-ball-detection.pt"
    #: Ball representation: ``box`` is the shipped YOLO detector; ``heatmap``
    #: loads a centre-heatmap checkpoint instead. Default unchanged, so no
    #: existing run, fingerprint or result moves; the alternative exists so the
    #: representation study can be measured end to end rather than on a
    #: benchmark alone.
    representation: Literal["box", "heatmap"] = "box"
    imgsz: int = Field(640, ge=160)
    conf_threshold: float = Field(0.08, ge=0, le=1)
    #: side length in source pixels of the ROI crop centred on the ball prediction
    roi_size_px: int = Field(640, ge=128)
    #: when the ball is lost, fall back to a tiled sweep of the whole frame
    tiled_fallback: bool = True
    tile_rows: int = Field(2, ge=1)
    tile_cols: int = Field(3, ge=1)
    tile_overlap: float = Field(0.2, ge=0, lt=0.9)
    #: Run the tiled sweep every N frames while the ball is lost.
    #:
    #: 1, not 3. Measured on the validation clip: at 3, two thirds of the frames
    #: where the ROI missed got no second look, and the pipeline saw candidates
    #: in 64.8% of frames against the detector's 91.2% recall on stills. Setting
    #: it to 1 took direct ball observation from 53.4% to 60.2% for +17% runtime.
    #:
    #: Lowering the ball confidence floors on top of this was also measured and
    #: *rejected*: it added 3pp of pipeline observations but cost 11pp of
    #: detector precision (0.663 -> 0.550 on the ball test split), and the
    #: trajectory search accepts ~73% of candidates so it is not a strong enough
    #: filter to absorb that. Set the floors manually if you want that trade.
    tiled_every_n_frames: int = Field(1, ge=1)


class TrackingConfig(_Base):
    #: 'bytetrack' is detection-only; 'botsort' adds appearance + motion compensation
    tracker: str = Field("botsort", pattern="^(bytetrack|botsort)$")
    track_high_threshold: float = Field(0.5, ge=0, le=1)
    track_low_threshold: float = Field(0.1, ge=0, le=1)
    new_track_threshold: float = Field(0.6, ge=0, le=1)
    #: frames a track survives without an observation before being closed
    track_buffer: int = Field(60, ge=1)
    match_threshold: float = Field(0.8, ge=0, le=1)
    #: global motion compensation - critical on panning broadcast footage
    gmc_enabled: bool = True
    gmc_method: str = Field("sparseOptFlow", pattern="^(sparseOptFlow|ecc|orb|none)$")
    gmc_downscale: int = Field(2, ge=1)
    #: appearance embedding association
    reid_enabled: bool = True
    reid_weight: float = Field(0.4, ge=0, le=1)
    #: post-processing: discard tracks shorter than this many observations
    min_track_length: int = Field(5, ge=1)
    #: stitch two tracks separated by <= this many frames if they agree spatially
    stitch_max_gap_frames: int = Field(30, ge=0)
    stitch_max_distance_px: float = Field(120.0, ge=0)

    # -- global tracklet association ---------------------------------------- #
    #: 'global' solves all joins simultaneously with the Hungarian algorithm;
    #: 'greedy' is the original pairwise stitcher, kept for A/B comparison.
    association: str = Field("global", pattern="^(global|greedy|none)$")
    #: A player can be occluded for a long time. This is deliberately larger
    #: than stitch_max_gap_frames because the global solver has stronger
    #: evidence available and can afford a wider search.
    assoc_max_gap_frames: int = Field(90, ge=1)
    #: matching rounds; chains of fragments collapse one link per round
    assoc_max_rounds: int = Field(6, ge=1)
    #: Longest gap a join may bridge on image evidence alone. Past this, the
    #: pair must agree in pitch coordinates. Guards against the gap-scaled pixel
    #: allowance growing permissive enough to merge two different players.
    assoc_max_image_only_gap_frames: int = Field(25, ge=0)
    #: joins costing more than this are refused even if they are the best match
    assoc_max_cost: float = Field(0.55, gt=0)
    #: cost weights, applied to terms each normalised to roughly [0, 1]
    assoc_w_geometry: float = Field(0.55, ge=0)
    assoc_w_appearance: float = Field(0.15, ge=0)
    assoc_w_gap: float = Field(0.20, ge=0)
    assoc_w_size: float = Field(0.10, ge=0)
    #: appearance distance above which a join is refused outright
    assoc_appearance_reject: float = Field(0.75, ge=0, le=1)
    #: fastest a player can plausibly travel, for the pitch-space gate
    assoc_max_pitch_speed_m_s: float = Field(9.5, gt=0)
    #: metres of slack added to the pitch gate, absorbing calibration error
    assoc_pitch_slack_m: float = Field(4.0, ge=0)
    #: extra image-space allowance per frame of gap. A flat pixel budget assumes
    #: a static camera; a panning broadcast camera translates the scene by
    #: several pixels per frame independently of player motion.
    assoc_px_per_frame_allowance: float = Field(6.0, ge=0)
    #: frame rate assumed when converting a frame gap to seconds; the pipeline
    #: overwrites this from the video's actual rate
    assoc_fps: float = Field(25.0, gt=0)


class BallTrackingConfig(_Base):
    """Temporal ball trajectory estimation."""

    #: Kalman process noise, in px/frame^2 - governs how hard the filter is pulled
    process_noise: float = Field(9.0, gt=0)
    measurement_noise: float = Field(4.0, gt=0)
    #: reject an association beyond this Mahalanobis gate
    gating_threshold: float = Field(9.5, gt=0)
    #: physically implausible frame-to-frame jump (px), scaled by frame width
    max_speed_px_per_frame: float = Field(140.0, gt=0)
    #: never interpolate across a gap longer than this many frames
    max_interpolation_gap_frames: int = Field(12, ge=0)
    #: below this trajectory-support score the ball is reported as absent
    min_track_confidence: float = Field(0.25, ge=0, le=1)
    #: half-width of the smoothing window used by the offline pass
    smoothing_window: int = Field(5, ge=1)
    #: number of competing hypotheses kept during best-path stitching
    max_hypotheses: int = Field(4, ge=1)
    #: A trajectory segment shorter than this is discarded as a likely
    #: persistent false positive (a penalty spot, a boot).
    #:
    #: Measured on the validation clip: 3 gives 53.4% observed, 2 gives 52.5%,
    #: and 1 gives 46.0%. Lowering it *hurts*, because segments claim frames
    #: exclusively -- an isolated false positive accepted as a one-frame segment
    #: takes ownership of that frame and blocks a genuine longer path from
    #: passing through it. The floor is a precision guard, not a recall cost.
    min_segment_frames: int = Field(3, ge=1)


class RoleConfig(_Base):
    """Thresholds for track-level role resolution.

    Tuned for abstention. The detector's ``referee`` class has poor precision on
    broadcast footage, so a referee call has to clear three independent bars --
    persistence, no confident team assignment, and a kit that actually sits
    outside both team colour clusters -- before it is allowed to overwrite a
    player's team. See team_classification/roles.py.
    """

    referee_min_vote_share: float = Field(0.55, ge=0, le=1)
    referee_min_frames: int = Field(15, ge=1)
    referee_team_veto_confidence: float = Field(0.70, ge=0, le=1)
    referee_min_kit_outlier_ratio: float = Field(1.15, ge=0)
    referee_dominant_vote_share: float = Field(0.85, ge=0, le=1)
    goalkeeper_min_vote_share: float = Field(0.40, ge=0, le=1)
    goalkeeper_min_frames: int = Field(12, ge=1)
    goalkeeper_max_goal_distance_m: float = Field(30.0, gt=0)
    goalkeeper_min_calibrated_samples: int = Field(8, ge=1)
    goalkeeper_max_per_team: int = Field(2, ge=1)


class TeamClassificationConfig(_Base):
    #: 'color' (default) uses hue-saturation histograms; 'embedding' uses a
    #: vision backbone. Colour measured substantially better on broadcast-scale
    #: crops -- see the table in team_classification/embeddings.py.
    method: str = Field("color", pattern="^(embedding|color|colour)$")
    embedding_model: str = "google/siglip-base-patch16-224"
    #: number of crops sampled across the video to fit the clustering model
    fit_sample_size: int = Field(600, ge=50)
    #: How often to harvest a crop from each track. Crops are tiny (a few KB), so
    #: this is cheap; sampling too sparsely starves short tracks of the votes
    #: they need and leaves them permanently UNKNOWN.
    fit_stride_frames: int = Field(5, ge=1)
    #: Global crop budget for one run, in crops. Bounds memory; the harvester
    #: shares it fairly across tracks rather than chronologically, so raising it
    #: buys vote quality, not coverage. A crop is a few KB.
    max_crops: int = Field(30000, ge=100)
    #: Per-track ceiling, so no single long track consumes the budget.
    max_crops_per_track: int = Field(40, ge=1)
    #: jersey region as a fraction of the person box (torso, avoids shorts+socks)
    jersey_top_frac: float = Field(0.15, ge=0, lt=1)
    jersey_bottom_frac: float = Field(0.50, ge=0, le=1)
    jersey_side_margin_frac: float = Field(0.15, ge=0, lt=0.5)
    #: suppress green pitch pixels before computing appearance features
    remove_grass: bool = True
    grass_hue_range: tuple[int, int] = (30, 90)
    grass_sat_min: int = Field(40, ge=0, le=255)
    #: minimum crop size worth classifying
    min_crop_px: int = Field(16, ge=4)
    #: a track's team label needs this fraction of the vote to be accepted
    vote_confidence_threshold: float = Field(0.6, ge=0, le=1)
    min_votes: int = Field(3, ge=1)
    #: goalkeeper discovery
    goalkeeper_penalty_area_frac: float = Field(0.55, ge=0, le=1)
    goalkeeper_min_confidence: float = Field(0.5, ge=0, le=1)
    #: how many nearest players of each team are compared when attaching a
    #: keeper to a side
    goalkeeper_neighbours: int = Field(3, ge=1, le=11)
    roles: RoleConfig = Field(default_factory=RoleConfig)


class CalibrationConfig(_Base):
    enabled: bool = True
    model_path: str = "models/yolo-football-pitch-detection.pt"
    imgsz: int = Field(1280, ge=320)
    #: keypoints below this confidence are not fed to the homography solver
    keypoint_conf_threshold: float = Field(0.35, ge=0, le=1)
    #: A homography needs 4 points mathematically, but a 4-point fit has zero
    #: redundancy: it reproduces its own inputs exactly, so its reprojection
    #: error is uninformative and cannot be used to reject it. 5 is the smallest
    #: value at which the error term means anything. On broadcast footage many
    #: frames genuinely show only 4-9 landmarks, so demanding more than this
    #: trades calibration coverage away very quickly.
    min_keypoints: int = Field(5, ge=4)
    #: RANSAC inlier tolerance, in **metres of pitch** - the units the solver
    #: actually measures residuals in, since the destination points are pitch
    #: coordinates. The keypoint model localises landmarks to a couple of metres
    #: on a wide shot, so a tolerance far below that rejects correct fits.
    ransac_threshold_m: float = Field(2.0, gt=0)
    #: reject the homography if mean reprojection error exceeds this, in metres
    max_reprojection_error_m: float = Field(3.0, gt=0)
    #: below this the frame is marked LOW_CALIBRATION rather than dropped
    min_confidence: float = Field(0.4, ge=0, le=1)
    #: Fill unsolved frames by chaining the camera-motion warp from solved
    #: neighbours. Most unsolved frames are unsolved because the camera happened
    #: to be pointing at featureless grass, not because its pose is unknown.
    propagate_from_motion: bool = True
    max_propagation_frames: int = Field(45, ge=0)
    #: confidence lost per frame of propagation; error compounds each step
    propagation_decay_per_frame: float = Field(0.015, ge=0, le=1)
    #: Projections this far outside the region the landmarks constrained are
    #: marked EXTRAPOLATED. Near the horizon a homography fitted to one corner
    #: of the frame is wrong by tens of metres.
    max_extrapolation_risk: float = Field(0.35, ge=0, le=1)
    #: A frame whose homography places the pitch this far from where its temporal
    #: neighbours place it is rejected. A camera pans and zooms; it does not
    #: teleport. This catches geometrically-nonsense fits that nonetheless have a
    #: low reprojection error against their own few landmarks.
    max_temporal_jump_m: float = Field(8.0, gt=0)
    #: temporal smoothing of the homography across frames
    temporal_smoothing: bool = True
    smoothing_window: int = Field(9, ge=1)
    #: carry the last good homography forward for at most this many frames
    max_carry_forward_frames: int = Field(15, ge=0)
    #: shot-change detection - a large histogram jump invalidates carried calibration
    shot_change_threshold: float = Field(0.45, ge=0, le=1)
    #: run calibration every N frames and interpolate between (camera moves slowly)
    calibrate_every_n_frames: int = Field(1, ge=1)


class ReidConfig(_Base):
    """Player identity foundation. Experimental in Phase 1."""

    jersey_ocr_enabled: bool = False
    #: how many best-view crops per track are pushed through OCR
    crops_per_track: int = Field(24, ge=1)
    #: a digit read needs this confidence to enter the vote
    min_digit_confidence: float = Field(0.5, ge=0, le=1)
    #: a track-level number needs this share of the vote to be assigned
    vote_threshold: float = Field(0.45, ge=0, le=1)
    min_votes: int = Field(4, ge=1)
    #: upscale factor applied to jersey crops before OCR
    upscale_factor: int = Field(4, ge=1)


class ShowcaseConfig(_Base):
    """The presentation overlay: broadcast footage plus tactical structure.

    Separate from :class:`VisualizationConfig` on purpose. The debug renderer is
    tuned for diagnosis and the showcase renderer is tuned to look like the
    reference clip; sharing one set of knobs would mean every visual tweak risked
    the diagnostic view, and vice versa.
    """

    enabled: bool = False
    #: mutate the decoded frame instead of copying it. Safe in the render pass,
    #: which owns its frames, and it removes a full-frame copy per frame.
    in_place: bool = False

    # -- team dots ---------------------------------------------------------- #
    #: Small on purpose. The reference marks a player, it does not highlight one.
    dot_radius_px: int = Field(3, ge=1, le=20)
    dot_outline_px: int = Field(1, ge=0, le=5)
    dot_outline_bgr: tuple[int, int, int] = (15, 15, 15)
    #: RGB. Magenta and blue, matching the reference clip's two team markers.
    team_colors: dict[str, list[int]] = Field(
        default_factory=lambda: {"A": [255, 0, 200], "B": [10, 60, 255]}
    )

    #: Minimum team-eligible players in a frame before anything is drawn.
    #:
    #: Two is the smallest number that can express a tactical relation: one dot
    #: can never carry an edge, so it asserts structure the pipeline has not
    #: actually resolved. Measured on a 134s screen recording, 117 frames
    #: resolved exactly one player and 88 of those were not football at all --
    #: they were the video's promotional end card, where a lone dot was drawn
    #: over an inset clip. The cost in genuine play is 29 frames, about one
    #: second, all of them frames with nothing to connect anyway.
    min_team_players_per_frame: int = Field(2, ge=0)

    #: RGB. Officials are a third semantic class and must not read as either
    #: team, so this is deliberately far from both magenta and blue.
    referee_color: list[int] = Field(default_factory=lambda: [255, 176, 0])

    # -- tactical graph ------------------------------------------------------ #
    #: RGB edge colours per team. The reference uses near-black for one side and
    #: bright green for the other rather than the dot colours, which keeps thin
    #: 1px lines legible against both grass and kit.
    graph_colors: dict[str, list[int]] = Field(
        default_factory=lambda: {"A": [8, 8, 8], "B": [30, 235, 40]}
    )
    graph_thickness_px: int = Field(1, ge=1, le=6)
    graph_opacity: float = Field(0.85, ge=0, le=1)
    #: Delaunay is preferred; kNN is the fallback for degenerate point sets
    graph_knn: int = Field(3, ge=1, le=10)
    #: edges longer than this are tactically meaningless and visually dominant
    graph_max_edge_m: float = Field(20.0, gt=0)
    #: second, scale-free gate: an edge may not exceed this multiple of the
    #: team's own median nearest-neighbour distance. The absolute gate alone
    #: cannot tell a compact block from a team stretched over the whole pitch,
    #: and it is the stretched case that turns a local graph into one network.
    #: Measured on the reference clip, edges run 1.46x local spacing at the
    #: median and 2.87x at p90.
    graph_local_scale: float = Field(2.2, gt=0)
    #: Hard ceiling on how long an edge may *look*, as a fraction of frame
    #: width. Also the sole gate on uncalibrated frames, where there is no
    #: metric distance to test -- and those are the frames that produced the
    #: worst offenders, edges spanning 65 pitch-metres. Set at the reference
    #: clip's p90 (0.304), so a line longer than the reference ever draws is
    #: rejected outright.
    graph_max_edge_image_frac: float = Field(0.30, gt=0, le=2)
    graph_min_calibration_confidence: float = Field(0.3, ge=0, le=1)
    #: consecutive frames an edge must be proposed before it is drawn
    graph_hysteresis_on_frames: int = Field(3, ge=1, le=60)
    #: consecutive frames an edge must be absent before it is dropped
    graph_hysteresis_off_frames: int = Field(8, ge=1, le=120)

    #: the graph only needs the homography to be *currently* supported, not
    #: strongly so; below this it falls back to image-space adjacency
    graph_min_inliers: int = Field(1, ge=0)


class VisualizationConfig(_Base):
    write_annotated_video: bool = True
    write_tactical_map: bool = True
    #: side-by-side broadcast + radar in one file
    write_combined_video: bool = True
    output_fps: float | None = None  # None = inherit source fps
    ball_trail_frames: int = Field(30, ge=0)
    player_trail_frames: int = Field(0, ge=0)
    show_track_ids: bool = True
    show_confidence: bool = False
    radar_width_px: int = Field(1050, ge=200)
    radar_padding_px: int = Field(40, ge=0)
    team_colors: dict[str, list[int]] = Field(
        default_factory=lambda: {
            "A": [235, 64, 52],
            "B": [52, 132, 235],
            "none": [245, 200, 40],
            "unknown": [150, 150, 150],
        }
    )
    #: the presentation overlay; independent of every field above
    showcase: ShowcaseConfig = Field(default_factory=ShowcaseConfig)


class StorageConfig(_Base):
    output_dir: str = "outputs"
    #: 'parquet' is the reference format; json is available for inspection
    format: str = Field("parquet", pattern="^(parquet|json)$")
    compression: str = Field("zstd", pattern="^(zstd|snappy|gzip|none)$")
    write_raw_detections: bool = True
    write_tracks: bool = True
    write_game_state: bool = True
    #: keep stage checkpoints so later phases re-run without re-decoding video
    keep_intermediate: bool = True


class EvaluationConfig(_Base):
    annotations_dir: str = "data/annotations"
    #: IoU at which tracking identity metrics are computed
    tracking_iou_threshold: float = Field(0.5, ge=0, le=1)
    #: mAP sweep
    map_iou_thresholds: list[float] = Field(
        default_factory=lambda: [round(0.5 + 0.05 * i, 2) for i in range(10)]
    )
    #: HOTA alpha sweep
    hota_alphas: list[float] = Field(
        default_factory=lambda: [round(0.05 + 0.05 * i, 2) for i in range(19)]
    )
    #: boxes smaller than this area are the "small object" bucket
    small_object_area_px: float = Field(1024.0, gt=0)


# --------------------------------------------------------------------------- #
# Root
# --------------------------------------------------------------------------- #


class ChunkingConfig(_Base):
    """Bounded-memory processing for long videos."""

    enabled: bool = False
    #: frames each chunk owns. 9000 at 25 fps is six minutes, which keeps peak
    #: memory in the low hundreds of MB while amortising model load.
    chunk_frames: int = Field(9000, ge=10)
    #: Lead-in frames processed but not owned. They let the tracker and
    #: calibrator reach steady state before producing rows anyone keeps, and
    #: give the merger shared frames on which to re-link identities. Too small
    #: and every boundary breaks a track; too large and work is wasted.
    overlap_frames: int = Field(150, ge=0)
    #: maximum box-centre disagreement when re-linking an identity across a seam
    link_max_distance_px: float = Field(60.0, ge=0)
    #: shared frames required before two tracklets may be called the same player
    link_min_shared_frames: int = Field(2, ge=1)


class PitchConfig(_Base):
    length_m: float = Field(105.0, ge=90, le=120)
    width_m: float = Field(68.0, ge=45, le=90)


class Config(_Base):
    mode: AnalysisMode = AnalysisMode.BALANCED
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    ball_detection: BallDetectionConfig = Field(default_factory=BallDetectionConfig)
    ball_fusion: BallFusionConfig = Field(default_factory=BallFusionConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    ball_tracking: BallTrackingConfig = Field(default_factory=BallTrackingConfig)
    team_classification: TeamClassificationConfig = Field(
        default_factory=TeamClassificationConfig
    )
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    reid: ReidConfig = Field(default_factory=ReidConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    pitch: PitchConfig = Field(default_factory=PitchConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    # -- provenance --------------------------------------------------------- #

    def fingerprint(self) -> str:
        """Stable short hash of the resolved config, for the run manifest."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _coerce_scalar(raw: str) -> Any:
    """Parse a CLI override value using YAML scalar rules."""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def _apply_dotted(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node = target
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"override path {dotted_key!r} traverses a non-mapping")
    node[parts[-1]] = value


def _load_with_extends(path: Path, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Read one config file, resolving an ``extends:`` chain beneath it.

    The referenced path is relative to the *including* file's directory, so
    ``configs/showcase.yaml`` says ``extends: default.yaml``. Parents are merged
    first and the extending file wins, matching how mode overlays already read.
    """
    resolved = path.resolve()
    if resolved in _seen:
        chain = " -> ".join(p.name for p in (*_seen, resolved))
        raise ValueError(f"circular config extends: {chain}")

    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping")

    parent_ref = loaded.pop("extends", None)
    if parent_ref is None:
        return loaded

    parent_path = Path(parent_ref)
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    if not parent_path.exists():
        raise FileNotFoundError(
            f"{path.name} extends {parent_ref!r}, which does not exist "
            f"(looked in {parent_path})"
        )
    parent = _load_with_extends(parent_path, (*_seen, resolved))
    return _deep_merge(parent, loaded)


def load_config(
    config_path: str | Path | None = None,
    mode: AnalysisMode | str | None = None,
    overrides: dict[str, Any] | None = None,
    cli_sets: list[str] | None = None,
    config_root: str | Path | None = None,
) -> Config:
    """Resolve the layered configuration into a validated :class:`Config`.

    Parameters
    ----------
    config_path:
        Base YAML. Defaults to ``configs/default.yaml`` relative to ``config_root``.
    mode:
        Analysis mode whose overlay is applied on top of the base file. When
        omitted the base file's own ``mode`` value is used.
    overrides:
        Nested dict merged last-but-one. Used programmatically and by tests.
    cli_sets:
        ``["detection.imgsz=1920", "runtime.half_precision=false"]`` style strings.

    Mode overlays normally win over the base file, which is right for
    ``configs/default.yaml`` but wrong for a self-contained candidate config: a
    file that deliberately pins ``ball_detection.imgsz`` cannot express that,
    because ``configs/modes/balanced.yaml`` pins it too and is merged on top. A
    base file may therefore set ``apply_mode_overlay: false`` to declare that it
    is already fully resolved. The key is opt-in and absent everywhere else, so
    no existing configuration changes behaviour; the mode is still recorded in
    the resolved config and the run manifest.

    A base file may also set ``extends: <path>`` to layer itself on another
    file, resolved relative to its own directory. Without it, passing
    ``config_path`` *replaces* ``configs/default.yaml`` rather than overriding
    it, so a partial file silently drops every setting it does not mention back
    to the field defaults -- which is how ``configs/showcase.yaml``, a
    visualization-only overlay, came to resolve calibration ``imgsz`` to 1280
    instead of the validated 960 and to select the wrong ball checkpoint. The
    failure is silent because both values are individually legal.
    """
    root = Path(config_root) if config_root else Path.cwd()
    base_path = Path(config_path) if config_path else root / "configs" / "default.yaml"

    data: dict[str, Any] = {}
    if base_path.exists():
        data = _load_with_extends(base_path)
    elif config_path is not None:
        raise FileNotFoundError(f"config file not found: {base_path}")

    # Popped before validation: Config forbids extra keys, and this is a
    # directive about how to resolve the file, not a setting of the run.
    apply_overlay = bool(data.pop("apply_mode_overlay", True))

    resolved_mode = mode or data.get("mode")
    if resolved_mode is not None:
        resolved_mode = AnalysisMode(resolved_mode)
        if apply_overlay:
            overlay_path = root / "configs" / "modes" / f"{resolved_mode.value}.yaml"
            if overlay_path.exists():
                overlay = yaml.safe_load(overlay_path.read_text(encoding="utf-8")) or {}
                data = _deep_merge(data, overlay)
        data["mode"] = resolved_mode.value

    if overrides:
        data = _deep_merge(data, overrides)

    if cli_sets:
        patch: dict[str, Any] = {}
        for item in cli_sets:
            if "=" not in item:
                raise ValueError(f"--set expects key.path=value, got {item!r}")
            key, _, raw = item.partition("=")
            _apply_dotted(patch, key.strip(), _coerce_scalar(raw.strip()))
        data = _deep_merge(data, patch)

    return Config.model_validate(data)


def save_config(config: Config, path: str | Path) -> None:
    """Write the fully resolved config next to the results, for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
