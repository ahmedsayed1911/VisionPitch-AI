"""Phase 1 pipeline.

Two passes over the video, and no more:

**Pass 1 (analysis)** decodes each frame once and runs everything that needs
pixels -- detection, the specialist ball pass, calibration, tracking (which needs
the image for motion compensation and appearance) -- and harvests small jersey
crops as it goes.

**Offline stages** then run on stored data with no video access at all: ball
trajectory search, track cleaning, team discovery from the harvested crops,
match-setup inference, and game-state assembly.

**Pass 2 (rendering)** decodes again only if visualisation was requested.

Why crops are harvested rather than frames cached
-------------------------------------------------
Team discovery needs many frames spread across the clip. Caching those frames
costs ~6 MB each at 1080p and is untenable on a full match. Harvesting the torso
crop at the moment the frame is decoded costs ~5 KB per player and gives the
clustering stage exactly the same information.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from visionpitch.ball_tracking.kalman import BallKalmanFilter
from visionpitch.ball_tracking.trajectory import BallTrajectoryEstimator
from visionpitch.calibration.calibrator import Calibrator
from visionpitch.common.config import Config
from visionpitch.common.logging import get_logger, progress_bar
from visionpitch.common.schema import GameStateRow, display_name
from visionpitch.common.types import (
    BallState,
    CalibrationResult,
    Detection,
    ObjectClass,
    Track,
)
from visionpitch.detection.ball import build_ball_detector
from visionpitch.detection.fusion import fuse_detections
from visionpitch.detection.yolo import build_detector
from visionpitch.game_state.assembler import GameStateAssembler
from visionpitch.game_state.discovery import MatchSetup, discover_match_setup
from visionpitch.ingestion.video import Frame, VideoMetadata, VideoReader, probe_video
from visionpitch.pipeline.ball_fusion import (
    camera_motion_from_warps,
    gmc_provenance,
    legacy_report,
    run_ball_fusion,
)
from visionpitch.pitch.geometry import PitchConfiguration
from visionpitch.reid.jersey import JerseyNumberRecogniser
from visionpitch.storage.run import RunContext
from visionpitch.storage.tables import (
    calibration_rows,
    detection_rows,
    frame_rows,
    track_rows,
    write_calibration,
    write_detections,
    write_frames,
    write_game_state,
    write_tracks,
)
from visionpitch.team_classification.classifier import TeamClassifier
from visionpitch.team_classification.crops import JerseyCrop
from visionpitch.tracking.postprocess import clean_tracks
from visionpitch.tracking.tracker import MultiObjectTracker
from visionpitch.visualization.annotate import FrameAnnotator, stack_side_by_side
from visionpitch.visualization.radar import PitchRenderer
from visionpitch.visualization.writer import VideoWriter

log = get_logger("pipeline")


@dataclass
class PipelineResult:
    """Everything a caller needs after a run, without re-reading the tables."""

    run_dir: Path
    metadata: VideoMetadata
    frame_indices: list[int] = field(default_factory=list)
    timestamps: dict[int, float] = field(default_factory=dict)
    tracks: dict[int, Track] = field(default_factory=dict)
    ball_states: dict[int, BallState] = field(default_factory=dict)
    calibration: dict[int, CalibrationResult] = field(default_factory=dict)
    detections: dict[int, list[Detection]] = field(default_factory=dict)
    rows: list[GameStateRow] = field(default_factory=list)
    match_setup: MatchSetup | None = None
    stage_timings: dict[str, float] = field(default_factory=dict)
    reports: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    #: Per-frame image region the homography was constrained over.
    #:
    #: Part of the result, not private pipeline state, because the chunked
    #: runner has to merge it across chunks. It previously lived only on the
    #: pipeline object, so the chunked merge read it through
    #: ``getattr(result, "support_regions", {})`` and silently received nothing
    #: on every run -- the default hid a missing attribute rather than failing.
    support_regions: dict[int, tuple[float, float, float, float]] = field(
        default_factory=dict
    )


class _CropHarvester:
    """Collects torso crops during pass 1, under a hard memory budget.

    The budget is shared *fairly*, not chronologically. Two ceilings apply:

    ``max_per_track``
        No track hoards. A player on screen for the whole match contributes the
        same number of votes as one on screen for a minute; beyond a few dozen
        crops the team vote does not move.

    ``max_crops``
        The global memory bound. When it is reached the harvester does **not**
        start refusing new crops -- it evicts one from whichever track currently
        holds the most. Refusing is what made this budget chronological: on the
        528 s broadcast the pipeline wants ~48,700 crops against a 12,000 cap, so
        a first-come budget was spent in the opening minutes and every track born
        afterwards harvested *zero* crops and was forced to UNKNOWN. Measured on
        the full video: 63.9% UNKNOWN with neither ceiling, 53.5% with the
        per-track ceiling alone (which only binds on tracks long enough to reach
        it), and the eviction policy is what makes the budget independent of when
        a track happens to appear.
    """

    def __init__(
        self,
        classifier: TeamClassifier,
        stride: int,
        max_crops: int = 20000,
        max_per_track: int = 40,
    ) -> None:
        self.classifier = classifier
        self.stride = max(1, stride)
        self.max_crops = max(1, max_crops)
        self.max_per_track = max(1, max_per_track)
        self._by_track: dict[int, list[JerseyCrop]] = {}
        self._total = 0
        self.dropped_track_cap = 0
        self.evicted = 0

    # -- budget -------------------------------------------------------------- #

    @property
    def crops(self) -> list[JerseyCrop]:
        return [crop for crops in self._by_track.values() for crop in crops]

    @property
    def per_track(self) -> dict[int, int]:
        return {track_id: len(crops) for track_id, crops in self._by_track.items()}

    def _evict_one(self) -> None:
        """Drop the least informative crop held by the largest holder.

        Two choices matter here. Evicting from the *largest* holder is what makes
        the budget independent of arrival order -- a track that appears in the
        last minute takes its slots from whoever is hoarding, not from nobody.
        And evicting the most *redundant* crop within that track, rather than the
        oldest or the middle one, keeps its samples spread across the clip: a
        track's crops are only a vote if they sample the whole appearance, and
        repeatedly dropping from one position collapses them onto the ends.
        """
        donor_id = max(self._by_track, key=lambda k: len(self._by_track[k]))
        donor = self._by_track[donor_id]
        if len(donor) <= 2:
            donor.pop()
        else:
            # The crop whose removal leaves the smallest gap behind.
            index = min(
                range(1, len(donor) - 1),
                key=lambda i: donor[i + 1].frame_idx - donor[i - 1].frame_idx,
            )
            donor.pop(index)
        if not donor:
            del self._by_track[donor_id]
        self._total -= 1
        self.evicted += 1

    # -- collection ---------------------------------------------------------- #

    def harvest(self, frame: Frame, tracks: list) -> None:
        if frame.idx % self.stride:
            return
        for track in tracks:
            # Referee-class tracks are harvested as well. Their team vote is the
            # only evidence that can overrule a false referee call, and a track
            # with no crops cannot be defended.
            if track.object_class not in (
                ObjectClass.PLAYER,
                ObjectClass.GOALKEEPER,
                ObjectClass.REFEREE,
            ):
                continue
            held = self._by_track.setdefault(track.track_id, [])
            if len(held) >= self.max_per_track:
                self.dropped_track_cap += 1
                continue
            crop = self.classifier.extractor.extract(
                frame.image, track.bbox_array.tolist(), track.track_id, frame.idx
            )
            if crop is None:
                continue
            held.append(crop)
            self._total += 1
            # Accept first, then rebalance. Refusing at the door is what tied the
            # budget to arrival order in the first place.
            while self._total > self.max_crops:
                self._evict_one()

    def remap(self, id_map: dict[int, int], valid_ids: set[int]) -> None:
        """Follow track stitching, and discard crops whose track was removed."""
        merged: dict[int, list[JerseyCrop]] = {}
        for track_id, crops in self._by_track.items():
            new_id = id_map.get(track_id, track_id)
            if new_id not in valid_ids:
                continue
            for crop in crops:
                crop.track_id = new_id
            merged.setdefault(new_id, []).extend(crops)
        for crops in merged.values():
            crops.sort(key=lambda c: c.frame_idx)
        self._by_track = merged
        self._total = sum(len(c) for c in merged.values())

    def by_track(self) -> dict[int, list[JerseyCrop]]:
        return {track_id: list(crops) for track_id, crops in self._by_track.items()}

    def report(self) -> dict:
        counts = [len(c) for c in self._by_track.values()]
        return {
            "tracks_harvested": len(self._by_track),
            "crops_kept": self._total,
            "crops_evicted": self.evicted,
            "crops_dropped_track_cap": self.dropped_track_cap,
            "min_crops_per_track": min(counts) if counts else 0,
            "median_crops_per_track": (
                sorted(counts)[len(counts) // 2] if counts else 0
            ),
        }


class Phase1Pipeline:
    """Runs the full Phase 1 vision foundation."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.pitch = PitchConfiguration(
            length=config.pitch.length_m, width=config.pitch.width_m
        )
        self._timings: dict[str, float] = {}
        self._support_regions: dict[int, tuple[float, float, float, float]] = {}
        self._effective_fps: float = 25.0
        #: Person boxes per frame, kept **only** when the temporal ball-fusion
        #: engine needs the observability model. ~350 B/frame, so a full match
        #: costs a few MB; not worth paying on the default path, which never
        #: reads it.
        self._person_boxes: dict[int, np.ndarray] = {}

    # -- helpers ------------------------------------------------------------ #

    def _time(self, stage: str, start: float) -> None:
        self._timings[stage] = round(time.perf_counter() - start, 2)

    # -- entry point -------------------------------------------------------- #

    def run(
        self,
        video_path: str | Path,
        resume: bool = True,
        render: bool | None = None,
    ) -> PipelineResult:
        overall_start = time.perf_counter()

        metadata = probe_video(video_path)
        run = RunContext(self.config, metadata).ensure()
        log.info("run directory: %s", run.root)

        result = PipelineResult(run_dir=run.root, metadata=metadata)

        # ---------------- pass 1: everything that needs pixels ------------- #
        pass1 = self._analysis_pass(run, metadata, result)
        tracker, calibrator, harvester = pass1

        # ---------------- offline stages ------------------------------------ #
        # Association uses pitch coordinates when calibration supports them, so
        # calibration must be finalised before tracks are merged.
        # Effective frame rate for the association speed gate. Kept beside the
        # config rather than written into it: the config's fingerprint is the
        # run's provenance key and is recorded in the manifest, so mutating it
        # mid-run makes the stored manifest disagree with the config that
        # produced it.
        self._effective_fps = metadata.fps / self.config.ingestion.frame_stride
        self._ball_stage(result, metadata, tracker, calibrator)
        # The tracker already estimated background motion for every frame;
        # calibration reuses it to fill frames it could not solve on landmarks.
        calibrator.warps = tracker.motion_warps
        self._calibration_finalise(calibrator, result)
        self._track_stage(tracker, harvester, calibrator, result)
        self._team_stage(harvester, calibrator, result)
        self._identity_stage(harvester, result)
        self._setup_stage(result)
        self._assemble_stage(metadata, result)

        # ---------------- persistence --------------------------------------- #
        self._write_tables(run, metadata, result, calibrator)

        # ---------------- pass 2: rendering ---------------------------------- #
        should_render = (
            render
            if render is not None
            else (
                self.config.visualization.write_annotated_video
                or self.config.visualization.write_tactical_map
            )
        )
        if should_render:
            self._render_pass(run, metadata, result)

        result.stage_timings = dict(self._timings)
        result.stage_timings["total"] = round(time.perf_counter() - overall_start, 2)
        self._finalise_manifest(run, result, calibrator)

        log.info(
            "pipeline complete in %.1fs (%s)",
            result.stage_timings["total"],
            ", ".join(f"{k}={v}s" for k, v in self._timings.items()),
        )
        _ = resume
        return result

    # ------------------------------------------------------------------ #
    # Pass 1
    # ------------------------------------------------------------------ #

    def _analysis_pass(
        self, run: RunContext, metadata: VideoMetadata, result: PipelineResult
    ) -> tuple[MultiObjectTracker, Calibrator, _CropHarvester]:
        start = time.perf_counter()

        detector = build_detector(self.config)
        ball_detector = (
            build_ball_detector(self.config)
            if self.config.ball_detection.enabled else None
        )
        tracker = MultiObjectTracker(self.config)
        calibrator = Calibrator(
            self.config, self.pitch, (metadata.width, metadata.height)
        )
        classifier = TeamClassifier(self.config, self.pitch)
        harvester = _CropHarvester(
            classifier,
            self.config.team_classification.fit_stride_frames,
            max_crops=self.config.team_classification.max_crops,
            max_per_track=self.config.team_classification.max_crops_per_track,
        )
        self._classifier = classifier

        ball_filter = BallKalmanFilter(
            self.config.ball_tracking.process_noise,
            self.config.ball_tracking.measurement_noise,
        )
        frames_since_ball = 0

        detection_shard: list[dict] = []
        shard_index = 0
        batch_size = max(1, self.config.runtime.batch_size)

        reader = VideoReader(metadata, self.config.ingestion)
        expected = reader.expected_frames

        with reader, progress_bar() as progress:
            task = progress.add_task("pass 1: detect / calibrate / track", total=expected)
            batch: list[Frame] = []

            for frame in reader:
                batch.append(frame)
                if len(batch) < batch_size:
                    continue
                frames_since_ball, shard_index = self._process_batch(
                    batch,
                    detector,
                    ball_detector,
                    tracker,
                    calibrator,
                    harvester,
                    ball_filter,
                    frames_since_ball,
                    result,
                    detection_shard,
                    run,
                    shard_index,
                    metadata,
                )
                progress.update(task, advance=len(batch))
                batch = []

            if batch:
                frames_since_ball, shard_index = self._process_batch(
                    batch,
                    detector,
                    ball_detector,
                    tracker,
                    calibrator,
                    harvester,
                    ball_filter,
                    frames_since_ball,
                    result,
                    detection_shard,
                    run,
                    shard_index,
                    metadata,
                )
                progress.update(task, advance=len(batch))

        if detection_shard:
            self._flush_detections(run, detection_shard, shard_index)

        result.reports["ingestion"] = reader.counters.summary()
        result.reports["detector"] = detector.info.to_dict()
        if ball_detector is not None:
            result.reports["ball_detector"] = ball_detector.info.to_dict()
        result.reports["pitch_keypoints"] = calibrator.detector.info.to_dict()
        result.reports["crops_harvested"] = harvester.report()

        self._time("pass1_analysis", start)
        log.info(
            "pass 1 done: %d frames, %d ball candidates kept in memory",
            len(result.frame_indices),
            sum(len(v) for v in result.detections.values()),
        )
        return tracker, calibrator, harvester

    def _process_batch(
        self,
        batch: list[Frame],
        detector,
        ball_detector,
        tracker: MultiObjectTracker,
        calibrator: Calibrator,
        harvester: _CropHarvester,
        ball_filter: BallKalmanFilter,
        frames_since_ball: int,
        result: PipelineResult,
        detection_shard: list[dict],
        run: RunContext,
        shard_index: int,
        metadata: VideoMetadata,
    ) -> tuple[int, int]:
        images = [f.image for f in batch]
        indices = [f.idx for f in batch]

        multiclass = detector.detect_batch(images, indices)
        calibrations = calibrator.process_batch(images, indices)

        for frame, mc_dets, calib in zip(batch, multiclass, calibrations, strict=True):
            # -- ball: ROI-guided specialist pass --------------------------- #
            specialist: list[Detection] = []
            if ball_detector is not None:
                can_predict = ball_filter.initialised and frames_since_ball < 30
                predicted = ball_filter.peek(1) if can_predict else None
                specialist = ball_detector.detect(
                    frame.image, frame.idx, predicted, allow_tiled=True
                )

            fused = fuse_detections(mc_dets, specialist)
            balls = [d for d in fused if d.object_class is ObjectClass.BALL]

            if balls:
                best = max(balls, key=lambda d: d.confidence)
                centre = best.bbox.center
                if not ball_filter.initialised:
                    ball_filter.initiate(centre)
                else:
                    ball_filter.predict()
                    # Gate the online filter so a false positive cannot drag the
                    # ROI away. The offline estimator decides the truth later;
                    # this filter exists only to aim the next crop.
                    gate = self.config.ball_tracking.gating_threshold
                    if ball_filter.gating_distance(centre) < gate:
                        ball_filter.update(centre)
                frames_since_ball = 0
            else:
                if ball_filter.initialised:
                    ball_filter.predict()
                frames_since_ball += 1
                if frames_since_ball > 45:
                    ball_filter.reset()

            # -- persistence and bookkeeping -------------------------------- #
            result.frame_indices.append(frame.idx)
            result.timestamps[frame.idx] = frame.timestamp_s
            result.detections[frame.idx] = balls  # only the ball is needed later

            if self._needs_person_boxes:
                people = [d for d in fused if d.object_class.is_person]
                self._person_boxes[frame.idx] = (
                    np.array([d.bbox.to_array() for d in people], dtype=np.float32)
                    if people else np.zeros((0, 4), dtype=np.float32)
                )

            if self.config.storage.write_raw_detections:
                detection_shard.extend(
                    detection_rows(metadata.video_id, frame.idx, frame.timestamp_s, fused)
                )

            active = tracker.update(fused, frame.idx, frame.timestamp_s, frame.image)
            harvester.harvest(frame, active)
            _ = calib

        if len(detection_shard) >= 50000:
            self._flush_detections(run, list(detection_shard), shard_index)
            detection_shard.clear()
            shard_index += 1

        return frames_since_ball, shard_index

    def _flush_detections(self, run: RunContext, rows: list[dict], index: int) -> None:
        path = run.checkpoint_dir("detection") / f"shard_{index:05d}.parquet"
        write_detections(rows, path, self.config.storage.compression)

    # ------------------------------------------------------------------ #
    # Offline stages
    # ------------------------------------------------------------------ #

    @property
    def _needs_person_boxes(self) -> bool:
        cfg = self.config.ball_fusion
        return cfg.engine == "temporal" and cfg.observability_enabled

    def _fusion_stage(
        self,
        result: PipelineResult,
        metadata: VideoMetadata,
        tracker: MultiObjectTracker,
        calibrator: Calibrator,
    ) -> dict[int, list[Detection]]:
        """Candidate fusion, between detection and the trajectory search.

        Returns the candidate map the estimator should search. On the default
        (``legacy``) engine this is ``result.detections`` unchanged and by
        identity -- a run recorded before this stage existed reproduces exactly.
        """
        cfg = self.config.ball_fusion
        if cfg.engine == "legacy":
            result.reports["ball_fusion"] = legacy_report(
                result.detections, result.frame_indices
            )
            return result.detections

        start = time.perf_counter()
        # Real GMC, from the person tracker's own estimates. Same frame
        # indexing, same failure cases, no second estimation pass.
        shifts, camera_conf = camera_motion_from_warps(
            tracker.motion_warps, metadata.width, metadata.height
        )
        cuts = set(calibrator.shot_boundaries) if cfg.camera_cut_reset_enabled else set()

        observability: dict[int, str] | None = None
        if cfg.observability_enabled:
            from visionpitch.ball_tracking.observability import ObservabilityEstimator

            direct = {
                idx: max(dets, key=lambda d: d.confidence).bbox.center
                for idx, dets in result.detections.items() if dets
            }
            report = ObservabilityEstimator().label_sequence(
                result.frame_indices,
                (metadata.width, metadata.height),
                direct,
                self._person_boxes,
                camera_motion_by_frame={
                    idx: float(np.hypot(*shift)) for idx, shift in shifts.items()
                },
                calibration_confidence_by_frame={
                    idx: c.confidence for idx, c in calibrator.results.items()
                },
                pitch_keypoints_by_frame={
                    idx: c.n_keypoints for idx, c in calibrator.results.items()
                },
                cut_frames=cuts,
            )
            observability = {
                idx: frame.state.value for idx, frame in report.frames.items()
            }

        fused, fusion_report = run_ball_fusion(
            result.detections,
            result.frame_indices,
            cfg,
            camera_shifts=shifts,
            camera_confidence=camera_conf,
            cut_frames=cuts,
            observability=observability,
        )
        fusion_report["gmc"] = gmc_provenance(
            tracker.motion_warps, camera_conf, result.frame_indices,
            self.config.tracking.gmc_method, self.config.tracking.gmc_downscale,
            tracker.use_gmc,
        )
        fusion_report["camera_cuts"] = len(cuts)
        result.reports["ball_fusion"] = fusion_report
        self._time("ball_fusion", start)
        log.info(
            "ball fusion (temporal): %d -> %d candidates, %d frame(s) emptied "
            "by verification, GMC on %.1f%% of frames",
            fusion_report["candidates_in"], fusion_report["candidates_out"],
            fusion_report["frames_emptied_by_verification"],
            100 * fusion_report["gmc"]["estimate_ratio"],
        )
        return fused

    def _ball_stage(
        self,
        result: PipelineResult,
        metadata: VideoMetadata,
        tracker: MultiObjectTracker,
        calibrator: Calibrator,
    ) -> None:
        candidates = self._fusion_stage(result, metadata, tracker, calibrator)
        start = time.perf_counter()
        estimator = BallTrajectoryEstimator(self.config)
        result.ball_states = estimator.estimate(
            candidates, result.frame_indices, result.timestamps, metadata.width
        )
        result.reports["ball_tracking"] = {
            **estimator.quality_report(result.ball_states),
            # Separates "the detector never saw it" from "the search rejected
            # it". The headline observed-ratio conflates the two.
            "error_analysis": estimator.error_analysis(),
            **estimator.counters.summary(),
        }
        self._time("ball_trajectory", start)

    def _track_stage(
        self,
        tracker: MultiObjectTracker,
        harvester: _CropHarvester,
        calibrator: Calibrator,
        result: PipelineResult,
    ) -> None:
        start = time.perf_counter()
        raw = tracker.finalise()

        # Per-track appearance, aggregated over the harvested crops rather than
        # taken from any single frame: a crop captured mid-occlusion is half an
        # opponent, and letting it define the tracklet's appearance is how a
        # wrong merge gets authorised.
        appearance = self._aggregate_appearance(harvester, raw)

        tracking_config = self.config.tracking.model_copy(
            update={"assoc_fps": self._effective_fps}
        )
        cleaned, id_map, report = clean_tracks(
            raw,
            tracking_config,
            calibration=calibrator.results,
            min_calibration_confidence=max(
                self.config.calibration.min_confidence, 0.5
            ),
            appearance=appearance,
        )
        result.tracks = cleaned
        harvester.remap(id_map, set(cleaned))
        result.reports["tracking"] = {**report, **tracker.counters.summary()}
        self._time("track_cleaning", start)

    def _aggregate_appearance(
        self, harvester: _CropHarvester, tracks: dict[int, Track]
    ) -> dict[int, np.ndarray]:
        """Median hue-saturation descriptor per track."""
        extractor = self._classifier.extractor
        by_track = harvester.by_track()
        out: dict[int, np.ndarray] = {}
        for track_id in tracks:
            crops = by_track.get(track_id, [])
            if not crops:
                continue
            descriptors = [extractor.colour_descriptor(c) for c in crops]
            usable = [d for d in descriptors if np.any(d)]
            if not usable:
                continue
            median = np.median(np.stack(usable), axis=0)
            norm = np.linalg.norm(median)
            if norm > 0:
                out[track_id] = (median / norm).astype(np.float32)
        return out

    def _team_stage(
        self, harvester: _CropHarvester, calibrator: Calibrator, result: PipelineResult
    ) -> None:
        start = time.perf_counter()
        classifier: TeamClassifier = self._classifier

        outfield = [
            c
            for c in harvester.crops
            if result.tracks.get(c.track_id)
            and result.tracks[c.track_id].object_class is ObjectClass.PLAYER
        ]
        try:
            classifier.fit(outfield)
        except RuntimeError as exc:
            log.error("team discovery failed: %s", exc)
            result.reports["team_classification"] = {
                "status": "failed",
                "error": str(exc),
            }
            self._time("team_classification", start)
            return

        classifier.assign_from_crops(
            harvester.by_track(), result.tracks, calibrator.results
        )
        result.reports["team_classification"] = {
            "status": "ok",
            **classifier.report.to_dict(),
            "crop_budget": harvester.report(),
            **classifier.counters.summary(),
        }
        self._time("team_classification", start)

    def _identity_stage(self, harvester: _CropHarvester, result: PipelineResult) -> None:
        if not self.config.reid.jersey_ocr_enabled:
            result.reports["reid"] = {"status": "disabled"}
            return
        start = time.perf_counter()
        recogniser = JerseyNumberRecogniser(self.config)
        readings = recogniser.recognise_from_crops(harvester.by_track(), result.tracks)
        result.reports["reid"] = {
            "status": "experimental",
            "assigned": sum(1 for r in readings.values() if r.status == "assigned"),
            "ambiguous": sum(1 for r in readings.values() if r.status == "ambiguous"),
            "unknown": sum(1 for r in readings.values() if r.status == "unknown"),
            "readings": {str(k): v.to_dict() for k, v in readings.items()},
        }
        self._time("jersey_recognition", start)

    def _calibration_finalise(self, calibrator: Calibrator, result: PipelineResult) -> None:
        start = time.perf_counter()
        result.calibration = calibrator.finalise(result.frame_indices)
        self._support_regions = calibrator.support
        result.support_regions = dict(calibrator.support)
        result.reports["calibration"] = {
            **calibrator.report(result.frame_indices),
            "propagation": calibrator.propagation_report,
            **calibrator.counters.summary(),
        }
        self._time("calibration_smoothing", start)

    def _setup_stage(self, result: PipelineResult) -> None:
        start = time.perf_counter()
        result.match_setup = discover_match_setup(
            result.tracks, result.calibration, self.pitch, result.frame_indices
        )
        result.reports["match_setup"] = result.match_setup.to_dict()
        self._time("match_setup", start)

    def _assemble_stage(self, metadata: VideoMetadata, result: PipelineResult) -> None:
        start = time.perf_counter()
        assembler = GameStateAssembler(
            metadata.video_id,
            self.pitch,
            self.config.calibration.min_confidence,
            support_regions=self._support_regions,
            max_extrapolation_risk=self.config.calibration.max_extrapolation_risk,
        )
        result.rows = assembler.assemble(
            result.tracks,
            result.ball_states,
            result.calibration,
            result.timestamps,
            result.frame_indices,
        )
        result.reports["game_state"] = {
            **assembler.quality_report(result.rows),
            **assembler.counters.summary(),
        }
        self._time("game_state", start)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def _write_tables(
        self,
        run: RunContext,
        metadata: VideoMetadata,
        result: PipelineResult,
        calibrator: Calibrator,
    ) -> None:
        start = time.perf_counter()
        fmt = self.config.storage.format
        compression = self.config.storage.compression

        if self.config.storage.write_game_state:
            path = write_game_state(result.rows, run.table_path("game_state"), compression, fmt)
            result.outputs["game_state"] = str(path)

        # One row per processed frame, so a consumer can tell an empty frame
        # from an unprocessed one. Written unconditionally: it is small and it
        # is the denominator for every per-frame rate downstream.
        path = write_frames(
            frame_rows(
                metadata.video_id,
                result.frame_indices,
                result.timestamps,
                result.rows,
                result.calibration,
                getattr(result, "chunk_of", None),
            ),
            run.table_path("frames"),
            compression,
            fmt,
        )
        result.outputs["frames"] = str(path)

        if self.config.storage.write_tracks:
            path = write_tracks(
                track_rows(metadata.video_id, result.tracks.values()),
                run.table_path("tracks"),
                compression,
                fmt,
            )
            result.outputs["tracks"] = str(path)

        path = write_calibration(
            calibration_rows(metadata.video_id, result.timestamps, result.calibration.values()),
            run.table_path("calibration"),
            compression,
            fmt,
        )
        result.outputs["calibration"] = str(path)

        # Consolidate detection shards into one table.
        if self.config.storage.write_raw_detections:
            shards = sorted(run.checkpoint_dir("detection").glob("shard_*.parquet"))
            if shards:
                import pyarrow as pa
                import pyarrow.parquet as pq

                table = pa.concat_tables([pq.read_table(s) for s in shards])
                target = run.table_path("detections")
                pq.write_table(table, target, compression=compression)
                result.outputs["detections"] = str(target)
                if not self.config.storage.keep_intermediate:
                    for shard in shards:
                        shard.unlink()

        for stage in ("detection", "tracking", "ball_tracking", "team_classification",
                      "calibration", "game_state"):
            run.mark_complete(stage, result.reports.get(stage))

        self._time("write_tables", start)

    def _finalise_manifest(
        self, run: RunContext, result: PipelineResult, calibrator: Calibrator
    ) -> None:
        warnings: list[str] = []
        team_report = result.reports.get("team_classification", {})
        warnings.extend(team_report.get("warnings", []) or [])
        if result.match_setup:
            warnings.extend(result.match_setup.warnings)

        run.update_manifest(
            models={
                k: result.reports[k]
                for k in ("detector", "ball_detector", "pitch_keypoints")
                if k in result.reports
            },
            stages={**result.reports, "timings_s": dict(self._timings)},
            data_quality=self._data_quality(result),
            warnings=warnings,
        )
        _ = calibrator

        summary_path = run.path("summary.json")
        summary_path.write_text(
            json.dumps(
                {
                    "video": result.metadata.to_dict(),
                    "outputs": result.outputs,
                    "timings_s": dict(self._timings),
                    "data_quality": self._data_quality(result),
                    "match_setup": result.match_setup.to_dict() if result.match_setup else None,
                    "warnings": warnings,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        result.outputs["summary"] = str(summary_path)

    def _data_quality(self, result: PipelineResult) -> dict:
        """The numbers a reviewer should read before trusting anything else."""
        game_state = result.reports.get("game_state", {})
        calibration = result.reports.get("calibration", {})
        ball = result.reports.get("ball_tracking", {})
        tracking = result.reports.get("tracking", {})

        return {
            "frames_processed": len(result.frame_indices),
            "tracks": tracking.get("tracks_out"),
            "mean_people_per_frame": game_state.get("mean_people_per_frame"),
            "calibration_valid_ratio": calibration.get("valid_ratio"),
            "calibration_confident_ratio": calibration.get("confident_ratio"),
            "calibration_mean_reprojection_error_m": calibration.get(
                "mean_reprojection_error_m"
            ),
            "ball_observed_ratio": ball.get("observed_ratio"),
            "ball_visible_ratio": ball.get("visible_ratio"),
            "pitch_coordinate_ratio": game_state.get("pitch_coordinate_ratio"),
            "interpolated_row_ratio": game_state.get("interpolated_ratio"),
            "validation_status_counts": game_state.get("validation_status_counts"),
            "requires_manual_review": self._review_flags(result),
        }

    @staticmethod
    def _review_flags(result: PipelineResult) -> list[str]:
        flags: list[str] = []
        calibration = result.reports.get("calibration", {})
        ball = result.reports.get("ball_tracking", {})
        team = result.reports.get("team_classification", {})

        if (calibration.get("valid_ratio") or 0) < 0.7:
            flags.append(
                f"calibration succeeded on only "
                f"{100 * (calibration.get('valid_ratio') or 0):.0f}% of frames; "
                f"pitch coordinates are missing for the rest"
            )
        if (ball.get("observed_ratio") or 0) < 0.5:
            flags.append(
                f"the ball was directly observed in only "
                f"{100 * (ball.get('observed_ratio') or 0):.0f}% of frames"
            )
        separation = team.get("cluster_separation")
        if separation is not None and separation < 0.15:
            flags.append(
                f"team separation is weak (silhouette {separation:.2f}); "
                f"team assignments should be reviewed"
            )
        if team.get("status") == "failed":
            flags.append("team discovery failed entirely; all players are unassigned")
        return flags

    # ------------------------------------------------------------------ #
    # Pass 2: rendering
    # ------------------------------------------------------------------ #

    def _render_pass(
        self, run: RunContext, metadata: VideoMetadata, result: PipelineResult
    ) -> None:
        start = time.perf_counter()
        cfg = self.config.visualization

        if not (cfg.write_annotated_video or cfg.write_tactical_map or cfg.write_combined_video):
            # The showcase config disables every debug output. Decoding the whole
            # video to draw nothing costs as much as drawing everything, so say so
            # and stop; `visionpitch showcase` renders the presentation output
            # from the tables this run just wrote.
            log.info("no video outputs enabled; skipping the render pass")
            run.mark_complete("visualization")
            self._time("render", start)
            return

        annotator = FrameAnnotator(cfg)
        renderer = PitchRenderer(self.pitch, cfg)

        discovered = (result.reports.get("team_classification") or {}).get("team_colours")
        if discovered:
            renderer.set_discovered_colours(discovered)
            annotator.set_discovered_colours(discovered)
        fps = cfg.output_fps or (metadata.fps / self.config.ingestion.frame_stride)

        labels = {
            t.track_id: display_name(
                t.team_id.value, t.jersey_number, t.track_id, t.role.value
            )
            for t in result.tracks.values()
        }

        rows_by_frame: dict[int, list[dict]] = {}
        for row in result.rows:
            rows_by_frame.setdefault(row.frame_idx, []).append(
                {
                    "pitch_x": row.pitch_x,
                    "pitch_y": row.pitch_y,
                    "team_id": row.team_id,
                    "role": row.role,
                    "track_id": row.track_id,
                    "team_confidence": row.team_confidence,
                }
            )

        writers: dict[str, VideoWriter] = {}
        if cfg.write_annotated_video:
            writers["annotated"] = VideoWriter(
                run.path("video", "annotated.mp4"), fps, (metadata.width, metadata.height)
            )
        if cfg.write_tactical_map:
            writers["radar"] = VideoWriter(
                run.path("video", "radar.mp4"), fps, (renderer.width, renderer.height)
            )

        combined_writer: VideoWriter | None = None
        ball_trail: list[tuple[float, float]] = []

        reader = VideoReader(metadata, self.config.ingestion)
        with reader, progress_bar() as progress:
            task = progress.add_task("pass 2: rendering", total=reader.expected_frames)
            for frame in reader:
                calib = result.calibration.get(frame.idx)
                calib_conf = calib.confidence if calib else 0.0
                ball = result.ball_states.get(frame.idx)

                if "annotated" in writers or cfg.write_combined_video:
                    annotated = annotator.annotate(
                        frame.image,
                        frame.idx,
                        frame.timestamp_s,
                        result.tracks,
                        ball,
                        calib_conf,
                        labels,
                    )
                    if "annotated" in writers:
                        writers["annotated"].write(annotated)
                else:
                    annotated = None

                objects = rows_by_frame.get(frame.idx, [])
                ball_row = next((o for o in objects if o["role"] == "ball"), None)
                if ball_row and ball_row["pitch_x"] is not None:
                    ball_trail.append((ball_row["pitch_x"], ball_row["pitch_y"]))
                    if len(ball_trail) > cfg.ball_trail_frames:
                        ball_trail.pop(0)
                elif not objects:
                    ball_trail.clear()

                if calib is None or not calib.is_valid:
                    radar = renderer.blank(
                        frame.idx, frame.timestamp_s, "no valid calibration for this frame"
                    )
                else:
                    radar = renderer.render(
                        objects, frame.idx, frame.timestamp_s, ball_trail, calib_conf
                    )

                if "radar" in writers:
                    writers["radar"].write(radar)

                if cfg.write_combined_video and annotated is not None:
                    combined = stack_side_by_side(annotated, radar)
                    if combined_writer is None:
                        combined_writer = VideoWriter(
                            run.path("video", "combined.mp4"),
                            fps,
                            (combined.shape[1], combined.shape[0]),
                        )
                    combined_writer.write(combined)

                progress.update(task, advance=1)

        for name, writer in writers.items():
            writer.close()
            result.outputs[f"video_{name}"] = str(writer.path)
        if combined_writer is not None:
            combined_writer.close()
            result.outputs["video_combined"] = str(combined_writer.path)

        run.mark_complete("visualization")
        self._time("render", start)

    # ------------------------------------------------------------------ #

    @staticmethod
    def frame_width_scale(metadata: VideoMetadata) -> float:
        return metadata.width / 1920.0

    @staticmethod
    def _as_array(values: list[float]) -> np.ndarray:
        return np.asarray(values, dtype=np.float64)
