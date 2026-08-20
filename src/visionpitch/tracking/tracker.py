"""Multi-object tracker: ByteTrack association with optional BoT-SORT extensions.

Two backends behind one implementation, selected by config:

``bytetrack``
    Two-stage IoU association. Stage one matches high-confidence detections;
    stage two rescues tracks by matching them against the *low*-confidence
    detections that a normal tracker would discard. That second stage is what
    makes ByteTrack good in crowds -- an occluded player's detector confidence
    collapses but rarely to zero.

``botsort``
    ByteTrack plus camera motion compensation and appearance-gated association.

Why implement it here rather than call Ultralytics' built-in tracker
--------------------------------------------------------------------
The pipeline stores raw detections to Parquet before tracking. Tracking then
consumes that table, so tracker parameters can be re-tuned and re-run in seconds
without a second pass of detection over the video -- which is the expensive part
and, per rule 19 of the brief, the thing worth caching. Ultralytics' tracker is
coupled to its predictor loop and cannot be driven from stored detections.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from scipy.optimize import linear_sum_assignment

from visionpitch.common.config import Config
from visionpitch.common.geometry import iou_matrix
from visionpitch.common.logging import StageCounters, get_logger
from visionpitch.common.types import BBox, Detection, ObjectClass, Track, TrackObservation
from visionpitch.tracking.appearance import (
    ExponentialFeatureBank,
    TorsoHistogramAppearance,
    cosine_distance,
)
from visionpitch.tracking.gmc import GlobalMotionCompensator
from visionpitch.tracking.kalman import KalmanBoxFilter, state_to_xyxy, xyxy_to_state

log = get_logger("tracking")


class TrackState(str, Enum):
    NEW = "new"
    TRACKED = "tracked"
    LOST = "lost"
    REMOVED = "removed"


class _ActiveTrack:
    """Internal mutable track. Converted to the public :class:`Track` at the end."""

    __slots__ = (
        "track_id",
        "object_class",
        "mean",
        "covariance",
        "state",
        "score",
        "start_frame",
        "last_frame",
        "time_since_update",
        "hits",
        "observations",
        "class_votes",
        "class_counts",
        "_kf",
    )

    def __init__(
        self,
        track_id: int,
        object_class: ObjectClass,
        detection: Detection,
        timestamp_s: float,
        kf: KalmanBoxFilter,
    ) -> None:
        self.track_id = track_id
        self.object_class = object_class
        self._kf = kf
        self.mean, self.covariance = kf.initiate(xyxy_to_state(detection.bbox.to_array()))
        self.state = TrackState.NEW
        self.score = detection.confidence
        self.start_frame = detection.frame_idx
        self.last_frame = detection.frame_idx
        self.time_since_update = 0
        self.hits = 1
        # Detector class is evidence, not identity: accumulate it rather than
        # freezing the birth detection's label. See Track.class_votes.
        self.class_votes: dict[str, float] = {}
        self.class_counts: dict[str, int] = {}
        self._vote_class(detection)
        self.observations: list[TrackObservation] = [
            TrackObservation(
                frame_idx=detection.frame_idx,
                timestamp_s=timestamp_s,
                bbox=detection.bbox,
                det_confidence=detection.confidence,
                track_confidence=detection.confidence,
                interpolated=False,
            )
        ]

    # -- geometry ----------------------------------------------------------- #

    @property
    def bbox_array(self) -> np.ndarray:
        return state_to_xyxy(self.mean)

    def predict(self) -> None:
        if self.state is not TrackState.TRACKED:
            # A track that is not currently observed should not be assumed to be
            # accelerating; freeze the size velocities to stop it ballooning.
            self.mean[6] = 0.0
            self.mean[7] = 0.0
        self.mean, self.covariance = self._kf.predict(self.mean, self.covariance)
        self.time_since_update += 1

    def apply_warp(self, warp: np.ndarray) -> None:
        self.mean, self.covariance = self._kf.apply_motion_compensation(
            self.mean, self.covariance, warp
        )

    def _vote_class(self, detection: Detection) -> None:
        key = detection.object_class.value
        self.class_votes[key] = self.class_votes.get(key, 0.0) + float(detection.confidence)
        self.class_counts[key] = self.class_counts.get(key, 0) + 1

    def update(self, detection: Detection, timestamp_s: float) -> None:
        self._vote_class(detection)
        self.mean, self.covariance = self._kf.update(
            self.mean, self.covariance, xyxy_to_state(detection.bbox.to_array())
        )
        self.state = TrackState.TRACKED
        self.score = detection.confidence
        self.last_frame = detection.frame_idx
        self.time_since_update = 0
        self.hits += 1
        # Track confidence blends detector confidence with accumulated support:
        # a box seen for 40 consecutive frames is more trustworthy than the same
        # box seen once, even at identical detector score.
        support = min(1.0, self.hits / 20.0)
        self.observations.append(
            TrackObservation(
                frame_idx=detection.frame_idx,
                timestamp_s=timestamp_s,
                bbox=detection.bbox,
                det_confidence=detection.confidence,
                track_confidence=float(0.5 * detection.confidence + 0.5 * support),
                interpolated=False,
            )
        )

    def coast(self, frame_idx: int, timestamp_s: float) -> None:
        """Record a predicted (not observed) position while the track is lost."""
        self.state = TrackState.LOST
        self.last_frame = frame_idx
        box = self.bbox_array
        decay = max(0.0, 1.0 - self.time_since_update / 30.0)
        self.observations.append(
            TrackObservation(
                frame_idx=frame_idx,
                timestamp_s=timestamp_s,
                bbox=BBox.from_xyxy(box),
                det_confidence=0.0,
                track_confidence=float(0.4 * decay),
                interpolated=True,
            )
        )


def _linear_assignment(
    cost: np.ndarray, threshold: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian assignment with a cost ceiling."""
    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return [], list(range(n_rows)), list(range(n_cols))

    row_idx, col_idx = linear_sum_assignment(cost)
    matches, matched_rows, matched_cols = [], set(), set()
    for r, c in zip(row_idx, col_idx, strict=True):
        if cost[r, c] <= threshold:
            matches.append((int(r), int(c)))
            matched_rows.add(int(r))
            matched_cols.add(int(c))
    unmatched_rows = [r for r in range(n_rows) if r not in matched_rows]
    unmatched_cols = [c for c in range(n_cols) if c not in matched_cols]
    return matches, unmatched_rows, unmatched_cols


class MultiObjectTracker:
    """Tracks people frame by frame. The ball is handled by ``ball_tracking``."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.cfg = config.tracking
        self.use_appearance = self.cfg.reid_enabled and self.cfg.tracker == "botsort"
        self.use_gmc = self.cfg.gmc_enabled and self.cfg.tracker == "botsort"

        self._kf = KalmanBoxFilter()
        self._next_id = 1
        #: frame -> 2x3 affine mapping the previous processed frame onto it.
        #: Shared with calibration, which uses it to propagate camera pose.
        self.motion_warps: dict[int, np.ndarray] = {}
        self._tracked: list[_ActiveTrack] = []
        self._lost: list[_ActiveTrack] = []
        self._finished: list[_ActiveTrack] = []
        self.counters = StageCounters("tracking")

        tc = config.team_classification
        self._appearance = TorsoHistogramAppearance(
            top_frac=tc.jersey_top_frac,
            bottom_frac=tc.jersey_bottom_frac,
            side_margin_frac=tc.jersey_side_margin_frac,
            grass_hue_range=tc.grass_hue_range,
            grass_sat_min=tc.grass_sat_min,
        )
        self._features = ExponentialFeatureBank()
        self._gmc = (
            GlobalMotionCompensator(method=self.cfg.gmc_method, downscale=self.cfg.gmc_downscale)
            if self.use_gmc
            else None
        )

    # -- cost construction -------------------------------------------------- #

    def _cost(
        self,
        tracks: list[_ActiveTrack],
        detections: list[Detection],
        features: np.ndarray | None,
    ) -> np.ndarray:
        if not tracks or not detections:
            return np.zeros((len(tracks), len(detections)), dtype=np.float32)

        track_boxes = np.array([t.bbox_array for t in tracks])
        det_boxes = np.array([d.bbox.to_array() for d in detections])
        cost = 1.0 - iou_matrix(track_boxes, det_boxes).astype(np.float32)

        if features is None or not self.use_appearance or features.size == 0:
            return cost

        bank = np.array(
            [self._features.get(t.track_id, self._appearance.dim) for t in tracks],
            dtype=np.float32,
        )
        app_cost = cosine_distance(bank, features)
        w = self.cfg.reid_weight
        # Appearance can only *refine* a geometrically plausible match, never
        # create one across the frame: two players in identical kit have near
        # zero appearance cost, so an ungated appearance term would happily swap
        # them. The IoU term stays dominant and gates the result.
        has_appearance = np.isfinite(app_cost)
        weighted = (1.0 - w) * cost + np.where(has_appearance, w * app_cost, 0.0)
        available_weight = (1.0 - w) + np.where(has_appearance, w, 0.0)
        blended = weighted / available_weight
        gated = np.where(cost > 0.9, cost, blended)
        return gated.astype(np.float32)

    # -- main loop ---------------------------------------------------------- #

    def update(
        self,
        detections: list[Detection],
        frame_idx: int,
        timestamp_s: float,
        image: np.ndarray | None = None,
    ) -> list[_ActiveTrack]:
        """Advance the tracker by one frame. Returns the currently active tracks."""
        people = [d for d in detections if d.object_class.is_person]

        high = [d for d in people if d.confidence >= self.cfg.track_high_threshold]
        low = [
            d
            for d in people
            if self.cfg.track_low_threshold <= d.confidence < self.cfg.track_high_threshold
        ]

        # -- predict, then transport into this frame's coordinate system ----- #
        pool = self._tracked + self._lost
        for track in pool:
            track.predict()

        if self._gmc is not None and image is not None:
            boxes = np.array([d.bbox.to_array() for d in people]) if people else None
            warp = self._gmc.apply(image, boxes)
            # Kept so calibration can chain it across frames it could not solve.
            # Estimating background motion twice would be pure waste.
            self.motion_warps[frame_idx] = warp
            for track in pool:
                track.apply_warp(warp)

        features_high = None
        if self.use_appearance and image is not None and high:
            features_high = self._appearance.extract(
                image, np.array([d.bbox.to_array() for d in high])
            )

        # -- stage 1: confident detections against all tracks ---------------- #
        cost = self._cost(pool, high, features_high)
        matches, unmatched_tracks, unmatched_dets = _linear_assignment(
            cost, 1.0 - (1.0 - self.cfg.match_threshold)
        )

        activated: list[_ActiveTrack] = []
        for t_idx, d_idx in matches:
            track, det = pool[t_idx], high[d_idx]
            track.update(det, timestamp_s)
            if features_high is not None:
                self._features.update(track.track_id, features_high[d_idx], det.confidence)
            activated.append(track)

        # -- stage 2: the ByteTrack rescue pass ------------------------------ #
        remaining = [pool[i] for i in unmatched_tracks if pool[i].state is TrackState.TRACKED]
        if remaining and low:
            cost_low = self._cost(remaining, low, None)
            # A looser gate here on purpose: these are boxes the detector was
            # unsure about, and the track's own motion model is carrying most of
            # the evidence.
            matches_low, unmatched_low_tracks, _ = _linear_assignment(cost_low, 0.5)
            for t_idx, d_idx in matches_low:
                remaining[t_idx].update(low[d_idx], timestamp_s)
                activated.append(remaining[t_idx])
            still_unmatched = [remaining[i] for i in unmatched_low_tracks]
        else:
            still_unmatched = remaining

        matched_ids = {t.track_id for t in activated}
        for track in pool:
            if track.track_id in matched_ids:
                continue
            track.coast(frame_idx, timestamp_s)
        _ = still_unmatched  # coasting already covers these

        # -- births ---------------------------------------------------------- #
        for d_idx in unmatched_dets:
            det = high[d_idx]
            if det.confidence < self.cfg.new_track_threshold:
                continue
            track = _ActiveTrack(self._next_id, det.object_class, det, timestamp_s, self._kf)
            self._next_id += 1
            if features_high is not None:
                self._features.update(track.track_id, features_high[d_idx], det.confidence)
            activated.append(track)

        # -- lifecycle -------------------------------------------------------- #
        self._tracked = [t for t in activated if t.time_since_update == 0]
        self._lost = [
            t
            for t in pool + activated
            if t.time_since_update > 0 and t.time_since_update <= self.cfg.track_buffer
        ]
        # De-duplicate: a track can appear in both lists after a stage-2 rescue.
        tracked_ids = {t.track_id for t in self._tracked}
        self._lost = [t for t in self._lost if t.track_id not in tracked_ids]

        expired = [
            t for t in pool if t.time_since_update > self.cfg.track_buffer and t not in self._lost
        ]
        for track in expired:
            track.state = TrackState.REMOVED
            self._features.drop(track.track_id)
            if track not in self._finished:
                self._finished.append(track)

        self.counters.ok()
        if not people:
            self.counters.warn("frame_with_no_person_detections")
        return self._tracked

    # -- output -------------------------------------------------------------- #

    def finalise(self) -> dict[int, Track]:
        """Close every open track and convert to the public representation.

        Coasted (predicted) observations at the *end* of a track are trimmed:
        they are extrapolation past the last real evidence and would otherwise
        appear in the output as confident positions of a player who has left the
        frame.
        """
        everything = {t.track_id: t for t in self._finished + self._lost + self._tracked}

        out: dict[int, Track] = {}
        for track_id, active in everything.items():
            observations = list(active.observations)
            while observations and observations[-1].interpolated:
                observations.pop()
            if not observations:
                continue
            out[track_id] = Track(
                track_id=track_id,
                object_class=active.object_class,
                observations=observations,
                class_votes=dict(active.class_votes),
                class_counts=dict(active.class_counts),
            )
        log.info("tracker produced %d raw tracks", len(out))
        return out
