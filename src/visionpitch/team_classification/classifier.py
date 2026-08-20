"""Team, role and goalkeeper discovery.

The user uploads a video and picks a mode. Everything below is inferred:

* which two kits are on the pitch, and which players wear which
* which tracks are goalkeepers, and which team each keeper belongs to
* which tracks are match officials
* a stable temporary identity for every player

Method
------
Team assignment is a **track-level** decision made from many frames, never a
per-frame one. A single frame of a player mid-tackle, backlit, or half-occluded
will be misclassified by any method; the same player over eighty frames will not.
So crops are sampled across the clip, embedded, clustered into two groups, and
then each *track* takes the majority vote of its own crops with an explicit
confidence. A track whose vote is split stays ``UNKNOWN`` rather than being
forced into a team.

Clustering is fitted on outfield players only. Goalkeepers wear deliberately
distinct kit and referees wear a third colour; including them would corrupt the
two-cluster structure the whole method depends on.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from visionpitch.common.config import Config
from visionpitch.common.logging import StageCounters, get_logger
from visionpitch.common.types import CalibrationResult, ObjectClass, Role, TeamId, Track
from visionpitch.pitch.geometry import PitchConfiguration
from visionpitch.team_classification.crops import JerseyCrop, JerseyCropExtractor
from visionpitch.team_classification.embeddings import build_embedder
from visionpitch.team_classification.roles import RolePolicy, resolve_roles

log = get_logger("team_classification")


@dataclass
class TeamDiscoveryReport:
    """What the system worked out, for the review screen and the manifest."""

    embedder: str
    n_crops_fitted: int
    n_tracks: int
    team_counts: dict[str, int] = field(default_factory=dict)
    role_counts: dict[str, int] = field(default_factory=dict)
    #: mean BGR of each team's jerseys, for the UI swatch
    team_colours: dict[str, list[int]] = field(default_factory=dict)
    goalkeepers: dict[str, dict] = field(default_factory=dict)
    #: what the conservative role resolver decided, and what it overruled
    roles: dict = field(default_factory=dict)
    #: silhouette score of the two-cluster fit; low means the kits are similar
    cluster_separation: float | None = None
    unresolved_tracks: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "embedder": self.embedder,
            "n_crops_fitted": self.n_crops_fitted,
            "n_tracks": self.n_tracks,
            "team_counts": self.team_counts,
            "role_counts": self.role_counts,
            "team_colours": self.team_colours,
            "goalkeepers": self.goalkeepers,
            "roles": self.roles,
            "cluster_separation": self.cluster_separation,
            "unresolved_tracks": self.unresolved_tracks,
            "warnings": self.warnings,
        }


class TeamClassifier:
    """Fits two team clusters, then labels every track."""

    def __init__(self, config: Config, pitch: PitchConfiguration) -> None:
        self.config = config
        self.cfg = config.team_classification
        self.pitch = pitch
        self.extractor = JerseyCropExtractor(self.cfg)
        self.embedder = build_embedder(config, self.extractor)
        self.counters = StageCounters("team_classification")

        self._kmeans = None
        self._reducer = None
        self._cluster_colours: dict[int, np.ndarray] = {}
        self.report = TeamDiscoveryReport(
            embedder=self.embedder.name, n_crops_fitted=0, n_tracks=0
        )
        self.role_policy = RolePolicy(**self.cfg.roles.model_dump())
        self.role_report = None

    # -- crop collection ---------------------------------------------------- #

    def collect_crops(
        self,
        frames: dict[int, np.ndarray],
        tracks: dict[int, Track],
        classes: tuple[ObjectClass, ...] = (ObjectClass.PLAYER,),
    ) -> list[JerseyCrop]:
        """Sample torso crops across the clip for the requested track classes."""
        wanted = set(classes)
        crops: list[JerseyCrop] = []
        for track in tracks.values():
            if track.object_class not in wanted:
                continue
            for obs in track.observations:
                if obs.interpolated or obs.frame_idx % self.cfg.fit_stride_frames:
                    continue
                image = frames.get(obs.frame_idx)
                if image is None:
                    continue
                crop = self.extractor.extract(
                    image, obs.bbox.to_xyxy(), track.track_id, obs.frame_idx
                )
                if crop is not None:
                    crops.append(crop)
        return crops

    # -- fitting ------------------------------------------------------------ #

    def fit(self, crops: list[JerseyCrop]) -> None:
        """Fit the two-team model on a sample of outfield-player crops."""
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.metrics import silhouette_score

        if len(crops) < self.cfg.min_votes * 4:
            raise RuntimeError(
                f"only {len(crops)} usable jersey crops were found; team discovery "
                f"needs at least {self.cfg.min_votes * 4}. The clip may be too "
                f"short, too zoomed-in, or detection may have failed."
            )

        sample = crops
        if len(crops) > self.cfg.fit_sample_size:
            rng = np.random.default_rng(self.config.runtime.seed)
            idx = rng.choice(len(crops), self.cfg.fit_sample_size, replace=False)
            sample = [crops[i] for i in idx]

        features = self.embedder.embed(sample)
        usable = np.linalg.norm(features, axis=1) > 0
        features, sample = features[usable], [c for c, u in zip(sample, usable, strict=True) if u]
        if features.shape[0] < self.cfg.min_votes * 4:
            raise RuntimeError("too few usable embeddings after filtering empty crops")

        # PCA before clustering: high-dimensional embeddings put almost all
        # pairwise distances at the same value, which makes k-means arbitrary.
        n_components = int(min(32, features.shape[0] - 1, features.shape[1]))
        self._reducer = PCA(n_components=n_components, random_state=self.config.runtime.seed)
        reduced = self._reducer.fit_transform(features)

        self._kmeans = KMeans(
            n_clusters=2, n_init=20, random_state=self.config.runtime.seed
        ).fit(reduced)
        labels = self._kmeans.labels_

        try:
            separation = float(silhouette_score(reduced, labels))
        except ValueError:
            separation = float("nan")
        self.report.cluster_separation = round(separation, 4)
        self.report.n_crops_fitted = int(features.shape[0])

        # A low silhouette means the two kits are not separable in this feature
        # space. The run continues -- but the warning is surfaced so nobody reads
        # the team stats as reliable.
        if np.isfinite(separation) and separation < 0.15:
            msg = (
                f"weak team separation (silhouette {separation:.3f}): the two kits "
                f"may be visually similar. Team assignments should be reviewed."
            )
            log.warning(msg)
            self.report.warnings.append(msg)

        for cluster in (0, 1):
            member_colours = [
                sample[i].mean_colour for i in np.flatnonzero(labels == cluster)
            ]
            if member_colours:
                self._cluster_colours[cluster] = np.mean(member_colours, axis=0)

        counts = Counter(labels.tolist())
        log.info(
            "fitted 2 team clusters on %d crops (%d/%d), silhouette=%.3f",
            features.shape[0],
            counts.get(0, 0),
            counts.get(1, 0),
            separation,
        )

    def _predict(
        self, crops: list[JerseyCrop]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cluster label, per-crop confidence, and distance to nearest centroid.

        The third value is what separates a referee from a player. Both teams'
        kits sit close to their own centroid; an official's third colour sits far
        from both, and that stays true even when the *relative* confidence is
        high because one centroid happens to be marginally nearer than the other.
        """
        if self._kmeans is None or self._reducer is None:
            raise RuntimeError("TeamClassifier.fit must be called before prediction")
        if not crops:
            return np.zeros(0, dtype=int), np.zeros(0, dtype=float), np.zeros(0)

        features = self.embedder.embed(crops)
        reduced = self._reducer.transform(features)
        distances = self._kmeans.transform(reduced)
        labels = distances.argmin(axis=1)

        near = distances.min(axis=1)
        far = distances.max(axis=1)
        total = near + far
        # 0 when the crop sits equidistant between clusters, ->1 when it sits
        # firmly inside one. This is the per-crop weight in the track vote.
        confidence = np.where(total > 0, (far - near) / total, 0.0)
        return labels, confidence, near

    # -- track labelling ---------------------------------------------------- #

    def _dominant_class(self, track: Track) -> ObjectClass:
        return track.object_class

    def assign_tracks(
        self,
        frames: dict[int, np.ndarray],
        tracks: dict[int, Track],
        calibration: dict[int, CalibrationResult] | None = None,
    ) -> None:
        """Label every track from cached frames. Convenience for tests and tools.

        The pipeline uses :meth:`assign_from_crops` instead, because it harvests
        crops during decoding and never holds whole frames.
        """
        by_track: dict[int, list[JerseyCrop]] = defaultdict(list)
        for crop in self.collect_crops(
            frames,
            tracks,
            classes=(ObjectClass.PLAYER, ObjectClass.GOALKEEPER, ObjectClass.REFEREE),
        ):
            by_track[crop.track_id].append(crop)
        self.assign_from_crops(by_track, tracks, calibration)

    def assign_from_crops(
        self,
        crops_by_track: dict[int, list[JerseyCrop]],
        tracks: dict[int, Track],
        calibration: dict[int, CalibrationResult] | None = None,
    ) -> None:
        """Label every track in place with team, role and confidences."""
        by_track = crops_by_track
        kit_distance: dict[int, float] = {}

        for track in tracks.values():
            cls = self._dominant_class(track)

            if cls is ObjectClass.BALL:
                track.team_id = TeamId.NONE
                track.role = Role.BALL
                track.role_confidence = 1.0
                continue

            # Referee-class tracks are voted on like everyone else. Short-circuiting
            # them here is what made a single bad birth detection unrecoverable:
            # the track skipped team classification entirely, so no later stage
            # held the evidence needed to disagree with it.
            crops = by_track.get(track.track_id, [])
            if len(crops) < self.cfg.min_votes:
                track.team_id = TeamId.UNKNOWN
                track.team_confidence = 0.0
                track.role = Role.UNKNOWN
                track.role_confidence = 0.0
                self.report.unresolved_tracks.append(track.track_id)
                self.counters.warn("too_few_crops_for_vote")
                continue

            labels, confidences, distances = self._predict(crops)
            team_id, team_conf = self._vote(labels, confidences)
            track.team_id = team_id
            track.team_confidence = team_conf
            track.role = Role.UNKNOWN
            track.role_confidence = 0.0
            if distances.size:
                kit_distance[track.track_id] = float(np.median(distances))

            if team_id is TeamId.UNKNOWN:
                self.report.unresolved_tracks.append(track.track_id)
                self.counters.warn("split_team_vote")
            else:
                self.counters.ok()

        # Roles come from accumulated evidence, not from the birth class.
        self.role_report = resolve_roles(
            tracks,
            calibration,
            self.pitch,
            kit_distance=kit_distance,
            policy=self.role_policy,
            min_calibration_confidence=self.config.calibration.min_confidence,
        )
        self._assign_goalkeepers(tracks, calibration)
        self._build_report(tracks)

    def _vote(
        self, labels: np.ndarray, confidences: np.ndarray
    ) -> tuple[TeamId, float]:
        """Confidence-weighted majority vote over a track's crops."""
        if labels.size == 0:
            return TeamId.UNKNOWN, 0.0

        weights = np.zeros(2)
        for label, conf in zip(labels, confidences, strict=True):
            weights[int(label)] += float(conf)

        total = weights.sum()
        if total <= 0:
            return TeamId.UNKNOWN, 0.0

        winner = int(weights.argmax())
        share = float(weights[winner] / total)
        if share < self.cfg.vote_confidence_threshold:
            return TeamId.UNKNOWN, share
        return (TeamId.A if winner == 0 else TeamId.B), share

    # -- goalkeeper discovery ------------------------------------------------ #

    def _assign_goalkeepers(
        self, tracks: dict[int, Track], calibration: dict[int, CalibrationResult] | None
    ) -> None:
        """Attach each goalkeeper to a team using spatial evidence.

        Appearance cannot answer this: a keeper's kit is deliberately unlike
        both outfield kits, so the two-cluster model has no opinion worth
        trusting about which team they play for.

        The signal that does work is proximity. A keeper's own defenders spend
        the match between the keeper and the ball, so averaged over the clip the
        keeper's own team is measurably closer to them than the opposition is.
        When the two averages are close, the assignment is left UNKNOWN rather
        than guessed.
        """
        keepers = [t for t in tracks.values() if t.role is Role.GOALKEEPER]
        if not keepers:
            return
        # A keeper whose own colour vote was confident keeps it. Kit clash makes
        # that rare, but when it happens the direct evidence beats the proximity
        # heuristic below, which is only a tiebreak averaged over the whole clip.
        prior = {
            k.track_id: (k.team_id, k.team_confidence)
            for k in keepers
            if k.team_id in (TeamId.A, TeamId.B)
            and k.team_confidence >= self.cfg.goalkeeper_min_confidence
        }
        if calibration is None:
            for keeper in keepers:
                keeper.team_id = TeamId.UNKNOWN
                keeper.team_confidence = 0.0
            self.report.warnings.append(
                "goalkeepers could not be assigned to teams: no calibration available"
            )
            return

        # Pitch positions per frame, per team.
        positions: dict[int, dict[TeamId, list[np.ndarray]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for track in tracks.values():
            if track.role is not Role.OUTFIELD or track.team_id not in (TeamId.A, TeamId.B):
                continue
            for obs in track.observations:
                result = calibration.get(obs.frame_idx)
                if result is None or not result.is_valid:
                    continue
                point = self._project(result, obs.bbox.ground_contact)
                if point is not None:
                    positions[obs.frame_idx][track.team_id].append(point)

        for keeper in keepers:
            distances: dict[TeamId, list[float]] = {TeamId.A: [], TeamId.B: []}
            keeper_points: list[np.ndarray] = []
            prior_team = prior.get(keeper.track_id)

            for obs in keeper.observations:
                result = calibration.get(obs.frame_idx)
                if result is None or not result.is_valid:
                    continue
                keeper_point = self._project(result, obs.bbox.ground_contact)
                if keeper_point is None:
                    continue
                keeper_points.append(keeper_point)
                for team in (TeamId.A, TeamId.B):
                    teammates = positions[obs.frame_idx].get(team, [])
                    if teammates:
                        spread = np.linalg.norm(np.array(teammates) - keeper_point, axis=1)
                        # Nearest few, not the whole team. Averaging over all
                        # eleven is dominated by the opposition's shape upfield
                        # and washes out the one signal that actually separates
                        # the two: a keeper is surrounded by their own defenders.
                        nearest = np.sort(spread)[: self.cfg.goalkeeper_neighbours]
                        distances[team].append(float(np.mean(nearest)))

            mean_a = float(np.mean(distances[TeamId.A])) if distances[TeamId.A] else np.inf
            mean_b = float(np.mean(distances[TeamId.B])) if distances[TeamId.B] else np.inf

            if not np.isfinite(mean_a) and not np.isfinite(mean_b):
                keeper.team_id = TeamId.UNKNOWN
                keeper.team_confidence = 0.0
                continue

            nearer = TeamId.A if mean_a <= mean_b else TeamId.B
            far, near = max(mean_a, mean_b), min(mean_a, mean_b)
            margin = (far - near) / far if np.isfinite(far) and far > 0 else 0.0

            if prior_team is not None:
                keeper.team_id, keeper.team_confidence = prior_team
            elif margin < 0.05:
                keeper.team_id = TeamId.UNKNOWN
                keeper.team_confidence = float(margin)
                self.report.warnings.append(
                    f"goalkeeper track {keeper.track_id}: team assignment ambiguous "
                    f"(mean distance to A {mean_a:.1f}m vs B {mean_b:.1f}m)"
                )
            else:
                keeper.team_id = nearer
                keeper.team_confidence = float(min(1.0, margin * 4))

            side = None
            if keeper_points:
                mean_x = float(np.mean([p[0] for p in keeper_points]))
                side = "left" if mean_x < self.pitch.length / 2 else "right"

            self.report.goalkeepers[f"track_{keeper.track_id}"] = {
                "track_id": keeper.track_id,
                "team_id": keeper.team_id.value,
                "confidence": round(keeper.team_confidence, 4),
                "defends_side": side,
                "mean_distance_to_team_a_m": (
                    round(mean_a, 2) if np.isfinite(mean_a) else None
                ),
                "mean_distance_to_team_b_m": (
                    round(mean_b, 2) if np.isfinite(mean_b) else None
                ),
            }

    @staticmethod
    def _project(result: CalibrationResult, point: tuple[float, float]) -> np.ndarray | None:
        from visionpitch.common.geometry import apply_homography

        projected = apply_homography(result.homography, np.array([point]))
        if not np.isfinite(projected).all():
            return None
        return projected[0]

    # -- reporting ---------------------------------------------------------- #

    def _build_report(self, tracks: dict[int, Track]) -> None:
        self.report.n_tracks = len(tracks)
        if self.role_report is not None:
            self.report.roles = self.role_report.to_dict()
        self.report.team_counts = dict(Counter(t.team_id.value for t in tracks.values()))
        self.report.role_counts = dict(Counter(t.role.value for t in tracks.values()))
        for cluster, colour in self._cluster_colours.items():
            key = TeamId.A.value if cluster == 0 else TeamId.B.value
            self.report.team_colours[key] = [int(round(c)) for c in colour]

        n_unknown = self.report.team_counts.get(TeamId.UNKNOWN.value, 0)
        if n_unknown > 0.25 * max(1, len(tracks)):
            msg = (
                f"{n_unknown} of {len(tracks)} tracks could not be assigned to a team; "
                f"team-level statistics will be incomplete"
            )
            log.warning(msg)
            self.report.warnings.append(msg)

    # -- manual correction --------------------------------------------------- #

    @staticmethod
    def apply_corrections(tracks: dict[int, Track], corrections: dict) -> int:
        """Apply reviewer overrides without re-running detection or tracking.

        ``corrections`` maps track id to any of ``team_id``, ``role``,
        ``jersey_number``. Corrected fields are set to confidence 1.0 so that
        downstream consumers treat a human decision as ground truth, and so the
        difference between model output and human output stays visible.
        """
        applied = 0
        for raw_id, override in corrections.items():
            track = tracks.get(int(raw_id))
            if track is None:
                log.warning("correction for unknown track %s ignored", raw_id)
                continue
            if "team_id" in override:
                track.team_id = TeamId(override["team_id"])
                track.team_confidence = 1.0
            if "role" in override:
                track.role = Role(override["role"])
                track.role_confidence = 1.0
            if "jersey_number" in override:
                value = override["jersey_number"]
                track.jersey_number = int(value) if value is not None else None
                track.jersey_confidence = 1.0
            applied += 1
        if applied:
            log.info("applied %d manual track correction(s)", applied)
        return applied
