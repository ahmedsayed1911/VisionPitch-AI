"""Offline ball trajectory estimation by best-path search.

The problem
-----------
Per-frame ball detection on broadcast footage produces a noisy candidate set:
the true ball, plus a rotating cast of false positives -- the penalty spot, a
distant head, a boot, the corner flag base, a patch of sunlit line paint. The
true ball is frequently *missing* for runs of frames when it is occluded by a
player, leaves frame, or blurs into the crowd background.

Running a greedy nearest-neighbour tracker on that candidate set fails in a
characteristic way: one false positive near the prediction captures the track,
and because the filter then follows the false positive, it never recovers.

The approach
------------
Because Phase 1 prioritises accuracy over real-time (rule 20), this estimator is
**offline** and looks at the whole clip at once. It treats the candidates as a
directed acyclic graph -- one node per candidate detection, edges between
candidates in nearby frames whose separation implies a physically possible
speed -- and finds the single highest-scoring path through it by dynamic
programming.

That is a Viterbi decode over the candidate lattice, and its key property is
that a false positive can only capture the trajectory if doing so improves the
score of the *entire sequence*. An isolated blob near one frame's prediction
cannot, because the path would have to teleport out to it and back.

What it refuses to do
---------------------
Gaps longer than ``max_interpolation_gap_frames`` are left as ``None``. A ball
that has been invisible for a second is genuinely unknown, and inventing a
smooth interpolation across it would silently feed a fabricated position to the
Phase 2 possession engine. Every filled position is flagged ``interpolated``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from visionpitch.common.config import Config
from visionpitch.common.geometry import smooth_series
from visionpitch.common.logging import StageCounters, get_logger
from visionpitch.common.types import BallState, BBox, Detection, ObjectClass

log = get_logger("ball_tracking")


@dataclass(slots=True)
class _Candidate:
    frame_idx: int
    position: tuple[float, float]
    bbox: BBox
    confidence: float
    source: str


@dataclass(slots=True)
class _Node:
    candidate: _Candidate
    #: best cumulative score of any path ending at this node
    score: float = 0.0
    #: index of the predecessor node in the flat node list
    parent: int = -1


class BallTrajectoryEstimator:
    """Turns per-frame ball candidates into a validated temporal trajectory."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.cfg = config.ball_tracking
        self.counters = StageCounters("ball_tracking")
        #: why each rejection happened, aggregated
        self.rejections: Counter[str] = Counter()
        #: frame -> reason its candidate did not become an observation
        self._frame_reason: dict[int, str] = {}
        # Speed limits are expressed for a 1920px-wide frame and rescaled, so a
        # 720p or 4K source does not silently change the physics.
        self._reference_width = 1920.0

    # -- scoring ------------------------------------------------------------ #

    def _max_step(self, frame_width: int, gap: int) -> float:
        scale = frame_width / self._reference_width
        return self.cfg.max_speed_px_per_frame * scale * gap

    def _transition_score(
        self, a: _Candidate, b: _Candidate, frame_width: int
    ) -> float | None:
        """Score of following candidate ``a`` with candidate ``b``.

        Higher is better. ``None`` means the transition is physically
        impossible and no path may use it.
        """
        gap = b.frame_idx - a.frame_idx
        if gap <= 0 or gap > self.cfg.max_interpolation_gap_frames + 1:
            return None

        distance = float(np.hypot(b.position[0] - a.position[0], b.position[1] - a.position[1]))
        limit = self._max_step(frame_width, gap)
        if distance > limit:
            return None

        # Reward: the detector's own confidence in the destination.
        # Penalty: how much of the physical speed budget the step consumed, and
        # how many frames the ball had to be invisible to make the step work.
        speed_penalty = distance / max(1e-6, limit)
        gap_penalty = (gap - 1) / max(1, self.cfg.max_interpolation_gap_frames)
        return b.confidence - 0.6 * speed_penalty - 0.8 * gap_penalty

    # -- best path ---------------------------------------------------------- #

    def _all_paths(
        self, candidates: list[_Candidate], frame_width: int
    ) -> list[_Candidate]:
        """Extract every plausible trajectory segment, not just the best one.

        A single global best path is the wrong model for a football: the ball
        leaves the frame for a throw-in, is buried in a goalmouth scramble, or
        simply is not detected for two seconds, and no path may bridge a gap
        longer than ``max_interpolation_gap_frames`` by design.

        How segments are extracted
        -------------------------
        The dynamic program runs **once** over every candidate. Paths are then
        peeled off in descending order of score: take the best unclaimed node,
        walk back through its predecessors until reaching a node or a frame that
        another segment already owns, and accept the result if it is long enough.

        The earlier implementation instead re-ran the search and *stopped* the
        moment the best remaining path fell below the length floor. Because a
        short high-confidence pair can outscore a long chain of weak detections,
        that stop condition fired early and discarded every remaining candidate
        in the clip — measured at 26% of all frames that had a real ball
        detection. Peeling by score with a per-node claim removes the premature
        exit entirely.
        """
        if not candidates:
            return []

        nodes = self._run_dp(candidates, frame_width)
        order = sorted(range(len(nodes)), key=lambda i: -nodes[i].score)

        claimed_nodes = [False] * len(nodes)
        claimed_frames: set[int] = set()
        accepted: list[_Candidate] = []

        for start in order:
            if claimed_nodes[start]:
                continue

            path_indices: list[int] = []
            cursor = start
            while cursor != -1 and not claimed_nodes[cursor]:
                if nodes[cursor].candidate.frame_idx in claimed_frames:
                    break
                path_indices.append(cursor)
                cursor = nodes[cursor].parent

            if len(path_indices) < self.cfg.min_segment_frames:
                # Claim only this node, never its ancestors: they may still be
                # the backbone of a longer, better segment discovered later.
                claimed_nodes[start] = True
                self.rejections["segment_too_short"] += 1
                for i in path_indices:
                    self._frame_reason.setdefault(
                        nodes[i].candidate.frame_idx, "segment_too_short"
                    )
                continue

            for i in path_indices:
                claimed_nodes[i] = True
                claimed_frames.add(nodes[i].candidate.frame_idx)
                accepted.append(nodes[i].candidate)

        accepted.sort(key=lambda c: c.frame_idx)
        return accepted

    def _best_path(self, candidates: list[_Candidate], frame_width: int) -> list[_Candidate]:
        """Single highest-scoring path. Retained for tests and diagnostics."""
        if not candidates:
            return []
        nodes = self._run_dp(candidates, frame_width)
        best_idx = int(np.argmax([n.score for n in nodes]))
        path: list[_Candidate] = []
        while best_idx != -1:
            path.append(nodes[best_idx].candidate)
            best_idx = nodes[best_idx].parent
        path.reverse()
        return path

    def _run_dp(self, candidates: list[_Candidate], frame_width: int) -> list[_Node]:
        """Forward dynamic program over the candidate lattice."""
        nodes = [_Node(candidate=c, score=c.confidence) for c in sorted(
            candidates, key=lambda c: (c.frame_idx, -c.confidence)
        )]

        # Index the first node of each frame so the inner loop only scans the
        # reachable window rather than every earlier candidate.
        frame_starts: dict[int, int] = {}
        for i, node in enumerate(nodes):
            frame_starts.setdefault(node.candidate.frame_idx, i)

        window = self.cfg.max_interpolation_gap_frames + 1
        for node in nodes:
            f = node.candidate.frame_idx
            for prev_frame in range(f - window, f):
                start = frame_starts.get(prev_frame)
                if start is None:
                    continue
                j = start
                while j < len(nodes) and nodes[j].candidate.frame_idx == prev_frame:
                    step = self._transition_score(nodes[j].candidate, node.candidate, frame_width)
                    if step is not None:
                        total = nodes[j].score + step
                        if total > node.score:
                            node.score = total
                            node.parent = j
                    j += 1
        return nodes

    # -- physical plausibility ---------------------------------------------- #

    def _reject_implausible(
        self, path: list[_Candidate], frame_width: int
    ) -> list[_Candidate]:
        """Drop points whose implied acceleration is physically impossible.

        The first-order path search enforces a speed ceiling but is blind to
        *acceleration*: a candidate that sits mid-way between two real positions
        can be inserted at a modest speed cost while producing an absurd
        direction reversal. This second-order pass removes those.
        """
        if len(path) < 3:
            return path

        scale = frame_width / self._reference_width
        max_accel = 60.0 * scale  # px/frame^2; generous, a kicked ball is abrupt

        keep = [True] * len(path)
        for i in range(1, len(path) - 1):
            prev_c, cur_c, next_c = path[i - 1], path[i], path[i + 1]
            dt1 = max(1, cur_c.frame_idx - prev_c.frame_idx)
            dt2 = max(1, next_c.frame_idx - cur_c.frame_idx)
            # Only meaningful within one contiguous segment. Across the gap
            # between two segments the neighbours are unrelated sightings and
            # their implied acceleration says nothing.
            limit = self.cfg.max_interpolation_gap_frames + 1
            if dt1 > limit or dt2 > limit:
                continue
            v1 = (np.array(cur_c.position) - np.array(prev_c.position)) / dt1
            v2 = (np.array(next_c.position) - np.array(cur_c.position)) / dt2
            accel = np.linalg.norm(v2 - v1) / ((dt1 + dt2) / 2)
            if accel > max_accel:
                keep[i] = False
                self.counters.warn("implausible_acceleration_rejected")
                self.rejections["implausible_acceleration"] += 1
                self._frame_reason.setdefault(cur_c.frame_idx, "implausible_acceleration")

        return [c for c, k in zip(path, keep, strict=True) if k]

    # -- public API --------------------------------------------------------- #

    def estimate(
        self,
        detections_by_frame: dict[int, list[Detection]],
        frame_indices: list[int],
        timestamps: dict[int, float],
        frame_width: int,
    ) -> dict[int, BallState]:
        """Produce a ball state for every processed frame.

        Frames where the ball's position is genuinely unknown get a state with
        ``position=None``, not a guess.
        """
        candidates: list[_Candidate] = []
        for frame_idx, detections in detections_by_frame.items():
            for det in detections:
                if det.object_class is not ObjectClass.BALL:
                    continue
                candidates.append(
                    _Candidate(
                        frame_idx=frame_idx,
                        position=det.bbox.center,
                        bbox=det.bbox,
                        confidence=det.confidence,
                        source=det.source,
                    )
                )

        n_candidates = len(candidates)
        frames_with_candidates = {c.frame_idx for c in candidates}

        path = self._all_paths(candidates, frame_width)
        path = self._reject_implausible(path, frame_width)
        accepted = {c.frame_idx: c for c in path}

        # Anything that had evidence but produced no observation is accounted
        # for by name. Without this, a drop in ball coverage is invisible.
        for frame_idx in frames_with_candidates - set(accepted):
            reason = self._frame_reason.get(frame_idx, "not_on_any_accepted_path")
            self.rejections[reason] += 1
        self.rejections["frames_with_candidates"] = len(frames_with_candidates)
        self.rejections["frames_accepted"] = len(accepted)

        log.info(
            "ball: %d candidates over %d frames -> %d accepted on the best path",
            n_candidates,
            len(frame_indices),
            len(accepted),
        )
        self.counters.ok(len(accepted))
        self.counters.warn("candidates_rejected", max(0, n_candidates - len(accepted)))

        states = self._fill(accepted, frame_indices, timestamps)
        self._smooth(states, frame_indices)
        return states

    # -- gap filling -------------------------------------------------------- #

    def _fill(
        self,
        accepted: dict[int, _Candidate],
        frame_indices: list[int],
        timestamps: dict[int, float],
    ) -> dict[int, BallState]:
        observed_frames = sorted(accepted)
        states: dict[int, BallState] = {}

        for frame_idx in frame_indices:
            timestamp = timestamps.get(frame_idx, 0.0)
            candidate = accepted.get(frame_idx)

            if candidate is not None:
                states[frame_idx] = BallState(
                    frame_idx=frame_idx,
                    timestamp_s=timestamp,
                    position=candidate.position,
                    bbox=candidate.bbox,
                    velocity=None,
                    confidence=candidate.confidence,
                    observed=True,
                    interpolated=False,
                    uncertainty_px=2.0,
                )
                continue

            before = self._nearest(observed_frames, frame_idx, before=True)
            after = self._nearest(observed_frames, frame_idx, before=False)

            if before is None or after is None:
                # Outside the observed span: extrapolation here would be pure
                # invention, so the ball is reported as unknown.
                states[frame_idx] = self._unknown(frame_idx, timestamp)
                continue

            gap = after - before
            if gap > self.cfg.max_interpolation_gap_frames:
                states[frame_idx] = self._unknown(frame_idx, timestamp)
                self.counters.warn("gap_too_long_to_interpolate")
                continue

            a, b = accepted[before], accepted[after]
            t = (frame_idx - before) / gap
            position = (
                a.position[0] + t * (b.position[0] - a.position[0]),
                a.position[1] + t * (b.position[1] - a.position[1]),
            )
            # Uncertainty peaks in the middle of the gap and scales with its
            # length -- the honest shape for linear interpolation error.
            distance_from_anchor = min(frame_idx - before, after - frame_idx)
            uncertainty = 2.0 + 4.0 * distance_from_anchor

            width = 0.5 * (a.bbox.width + b.bbox.width)
            height = 0.5 * (a.bbox.height + b.bbox.height)
            confidence = float(
                min(a.confidence, b.confidence) * max(0.1, 1.0 - gap / (2 * (
                    self.cfg.max_interpolation_gap_frames + 1)))
            )

            states[frame_idx] = BallState(
                frame_idx=frame_idx,
                timestamp_s=timestamp,
                position=position,
                bbox=BBox(
                    position[0] - width / 2,
                    position[1] - height / 2,
                    position[0] + width / 2,
                    position[1] + height / 2,
                ),
                velocity=None,
                confidence=confidence,
                observed=False,
                interpolated=True,
                uncertainty_px=float(uncertainty),
            )

        return states

    @staticmethod
    def _nearest(sorted_frames: list[int], frame_idx: int, before: bool) -> int | None:
        import bisect

        if not sorted_frames:
            return None
        if before:
            pos = bisect.bisect_left(sorted_frames, frame_idx) - 1
            return sorted_frames[pos] if pos >= 0 else None
        pos = bisect.bisect_right(sorted_frames, frame_idx)
        return sorted_frames[pos] if pos < len(sorted_frames) else None

    @staticmethod
    def _unknown(frame_idx: int, timestamp: float) -> BallState:
        return BallState(
            frame_idx=frame_idx,
            timestamp_s=timestamp,
            position=None,
            bbox=None,
            velocity=None,
            confidence=0.0,
            observed=False,
            interpolated=False,
            uncertainty_px=float("inf"),
        )

    # -- smoothing and velocity --------------------------------------------- #

    def _smooth(self, states: dict[int, BallState], frame_indices: list[int]) -> None:
        """Smooth the trajectory and derive velocity, in place.

        Smoothing runs over *contiguous runs* of known positions. Averaging
        across an unknown gap would pull real positions toward a segment the
        estimator explicitly declined to fill.
        """
        run: list[int] = []
        for frame_idx in list(frame_indices) + [None]:
            known = frame_idx is not None and states[frame_idx].position is not None
            if known:
                run.append(frame_idx)
                continue
            if len(run) >= 3:
                self._smooth_run(states, run)
            elif len(run) >= 2:
                self._velocity_only(states, run)
            run = []

    def _smooth_run(self, states: dict[int, BallState], run: list[int]) -> None:
        xs = np.array([states[f].position[0] for f in run])
        ys = np.array([states[f].position[1] for f in run])
        window = min(self.cfg.smoothing_window, len(run) if len(run) % 2 else len(run) - 1)
        sx = smooth_series(xs, window)
        sy = smooth_series(ys, window)

        for i, frame_idx in enumerate(run):
            state = states[frame_idx]
            # Observed positions are only nudged toward the smoothed curve;
            # interpolated ones are replaced by it. Real measurements should not
            # be overwritten by a model of themselves.
            if state.observed:
                px = 0.7 * state.position[0] + 0.3 * sx[i]
                py = 0.7 * state.position[1] + 0.3 * sy[i]
            else:
                px, py = float(sx[i]), float(sy[i])
            state.position = (float(px), float(py))

        self._velocity_only(states, run)

    @staticmethod
    def _velocity_only(states: dict[int, BallState], run: list[int]) -> None:
        for i, frame_idx in enumerate(run):
            if i == 0:
                nxt = states[run[min(1, len(run) - 1)]]
                cur = states[frame_idx]
                dt = max(1, nxt.frame_idx - cur.frame_idx)
                vx = (nxt.position[0] - cur.position[0]) / dt
                vy = (nxt.position[1] - cur.position[1]) / dt
            else:
                prev = states[run[i - 1]]
                cur = states[frame_idx]
                dt = max(1, cur.frame_idx - prev.frame_idx)
                vx = (cur.position[0] - prev.position[0]) / dt
                vy = (cur.position[1] - prev.position[1]) / dt
            states[frame_idx].velocity = (float(vx), float(vy))

    # -- reporting ----------------------------------------------------------- #

    @staticmethod
    def quality_report(states: dict[int, BallState]) -> dict[str, float | int]:
        """Ball data-quality summary for the run manifest."""
        total = len(states)
        if total == 0:
            return {"frames": 0, "observed": 0, "interpolated": 0, "unknown": 0}
        observed = sum(1 for s in states.values() if s.observed)
        interpolated = sum(1 for s in states.values() if s.interpolated)
        unknown = total - observed - interpolated
        return {
            "frames": total,
            "observed": observed,
            "interpolated": interpolated,
            "unknown": unknown,
            "observed_ratio": round(observed / total, 4),
            "visible_ratio": round((observed + interpolated) / total, 4),
            "mean_confidence": round(
                float(np.mean([s.confidence for s in states.values()])), 4
            ),
        }

    def error_analysis(self) -> dict:
        """Where ball evidence was lost, by cause.

        ``frames_with_candidates`` is the ceiling the detector handed to this
        stage; ``frames_accepted`` is what survived. The difference, itemised,
        is what to attack — and it distinguishes a detector problem from a
        trajectory-search problem, which look identical in the headline number.
        """
        with_candidates = self.rejections.get("frames_with_candidates", 0)
        accepted = self.rejections.get("frames_accepted", 0)
        causes = {
            k: v
            for k, v in self.rejections.items()
            if k not in ("frames_with_candidates", "frames_accepted")
        }
        return {
            "frames_with_candidates": with_candidates,
            "frames_accepted": accepted,
            "acceptance_rate": (
                round(accepted / with_candidates, 4) if with_candidates else None
            ),
            "lost_frames": max(0, with_candidates - accepted),
            "causes": dict(sorted(causes.items(), key=lambda kv: -kv[1])),
        }
