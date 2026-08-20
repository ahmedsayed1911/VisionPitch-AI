"""Regression coverage for the crop budget and the showcase overlay.

The crop-budget tests exist because the failure they guard against was silent:
the harvester filled its global cap in the opening two minutes of a nine-minute
broadcast, every track that first appeared afterwards harvested zero crops, and
those tracks were reported as UNKNOWN team rather than as starved. Nothing threw,
nothing warned loudly, and the only visible symptom was a team-classification
rate that looked like a model problem instead of a budget problem.
"""

from __future__ import annotations

import numpy as np
import pytest

from visionpitch.common.config import ShowcaseConfig, load_config
from visionpitch.common.types import ObjectClass, Role, TeamId
from visionpitch.ingestion.video import Frame
from visionpitch.pipeline.runner import _CropHarvester
from visionpitch.pitch.geometry import PitchConfiguration
from visionpitch.visualization.showcase import (
    Eligibility,
    ShowcasePlayer,
    ShowcaseRenderer,
    _delaunay_edges,
    _EdgeMemory,
    _knn_edges,
    eligibility,
)

# --------------------------------------------------------------------------- #
# Crop budget
# --------------------------------------------------------------------------- #


class _FakeExtractor:
    """Returns a distinct sentinel per call so crops stay attributable."""

    def __init__(self) -> None:
        self.calls = 0

    def extract(self, image, bbox, track_id, frame_idx):
        self.calls += 1
        return _FakeCrop(track_id=track_id, frame_idx=frame_idx)


class _FakeCrop:
    __slots__ = ("track_id", "frame_idx")

    def __init__(self, track_id: int, frame_idx: int) -> None:
        self.track_id = track_id
        self.frame_idx = frame_idx


class _FakeClassifier:
    def __init__(self) -> None:
        self.extractor = _FakeExtractor()


class _FakeTrack:
    """The shape ``_CropHarvester.harvest`` reads off a live tracker track."""

    def __init__(self, track_id: int, object_class=ObjectClass.PLAYER) -> None:
        self.track_id = track_id
        self.object_class = object_class
        self.bbox_array = np.array([10.0, 20.0, 50.0, 110.0])


def _harvest_video(
    harvester: _CropHarvester,
    n_frames: int,
    tracks_at: dict[int, list[_FakeTrack]],
) -> None:
    for idx in range(n_frames):
        frame = Frame(idx=idx, timestamp_s=idx / 50.0, image=np.zeros((8, 8, 3), np.uint8))
        harvester.harvest(frame, tracks_at.get(idx, []))


class TestCropBudget:
    def test_per_track_cap_is_enforced(self) -> None:
        harvester = _CropHarvester(_FakeClassifier(), stride=1, max_crops=10_000, max_per_track=40)
        track = _FakeTrack(1)
        _harvest_video(harvester, 500, dict.fromkeys(range(500), [track]))

        assert harvester.per_track[1] == 40
        assert len(harvester.crops) == 40
        assert harvester.dropped_track_cap == 460

    def test_budget_is_shared_across_the_whole_clip(self) -> None:
        """The regression, stated as the property that actually broke.

        Tracks appear across the whole broadcast, not at the start, and together
        they want far more crops than the budget allows. The bug was that the
        budget was spent chronologically, so tracks born after it ran out
        harvested nothing at all and were reported UNKNOWN team rather than as
        starved. What must hold is that *when* a track appears does not decide
        whether it gets a vote.
        """
        config = load_config()
        n_tracks, visible_frames, budget = 20, 300, 700
        n_frames = n_tracks * visible_frames
        tracks = [_FakeTrack(i) for i in range(n_tracks)]
        schedule = {idx: [tracks[idx // visible_frames]] for idx in range(n_frames)}

        harvester = _CropHarvester(
            _FakeClassifier(),
            stride=config.team_classification.fit_stride_frames,
            max_crops=budget,
            max_per_track=40,
        )
        _harvest_video(harvester, n_frames, schedule)

        counts = harvester.per_track
        assert len(counts) == n_tracks, "every track must hold at least one crop"
        assert min(counts.values()) >= config.team_classification.min_votes
        # Fair, not merely non-zero: the earliest and latest tracks end up with
        # comparable evidence.
        assert max(counts.values()) - min(counts.values()) <= 0.4 * budget / n_tracks
        first_half = sum(counts[i] for i in range(n_tracks // 2))
        second_half = sum(counts[i] for i in range(n_tracks // 2, n_tracks))
        assert 0.7 <= first_half / second_half <= 1.4

    def test_global_budget_is_never_exceeded(self) -> None:
        """Eviction keeps the memory bound exact, not approximate."""
        budget = 250
        harvester = _CropHarvester(
            _FakeClassifier(), stride=1, max_crops=budget, max_per_track=10_000
        )
        tracks = [_FakeTrack(i) for i in range(12)]
        _harvest_video(harvester, 400, dict.fromkeys(range(400), tracks))
        assert len(harvester.crops) <= budget
        assert harvester.evicted > 0

    def test_eviction_preserves_temporal_spread(self) -> None:
        """A track's crops are only a vote if they sample the clip, not a moment."""
        harvester = _CropHarvester(
            _FakeClassifier(), stride=1, max_crops=20, max_per_track=10_000
        )
        tracks = [_FakeTrack(0), _FakeTrack(1)]
        _harvest_video(harvester, 200, dict.fromkeys(range(200), tracks))
        for crops in harvester.by_track().values():
            frames = [c.frame_idx for c in crops]
            assert frames == sorted(frames)
            assert frames[0] < 20, "the earliest evidence must survive"
            assert frames[-1] > 150, "so must the latest"

    def test_a_sufficient_budget_lets_every_track_be_voted_on(self) -> None:
        """With headroom, no track is left below ``min_votes``."""
        config = load_config()
        min_votes = config.team_classification.min_votes
        n_tracks, visible_frames = 60, 60
        n_frames = n_tracks * visible_frames
        tracks = [_FakeTrack(i) for i in range(n_tracks)]
        schedule = {idx: [tracks[idx // visible_frames]] for idx in range(n_frames)}

        harvester = _CropHarvester(
            _FakeClassifier(),
            stride=config.team_classification.fit_stride_frames,
            max_crops=n_tracks * 40,
            max_per_track=40,
        )
        _harvest_video(harvester, n_frames, schedule)

        assert len(harvester.per_track) == n_tracks
        assert min(harvester.per_track.values()) >= min_votes

    def test_referee_class_tracks_are_harvested(self) -> None:
        """Referee tracks need crops, or a false referee call cannot be overruled."""
        harvester = _CropHarvester(_FakeClassifier(), stride=1, max_crops=100, max_per_track=40)
        referee = _FakeTrack(7, object_class=ObjectClass.REFEREE)
        ball = _FakeTrack(8, object_class=ObjectClass.BALL)
        _harvest_video(harvester, 30, dict.fromkeys(range(30), [referee, ball]))

        assert harvester.per_track.get(7) == 30
        assert 8 not in harvester.per_track

    def test_remap_follows_stitching_and_drops_dead_tracks(self) -> None:
        harvester = _CropHarvester(_FakeClassifier(), stride=1, max_crops=100, max_per_track=40)
        _harvest_video(
            harvester, 6, dict.fromkeys(range(6), [_FakeTrack(3), _FakeTrack(4), _FakeTrack(9)])
        )
        harvester.remap({4: 3}, valid_ids={3})

        assert {c.track_id for c in harvester.crops} == {3}
        assert len(harvester.by_track()[3]) == 12


# --------------------------------------------------------------------------- #
# Showcase overlay
# --------------------------------------------------------------------------- #


@pytest.fixture
def renderer() -> ShowcaseRenderer:
    return ShowcaseRenderer(ShowcaseConfig(enabled=True), PitchConfiguration())


def _player(track_id: int, team: str, x: float, y: float) -> ShowcasePlayer:
    return ShowcasePlayer(
        track_id=track_id,
        team_id=team,
        role=Role.OUTFIELD.value,
        image_xy=(200.0 + 12.0 * x, 400.0 + 8.0 * y),
        pitch_xy=(x, y),
    )


class TestShowcaseContent:
    """What the showcase is allowed to put on screen, and nothing more."""

    def test_the_ball_is_never_drawn(self, renderer: ShowcaseRenderer) -> None:
        """The ball must read exactly as it does in the source broadcast.

        The renderer takes no ball argument at all, so this pins the contract as
        much as the pixels: a ball position cannot reach it.
        """
        import inspect

        params = set(inspect.signature(renderer.render).parameters)
        assert not any("ball" in name for name in params)
        assert not hasattr(renderer, "_draw_ball")

    def test_no_pitch_lanes_are_drawn_on_a_calibrated_frame(self) -> None:
        """Projected red/yellow lanes are gone, including when calibration is good."""
        renderer = ShowcaseRenderer(ShowcaseConfig(enabled=True), PitchConfiguration())
        image = np.full((720, 1280, 3), 60, np.uint8)
        out = renderer.render(image.copy(), [], calibration_confidence=0.95, n_inliers=40)
        assert np.array_equal(out, image), "a lane or guide was drawn"
        assert not hasattr(renderer, "_draw_lanes")

    def test_an_empty_frame_is_returned_untouched(self, renderer: ShowcaseRenderer) -> None:
        image = np.full((720, 1280, 3), 60, np.uint8)
        assert np.array_equal(renderer.render(image.copy(), [], 0.0, 0), image)


class TestMinimumSupport:
    """A frame with nothing to connect is abstained on, not partly drawn."""

    def test_a_lone_player_is_not_drawn(self, renderer: ShowcaseRenderer) -> None:
        """One dot can carry no edge, so it shows no tactical relation at all.

        This is how a team dot came to be drawn over the inset clip on a
        YouTube end card: every other person in the frame was UNKNOWN, one
        resolved, and the renderer drew it.
        """
        image = np.full((720, 1280, 3), 60, np.uint8)
        lone = [_player(1, TeamId.A.value, 40, 30)]
        assert np.array_equal(renderer.render(image.copy(), lone, 0.9, 20), image)

    def test_a_lone_player_plus_a_referee_is_still_abstained_on(
        self, renderer: ShowcaseRenderer
    ) -> None:
        image = np.full((720, 1280, 3), 60, np.uint8)
        people = [
            _player(1, TeamId.A.value, 40, 30),
            ShowcasePlayer(9, TeamId.NONE.value, Role.REFEREE.value, (600.0, 400.0), None),
        ]
        assert np.array_equal(renderer.render(image.copy(), people, 0.9, 20), image)

    def test_two_players_are_drawn(self, renderer: ShowcaseRenderer) -> None:
        image = np.full((720, 1280, 3), 60, np.uint8)
        pair = [_player(1, TeamId.A.value, 40, 30), _player(2, TeamId.B.value, 44, 31)]
        assert not np.array_equal(renderer.render(image.copy(), pair, 0.9, 20), image)

    def test_the_threshold_is_configurable_and_zero_disables_it(self) -> None:
        cfg = ShowcaseConfig(enabled=True, min_team_players_per_frame=0)
        renderer = ShowcaseRenderer(cfg, PitchConfiguration())
        image = np.full((720, 1280, 3), 60, np.uint8)
        lone = [_player(1, TeamId.A.value, 40, 30)]
        assert not np.array_equal(renderer.render(image.copy(), lone, 0.9, 20), image)


class TestEligibility:
    """Role outranks team assignment. This is the boundary that decides."""

    def test_a_referee_carrying_a_team_id_is_still_a_referee(self) -> None:
        """The case that put officials into team graphs on the real broadcast.

        Team classification fits exactly two clusters, so an official's kit is
        forced into whichever is nearer and reports high confidence for a colour
        matching neither. The renderer must not believe it.
        """
        official = ShowcasePlayer(
            1, TeamId.A.value, Role.REFEREE.value, (500.0, 400.0), (50.0, 30.0)
        )
        assert eligibility(official) is Eligibility.REFEREE

    def test_players_and_goalkeepers_are_team_eligible(self) -> None:
        for role in (Role.OUTFIELD.value, Role.GOALKEEPER.value):
            for team in (TeamId.A.value, TeamId.B.value):
                player = ShowcasePlayer(1, team, role, (500.0, 400.0), (50.0, 30.0))
                assert eligibility(player) is Eligibility.TEAM, (role, team)

    def test_unresolved_tracks_abstain(self) -> None:
        for team, role in (
            (TeamId.UNKNOWN.value, Role.OUTFIELD.value),
            (TeamId.A.value, Role.UNKNOWN.value),
            (TeamId.NONE.value, Role.OUTFIELD.value),
        ):
            player = ShowcasePlayer(1, team, role, (500.0, 400.0), (50.0, 30.0))
            assert eligibility(player) is Eligibility.NONE, (team, role)

    def test_a_referee_is_drawn_in_its_own_colour(self) -> None:
        """One dot, in a colour that is neither team's.

        Two teammates sit far away on the other side of the frame purely to give
        the frame enough resolved players to be drawn at all; the assertions
        below look only at the referee's own neighbourhood.
        """
        cfg = ShowcaseConfig(enabled=True)
        renderer = ShowcaseRenderer(cfg, PitchConfiguration())
        image = np.full((720, 1280, 3), 60, np.uint8)
        official = ShowcasePlayer(
            1, TeamId.NONE.value, Role.REFEREE.value, (640.0, 360.0), None
        )
        elsewhere = [_player(2, TeamId.A.value, 4, 4), _player(3, TeamId.A.value, 7, 5)]
        out = renderer.render(image.copy(), [official, *elsewhere], 0.0, 0)

        near = out[344:376, 624:656].reshape(-1, 3)
        near_colours = {tuple(int(v) for v in px) for px in near}
        assert tuple(reversed(cfg.referee_color)) in near_colours, "referee not drawn"
        for team_rgb in cfg.team_colors.values():
            assert tuple(reversed(team_rgb)) not in near_colours, (
                "referee got a team colour"
            )


class TestTacticalGraph:
    def test_referees_never_enter_a_team_graph(self, renderer: ShowcaseRenderer) -> None:
        """Hard invariant: zero referee-involving edges, however the data arrives.

        The official sits in the middle of a tight group of team A players *and*
        carries team A's id, so a renderer filtering on team rather than role
        would connect it to everything nearby.
        """
        image = np.full((720, 1280, 3), 60, np.uint8)
        squad = [_player(i, TeamId.A.value, 40 + 3 * i, 30) for i in range(1, 6)]
        official = ShowcasePlayer(
            99,
            TeamId.A.value,
            Role.REFEREE.value,
            (200.0 + 12.0 * 46, 400.0 + 8.0 * 30),
            (46.0, 30.0),
        )
        for _ in range(8):
            renderer.render(image.copy(), [*squad, official], 0.9, 20)

        assert renderer.last_edges, "the squad should be connected"
        for _team, pa, pb in renderer.last_edges:
            assert Role.REFEREE.value not in (pa.role, pb.role)
            assert 99 not in (pa.track_id, pb.track_id)

    def test_no_cross_team_edge_is_possible(self, renderer: ShowcaseRenderer) -> None:
        """Interleaved teams: every edge must still stay inside one team."""
        image = np.full((720, 1280, 3), 60, np.uint8)
        players = []
        for i in range(8):
            team = TeamId.A.value if i % 2 == 0 else TeamId.B.value
            players.append(_player(i + 1, team, 40 + 2.0 * i, 30 + (i % 3)))
        for _ in range(8):
            renderer.render(image.copy(), players, 0.9, 20)

        assert renderer.last_edges
        for team, pa, pb in renderer.last_edges:
            assert pa.team_id == pb.team_id == team

    def test_distant_teammates_are_not_connected(self, renderer: ShowcaseRenderer) -> None:
        """The core visual rule: locality, not team membership, decides."""
        near = _player(1, TeamId.A.value, 10, 30)
        far = _player(2, TeamId.A.value, 100, 30)
        for _ in range(10):
            edges = renderer._team_edges(TeamId.A.value, [near, far], calibrated=True)
        assert edges == []

    def test_separate_local_groups_stay_separate(self, renderer: ShowcaseRenderer) -> None:
        """Multiple disconnected groups are expected and desired."""
        left = [_player(i, TeamId.A.value, 12 + 2.0 * i, 20 + i) for i in range(4)]
        right = [_player(10 + i, TeamId.A.value, 88 + 2.0 * i, 45 + i) for i in range(4)]
        for _ in range(8):
            edges = renderer._team_edges(TeamId.A.value, left + right, calibrated=True)

        assert edges, "each cluster should be internally connected"
        left_ids = {p.track_id for p in left}
        for pa, pb in edges:
            assert (pa.track_id in left_ids) == (pb.track_id in left_ids), (
                "a group boundary was crossed"
            )

    def test_hysteresis_never_outlives_the_distance_rule(self) -> None:
        """Distance validity always wins over temporal smoothing."""
        renderer = ShowcaseRenderer(ShowcaseConfig(enabled=True), PitchConfiguration())
        pair = [_player(1, TeamId.A.value, 40, 30), _player(2, TeamId.A.value, 44, 30)]
        for _ in range(8):
            edges = renderer._team_edges(TeamId.A.value, pair, calibrated=True)
        assert edges, "a close pair should connect"

        separated = [_player(1, TeamId.A.value, 10, 30), _player(2, TeamId.A.value, 95, 30)]
        edges = renderer._team_edges(TeamId.A.value, separated, calibrated=True)
        assert edges == [], "an edge survived the players separating"

    def test_edges_survive_a_dropped_proposal(self) -> None:
        """Hysteresis: a triangulation flicker must not blink the graph off.

        Both players stay on screen throughout; only the *proposal* lapses, which
        is exactly what Delaunay does when four players approach co-circularity.
        """
        memory = _EdgeMemory(on_frames=3, off_frames=8)
        edge, alive = (1, 2), {1, 2}
        for _ in range(3):
            memory.step({edge}, alive)
        assert edge in memory.drawn

        for _ in range(4):
            assert memory.step(set(), alive) == {edge}
        assert memory.step({edge}, alive) == {edge}

    def test_edges_drop_once_a_pair_leaves_the_frame(self) -> None:
        memory = _EdgeMemory(on_frames=2, off_frames=8)
        for _ in range(3):
            memory.step({(1, 2)}, {1, 2})
        assert memory.step(set(), {1}) == set()

    def test_delaunay_falls_back_on_collinear_points(self) -> None:
        """A flat defensive line is degenerate; kNN has to cover it."""
        collinear = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        assert _delaunay_edges(collinear) is None
        assert _knn_edges(collinear, 2)
