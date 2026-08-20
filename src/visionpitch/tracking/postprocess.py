"""Offline track cleaning.

Online tracking is causal -- it must decide at frame *t* using only frames up to
*t*. Once the whole clip is tracked we know the future too, and can repair two
failure modes that no causal tracker can:

**Fragmentation.** A player occluded behind a group for a second reappears as a
new id. Offline we can see that track 47 ends where track 63 begins, moving in
the same direction at the same speed, and rejoin them. This is the single
largest lever on IDF1 and HOTA in football footage.

**Spurious tracks.** A detector false positive on a linesman, a ball boy, or a
sponsor board survives for three frames. A minimum-length filter removes it.

Both are reported as counters, never applied silently.
"""

from __future__ import annotations

import numpy as np

from visionpitch.common.config import TrackingConfig
from visionpitch.common.logging import get_logger
from visionpitch.common.types import Track

log = get_logger("tracking.postprocess")


def merge_class_votes(target: Track, source: Track) -> None:
    """Fold ``source``'s detector class evidence into ``target`` on a stitch.

    Losing this on merge would silently shrink the evidence base that role
    resolution depends on, and asymmetrically so: the absorbed fragment is often
    exactly the stretch where the detector changed its mind.
    """
    for key, weight in source.class_votes.items():
        target.class_votes[key] = target.class_votes.get(key, 0.0) + weight
    for key, count in source.class_counts.items():
        target.class_counts[key] = target.class_counts.get(key, 0) + count


def _velocity(track: Track, n: int = 5, at_end: bool = True) -> np.ndarray:
    """Mean per-frame velocity of the box centre over the last/first ``n`` frames."""
    obs = track.observations[-n:] if at_end else track.observations[:n]
    if len(obs) < 2:
        return np.zeros(2)
    p0 = np.array(obs[0].bbox.center)
    p1 = np.array(obs[-1].bbox.center)
    dt = max(1, obs[-1].frame_idx - obs[0].frame_idx)
    return (p1 - p0) / dt


def _stitch_score(
    a: Track, b: Track, max_gap: int, max_distance: float
) -> float | None:
    """Cost of joining track ``a`` (earlier) to track ``b`` (later), or ``None``.

    Returns ``None`` when the pair is inadmissible, so callers cannot
    accidentally stitch on a large finite cost.
    """
    gap = b.first_frame - a.last_frame
    if gap <= 0 or gap > max_gap:
        return None
    if a.object_class is not b.object_class:
        return None

    tail = a.observations[-1]
    head = b.observations[0]

    # Where would a's motion have put it at b's first frame?
    predicted = np.array(tail.bbox.center) + _velocity(a) * gap
    actual = np.array(head.bbox.center)
    distance = float(np.linalg.norm(predicted - actual))
    if distance > max_distance:
        return None

    # Box size must be consistent: a player does not double in height across an
    # occlusion, and size is our best proxy for depth.
    size_ratio = tail.bbox.height / max(1e-6, head.bbox.height)
    if not 0.6 <= size_ratio <= 1.67:
        return None

    # Prefer short gaps and small positional error, with the gap weighted
    # lightly -- a clean 20-frame rejoin beats a sloppy 3-frame one.
    return distance + 2.0 * gap


def stitch_tracks(
    tracks: dict[int, Track], config: TrackingConfig
) -> tuple[dict[int, Track], dict[int, int]]:
    """Greedily rejoin fragmented tracks.

    Returns the surviving tracks and a mapping from every absorbed track id to
    the id that absorbed it. Callers hold data keyed by the old ids -- jersey
    crops, appearance features -- and silently dropping that mapping would
    detach a merged track from half its own evidence.
    """
    if config.stitch_max_gap_frames <= 0:
        return tracks, {}

    ordered = sorted(tracks.values(), key=lambda t: t.first_frame)
    merged_into: dict[int, int] = {}
    result = {t.track_id: t for t in ordered}
    n_merged = 0

    for later in ordered:
        if later.track_id in merged_into:
            continue
        best_id, best_score = None, None
        for earlier in ordered:
            if earlier.track_id == later.track_id or earlier.track_id in merged_into:
                continue
            if earlier.track_id not in result:
                continue
            score = _stitch_score(
                result[earlier.track_id],
                later,
                config.stitch_max_gap_frames,
                config.stitch_max_distance_px,
            )
            if score is not None and (best_score is None or score < best_score):
                best_id, best_score = earlier.track_id, score

        if best_id is not None:
            target = result[best_id]
            target.observations.extend(later.observations)
            target.observations.sort(key=lambda o: o.frame_idx)
            merge_class_votes(target, later)
            merged_into[later.track_id] = best_id
            result.pop(later.track_id, None)
            n_merged += 1

    if n_merged:
        log.info("stitched %d fragmented track(s)", n_merged)

    # Collapse chains: if 9 merged into 5 and 5 merged into 2, then 9 -> 2.
    resolved: dict[int, int] = {}
    for source in merged_into:
        target = source
        seen = set()
        while target in merged_into and target not in seen:
            seen.add(target)
            target = merged_into[target]
        resolved[source] = target
    return result, resolved


def drop_short_tracks(
    tracks: dict[int, Track], min_length: int
) -> tuple[dict[int, Track], int]:
    """Remove tracks with too few real observations to be trustworthy."""
    kept, dropped = {}, 0
    for track_id, track in tracks.items():
        real = sum(1 for o in track.observations if not o.interpolated)
        if real >= min_length:
            kept[track_id] = track
        else:
            dropped += 1
    if dropped:
        log.info("dropped %d track(s) shorter than %d observations", dropped, min_length)
    return kept, dropped


def clean_tracks(
    tracks: dict[int, Track],
    config: TrackingConfig,
    calibration: dict | None = None,
    min_calibration_confidence: float = 0.5,
    appearance: dict[int, np.ndarray] | None = None,
) -> tuple[dict[int, Track], dict[int, int], dict]:
    """Full offline cleaning pass.

    Returns ``(cleaned_tracks, id_map, report)``. ``id_map`` maps absorbed track
    ids onto their survivor; ids not present were unchanged.

    Association strategy is configurable so the global solver can be measured
    against the greedy stitcher it replaced, rather than assumed better.
    """
    n_input = len(tracks)
    association_report: dict = {}

    if config.association == "global":
        from visionpitch.tracking.association import associate_tracklets

        stitched, id_map, assoc = associate_tracklets(
            tracks,
            config,
            calibration=calibration,
            min_calibration_confidence=min_calibration_confidence,
            appearance=appearance,
        )
        association_report = assoc.to_dict()
    elif config.association == "greedy":
        stitched, id_map = stitch_tracks(tracks, config)
    else:
        stitched, id_map = tracks, {}

    cleaned, n_dropped = drop_short_tracks(stitched, config.min_track_length)

    durations = [t.length for t in cleaned.values()]
    report = {
        "association": config.association,
        "association_detail": association_report,
        "tracks_in": n_input,
        "tracks_out": len(cleaned),
        "stitched": len(id_map),
        "p90_track_length": (
            float(np.percentile(durations, 90)) if durations else 0.0
        ),
        "max_track_length": float(max(durations)) if durations else 0.0,
        "dropped_short": n_dropped,
        "mean_track_length": (
            float(np.mean([t.length for t in cleaned.values()])) if cleaned else 0.0
        ),
        "median_track_length": (
            float(np.median([t.length for t in cleaned.values()])) if cleaned else 0.0
        ),
    }
    log.info(
        "track cleaning: %d -> %d (%d stitched, %d dropped), mean length %.1f frames",
        n_input,
        len(cleaned),
        len(id_map),
        n_dropped,
        report["mean_track_length"],
    )
    # An id that was absorbed into a track that was then dropped as too short
    # must not be remapped onto a track that no longer exists.
    id_map = {src: dst for src, dst in id_map.items() if dst in cleaned}
    return cleaned, id_map, report
