"""Bounded-memory processing for full matches.

Why chunking is needed
----------------------
The single-pass pipeline holds every track, ball state, calibration result and
assembled row in memory for the whole clip. That is fine for a 45-second
validation clip and untenable for 90 minutes: at 25 fps a full match is ~135,000
frames and roughly 3 million game-state rows.

The design
----------
The video is split into fixed-length chunks that **overlap**. Each chunk is
processed independently and flushed to disk, so peak memory is a function of
chunk length rather than match length. The overlap is what makes the seams
recoverable: a track alive at the boundary appears in both chunks, and the
overlap gives the merger enough shared frames to recognise it as the same object.

Boundary correctness
--------------------
Two failure modes have to be prevented, and they pull in opposite directions:

* **Duplicate rows.** A frame inside the overlap is processed twice. Exactly one
  chunk owns each frame -- the earlier one -- and rows from the other are
  dropped at merge time. Ownership is decided by frame index alone, so it is
  deterministic and independent of processing order.
* **Broken identities.** A player crossing a boundary would otherwise get a new
  track id. Tracks are matched across the overlap on the frames they share, and
  the later chunk's ids are rewritten to the earlier chunk's.

Determinism
-----------
Merging depends only on frame indices and on the per-chunk outputs, never on
wall-clock order or on which chunk finished first. Re-running a chunk after a
failure therefore produces the same merged result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from visionpitch.common.logging import get_logger
from visionpitch.common.types import BallState, CalibrationResult, Track

log = get_logger("pipeline.chunking")


@dataclass(frozen=True)
class Chunk:
    """One unit of bounded-memory work."""

    index: int
    #: first frame this chunk decodes, including its lead-in overlap
    start_frame: int
    #: one past the last frame this chunk decodes
    end_frame: int
    #: first frame this chunk *owns*; frames before it belong to the previous
    #: chunk and exist here only to warm up the tracker and calibrator
    owned_start: int
    #: one past the last frame this chunk owns
    owned_end: int

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def warmup_frames(self) -> int:
        return self.owned_start - self.start_frame

    def owns(self, frame_idx: int) -> bool:
        return self.owned_start <= frame_idx < self.owned_end


def plan_chunks(
    first_frame: int, last_frame: int, chunk_frames: int, overlap_frames: int
) -> list[Chunk]:
    """Split a frame range into overlapping chunks with disjoint ownership.

    ``last_frame`` is exclusive. Ownership tiles the range exactly once, so
    concatenating owned frames reproduces the input range with no gaps and no
    duplicates -- which is the property the merge relies on.
    """
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")
    if overlap_frames < 0:
        raise ValueError("overlap_frames must not be negative")
    if overlap_frames >= chunk_frames:
        raise ValueError(
            f"overlap_frames ({overlap_frames}) must be smaller than "
            f"chunk_frames ({chunk_frames}), or chunks would not advance"
        )

    chunks: list[Chunk] = []
    owned_start = first_frame
    index = 0
    while owned_start < last_frame:
        owned_end = min(last_frame, owned_start + chunk_frames)
        # The lead-in exists so the tracker and calibrator reach steady state
        # before they produce rows anyone keeps.
        start = max(first_frame, owned_start - overlap_frames)
        chunks.append(
            Chunk(
                index=index,
                start_frame=start,
                end_frame=owned_end,
                owned_start=owned_start,
                owned_end=owned_end,
            )
        )
        owned_start = owned_end
        index += 1
    return chunks


# --------------------------------------------------------------------------- #
# Cross-chunk identity
# --------------------------------------------------------------------------- #


@dataclass
class MergeReport:
    chunks: int = 0
    tracks_before: int = 0
    tracks_after: int = 0
    identities_linked: int = 0
    duplicate_rows_dropped: int = 0
    unlinked_boundary_tracks: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "chunks": self.chunks,
            "tracks_before_merge": self.tracks_before,
            "tracks_after_merge": self.tracks_after,
            "identities_linked": self.identities_linked,
            "duplicate_rows_dropped": self.duplicate_rows_dropped,
            "unlinked_boundary_tracks": self.unlinked_boundary_tracks,
            "notes": self.notes,
        }


def _overlap_signature(track: Track, lo: int, hi: int) -> dict[int, np.ndarray]:
    """Box centres this track occupies within ``[lo, hi)``."""
    return {
        o.frame_idx: np.array(o.bbox.center, dtype=np.float64)
        for o in track.observations
        if lo <= o.frame_idx < hi and not o.interpolated
    }


def link_identities(
    earlier: dict[int, Track],
    later: dict[int, Track],
    overlap_lo: int,
    overlap_hi: int,
    max_centre_distance_px: float = 60.0,
    min_shared_frames: int = 2,
) -> dict[int, int]:
    """Map ``later``'s track ids onto ``earlier``'s where they are the same object.

    Matching is on the frames the two chunks genuinely share: the same physical
    player was tracked by both, so their boxes coincide there. Requiring several
    shared frames in agreement, rather than one, is what stops two players who
    happen to cross near the boundary from being merged.
    """
    if overlap_hi <= overlap_lo:
        return {}

    earlier_sigs = {
        tid: sig
        for tid, t in earlier.items()
        if (sig := _overlap_signature(t, overlap_lo, overlap_hi))
    }
    later_sigs = {
        tid: sig
        for tid, t in later.items()
        if (sig := _overlap_signature(t, overlap_lo, overlap_hi))
    }
    if not earlier_sigs or not later_sigs:
        return {}

    earlier_ids = sorted(earlier_sigs)
    later_ids = sorted(later_sigs)

    cost = np.full((len(later_ids), len(earlier_ids)), np.inf)
    for i, later_id in enumerate(later_ids):
        for j, earlier_id in enumerate(earlier_ids):
            shared = set(later_sigs[later_id]) & set(earlier_sigs[earlier_id])
            if len(shared) < min_shared_frames:
                continue
            distances = [
                float(np.linalg.norm(later_sigs[later_id][f] - earlier_sigs[earlier_id][f]))
                for f in shared
            ]
            mean_distance = float(np.mean(distances))
            if mean_distance > max_centre_distance_px:
                continue
            # Prefer more shared evidence at equal distance.
            cost[i, j] = mean_distance - 0.5 * len(shared)

    from scipy.optimize import linear_sum_assignment

    finite = np.isfinite(cost)
    if not finite.any():
        return {}
    workable = np.where(finite, cost, 1e6)
    rows, cols = linear_sum_assignment(workable)

    mapping: dict[int, int] = {}
    for r, c in zip(rows, cols, strict=True):
        if not finite[r, c]:
            continue
        mapping[later_ids[r]] = earlier_ids[c]
    return mapping


def merge_tracks(
    accumulated: dict[int, Track],
    incoming: dict[int, Track],
    mapping: dict[int, int],
    id_offset: int,
) -> tuple[dict[int, Track], int]:
    """Fold one chunk's tracks into the accumulated set.

    Unmatched incoming tracks are re-numbered above ``id_offset`` so ids stay
    globally unique across the whole match.
    """
    next_id = id_offset
    for track_id, track in sorted(incoming.items()):
        target_id = mapping.get(track_id)
        if target_id is not None and target_id in accumulated:
            target = accumulated[target_id]
            seen = {o.frame_idx for o in target.observations}
            # Only frames the target does not already have: the overlap is
            # processed twice and must not contribute twice.
            target.observations.extend(
                o for o in track.observations if o.frame_idx not in seen
            )
            target.observations.sort(key=lambda o: o.frame_idx)
            continue

        next_id += 1
        track.track_id = next_id
        accumulated[next_id] = track
    return accumulated, next_id


def merge_frame_keyed(
    accumulated: dict[int, object], incoming: dict[int, object], chunk: Chunk
) -> int:
    """Merge any frame-keyed mapping, keeping only frames this chunk owns.

    Returns how many incoming entries were dropped as duplicates.
    """
    dropped = 0
    for frame_idx, value in incoming.items():
        if not chunk.owns(frame_idx):
            dropped += 1
            continue
        accumulated[frame_idx] = value
    return dropped


def merge_ball_states(
    accumulated: dict[int, BallState], incoming: dict[int, BallState], chunk: Chunk
) -> int:
    return merge_frame_keyed(accumulated, incoming, chunk)  # type: ignore[arg-type]


def merge_calibration(
    accumulated: dict[int, CalibrationResult],
    incoming: dict[int, CalibrationResult],
    chunk: Chunk,
) -> int:
    return merge_frame_keyed(accumulated, incoming, chunk)  # type: ignore[arg-type]


def trim_tracks_to_owned(
    tracks: dict[int, Track], first_owned: int, last_owned: int
) -> dict[int, Track]:
    """Drop observations outside the owned range, and tracks left empty."""
    out: dict[int, Track] = {}
    for track_id, track in tracks.items():
        kept = [o for o in track.observations if first_owned <= o.frame_idx < last_owned]
        if not kept:
            continue
        track.observations = kept
        out[track_id] = track
    return out
