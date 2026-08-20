"""Event-level ground truth: schema, validation, and public-corpus adapters.

Phase 2B, Parts 1 and 2.

Schema design
-------------
An event annotation is an *interval*, not a point, because half the events that
matter (carries, possession spells) have a duration and the other half (a pass)
have a start the annotator can place precisely and an end they often cannot.
``end_time_s`` may equal ``start_time_s`` for a point event.

``UNKNOWN``, ``AMBIGUOUS`` and ``IGNORE`` are first-class labels. An annotator
who cannot tell what happened must be able to say so: forcing a choice
manufactures ground truth that is worse than no ground truth, because it looks
authoritative. ``IGNORE`` intervals are excluded from scoring entirely rather
than counted as either hits or misses.

Player identity is optional, and its absence is recorded rather than implied.
This matters: the largest public event corpus available here (SoccerNet Ball
Action Spotting) has no player labels at all, so any player-attribution metric
computed from it would be fabricated.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from visionpitch.common.logging import get_logger

log = get_logger("evaluation.event_gt")

EVENT_GT_SCHEMA_VERSION = "1.0.0"


class GTEventType(str, Enum):
    """Annotatable event vocabulary.

    Deliberately wider than the engine's own vocabulary: an annotator records
    what they saw, not what the engine can produce, and the mapping between the
    two is an explicit, reviewable decision made at evaluation time.
    """

    BALL_TOUCH = "ball_touch"
    POSSESSION_START = "possession_start"
    POSSESSION_END = "possession_end"
    PASS_START = "pass_start"
    PASS_RECEPTION = "pass_reception"
    PASS_SUCCESSFUL = "pass_successful"
    PASS_FAILED = "pass_failed"
    CARRY_START = "carry_start"
    CARRY_END = "carry_end"
    SHOT = "shot"
    INTERCEPTION = "interception"
    RECOVERY = "recovery"
    TURNOVER = "turnover"
    BALL_OUT = "ball_out"
    CROSS = "cross"
    HEADER = "header"
    RESTART = "restart"

    # -- explicit non-answers, never inferred -------------------------------- #
    UNKNOWN = "unknown"
    AMBIGUOUS = "ambiguous"
    IGNORE = "ignore"

    @property
    def is_scorable(self) -> bool:
        return self not in (
            GTEventType.UNKNOWN, GTEventType.AMBIGUOUS, GTEventType.IGNORE
        )


@dataclass
class GTEvent:
    """One annotated event."""

    event_type: GTEventType
    start_time_s: float
    end_time_s: float | None = None
    start_frame: int | None = None
    end_frame: int | None = None
    #: stable identity when known; ``None`` when the corpus has no player labels
    player_id: str | None = None
    track_id: int | None = None
    team: str | None = None
    #: annotator confidence in [0, 1]; 1.0 for expert corpora
    confidence: float = 1.0
    #: True when the frame index is derived from a timestamp rather than marked
    frame_is_approximate: bool = False
    notes: str = ""

    @property
    def duration_s(self) -> float:
        if self.end_time_s is None:
            return 0.0
        return max(0.0, self.end_time_s - self.start_time_s)

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type.value,
            "start_time_s": round(self.start_time_s, 4),
            "end_time_s": round(self.end_time_s, 4) if self.end_time_s is not None else None,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "player_id": self.player_id,
            "track_id": self.track_id,
            "team": self.team,
            "confidence": round(self.confidence, 4),
            "frame_is_approximate": self.frame_is_approximate,
            "notes": self.notes,
        }

    @staticmethod
    def from_dict(data: dict) -> GTEvent:
        return GTEvent(
            event_type=GTEventType(data["event_type"]),
            start_time_s=float(data["start_time_s"]),
            end_time_s=(
                float(data["end_time_s"]) if data.get("end_time_s") is not None else None
            ),
            start_frame=data.get("start_frame"),
            end_frame=data.get("end_frame"),
            player_id=data.get("player_id"),
            track_id=data.get("track_id"),
            team=data.get("team"),
            confidence=float(data.get("confidence", 1.0)),
            frame_is_approximate=bool(data.get("frame_is_approximate", False)),
            notes=data.get("notes", ""),
        )


@dataclass
class IgnoreInterval:
    """A stretch the evaluator must not score in either direction.

    Used for replays, crowd shots, and any interval the annotator judged
    unobservable. Counting a missed event inside a replay as a false negative
    would penalise the engine for correctly declining to hallucinate.
    """

    start_time_s: float
    end_time_s: float
    reason: str = ""

    def contains(self, timestamp_s: float) -> bool:
        return self.start_time_s <= timestamp_s <= self.end_time_s

    def to_dict(self) -> dict:
        return {
            "start_time_s": round(self.start_time_s, 4),
            "end_time_s": round(self.end_time_s, 4),
            "reason": self.reason,
        }


@dataclass
class EventGroundTruth:
    """A versioned, provenance-carrying event annotation set."""

    clip_id: str
    fps: float
    source: str
    licence: str
    events: list[GTEvent] = field(default_factory=list)
    ignore_intervals: list[IgnoreInterval] = field(default_factory=list)
    #: whether this corpus can support player-attribution metrics at all
    has_player_identity: bool = False
    annotator: str = ""
    notes: str = ""
    schema_version: str = EVENT_GT_SCHEMA_VERSION

    # -- selection ------------------------------------------------------------ #

    def scorable(self, event_types: set[GTEventType] | None = None) -> list[GTEvent]:
        """Events eligible for scoring: known type, outside every ignore span."""
        out = []
        for event in self.events:
            if not event.event_type.is_scorable:
                continue
            if event_types is not None and event.event_type not in event_types:
                continue
            if self.is_ignored(event.start_time_s):
                continue
            out.append(event)
        return out

    def is_ignored(self, timestamp_s: float) -> bool:
        return any(span.contains(timestamp_s) for span in self.ignore_intervals)

    @property
    def observable_duration_s(self) -> float:
        """Annotated span minus ignored intervals."""
        if not self.events:
            return 0.0
        span = max(e.start_time_s for e in self.events) - min(
            e.start_time_s for e in self.events
        )
        ignored = sum(i.end_time_s - i.start_time_s for i in self.ignore_intervals)
        return max(0.0, span - ignored)

    # -- provenance ------------------------------------------------------------ #

    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "clip_id": self.clip_id,
                "schema_version": self.schema_version,
                "events": [e.to_dict() for e in self.events],
                "ignore": [i.to_dict() for i in self.ignore_intervals],
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def summary(self) -> dict:
        from collections import Counter

        counts = Counter(e.event_type.value for e in self.events)
        return {
            "clip_id": self.clip_id,
            "schema_version": self.schema_version,
            "fingerprint": self.fingerprint(),
            "source": self.source,
            "licence": self.licence,
            "fps": round(self.fps, 4),
            "n_events": len(self.events),
            "n_scorable": len(self.scorable()),
            "n_ignore_intervals": len(self.ignore_intervals),
            "has_player_identity": self.has_player_identity,
            "counts": dict(counts.most_common()),
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    **self.summary(),
                    "annotator": self.annotator,
                    "notes": self.notes,
                    "events": [e.to_dict() for e in self.events],
                    "ignore_intervals": [i.to_dict() for i in self.ignore_intervals],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def load(path: str | Path) -> EventGroundTruth:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        gt = EventGroundTruth(
            clip_id=data["clip_id"],
            fps=float(data["fps"]),
            source=data.get("source", ""),
            licence=data.get("licence", ""),
            events=[GTEvent.from_dict(e) for e in data.get("events", [])],
            ignore_intervals=[
                IgnoreInterval(i["start_time_s"], i["end_time_s"], i.get("reason", ""))
                for i in data.get("ignore_intervals", [])
            ],
            has_player_identity=bool(data.get("has_player_identity", False)),
            annotator=data.get("annotator", ""),
            notes=data.get("notes", ""),
            schema_version=data.get("schema_version", EVENT_GT_SCHEMA_VERSION),
        )
        stored = data.get("fingerprint")
        if stored and stored != gt.fingerprint():
            log.warning(
                "%s: fingerprint changed since it was written (%s -> %s); the file "
                "has been edited outside the tool",
                Path(path).name, stored, gt.fingerprint(),
            )
        return gt


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def validate(gt: EventGroundTruth) -> dict:
    """Structural checks. Returns a report; never raises on annotator error."""
    issues: dict[str, list] = {
        "negative_duration": [],
        "end_before_start": [],
        "out_of_order": [],
        "duplicate_timestamps": [],
        "overlapping_ignore": [],
        "missing_team": [],
        "confidence_out_of_range": [],
        "identity_claimed_without_source": [],
    }

    previous = None
    seen: dict[tuple[str, float], int] = {}
    for event in gt.events:
        if event.end_time_s is not None and event.end_time_s < event.start_time_s:
            issues["end_before_start"].append(event.to_dict())
        if event.duration_s < 0:
            issues["negative_duration"].append(event.to_dict())
        if previous is not None and event.start_time_s < previous:
            issues["out_of_order"].append(event.start_time_s)
        previous = event.start_time_s

        key = (event.event_type.value, round(event.start_time_s, 3))
        seen[key] = seen.get(key, 0) + 1
        if not 0.0 <= event.confidence <= 1.0:
            issues["confidence_out_of_range"].append(event.to_dict())
        if event.event_type.is_scorable and event.team is None:
            issues["missing_team"].append(event.to_dict())
        if event.player_id is not None and not gt.has_player_identity:
            issues["identity_claimed_without_source"].append(event.to_dict())

    issues["duplicate_timestamps"] = [k for k, v in seen.items() if v > 1]

    spans = sorted(gt.ignore_intervals, key=lambda i: i.start_time_s)
    for a, b in zip(spans, spans[1:], strict=False):
        if b.start_time_s < a.end_time_s:
            issues["overlapping_ignore"].append((a.to_dict(), b.to_dict()))

    return {
        "valid": not any(issues.values()),
        "issue_counts": {k: len(v) for k, v in issues.items() if v},
        "issues": {k: v[:10] for k, v in issues.items() if v},
        "summary": gt.summary(),
    }


# --------------------------------------------------------------------------- #
# SoccerNet Ball Action Spotting adapter
# --------------------------------------------------------------------------- #

#: SN-BAS action vocabulary mapped onto the annotation schema.
#:
#: Several SN-BAS labels have no clean counterpart and are mapped to the nearest
#: honest option rather than forced. DRIVE is a player moving with the ball, so
#: it becomes a carry; HEADER is a touch, not necessarily a pass.
_BAS_MAPPING: dict[str, GTEventType] = {
    "PASS": GTEventType.PASS_START,
    "HIGH PASS": GTEventType.PASS_START,
    "CROSS": GTEventType.CROSS,
    "SHOT": GTEventType.SHOT,
    "OUT": GTEventType.BALL_OUT,
    "THROW IN": GTEventType.RESTART,
    "FREE KICK": GTEventType.RESTART,
    "DRIVE": GTEventType.CARRY_START,
    "HEADER": GTEventType.HEADER,
    "BALL PLAYER BLOCK": GTEventType.INTERCEPTION,
    "PLAYER SUCCESSFUL TACKLE": GTEventType.TURNOVER,
    # GOAL is spotted by SN-BAS but the engine emits only goal *candidates*, so
    # it is carried as AMBIGUOUS and never scored.
    "GOAL": GTEventType.AMBIGUOUS,
}

SOCCERNET_PASSWORD = b"s0cc3rn3t"


def load_soccernet_bas(
    archive: str | Path,
    half: int = 1,
    clip_id: str | None = None,
    fps: float = 25.0,
    time_offset_s: float = 0.0,
    max_time_s: float | None = None,
) -> EventGroundTruth:
    """Read SoccerNet Ball Action Spotting labels into the annotation schema.

    ``position`` in the SN-BAS label file is milliseconds from the start of the
    half, which is the anchor used here. ``gameTime`` is only second-resolution
    and would throw away most of the temporal precision the corpus provides.

    The corpus carries **no player identity**, so ``has_player_identity`` is
    False and every ``player_id`` is None. Any player-attribution metric
    computed against this data would be invented.
    """
    import pyzipper

    archive = Path(archive)
    with pyzipper.AESZipFile(archive) as zf:
        zf.setpassword(SOCCERNET_PASSWORD)
        label_names = [n for n in zf.namelist() if n.endswith("Labels-ball.json")]
        if not label_names:
            raise FileNotFoundError(f"no Labels-ball.json inside {archive}")
        raw = json.loads(zf.read(label_names[0]).decode("utf-8"))

    events: list[GTEvent] = []
    for entry in raw.get("annotations", []):
        game_time = str(entry.get("gameTime", ""))
        entry_half = int(game_time.split("-")[0].strip() or 1) if "-" in game_time else 1
        if entry_half != half:
            continue

        seconds = float(entry.get("position", 0)) / 1000.0 - time_offset_s
        if seconds < 0 or (max_time_s is not None and seconds > max_time_s):
            continue

        label = str(entry.get("label", "")).strip().upper()
        event_type = _BAS_MAPPING.get(label)
        if event_type is None:
            event_type = GTEventType.UNKNOWN

        # 'visibility' records whether the annotator could actually see the
        # action. A non-visible action is real but unobservable from this
        # footage, so it is annotated with reduced confidence rather than
        # dropped or treated as fully observable.
        visible = str(entry.get("visibility", "visible")).lower() == "visible"

        events.append(
            GTEvent(
                event_type=event_type,
                start_time_s=seconds,
                end_time_s=None,
                start_frame=int(round(seconds * fps)),
                # SN-BAS anchors on a timestamp, not a marked frame.
                frame_is_approximate=True,
                player_id=None,
                team=str(entry.get("team")) if entry.get("team") else None,
                confidence=1.0 if visible else 0.5,
                notes=f"SN-BAS:{label}" + ("" if visible else " (not visible)"),
            )
        )

    events.sort(key=lambda e: e.start_time_s)
    gt = EventGroundTruth(
        clip_id=clip_id or archive.stem,
        fps=fps,
        source="SoccerNet SN-BAS-2025 (Ball Action Spotting)",
        licence="SoccerNet terms; non-commercial research use",
        events=events,
        has_player_identity=False,
        annotator="SoccerNet expert annotators",
        notes=(
            "Ball actions anchored to a single millisecond timestamp. No interval "
            "ends and no player identity, so carry duration and player attribution "
            "cannot be scored against this corpus."
        ),
    )
    log.info(
        "SN-BAS half %d: %d events, %d scorable (%s)",
        half, len(gt.events), len(gt.scorable()), gt.fingerprint(),
    )
    return gt


def extract_bas_video(
    archive: str | Path, destination: str | Path, quality: str = "720p"
) -> Path:
    """Extract the match video from a SN-BAS archive."""
    import pyzipper

    archive, destination = Path(archive), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination

    with pyzipper.AESZipFile(archive) as zf:
        zf.setpassword(SOCCERNET_PASSWORD)
        matches = [n for n in zf.namelist() if n.endswith(f"{quality}.mp4")]
        if not matches:
            raise FileNotFoundError(f"no {quality}.mp4 inside {archive}")
        with zf.open(matches[0]) as source, destination.open("wb") as target:
            while chunk := source.read(1 << 22):
                target.write(chunk)
    log.info("extracted %s -> %s (%.2f GB)",
             matches[0], destination, destination.stat().st_size / 1e9)
    return destination
