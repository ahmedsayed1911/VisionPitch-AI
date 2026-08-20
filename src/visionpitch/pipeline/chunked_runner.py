"""Full-match orchestration on top of the single-pass pipeline.

Runs :class:`Phase1Pipeline` over overlapping chunks and merges the results.
Peak memory is bounded by chunk length, not match length.

Each chunk's merged-so-far state is checkpointed after it completes, so a run
interrupted at chunk 47 of 90 resumes at 47 rather than restarting. The
checkpoint is written with an atomic replace, so a crash mid-write cannot leave
a corrupt state that a later run would trust.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from visionpitch.common.config import Config
from visionpitch.common.logging import get_logger
from visionpitch.common.schema import GameStateRow
from visionpitch.game_state.assembler import GameStateAssembler
from visionpitch.game_state.discovery import discover_match_setup
from visionpitch.ingestion.video import VideoMetadata, probe_video
from visionpitch.pipeline.chunking import (
    Chunk,
    MergeReport,
    link_identities,
    merge_ball_states,
    merge_calibration,
    merge_tracks,
    plan_chunks,
    trim_tracks_to_owned,
)
from visionpitch.pipeline.runner import Phase1Pipeline, PipelineResult
from visionpitch.pitch.geometry import PitchConfiguration
from visionpitch.storage.run import RunContext
from visionpitch.storage.tables import (
    calibration_rows,
    frame_rows,
    track_rows,
    write_calibration,
    write_frames,
    write_game_state,
    write_tracks,
)

log = get_logger("pipeline.chunked")


@dataclass
class ChunkedResult:
    run_dir: Path
    metadata: VideoMetadata
    chunks: list[Chunk] = field(default_factory=list)
    merge: MergeReport = field(default_factory=MergeReport)
    rows: int = 0
    tracks: int = 0
    outputs: dict[str, str] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    reports: dict = field(default_factory=dict)


class ChunkedPipeline:
    """Processes a long video in bounded-memory chunks."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.pitch = PitchConfiguration(
            length=config.pitch.length_m, width=config.pitch.width_m
        )

    # -- chunk execution ----------------------------------------------------- #

    def _run_chunk(
        self, video_path: Path, chunk: Chunk, metadata: VideoMetadata
    ) -> PipelineResult:
        """Process one chunk with the ordinary single-pass pipeline."""
        fps = metadata.fps
        chunk_config = self.config.model_copy(deep=True)
        chunk_config.ingestion.start_time_s = chunk.start_frame / fps
        chunk_config.ingestion.end_time_s = chunk.end_frame / fps
        chunk_config.ingestion.max_frames = None
        # Chunk artefacts are merged and rewritten; per-chunk videos and tables
        # would be redundant output measured in gigabytes on a full match.
        chunk_config.visualization.write_annotated_video = False
        chunk_config.visualization.write_tactical_map = False
        chunk_config.visualization.write_combined_video = False
        chunk_config.storage.write_raw_detections = self.config.storage.write_raw_detections

        pipeline = Phase1Pipeline(chunk_config)
        return pipeline.run(video_path, render=False)

    # -- entry point --------------------------------------------------------- #

    def run(self, video_path: str | Path, resume: bool = True) -> ChunkedResult:
        video_path = Path(video_path)
        metadata = probe_video(video_path)
        run = RunContext(self.config, metadata).ensure()
        started = time.perf_counter()

        fps = metadata.fps
        first = int(round((self.config.ingestion.start_time_s or 0.0) * fps))
        last = (
            int(round(self.config.ingestion.end_time_s * fps))
            if self.config.ingestion.end_time_s is not None
            else metadata.frame_count
        )
        chunks = plan_chunks(
            first,
            last,
            self.config.chunking.chunk_frames,
            self.config.chunking.overlap_frames,
        )
        log.info(
            "planned %d chunk(s) of %d frames with %d-frame overlap over frames %d-%d",
            len(chunks),
            self.config.chunking.chunk_frames,
            self.config.chunking.overlap_frames,
            first,
            last,
        )

        state_path = run.path("checkpoints", "chunked_state.json")
        completed = self._load_state(state_path) if resume else set()

        accumulated_tracks: dict = {}
        accumulated_ball: dict = {}
        accumulated_calibration: dict = {}
        timestamps: dict[int, float] = {}
        chunk_of: dict[int, int] = {}
        support_regions: dict = {}
        next_id = 0
        report = MergeReport(chunks=len(chunks))

        previous_tracks: dict = {}
        previous_chunk: Chunk | None = None
        #: Model provenance, taken from the chunks that actually ran. Without
        #: this the chunked path writes a manifest with an empty `models` block,
        #: so nothing downstream -- an event review, a report, a re-run -- can
        #: say which weights produced the results.
        models: dict[str, dict] = {}

        for chunk in chunks:
            if chunk.index in completed and resume:
                log.info("chunk %d already complete; skipping", chunk.index)
                continue

            log.info(
                "chunk %d/%d: frames %d-%d (owns %d-%d)",
                chunk.index + 1,
                len(chunks),
                chunk.start_frame,
                chunk.end_frame,
                chunk.owned_start,
                chunk.owned_end,
            )
            result = self._run_chunk(video_path, chunk, metadata)

            timestamps.update(result.timestamps)
            # Direct attribute access, not getattr with a default: if this field
            # ever disappears again the merge must fail loudly rather than
            # quietly assemble a run in which nothing can be marked extrapolated
            # -- which is exactly how a chunked run once produced zero usable
            # player rows without a single error.
            support_regions.update(result.support_regions)

            for key in ("detector", "ball_detector", "pitch_keypoints"):
                report_for_key = result.reports.get(key)
                if report_for_key is None:
                    continue
                previous = models.get(key)
                if previous is not None and previous.get(
                    "weights_sha256"
                ) != report_for_key.get("weights_sha256"):
                    # Chunks are meant to be the same pipeline over a longer
                    # video. Different weights mid-run means the merged tables
                    # mix two models, which no single manifest can describe.
                    raise RuntimeError(
                        f"chunk {chunk.index} used different {key} weights than "
                        f"earlier chunks ({report_for_key.get('weights_sha256')} != "
                        f"{previous.get('weights_sha256')}); the merged run would "
                        f"have no single model provenance"
                    )
                models[key] = report_for_key

            for frame_idx in result.frame_indices:
                if chunk.owns(frame_idx):
                    chunk_of[frame_idx] = chunk.index

            mapping: dict[int, int] = {}
            if previous_chunk is not None and previous_tracks:
                mapping = link_identities(
                    previous_tracks,
                    result.tracks,
                    overlap_lo=chunk.start_frame,
                    overlap_hi=previous_chunk.owned_end,
                )
                report.identities_linked += len(mapping)
                unlinked = sum(
                    1
                    for t in result.tracks.values()
                    if t.first_frame < chunk.owned_start and t.track_id not in mapping
                )
                report.unlinked_boundary_tracks += unlinked

            owned = trim_tracks_to_owned(
                result.tracks, chunk.owned_start, chunk.owned_end
            )
            # Identity linking is computed on the overlap, but only owned
            # observations are folded in, so the overlap contributes once.
            accumulated_tracks, next_id = merge_tracks(
                accumulated_tracks, owned, mapping, next_id
            )
            report.duplicate_rows_dropped += merge_ball_states(
                accumulated_ball, result.ball_states, chunk
            )
            report.duplicate_rows_dropped += merge_calibration(
                accumulated_calibration, result.calibration, chunk
            )

            previous_tracks = result.tracks
            previous_chunk = chunk
            completed.add(chunk.index)
            self._save_state(state_path, completed)

        report.tracks_before = report.tracks_after = len(accumulated_tracks)

        # -- assemble the merged match ---------------------------------------- #
        frame_indices = sorted(timestamps)
        assembler = GameStateAssembler(
            metadata.video_id,
            self.pitch,
            self.config.calibration.min_confidence,
            support_regions=support_regions,
            max_extrapolation_risk=self.config.calibration.max_extrapolation_risk,
        )
        rows: list[GameStateRow] = assembler.assemble(
            accumulated_tracks,
            accumulated_ball,
            accumulated_calibration,
            timestamps,
            frame_indices,
        )

        setup = discover_match_setup(
            accumulated_tracks, accumulated_calibration, self.pitch, frame_indices
        )

        outputs = self._write(
            run, metadata, rows, accumulated_tracks, accumulated_calibration,
            timestamps, frame_indices, chunk_of,
        )

        elapsed = round(time.perf_counter() - started, 2)
        # On a full resume every chunk is skipped and no model report is
        # produced. Passing the empty dict would erase the provenance the
        # original run recorded, so the field is only written when observed.
        manifest_models = {"models": models} if models else {}
        run.update_manifest(
            **manifest_models,
            stages={
                "chunking": {
                    "chunks": len(chunks),
                    "chunk_frames": self.config.chunking.chunk_frames,
                    "overlap_frames": self.config.chunking.overlap_frames,
                    "merge": report.to_dict(),
                },
                "game_state": assembler.quality_report(rows),
                "match_setup": setup.to_dict(),
                "timings_s": {"total": elapsed},
            },
            data_quality={
                "frames_processed": len(frame_indices),
                "tracks": len(accumulated_tracks),
                "rows": len(rows),
            },
        )

        log.info(
            "chunked run complete: %d chunk(s), %d tracks, %d rows in %.1fs",
            len(chunks),
            len(accumulated_tracks),
            len(rows),
            elapsed,
        )
        return ChunkedResult(
            run_dir=run.root,
            metadata=metadata,
            chunks=chunks,
            merge=report,
            rows=len(rows),
            tracks=len(accumulated_tracks),
            outputs=outputs,
            timings={"total": elapsed},
            reports={"match_setup": setup.to_dict()},
        )

    # -- persistence --------------------------------------------------------- #

    def _write(
        self, run, metadata, rows, tracks, calibration, timestamps, frame_indices, chunk_of
    ) -> dict[str, str]:
        compression = self.config.storage.compression
        fmt = self.config.storage.format
        outputs: dict[str, str] = {}

        outputs["game_state"] = str(
            write_game_state(rows, run.table_path("game_state"), compression, fmt)
        )
        outputs["tracks"] = str(
            write_tracks(
                track_rows(metadata.video_id, tracks.values()),
                run.table_path("tracks"), compression, fmt,
            )
        )
        outputs["calibration"] = str(
            write_calibration(
                calibration_rows(metadata.video_id, timestamps, calibration.values()),
                run.table_path("calibration"), compression, fmt,
            )
        )
        outputs["frames"] = str(
            write_frames(
                frame_rows(
                    metadata.video_id, frame_indices, timestamps, rows, calibration, chunk_of
                ),
                run.table_path("frames"), compression, fmt,
            )
        )
        return outputs

    @staticmethod
    def _load_state(path: Path) -> set[int]:
        if not path.exists():
            return set()
        try:
            return set(json.loads(path.read_text(encoding="utf-8"))["completed_chunks"])
        except (json.JSONDecodeError, KeyError) as exc:
            log.warning("unreadable chunk checkpoint, restarting: %s", exc)
            return set()

    @staticmethod
    def _save_state(path: Path, completed: set[int]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({"completed_chunks": sorted(completed)}, indent=2), encoding="utf-8"
        )
        tmp.replace(path)
