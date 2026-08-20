"""Render the showcase overlay from a completed run.

Rendering is separated from analysis here because the two have wildly different
costs. Pass 1 on the reference broadcast takes ~29 minutes; compositing an
overlay over the same footage takes a few. Reading the stored tables instead of
re-running detection means the visual design can be iterated on real
full-length data, and a style change never risks re-deriving the geometry it is
drawing.

Everything this module needs is already in the Phase 1 contract:

``game_state.parquet``  feet position, team, role, per-frame, per-track
``calibration.parquet``  the image->pitch homography and its confidence

Audio is copied from the source with ffmpeg after the video is written; OpenCV's
writer has no audio path at all, and a silent showcase video is a regression a
viewer notices immediately.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from visionpitch.common.config import Config
from visionpitch.common.logging import get_logger, progress_bar
from visionpitch.common.types import Role, SegmentKind
from visionpitch.ingestion.video import VideoReader, probe_video
from visionpitch.pitch.geometry import PitchConfiguration
from visionpitch.storage.tables import homography_from_row
from visionpitch.visualization.showcase import (
    ShowcasePlayer,
    ShowcaseRenderer,
)
from visionpitch.visualization.writer import VideoWriter

log = get_logger("pipeline.showcase")


@dataclass
class ShowcaseRenderResult:
    path: Path
    n_frames: int
    seconds: float
    fps: float
    audio: bool
    stats: dict

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "n_frames": self.n_frames,
            "render_seconds": round(self.seconds, 2),
            "render_fps": round(self.fps, 2),
            "audio": self.audio,
            **self.stats,
        }


# --------------------------------------------------------------------------- #
# Table loading
# --------------------------------------------------------------------------- #


def load_frame_players(
    game_state_path: Path,
) -> dict[int, list[ShowcasePlayer]]:
    """Per-frame people from the game-state table.

    Ball rows are skipped rather than returned: the showcase draws nothing on the
    ball, so handing one to the renderer would only invite it back on screen.
    Ball tracking itself is untouched and still runs in pass 1.
    """
    table = pq.read_table(
        game_state_path,
        columns=[
            "frame_idx",
            "track_id",
            "team_id",
            "role",
            "image_x",
            "image_y",
            "pitch_x",
            "pitch_y",
        ],
    )
    data = table.to_pydict()

    players: dict[int, list[ShowcasePlayer]] = {}

    for i in range(table.num_rows):
        frame_idx = int(data["frame_idx"][i])
        role = data["role"][i]
        if role == "ball":
            continue
        px, py = data["pitch_x"][i], data["pitch_y"][i]
        players.setdefault(frame_idx, []).append(
            ShowcasePlayer(
                track_id=int(data["track_id"][i] or -1),
                team_id=data["team_id"][i],
                role=role,
                image_xy=(float(data["image_x"][i]), float(data["image_y"][i])),
                pitch_xy=(
                    (float(px), float(py)) if px is not None and py is not None else None
                ),
            )
        )
    return players


def load_calibration(
    calibration_path: Path,
) -> dict[int, tuple[np.ndarray | None, float, str, int]]:
    """Per-frame ``(homography, confidence, segment_kind, n_inliers)``.

    ``n_inliers`` is carried because it, not confidence, is what says whether the
    homography was solved from *this* frame or carried into it.
    """
    table = pq.read_table(
        calibration_path,
        columns=["frame_idx", "homography", "confidence", "segment_kind", "n_inliers"],
    )
    data = table.to_pydict()
    out: dict[int, tuple[np.ndarray | None, float, str, int]] = {}
    for i in range(table.num_rows):
        out[int(data["frame_idx"][i])] = (
            homography_from_row(data["homography"][i]),
            float(data["confidence"][i]),
            data["segment_kind"][i],
            int(data["n_inliers"][i]),
        )
    return out


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #


def has_audio_stream(video: Path) -> bool:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return False
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return "audio" in result.stdout


def mux_audio(
    silent_video: Path, source_video: Path, output: Path, start_s: float = 0.0
) -> bool:
    """Copy the source's audio onto the rendered video. Returns success.

    The video stream is re-encoded rather than copied: OpenCV writes mp4v, which
    plays badly in browsers and on phones, and this is the file people actually
    watch. ``-shortest`` guards the case where a partial render is shorter than
    the source audio.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        log.warning("ffmpeg not found; showcase video will have no audio")
        return False
    if not has_audio_stream(source_video):
        log.warning("source video has no audio stream; nothing to mux")
        return False

    command = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-i",
        str(silent_video),
    ]
    if start_s > 0:
        command += ["-ss", f"{start_s:.3f}"]
    command += [
        "-i",
        str(source_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        log.error("ffmpeg mux failed: %s", result.stderr.strip()[:500])
        return False
    return True


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #


def render_showcase(
    run_dir: Path,
    video_path: Path,
    config: Config,
    output_path: Path | None = None,
    pitch: PitchConfiguration | None = None,
) -> ShowcaseRenderResult:
    """Composite the showcase overlay over ``video_path`` using ``run_dir``'s tables."""
    run_dir = Path(run_dir)
    cfg = config.visualization.showcase
    pitch = pitch or PitchConfiguration(
        length=config.pitch.length_m, width=config.pitch.width_m
    )

    players_by_frame = load_frame_players(run_dir / "game_state.parquet")
    calibration = load_calibration(run_dir / "calibration.parquet")

    metadata = probe_video(video_path)
    reader = VideoReader(metadata, config.ingestion)
    renderer = ShowcaseRenderer(cfg, pitch)

    fps = config.visualization.output_fps or (
        metadata.fps / max(1, config.ingestion.frame_stride)
    )
    output_path = Path(output_path or run_dir / "video" / "showcase.mp4")
    silent_path = output_path.with_name(output_path.stem + "_silent.mp4")

    stats = {
        "frames_with_players": 0,
        "frames_abstained": 0,
        "team_dots": 0,
        "team_edges": 0,
        "referee_edges": 0,
        "cross_team_edges": 0,
        "referee_dots": 0,
    }
    edge_lengths_m: list[float] = []
    start = time.perf_counter()

    writer = VideoWriter(silent_path, fps, (metadata.width, metadata.height))
    with reader, progress_bar() as progress:
        task = progress.add_task("showcase render", total=reader.expected_frames)
        for frame in reader:
            _, confidence, _, n_inliers = calibration.get(
                frame.idx, (None, 0.0, SegmentKind.UNKNOWN.value, 0)
            )
            players = players_by_frame.get(frame.idx, [])
            canvas = renderer.render(frame.image, players, confidence, n_inliers)

            # Counted from what the renderer painted, not from what was eligible:
            # a frame below the minimum-support threshold draws nothing at all.
            if renderer.last_team_dots or renderer.last_referee_dots:
                stats["frames_with_players"] += 1
            stats["team_dots"] += renderer.last_team_dots
            stats["referee_dots"] += renderer.last_referee_dots
            stats["frames_abstained"] += bool(players) and not (
                renderer.last_team_dots or renderer.last_referee_dots
            )
            # Metric lengths only mean something when the frame's graph was
            # actually gated in metres. On uncalibrated frames the gate is the
            # image-space one, and recording pitch distances there reports
            # 60-metre "edges" that no metric rule ever admitted.
            calibrated = (
                confidence >= cfg.graph_min_calibration_confidence
                and n_inliers >= cfg.graph_min_inliers
            )

            # The graph invariants, measured on every frame actually rendered
            # rather than only in a unit test.
            for team_id, pa, pb in renderer.last_edges:
                stats["team_edges"] += 1
                if Role.REFEREE.value in (pa.role, pb.role):
                    stats["referee_edges"] += 1
                if pa.team_id != pb.team_id or pa.team_id != team_id:
                    stats["cross_team_edges"] += 1
                if calibrated and pa.pitch_xy is not None and pb.pitch_xy is not None:
                    edge_lengths_m.append(
                        float(np.hypot(pa.pitch_xy[0] - pb.pitch_xy[0],
                                       pa.pitch_xy[1] - pb.pitch_xy[1]))
                    )
            writer.write(canvas)
            progress.update(task, advance=1)
    writer.close()

    elapsed = time.perf_counter() - start
    n_frames = writer.n_frames
    if edge_lengths_m:
        lengths = np.asarray(edge_lengths_m)
        stats["edge_length_m_median"] = round(float(np.median(lengths)), 2)
        stats["edge_length_m_p90"] = round(float(np.percentile(lengths, 90)), 2)
        stats["edge_length_m_max"] = round(float(lengths.max()), 2)
    stats["edges_per_frame"] = round(stats["team_edges"] / max(1, n_frames), 2)
    stats["edge_length_m_samples"] = len(edge_lengths_m)

    start_s = config.ingestion.start_time_s or 0.0
    audio = mux_audio(silent_path, video_path, output_path, start_s=start_s)
    if audio:
        silent_path.unlink(missing_ok=True)
    else:
        # Better a playable silent file at the promised path than nothing.
        shutil.move(str(silent_path), str(output_path))

    log.info(
        "showcase render: %d frames in %.1fs (%.1f fps) -> %s",
        n_frames,
        elapsed,
        n_frames / elapsed if elapsed else 0.0,
        output_path.name,
    )
    return ShowcaseRenderResult(
        path=output_path,
        n_frames=n_frames,
        seconds=elapsed,
        fps=n_frames / elapsed if elapsed else 0.0,
        audio=audio,
        stats=stats,
    )
