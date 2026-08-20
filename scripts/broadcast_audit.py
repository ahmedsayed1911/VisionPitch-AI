"""Audit a broadcast video: cuts, shot types, live-play share.

Broadcast ball annotation workflow, step 1.

Two passes. The first decodes every frame for cheap signals (histogram distance,
camera motion, blur, edge density) and cuts the video into shots. The second runs
the person detector and the pitch-keypoint model on a few frames per shot and
classifies the shot from that evidence.

Usage::

    python scripts/broadcast_audit.py --video "<path>.mp4"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionpitch.annotation.broadcast_audit import (  # noqa: E402
    AuditResult,
    Shot,
    ShotType,
    build_shots,
    detect_cuts,
    scan_frames,
)
from visionpitch.common.config import load_config  # noqa: E402
from visionpitch.common.logging import configure_logging, get_logger  # noqa: E402
from visionpitch.common.types import ObjectClass  # noqa: E402
from visionpitch.ingestion.video import probe_video  # noqa: E402

log = get_logger("broadcast.audit")

#: A person taller than this share of frame height means the camera is close.
CLOSE_UP_HEIGHT_SHARE = 0.45
MEDIUM_HEIGHT_SHARE = 0.22
#: Pitch keypoints needed before a shot is treated as having usable geometry.
MIN_PITCH_KEYPOINTS = 4


def classify(
    shot: Shot,
    n_people: list[int],
    heights: list[float],
    keypoints: list[int],
    frame_height: int,
    live_motion_median: float,
) -> None:
    """Assign a shot type from measured evidence, and say how sure it is."""
    shot.n_people_median = float(np.median(n_people)) if n_people else 0.0
    shot.person_height_median_px = float(np.median(heights)) if heights else 0.0
    shot.pitch_keypoints_median = float(np.median(keypoints)) if keypoints else 0.0

    height_share = shot.person_height_median_px / max(1, frame_height)
    has_pitch = shot.pitch_keypoints_median >= MIN_PITCH_KEYPOINTS

    if shot.n_people_median <= 0.5 and not has_pitch:
        # Nothing detected and no pitch: a graphic, a logo sting or a black frame.
        # Saturation separates a designed graphic from a dim crowd shot.
        shot.shot_type = (
            ShotType.GRAPHIC if shot.saturation_median > 90 else ShotType.CROWD_OR_BENCH
        )
        shot.classification_confidence = 0.5
        shot.notes = "no people and no pitch geometry"
    elif height_share >= CLOSE_UP_HEIGHT_SHARE:
        shot.shot_type = ShotType.CLOSE_UP
        shot.classification_confidence = 0.8
        shot.notes = f"median person spans {height_share:.0%} of frame height"
    elif not has_pitch:
        shot.shot_type = ShotType.CROWD_OR_BENCH
        shot.classification_confidence = 0.6
        shot.notes = f"only {shot.pitch_keypoints_median:.0f} pitch keypoints"
    elif height_share >= MEDIUM_HEIGHT_SHARE or shot.n_people_median < 5:
        shot.shot_type = ShotType.MEDIUM_PLAY
        shot.classification_confidence = 0.7
        shot.notes = (
            f"{shot.n_people_median:.0f} people, {height_share:.0%} frame height"
        )
    else:
        shot.shot_type = ShotType.WIDE_PLAY
        shot.classification_confidence = 0.8
        shot.notes = (
            f"{shot.n_people_median:.0f} people, {shot.pitch_keypoints_median:.0f} "
            f"pitch keypoints"
        )

    # Slow motion: a pitch shot moving far more slowly than live play does.
    # Explicitly a heuristic -- the reviewer has IGNORE_REPLAY for the truth.
    if shot.shot_type.is_live_play_candidate and live_motion_median > 0:
        shot.likely_slow_motion = shot.motion_median_px < 0.35 * live_motion_median


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--frames-per-shot", type=int, default=3)
    parser.add_argument("--min-shot-frames", type=int, default=12)
    parser.add_argument("--out", type=Path, default=Path("data/annotation"))
    args = parser.parse_args()

    configure_logging("INFO")
    video = Path(args.video)
    if not video.exists():
        log.error("no video at %s", video)
        return 1

    metadata = probe_video(video)
    log.info(
        "%dx%d @ %.3f fps, %d frames, %.1fs, codec=%s",
        metadata.width, metadata.height, metadata.fps,
        metadata.frame_count, metadata.duration_s, metadata.codec,
    )

    log.info("pass 1: scanning every frame for cuts and motion")
    features = scan_frames(video)
    cuts, threshold = detect_cuts(features, min_shot_frames=args.min_shot_frames)
    shots = build_shots(cuts, features, metadata.fps)
    log.info("found %d cut(s) -> %d shot(s), threshold %.4f",
             len(cuts), len(shots), threshold)

    # -- pass 2: classify shots with the real models --------------------------- #
    log.info("pass 2: classifying %d shot(s) with detector and pitch keypoints", len(shots))
    config = load_config()
    from visionpitch.calibration.keypoints import PitchKeypointDetector
    from visionpitch.detection.yolo import build_detector

    detector = build_detector(config)
    keypoint_detector = PitchKeypointDetector(config)

    wanted: dict[int, list[int]] = {}
    for shot in shots:
        if shot.n_frames <= 0:
            continue
        picks = np.linspace(
            shot.start_frame, shot.end_frame,
            min(args.frames_per_shot, shot.n_frames), dtype=int,
        )
        wanted[shot.index] = sorted(set(int(p) for p in picks))

    needed = {idx for picks in wanted.values() for idx in picks}
    grabbed: dict[int, np.ndarray] = {}
    capture = cv2.VideoCapture(str(video))
    frame_idx = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_idx in needed:
            grabbed[frame_idx] = frame
        frame_idx += 1
    capture.release()
    log.info("  decoded %d probe frame(s)", len(grabbed))

    per_shot: dict[int, tuple[list[int], list[float], list[int]]] = {}
    for shot in shots:
        counts: list[int] = []
        heights: list[float] = []
        keypoint_counts: list[int] = []
        for index in wanted.get(shot.index, []):
            frame = grabbed.get(index)
            if frame is None:
                continue
            detections = detector.detect_batch([frame], [index])[0]
            people = [
                d for d in detections
                if d.object_class in (
                    ObjectClass.PLAYER, ObjectClass.GOALKEEPER, ObjectClass.REFEREE
                )
            ]
            counts.append(len(people))
            if people:
                heights.append(float(np.median([d.bbox.height for d in people])))
            observation = keypoint_detector.detect_batch([frame], [index])[0]
            if observation is None:
                keypoint_counts.append(0)
            else:
                _, points, _ = observation.confident(config.calibration.min_confidence)
                keypoint_counts.append(int(len(points)))
        per_shot[shot.index] = (counts, heights, keypoint_counts)

    # Live-play motion norm, from shots that look like play, so the slow-motion
    # test compares against this broadcast rather than an assumed constant.
    provisional = []
    for shot in shots:
        counts, heights, keypoints = per_shot[shot.index]
        enough_people = bool(counts) and np.median(counts) >= 5
        enough_pitch = bool(keypoints) and np.median(keypoints) >= MIN_PITCH_KEYPOINTS
        if enough_people and enough_pitch:
            provisional.append(shot.motion_median_px)
    live_motion = float(np.median(provisional)) if provisional else 0.0
    log.info("live-play motion norm: %.2f px/frame", live_motion)

    for shot in shots:
        counts, heights, keypoints = per_shot[shot.index]
        classify(shot, counts, heights, keypoints, metadata.height, live_motion)

    result = AuditResult(
        video_path=str(video),
        content_hash=metadata.content_hash,
        width=metadata.width, height=metadata.height, fps=metadata.fps,
        frame_count=metadata.frame_count, duration_s=metadata.duration_s,
        codec=metadata.codec, shots=shots, cut_threshold=threshold,
    )
    destination = result.save(args.out / "broadcast_audit.json")

    print(f"\nvideo      : {video.name}")
    print(f"resolution : {metadata.width}x{metadata.height} @ {metadata.fps:g} fps")
    print(f"duration   : {metadata.duration_s:.1f}s, {metadata.frame_count} frames")
    print(f"codec      : {metadata.codec}")
    print(f"content    : {metadata.content_hash[:16]}")
    print(f"\ncuts       : {len(cuts)}  ->  {len(shots)} shots "
          f"(median {np.median([s.n_frames for s in shots]):.0f} frames)")
    print(f"\n{'shot type':<18}{'shots':>7}{'frames':>9}{'seconds':>10}{'share':>8}")
    for name, block in result.by_type().items():
        print(f"  {name:<16}{block['n_shots']:>7}{block['n_frames']:>9}"
              f"{block['seconds']:>10.1f}{block['share_of_video']:>8.3f}")
    print(f"\nlive-play share        : {result.live_play_share:.3f}")
    print(f"likely slow-motion     : {result.slow_motion_share:.3f}  (heuristic)")
    print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
