"""Interactive ground-truth annotation tool.

Produces the JSON consumed by ``visionpitch evaluate``. Two modes:

``boxes``
    Draw player / goalkeeper / referee / ball boxes and assign track ids.
    Track ids must be *consistent across frames* -- that consistency is exactly
    what IDF1 and HOTA measure, so an annotator who renumbers players each frame
    produces ground truth that scores every tracker at zero.

``pitch``
    Click pitch landmarks and name them by index. This is the only way to detect
    a systematic homography bias: a homography fitted to the model's own
    keypoints always reproduces those keypoints well, whether or not it is right.

Usage::

    python scripts/annotate.py boxes data/raw/clip.mp4 --frames 100,300,500
    python scripts/annotate.py pitch data/raw/clip.mp4 --frames 100,300,500

Keys (boxes): 1 player  2 goalkeeper  3 referee  4 ball
              drag to draw, u undo, n next frame, p previous, s save, q quit
Keys (pitch): click a point then type the landmark index and press Enter
              u undo, n next, s save, q quit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionpitch.common.types import BBox, ObjectClass  # noqa: E402
from visionpitch.evaluation.ground_truth import (  # noqa: E402
    GroundTruth,
    GTObject,
    load_ground_truth,
    save_ground_truth,
)
from visionpitch.ingestion.video import probe_video  # noqa: E402
from visionpitch.pitch.geometry import PitchConfiguration  # noqa: E402

CLASS_KEYS = {
    ord("1"): ObjectClass.PLAYER,
    ord("2"): ObjectClass.GOALKEEPER,
    ord("3"): ObjectClass.REFEREE,
    ord("4"): ObjectClass.BALL,
}
CLASS_COLOURS = {
    ObjectClass.PLAYER: (80, 220, 80),
    ObjectClass.GOALKEEPER: (255, 200, 60),
    ObjectClass.REFEREE: (60, 60, 240),
    ObjectClass.BALL: (255, 255, 255),
}


def read_frames(video: Path, frame_indices: list[int]) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    out: dict[int, np.ndarray] = {}
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, image = cap.read()
        if ok:
            out[idx] = image
        else:
            print(f"warning: could not read frame {idx}")
    cap.release()
    return out


def annotate_boxes(video: Path, frame_indices: list[int], out_path: Path) -> None:
    metadata = probe_video(video)
    frames = read_frames(video, frame_indices)
    gt = load_ground_truth(out_path) if out_path.exists() else GroundTruth(
        video_id=metadata.video_id, fps=metadata.fps
    )
    gt.annotator = gt.annotator or "manual"

    state = {"class": ObjectClass.PLAYER, "drawing": False, "start": (0, 0), "cur": (0, 0)}
    order = sorted(frames)
    position = 0

    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True
            state["start"] = (x, y)
            state["cur"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            state["cur"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["drawing"]:
            state["drawing"] = False
            x1, y1 = state["start"]
            x2, y2 = x, y
            box = BBox(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            if box.width < 3 or box.height < 3:
                return
            frame_idx = order[position]
            existing = gt.frames.setdefault(frame_idx, [])
            prompt = f"track id for this {state['class'].value} (enter in console): "
            print(prompt, end="", flush=True)
            try:
                track_id = int(input().strip())
            except ValueError:
                print("  not a number - discarded")
                return
            existing.append(GTObject(state["class"], track_id, box))
        _ = flags

    cv2.namedWindow("annotate", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("annotate", on_mouse)

    while True:
        frame_idx = order[position]
        canvas = frames[frame_idx].copy()
        for obj in gt.frames.get(frame_idx, []):
            colour = CLASS_COLOURS[obj.object_class]
            x1, y1, x2, y2 = (int(v) for v in obj.bbox.to_xyxy())
            cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(canvas, str(obj.track_id), (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        if state["drawing"]:
            cv2.rectangle(canvas, state["start"], state["cur"],
                          CLASS_COLOURS[state["class"]], 1)

        header = (f"frame {frame_idx} ({position + 1}/{len(order)})  "
                  f"class={state['class'].value}  objects={len(gt.frames.get(frame_idx, []))}")
        cv2.putText(canvas, header, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("annotate", canvas)

        key = cv2.waitKey(20) & 0xFF
        if key in CLASS_KEYS:
            state["class"] = CLASS_KEYS[key]
        elif key == ord("u"):
            if gt.frames.get(frame_idx):
                gt.frames[frame_idx].pop()
        elif key == ord("n"):
            position = min(len(order) - 1, position + 1)
        elif key == ord("p"):
            position = max(0, position - 1)
        elif key == ord("s"):
            save_ground_truth(gt, out_path)
            print(f"saved -> {out_path}")
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()
    save_ground_truth(gt, out_path)
    print(f"saved -> {out_path}")
    print(gt.summary())


def annotate_pitch(video: Path, frame_indices: list[int], out_path: Path) -> None:
    metadata = probe_video(video)
    frames = read_frames(video, frame_indices)
    pitch = PitchConfiguration()
    gt = load_ground_truth(out_path) if out_path.exists() else GroundTruth(
        video_id=metadata.video_id, fps=metadata.fps
    )

    print("\nLandmark indices (see PitchConfiguration.vertices):")
    for i, (x, y) in enumerate(pitch.vertices):
        print(f"  {i:2d}: ({x:6.2f}, {y:5.2f}) m")

    order = sorted(frames)
    position = 0
    clicks: dict[int, list[tuple[np.ndarray, int]]] = {
        f: [(p, int(i)) for p, i in zip(*gt.calibration[f], strict=True)]
        if f in gt.calibration else []
        for f in order
    }
    pending = {"point": None}

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pending["point"] = (x, y)
            print(f"clicked ({x}, {y}) - landmark index: ", end="", flush=True)
            try:
                index = int(input().strip())
            except ValueError:
                print("  not a number - discarded")
                pending["point"] = None
                return
            if not 0 <= index < pitch.n_vertices:
                print(f"  index must be 0-{pitch.n_vertices - 1} - discarded")
                pending["point"] = None
                return
            clicks[order[position]].append((np.array([x, y], dtype=float), index))
            pending["point"] = None

    cv2.namedWindow("pitch", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("pitch", on_mouse)

    while True:
        frame_idx = order[position]
        canvas = frames[frame_idx].copy()
        for point, index in clicks[frame_idx]:
            p = (int(point[0]), int(point[1]))
            cv2.drawMarker(canvas, p, (60, 220, 255), cv2.MARKER_CROSS, 16, 2)
            cv2.putText(canvas, str(index), (p[0] + 8, p[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 220, 255), 2, cv2.LINE_AA)

        header = (f"frame {frame_idx} ({position + 1}/{len(order)})  "
                  f"landmarks={len(clicks[frame_idx])}")
        cv2.putText(canvas, header, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("pitch", canvas)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("u"):
            if clicks[frame_idx]:
                clicks[frame_idx].pop()
        elif key == ord("n"):
            position = min(len(order) - 1, position + 1)
        elif key == ord("p"):
            position = max(0, position - 1)
        elif key in (ord("s"), ord("q")):
            for f, entries in clicks.items():
                if len(entries) >= 4:
                    gt.calibration[f] = (
                        np.array([p for p, _ in entries]),
                        np.array([i for _, i in entries]),
                    )
            save_ground_truth(gt, out_path)
            print(f"saved -> {out_path}")
            if key == ord("q"):
                break

    cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["boxes", "pitch"])
    parser.add_argument("video", type=Path)
    parser.add_argument("--frames", required=True, help="comma-separated frame indices")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    frame_indices = [int(v) for v in args.frames.split(",") if v.strip()]
    out = args.out or Path("data/annotations") / f"{args.video.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "boxes":
        annotate_boxes(args.video, frame_indices, out)
    else:
        annotate_pitch(args.video, frame_indices, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
