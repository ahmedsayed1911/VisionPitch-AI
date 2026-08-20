"""Tracking metrics: HOTA, IDF1, CLEAR (MOTA), ID switches, fragmentation.

Implemented here rather than pulled from TrackEval so the metrics run against
this project's own data structures with no format conversion step, and so their
definitions are auditable in-repo.

The three families measure genuinely different things, which is why all three
are reported:

* **MOTA** is detection-dominated. A tracker can score well on MOTA while
  switching identities constantly, because a switch costs one frame.
* **IDF1** is identity-dominated. It asks whether a real player maps to one
  predicted track over their whole life.
* **HOTA** balances detection and association explicitly, and is the metric to
  quote for football, where both matter and neither should mask the other.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from visionpitch.common.geometry import iou_matrix
from visionpitch.common.logging import get_logger
from visionpitch.common.types import ObjectClass, Track
from visionpitch.evaluation.ground_truth import GroundTruth

log = get_logger("evaluation.tracking")


#: per-frame ``{frame_idx: (ids, boxes)}``
FrameData = dict[int, tuple[np.ndarray, np.ndarray]]


def _build_frame_data(
    ground_truth: GroundTruth,
    tracks: dict[int, Track],
    classes: tuple[ObjectClass, ...],
) -> tuple[list[int], FrameData, FrameData]:
    """Extract per-frame ``(ids, boxes)`` for ground truth and predictions."""
    wanted = set(classes)
    frames = sorted(ground_truth.annotated_frames)

    gt_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for frame_idx in frames:
        objs = [o for o in ground_truth.frames[frame_idx] if o.object_class in wanted]
        gt_data[frame_idx] = (
            np.array([o.track_id for o in objs], dtype=int),
            np.array([o.bbox.to_array() for o in objs]) if objs else np.zeros((0, 4)),
        )

    per_frame: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for track in tracks.values():
        if track.object_class not in wanted:
            continue
        for obs in track.observations:
            if obs.interpolated:
                # Interpolated boxes are the tracker's guesses, not observations.
                # Including them would let a tracker inflate recall by coasting
                # through occlusions. They are excluded, and that exclusion is
                # part of the metric's definition here.
                continue
            if obs.frame_idx in gt_data:
                per_frame[obs.frame_idx].append((track.track_id, obs.bbox.to_array()))

    pred_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for frame_idx in frames:
        entries = per_frame.get(frame_idx, [])
        pred_data[frame_idx] = (
            np.array([e[0] for e in entries], dtype=int),
            np.array([e[1] for e in entries]) if entries else np.zeros((0, 4)),
        )
    return frames, gt_data, pred_data


# --------------------------------------------------------------------------- #
# HOTA
# --------------------------------------------------------------------------- #


def compute_hota(
    frames: list[int],
    gt_data: dict[int, tuple[np.ndarray, np.ndarray]],
    pred_data: dict[int, tuple[np.ndarray, np.ndarray]],
    alphas: list[float] | None = None,
) -> dict:
    """HOTA and its DetA / AssA components, averaged over the alpha sweep.

    Follows the reference formulation: matching at each alpha is done on
    similarity weighted by a *global* alignment score, so a frame-local match
    that would break a long-lived correspondence is penalised.
    """
    alphas = alphas or [round(0.05 + 0.05 * i, 2) for i in range(19)]

    gt_ids = sorted({int(i) for f in frames for i in gt_data[f][0]})
    pred_ids = sorted({int(i) for f in frames for i in pred_data[f][0]})
    if not gt_ids:
        return {"HOTA": None, "DetA": None, "AssA": None, "note": "no ground-truth tracks"}
    if not pred_ids:
        return {"HOTA": 0.0, "DetA": 0.0, "AssA": 0.0, "note": "no predicted tracks"}

    gt_index = {g: i for i, g in enumerate(gt_ids)}
    pred_index = {p: i for i, p in enumerate(pred_ids)}
    n_gt, n_pred = len(gt_ids), len(pred_ids)

    # Global co-occurrence: how much each (gt, pred) pair overlaps across time.
    potential = np.zeros((n_gt, n_pred))
    gt_count = np.zeros(n_gt)
    pred_count = np.zeros(n_pred)

    similarities: dict[int, np.ndarray] = {}
    for frame_idx in frames:
        g_ids, g_boxes = gt_data[frame_idx]
        p_ids, p_boxes = pred_data[frame_idx]
        sim = iou_matrix(g_boxes, p_boxes)
        similarities[frame_idx] = sim
        for g in g_ids:
            gt_count[gt_index[int(g)]] += 1
        for p in p_ids:
            pred_count[pred_index[int(p)]] += 1
        for gi, g in enumerate(g_ids):
            for pi, p in enumerate(p_ids):
                potential[gt_index[int(g)], pred_index[int(p)]] += sim[gi, pi]

    denominator = gt_count[:, None] + pred_count[None, :] - potential
    global_alignment = np.divide(
        potential, denominator, out=np.zeros_like(potential), where=denominator > 0
    )

    hotas, detas, assas = [], [], []
    for alpha in alphas:
        tp = fp = fn = 0
        matches = np.zeros((n_gt, n_pred))

        for frame_idx in frames:
            g_ids, _ = gt_data[frame_idx]
            p_ids, _ = pred_data[frame_idx]
            sim = similarities[frame_idx]

            if g_ids.size == 0 or p_ids.size == 0:
                fn += int(g_ids.size)
                fp += int(p_ids.size)
                continue

            rows = [gt_index[int(g)] for g in g_ids]
            cols = [pred_index[int(p)] for p in p_ids]
            score = global_alignment[np.ix_(rows, cols)] * sim
            # Tiny epsilon keeps the assignment deterministic when scores tie.
            r_idx, c_idx = linear_sum_assignment(-(score + 1e-12))

            n_matched = 0
            for r, c in zip(r_idx, c_idx, strict=True):
                if sim[r, c] >= alpha:
                    matches[rows[r], cols[c]] += 1
                    n_matched += 1
            tp += n_matched
            fn += int(g_ids.size) - n_matched
            fp += int(p_ids.size) - n_matched

        det_a = tp / max(1e-9, tp + fn + fp)
        if tp > 0:
            ass_denominator = gt_count[:, None] + pred_count[None, :] - matches
            ass_iou = np.divide(
                matches, ass_denominator, out=np.zeros_like(matches), where=ass_denominator > 0
            )
            ass_a = float((matches * ass_iou).sum() / tp)
        else:
            ass_a = 0.0

        detas.append(det_a)
        assas.append(ass_a)
        hotas.append(float(np.sqrt(det_a * ass_a)))

    return {
        "HOTA": round(float(np.mean(hotas)), 4),
        "DetA": round(float(np.mean(detas)), 4),
        "AssA": round(float(np.mean(assas)), 4),
        "HOTA_at_alpha_0.5": round(float(hotas[alphas.index(0.5)]), 4)
        if 0.5 in alphas
        else None,
        "alphas": alphas,
    }


# --------------------------------------------------------------------------- #
# IDF1
# --------------------------------------------------------------------------- #


def compute_idf1(
    frames: list[int],
    gt_data: dict[int, tuple[np.ndarray, np.ndarray]],
    pred_data: dict[int, tuple[np.ndarray, np.ndarray]],
    iou_threshold: float = 0.5,
) -> dict:
    """Identity F1: one global one-to-one assignment between GT and pred ids."""
    gt_ids = sorted({int(i) for f in frames for i in gt_data[f][0]})
    pred_ids = sorted({int(i) for f in frames for i in pred_data[f][0]})
    n_gt_dets = sum(int(gt_data[f][0].size) for f in frames)
    n_pred_dets = sum(int(pred_data[f][0].size) for f in frames)

    if not gt_ids or not pred_ids:
        return {
            "IDF1": 0.0 if gt_ids else None,
            "IDP": 0.0,
            "IDR": 0.0,
            "IDTP": 0,
            "IDFP": n_pred_dets,
            "IDFN": n_gt_dets,
        }

    gt_index = {g: i for i, g in enumerate(gt_ids)}
    pred_index = {p: i for i, p in enumerate(pred_ids)}
    overlap = np.zeros((len(gt_ids), len(pred_ids)))

    for frame_idx in frames:
        g_ids, g_boxes = gt_data[frame_idx]
        p_ids, p_boxes = pred_data[frame_idx]
        if g_ids.size == 0 or p_ids.size == 0:
            continue
        ious = iou_matrix(g_boxes, p_boxes)
        for gi, g in enumerate(g_ids):
            for pi, p in enumerate(p_ids):
                if ious[gi, pi] >= iou_threshold:
                    overlap[gt_index[int(g)], pred_index[int(p)]] += 1

    r_idx, c_idx = linear_sum_assignment(-overlap)
    idtp = int(sum(overlap[r, c] for r, c in zip(r_idx, c_idx, strict=True)))
    idfn = n_gt_dets - idtp
    idfp = n_pred_dets - idtp

    idp = idtp / max(1e-9, idtp + idfp)
    idr = idtp / max(1e-9, idtp + idfn)
    idf1 = 2 * idtp / max(1e-9, 2 * idtp + idfp + idfn)

    return {
        "IDF1": round(float(idf1), 4),
        "IDP": round(float(idp), 4),
        "IDR": round(float(idr), 4),
        "IDTP": idtp,
        "IDFP": idfp,
        "IDFN": idfn,
    }


# --------------------------------------------------------------------------- #
# CLEAR MOT
# --------------------------------------------------------------------------- #


def compute_clear(
    frames: list[int],
    gt_data: dict[int, tuple[np.ndarray, np.ndarray]],
    pred_data: dict[int, tuple[np.ndarray, np.ndarray]],
    iou_threshold: float = 0.5,
) -> dict:
    """MOTA, MOTP, ID switches and fragmentation.

    Matching preserves the previous frame's correspondence when it is still
    valid, which is the CLEAR convention and the reason ID switches are counted
    at all: without that stickiness, a fresh optimal assignment each frame would
    report switches that the tracker never made.
    """
    previous: dict[int, int] = {}
    total_gt = 0
    fp = fn = switches = 0
    distances: list[float] = []

    # Per-GT-track history for fragmentation.
    tracked_flags: dict[int, list[tuple[int, bool]]] = defaultdict(list)

    for frame_idx in frames:
        g_ids, g_boxes = gt_data[frame_idx]
        p_ids, p_boxes = pred_data[frame_idx]
        total_gt += int(g_ids.size)

        matched: dict[int, int] = {}
        if g_ids.size and p_ids.size:
            ious = iou_matrix(g_boxes, p_boxes)
            pred_pos = {int(p): i for i, p in enumerate(p_ids)}

            used_pred: set[int] = set()
            used_gt: set[int] = set()

            # Preserve existing correspondences first.
            for gi, g in enumerate(g_ids):
                prev_pred = previous.get(int(g))
                if prev_pred is None or prev_pred not in pred_pos:
                    continue
                pi = pred_pos[prev_pred]
                if ious[gi, pi] >= iou_threshold:
                    matched[int(g)] = prev_pred
                    used_gt.add(gi)
                    used_pred.add(pi)
                    distances.append(float(ious[gi, pi]))

            free_gt = [i for i in range(g_ids.size) if i not in used_gt]
            free_pred = [i for i in range(p_ids.size) if i not in used_pred]
            if free_gt and free_pred:
                sub = ious[np.ix_(free_gt, free_pred)]
                r_idx, c_idx = linear_sum_assignment(-sub)
                for r, c in zip(r_idx, c_idx, strict=True):
                    if sub[r, c] < iou_threshold:
                        continue
                    gi, pi = free_gt[r], free_pred[c]
                    g, p = int(g_ids[gi]), int(p_ids[pi])
                    if g in previous and previous[g] != p:
                        switches += 1
                    matched[g] = p
                    distances.append(float(sub[r, c]))

        fn += int(g_ids.size) - len(matched)
        fp += int(p_ids.size) - len(matched)

        for g in g_ids:
            tracked_flags[int(g)].append((frame_idx, int(g) in matched))
        previous.update(matched)
        # Forget correspondences for GT tracks absent this frame, so a track
        # reappearing after a long gap does not register a spurious switch.
        for g in list(previous):
            if g not in set(int(i) for i in g_ids):
                previous.pop(g, None)

    fragments = 0
    for history in tracked_flags.values():
        was_tracked = False
        seen_tracked = False
        for _, is_tracked in history:
            if is_tracked and not was_tracked and seen_tracked:
                fragments += 1
            if is_tracked:
                seen_tracked = True
            was_tracked = is_tracked

    mota = 1.0 - (fn + fp + switches) / max(1, total_gt)
    motp = float(np.mean(distances)) if distances else 0.0

    # Mostly-tracked / mostly-lost, the standard coverage buckets.
    mt = ml = pt = 0
    for history in tracked_flags.values():
        ratio = sum(1 for _, t in history if t) / max(1, len(history))
        if ratio >= 0.8:
            mt += 1
        elif ratio <= 0.2:
            ml += 1
        else:
            pt += 1

    return {
        "MOTA": round(float(mota), 4),
        "MOTP_iou": round(motp, 4),
        "id_switches": switches,
        "fragmentations": fragments,
        "false_positives": fp,
        "false_negatives": fn,
        "ground_truth_detections": total_gt,
        "mostly_tracked": mt,
        "partially_tracked": pt,
        "mostly_lost": ml,
    }


# --------------------------------------------------------------------------- #


def evaluate_tracking(
    ground_truth: GroundTruth,
    tracks: dict[int, Track],
    iou_threshold: float = 0.5,
    hota_alphas: list[float] | None = None,
    classes: tuple[ObjectClass, ...] = (
        ObjectClass.PLAYER,
        ObjectClass.GOALKEEPER,
        ObjectClass.REFEREE,
    ),
) -> dict:
    """Full tracking evaluation on the annotated frames."""
    frames, gt_data, pred_data = _build_frame_data(ground_truth, tracks, classes)
    if not frames:
        raise ValueError("ground truth contains no annotated frames")

    result = {
        "classes_evaluated": [c.value for c in classes],
        "annotated_frames": len(frames),
        "n_ground_truth_tracks": len({int(i) for f in frames for i in gt_data[f][0]}),
        "n_predicted_tracks": len({int(i) for f in frames for i in pred_data[f][0]}),
        "iou_threshold": iou_threshold,
        **compute_hota(frames, gt_data, pred_data, hota_alphas),
        **compute_idf1(frames, gt_data, pred_data, iou_threshold),
        **compute_clear(frames, gt_data, pred_data, iou_threshold),
    }
    log.info(
        "tracking: HOTA=%s IDF1=%s MOTA=%s switches=%s",
        result.get("HOTA"),
        result.get("IDF1"),
        result.get("MOTA"),
        result.get("id_switches"),
    )
    return result
