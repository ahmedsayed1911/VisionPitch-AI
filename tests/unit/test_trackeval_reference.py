"""Cross-check the in-repo tracking metrics against official TrackEval."""

from __future__ import annotations

import importlib.util
import sys
import types

import numpy as np
import pytest

from visionpitch.common.geometry import iou_matrix
from visionpitch.evaluation.tracking import FrameData, compute_clear, compute_hota, compute_idf1

if importlib.util.find_spec("trackeval") is None:
    pytest.skip("TrackEval is not installed", allow_module_level=True)
# TrackEval imports every optional dataset adapter at package import time.  Its
# BURST adapter pulls in pycocotools even though these metric-only tests do not
# need any dataset adapter; stub that namespace to keep the reference check
# independent of optional binary extensions.
sys.modules.setdefault("trackeval.datasets", types.ModuleType("trackeval.datasets"))
trackeval = pytest.importorskip("trackeval")


def _box(slot: int) -> np.ndarray:
    x = float(slot * 100)
    return np.array([x, 0.0, x + 40.0, 80.0])


def _frame(entries: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([track_id for track_id, _ in entries], dtype=int),
        np.array([_box(slot) for _, slot in entries], dtype=float).reshape(-1, 4),
    )


SCENARIOS = {
    "perfect": (
        [[(10, 0), (20, 1)], [(10, 0), (20, 1)], [(10, 0), (20, 1)]],
        [[(100, 0), (200, 1)], [(100, 0), (200, 1)], [(100, 0), (200, 1)]],
    ),
    "miss": (
        [[(10, 0), (20, 1)], [(10, 0), (20, 1)], [(10, 0), (20, 1)]],
        [[(100, 0), (200, 1)], [(100, 0)], [(100, 0), (200, 1)]],
    ),
    "false_positive": (
        [[(10, 0)], [(10, 0)], [(10, 0)]],
        [[(100, 0)], [(100, 0), (999, 2)], [(100, 0)]],
    ),
    "identity_switch": (
        [[(10, 0)], [(10, 0)], [(10, 0)]],
        [[(100, 0)], [(100, 0)], [(200, 0)]],
    ),
    "fragment": (
        [[(10, 0)], [(10, 0)], [(10, 0)], [(10, 0)]],
        [[(100, 0)], [], [(100, 0)], [(100, 0)]],
    ),
}


def _trackeval_data(frames: list[int], gt: FrameData, pred: FrameData) -> dict:
    gt_values = sorted({int(value) for frame in frames for value in gt[frame][0]})
    pred_values = sorted({int(value) for frame in frames for value in pred[frame][0]})
    gt_index = {value: index for index, value in enumerate(gt_values)}
    pred_index = {value: index for index, value in enumerate(pred_values)}

    gt_ids = [
        np.array([gt_index[int(value)] for value in gt[frame][0]], dtype=int) for frame in frames
    ]
    tracker_ids = [
        np.array([pred_index[int(value)] for value in pred[frame][0]], dtype=int)
        for frame in frames
    ]
    return {
        "num_timesteps": len(frames),
        "num_gt_ids": len(gt_values),
        "num_tracker_ids": len(pred_values),
        "num_gt_dets": sum(len(values) for values in gt_ids),
        "num_tracker_dets": sum(len(values) for values in tracker_ids),
        "gt_ids": gt_ids,
        "tracker_ids": tracker_ids,
        "similarity_scores": [iou_matrix(gt[frame][1], pred[frame][1]) for frame in frames],
    }


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_metrics_match_trackeval_reference(scenario: str) -> None:
    gt_rows, pred_rows = SCENARIOS[scenario]
    frames = list(range(len(gt_rows)))
    gt = {frame: _frame(rows) for frame, rows in enumerate(gt_rows)}
    pred = {frame: _frame(rows) for frame, rows in enumerate(pred_rows)}
    data = _trackeval_data(frames, gt, pred)

    ours_hota = compute_hota(frames, gt, pred)
    ours_identity = compute_idf1(frames, gt, pred)
    ours_clear = compute_clear(frames, gt, pred)
    ref_hota = trackeval.metrics.HOTA().eval_sequence(data)
    ref_identity = trackeval.metrics.Identity({"PRINT_CONFIG": False}).eval_sequence(data)
    ref_clear = trackeval.metrics.CLEAR({"PRINT_CONFIG": False}).eval_sequence(data)

    assert ours_hota["HOTA"] == round(float(np.mean(ref_hota["HOTA"])), 4)
    assert ours_hota["DetA"] == round(float(np.mean(ref_hota["DetA"])), 4)
    assert ours_hota["AssA"] == round(float(np.mean(ref_hota["AssA"])), 4)
    assert ours_hota["HOTA_at_alpha_0.5"] == round(float(ref_hota["HOTA"][9]), 4)
    for key in ("IDF1", "IDP", "IDR"):
        assert ours_identity[key] == round(float(ref_identity[key]), 4)
    for key in ("IDTP", "IDFP", "IDFN"):
        assert ours_identity[key] == int(ref_identity[key])
    assert ours_clear["MOTA"] == round(float(ref_clear["MOTA"]), 4)
    assert ours_clear["id_switches"] == int(ref_clear["IDSW"])
