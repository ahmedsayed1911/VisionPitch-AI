"""Frozen row definitions for the fusion production evaluation.

Every configuration in this module was fixed **before** the four-way rows were
scored, alongside the pinned thresholds in
:mod:`visionpitch.evaluation.fusion_thresholds`. A row is a detector checkpoint
plus a fully specified ``ball_fusion`` block -- nothing else varies. Frames,
calibration, tracking, possession, events, detector confidence and every
threshold outside fusion are identical across rows by construction, because
they are simply not mentioned here.

Freezing them in code rather than in a script argument is deliberate: a
`--set` typed at a shell prompt is not a record of what was run.
"""

from __future__ import annotations

import hashlib
import json

ROW_SPEC_VERSION = "1.0.0"

#: Locked checkpoints. Neither is retrained, rebuilt or reselected here.
DETECTORS = {
    "default": {
        "weights": "models/yolo-football-ball-detection.pt",
        "fingerprint": "fb37942448e7de08",
    },
    "candidate_c": {
        "weights": "models/finetune/bcast_adapt/weights/best.pt",
        "fingerprint": "e1e373009e4a8c96",
    },
}

#: Locked detector confidence, unchanged from the broadcast comparison.
CONF_THRESHOLD = 0.12

LEGACY = {
    "engine": "legacy",
}

#: The temporal configuration under test. Its knobs are the defaults measured in
#: the isolated ablation's full stack, translated into the run config.
TEMPORAL_FULL = {
    "engine": "temporal",
    "suppression": "weighted_centre",
    "merge_radius_px": 22.0,
    "temporal_filter_enabled": True,
    "camera_motion_enabled": True,
    "camera_cut_reset_enabled": True,
    "observability_enabled": True,
    "min_support_frames": 2,
    "trust_confidence": 0.75,
    "camera_motion_px": 6.0,
    "min_camera_confidence": 0.35,
}


#: The four-way comparison. A/C are the legacy references; B/D are under test.
ROWS: dict[str, dict] = {
    "A": {
        "label": "A_default_legacy",
        "detector": "default",
        "fusion": LEGACY,
        "reference": None,
        "description": "current default detector + legacy fusion (shipped today)",
    },
    "B": {
        "label": "B_default_temporal",
        "detector": "default",
        "fusion": TEMPORAL_FULL,
        "reference": "A",
        "description": "current default detector + new temporal fusion",
    },
    "C": {
        "label": "C_adapt_legacy",
        "detector": "candidate_c",
        "fusion": LEGACY,
        "reference": None,
        "description": "Candidate C detector + legacy fusion",
    },
    "D": {
        "label": "D_adapt_temporal",
        "detector": "candidate_c",
        "fusion": TEMPORAL_FULL,
        "reference": "C",
        "description": "Candidate C detector + new temporal fusion",
    },
}


#: Part 7: the same ablation the isolated study ran, but every row now passes
#: through possession and the event engine. Rows 1 and 7 are byte-identical to
#: four-way rows C and D and are reused rather than re-run.
ABLATION: dict[str, dict] = {
    "1_legacy": {"fusion": LEGACY, "same_as_row": "C"},
    "2_centre_suppression_only": {
        "fusion": {
            **TEMPORAL_FULL,
            "suppression": "centre_distance",
            "temporal_filter_enabled": False,
            "camera_motion_enabled": False,
            "camera_cut_reset_enabled": False,
        },
    },
    "3_temporal_persistence_only": {
        "fusion": {
            **TEMPORAL_FULL,
            "suppression": "iou",
            "camera_motion_enabled": False,
            "camera_cut_reset_enabled": False,
        },
        "note": "IoU suppression reproduces legacy de-duplication, isolating "
                "persistence",
    },
    "4_real_gmc_only": {
        "fusion": {
            **TEMPORAL_FULL,
            "suppression": "iou",
            "min_support_frames": 0,
            "trust_confidence": 0.0,
        },
        "note": "camera tests active, persistence neutralised",
    },
    "5_centre_suppression_plus_persistence": {
        "fusion": {
            **TEMPORAL_FULL,
            "suppression": "centre_distance",
            "camera_motion_enabled": False,
            "camera_cut_reset_enabled": False,
        },
    },
    "6_centre_suppression_plus_real_gmc": {
        "fusion": {
            **TEMPORAL_FULL,
            "suppression": "centre_distance",
            "min_support_frames": 0,
            "trust_confidence": 0.0,
        },
    },
    "7_full_temporal_fusion": {"fusion": TEMPORAL_FULL, "same_as_row": "D"},
}


def overrides(row: dict) -> list[str]:
    """The exact ``--set`` arguments for a row. One source of truth."""
    detector = DETECTORS[row["detector"]]
    settings = [
        f"ball_detection.model_path={detector['weights']}",
        f"ball_detection.conf_threshold={CONF_THRESHOLD}",
    ]
    settings += [f"ball_fusion.{k}={v}" for k, v in row["fusion"].items()]
    return settings


def fingerprint() -> str:
    return hashlib.sha256(
        json.dumps(
            {"version": ROW_SPEC_VERSION, "detectors": DETECTORS,
             "conf": CONF_THRESHOLD, "rows": ROWS, "ablation": ABLATION},
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
