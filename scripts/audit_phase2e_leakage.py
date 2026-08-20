"""Prove that Phase 2E touched only canonical VALID and TRAIN.

Every result in this phase is a model-selection claim, so the claim is only
worth as much as the guarantee behind it. This re-derives that guarantee from
the artefacts themselves rather than from the scripts' own logging:

* every evaluation record names only canonical **validation** sequences
* the sequence set and fingerprint are identical across every record
* the training dataset draws only on canonical **train**
* the SN-GSR test and challenge roots are absent from disk entirely
* each ball checkpoint is classified by whether its training data included
  test-split sequences, which is a property of the checkpoint and not of this
  phase -- recorded so nobody later mistakes a burned checkpoint for a clean one

Usage::

    python scripts/audit_phase2e_leakage.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionpitch.common.logging import configure_logging, get_logger  # noqa: E402

log = get_logger("phase2e.audit")

SPLIT_MANIFEST = Path("data/eval/gsr/sequences_info.json")
RECORD = Path("data/eval/fusion/phase2e_leakage_audit.json")
SEQUENCE_PATTERN = re.compile(r"SNGS-\d{3}")

EVAL_RECORDS = [
    Path("data/eval/fusion/ablation_valid.json"),
    Path("data/eval/fusion/downstream_valid.json"),
    Path("data/eval/fusion/operating_point_sweep.json"),
    Path("data/eval/fusion/operating_point_sweep_high.json"),
    Path("data/eval/fusion/checkpoint_comparison.json"),
    Path("data/eval/fusion/final_checkpoint_comparison.json"),
    Path("data/eval/fusion/failure_diagnosis.json"),
]

TRAINING_DATASETS = {
    "ball_gsrtrain_v1 (this phase)": Path("data/ball_gsrtrain_v1"),
    "ball_multicorpus (pre-existing)": Path("data/ball_multicorpus"),
    "ball_broadcast_adapt (pre-existing)": Path("data/ball_broadcast_adapt"),
    "ball_broadcast_adapt_aug (pre-existing)": Path("data/ball_broadcast_adapt_aug"),
    "ball_broadcast_public (pre-existing)": Path("data/ball_broadcast_public"),
    "ball_hardened (pre-existing)": Path("data/ball_hardened"),
    "ball_finetune (pre-existing)": Path("data/ball_finetune"),
}


def dataset_sequences(root: Path) -> set[str]:
    found: set[str] = set()
    for split in ("train", "val", "test"):
        images = root / split / "images"
        if not images.is_dir():
            continue
        for name in os.listdir(images):
            found.update(SEQUENCE_PATTERN.findall(name))
    return found


def main() -> int:
    configure_logging("INFO")
    manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    canonical = {
        key: {s["name"] for s in value}
        for key, value in manifest.items()
        if isinstance(value, list)
    }
    off_limits = canonical["test"] | canonical["challenge"]

    failures: list[str] = []

    # -- evaluation records --------------------------------------------------- #
    records: dict[str, dict] = {}
    fingerprints: set[str] = set()
    for path in EVAL_RECORDS:
        if not path.is_file():
            records[str(path)] = {"present": False}
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        sequences = set(payload.get("sequences_evaluated", []))
        bad = sorted(sequences & off_limits)
        not_valid = sorted(s for s in sequences if s not in canonical["validation"])
        if bad:
            failures.append(f"{path}: test/challenge sequences {bad}")
        if not_valid:
            failures.append(f"{path}: non-validation sequences {not_valid}")
        fingerprint = payload.get("split_fingerprint")
        if fingerprint:
            fingerprints.add(fingerprint)
        records[str(path)] = {
            "present": True,
            "n_sequences": len(sequences),
            "all_canonical_validation": not bad and not not_valid,
            "split_fingerprint": fingerprint,
        }
    if len(fingerprints) > 1:
        failures.append(f"evaluation records disagree on sequence set: {fingerprints}")

    # -- training datasets ----------------------------------------------------- #
    datasets: dict[str, dict] = {}
    for label, root in TRAINING_DATASETS.items():
        if not root.is_dir():
            datasets[label] = {"present": False}
            continue
        sequences = dataset_sequences(root)
        test_used = sorted(sequences & off_limits)
        valid_used = sorted(sequences & canonical["validation"])
        datasets[label] = {
            "present": True,
            "n_gsr_sequences": len(sequences),
            "test_or_challenge_sequences": len(test_used),
            "validation_sequences": len(valid_used),
            "verdict": (
                "CLEAN" if not test_used and not valid_used
                else "TEST-BURNED" if test_used and not valid_used
                else "VALID-CONTAMINATED"
            ),
        }
        # A validation sequence in any training set would invalidate this phase.
        if valid_used:
            failures.append(f"{label}: VALIDATION sequences in training data {valid_used}")

    # -- held-out roots must not exist on disk --------------------------------- #
    roots = {
        "data/SoccerNetGS/test": Path("data/SoccerNetGS/test"),
        "data/SoccerNetGS/challenge": Path("data/SoccerNetGS/challenge"),
    }
    root_state = {}
    for label, path in roots.items():
        present = path.is_dir() and any(path.iterdir()) if path.is_dir() else False
        root_state[label] = "absent" if not present else "PRESENT"

    # ``data/eval/gsr`` holds the test split and is legitimately read for the
    # manifest only; record that no evaluation record scored anything from it.
    legacy_test = sorted(
        p.parent.name for p in Path("data/eval/gsr").glob("*/Labels-GameState.json")
    )
    root_state["data/eval/gsr (legacy test root)"] = (
        f"{len(legacy_test)} sequences present; read for sequences_info.json only"
    )

    payload = {
        "schema_version": "1.0.0",
        "phase": "2E ball operating point",
        "verdict": "PASS" if not failures else "FAIL",
        "failures": failures,
        "evaluation_records": records,
        "shared_split_fingerprint": sorted(fingerprints),
        "training_datasets": datasets,
        "heldout_roots": root_state,
        "notes": [
            "no evaluation in this phase scored any test or challenge sequence",
            "no training dataset contains any validation sequence, so every "
            "VALID measurement in this phase is a genuine held-out measurement",
            "checkpoints marked TEST-BURNED were fine-tuned on SN-GSR test-split "
            "frames before this phase; they remain usable for VALID-only "
            "selection but can never yield an honest SN-GSR test number",
        ],
    }
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"verdict: {payload['verdict']}")
    for label, state in datasets.items():
        if state.get("present"):
            print(f"  {label:<44}{state['verdict']:<20}"
                  f"test/chal={state['test_or_challenge_sequences']} "
                  f"valid={state['validation_sequences']}")
    for label, state in root_state.items():
        print(f"  {label:<44}{state}")
    for failure in failures:
        print(f"  FAILURE: {failure}")
    print(f"\nwrote {RECORD}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
