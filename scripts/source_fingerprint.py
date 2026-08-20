"""Deterministic source-tree fingerprint, standing in for a git revision.

This repository has no git commits, so results cannot cite a revision hash. This
computes a stable hash over the tracked source instead: every ``.py``, ``.yaml``
and ``.md`` under the directories that determine behaviour, hashed by relative
path and content, with line endings normalised so a checkout on another platform
produces the same value.

It is not a substitute for version control -- it cannot tell you *what* changed,
only that something did. Initialising a repository would be strictly better and
is a one-line change.

Usage::

    python scripts/source_fingerprint.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOTS = ("src", "scripts", "tests", "configs")
SUFFIXES = {".py", ".yaml", ".yml"}
EXCLUDE_PARTS = {"__pycache__", ".venv", "node_modules", ".git", ".pytest_cache"}


def iter_files(base: Path) -> list[Path]:
    out: list[Path] = []
    for root in ROOTS:
        directory = base / root
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUFFIXES:
                continue
            if EXCLUDE_PARTS & set(path.parts):
                continue
            out.append(path)
    return sorted(out)


def fingerprint(base: Path) -> dict:
    digest = hashlib.sha256()
    entries = []
    for path in iter_files(base):
        relative = path.relative_to(base).as_posix()
        # Normalise line endings so the value does not depend on the checkout's
        # autocrlf setting.
        content = path.read_bytes().replace(b"\r\n", b"\n")
        file_hash = hashlib.sha256(content).hexdigest()
        digest.update(relative.encode())
        digest.update(file_hash.encode())
        entries.append({"path": relative, "sha256": file_hash[:16], "bytes": len(content)})
    return {
        "source_fingerprint": digest.hexdigest()[:16],
        "n_files": len(entries),
        "roots": list(ROOTS),
        "suffixes": sorted(SUFFIXES),
        "files": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("data/eval/source_fingerprint.json"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    payload = fingerprint(args.base.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.quiet:
        print(payload["source_fingerprint"])
    else:
        print(f"source fingerprint: {payload['source_fingerprint']}")
        print(f"files hashed      : {payload['n_files']}")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
