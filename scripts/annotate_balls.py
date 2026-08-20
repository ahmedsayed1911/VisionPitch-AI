"""Launch the local ball-annotation interface.

Broadcast ball annotation workflow, step 4.

Serves the review package on localhost. Nothing leaves the machine: images are
read from disk, annotations are appended to a local JSONL file, and no request
is made to any network service.

Usage::

    python scripts/annotate_balls.py
    python scripts/annotate_balls.py --package data/annotation/package --port 8009
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visionpitch.annotation.schema import AnnotationStore  # noqa: E402
from visionpitch.common.logging import configure_logging, get_logger  # noqa: E402

log = get_logger("annotate")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path("data/annotation/package"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8009)
    parser.add_argument("--status", action="store_true",
                        help="print progress and exit without serving")
    parser.add_argument(
        "--qc", action="store_true",
        help=(
            "focused mode: serve only the prioritised unreviewed queue, ordered "
            "so the scarcest coverage quota is filled first"
        ),
    )
    parser.add_argument("--qc-total", type=int, default=125,
                        help="approximate size of the focused queue")
    args = parser.parse_args()

    configure_logging("INFO")
    store = AnnotationStore(args.package)
    if not store.samples_path.exists():
        log.error(
            "no review package at %s; run scripts/build_annotation_package.py first",
            args.package,
        )
        return 1

    progress = store.progress()
    manifest = store.manifest()
    print(f"package : {args.package.resolve()}")
    print(f"video   : {Path(manifest.get('source_video', '?')).name}")
    print(f"frames  : {progress['n_samples']}")
    print(f"done    : {progress['n_annotated']}  |  remaining: {progress['n_remaining']}")
    if progress["by_visibility"]:
        print(f"labels  : {progress['by_visibility']}")

    if args.qc or args.status:
        from visionpitch.annotation.qc import progress_report

        report = progress_report(args.package, args.qc_total)
        print(f"\nQC queue: {report['n_queued']} frame(s), priority order "
              f"{' > '.join(report['priority_order'])}")
        negative = report["negative_quota"]
        print(f"\n{'quota':<22}{'target':>8}{'have':>7}{'queued':>8}{'avail':>7}  reachable")
        print(f"  {'genuine negatives':<20}{negative['target']:>8}"
              f"{negative['achieved']:>7}{negative['queued']:>8}"
              f"{negative['available_unreviewed']:>7}  "
              f"{'yes' if negative['reachable_from_this_package'] else 'NOT FROM THIS PACKAGE'}")
        for name, q in report["category_quotas"].items():
            print(f"  {name:<20}{q['target']:>8}{q['achieved']:>7}{q['queued']:>8}"
                  f"{q['available_unreviewed']:>7}  "
                  f"{'yes' if q['reachable_from_this_package'] else 'no'}")
        windows = report["temporal_windows"]
        print(f"\ncomplete temporal windows: {windows['complete_now']} of "
              f"{windows['target_complete']}")
        if report["genuine_negatives_by_kind"]:
            print(f"negatives so far: {report['genuine_negatives_by_kind']}")

    if args.status:
        return 0

    from visionpitch.annotation.server import serve

    print(f"\n  ->  http://{args.host}:{args.port}"
          f"{'   [QC MODE]' if args.qc else ''}\n")
    print("  click = ball centre   n = not visible   o = outside frame   a = ambiguous")
    print("  r = ignore replay     l = ignore non-live")
    print("  1/2 = show proposals  q/w = accept proposal   [ ] = context frames")
    print("  drag handle / wheel over ball = resize   + - = radius   d = default")
    print("  j = next unreviewed   Enter = save    (Ctrl+C here to stop)\n")
    if args.qc:
        print("  QC mode serves only the prioritised queue. A frame the sampler")
        print("  called crowd or graphics is NOT automatically a negative --")
        print("  mark what you actually see.\n")
    serve(args.package, host=args.host, port=args.port,
          qc=args.qc, qc_total=args.qc_total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
