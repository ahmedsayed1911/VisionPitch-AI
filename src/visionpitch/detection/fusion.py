"""Merging detections from the multiclass and specialist detectors.

The two detectors disagree in predictable ways, and the merge rules encode that:

* **Ball.** The specialist is more reliable, so where the two overlap the
  specialist's box wins, but its confidence is *boosted* by agreement rather
  than replaced -- two independent models agreeing is stronger evidence than
  either alone. Non-overlapping candidates from both are kept, because the
  trajectory estimator downstream is designed to reject false positives and
  cannot recover a miss.
* **People.** Only the multiclass detector produces them, so they pass through
  untouched apart from de-duplication.
"""

from __future__ import annotations

from visionpitch.common.types import Detection, ObjectClass


def _dedupe(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    ordered = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: list[Detection] = []
    for candidate in ordered:
        if all(candidate.bbox.iou(k.bbox) <= iou_threshold for k in kept):
            kept.append(candidate)
    return kept


def fuse_detections(
    multiclass: list[Detection],
    ball_specialist: list[Detection],
    iou_threshold: float = 0.5,
    agreement_boost: float = 0.15,
    max_ball_candidates: int = 4,
) -> list[Detection]:
    """Combine the two detectors' outputs for one frame.

    Returns detections sorted by class then descending confidence. Ball
    candidates are capped: keeping every low-confidence blob would swamp the
    association step without improving recall of the true ball.
    """
    people = _dedupe([d for d in multiclass if d.object_class.is_person], iou_threshold)

    mc_balls = [d for d in multiclass if d.object_class is ObjectClass.BALL]
    if not ball_specialist:
        balls = _dedupe(mc_balls, iou_threshold)
    else:
        fused: list[Detection] = []
        for spec in ball_specialist:
            partner = max(
                (m for m in mc_balls if m.bbox.iou(spec.bbox) > iou_threshold),
                key=lambda m: m.confidence,
                default=None,
            )
            if partner is None:
                fused.append(spec)
            else:
                fused.append(
                    Detection(
                        frame_idx=spec.frame_idx,
                        object_class=ObjectClass.BALL,
                        bbox=spec.bbox,
                        confidence=min(1.0, spec.confidence + agreement_boost * partner.confidence),
                        source="ball_consensus",
                    )
                )
        # Multiclass balls the specialist did not see are still evidence.
        for mc in mc_balls:
            if all(mc.bbox.iou(f.bbox) <= iou_threshold for f in fused):
                fused.append(mc)
        balls = _dedupe(fused, iou_threshold)

    balls = sorted(balls, key=lambda d: d.confidence, reverse=True)[:max_ball_candidates]
    return people + balls
