# Annotation guide

Phase 1B measures accuracy on public expert-annotated corpora (see
[PHASE1B_REPORT.md](PHASE1B_REPORT.md)). Annotating **your own** footage is still
worth doing, because those corpora cannot tell you how the system behaves on
your cameras, your kits and your compression. This is how to spend that effort
so the resulting numbers are trustworthy.

## The rule that matters most

**Label every visible object in an annotated frame.** A partially annotated
frame turns real detections into false positives and destroys precision. If a
frame contains something you cannot label confidently, exclude the whole frame
rather than labelling part of it.

## Frame-sampling strategy

Do not annotate 100 consecutive frames. Consecutive frames are nearly identical,
so they cost the same effort as a diverse sample and measure far less. Aim for
**100+ frames spread across these conditions**, roughly evenly:

| Condition | Why it must be represented | Suggested share |
|---|---|---|
| Wide broadcast view | the dominant case | 25% |
| Penalty-area view | dense occlusion, most goalmouth events | 15% |
| Midfield view | few pitch landmarks — the calibration stress case | 15% |
| Active camera pan | the main source of ID switches | 15% |
| Zoom change | breaks size-consistency assumptions | 5% |
| Heavy occlusion / player cluster | corners, walls, celebrations | 10% |
| Ball clearly visible | ball detection ceiling | 5% |
| Ball hidden or airborne | trajectory-search behaviour | 5% |
| Goalkeepers in frame | rarest class, weakest recall (0.842) | 3% |
| Referees in frame | third kit colour, confuses team clustering | 2% |

Sample from **both halves** if the footage spans them: light, shadow and pitch
wear all change.

## Detection vs tracking annotations

These have different requirements and it is fine to produce them separately.

**Detection-only** — boxes and classes, no identities needed. Frames can be
scattered arbitrarily. Use `track_id = -1`.

**Tracking** — needs identity *consistency across temporally adjacent frames*.
That consistency is exactly what IDF1 and HOTA measure; an annotator who
renumbers players each frame produces ground truth that scores every tracker at
zero. Annotate **short contiguous runs** (say 50–100 consecutive frames, several
runs) rather than scattered singles.

## When identity is genuinely ambiguous

Do not guess. Two supported options:

1. **Exclude the frame** from the tracking subset — keep it in the
   detection-only set, where identity does not matter.
2. **Split the run** at the ambiguity and treat the two halves as separate
   sequences, so no metric spans the uncertain moment.

A guessed identity is worse than a missing one: it produces a confident wrong
number instead of a smaller sample.

## Tools

```bash
python scripts/annotate.py boxes data/raw/clip.mp4 --frames 100,160,220,280
```

```bash
python scripts/annotate.py pitch data/raw/clip.mp4 --frames 100,400,700,1000
```

Keys — boxes: `1` player, `2` goalkeeper, `3` referee, `4` ball; drag to draw,
`u` undo, `n`/`p` next/previous frame, `s` save, `q` quit.

Pitch landmarks: click, then type the landmark index. Mark landmarks in frames
where the **camera has moved**, not four consecutive frames — independent
observations are the only way to detect a systematic homography bias, since a
homography fitted to the model's own keypoints always reproduces those keypoints
well whether or not it is right.

## Validating the annotation before trusting it

```python
from visionpitch.evaluation.datasets import validate_ground_truth
from visionpitch.evaluation.ground_truth import load_ground_truth

report = validate_ground_truth(load_ground_truth("data/annotations/clip.json"),
                               require_identity=True)
print(report["issue_counts"])
```

Checks applied: duplicate track ids within a frame, degenerate boxes, frames
annotated with nothing, missing identities where identity is required,
implausible aspect ratios, and which classes are absent entirely.

## Running the evaluation

```bash
visionpitch evaluate outputs/<video_id>/<fingerprint> --annotations data/annotations/clip.json
```

Without `--annotations` the report emits only reference-free diagnostics and
says so explicitly, rather than implying the system went unmeasured by choice.

## How many frames is enough

- **< 30 frames** — indicative direction only; the report says so.
- **100+ frames** across the conditions above — a usable estimate, and the
  bootstrap confidence intervals become meaningful.
- **500+ frames** with several contiguous tracking runs — a number worth
  publishing.
