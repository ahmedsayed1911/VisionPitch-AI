# Broadcast ball dataset — coverage audit at 115 reviewed frames

**Verdict: diversity is insufficient. Do not split or train yet.** The gap is
review completion, not sample size — almost everything missing is already sitting
in the unreviewed 285 frames of the existing package. Details in §5.

| | |
|---|---|
| Package | `data/annotation/package` |
| Source video hash | `3f0916b7d5cf7754` |
| Sampling fingerprint | `22f9883813ea93ba` |
| Annotation fingerprint | `41f769ac659b1067` |
| Reviewed | **115 of 400 (28.7%)** |

## 1. What was reviewed

| | |
|---|---|
| Unique broadcast shots | **26 of 76** |
| Frames per shot | min 1, median 3, max 12 |
| Video span covered | 4.8 s – 495.7 s of 528.3 s (**93.8%**) |
| Scorable frames | 115 |
| Independent clicks | **115 of 115** — no proposal was accepted unchanged |

Review was **not** sequential: it spans nearly the whole video with 50 gaps in
the frame ordering. That is a genuinely good temporal spread for 115 frames, and
it means the reviewed subset is not just the opening minutes.

Zero accepted proposals matters more than it looks. Every ball position in this
dataset was placed by hand, so nothing here is a model grading itself.

## 2. Distribution

| category | n | source | status |
|---|---|---|---|
| wide play | 64 | shot type | sufficient |
| medium play | 32 | shot type | sufficient |
| midfield | 26 | sampling | sufficient |
| **aerial ball** | **33** | derived | sufficient |
| **occlusion / body-adjacent** | **20** | derived | splittable, not separately measurable |
| crowded scene | 18 | sampling | splittable, not separately measurable |
| near goal | 13 | sampling | too few |
| camera pan | 13 | sampling | too few |
| fast transition | 9 | sampling | too few |
| temporal window | 9 | sampling | too few |
| motion blur | **4** | sampling | too few |
| low contrast | **4** | sampling | too few |
| graphics | **0** | sampling | **absent** |
| crowd negatives | 19 sampled → **0 genuine** | sampling | **absent** (see §4) |
| ball not visible | **0** | annotation | **absent** |
| ball outside frame | **0** | annotation | **absent** |
| ambiguous | 0 | annotation | none needed |

Aerial and occlusion could not be labelled before review — they depend on where
the ball is. They are **derived** here by running the person detector on the
reviewed frames and comparing player boxes against the human ball position. No
model output became ground truth; the detector only characterises frames a human
already labelled. Median ball-to-nearest-player distance is 43 px.

### Ball size

| | min | p25 | median | p75 | p90 | max |
|---|---|---|---|---|---|---|
| radius px | 5.0 | 6.0 | **8.5** | 18.8 | 30.0 | 60.0 |

26 distinct values across 115 frames, so the resizable annotation is being used
rather than left at the default. The wide spread is real, not error: spot checks
confirmed a 5.5 px radius on a distant midfield ball, 33 px on a tight tracking
shot, and 60 px on a behind-goal camera. This is the size signal no fixed-radius
annotation could have captured.

## 3. Quality

32 of 115 frames (27.8%) are flagged, all for **detector disagreement** — the
human position is more than 40 px from every model that fired. That is expected
and desirable: 103 of the 400 sampled frames were deliberately drawn from
detector disagreements, and those are exactly the frames the dataset exists to
capture. No temporal jumps, no isolated positions, no neighbour inconsistencies.

Four frames were inspected directly against their annotations. All four were
correct, including both extremes of the radius range.

## 4. A correction to my own earlier audit

**My shot classifier is unreliable, and the video audit numbers I reported
earlier are wrong.**

Frame `f007376` was classified `crowd_or_bench`. It is a textbook wide-play
frame: full pitch, both teams, a clearly visible ball that the reviewer
annotated correctly. The cause is that shot classification required at least 4
confident pitch keypoints, and a tight wide shot of midfield shows only the
centre circle and the halfway line — so real play with sparse line markings gets
filed as "no pitch".

The error runs both ways: `f000240` was classified `wide_play` but is a tight
tracking shot with a 66 px ball.

Consequences, stated plainly:

- **Live-play share 62.9% is an underestimate.** The true figure is higher.
- **`crowd_or_bench` 31.5% is an overestimate.**
- **The negative stratum is contaminated.** All 19 reviewed "crowd negatives"
  are real play with a visible ball, which is why the reviewer marked every one
  visible. That was the right call.
- The 0 close-ups reported earlier was also a symptom of the same weakness.

Genuine negatives *do* exist in the package — `f005162` is a goal-celebration
close-up with no ball and no pitch — but **none of them have been reviewed
yet**. All 5 graphic-stratum frames and 16 crowd-stratum frames remain
unreviewed.

## 5. What is missing, and how much of it is already on disk

The decisive gap is that **115 of 115 annotations are `visible`**. There is not
one frame where the ball is absent. Without those:

- **Hallucination cannot be measured.** Three separate interventions across
  Phases 2B–2D improved benchmarks and degraded real footage, and a false-positive
  rate is the metric that would have caught it soonest.
- **`not_visible` and `outside_frame` accuracy cannot be computed**, though the
  workflow's own step 8 requires both.
- **Nothing teaches the detector to suppress a false positive.** A model trained
  only on frames containing a ball learns that a ball is always present.

### Coverage floors used

- **15 frames** — below this a category cannot be split 60/20/20 and still leave
  anything in test.
- **25 frames** — below this a per-category test rate has a 95% interval wider
  than roughly ±0.20, which decorates a report rather than informing a decision.

### The gap, and where it is

| category | now | target | short by | available unreviewed |
|---|---|---|---|---|
| genuine negatives (no ball) | 0 | 25 | **25** | 21 |
| motion blur | 4 | 25 | 21 | 35 |
| low contrast | 4 | 25 | 21 | 21 |
| fast transition | 9 | 25 | 16 | 27 |
| near goal | 13 | 25 | 12 | 19 |
| camera pan | 13 | 25 | 12 | 21 |
| complete temporal windows | 0 | 3 (21 frames) | 18 | 54 |

**Recommendation: review approximately 125 more frames from the package you
already have. Do not create new annotations.**

Every shortfall except negatives is already covered by the unreviewed remainder.
Reviewing those 125 brings the total to ~240 — inside the original 300–500 target
— and lifts every category above the evaluation floor.

Negatives are the one place the existing package may not be enough: 21
negative-stratum frames remain, and §4 shows some will turn out to be real play.
**Review all 21 first.** If fewer than 25 are genuine negatives, additional
sampling is justified at that point, and the exact number will be known rather
than guessed. Recommending it now would be recommending against a number I have
not measured.

## Reproduction

```bash
python scripts/audit_annotations.py
```
