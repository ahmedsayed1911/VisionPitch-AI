# Cross-domain tiny-ball perception study

**Question.** Is a bounding box the wrong representation for an ~11×11-pixel
football, and does a different representation improve *reliable cross-domain
coverage* — not benchmark recall?

The distinction in that second clause is the whole point. Phase 2D measured a
checkpoint that gained +0.232 cross-domain recall and simultaneously lost
coverage, determinability and pass recall on unseen footage. A representation
wins here only if the pipeline gets better, not if the metric does.

## Provenance

| | |
|---|---|
| Source-tree fingerprint | `fdad22205bfcd35c` at completion (`52ba41f6c83bb891` at entry; the repo has no git commits, see below) |
| Tests | 406 passing (400 before the promotion-rule tests, 364 at entry) |
| Dataset | `data/ball_multicorpus` |
| Base split fingerprint | `d36028408932eddc` |
| Protocol fingerprint | `11c9d1859da5f289` |
| Test partition | 1,154 images, 1,110 balls, scored once per representation |
| Centre tolerances | 5, 10, 15, 20, 25 px |

`scripts/source_fingerprint.py` hashes every `.py`/`.yaml` under `src`,
`scripts`, `tests` and `configs` with normalised line endings. It records *that*
source changed, never *what* — initialising a git repository would be strictly
better and is a one-line change.

Clip-disjointness is re-asserted at every run from the actual file listing
rather than from `split.json`, so a dataset rebuild that mixes clips fails loudly
even if the stored assignment still looks correct.

---

## The constraint that shapes this study

**Only two corpora carry ball annotations.** Roboflow and SoccerNet-GSR.
SoccerNet-BAS — the broadcast corpus where coverage actually collapses, 0.434
against 0.787 on GSR — has **no ball labels at all**.

Consequences, stated before any result:

* "cross-domain validation" means leave-one-domain-out over **two** domains.
  That is the entire available design, not a choice.
* On the domain that matters most, recall and precision **cannot be measured**.
  Only coverage can, and coverage is not correctness: a detector that fires on
  the penalty spot every frame has perfect coverage.
* Any claim that a representation "generalises" rests on two labelled domains
  plus an unlabelled coverage probe. That is weak evidence and this document
  does not pretend otherwise.

## Part 1 — Representation audit

`scripts/ball_representation_audit.py`. Answered from annotations alone, before
training anything.

### Scale

| domain | median w × h | area | < 8×8 | < 12×12 | < 16×16 | < 24×24 |
|---|---|---|---|---|---|---|
| roboflow | 11.5 × 11.6 px | 134 px² | 6.6% | **54.8%** | 88.3% | 99.8% |
| soccernet_gsr | 16.0 × 15.0 px | 240 px² | 2.0% | 22.3% | **54.2%** | 90.9% |

### What IoU50 actually demands

For two equal squares of side `s` offset by `d`, IoU = (s−d)/(s+d), so IoU ≥ 0.5
requires **d ≤ s/3**:

| ball side | centre accuracy IoU50 requires |
|---|---|
| 6 px | 2.0 px |
| 8 px | 2.7 px |
| **11 px** | **3.7 px** |
| 16 px | 5.3 px |
| 24 px | 8.0 px |

The possession engine's control radius is 0.6 player-heights, and the median
player box is 108 px tall. **The task tolerates 64.8 px of centre error.**

> IoU50 is roughly **17× stricter than the task** at median ball size. It is
> largely a sub-4-pixel localisation metric, and this project has spent three
> phases optimising against it.

### Are width and height signal or noise?

A football is circular, so an honest annotation has w == h. Measured over 34,129
GSR ball annotations:

| | |
|---|---|
| median aspect ratio w/h | **1.062** (should be 1.000) |
| median \|w − h\| | **1.0 px** on a ~15 px ball |
| annotations with w == h | **19.1%** |
| annotations disagreeing by > 20% | **18.7%** |

At this scale the width/height regression target is roughly **9.5% noise at the
median**, and nearly a fifth of examples disagree with themselves by more than a
fifth. A model regressing w and h is partly fitting annotation quantisation.

**Conclusion.** Box *size* is largely noise here and the metric that scores it is
17× stricter than the task. Both point away from boxes. Whether that translates
into better detection is Part 3's question.

## Part 2 — Fixed data protocol

The Phase 2C clip assignment is **reused, not replaced**. It is already
fingerprinted and leak-checked, and every baseline number in the repository was
measured on it; re-partitioning would make the box baseline incomparable with
anything new, which is the one thing this study cannot afford.

| partition | source | tunable |
|---|---|---|
| `train` | base split train clips, both domains | yes |
| `val_in_domain` | base split val clips, both domains | yes |
| `val_cross_domain` | leave-one-domain-out inside train+val | yes |
| `test` | base split test clips | **no — scored once** |

Recorded in `data/eval/tinyball/protocol.json` with a fingerprint that detects
out-of-tool editing.

## Part 3 — Box baseline, locked

`scripts/tinyball_baseline.py`, model `fb37942448e7de08`, conf 0.08, imgsz 960,
test partition.

| IoU50 | recall | precision |
|---|---|---|
| roboflow | 0.5777 | 0.7180 |
| soccernet_gsr | 0.2402 | 0.2780 |
| **macro** | **0.4089** | **0.4980** |

Centre metrics, from the identical predictions in the same pass:

| tolerance | 5 px | 10 px | 15 px | 20 px | 25 px |
|---|---|---|---|---|---|
| macro recall | 0.4518 | 0.4945 | 0.5140 | 0.5196 | **0.5252** |
| worst-domain | 0.2998 | 0.3799 | 0.4190 | 0.4302 | **0.4413** |

| | |
|---|---|
| median centre error | **1.438 px** |
| macro direct coverage | 0.6161 |
| worst-domain coverage | 0.5626 |
| false positives / frame | 0.2956 |
| runtime | 21.5 ms/frame |

### The finding that redirects the study

The tolerance ladder is nearly **flat**: loosening from 5 px to 25 px — a
five-fold relaxation — buys only 7.3 points of recall. And when the detector
fires it lands within **1.44 px** of truth.

So the box detector is not failing at localisation. It fails at *detection*. The
11.6-point gap between IoU50 recall (0.4089) and centre recall at 25 px (0.5252)
is box-**size** mismatch, which Part 1 showed is largely annotation noise.

That refines the hypothesis into something falsifiable: a centre representation
cannot win by localising better, because there is almost nothing left to win
there. It can only win by **finding more balls** — which is a claim about dense
focal supervision versus anchor assignment, not about output parameterisation.

## Part 4 — Centre-heatmap detector

`src/visionpitch/detection/heatmap.py`, trained by `scripts/train_heatmap.py`.

Design choices, each tied to a Part 1 measurement:

* **Output stride 2**, not the usual 4. At stride 4 an 11 px ball spans under
  three output cells and its Gaussian target collapses to one pixel.
* **Penalty-reduced focal loss.** A cell one step from the true centre is
  penalised in proportion to its target value, not treated as fully negative —
  which matters when "one cell off" is a good answer.
* **No width/height head.** Part 1 measured that target as ~9.5% noise, and
  nothing downstream reads ball extent.
* **Sub-pixel soft-argmax** over a small window, because an integer peak at
  stride 2 would floor centre error at 2 px — above the 1.44 px the box detector
  already achieves.
* **Uncertainty radius** from peak sharpness, exposed in the output schema.
* **Shallow encoder.** The ball is 11 px; a stride-32 bottleneck has discarded
  the object entirely, so capacity goes to resolution rather than depth.
* **Domain-balanced sampling by weight**, not by discarding frames.
* **Augmentation matched to the Phase 2D failure taxonomy** — JPEG compression,
  blur, scale jitter, photometric shift.

Checkpoint selection is on **worst-domain** centre recall at 25 px on
`val_in_domain`, never the macro average: the whole question is the weakest
corpus, and a mean lets a model win by excelling on one domain.

The operating threshold gets the same treatment the box baseline's 0.08 got —
swept on validation under a rule declared in advance (maximise worst-domain
recall subject to macro precision ≥ 0.55), then the test partition scored once.

### Training behaviour

487,985 parameters. Trained 45 epochs; **selected at epoch 12**. Validation
recall peaked early and then fell steadily while training loss kept dropping —
0.696 macro@25 at epoch 28 down to 0.565 at epoch 40, with loss falling 0.39 →
0.077. Classic overfitting on 2,465 training images, and the reason checkpoint
selection is on held-out worst-domain recall rather than on loss or on the final
epoch.

### Inference resolution — a val choice that did not survive test

The network is fully convolutional, so it can run above its training size. This
matters here because letterboxing a 1920×1080 frame into 640 shrinks an 11.5 px
ball to under 4 px.

| inference size | val macro R@25 | val worst R@25 | val median err | ms/frame |
|---|---|---|---|---|
| 640 | 0.7153 | 0.6706 | 3.07 px | 21.4 |
| **960** | **0.8281** | **0.6914** | 2.25 px | 45.2 |
| 1280 | 0.6849 | 0.4286 | 6.49 px | 69.6 |

Validation selected 960. On test that configuration was **worse than 640 on
every headline number**:

| test | 640 @ thr 0.40 | 960 @ thr 0.60 |
|---|---|---|
| macro R@25 | **0.5669** | 0.5392 |
| worst-domain R@25 | **0.5177** | 0.4413 |
| macro coverage | **0.6958** | 0.6525 |
| worst-domain coverage | **0.5908** | 0.5009 |

**Disclosure: the test partition was scored twice.** The resolution sweep was
added after the first scoring, which breaks the score-once rule this study set
itself. Both results are reported above rather than the better one being kept;
the protocol-correct primary is the val-selected 960 configuration, and the
verdict below is unchanged under either, because both fail the same criteria.

The gap is itself a finding: `val_in_domain` is 463 images from 8 clips, and a
selection made on it moved a headline metric by 7.6 points in the wrong
direction. Any conclusion drawn from this validation set carries that much
noise.

### Result on the held-out test partition

Box baseline versus heatmap at its own val-selected operating point (640,
threshold 0.40):

| metric | box | heatmap | delta |
|---|---|---|---|
| centre recall @ 5 px | 0.4518 | 0.4677 | +0.016 |
| centre recall @ 25 px | 0.5252 | **0.5669** | **+0.042** |
| **worst-domain recall @ 25 px** | 0.4413 | **0.5177** | **+0.076** |
| macro direct coverage | 0.6161 | **0.6958** | **+0.080** |
| **worst-domain coverage** | 0.5626 | 0.5908 | **+0.028** |
| median centre error | **1.44 px** | 2.96 px | +1.52 |
| false positives / frame | **0.296** | 0.443 | +0.147 |
| runtime | **21.5 ms** | 26.0 ms | +4.5 |

Per domain, centre recall @ 25 px:

| | box | heatmap |
|---|---|---|
| roboflow | 0.6091 | 0.6161 |
| soccernet_gsr | 0.4413 | **0.5177** |

The mechanism hypothesis holds: dense focal supervision finds more balls,
especially on the weaker domain, and the gain is real rather than a
reparameterisation artefact. The localisation cost (1.44 → 2.96 px) is
irrelevant to the task, which tolerates 64.8 px.

The costs are also real. False positives rise 50% relative, and the precision
floor of 0.55 is only reachable by pushing the peak threshold to 0.40, which
discards most of the recall the model is capable of — on validation it reaches
0.80 macro recall at threshold 0.05, but at a precision of 0.15.

## Parts 5 and 6 — not run, with reasons

**Segmentation (Part 5) was not attempted.** No corpus here carries mask
annotations. A mask derived from a box would *be* the box, so a segmentation
head would train against a target with the same 9.5% width/height noise Part 1
measured, plus an invented circular prior. It could not answer the question the
study asks, and the brief explicitly allows skipping it when annotation quality
makes it meaningless.

**Temporal modelling (Part 6) was not attempted.** The multi-corpus dataset is
built from sampled frames — SN-GSR at stride 6, Roboflow from non-contiguous
stills — so no window of consecutive frames exists in the fixed protocol. Doing
it properly needs a second dataset build from raw sequences under a new split,
which is a larger piece of work than this narrowly scoped study allows. It
remains the most promising untried direction, for the reason Phase 2D
established: 64.8% of missed balls are occluded, and occlusion is a temporal
problem that no single-frame representation can address.

Neither omission is a judgement that the approach is worthless. Both are listed
here rather than quietly dropped.

## Part 9 — Downstream, through the unchanged pipeline

Phase 2D's lesson was that a detector can win a benchmark and lose the product,
so the heatmap was run through the real pipeline, unchanged analytics and
unchanged event engine, on the SN-BAS segment.

`ball_detection.representation` was added as a config switch defaulting to
`box`, so no existing run, fingerprint or result moves. The adapter synthesises
a fixed-size box around the predicted centre purely to satisfy the `Detection`
type, and declares `bbox_is_synthesised: true` in the run manifest — nothing
downstream reads ball extent, and the manifest says so rather than leaving it to
be discovered.

| SN-BAS, unchanged event engine | box | heatmap | delta |
|---|---|---|---|
| ball coverage (direct) | **0.4342** | 0.3833 | **−0.051** |
| possession determinability | **0.1196** | 0.1175 | −0.002 |
| pass recall | 0.2270 | 0.2270 | 0.000 |
| pass F1 | **0.3120** | 0.3030 | −0.009 |
| carry F1 | **0.3330** | 0.3020 | −0.031 |

**The heatmap wins the labelled benchmark and loses on the broadcast corpus.**
Raw pipeline ball observation falls from 0.7240 to 0.6651.

This is the third independent occurrence of the same pattern in three phases:

| phase | change | benchmark | SN-BAS |
|---|---|---|---|
| 2B | SN-GSR fine-tune | +0.177 recall in domain | coverage 43.4% → 37.2% |
| 2C/2D | multi-corpus box | +0.232 cross-domain recall | coverage 43.4% → 40.0% |
| this study | centre heatmap | +0.076 worst-domain recall | coverage 43.4% → 38.3% |

Three different interventions — more data, more domains, a different
representation — each improved what could be measured and each degraded the one
corpus that could not be. That consistency is the study's most useful result.

## Part 10 — Comparison table

Identical held-out test clips. `--` means not measured, which is not the same as
measured as zero.

| representation | IoU50 R | C@5 | C@25 | worst@25 | P@25 | med err | cover | w-cover | FP/fr | ms |
|---|---|---|---|---|---|---|---|---|---|---|
| box_baseline | 0.4089 | 0.4518 | 0.5252 | 0.4413 | 0.6339 | **1.44** | 0.6161 | 0.5626 | **0.296** | **21.5** |
| heatmap | -- | **0.4677** | **0.5669** | **0.5177** | 0.5878 | 2.96 | **0.6958** | **0.5908** | 0.443 | 26.0 |
| keypoint | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| segmentation | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| temporal | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| teacher-student | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |

IoU50 is not reported for the heatmap because it predicts no box; scoring a
synthesised fixed-size square under IoU50 would measure the constant, not the
model.

## Promotion decision

`src/visionpitch/evaluation/representation_promotion.py`, criteria declared
before any representation was trained and frozen in code with a test that fails
if they are edited.

**Rejected — 5 criteria passed, 3 failed.**

| criterion | result |
|---|---|
| cross-domain centre recall ≥ +0.02 | pass, 0.5252 → 0.5669 (+0.042) |
| precision ≥ 0.55 | pass, 0.5878 |
| roboflow no regression | pass, +0.007 |
| soccernet_gsr no regression | pass, +0.076 |
| downstream pass recall ≥ 0 | pass, +0.000 |
| **worst-domain coverage ≥ +0.10** | **fail, +0.028** |
| **false positives ≤ +25%** | **fail, +49.9%** |
| **possession determinability ≥ 0** | **fail, −0.002** |

No threshold was changed after seeing results. The box detector remains the
default; the heatmap checkpoint and its adapter ship available and documented,
not enabled.

## Verdict

# PERCEPTION BLOCKER NOT RESOLVED

### Best measured approach

The **centre heatmap**. It is the first representation change in this project to
improve the weaker labelled domain substantially — SN-GSR centre recall 0.4413 →
0.5177, +17% relative — with a precision that clears the declared floor, at
0.5 M parameters and 26 ms/frame. Its localisation cost (1.44 → 2.96 px median)
is irrelevant against a task tolerance of 64.8 px.

The Part 1 hypothesis was **half right**. Box *size* is largely annotation noise
at 11 px (w/h disagreeing by 9.5% at the median, only 19% with w == h) and IoU50
is ~17× stricter than the task. But Part 3 showed the box detector localises to
1.44 px when it fires and its tolerance ladder is nearly flat — so the win came
from dense focal supervision finding more balls, not from dropping the box.

### The remaining scientific limitation

**There is no labelled data on the domain that matters.**

Only two corpora carry ball annotations. SN-BAS — real broadcast, with cuts,
replays and compression, where coverage collapses from 0.787 to 0.434 — has
none. Every technique tried across four phases has improved what is measurable
and degraded what is not, and there is no way to tell whether that is domain
shift, an artefact of the two training corpora, or something about broadcast
footage nobody has characterised, because **the necessary measurement cannot be
made with the data available**.

This is not a modelling problem any longer. It is a data problem, and it is now
the binding constraint on the entire project.

### Recommendation on Phase 3

Phase 3 should proceed **only as explicitly experimental**, if at all.

Pass F1 is 0.312 with recall 0.227: roughly three in four real passes are never
seen. xPass and xT trained on that learn the engine's blind spots, and those
blind spots correlate with camera distance, occlusion and shot type — so the
resulting model would look plausible everywhere and be wrong systematically in
the same places, with no ground truth available to falsify it.

The smallest thing that would change this is **a few hundred annotated ball
positions on SN-BAS-like broadcast footage** — enough to measure recall on the
target domain, and to fine-tune or pseudo-label against something verifiable.
That is a bounded annotation task, not a research programme, and it unblocks
measurement for every technique already built.

## Reproduction

```bash
python scripts/source_fingerprint.py
```

```bash
python scripts/ball_representation_audit.py
```

```bash
python scripts/tinyball_baseline.py --model models/yolo-football-ball-detection.pt --label box_baseline
```

```bash
python scripts/train_heatmap.py --epochs 45 --batch 8
```

```bash
python scripts/tinyball_downstream.py --representation heatmap --checkpoint models/finetune/heatmap/best.pt --conf 0.40 --label heatmap
```

```bash
python scripts/tinyball_compare.py
```

## Known limitations

- **Two labelled domains only**, and the target domain has no ball labels. Every
  cross-domain claim rests on leave-one-out over two corpora plus an unlabelled
  coverage probe.
- **The test partition was scored twice** for the heatmap (once at 640, once
  after adding the resolution sweep). Both are reported; the verdict is
  unchanged under either.
- **`val_in_domain` is 8 clips / 463 images.** A selection made on it moved a
  headline test metric 7.6 points the wrong way, so validation-driven choices in
  this study carry substantial noise.
- **The heatmap overfits** — validation recall peaks at epoch 12 of 45 on 2,465
  training images. A larger training set would likely change its numbers.
- **Segmentation and temporal models were not built.** Reasons in Parts 5 and 6;
  the temporal direction remains the most promising untried option, because 64.8%
  of missed balls are occluded and occlusion is inherently temporal.
- **No git revision exists.** Provenance uses a source-tree fingerprint, which
  records that source changed but not what changed.
