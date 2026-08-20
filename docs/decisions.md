# Architectural decision record

Decisions that were non-obvious, contested, or that a future contributor would
otherwise be likely to reverse by accident. Newest last.

---

## ADR-001 — Reuse the working system PyTorch instead of a clean venv

**Context.** The machine already had a working `torch 2.12.1+cu126` on Python
3.14 with CUDA verified against the RTX 3090 Ti. A hermetic venv would have
meant a ~2.5 GB re-download.

**Decision.** Create the venv with `--system-site-packages` and install only
the missing packages.

**Consequence.** Fast setup, but the venv is not hermetic. `docs/setup.md`
documents the clean-install path via `pyproject.toml` for other machines, and
`pyproject.toml` pins real version floors so reproduction elsewhere does not
depend on this machine's state.

---

## ADR-002 — Define our own named pitch landmark set

**Context.** Third-party football keypoint models use unlabelled index
conventions (commonly 32 points) with no authoritative documentation.

**Decision.** Define 37 explicitly named landmarks in
`visionpitch.geometry.pitch`, each tagged with a `LandmarkKind` describing how
it is geometrically defined.

**Rationale.** Named landmarks are self-documenting and testable; symmetry and
circle-membership can be asserted in unit tests. We train our own keypoint
model, so we are not bound to anyone else's ordering. External conventions can
be mapped in one adapter table.

**Consequence.** The landmark order is a **stable contract** between the
template and any trained checkpoint. New landmarks must be appended, never
inserted. `PitchKeypointDetector` raises on a count mismatch rather than
misinterpreting channels.

Circle extrema (`centre_circle_left/right`) carry weight 0 and are excluded
from fitting: they are tangency points that shift under perspective, so using
them as correspondences is simply wrong.

---

## ADR-003 — Plain RANSAC, not USAC_MAGSAC, for homography

**Context.** `cv2.USAC_MAGSAC` is generally the stronger robust estimator.

**Decision.** Use `cv2.RANSAC`, then refit on the consensus set with a weighted
normalised DLT.

**Evidence.** On synthetic correspondences from a known camera where plain
RANSAC recovers the homography to 0.0 px, OpenCV 4.13's `USAC_MAGSAC` returned
`None` for every input tried — both dtypes, both array shapes, and with Hartley
normalisation applied. It fails silently by returning `None`, not by raising,
so an earlier fallback guarded only against `cv2.error` and never triggered.

**Consequence.** Accuracy comes from the weighted refit rather than the
consensus method. Per-landmark reliability weights (which OpenCV cannot
express) are applied there. Revisit if a future OpenCV fixes USAC.

---

## ADR-004 — Smooth calibration in image space, not matrix space

**Context.** Per-frame homographies jitter; the minimap is unwatchable without
smoothing.

**Decision.** Project a fixed grid of pitch control points into the image,
smooth those 2D tracks with an adaptive gain, and refit a homography from the
smoothed positions.

**Rationale.** Homographies live in a projective space; linear interpolation of
matrix entries does not correspond to interpolation of the induced transform.
Averaging coefficients is not a meaningful operation.

**Consequence.** The gain yields fully to observations when displacement
exceeds `motion_gate_px`, so a genuine pan is followed rather than dragged
behind. During dropouts the last good calibration is held for at most
`max_gap_frames` with **decaying confidence**, then reported unavailable rather
than extrapolated indefinitely.

---

## ADR-005 — Cost terms abstain instead of reporting maximum distance

**Context.** `embedding_distance` originally returned `1.0` when a track had no
appearance gallery, and `pitch_distance` returned `1.0` when either side lacked
a pitch position.

**Problem found in integration testing.** `assess_quality` rejects crops under
42 px, so in wide shots most tracks have empty galleries. The appearance gate
then read `1.0` as "looks completely different" and vetoed **every** candidate
match, so no track ever confirmed and the pipeline produced zero output.

**Decision.** Missing evidence returns `nan`. `blend_costs` normalises **per
entry**, so a pair scored by one cue is directly comparable to one scored by
three. Gates apply only where the cue actually spoke.

**Consequence.** "No evidence" and "strong disagreement" are now distinct
throughout association. This distinction is load-bearing; do not collapse it.

---

## ADR-006 — DIoU rather than IoU for primary association

**Context.** Distant broadcast players are under ten pixels wide. IoU is flat
at zero for all non-overlapping pairs, so a few pixels of residual
camera-compensation error erases all ranking information.

**Decision.** Use DIoU (IoU minus normalised centre separation) in the primary
association stage. Plain IoU is retained for the low-confidence stage, where
strictness is wanted.

**Evidence.** Synthetic pan benchmark, 10 players over 30 frames with real
GMC: IoU produced **17** distinct identities, DIoU produced exactly **10**.

---

## ADR-007 — A scene cut requires positive evidence

**Context.** The motion compensator originally reported `scene_cut=True`
whenever it could not find enough features to track.

**Problem found in integration testing.** On low-texture footage this fired
every frame, marking all tracks `LOST` continuously and destroying tracking.

**Decision.** A cut is asserted only when features that *were* being tracked
confidently are lost wholesale (`tracked_ratio < 0.25` from a healthy prior
count), or when a well-supported fit implies an impossible jump. Insufficient
texture yields `ok=False, scene_cut=False` — unusable motion, but no claim
about a cut.

**Principle.** Absence of evidence is not evidence. This mirrors ADR-005.

---

## ADR-008 — Implement tracking metrics in-repo, pinned to TrackEval

**Context.** `trackeval` is the reference implementation but is built around
MOTChallenge file layouts, which is awkward for in-memory evaluation.

**Decision.** Implement HOTA/IDF1/CLEAR directly against our structures, and
pin them to `trackeval` in `tests/test_metrics_reference.py`.

**Evidence.** Max absolute deviation **0.0** across five scenarios (perfect,
ID-swap, missing detections, noisy with false positives, fragmented) for HOTA,
DetA, AssA, IDF1, IDP, IDR, MOTA and IDSW, including the full per-alpha HOTA
curve.

**Consequence.** `trackeval` stays an evaluation-only dependency. If the two
ever disagree, the reference wins and our implementation is the bug.

---

## ADR-009 — The ball pipeline is offline

**Context.** Correctly filling a gap in the ball trajectory requires the
observation *after* the gap.

**Decision.** Buffer frames and resolve in `finalise()`. Rendering therefore
needs a second decode pass.

**Rationale.** Phase 1 optimises accuracy over latency. The alternatives were a
rendering lag or forward-only extrapolation, and extrapolation would mean
fabricating positions — the one thing this project must not do.

**Consequence.** Interpolation is capped at `max_interpolation_gap` and
additionally requires the two anchors to be mutually reachable, so two
unrelated ball sightings are never joined. Every filled frame is labelled
`INTERPOLATED`, never `OBSERVED`.

---

## ADR-010 — Ship a colour kit encoder as the fallback, not the target

**Context.** A learned embedding backbone (DINOv2, ~90 MB) was not approved for
download.

**Decision.** Implement a pluggable `CropEncoder` interface with two
implementations: `KitColourEncoder` (no download) and `DinoV2Encoder` (fetches
weights, fails loudly if unavailable).

**Rationale.** The colour encoder is a considered descriptor — torso band only,
grey-world colour constancy, CIELAB chromatic channels, grass masking,
histogram rather than mean — not naive RGB averaging. But it is not a learned
embedding, and the project should not pretend otherwise.

**Consequence.** The pipeline runs end-to-end today. Team-classification
quality is expected to be materially lower than with a learned encoder, and
this is recorded in `docs/limitations.md`. `DinoV2Encoder` raises rather than
silently degrading, so a run configured for learned embeddings never quietly
produces colour-histogram results.

---

## ADR-011 — Detector class is evidence for a role, not the role itself

**Context.** Real-video QA showed ordinary players rendered as match officials,
and goalkeepers assigned unreliably. The cause was structural rather than a
model deficiency: a track's `object_class` was fixed by its **birth detection**
and never revisited, and referee-class tracks then skipped team classification
entirely. One spurious `referee` box at the moment a track was created was
therefore enough to relabel a player for the rest of the clip *and* erase their
team. The detector's `referee` class has the weakest precision of the four.

**Decision.** Accumulate confidence-weighted class votes over a track's whole
life (`Track.class_votes`, merged on stitching), vote every person track for a
team including referee-class ones, and resolve roles afterwards in
`team_classification/roles.py` from four independent signals: class votes, team
vote confidence, kit distance from both team colour centroids, and median
longitudinal pitch position.

**Rationale.** A referee call has to clear three bars before it may overwrite a
team: persistence, no confident team assignment, and a kit that actually sits
outside both colour clusters. A goalkeeper needs class evidence *and* to spend
the clip near a goal line, because a keeper is defined by where they stand.

**Consequence.** Measured on the full broadcast, referee-labelled person-frames
fell from 6.9% to ~1%, with 31 referee calls overruled. Where the evidence
conflicts the resolver abstains — `Role.UNKNOWN`, which the showcase renderer
draws as nothing. Drawing nothing is a gap; drawing a midfielder as an official
is a visible error. Thresholds live under `team_classification.roles`.

---

## ADR-012 — The crop budget is shared fairly, not chronologically

**Context.** The team-crop harvester enforced a global cap by refusing new crops
once full. On the 528 s broadcast the pipeline wants ~48,700 crops against a
12,000 cap, so the budget was spent in the opening minutes and every track born
afterwards harvested *zero* crops — reported as UNKNOWN team rather than as
starved. A per-track ceiling was added first and moved the full-video UNKNOWN
rate from 63.9% to 53.5%, because it only binds on tracks long enough to reach
it.

**Decision.** Accept every crop, then evict one from whichever track currently
holds the most, choosing the most redundant sample within that track.

**Rationale.** Refusing at the door is precisely what tied the budget to arrival
order. Evicting from the largest holder makes coverage independent of when a
track appears; evicting the sample whose removal leaves the smallest temporal
gap keeps each track's crops spread across its appearance rather than collapsed
onto one moment.

**Consequence.** Memory stays bounded by `team_classification.max_crops`
exactly, and raising it now buys vote *quality* rather than coverage. The
per-track ceiling is retained for the opposite reason it was introduced: to stop
one very long track dominating the sample it is voting with.
