# Broadcast ball annotation — reviewer guide

Read this once before starting. It takes about four minutes and it is the
difference between a dataset that measures something and one that does not.

## What you are producing

An independent answer key. Two detectors have already guessed where the ball is
in every frame, and **their guesses are not the answer** — they are there to save
you clicks when they happen to be right. Wherever you are unsure, the honest
label is "ambiguous", not "whatever the model said". A dataset that quietly
agrees with a model cannot be used to measure that model.

You will see roughly 400 frames. Most take two or three seconds.

## Launching

```bash
python scripts/annotate_balls.py
```

Then open <http://127.0.0.1:8009>. Work is saved frame by frame; close the
browser and re-run the command whenever you like — it resumes at the first
unreviewed frame.

## The decision, in order

For each frame ask, in this order:

**1. Is this live play?**
If the frame is a replay, a slow-motion repeat, a studio graphic, a crowd shot
or a bench reaction, press `r` (replay) or `l` (non-live) and save. These are
excluded from detection scoring entirely. You do not need to find the ball in
them.

The tool guesses shot type and shows it, but the guess is often wrong about
replays specifically — trust your eyes, not the label.

**2. Can you see the ball?**

| Situation | Key | Meaning |
|---|---|---|
| You can see it | click on it | the ball's centre, as precisely as you can |
| It is in shot but hidden — under players, behind a body | `n` | not visible |
| It is out of the picture entirely | `o` | outside frame |
| You genuinely cannot tell | `a` | ambiguous — excluded from scoring |

These four are **not interchangeable**, and the difference matters more than it
looks:

- `n` (not visible) says *a detector finding nothing here is correct*.
- `o` (outside frame) says *a detector finding anything here is hallucinating*.
- `a` (ambiguous) says *do not score this frame at all*.

Guessing between them is worse than marking `a`.

**3. Where exactly, and how big?**
Zoom in with the scroll wheel before clicking. On this broadcast the ball is
about **14 pixels across**, so at 100% zoom a careless click is easily 10 px out.
Zoom to 400–800%, click the centre, and the readout in the corner confirms both
the position and the radius.

Clicking places a circle at the current radius. **Resize it so the circle matches
the ball you can actually see** — that size is stored as real ball size, not as a
marker size, and it is what lets us later report accuracy separately for tiny
far-side balls and large close-shot ones.

Four ways to resize, use whichever is quickest:

- **drag the green handle** on the right edge of the circle — centre stays put
- **scroll the wheel while the cursor is over the ball** — resizes instead of
  zooming
- **`+` / `-`** in 0.5 px steps, **`Shift`+`+` / `Shift`+`-`** in 2 px steps
- the **Ball radius** slider and number box in the panel

The radius **carries over to the next frame**, because ball size changes slowly
across a passage of play. Press `d` to return to the 7 px default.

If you click and then see you were off, drag from inside the circle to move it —
that keeps the radius. Clicking outside it places a fresh ball. The last state
before you press save is what gets stored.

## Using the proposals

Press `1` to show Model A (orange) and `2` to show Model B (blue). Both start
hidden on every frame, deliberately: seeing a confident marker before you have
looked makes you agree with it.

Look first. Then, if a proposal is genuinely on the ball, press `q` (accept A)
or `w` (accept B) instead of clicking — it is faster and just as accurate.

Accepting is recorded separately from clicking, so we can check afterwards how
much of the dataset came from agreement rather than independent placement. Do
not accept a proposal that is "close enough"; click the right spot instead.

## When the ball is hard to find

Press `[` and `]` to step through the two frames either side. A ball that is
invisible in a still is often obvious in motion — you can see where it came from
and where it goes. Use this before reaching for `a`.

If after looking at the neighbours you still cannot place it within a few pixels,
mark `a` and move on. Roughly 5–15% ambiguous is normal and healthy.

## Full controls

| | |
|---|---|
| click empty space | place a ball at the current radius (marks the frame visible) |
| drag from inside the circle | move it, radius unchanged |
| drag the edge handle | resize, centre unchanged |
| scroll **over the ball** | resize (`Shift` for coarse steps) |
| scroll **anywhere else** | zoom at the cursor |
| `+` / `=` , `-` | radius ± 0.5 px |
| `Shift` + `+` / `-` | radius ± 2 px |
| `d` | reset radius to the 7 px default |
| middle-drag or space-drag | pan |
| `0` | reset the view |
| `1` / `2` | toggle Model A / Model B proposal |
| `q` / `w` | accept Model A / Model B proposal |
| `n` | ball not visible |
| `o` | ball outside frame |
| `a` | ambiguous |
| `r` | ignore — replay |
| `l` | ignore — non-live |
| `u` | undo, clear the frame's marks |
| `[` / `]` | previous / next context frame |
| `←` / `→` | previous / next sample |
| `j` | jump to the next unreviewed frame |
| `Enter` | save and advance |

## Things that are easy to get wrong

**A ball at the very edge of the picture is still visible.** Use `o` only when
it is fully out of shot.

**A ball behind the goal net, in shadow, or against a white line is visible** if
you can locate it. Zoom in.

**Do not mark the ball where you think it should be.** If it is behind a player,
that is `n`, not a guess at the hidden position. The whole point of `n` is that
"no detection" is the correct answer there.

**Motion-blurred balls are visible.** Click the centre of the smear.

**If two things look like the ball**, use the context frames to disambiguate. If
they still both look plausible, mark `a`.

## What happens next

Once every frame has a decision:

1. A quality-control pass flags frames for a second look — detector
   disagreements, positions inconsistent with their neighbours, isolated
   single-frame placements, and everything marked ambiguous.
2. The reviewed frames are split by shot into train / validation / test, so
   near-duplicate frames cannot land on both sides of a split.
3. The test split is locked before anything is trained, and is scored once.

Your annotations are appended to `annotations.jsonl`; earlier decisions are never
overwritten, so changing your mind is safe and leaves a trail.
