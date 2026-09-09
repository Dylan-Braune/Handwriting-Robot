# Rewriting pipeline — current status & context

**Best / shipping version: commit `dffeab0` (HEAD `141d00f`).** Everything
below describes that state. An experimental neural attempt lives in
`experimental/` and is NOT part of this — see the end.

---

## What it does

`text string + author id  ->  a pen trajectory in that author's handwriting
->  G-code for the gantry`, with a hard requirement that the output be
**completely legible** and a soft goal of looking like the chosen author.

Ten target authors: IAM writers `150 151 152 153 154 155 384 551 552 588`
(`Data/Datasets/IAMpages10`).

## Data flow / files (all in `Software/CNN/`, top level)

| File | Role |
|---|---|
| `BuildStyleProfile.py` | Offline: per-author profile from non-holdout pages. CTC forced-alignment (frozen `TrainText` recognizer, inference only) gives per-character x-spans in each line; ink is binarized, Zhang-Suen skeletonized, traced to ordered polylines, normalized to a de-slanted baseline/x-height frame, stored as glyph **variants**. Also builds a **denoised prototype** per letter = robust per-stroke median of that author's aligned variants (cancels the ~random cut noise). Plus measured params: slant, x-height, ascender/descender ratios, word spacing, `connectedness`. Writes `NOGIT/StyleProfiles10/<id>.json`. |
| `SynthesizeHandwriting.py` | Online: `SynthesizeText(text, profile, ...)` -> `Trajectory` (list of pen-down polylines, mm, y-up). `SynthesizeLegible(...)` = best-of-N (frozen recognizer scores each) + a **repair pass** that pins still-misread letters to the print anchor. |
| `WriteGCode.py` | `Trajectory` -> G-code + step/dir schedule + plotter preview. All machine constants in the `GantryConfig` dataclass at the top. `SimulateAndCompare` rasterizes the emitted G-code vs the trajectory (IoU). |
| `WriteAsAuthor.py` | End-to-end entry point: `python WriteAsAuthor.py "some text" 3` -> `NOGIT/WriteJobs/<id>/`. |
| `EvaluateLegibility.py` | Headline metric. ~40 present-day sentences, none in IAM. Per author x sentence: uniform-ink render -> frozen recognizer char/word accuracy + shape-only writer-ID, direct and through the emitted G-code. |
| `VerifyRewrite.py` / `VerifyShapeStyle.py` / `TrainAuthorShape.py` | The **shape-only** writer-ID model: real lines stroke-normalized to one constant pen width (= what the gantry produces). Scores 98.9% on the authors' real held-out crops, so it is the honest judge of physical style. The ink-weight model is not usable here (collapses to ~12% under uniform ink). |

## How legibility is enforced (the core mechanism)

1. **Uniform ink**: `RenderTrajectory(uniformInk=True)` (default) draws every
   author at one constant stroke width (`UNIFORM_PEN_WIDTH_XH = 0.11`). Ink
   weight is not a style channel the one-pen gantry can reproduce.
2. **Print anchor for every author**: `CORE_CURSIVE_CONN = 2.0` disables the
   joined-cursive anchor — a glyph that isn't the author's own is drawn from
   the hand-tuned single-stroke print font `_FB`, reshaped to the author's
   slant (capped `CORE_SLANT_CAP_DEG = 22`), x-height and ascender reach.
3. **Keep only clean own-letterforms**: `_SwapProb(lam, priorD)` keeps the
   author's own glyph only when `priorD <= _GLYPH_CLEAN (0.12)` — in practice
   the denoised prototypes; everything else swaps to the anchor, scaled by
   the legibility dial `lam`.
4. **Legibility dial**: `legibility` / profile `legibilityLambda`
   (`DEFAULT_LEGIBILITY = 0.65`). 0 = the author's raw hand, 1 = near plain
   print. All 10 profiles currently carry `0.65`.
5. **Repair pass** in `SynthesizeLegible`: `REPAIR_ROUNDS = 3`,
   `REPAIR_WORST_K = 3` — re-draw, CTC-align to the target, force the worst
   characters to the anchor, keep if the recognizer reads it better.
6. Collision guard + slant cap + connectedness-scaled join thinning in
   `SynthesizeText`.

## Measured performance (novel text, uniform ink, full delivery path)

| | value |
|---|---|
| char legibility | **85.6 %** |
| word legibility | **51.1 %** |
| shape-only style match | 32 % (bimodal — see below) |
| G-code round-trip loss | negligible (char within 0.3 pt; sim IoU ~0.99) |

Per author: **150, 152, 551, 552** keep visible character (style 4–6 / 6).
**151, 153, 154, 155, 384, 588** are essentially printed now (style 0) —
they read cleanly but don't look like the author.

## The ceiling / where to push next

The blocker for the cursive hands is **upstream, in glyph extraction**, not
in synthesis. `BuildStyleProfile.ExtractLineGlyphs` cuts a cursive line into
per-character glyphs by placing boundaries between CTC span centres and
snapping to thin+low ink columns. For heavily joined hands the cuts land
mid-ligature -> stored glyphs are fragments or doubled letters (median
`priorD` ~0.25–0.33 vs ~0.22 for print hands; author 155 has almost no clean
glyph). Approaches already tried and rejected (all measured, no end-to-end
gain): word-crop extraction from `IAMwordsFULL`, width-prior CTC boundary
splitting, joined-cursive anchor, blend-toward-consensus core, repair
escalation ladder.

**The real lever:** a better cursive character segmenter in
`ExtractLineGlyphs` — proper ligature-crossing detection so a letter is
never cut through. That would give the cursive authors clean own-letterforms
and let `legibilityLambda` drop for them without losing legibility.

## Run / test

```
python WriteAsAuthor.py "A fresh sentence, 2026." 153      # -> NOGIT/WriteJobs/153/
python EvaluateLegibility.py                               # full scoreboard (~40 min CPU)
python BuildStyleProfile.py                                # rebuild all 10 profiles (line cache in NOGIT/GlyphCache10)
```

Weights (gitignored, in `NOGIT/weights/`): `paper_cnn_bilstm_ctc_best.pt`
(text recognizer), `author_shape_10_weights.pt` (shape-only writer-ID).
Profiles in `NOGIT/StyleProfiles10/`. `Archive/rewriting_v1/` has the
pre-legibility-work snapshot.

## experimental/ (not shipping)

`experimental/hw_rnn.py` — a compact conditional-LSTM handwriting
synthesizer (Graves-style MDN, per-author embedding), trained on ~5.6k
stroke sequences derived from skeletonized IAM word crops
(`build_strokes.py` -> `NOGIT/strokes10.pt`). Result: it learns per-author
**style** (slant, roundness, connectedness clearly differ) but **not legible
letterforms** — the opposite failure to the rule-based system. Needs real
online stroke data (IAM-OnDB) and GPU time to go further. Kept as a
documented dead-end / starting point.
