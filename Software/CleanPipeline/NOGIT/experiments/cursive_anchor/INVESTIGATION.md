# Investigation: why thiya's reproduced handwriting is bad, and what was tried

Handoff doc for a fresh agent. All claims below were verified against the real
repo and real models in this session, not assumed. All scripts/data/images
referenced are in this same folder
(`Software/CleanPipeline/NOGIT/experiments/cursive_anchor/`) unless noted
otherwise. Run everything with cwd = `Software/CleanPipeline/`.

## 1. The problem

thiya is the most cursive of the 13 stored writers (`connectedness = 0.962` in
her profile, `Software/CNN/NOGIT/StyleProfiles10/thiya.json` — the next
highest is abhinav at 0.789, most writers are well below that). Her
synthesised/reproduced handwriting (via `SynthesizeHandwriting.SynthesizeText`
+ `RenderTrajectory`) looks noticeably worse than other writers', both
visually and when scored by the real text recogniser and writer-ID
classifier.

Her **real** handwriting is not the problem — a held-out real line
(`thiya/Thiya03_13.png`, text `"most likely that patients live a sedentary
lifestyle"`, never used to build her profile) scores 96.2% character accuracy
/ 75% word accuracy on the text recogniser
(`paper_cnn_bilstm_ctc_joint_best.npz`) and 98.95% confidence, correctly
identified, on both writer-ID classifiers (`author_shape_10new_weights.npz`
and the ink classifier). See `thiya_holdout_line.png`. So whatever is wrong
is entirely in the **reproduction pipeline**, not in reading/recognising her.

## 2. Diagnosis: her extracted glyph library is structurally broken, not just noisy

`BuildStyleProfile.ExtractLineGlyphs` (line ~310) force-aligns a line against
its transcript with `ProfileIO.CtcForcedAlign`, then cuts the line into one
glyph per character at the cheapest nearby column (thinnest ink, lowest
height — see `cutCost` computation, same file ~line 331-356). This assumes
there is a real gap to cut at between letters.

For thiya there usually isn't one. `show_thiya_cuts.py` replicates that exact
cut computation and draws the resulting boundaries over her first 10 real
lines (output: `thiya_cuts_10lines.png`). Red solid lines are the final
snapped glyph-cell boundaries that `BuildStyleProfile.py` actually cuts on;
blue dashes are the raw CTC-aligned span before snapping. In words like
"transistor", "circuitry", "recommended" the red lines visibly slice through
the middle of connecting strokes and loops, not between letters — because her
pen genuinely never leaves the paper between characters in these lines, so
there is no low-ink column for the cutter to find.

Consequence: her per-character glyph "variants" aren't noisy copies of one
letterform, they're different arbitrary fragments of one continuous curve.
`glyph_averaging_thiya.png` (same rendering as used for abhinav/yeukita, see
`render_glyph_averaging` logic in the main report thread — not saved in this
folder, trivially rebuildable from `BuildStyleProfile.py`'s `idealMedoid`/
`idealGlyphs` per author) shows this directly for `a,e,n,o,t,s`: e.g. her `o`
variants include a closed loop, an open C, a near-straight diagonal, and a
zigzag — not 12 versions of one shape.

### Quantified with `priorD`

Every extracted glyph variant carries a `priorD` score (`BuildStyleProfile.py`
`_PriorScore`/`BuildLetterPrior`/`PriorFilter`, ~line 689-725): cosine distance
from a cross-author shape prior built from all 13 writers' variants of that
letter together. `priord_distribution.py` computes/plots this across all
glyphs for abhinav, yeukita, thiya (output `priord_distribution.png`):

| author | connectedness | n glyphs | mean priorD | median | p90 | % clean (≤0.12) | % garbage (≥0.55) |
|---|---|---|---|---|---|---|---|
| abhinav | 0.789 | 335 | 0.213 | 0.180 | 0.376 | 20.0% | 2.1% |
| yeukita | 0.417 | 333 | 0.242 | 0.224 | 0.418 | 21.0% | 4.8% |
| thiya | 0.962 | 239 | 0.330 | 0.318 | 0.500 | **7.1%** | 5.4% |

`_GLYPH_CLEAN = 0.12` and `_GLYPH_GARBAGE = 0.55` are the thresholds
`SynthesizeHandwriting._SwapProb` (line 364) uses to decide whether to draw an
author's own glyph or swap to the generic anchor letterform. thiya has barely
a third the "clean" fraction of the other two authors, and even her best
(lowest-priorD) variants are often worse than their worst ones.

## 3. Experiment: how much does forcing the generic anchor help?

`SynthesizeText(..., legibility=lam, perCharLam=...)` controls how often a
character is drawn from the author's own glyph library vs. the generic
anchor letterform (`_CoreGlyph`/`_BaselineGlyph`, `_FB` for print /
`_CURSIVE` for joined cursive, chosen by `profile['connectedness']`).
`perCharLam={i: 1.0 for i in range(len(text))}` forces **every** character to
the anchor (`forceCore=True` inside `_GlyphSource`), stronger than the
`legibility` knob's own max.

`test_thiya_anchor_levels.py` renders 4 novel sentences × 2 seeds under 3
conditions, then scores each render with the real text recogniser
(`paper_cnn_bilstm_ctc_joint_best.npz` — **not** the default-resolved
checkpoint, which is a weaker undertrained one, see §5) and the real shape
writer-ID classifier (`author_shape_10new_weights.npz`):

| condition | char acc | word acc | p(thiya) | correctly ID'd |
|---|---|---|---|---|
| her own glyphs only (`legibility=0.0`) | 23.7% | 1.4% | 76.5% | 100% |
| current default (`legibility=0.65`) | 31.7% | 5.4% | 63.4% | 87.5% |
| forced full anchor (`perCharLam` all 1.0) | 33.6% | 2.5% | **94.3%** | 100% |

Images: `thiya_legibility_levels.png` (0.0/0.3/0.65/1.0 sweep, visual),
`thiya_anchor_level_test.png` (the 3-condition comparison above, visual +
numbers).

**Finding 1 (fairly confident): forcing full anchor is a clear win for
writer-ID**, and the *default mixed condition is the worst of the three*, not
a safe middle ground — one default sample dropped to p(thiya)=20.8% and got
misclassified as writer 551 entirely. Mixing her own broken fragments with
anchor strokes mid-line appears to hurt style consistency more than
committing to either pure extreme.

**Finding 2 (weaker): legibility improves with full anchor but stays poor in
absolute terms** (33.6% char / 2.5% word is still "wrong word salad", just
made of better letter-shapes). Word accuracy doesn't even monotonically
improve (default's 5.4% beats full-anchor's 2.5%). This suggests something
beyond glyph source — possibly the cursive-join geometry in `SynthesizeText`
itself — is also capping her legibility, independent of which glyph source is
used. Not investigated further.

## 4. Experiment: replacing the hand-coded `_CURSIVE` anchor with a public cursive font

`_CURSIVE` (`SynthesizeHandwriting.py` lines 143-216) is 26 hand-typed
single-stroke letterforms, generic (not any author's), used whenever
`connectedness >= 0.5`. Visually rough — several letters (`b`,`f`,`g`,`h`,`k`)
have noticeable zigzags rather than smooth strokes (see left column of
`cursive_comparison.png`).

### What was tried

Fetched the Hershey "cursive" font (`Script 1-stroke (alt)`), a genuine
public-domain single-stroke engraving/plotter font family, as plain JSON data
from `https://raw.githubusercontent.com/techninja/hersheytextjs/master/hersheytext.json`
(`data['cursive']['chars']`, a list of 95 entries for ASCII 33–127, i.e.
**`chars[ord(ch) - 33]`**, each `{'d': <SVG-path-like string, M=pen-up/new
stroke, L=pen-down line>, 'o': <advance width>}`). Saved raw as
`hershey_cursive.json`.

`build_hershey_cursive.py` parses the `d` strings into stroke lists and
normalizes to the project's own glyph convention (x right from glyph start,
y up from baseline, `1.0 = one x-height`), calibrated from the font's own
data: the main-body strokes of non-ascender/descender letters (`n,m,u,v,w,x`)
consistently span raw y in `[13, 22]`, so `BASELINE_Y=22`, `XH_TOP_Y=13`,
`SCALE=9`; `local_y = (22 - raw_y) / 9`, `local_x = (raw_x - x0) / 9` where
`x0` is the glyph's own first point. Output: `hershey_cursive_normalized.json`
(keys `a`-`z`, same nested-list-of-strokes-of-(x,y)-tuples shape as
`_CURSIVE`).

**Note the off-by-one bug that cost time**: the first attempt indexed with
`ord(ch) - 32`, which is wrong — `chars[0]` is `!` (ASCII 33), not space
(ASCII 32), so every letter was shifted by one. Confirmed by checking that
`chars[0]['d']` traces an exclamation mark (vertical line + dot) and
`chars[1]` traces a double-quote (two short parallel strokes). Fixed to
`ord(ch) - 33`. If re-deriving this, sanity-check a known glyph (e.g. that
index for `'a'` actually looks like an `a`) before trusting it.

Individual letterforms came out **visibly much better** than the hand-coded
set — real, recognizable cursive shapes, not zigzags (`cursive_comparison.png`,
right column).

### Result: worse end-to-end, despite nicer individual letters

`test_hershey_cursive_swap.py` monkeypatches `SynthesizeHandwriting._CURSIVE`
to the Hershey-derived dict and reruns the exact same forced-full-anchor test
as §3:

| condition | char acc | word acc | p(thiya) |
|---|---|---|---|
| hand-coded `_CURSIVE`, full anchor | 33.6% | 2.5% | 94.3% |
| Hershey `_CURSIVE`, full anchor | **19.3%** | **0.0%** | **56.4%** |

(Still 100% correctly classified as thiya in both cases, just much lower
margin with Hershey.)

`thiya_hershey_vs_current.png` shows why: the Hershey-rendered line has huge
vertical spikes — ascenders and approach strokes swinging far above the
writing line, letters overlapping/colliding, baseline jumping around. The
current hand-coded `_CURSIVE` was deliberately authored so **every letter
enters and exits near the same height** (`_CX_IN, _CX_OUT = 0.0, 0.30` near
the top of the `_CURSIVE` block) — that convention is what lets
`SynthesizeText`'s join logic (`wantEntryY` computed from the previous
letter's exit point, ~line 573-577, consumed by `_GlyphSource`'s
`prevExitY` parameter) chain letters smoothly. A normal flowing script font
does **not** keep every letter's entry/exit at one fixed height by design —
that variation is part of what makes a real alphabet look natural when a
human free-hands it — but it's exactly what breaks this project's joining
logic, which expects a shared connection height.

**The raw letterforms are a real upgrade; the naive port is not enough.**
Global baseline/x-height normalization (what `build_hershey_cursive.py` does)
is not sufficient — each glyph's own entry and exit point needs to be
pulled to a consistent height, the way real extracted glyphs carry
`entryY`/`exitY` fields that the rest of the pipeline already uses
(`BuildStyleProfile.py`'s glyph dicts have `entryY`/`exitY`/`lead` keys; see
e.g. `ExtractLineGlyphs`'s output shape, or `_GlyphSource`'s `pack()` helper
in `SynthesizeHandwriting.py` which reads `g.get('lead',0.0)`).

## 5. Unrelated but relevant gotcha found along the way

`BuildStyleProfile.py`'s own default `TEXT_WEIGHTS` points at
`paper_cnn_bilstm_ctc_best.npz`, which **does not exist on disk**. For
testing, the correct checkpoints that do exist and are known-good are
`paper_cnn_bilstm_ctc_joint_best.npz` (general/production) and
`paper_cnn_bilstm_ctc_personal_best.npz` (used for personal-author alignment
in `BuildAll`, see `BuildStyleProfile.py` ~line 1118). Also,
`Evaluate.TEXT_WEIGHTS`'s auto-resolution currently prefers
`paper_cnn_bilstm_ctc_scratch_best.npz` if present, which is a much weaker,
undertrained from-scratch checkpoint (~0.33 char accuracy on a 4-line sanity
check vs. ~0.98 for `joint_best`, per an earlier memory note
`cleanpipeline-broken-imports.md`) — don't trust default-resolved text-model
paths in test scripts without checking which file they actually picked;
pass `checkpoint_path=` explicitly as done in the scripts in this folder.

## 6. Suggested next steps, roughly in order of expected value

1. **Decide the cheap, already-validated fix first**: detect very-high-
   connectedness authors (or more directly, authors whose own glyph-library
   `priorD` distribution is this bad — e.g. median > ~0.3 or %clean < ~10%)
   and force near-full anchor for them by default, rather than relying on the
   same `_SwapProb` probability curve tuned for authors whose own glyphs are
   worth partially trusting. This alone is confirmed to raise p(thiya) from
   ~63-76% to ~94% and char accuracy from ~24-32% to ~34%, with no new
   letterform work needed. Where to change: `SynthesizeText`'s call into
   `_GlyphSource`, or a new per-profile flag set at profile-build time in
   `BuildStyleProfile.py`.
2. **If pursuing the better cursive anchor**: give `build_hershey_cursive.py`
   (or a successor) a per-letter entry/exit normalization pass — e.g. find
   each letter's first/last stroke point, and either (a) rigidly rotate/
   translate so entry lands at a fixed target point the way `_CURSIVE` does,
   or (b) extend/trim the connector stub to meet a fixed height — then rerun
   `test_hershey_cursive_swap.py` to see if it closes the gap with (and
   hopefully beats) the hand-coded set. This is the natural continuation of
   §4, not a restart.
3. **Not investigated**: whether the join/geometry logic in `SynthesizeText`
   itself (independent of glyph source) is capping legibility even under
   full anchor (Finding 2 in §3) — word accuracy stayed under 10% in every
   condition tested. Worth a separate look if #1 and #2 don't move word
   accuracy much.
4. Re-run `priord_distribution.py` and `show_thiya_cuts.py` against any other
   very-cursive authors added later, as a quick health check before
   assuming the cross-author prior (built from all 13 pooled together) isn't
   itself biased toward less-joined hands — not ruled out as a secondary
   contributing factor, just shown to be smaller than the segmentation
   problem for thiya specifically (§2's visual evidence of genuinely wrong
   cuts, not just atypical-but-valid ones).

## 7. How to reproduce anything above

All scripts assume cwd = `Software/CleanPipeline/` and the line-image cache
already built (`Software/CNN/NOGIT/line_cache_authors10/`, ~99 IAM pages,
one-time cost, already built as of this investigation).

```bash
cd Software/CleanPipeline
python NOGIT/experiments/cursive_anchor/show_thiya_cuts.py          # cut-boundary viz, 10 lines
python NOGIT/experiments/cursive_anchor/priord_distribution.py      # priorD stats + histogram
python NOGIT/experiments/cursive_anchor/test_thiya_anchor_levels.py # own/default/full-anchor sweep
python NOGIT/experiments/cursive_anchor/build_hershey_cursive.py    # rebuild the Hershey conversion
python NOGIT/experiments/cursive_anchor/test_hershey_cursive_swap.py # hand-coded vs Hershey anchor test
```
