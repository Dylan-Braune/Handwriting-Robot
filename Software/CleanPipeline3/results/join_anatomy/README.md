# Where a join attaches to a letter

Three runs of `results/all13_sentences/make.py`, 13 writers x 2 sentence
lengths, curated profiles where one exists, nTries 14, seed 23.

## The question

Did the synthesiser know where a writer's pen actually leaves a letter, or
was it guessing?  Neither, as it turned out.  Every glyph stores
`entryX/entryY/exitX/exitY`, but `style._TrimLigatureTails` defines them as

    entry, exitPt = strokes[0][0], strokes[-1][-1]

the first point of the first stroke and the last point of the last one --
stroke point **ordering**, not handwriting anatomy.  Synthesis did not even
read those fields when drawing: it ran the connector between `chain[-1]`
and `body[0]`, the ends of whichever stroke was widest in x.  The stored
field agrees with the point actually used at a median difference of 0.000
x-heights, so it was no better a reference.

Two consequences, measured over all 25 profiles (6467 glyphs):

| defect | cause | scale |
|---|---|---|
| a t or f **crossbar** carries the join | widest in x picks the bar over the stem | 269 glyphs |
| `l`, `i` leave from the **top** | strokes are ordered left-to-right, which for a near-vertical letter is decided by pixel noise | 9% of glyphs have a body x-span under a quarter of their height |

33% of exit points and 42% of entry points sat above 0.7 x-heights.

## What was changed

`_BodyStroke` in `synthesise.py`, behind `JOIN_ANATOMY`:

- a stroke flatter than `JOIN_FLAT_MAX_ASPECT = 0.35` (height/width) is a
  bar, and may not carry the connection whatever its x-span.  Dots never won
  on width, so they were already safe.
- below `JOIN_VERT_MIN_XSPAN_XH = 0.25` the left-to-right ordering is noise,
  so the stroke is walked to end at its **low** end instead.

Crossbar-as-join went 269 -> 0; `l` leaving from its top 28/268 -> 18/268;
exits above 0.7 xh 33.4% -> 30.1%.

## What was tried and rejected

Refusing a join that would attach above x-height (`JOIN_MAX_ATTACH_XH`,
`JOIN_NO_ENTRY_ABOVE_XH`).  The glyphs separate cleanly on that line --
entry height median b 1.42, l 1.47, t 1.30 against v 0.84, w 0.75, o 0.70,
no short letter above 1.08 at p90 -- so 1.15 looked like a principled cut.

It measured worse and was turned off.  Both constants sit at 99.0, higher
than any glyph reaches, so neither test fires.

## Results

| run | file | char both | long | short | writer-ID | worst page |
|---|---|---|---|---|---|---|
| baseline | `RESULTS_prejoin.txt` | **87.1%** | 86.3 | 87.8 | 22/26 | 60.0 |
| + attach gates | `RESULTS_joingates.txt` | 85.5% | 86.2 | 84.9 | 22/26 | 57.8 |
| structural only (**shipped**) | `RESULTS.txt` | 86.7% | 86.2 | 87.2 | **24/26** | 55.6 |

The gates cost 1.2 points and fell hardest on the lightly joined hands,
151 by 10.6 and dylan by 7.4.

Against the baseline the shipped version is -0.4 points of character
accuracy -- 5 writers up, 6 down, 2 unchanged, which on a single seed is
not a difference -- while writer identification goes 22/26 to **24/26**,
short sentences to 13/13, and the five photographed hands to 5/5 on both
lengths.  151, 552 and yeukita each gain a page; 588 loses one.  Two pages
on one seed is a thin margin and has not been repeated.

## The defect this did NOT fix

The strange diagonals running to the tops of ascenders are **not
connectors**.  Rendering with every ligature disabled (`JOIN_PROB_CAP = 0`)
leaves them all in place -- see `thiya_nojoin.png` against
`thiya_before.png`.

They are the letters themselves.  thiya's `l`, straight from the profile:

    stroke0  (0.00, 0.17) ... (0.49, 0.33)      a low fragment at the baseline
    stroke1  (0.18, 2.28) ... (1.33, 0.17)      ascender tip, down to the right

`connL` is true, so `_TrimLigatureTails` removed the up-leg as if it
belonged to the previous letter.  In real cursive that up-leg *is* the
letter's beginning, so what is stored is half an `l` hanging from a point
in mid-air.  `entryY` is 2.143, the ascender tip.

Fixing that is a cut-time change in `style.py` -- keep or regrow the
up-leg when `connL` is set and the body starts above x-height -- and like
the crossbar fix in `_BarOwner` it needs the profiles rebuilt before any
of it reaches a page.

## Files

| file | what it is |
|---|---|
| `<writer>_before.png` / `_after.png` | the structural change, four writers |
| `thiya_nojoin.png` | every ligature disabled, proving the diagonals are glyph ink |
