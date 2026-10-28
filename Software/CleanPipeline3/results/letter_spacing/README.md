# Letter spacing and when to join

`"Another method to drastically lower the number of connections needed"`, owen's
curated profile, seed 23, rendered at 9 px/mm.

## What was wrong

A glyph's stored `advance` is the width of the **cell it was cut from**, not of
its ink. Where the cut took in part of a neighbour the cell is far too wide, so
the next letter is pushed away from it. Measured over the five photographed
writers the median trailing gap is 0.04 to 0.13 x-heights, but the tail runs to
**1.38** — wider than a letter.

Separately, the join decision asked only what the two letters were and how
cursive the hand is. It never asked how far apart they had ended up, so a
connector was drawn across any distance at all. A long gap plus a mandatory
connector is a long flat dash across empty paper.

## What changed

- the gap after each letter is held to the writer's own 90th percentile, hard
  capped at `MAX_LETTER_GAP_XH = 0.30`. Only the top of the range is taken from
  the writer; the floor stays where it was.
- two letters more than `MAX_JOIN_GAP_XH = 0.80` x-heights apart are not joined.
  Measured afterwards, the drawn joins top out at exactly 0.80, so it binds.

| writer | gap cap used |
|---|---|
| abhinav | 0.222 |
| owen | 0.226 |
| yeukita | 0.266 |
| dylan | 0.298 |
| thiya | 0.300 (its own p90 is 0.41, so the hard cap binds) |

## The images

| file | what it is |
|---|---|
| `before.png` | the original advance formula |
| `after.png` | floor raised to the writer's p25 as well — **rejected** |
| `caponly.png` | cap only, floor untouched — **shipped** |
| `cap045/030/022/015.png` | the cap swept; 0.45, 0.30 and 0.22 are identical for owen because his own p90 of 0.226 binds first |
| `final_owen.png`, `final_thiya.png` | the shipped setting |

Reading them: `before` splits `drasti cally` and breaks `connections` into four
pieces, `co nne ctio ns`. `caponly` joins `drastically` and renders
`connections` as one word. The rejected `after` fixed those too but pushed
apart letters that were already correct, giving `num ber` and `t he` — raising
the floor cost more than it bought. A tighter cap of 0.15 closed `the` but
broke `connections` and `needed` instead.
