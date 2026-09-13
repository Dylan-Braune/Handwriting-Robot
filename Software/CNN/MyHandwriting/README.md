# MyHandwriting -- reproducing an arbitrary writer's hand

Three independent ways to reproduce new text in a chosen writer's style. Each
is **trainable on any handwriting** -- point `pages/` at a different writer.
Here they are all trained on the student's own 3 lab-book pages
(`pages/page{1,2,3}.jpg` + `_labels.txt`, transcribed line by line).

| | Method | How it works | Output | On the SBC? | First-principles? |
|---|---|---|---|---|---|
| 1 | `method1_exemplar.py` | stores real image fragments the writer wrote (words, letter-pairs, letters) via CTC forced-alignment, tiles the target sentence from them, stitches seams | writer's own ink pixels | yes | yes |
| 2 | `method2_features.py` | skeletonises every letter into an ordered **pen path**, keeps denoised variants + measures style numbers (slant, x-height, stroke width, spacing, connectedness), **draws** new text from the paths | pen trajectory + render | yes | yes |
| 3 | `method3_library.py` | pretrained DiffBrush diffusion model, conditioned on one clean strip of the writer's hand | raster image | no (needs GPU + `NOGIT/_vendor/`) | no -- reference only |

## Result on the student's hand (novel sentence)

`out/all3_novel.png` -- "the quick brown fox jumps over the lazy dog":

- **Method 2 is the best**: every word rendered, clean, uniform stroke,
  clearly the student's print style, and it produces the pen trajectory the
  gantry needs. This is the method that matches the project proposal
  (FU 2.5 "character image -> trajectory map").
- **Method 1** is legible and uses the student's literal ink, but only 3
  pages -> gaps for letters/words the student never wrote (`jumps` -> `umps`).
- **Method 3** captures the look but garbles content on an unseen writer
  from a noisy strip -- it is the "yes this is possible" upper bound, not a
  deliverable.

## Run

```
python method1_exemplar.py --build            # build fragment library from pages/
python method1_exemplar.py "any new text"

python method2_features.py --build            # extract style profile from pages/
python method2_features.py "any new text"
python method2_features.py --iam              # also build the 10 IAM authors

python method3_library.py "any new text"      # needs NOGIT/_vendor/ (see
                                              # ../AuthorReproductionStuff/REVISED_METHOD.md)
```

Libraries/profiles are written to `lib/`, renders to `out/`.

## Shared building blocks (all the student's own code, in the repo)

- `SegmentPage.py` -- photo -> deskewed line crops
- `BuildStyleProfile.py` -- CTC forced-alignment (Viterbi), Zhang-Suen
  skeletonisation, polyline tracing, Douglas-Peucker simplification, style
  measurement
- `TrainText.py` -- the CNN-BiLSTM-CTC recogniser used for alignment + a
  legibility check
