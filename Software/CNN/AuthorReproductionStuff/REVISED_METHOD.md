# Revised author-reproduction method (image output only)

**Goal:** given a text string and an author id (150, 151, 152, 153, 154, 155,
384, 551, 552, 588), produce a **PNG line image** of that text in that
author's hand that is (a) legible and (b) recognisably that writer -- for
*every* author, including the heavily cursive ones (151, 153, 154, 155, 552)
that the current glyph-library pipeline breaks on.

No G-code, no motor code, no gantry. The output is an image and nothing else.

---

## 1. Why the current pipeline fails on cursive

From `EvaluateStyle.py` on the committed `dffeab0` synthesiser:

| Signal | Result | Reading |
|---|---|---|
| Feature agreement (slant, word-gap, ink/x-height) | within ~10-20 % for all 10 authors | **the low-frequency style channel is fine** |
| Visual legibility, print hands (150, 152, 384, 588) | readable | glyph library is usable when letters are separate |
| Visual legibility, cursive hands (151, 153, 154, 155, 552) | dissolves into scribble mid-word | **the high-frequency channel is destroyed** |

Two independent papers explain this exactly:

- **SDT** (Dai et al., CVPR 2023, *Disentangling Writer and Character Styles
  for Handwriting Generation*) -- writer identity lives in **low-frequency**
  features (slant, aspect ratio, x-height); the distinctive per-letter shape
  and **how letters join** live in **high-frequency** features (stroke
  length, curvature). They use two separate contrastive heads for the two
  bands.
- **One-DM** (Dai et al., ECCV 2024, *One-Shot Diffusion Mimicker*) -- "the
  high-frequency information of the individual sample often contains distinct
  style patterns (e.g., character slant and letter joining)". They add a
  dedicated high-pass branch to the style encoder.

Our CTC-forced-alignment + Zhang-Suen skeleton extraction reconstructs the
low-frequency envelope well (hence the good feature agreement) but the
per-line cut noise wipes out the high-frequency join information -- which on
a cursive hand *is* the hand. No amount of tuning the extractor fixes this;
the information is gone before synthesis starts.

A third paper closes the argument:

- **DiffInk** (arXiv 2509.23624, Sept 2025) -- character-level pipelines
  (SDT included) suffer "cumulative error propagation" and "unnatural
  stitching artifacts at character boundaries"; modelling the whole line
  instead beats them by **30 points of writer-style accuracy**. Our glyph
  library *is* the character-level decomposition they moved away from.

**Conclusion:** stop hand-rolling the glyph shapes. Keep the parts that work
(the layout parameters, the legibility gate, the eval harness); replace the
glyph-assembly core with a learned generator that has seen cursive joins at
IAM scale.

---

## 2. The revised method

Seven stages. Stages B, E, F, G, H reuse code we already have. Only stage D
is new, and it is *inference against a frozen published model* -- no training
on our hardware.

```
  text ─► [B] content prep ─┐
                            ├─► [D] neural word generator ─► [E] legibility ─► [F] style ─► [G] line
  author ─► [C] style enc ──┘        (K candidates/word)        gate (CTC)      re-rank      compose ─► PNG
                                                                   │
                                                          archetype fallback
```

### B. Content preparation -- "visual archetypes" (VATr, CVPR 2023)

Split the target text into words. Render **each word as a printed-font
image** (our existing `_FB` single-stroke anchor font, or a plain sans) at
the generator's expected input height.

This printed rendering is used twice: as the *content signal* for the
generator, and as the *legibility reference* + *graceful-degradation
fallback* in stage E. VATr's whole point is that feeding rendered glyph
images (not one-hot class ids) as the content channel makes rare / hard
characters degrade toward a legible prototype instead of toward noise. This
is a principled version of our current hard `legibilityLambda` anchor swap.

### C. Style encoding (DiffusionPen ECCV 2024 + One-DM ECCV 2024)

Per author, collect **1-5 real reference line crops** from that author's
**held-out** IAM pages (never the pages used to fit anything).

Two style representations, both needed:

1. **Generator style condition** -- whatever the chosen backend wants. One-DM
   needs a single reference crop plus its high-pass (Laplacian) component.
   VATr++ needs ~15 word crops.
2. **Re-ranking embedding** -- a metric-learned style vector. **Reuse
   `NOGIT/weights/author_shape_10_weights.pt`** (the shape-only writer-ID
   model from `TrainAuthorShape.py`, 98.9 % on real held-out crops). It is
   already a contrastive-style encoder over stroke-normalised lines -- the
   domain our synthetic output lives in. Take its penultimate-layer
   activation as the embedding; store a per-author centroid over the real
   held-out crops.

   DiffusionPen's finding: train the style encoder with *metric learning +
   classification together*, not classification alone. `TrainAuthorShape.py`
   is close; if we retrain, add a triplet/NT-Xent term. Optional -- the
   current weights are good enough to start.

### D. Neural generator (NEW -- frozen pretrained model, inference only)

Backend interface (`htg_backend.py`), so the model is swappable. A backend
is either **line-level** (`generate_line(text, style, k, seed)`) or
**word-level** (`generate(word, style, k, seed)`); the orchestrator adapts.

**Primary backend: DiffBrush** (`github.com/dailenson/DiffBrush`, ICCV 2025,
MIT, IAM-trained -- the same lab's follow-up to One-DM).
- **Generates the whole line in one diffusion pass** -- no per-word
  composition, so the hand stays consistent across the line (this is the
  DiffInk lesson, solved by the model instead of by stage G).
- "Content-decoupled style learning" -- a two-head (vertical / horizontal)
  style encoder that transfers writer style more strongly than One-DM's
  single high-freq branch. Verified on our authors: 150/384 print, 151/153
  slanted cursive, 155 rounder semi-cursive all come through recognisably.
- Full a-z/A-Z/0-9/punctuation content set (One-DM's is a restricted
  ~80-char list that mangles apostrophes / long words).
- Style ref = **one wide strip (>512 px)** of the author's real writing --
  we use their held-out IAM line, resized to h=64. Works for all 10
  authors (One-DM's IAM64 word-crop set is missing 384).
- CPU: ~35-45 s per line at 25-30 DDIM steps. Batch, not interactive.

**Alternative: One-DM** (`github.com/dailenson/One-DM`, ECCV 2024, MIT).
Word-level. Kept as a backend but superseded by DiffBrush -- its per-word
output has to be stitched into a line (stage G), which drifts in size and
mixes print/cursive candidates. Good standalone word quality; weaker as a
line.

**Not used: DiffusionPen** (5-sample style, metric-learned encoder) and
**VATr++** (fast GAN). Both viable drop-ins if DiffBrush ever needs
replacing; not needed now.

Generate **K = 2-4 candidates** (different noise seeds).

### E. Legibility gate -- keep `SynthesizeLegible`'s logic verbatim

For each candidate word image: run it through **`paper_cnn_bilstm_ctc_best.pt`**
(our `PaperCRNN` CTC reader) and compute `CER(pred, target_word)`.

- Keep candidates with `CER <= tau` (start `tau = 0.15`).
- If **every** candidate for a word fails: emit the **printed archetype** for
  that word, slant/size-matched to the author (stage B image, sheared to
  `slantDeg`, scaled to `xHeightPx`). Never emit an unreadable word. This is
  the current repair-pass philosophy, unchanged.

### F. Style re-rank

Among the legible candidates for a word:

```
score = w_leg * (1 - CER) + w_sty * cosine(shape_embed(cand), author_centroid)
```

Start `w_leg = 0.5`, `w_sty = 0.5`. Pick the argmax. This is best-of-N scored
by the frozen reader + the frozen style judge -- the same pattern every
paper's demo pipeline uses, and the same pattern our `SynthesizeLegible`
already uses for legibility alone.

### G. Line composition -- our style profile becomes the layout model

This is where the DiffInk "the line matters, not the isolated glyph" lesson
is handled *without* needing a line-level model: the generator gives us
good words, our **measured per-author parameters** place them.

From `NOGIT/StyleProfiles10/<author>.json` (already built):
- `wordSpaceXh` -> gap between words
- `xHeightPx`, `slantDeg` -> target scale (slant is already baked in by the
  generator; use it only to sanity-check / to shape the archetype fallback)
- add small per-word **baseline drift** and a slight whole-line **slope**
  (both are cheap noise, and both are cues a plotter-flat line lacks)

Render one constant stroke width (`uniformInk`, already implemented) or
sample ink from `inkLevel` -- it is an image, so either is free.

Output: a single grayscale line PNG, identical format to
`EvaluateStyle.BuildComparisonSheet` so the eval sheet drops in unchanged.

### H. Evaluation -- revised

`evaluate_v2.py`, backend-agnostic, mirrors `EvaluateStyle.RunFullEvaluation`:

| Metric | Tool | Note |
|---|---|---|
| Real-vs-synth sheet | `BuildComparisonSheet` | unchanged |
| CER (legibility) | `PaperCRNN` | unchanged |
| Writer-ID accuracy | **`author_shape_10_weights.pt`** | **shape-only, NOT the ink model** -- the ink model collapses to ~12 % on uniform-width output (documented in `README.md`); it is not a valid judge here |
| Style cosine | shape embedding vs held-out real centroid | new, continuous, more informative than top-1 ID |
| HWD (optional) | `pip install hwd` (DiffusionPen authors' Handwriting Distance) | perceptual style metric, better than FID for handwriting |

Targets: shape-ID >= 85 % of an author's synth lines classified as that
author; CER <= the `PaperCRNN` CER on that author's *real* held-out lines
(i.e. "as readable as the real thing, to our own reader").

---

## 3. Why this is the secure choice

- **No training on our hardware.** The one heavy component is a frozen,
  peer-reviewed, MIT-licensed model with released weights. We run it in
  `eval()` and never touch its parameters.
- **In-distribution.** Authors 150-588 are IAM writers; IAM-trained
  generators have seen them. This is the easiest case for a learned model,
  not the hardest.
- **Every failure mode has a deterministic fallback** (the printed archetype
  word). The pipeline cannot emit scribble.
- **Reuses everything that already works** -- CTC reader, shape-ID judge,
  style profiles, comparison-sheet eval. The only thing discarded is the
  glyph-library assembly core, i.e. the part that was failing.
- **Cursive is handled by learned priors**, plus One-DM's explicit
  high-frequency style branch that targets our exact failure channel.

---

## 4. No-download fallback (Tier 0) -- strict improvement, ships today

If we don't want the pretrained dependency yet, these four changes to the
existing synthesiser remove the worst cursive breakdowns with **no new
infra**:

1. **Line-level assembly, not glyph-level** (DiffInk). Synthesise the whole
   line in one trajectory pass; let the CTC gate score the line, not each
   glyph in isolation.
2. **Soft archetype blend** (VATr). Replace the hard `legibilityLambda`
   swap with a per-glyph interpolation between the author's extracted variant
   and the slant-matched print archetype, weight = f(CTC confidence on that
   glyph) -- continuous, not a cliff.
3. **Frequency-preserving substitution** (SDT / One-DM). When a glyph must be
   substituted, keep the author's low-frequency envelope (bbox, slant,
   x-height, advance -- the channel we measure well) and replace only the
   high-frequency skeleton path.
4. **Best-of-N re-rank** by shape-ID cosine + CER (we already do CER alone).

Expected: closes the "therefore -> AttoraRanah" class of failure; will not
reach One-DM quality.

---

## 5. Runtime (measured, CPU-only, this machine)

| Backend | Per line | k=2, 10 authors | Use |
|---|---|---|---|
| `legacy_synth` | ~3-8 s | ~3 min | iteration, baseline |
| `diffbrush` (25 steps) | ~35-45 s | ~20-25 min | **primary** |
| `one_dm` (25 steps, k words) | ~2-3 min | ~40 min | alternative |

Model load is a one-off ~15-17 s. Run `evaluate_v2` in the background.

---

## 6. Ethics / scope

Reproducing IAM-dataset writers is standard research practice -- every paper
cited here does exactly this on IAM. This pipeline is for **dataset-author
reproduction in a research report**, not impersonation of a named
individual. Do not point it at signatures or a real person's private
correspondence. Keep reference material to the IAM corpus.

---

## 7. Build status -- INTEGRATED

Everything lives in `AuthorReproductionStuff/ReproduceV2/`. It also carries
the path fix the moved folder needs (`_env.py`), so it runs from anywhere.

| File | State |
|---|---|
| `_env.py` -- path setup + re-pins the stale `BuildStyleProfile` NOGIT paths | done |
| `htg_backend.py` -- `StyledHTGBackend` interface + backends | done |
| `reproduce_v2.py` -- orchestrator (line-level pick-best for DiffBrush/legacy; per-word best-of-K + compose for One-DM) | done, runs |
| `evaluate_v2.py` -- backend-agnostic real-vs-synth sheet + CER + shape-ID + style | done, runs |
| `diffbrush_infer.py` -- CPU single-process DiffBrush wrapper | **done, runs -- primary** |
| `onedm_infer.py` -- CPU single-process One-DM wrapper | done, runs -- alternative |
| `DiffBrushBackend` -- frozen DiffBrush, whole-line, ~40 s/line | **done, runs** |
| `OneDMBackend` -- frozen One-DM, word-level | done, runs |
| `LegacySynthBackend` -- wraps the current synthesiser (Tier 0 baseline) | done |
| `PrintArchetypeBackend` -- printed-font last-resort floor | done |
| `VATrBackend` -- fast GAN alternative | stub (not needed) |

Run:
```
python reproduce_v2.py "some text" 153 --backend diffbrush --k 3
python evaluate_v2.py --backend diffbrush --k 2       # full 10-author sheet
python evaluate_v2.py --backend legacy_synth --k 6    # baseline for comparison
python compare3.py 153 155 150 384                    # real | glyph | one-DM spot check
```

### What was downloaded (all under `NOGIT/_vendor/`, gitignored, ~3 GB)

| Path | What | Source |
|---|---|---|
| `DiffBrush/` | repo + `files/unifont.pickle` | `github.com/dailenson/DiffBrush` |
| `ckpt/DiffBrush-ckpt.pt` | UNet weights (1.17 GB) | Google Drive `1EWzBmLt...` via `gdown` |
| `One-DM/` | repo (code only) | `github.com/dailenson/One-DM` |
| `ckpt/onedm_models/One-DM-ckpt.pt` | UNet weights (1.25 GB) | Google Drive `10KOQ05...` via `gdown` |
| `ckpt/sd15/vae/` | SD-1.5 `AutoencoderKL` (~320 MB) | HF `stable-diffusion-v1-5/stable-diffusion-v1-5`, `vae/` only |
| `ckpt/unifont.pickle` + `ckpt/style/<author>/` | One-DM content symbols + IAM64 style crops | One-DM `English_data.zip` |

If disk is tight, `ckpt/onedm_models/One-DM-ckpt.pt` (1.25 GB) can go --
DiffBrush is the primary and doesn't need it. The One-DM training-only
checkpoints (`RN18_class_10400.pth`, `vae_HTR138.pth`, `English_data.zip`)
were already deleted.

### CPU port -- what the `*_infer.py` wrappers do instead of `test.py` / `generate.py`

- build `UNetModel` + `Diffusion` directly, no `torchrun` / `dist` / config file
- `torch.load(..., map_location="cpu")`, `.to("cpu")` throughout
- monkey-patch `torchvision.models.resnet18` to `weights=None` so construction
  doesn't download ImageNet weights (overwritten by the checkpoint anyway)
- DiffBrush only: neutralise the hardcoded `.cuda()` on `Proxy_Anchor`'s
  train-only proxy params during construction
- build the style tensor from our own crop (One-DM: one word crop + its
  Laplacian; DiffBrush: one wide held-out-line strip) instead of the fixed
  `Random_StyleIAMDataset` / `IAMGenerateDataset`
- `--generate_type` is irrelevant: we pass our own text + style crop

---

## 8. Papers pulled from

| # | Paper | What we take |
|---|---|---|
| 1 | **DiffBrush**, Dai et al., ICCV 2025 | **the primary generator** -- whole-line diffusion, content-decoupled two-head style encoder |
| 2 | SDT, Dai et al., CVPR 2023 | low-freq = writer, high-freq = joins; frequency-preserving substitution (Tier-0) |
| 3 | One-DM, Dai et al., ECCV 2024 | alternative generator; one-shot style; high-pass style branch |
| 4 | DiffInk, arXiv 2509.23624, 2025 | line matters, not the isolated glyph -- why DiffBrush > One-DM here |
| 5 | VATr / Visual Archetypes, Pippi et al., CVPR 2023 | rendered-glyph content channel; graceful degradation; archetype fallback |
| 6 | DiffusionPen, Nikolaidou et al., ECCV 2024 | metric-learning + classification style encoder; HWD metric |
| 7 | Emuru, Pippi et al., CVPR 2025 | (considered) font-only training route if we ever need an in-house generator |
| 8 | Díaz et al. survey, Pattern Recognition 2025 | off-to-image is the mature direction; standard metric set (FID/HWD/DTW/CER/writer-ID) |
| 9 | DSD, Kotani et al., ECCV 2020 | closest prior art; per-character + global style split -- design reference |
| 10 | GANwriting / SmartPatch, Kang et al. | patch discriminator idea for letter-join realism (informs re-rank) |
| 11 | Differentiable Physical Rendering, arXiv 2608.03198, 2025 | (parked -- only relevant if we return to trajectory output) |
