# Handwriting synthesis techniques — survey (Sept 2026)

Research pass across the field, evaluated specifically against this
project's constraints: output must be a **pen trajectory** (for G-code, not
a picture), style data is **~10 offline scanned pages per author** (no real
online stroke recordings), compute is **CPU-only**, and the goal is
**legible AND recognizable as the author**. See `REWRITING_STATUS.md` for
where the shipped rule-based rewriter (commit `dffeab0`) currently stands
(85.6% char legibility, style recognizable for 4/10 authors) and
`experimental/` for the two from-scratch RNN attempts already tried this
session.

## The technique families

### 1. RNN sequence generation — Graves 2013 (`arXiv:1308.0850`)
3-stacked LSTM + learned Gaussian attention window over the text + mixture-
density output; style via **priming** (run the net over a real stroke+text
sample first). This is what `sjvasquez/handwriting-synthesis` implements
and what the user's linked video demos. **Already tried twice this
session** (`experimental/hw_rnn.py`, `experimental/graves_rnn.py` — the
second is a faithful port, verified line-by-line against the reference
repo). Conclusion: architecture is right, but it needs the ~12,000 *real*
recorded online sequences of IAM-OnDB and tens of thousands of GPU
minibatch steps to converge to legible letters. We have ~3,700 sequences
*reconstructed by skeletonizing offline scans* (no real pen order/speed)
and CPU-only training — two orders of magnitude short. Their pretrained
checkpoint is TensorFlow 1.6 (`.data`/`.index`/`.meta`) and cannot be run or
parsed without TensorFlow, which has no wheel for this environment's Python
3.14.

### 2. GAN-based offline generation — ScrabbleGAN, GANwriting, HiGAN/HiGAN+
CNN encodes a handful of a writer's real word images into a style vector;
a GAN generator renders new text in that style **as a raster image**.
Multi-shot (GANwriting: ~15 reference images). Mature, well-documented,
several PyTorch implementations exist. **Output is a picture, not strokes**
— would need a vectorization step afterward, which re-introduces exactly
the cursive-cut problem this project has been fighting (`ExtractLineGlyphs`
mangling ligatures). Lower priority for this use case unless paired with a
strong vectorizer (see #5).

### 3. Transformer-based few-shot — HWT, VATr / VATr++, WriteViT
Handwriting Transformers (HWT, ICCV 2021) and VATr (CVPR 2023, "Visual
Archetypes") use self-attention over a few style-reference images +
GNU-Unifont glyph archetypes as content queries, decoding to a **raster**
image. VATr++ improves rare-character generalization. Same limitation as
#2: image output, needs vectorization. Well-suited if the deliverable were
a printed page; not directly to a plotter.

### 4. Style-Disentangled Transformer — **SDT** (`dailenson/SDT`, CVPR 2023) — best fit found
Directly generates an **online stroke sequence** — `(x, y, pen-up)` points,
exactly `WriteGCode.py`'s input — conditioned on a **few offline OR online
reference samples** of the target writer, via two contrastive objectives
that separate writer-level style from character-level style. Has a
**pretrained English checkpoint** (also Chinese, Japanese), PyTorch,
Python 3.8, code on GitHub with an inference tutorial for custom styles.
This is the one technique found that matches this project's exact shape:
vector output, few-shot from *offline* images, pretrained already on
Latin/English handwriting. Feasibility check done: this environment can
`pip install` real packages (verified) even though the system Python is
3.14 (no TF wheel) — a secondary Python 3.13 install is present and
installable-into, which is far more likely to satisfy SDT's pinned deps
than fighting 3.14. Not yet attempted here; would need downloading their
released weights (Google Drive/Baidu links in the repo) and adapting our
authors' real IAM line/word crops as the style reference set.

### 5. Diffusion-based — DiffusionPen, One-DM, WordStylist, DiffInk
Iterative denoising conditioned on a style encoding + target text.
**One-DM** (2024) is notable for working from a **single** reference image
(most others want ~10-15) and explicitly extracts slant/joining from
high-frequency image detail — a good match for our thin per-author data,
but its output is a **raster image**. **DiffInk** (2025,
`arXiv:2509.23624`) is the one diffusion method claiming direct **online**
handwriting output via a latent diffusion transformer — newer, less
battle-tested, no confirmed public pretrained weights found in this pass.
**Trajectory-recovery diffusion** (`arXiv:2607.03422`, 2025) targets the
adjacent problem directly: image-conditioned diffusion that recovers a
stroke trajectory *from* an offline image, i.e. a much better version of
our own `Skeletonize`/`TracePolylines`/CTC-cut pipeline (see #6).

### 6. Offline → online trajectory recovery (fixes *our* actual bottleneck)
This is a distinct problem from generation: given a real offline image,
recover the pen order that drew it. **TRACE** (ICDAR 2021,
`arXiv:2105.11559`) is a CNN+LSTM+attention model trained end-to-end on
*whole lines* of arbitrary width with a differentiable DTW alignment loss
— built for exactly the "cursive line → per-character/point stroke order"
problem that `BuildStyleProfile.ExtractLineGlyphs` currently solves with
CTC-alignment + heuristic ink-column cuts (and which `REWRITING_STATUS.md`
names as the actual ceiling on the 10 authors' style fidelity). No public
code repo was found for TRACE in this pass (paper only). The 2025
diffusion trajectory-recovery paper is the newer entrant in the same
niche. **This family is the most direct fix for the documented root
cause** — better extraction feeds the *existing, working* rule-based
synthesizer better glyphs, rather than replacing the synthesizer.

### 7. Unsupervised stroke discovery for robot arms — **CalliRewrite** (ICRA 2024, `LoYuXr/CalliRewrite`)
Built for exactly this deployment shape: a low-cost robot arm reproducing
calligraphy from **image references only**, no labeled stroke-order data.
An unsupervised LSTM proposes a coarse stroke segmentation from the image,
then reinforcement learning refines it into a physically executable
trajectory for the specific tool/arm. Code is public. Caveat: demonstrated
on Chinese calligraphy glyphs (single characters), not English cursive
lines — the method (unsupervised order discovery + RL trajectory
refinement) would need real adaptation, not a drop-in port, but it is the
closest published system to "robot arm + offline images + no online
ground truth."

## Ranked recommendation

| # | Approach | Output | Fits our data? | Effort to try | Risk |
|---|---|---|---|---|---|
| 1 | **SDT**, pretrained English | strokes | yes — offline style refs | medium (env + weight download) | pretrained on generic writers, not our 10 specifically — may need fine-tuning or accept generic-but-legible-and-stylish output |
| 2 | **Trajectory-recovery model** (TRACE-like/diffusion) feeding our existing synthesizer | strokes (upstream fix) | yes — this is what it's for | high (no off-the-shelf code found; would mean training one) | best long-term payoff, most engineering |
| 3 | Keep tuning the rule-based system (current) | strokes | yes | low | already plateaued this session per `REWRITING_STATUS.md` |
| 4 | GAN/Transformer/most-diffusion (raster) | image | needs vectorization after | medium-high | reintroduces the cursive-cut problem downstream |
| 5 | CalliRewrite-style unsupervised RL | strokes | partial (wrong script/scale) | high | biggest architecture mismatch |
| — | Graves RNN from scratch | strokes | no — needs real online data | *(done, 2x)* | confirmed insufficient data/compute here |

**My recommendation: try SDT next.** It is the only technique surveyed that
(a) outputs vector strokes directly, (b) has a working pretrained English
model so no from-scratch training is needed, (c) is designed to condition
on offline reference images — which is exactly what we have for the 10
authors — and (d) is checked feasible to actually run in this environment
(PyTorch, installable Python, confirmed network access for `pip`/model
downloads). Say the word and I'll set up the environment and attempt
inference with our authors' real line crops as style references, comparing
its output against the shipped `dffeab0` rewriter on the same legibility +
shape-ID harness (`EvaluateLegibility.py`).

## Sources

- [A survey of handwriting synthesis from 2019 to 2024: A comprehensive review](https://www.sciencedirect.com/science/article/pii/S0031320325000172)
- [A Survey of Modern Handwriting Generation Models](https://www.ijsat.org/papers/2025/4/9815.pdf)
- [Generating Sequences With Recurrent Neural Networks (Graves, 2013)](https://arxiv.org/abs/1308.0850) · [sjvasquez/handwriting-synthesis](https://github.com/sjvasquez/handwriting-synthesis)
- [Disentangling Writer and Character Styles for Handwriting Generation (SDT, CVPR 2023)](https://arxiv.org/abs/2303.14736) · [dailenson/SDT](https://github.com/dailenson/SDT)
- [Handwritten Text Generation From Visual Archetypes (VATr, CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Pippi_Handwritten_Text_Generation_From_Visual_Archetypes_CVPR_2023_paper.pdf) · [VATr++ overview](https://www.azoai.com/news/20240223/VATr2b2b-Advanced-Few-Shot-Styled-Handwritten-Text-Generation.aspx)
- [One-DM: One-Shot Diffusion Mimicker for Handwritten Text Generation](https://arxiv.org/abs/2409.04004)
- [DiffInk: Glyph- and Style-Aware Latent Diffusion Transformer for Text to Online Handwriting Generation](https://arxiv.org/pdf/2509.23624)
- [TRACE: A Differentiable Approach to Line-Level Stroke Recovery for Offline Handwritten Text (ICDAR 2021)](https://arxiv.org/abs/2105.11559)
- [Handwriting Trajectory Recovery with Diffusion Models (2025)](https://arxiv.org/html/2607.03422v1)
- [CalliRewrite: Recovering Handwriting Behaviors from Calligraphy Images without Supervision (ICRA 2024)](https://arxiv.org/abs/2405.15776) · [LoYuXr/CalliRewrite](https://github.com/LoYuXr/CalliRewrite)
- [WriteViT: Handwritten Text Generation with Vision Transformer](https://arxiv.org/pdf/2505.13235)
- [Quo Vadis Handwritten Text Generation for Handwritten Text Recognition?](https://arxiv.org/pdf/2508.09936)
