# Code Explained — Part B: Training & Classification

This document explains, function by function, every piece of code in the project's **neural-network training and classification** block: the text-recognition model (`PaperCRNN`, a CNN-BiLSTM-CTC recognizer), the writer-identification model (`AuthorClassifierCNN`), the IAM dataset line-extraction utility that feeds both, the shared author/dataset configuration, and the from-scratch numpy re-implementation of both models' forward passes used for dependency-free inference. It is written so the author can (a) understand exactly what their own training code does and why, and (b) lift material almost directly into the "Design and implementation" and "Literature study" sections of the final report.

---

## 1. `Software/CNN/TrainText.py`

### Role in the pipeline

This file is the **text-recognition training block**. It defines `PaperCRNN` — a CNN + bidirectional LSTM + CTC handwriting-line recognizer reproducing the architecture of Kizilirmak & Yanikoglu (2023), *"CNN-BiLSTM model for English Handwriting Recognition: Comprehensive Evaluation on the IAM Dataset"* — and four independently runnable training recipes (`base`, `hf`, `joint`, `personal`) that all share the same model class, preprocessing, charset, and CTC decoding logic. `TrainAuthor.py` imports several of its functions/classes directly (`IAMLineDatasetRaw`, `PaperCRNN`, `conv_block`, `resize_line_image_fixed`, `tensor_from_resized`, `encode_text`, `_decode_png`, `augment_line_image`) rather than duplicating them, so this file is also a dependency of the writer-ID trainer. Its trained checkpoints (`weights/paper_cnn_bilstm_ctc_*_best.pt`) are consumed downstream by inference/evaluation scripts (`VerifyRewrite.py`, `np_inference/text_model.py`) and by `TrainAuthor.py` as a backbone warm-start source. It belongs entirely to the **Training & Classification** functional block.

The module docstring itself is an important piece of documentation: it states explicitly which parts of the Kizilirmak & Yanikoglu paper are reproduced (12-conv backbone, 2 max-pools, 32→64→128 channel progression, 2-layer BiLSTM hidden=256 no dropout, RMSprop lr=1e-3/wd=1e-5/batch=16, the Table-3 "combined" augmentation recipe) and which are deliberately **not** reproduced because they need infrastructure this project doesn't have (2.5M-image synthetic pretraining, lexicon-constrained word-beam-search decoding, test-time augmentation) — greedy CTC decoding is used instead throughout.

### `encode_text(text)`

**Purpose**: Converts a ground-truth transcript string into the integer label sequence CTC loss needs.

**Inputs**: `text` (`str`) — a line's transcript, assumed to consist mostly of characters in `CHARSET`.

**Outputs**: `torch.Tensor`, dtype `torch.long`, shape `(L,)` where `L` is the number of *recognised* characters (characters not in `CHARSET` are silently dropped, not errored).

**Inner workings**: Looks each character up in `CHAR_TO_IDX`, a dict built as `{char: idx+1 for idx, char in enumerate(CHARSET)}` — i.e. indices start at **1**, not 0, because index 0 is reserved for the CTC blank symbol (this reservation is structural to the CTC algorithm, see `decode_ctc` below).

**Literature connection**: Standard integer-encoding preprocessing for CTC-based sequence models; the blank-reserved-at-0 convention follows Graves et al. (2006).

### `decode_ctc(log_probs)`

**Purpose**: Turns the network's raw per-timestep class probabilities into decoded text strings via **greedy CTC decoding** (as opposed to the paper's lexicon-constrained word-beam search, which this project doesn't implement).

**Inputs**: `log_probs`, shape `(T, B, C)` — `T` timesteps (CTC frames, here fixed at `INPUT_WIDTH // 4 = 160`), `B` batch, `C = len(CHARSET)+1` classes including blank.

**Outputs**: `list[str]`, length `B` — one decoded string per batch element.

**Inner workings and maths**: `torch.argmax(log_probs, dim=2)` picks the single most likely class at every timestep ("best path" decoding). For each batch element's sequence of per-frame best-class indices, the standard CTC collapse rule is applied: consecutive repeated tokens are merged into one, and the blank token (index 0) is dropped entirely. E.g. a frame sequence `[a,a,blank,a,b,b]` decodes to `"ab"` (the repeated `a`s before the blank collapse to one `a`; the blank then lets a *second*, distinct `a` symbol follow immediately after; `b,b` collapses to one `b`). This is the decoding half of Graves et al.'s CTC formulation — it does **not** attempt to find the output sequence with the globally highest total probability (summed over all frame-paths mapping to that sequence), only the single highest-probability *frame path*, then collapses it. This is an approximation, cheap but occasionally suboptimal versus a proper forward-backward/beam search over the full CTC lattice.

**Literature connection**: Graves, Fernández, Gomez & Schmidhuber (2006), *"Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks"* — this is their "best path decoding" (Section 3.1 approximation to the true most-probable labelling), as distinct from the lexicon-constrained word-beam search the source paper actually reports its best numbers with. Forced alignment / beam search decoding (a different, more exact decoding strategy) lives in `ProfileIO.py` and is documented separately — not duplicated here.

### `levenshtein(a, b)`

**Purpose**: Computes character-level edit distance between predicted and ground-truth strings, the basis of Character Error Rate (CER).

**Inputs**: two strings `a`, `b`.

**Outputs**: `int`, minimum number of single-character insertions/deletions/substitutions to turn `a` into `b`.

**Inner workings**: Classic Wagner–Fischer dynamic-programming edit distance, implemented with a rolling 1-D array (`previous`/`current`) rather than a full 2-D matrix — O(len(a)·len(b)) time, O(min(len(a),len(b))) space. The early swap (`if len(a) < len(b): return levenshtein(b, a)`) just ensures the shorter string is iterated as the inner dimension; it doesn't change the result (edit distance is symmetric).

**Literature connection**: Levenshtein (1966) edit distance; standard technique, not specific to this project's novel contributions.

### `resize_line_image_fixed(pil_img, height, width, pad_value)`

**Purpose**: Fits a variable-size line-image crop onto the model's fixed `(INPUT_HEIGHT=64, INPUT_WIDTH=640)` canvas by **uniform scaling + white padding**, deliberately *not* independent width/height stretching.

**Inputs**: `pil_img` (PIL `Image`, arbitrary size, any mode — converted to `"L"` grayscale first); `height`, `width` (ints, default 64×640); `pad_value` (int, default 255 = white).

**Outputs**: PIL `Image`, mode `"L"`, exactly `(width, height)` pixels.

**Inner workings**: Computes a single scale factor `s = min(height/origH, width/origW)` — the binding constraint of the two axes — so the whole line is scaled isotropically (same factor on both axes), never squashed. Resizes to `(round(w·s), round(h·s))` via bilinear interpolation, then pastes that onto a blank white canvas of the target size, **left-aligned** at x=0 and **vertically centred**. The in-code comment explains why this matters beyond cosmetics: real line crops sit around a 14:1 width:height ratio, not the canvas's 10:1 ratio, so a plain `.resize()` to a fixed size would scale height more aggressively than width, distorting every letter's true stroke proportions — a real problem for a system whose downstream forced-alignment step needs to trust exact stroke geometry. Left-alignment specifically is what makes `frame_x_to_pixel` (below) a correct geometric inverse of this placement.

**Literature connection**: Standard "letterbox" resize-with-padding technique common in CNN preprocessing pipelines; not attributable to a single paper.

### `frame_x_to_pixel(frameX, origW, origH, width, height)`

**Purpose**: Inverse-maps a CTC frame-index-derived canvas x-coordinate back into the *original, unresized* line image's pixel coordinates — needed anywhere forced-alignment cut boundaries (computed in canvas space) must be expressed in the coordinate system of the actual photographed/scanned line.

**Inputs**: `frameX` (float, canvas-space x pixel, itself computed elsewhere as `t/T * width` for a CTC frame index `t`); `origW`, `origH` (the original crop's dimensions before resizing); `width`, `height` (canvas dimensions).

**Outputs**: `float` — x-coordinate in the original image's pixel space.

**Inner workings**: Recomputes the *same* scale factor `s = min(height/origH, width/origW)` that `resize_line_image_fixed` used, then returns `frameX / s`. Because the model's CTC time axis only encodes *horizontal* position (width is downsampled by the two 2×2 max-pools; height is fully collapsed), only the x-coordinate needs inverting — there is no y-coordinate analogue. This function's correctness depends entirely on `resize_line_image_fixed`'s placement convention (left-aligned, single isotropic scale) never silently changing.

### `tensor_from_resized(pil_img)`

**Purpose**: Converts a resized-and-padded PIL grayscale image into the normalized float tensor the CNN consumes.

**Inputs**: PIL `Image`, mode `"L"`, shape `(INPUT_HEIGHT, INPUT_WIDTH)`.

**Outputs**: `torch.Tensor`, dtype float32, shape `(1, H, W)` — single "channel" dimension added via `unsqueeze(0)`.

**Inner workings**: `arr = pixel/255` maps to `[0,1]`; `arr = 1 - arr` **inverts** so ink (originally dark, low pixel value) becomes high and background (white, high pixel value) becomes low — a convention used consistently across the whole codebase so "ink = high activation"; then `(arr - 0.5)/0.5` rescales to roughly `[-1, 1]`, a fixed (not per-sample/per-batch) normalization constant.

**Literature connection**: Fixed-constant normalization (vs. per-sample mean/std) is a deliberate choice documented in the module's preprocessing comment: because every sample is forced onto an identically-sized canvas with no variable amount of blank padding, a fixed constant avoids the contrast-normalization instability a padded, variable-content canvas would otherwise introduce.

### `_augment_shear(img)`, `_augment_rotate(img)`, `_augment_perspective(img)` / `torchvision_perspective_points(...)`, `_augment_elastic(img)`

**Purpose**: Four independent data-augmentation transforms matching Kizilirmak & Yanikoglu's Table 3 "combined" augmentation recipe (their best-performing configuration).

**Inputs**: each takes a PIL image (already resized to the fixed canvas — critical, see `augment_line_image` below).

**Outputs**: a transformed PIL image, same size, white-filled background outside the warped content.

**Inner workings**:
- `_augment_shear`: samples shear factor `k ~ U(-0.6, 0.6)`, converts to a shear angle via `atan(k)` in degrees, applies via `torchvision.transforms.functional.affine` with zero rotation/translation/scale.
- `_augment_rotate`: samples rotation angle `~ U(-2.5°, 2.5°)`, applies an affine rotation.
- `_augment_perspective`: samples a distortion scale `~ U(0.03, 0.08)`, builds `startpoints` (exact image corners) and `endpoints` (corners jittered by up to `distortion_scale × half-dimension` in each direction) via `torchvision_perspective_points`, then applies `TF.perspective`. The module docstring notes this stands in for the source paper's Luo et al. "geometric transformations" category, whose exact method needs an external repo this project doesn't vendor — `RandomPerspective`-style jitter is used as an approximation of the same spirit.
- `_augment_elastic`: implements **Simard et al. elastic distortion** — samples two random displacement fields (`dx`, `dy`, uniform noise in `[-1,1]` per pixel), smooths each with a Gaussian filter of std `sigma ∈ {3,4}` (`scipy.ndimage.gaussian_filter`), scales by `alpha ∈ {15,20}`, then warps the image by resampling at `(y+dy, x+dx)` via `scipy.ndimage.map_coordinates` (bilinear, `order=1`, white fill for out-of-bounds).

**Literature connection**: `_augment_elastic` is a direct, named implementation of Simard, Steinkraus & Platt (2003), *"Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis"* — the canonical elastic-distortion augmentation for handwritten-character recognition (also used in their classic MNIST augmentation work). Shear/rotation/perspective are standard affine/projective image-augmentation techniques.

### `augment_line_image(pil_img)`

**Purpose**: Applies **at most one** of the four augmentations per training image, matching the paper's independent-per-augmentation `p=0.5` gating (so most training images get either nothing or exactly one mild deformation, never multiple stacked).

**Inputs**: a PIL image that **must already be resized** to the fixed canvas (the function's docstring explicitly warns this — `_augment_elastic` allocates several full-resolution float arrays proportional to pixel count, and calling it on a raw, high-DPI scan/photo crop caused real `MemoryError` crashes across parallel DataLoader workers before this was fixed).

**Outputs**: possibly-augmented PIL image, same size as input.

**Inner workings**: Builds a candidate list by independently flipping a `p=0.5` coin for each of the four augmentation functions; if none pass, returns the image unchanged; otherwise picks uniformly at random among the candidates that passed and applies only that one.

### `_encode_png(pil_img)` / `_decode_png(png_bytes)`

**Purpose**: Serialize/deserialize a line-image crop to/from PNG bytes for on-disk caching.

**Inner workings**: Thin wrappers around `PIL.Image.save(..., format="PNG")` / `Image.open(io.BytesIO(...))`. PNG (lossless, entropy-coded) is chosen because handwriting scans are mostly white background, which compresses very well — keeping the cache from ballooning the way raw pixel arrays would.

### `class LineImageCache`

**Purpose**: Caches each IAM page's **segmented line crops** (as PNG bytes), keyed only by `page_key` (`"author/page.png"`) — deliberately independent of label text, so correcting a page's transcript never requires re-running (expensive) line segmentation.

**Inputs to `__init__`**: `root_dir` (dataset root containing per-author subfolders of page PNGs); `cache_dir`; `force_rebuild` (bool, ignore existing cache); `save_every_n_pages` (checkpoint interval); `max_pages` (optional cap, for quick smoke tests).

**Inner workings**:
- On construction, if a `manifest.pt` already exists and `force_rebuild` is false, loads the list of already-processed pages and the shard files containing their cached line-PNG-byte-lists, merging all shards into `self.pages` (a dict `page_key -> list[bytes]`).
- For every not-yet-cached page, calls `ExtractLinePatches(..., expectedLineCount=None, labelLines=None)` (from `ExtractIAMLines.py`, documented in Section 3 below) to segment that page into lines **without** any influence from a particular label file's line count — ensuring image segmentation and label text stay genuinely independent sources of truth. Each detected line's `raw_crop` (the *unresized*, un-padded crop — not the `processed_patch` other scripts use) is PNG-encoded and stored.
- Writes incremental checkpoints: every `save_every_n_pages` pages, flushes the newly-collected pages into a brand-new shard file (`shard_NNNNNN.pt`) and updates `manifest.pt` — previously-written shards are never rewritten. A `KeyboardInterrupt` handler ensures a Ctrl-C during a long run still flushes pending work before re-raising, so partial progress up to the last full page is never lost.

**Outputs**: `self.pages`, a dict mapping `page_key -> list[PNG-bytes]` (one entry per detected line, top-to-bottom order); `get(page_key)` returns that list or `None`.

**Literature connection**: Software-engineering pattern (incremental checkpointing / shard-and-manifest caching), not a machine-learning technique per se.

### `class IAMLineDatasetRaw(Dataset)`

**Purpose**: PyTorch `Dataset` that combines `LineImageCache`'s label-independent cached line images with each page's **current** ground-truth text (re-read fresh from `_labels.txt` every run — cheap, so label corrections are free without re-segmenting) into the final flat list of trainable samples.

**`__init__` inputs**: `root_dir`, `min_pages_for_holdout` (default 3 — an author needs at least this many labeled pages before one gets held out for validation), `cache_dir`, `force_rebuild`, `save_every_n_pages`, `max_pages`, `is_train`.

**Inner workings**:
1. Resolves the dataset root (`resolve_dataset_root`, below).
2. Builds/loads a `LineImageCache` over that root.
3. Builds `author_folders` (sorted subfolder names) and `author_to_idx`.
4. For each author, finds `labeled_pages` (pages with a non-empty `ReadLabelLines` result) and designates the **last** such page (alphabetically) as the per-author holdout page — but only if the author has at least `min_pages_for_holdout` labeled pages; otherwise no page is held out for that author.
5. For every page: reads its current label lines; fetches the cached PNG list; if the counts of cached images and current labels disagree, **skips the whole page** with a warning (segmentation and label text have drifted out of sync — this page needs cache regeneration before reuse) rather than silently misaligning images and text.
6. For every (image, label) pair that survives: encodes the text (`encode_text`); skips zero-length encoded targets; **skips any label longer than `MAX_SAFE_LABEL_CHARS`** (120 at current 64×640/stride-4 settings) with a loud warning — an oversized label almost certainly indicates line segmentation has collapsed several real printed/handwritten lines into one detected region, and such samples are mathematically unalignable under CTC anyway (see `MAX_SAFE_LABEL_CHARS` discussion below) and would otherwise silently waste a training slot or, under `zero_infinity=True`, contribute exactly zero loss.
7. Appends a dict per surviving sample: `{page_key, line_idx, image_png, target, text, is_holdout}`.

**`resolve_dataset_root(root_dir)`** (static): if `root_dir/data` exists and has subfolders, uses that as the real root instead (handles a nested dataset layout variant).

**`__getitem__(idx)`**: decodes the cached PNG, resizes it onto the fixed canvas, applies augmentation if `is_train`, converts to a normalized tensor, and returns `(img_tensor, target_tensor, text_str)`.

**`holdout_split_indices()`**: returns `(train_idx, test_idx)` — a simple partition of `self.samples` by the `is_holdout` flag set during construction (one whole page per author reserved for validation, not a random per-line split — this avoids leaking near-duplicate handwriting style/vocabulary from the same page into both splits).

**Literature connection**: The "hold out one whole page per author/writer, not random lines" split strategy follows standard writer-independent evaluation practice in handwriting-recognition literature (avoids optimistic bias from training and testing on lines from the same physical page).

### `MAX_SAFE_LABEL_CHARS` constant and its reasoning

Not a function, but worth documenting explicitly since the prompt calls for rigor: `INPUT_WIDTH // 4 = 160` is the CTC time dimension `T` — fixed purely by architecture (two 2×2 max-pools each halve width). A fundamental constraint of CTC is that **the output label length cannot exceed the number of input frames** (even ignoring the blank-insertion requirement for repeated characters, which tightens this further) — so any label with ≥160 characters is literally impossible to align, and PyTorch's `CTCLoss(..., zero_infinity=True)` silently zeroes such a sample's loss/gradient rather than erroring. `MAX_SAFE_LABEL_CHARS = int(160 * 0.75) = 120` caps comfortably under that hard limit.

### `class TrainEvalView(Dataset)`

**Purpose**: A thin wrapper letting a train subset and a validation subset of the *same* underlying `IAMLineDatasetRaw` independently toggle augmentation (train: on; val: off) without the cost of `copy.deepcopy`-ing the whole dataset.

**Inputs**: `base_dataset` (the full `IAMLineDatasetRaw`), `indices` (the subset's sample indices), `is_train` (bool, controls augmentation only).

**`__getitem__`**: identical processing to `IAMLineDatasetRaw.__getitem__`, just indexed through `self.indices[i]` into the base dataset's `samples` list, and gating augmentation on this view's own `is_train` flag rather than the base dataset's.

### `collate_fn(batch)`

**Purpose**: Batches a list of `(image, target, text)` samples into the flattened-target representation PyTorch's `CTCLoss` expects.

**Inputs**: `batch`, a list of `(img_tensor (1,H,W), target_tensor (Li,), text_str)` tuples.

**Outputs**: `(images, flat_targets, target_lengths, texts)` where `images` is `torch.stack` of shape `(B,1,H,W)`; `flat_targets` is `torch.cat(targets)` — **all** targets concatenated into one 1-D tensor of total length `ΣLi` (not padded to a common length); `target_lengths` is a `(B,)` tensor recording each sample's own `Li` so the loss function can re-split `flat_targets` internally; `texts` is the plain list of ground-truth strings (used only for CER computation, not loss).

**Literature connection**: This flattened-targets-plus-lengths format is exactly PyTorch's documented `CTCLoss` input convention (itself following Graves et al. 2006's formulation where each sequence's label length varies independently of input length).

### `conv_block(in_ch, out_ch, n_layers)`

**Purpose**: Builds one "stage" of the CNN backbone — `n_layers` repetitions of `Conv2d → BatchNorm2d → ReLU`, matching Table 1 of the source paper's per-stage structure.

**Inputs**: `in_ch` (int, input channel count — only the first layer in the stage differs, remaining layers go `out_ch → out_ch`); `out_ch` (int); `n_layers` (int, 2/4/6 for the three stages).

**Outputs**: `nn.Sequential` of `3 × n_layers` modules.

**Inner workings**: Each `Conv2d` uses a fixed `kernel_size=3, padding=1` — "same" padding, meaning each conv alone does not change spatial dimensions (only pooling layers, applied separately, do). `BatchNorm2d(out_ch)` normalizes each of the `out_ch` channels independently across the batch+spatial dimensions during training, then applies a learned affine transform (see batchnorm maths under `batchnorm2d` in Section 5 below — training-mode batchnorm uses live batch statistics, whereas the numpy inference re-implementation uses the frozen running statistics; both apply the same functional form once the normalizing statistics are fixed). `ReLU(inplace=True)` is the standard rectified-linear nonlinearity, `max(0, x)`.

**Literature connection**: Standard CNN block design. BatchNorm specifically: Ioffe & Szegedy (2015), *"Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"*.

### `class PaperCRNN(nn.Module)`

**Purpose**: The text-recognition model itself — the CNN-BiLSTM-CTC architecture from Kizilirmak & Yanikoglu (2023).

**`__init__` inputs**: `num_classes` (int, `len(CHARSET)+1` including blank); `lstm_hidden` (default 256); `lstm_layers` (default 2); `conv_width_mult` (default 1.0 — a later addition for an architecture-search sweep script, scales all channel counts together, default preserves the exact paper architecture); `dropout` (default 0.0, applied between BiLSTM layers and after each conv stage when >0 — also a later addition, default preserves the paper's "no dropout" setting).

**Architecture / inner workings**:
- **Stage 1**: `conv_block(1, w1=32, n_layers=2)` then `MaxPool2d(2,2)` — halves both H and W.
- **Stage 2**: `conv_block(w1, w2=64, n_layers=4)` then another `MaxPool2d(2,2)` — halves H and W again (so after stages 1–2, H and W are each `/4` of input).
- **Stage 3**: `conv_block(w2, w3=128, n_layers=6)`, **no further pooling** (matches the paper's "2 max-pooling operations" spec in Table 1 — pooling happens exactly twice, not once per stage).
- `conv_dropout`: `nn.Dropout2d` if `dropout>0` else identity.
- `height_pool = nn.AdaptiveMaxPool2d((1, None))`: collapses whatever height remains (after two /2 pools, `H_in/4`) down to exactly 1 row via a max operation, leaving width untouched (`None` means "keep as is"). This is what converts a 2-D feature map into a 1-D sequence-over-width.
- `sequence = nn.LSTM(input_size=w3, hidden_size=256, num_layers=2, bidirectional=True, dropout=... , batch_first=False)`: the BiLSTM sequence encoder (full gate maths given in Section 5's `lstm_forward` entry, which is the exact numpy re-implementation of what this PyTorch module computes internally).
- `text_head = nn.Linear(lstm_hidden*2, num_classes)`: maps each timestep's `2×256=512`-dim BiLSTM output (forward+backward concatenated) to per-class logits.

**`forward(x)`**: `x` is `(B, 1, H, W)` = `(B, 1, 64, 640)`.
1. `pool1(stage1(x))` → `(B, 32, 32, 320)`.
2. `pool2(stage2(x))` → `(B, 64, 16, 160)`.
3. `conv_dropout(stage3(x))` → `(B, 128, 16, 160)` (stage 3 does not change spatial size; only channels change to 128).
4. `height_pool(x)` → `(B, 128, 1, 160)`.
5. `x.squeeze(2).permute(2, 0, 1)` → `(160, B, 128)` — this is `(T, B, D)`, the sequence-major format `nn.LSTM(batch_first=False)` expects: `T=160` is the CTC time dimension, `D=128` is the per-timestep feature vector (one "column" of the final conv feature map, height-collapsed).
6. `seq, _ = self.sequence(x)` → `(160, B, 512)` (512 = 256 hidden × 2 directions, concatenated).
7. `logits = self.text_head(seq)` → `(160, B, num_classes)`.
8. Returns `log_softmax(logits, dim=2)` — **log**-probabilities, because `nn.CTCLoss` expects log-probabilities as input (it internally exponentiates/combines them via the forward-backward algorithm).

**Literature connection**: This entire class is a direct architectural reproduction of Kizilirmak & Yanikoglu (2023)'s best-scoring configuration (their Table 1/Table 2, CER 5.01% on their own validation split). The BiLSTM itself traces to Hochreiter & Schmidhuber (1997) for LSTM and Schuster & Paliwal (1997) for the bidirectional-RNN construction (running one LSTM forward over the sequence and a second over the reversed sequence, then concatenating outputs at each timestep — exactly what `bidirectional_lstm_forward` in the numpy re-implementation does explicitly, Section 5). The CTC decoder head (linear + log_softmax, trained with `CTCLoss`) follows Graves et al. (2006).

### `evaluate(model, dataloader, device, ctc_loss_fn)`

**Purpose**: Computes validation-set CTC loss and character accuracy (1 − CER), matching the paper's stated early-stopping criterion of validation **loss** (not accuracy).

**Inputs**: a `PaperCRNN` instance, a `DataLoader` yielding `(images, targets, target_lengths, texts)`, the torch device, and the `nn.CTCLoss` instance.

**Outputs**: `(avg_loss: float, char_acc: float, cer: float)`.

**Inner workings**: Puts the model in `eval()` mode (disables dropout, and — importantly for later discussion — switches any `BatchNorm` layers to use their frozen running statistics rather than live batch statistics) and runs under `torch.no_grad()`. For each batch: computes `log_probs = model(images)`; builds `input_lengths` as a constant `log_probs.size(0)` (=160) for every sample in the batch, since every sample always produces exactly `T=160` output frames regardless of actual line content; computes CTC loss; greedily decodes predictions (`decode_ctc`); accumulates total ground-truth character count and total Levenshtein edit distance across all samples. Final `cer = total_char_errors / total_chars`; `char_acc = 1 - cer`.

**Literature connection**: Character Error Rate is the standard evaluation metric for handwriting/speech recognition (used throughout the source paper and the wider OCR/ASR literature).

### `train(config, device)`

**Purpose**: The main training loop for the `base` subcommand (own segmented `IAMpages671` page scans) — implements the paper's Section 6.1 training recipe plus several deliberate deviations the module docstring documents and justifies.

**Inputs**: `config` dict (keys: `data_dir`, `cache_dir`, `force_rebuild`, `max_pages`, `batch_size`, `lr`, `weight_decay`, `epochs`, `early_stop_patience`, `eval_every`, `max_restarts`, `resume`, `name`); `device` (`torch.device`).

**Inner workings, step by step**:
1. Builds `IAMLineDatasetRaw`, requires ≥4 total samples and ≥1 held-out sample.
2. Splits into `TrainEvalView` (augmented) / `TrainEvalView` (not augmented) via `holdout_split_indices()`.
3. Builds `DataLoader`s (4 workers + pinned memory if CUDA available, else single-process).
4. Builds the model: `PaperCRNN(num_classes=len(CHARSET)+1)`.
5. Loss: `nn.CTCLoss(blank=0, zero_infinity=True)` — `blank=0` matches the index-0-reserved convention from `encode_text`; `zero_infinity=True` prevents a numerically-impossible alignment (label longer than input) from producing `inf`/`nan` gradients that would poison the whole batch — instead that one sample's loss/gradient contribution is silently zeroed (this is exactly why `MAX_SAFE_LABEL_CHARS` filtering at the dataset level matters: without it, such samples wouldn't crash training, they'd just waste a training slot doing nothing).
6. Optimizer: `RMSprop(lr=1e-3, weight_decay=1e-5)` — directly per the paper's Section 6.1 recipe.
7. Scheduler: `ReduceLROnPlateau(mode="min", factor=0.5, patience=5, min_lr=1e-5)` on validation loss — **a deliberate deviation** from the paper's literal fixed-LR recipe, justified in-code: the paper's much larger IAM training split makes validation loss move smoothly enough for a fixed LR + patience-10 early stop; on this project's much smaller (~10k-line) dataset, validation loss is noisier epoch-to-epoch, so a fixed high LR keeps bouncing around a loss basin rather than settling into it — decaying LR on plateau lets it settle. The paper's own text even mentions "tailoring" LR in some experiments, so the module docstring argues this isn't out of spirit.
8. Full training-state checkpoint/resume: unlike a weights-only save, this saves/restores `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `epoch`, `best_val_loss`, `patience_counter`, and `restart_count` together — a comment explicitly notes that a partial (weights-only) resume would silently reset RMSprop's running averages and the early-stop counter, visibly disrupting an in-progress run.
9. **Per-epoch loop**: standard forward→loss→backward→step on every training batch, with a throttled progress heartbeat (`\r`-overwritten, printed every 5th batch) showing running loss, seconds/batch, and ETA.
10. **Validation gating** via `eval_every`: validation (and its relatively expensive pure-Python Levenshtein-based CER computation) is only run every `eval_every` epochs (always on the final epoch) — a documented tradeoff, since `patience_counter` then advances per *validation check*, not per raw epoch, so `early_stop_patience` means "N validation checks with no improvement," which can span up to `N × eval_every` wall-clock epochs when `eval_every > 1`.
11. On each validation check: steps the LR scheduler on the fresh val loss (the in-code comment flags that this `.step()` call was **previously missing entirely** — a real bug where the scheduler object existed but was never advanced, so LR silently stayed fixed at its initial value regardless of how long validation loss plateaued); if val loss improved, saves `best_path` and resets `patience_counter`; else increments it.
12. Always saves the full training-state checkpoint at the end of every epoch (regardless of whether validation ran).
13. **Warm restarts**: when `patience_counter >= early_stop_patience` on a validation-checked epoch, rather than stopping outright, the code reloads the best-known weights, discards the optimizer's momentum state and the LR scheduler's decayed state (both rebuilt fresh at the original `config['lr']`), resets `patience_counter` to 0, and continues training — up to `max_restarts` times (config default 3, `base` subcommand uses its own CLI-exposed default). The rationale given in-code: by the time patience triggers, `ReduceLROnPlateau` may have already decayed LR down to something tiny, which can itself be *why* progress stalled; a fresh, larger LR re-applied to the best-known weights sometimes escapes that stall. After `max_restarts` exhausted restarts all plateau too, training stops for good.

**Literature connection**: RMSprop (Tieleman & Hinton, unpublished lecture notes, 2012) is a standard adaptive-learning-rate optimizer; `ReduceLROnPlateau` and warm restarts are standard training-engineering techniques (the warm-restart idea echoes, loosely, the spirit of Loshchilov & Hutter's SGDR warm-restart scheduling, though this implementation is a simpler ad hoc variant, not a direct reproduction of that paper's cosine-annealing schedule).

### `main_base(args)`

**Purpose**: Entry point for `python TrainText.py base` — sets up the original (pre-merge) `TrainText.py`'s exact behaviour: training on the project's own segmented `IAMpages671` page scans.

**Inner workings**: Resolves `data_dir` to `Data/Datasets/IAMpages671`; if `--max-pages` wasn't passed on the CLI, interactively prompts for it; selects CUDA if available; redirects **both** `stdout` and `stderr` through a `DualLogger` (so a crash traceback — previously only visible on stderr, never persisted — now also lands in the `.log` file, a documented bug fix); builds the `config` dict with all the paper's Section 6.1 hyperparameters hardcoded (`batch_size=16`, `lr=1e-3`, `weight_decay=1e-5`, `epochs=200`, `early_stop_patience=15`); seeds `random`/`numpy`/`torch` with `7` for reproducibility; calls `train(config, device)` inside a `try/except` that prints a full traceback (to both console and log file) before re-raising on any crash.

### `class DualLogger`

**Purpose**: Mirrors everything written to `sys.stdout`/`sys.stderr` into both the real console and a log file simultaneously.

**Inner workings**: Implements the minimal file-like interface (`write`, `flush`) PyTorch/Python's print machinery needs; each `write` call forwards to both `self.terminal` (the original stdout) and `self.log_file` (opened in append mode).

### HuggingFace-dataset training path (`hf` subcommand, merged in from the former `TrainTextHF.py`)

**`class HFLineDataset(Dataset)`**

**Purpose**: Wraps the HuggingFace `Teklia/IAM-line` dataset (pre-segmented line images + transcripts, no custom line-segmentation needed) in the same interface the rest of the file expects.

**`__init__`**: does one cheap pass over just the `text` column (no image decoding yet) to filter out empty-after-charset-cleaning rows and rows exceeding `MAX_SAFE_LABEL_CHARS`, building `self.rows = [(hf_index, cleaned_label), ...]`; optionally truncates to `max_samples` for smoke testing.

**`__getitem__(i)`**: looks up the HF dataset row by stored index (image decode deferred to here, lazily), resizes/augments/tensorizes exactly like the base dataset path.

**`load_hf(cache_dir)`**: thin wrapper around `datasets.load_dataset("Teklia/IAM-line", cache_dir=...)`, raising a clear `SystemExit` with install instructions if the `datasets` package isn't installed.

**`train_hf(config, device)`**: structurally identical training loop to `train()` above (same RMSprop/ReduceLROnPlateau/checkpoint/warm-restart logic), differing only in its data source (`HFLineDataset` wrapping the HF splits) and an added option (`config["init_from"]`) to warm-start from an existing checkpoint of the same architecture when there's no HF-specific checkpoint yet to resume — explicitly noted as saving the epochs that would otherwise be spent relearning generic stroke features, which matters without a GPU.

**`eval_only_hf(config, device)`**: loads the best (or latest checkpoint's) HF-trained weights and reports CER on both the `validation` and `test` HF splits, no training performed.

**`main_hf(args)`**: CLI entry point assembling `config` from argparse args, dispatching to either `eval_only_hf` or `train_hf`.

### Joint training path (`joint` subcommand, merged from `TrainTextJoint.py`)

**Purpose**: Trains **one** recognizer on both the full Teklia/IAM-line dataset **and** personal handwriting pages simultaneously every epoch, specifically to avoid "catastrophic forgetting" of general handwriting recognition that naive sequential fine-tuning on a small personal dataset alone would cause (this forgetting is the exact failure mode later fixed/explained in the `personal` subcommand's `freeze_conv` option, see below).

**`collect_personal_lines_joint()`**: for every `PERSONAL_AUTHORS` folder, segments each page's lines via `SegmentLean.SegmentLines` (the **same** segmenter used by writer-ID training and every evaluation script, so training and scoring see consistent line crops), pairs each with its ground-truth label (dropping any marked `"MESS"`), filters by charset/length exactly like `HFLineDataset`, then does a **per-page** shuffle+split into train/val at `VAL_FRACTION` using `SPLIT_SEED` — i.e. every page contributes both train and val lines (rather than holding out whole pages), to avoid starving either split of page-specific vocabulary.

**`class PersonalLineDataset(Dataset)`**: shared by `joint` and `personal` — stores `(crop_ndarray, label)` rows, resizes/augments/tensorizes identically to the other dataset classes.

**`build_train_loader(teklia_train, personal_train, batch_size, personal_fraction)`**: builds a single `DataLoader` over a `ConcatDataset` of both sources, using a `WeightedRandomSampler` whose per-sample weights are chosen so that, **in expectation**, `personal_fraction` of every sampled epoch's lines come from the personal set — without literally duplicating personal samples on disk. Concretely `w_tek = (1-f)/n_tek` and `w_per = f/n_per` per-sample, with `num_samples = n_tek` drawn with replacement.

**`main_joint(args)`**: loads both data sources, builds the weighted joint loader plus two **separate** validation loaders (Teklia val, personal val — evaluated and reported independently, never mixed), requires `--init-from` (the function refuses to train from scratch — a hard requirement enforced at the CLI parser level, since training this mixed an objective from random weights would be unreasonable). Each epoch: trains on the weighted mix, evaluates both validation sets separately, and — notably — tracks `best_min_acc = min(tek_acc, per_acc)` as the checkpoint-selection criterion rather than either accuracy alone, specifically so the saved "best" checkpoint can't be one that improved on personal data at the Teklia domain's expense (or vice versa) without the min of the two actually going up. Also guards against non-finite loss (`torch.isfinite(loss)`) by skipping that batch's backward/step entirely, and clips gradient norm to 5.0.

### Personal-only fine-tuning path (`personal` subcommand, merged from `TrainTextPersonal.py`)

**`collect_personal_lines_personal()`**: nearly identical to `collect_personal_lines_joint()` (same segmenter, same per-page train/val split logic) but uses its own separate constants `PERSONAL_VAL_FRACTION=0.15`/`PERSONAL_SPLIT_SEED=0` (currently numerically identical to the shared `authors_config` constants, kept as a separate local copy in this merged file rather than refactored to share — a minor duplication worth flagging as a simplification opportunity, though not a correctness bug).

**`train_personal(config, device)`**: same RMSprop/ReduceLROnPlateau/checkpoint loop structure, with one architecturally significant option: `config["freeze_conv"]`. When set, the three convolutional stages (`stage1`,`stage2`,`stage3`) have `requires_grad=False` applied to every parameter, so **only the LSTM + text_head** receive gradient updates; only those parameters are passed to the optimizer. The in-code comment gives a measured justification: naive full-fine-tuning on this small, personal-handwriting-only dataset was confirmed to lose ~10–11 percentage points of character accuracy on clean Teklia val/test after only 16 epochs — i.e. full fine-tuning on a small, narrow dataset overwrites the generic stroke/edge features the CNN backbone learned from the much larger general dataset, a textbook instance of **catastrophic forgetting**. Freezing the backbone constrains how far the model can drift, trading some personal-domain accuracy for much less forgetting of general handwriting recognition. Checkpoints for this frozen variant are saved under a distinct `_frozen` suffix so they're never confused with the unfrozen personal checkpoint.

*(Note the deliberate contrast documented in the final "Known limitations" section below: this exact frozen-vs-fine-tuned tradeoff reasoning is inverted in `TrainAuthor.py`'s writer-ID classifier, where freezing was the wrong choice — because the two tasks need opposite kinds of invariance.)*

### Argument parsing / dispatch (`build_arg_parser`, `main`)

Four subparsers (`base`, `hf`, `joint`, `personal`) each exposing that subcommand's original standalone script's exact CLI flags/defaults/help text, preserved verbatim through the merge. Calling with no arguments at all preserves the original `TrainText.py`'s default behaviour of running `base` directly (no subcommand needed for the common case).

---

## 2. `Software/CNN/TrainAuthor.py`

### Role in the pipeline

This is the **writer-identification training block**: three independently runnable trainers (`original`, `10authors`, `shape`) that all instantiate `AuthorClassifierCNN`, a CNN classifier architecturally identical in its backbone to `PaperCRNN` but with a classification head instead of a sequence+CTC head. `10authors` produces the **live checkpoint** `server.py` loads to classify real photographed handwriting by author. `shape` produces a second, stroke-normalized classifier used purely for judging synthesis quality (how well the robot's reproduced handwriting matches a target author's *geometry*, independent of pen/ink appearance). It imports dataset/preprocessing machinery directly from `TrainText.py` (no duplication) and shared author-list/split constants from `authors_config.py`. Functional block: **Training & Classification**.

The module docstring gives important historical/methodological context worth citing in a literature review: an earlier, now-deleted script trained a single `MultiTaskLineCRNN` with both a text-CTC head and an author-classification head sharing one backbone; on the same 10 authors, writer-ID accuracy reached 98.9–100% by epoch 20 while text recognition stalled at ~28–29% character accuracy for the entire 200-epoch run. The documented interpretation: writer-ID over a small fixed set of known classes is a far easier task than open-vocabulary text recognition, so a shared-backbone multi-task setup keeps spending gradient budget on the already-solved easy task while the harder task stays undertrained — motivating the split into two dedicated, single-task models, which the docstring explicitly ties to how the writer-identification literature does it too (citing arXiv:2009.04877, a dedicated single-task writer-ID CNN).

### `class AuthorLabeledView(Dataset)`

**Purpose**: Wraps a base `IAMLineDatasetRaw` to additionally surface an **author index** per sample (the base class already tracks `page_key`/`author_to_idx` internally but never needed to expose it for text-recognition training).

**`__getitem__(i)`**: identical image loading/resize/augment/tensorize pipeline as `TrainEvalView`, plus `authorIdx = self.base.author_to_idx[item["page_key"].split("/")[0]]` — the author ID is simply the first path component of `page_key`.

**Output**: `(img_tensor, authorIdx: int, page_key: str)`.

### `collate_fn(batch)` (TrainAuthor's own)

**Purpose**: Batches `(image, authorIdx, pageKey)` triples. Unlike `TrainText.py`'s `collate_fn`, no variable-length target flattening is needed here — classification targets are just a scalar per sample — so this just stacks images and converts author indices to a `torch.long` tensor, keeping `pageKeys` as a plain list (used only for per-author accuracy reporting, not loss).

### `class AuthorClassifierCNN(nn.Module)`

**Purpose**: The writer-identification CNN — same conv backbone shape as `PaperCRNN` (by design — see "Transfer learning" discussion below), but global-pooled over width and fed through a small MLP classification head instead of a sequence model.

**`__init__(num_authors)`**: builds `stage1`/`pool1`/`stage2`/`pool2`/`stage3`/`height_pool` **identically** to `PaperCRNN` (same `conv_block` helper, same channel progression 1→32→64→128, same two 2×2 pools, same `AdaptiveMaxPool2d((1,None))` height collapse). Adds `width_pool = nn.AdaptiveAvgPool1d(1)` (collapses the remaining width dimension via **average**, not max, pooling — a deliberate difference from the height collapse, since author style is thought to be a property distributed across the whole line rather than concentrated in a single most-salient column). The classification head: `Linear(128, 96) → ReLU → Dropout(0.2) → Linear(96, num_authors)`.

**`forward(x)`**: `x` is `(B,1,64,640)`. After the shared backbone: `(B,128,1,160)` → `squeeze(2)` → `(B,128,160)` → `width_pool` → `(B,128,1)` → `squeeze(2)` → `(B,128)` pooled feature vector → `author_head` → `(B, num_authors)` raw logits (no softmax applied here — `nn.CrossEntropyLoss` expects raw logits and applies log-softmax internally).

**`load_backbone_from_text_model(textModelStateDict)`**: a targeted partial state-dict load. Builds `matched = {k: v for k,v in textModelStateDict.items() if k in own_state_dict and shapes match}`, updates only those keys, and loads the merged dict back. Because `PaperCRNN` and `AuthorClassifierCNN` share identical backbone module names/shapes by construction, every backbone tensor (`stage1.*`, `stage2.*`, `stage3.*`) matches and transfers; `author_head.*` (absent in `PaperCRNN`) and `sequence.*`/`text_head.*` (absent in `AuthorClassifierCNN`) are silently skipped by the shape/key-presence check — no error, no crash, just a smaller `matched` count reported back to the caller for logging.

**`freeze_backbone()`**: sets `requires_grad = False` on every parameter in `stage1`/`pool1`/`stage2`/`pool2`/`stage3` (pooling layers have no learnable parameters, included only for interface symmetry with the modules that do). **This method exists in the code but is used only by the `original` subcommand's (optional, user-controlled) backbone-freezing path — it is explicitly *not* used by the `10authors` or (after the fix) `shape` trainers**, for reasons discussed at length below.

**Literature connection**: The transfer-learning rationale — warm-starting a writer-ID CNN from an OCR-pretrained backbone because that backbone "already learned to notice pen-stroke shape as a side effect of learning to read handwriting" — is explicitly tied in the module docstring to Nguyen et al., *"Encoding CNN Activations for Writer Recognition"* (arXiv:1712.07923), which uses exactly this kind of pretrained-CNN-activation-as-writer-feature idea.

### `evaluate(model, dataloader, device, authorFolders)`

**Purpose**: Computes overall and per-author classification accuracy on held-out data.

**Inner workings**: `model.eval()`, `torch.no_grad()`; for each batch, `preds = argmax(logits, dim=1)`; accumulates both an overall `correct/total` and, by extracting each sample's author ID from its `pageKey`, per-author correct/total counts.

**Outputs**: `(overall_accuracy: float, perAuthorAccuracy: dict[str, float])`.

### `original` subcommand — `train(dataDir, cacheDir, numEpochs, batchSize, learningRate, initFromTextWeights, freezeBackbone, runName)` and `main_original()`

**Purpose**: The original standalone `TrainAuthor.py` behaviour — trains on the old 10-author set (including authors 154/155, later dropped as redundant with 150/151/152's style cluster). Superseded as the live checkpoint source but kept runnable since it exercises the shared `AuthorClassifierCNN` class.

**Inner workings**: Builds `IAMLineDatasetRaw` over `dataDir`; splits via `holdout_split_indices()`; optionally warm-starts the backbone from a `PaperCRNN` checkpoint (`load_backbone_from_text_model`) and optionally freezes it (`freezeBackbone`, offered here as a genuine user choice via an interactive prompt in `main_original()`, default "yes" — rationale given interactively: "recommended with only 10 authors' worth of data," i.e. a classical small-dataset regularization argument, freeze to reduce overfitting risk). Loss: `nn.CrossEntropyLoss()`. Optimizer: `AdamW(lr, weight_decay=1e-4)` over only the trainable (non-frozen) parameters. Scheduler: `CosineAnnealingLR(T_max=numEpochs)` — smoothly decays LR to near-zero following a cosine curve over the full training run (a different, smoother schedule than `TrainText.py`'s plateau-reactive scheduler — appropriate here because classification accuracy on a small fixed dataset tends to improve fairly monotonically rather than needing reactive LR drops). Gradient clipping at `max_norm=5.0`. Tracks `bestAcc`/`bestState` **in-memory** (a deep-copied state dict, not a disk checkpoint) across all epochs, reloads the best state at the end, and only then saves to disk — a different checkpointing strategy from the `10authors`/`shape` trainers below, which save to disk the moment a new best is found (more crash-resilient but slightly more disk I/O).

**`main_original()`**: fully interactive (`input()` prompts for every hyperparameter/path), preserving the exact original script's UX; running `python TrainAuthor.py` with zero arguments dispatches here by default.

### `10authors` subcommand — the live-checkpoint trainer

**`_encode_png(gray_crop)`**: PNG-encodes a raw numpy grayscale crop (used for personal-author samples, which arrive as numpy arrays from `SegmentLean`, not already as PNG bytes like the IAM cache).

**`add_personal_samples(base)`**: For each `PERSONAL_AUTHORS` entry, segments every one of that author's photographed pages via `SegmentLean.SegmentLines`, reads matching ground truth via `ReadLabelLines` (dropping `"MESS"`-marked lines), shuffles the page's lines with a seeded RNG (`SPLIT_SEED`), reserves the first `round(len(rows) * VAL_FRACTION)` shuffled lines as holdout, and appends each surviving line directly into `base.samples` using the **exact same dict schema** `IAMLineDatasetRaw` uses internally (`page_key`, `line_idx`, `image_png`, `target`, `text`, `is_holdout`) — meaning every downstream piece of machinery (`author_folders`, `holdout_split_indices`, `AuthorLabeledView`, `evaluate`) works on personal and IAM-derived samples completely interchangeably, with zero special-casing. This is a clean example of **designing a canonical sample schema once and reusing it across heterogeneous data sources** rather than writing a parallel code path per data source.

**`main_10authors(epochs=None)`**: Loads the IAM-derived base dataset, **filters it down** to only the 8 currently-kept `DATASET_AUTHORS` (dropping 154/155), adds the two personal authors via `add_personal_samples`, rebuilds `author_folders`/`author_to_idx` over the union of all 10. Warm-starts the backbone from the best available text-model checkpoint (a preference order resolved at import time — see below). Trains with `CrossEntropyLoss` + `AdamW(lr=1e-3, weight_decay=1e-4)` + `CosineAnnealingLR(T_max=20 by default)`, gradient clipping at 5.0. **Saves to disk the moment a new best validation accuracy is found** (not just once at the end) — the in-code comment explains why this matters here specifically: this particular run "oscillates a lot epoch to epoch (a small dataset + a fairly high LR)," so the best epoch is rarely the last one, and an early stop or crash must not lose it. After training, reloads the best saved checkpoint and reports final per-author accuracy.

**Checkpoint-preference logic at module scope** (lines ~392–396): a `for`-loop over candidate checkpoint filenames (`paper_cnn_bilstm_ctc_joint_best.pt` preferred over `paper_cnn_bilstm_ctc_hf_best.pt`) overwrites `DEFAULT_TEXT_WEIGHTS` with whichever exists first — preferring the joint (Teklia+personal, no-forgetting) checkpoint over the HF-only one as the warm-start source, on the stated reasoning that it "reads handwriting better in general," which the writer-ID backbone then inherits for free.

### `shape` subcommand — the stroke-normalized classifier, and the frozen-backbone bug fix (the report's key worked example)

**Why this model exists** (from the module docstring, worth quoting closely for the report): the `10authors` ink-based classifier reaches 100% accuracy on real held-out photographed pages, but this was found to be almost entirely driven by **ink weight** — when the same real lines are redrawn with every author's ink reduced to a centreline and re-inked at one constant width, that same model collapses to 11–16% accuracy (chance level for 10 classes is 10%). The diagnosis: the ink classifier is "reading the pen, not the hand." This matters mechanically because the robot gantry physically writes every author's reproduction with the *same* physical pen at a *constant* stroke width — ink density/pressure variation is simply not a channel the hardware can reproduce — so a judging model that leans on ink weight does not predict what the physical robot will actually achieve, and is also not how a human judges handwriting style by eye. The fix: train a **second**, independent classifier on **stroke-normalized** lines (binarized, skeletonized to a centreline, then re-inked at one fixed width, identically for every author) — this input literally cannot encode ink-weight information, so the only way the classifier can succeed is by learning slant, letterform proportions, inter-letter spacing, and connection habits: the geometric properties the gantry actually reproduces. This makes the shape classifier "the honest yardstick for physical style accuracy."

**`StrokeNormalize(gray, penWidthXh=PEN_WIDTH_XH)`**

**Purpose**: Implements exactly this ink-removal transform for a single line-image.

**Inputs**: `gray` (numpy 2-D grayscale array); `penWidthXh` (float, default 0.14 — target pen width expressed as a *fraction of the line's own x-height*, making the normalization resolution-independent).

**Outputs**: PIL `Image` (`"L"` mode) with ink re-drawn at a constant width, or `None` if the line has no ink or no extractable skeleton.

**Inner workings**: `ink = SP.BinarizeLine(gray)` (binarize via Otsu-style/adaptive thresholding, defined in `BuildStyleProfile.py`, not re-documented here); `band = SP.CoreBand(ink)` estimates the "core"/x-height band of the line; `xh = band[1]-band[0]` is the measured x-height in pixels, with a fallback (`max(8, height/3)`) if the measured band is implausibly small (<4px); the target pen width in pixels is `w = clip(round(penWidthXh * xh), 2, 9)` — a per-line-adaptive width clamped to a sane pixel range; `sk = SP.Skeletonize(ink)` reduces the binarized ink to a 1-pixel-wide centreline skeleton; `out = F.Dilate(sk, k, k)` with `k = 2*(w//2)+1` (forced odd) re-thickens that skeleton back out to the target constant width using a square structuring element; the result is inverted (`~out`) and scaled to `0/255` so it matches the "white background, black ink" convention of the rest of the pipeline before conversion to a PIL image.

**Literature connection**: Skeletonization-then-redilation to a constant width is this project's own compositional use of standard binary morphology primitives (erosion/dilation, skeleton extraction). The specific skeletonization algorithm (`SP.Skeletonize`, implemented in `BuildStyleProfile.py`/`ProfileIO.py`, not in this file) is very plausibly the classic **Zhang & Suen (1984)** thinning algorithm — a standard, widely-implemented iterative parallel thinning method frequently used for exactly this kind of stroke-centreline extraction in handwriting/OCR preprocessing; flagged here as "plausible, standard algorithm for this purpose" rather than confirmed line-by-line against this file (its implementation lives in a different file outside this document's scope).

**`BuildNormCache(force=False)`**

**Purpose**: Precomputes and caches (to a pickle file) the stroke-normalized version of every training line, since `StrokeNormalize` is relatively expensive (skeletonization) and the same normalized set is reused across every training run.

**Inner workings**: if the cache file exists and `force=False`, just unpickles and returns it. Otherwise: loads the IAM base dataset filtered to `DATASET_AUTHORS`, adds personal samples (identical filtering/joining logic to `main_10authors`), then for every sample decodes its cached PNG, applies `StrokeNormalize`, resizes the normalized image onto the fixed canvas, and stores `{author, holdout, png: uint8 array}` rows; pickles the full list to `SHAPE_NORM_CACHE`. The accompanying comment explains the cache was deliberately **renamed** (not overwriting the old `shape_norm_cache.pkl`) because the author set itself changed (154/155 dropped, yeukita/dylan added), and `BuildNormCache` trusts an existing cache file blindly — a stale cache under the old name would silently keep training on the wrong 10 authors with no error.

**`class ShapeNormalizedView(Dataset)`**: thin dataset wrapper over `BuildNormCache`'s rows — the normalization has already happened at cache-build time, so `__getitem__` only does the shared resize/augment/tensorize steps (identical to `AuthorLabeledView`).

**`Train(epochs=20, lr=1e-3, seed=0)`** — the central worked example

**Purpose**: Trains the shape (stroke-normalized) author classifier. **This is the function whose training strategy was fixed**, and the docstring states the fix explicitly and in detail — quoted/paraphrased closely below since it is the single most important piece of documented reasoning in this file.

**What it does now (current, fixed code)**: `AuthorClassifierCNN` is constructed, its backbone is warm-started from `SHAPE_TEXT_WEIGHTS` (the best available `PaperCRNN` text-model checkpoint, via `load_backbone_from_text_model`) — but **critically, `freeze_backbone()` is never called**, and the model is left in `.train()` mode for all `epochs` (default 20). The optimizer (`AdamW(lr=1e-3, weight_decay=1e-4)`) is constructed over **all** `model.parameters()`, not a filtered trainable subset — so every weight in the backbone *and* the head receives gradient updates every step, exactly the same full-fine-tuning recipe `main_10authors` uses for the ink classifier. `CosineAnnealingLR(T_max=epochs)` schedules LR smoothly to zero over the run. Gradient clipping at 5.0. The best validation-accuracy state dict is kept in-memory and reloaded at the end, then saved to `SHAPE_OUT_WEIGHTS`.

**What the old (buggy) version did, and why it failed — the worked example**: the docstring states explicitly that the previous version "only trained `author_head` on features extracted ONCE from the frozen, text-model-warm-started backbone (500 epochs, lr 3e-3, head only)." In precise terms: the backbone's convolutional weights were copied in from a `PaperCRNN` checkpoint, then every one of those parameters had `requires_grad` set to `False` (the `freeze_backbone()` method, defined earlier in this file, was being called) — meaning during backpropagation, gradients were still computed with respect to inputs flowing through those layers (necessary to reach the still-trainable head), but **no gradient was ever computed or applied with respect to the frozen layers' own weights**, so those weights never moved from their OCR-pretrained values for the entire 500-epoch run. Only `author_head`'s own weights (`Linear(128,96)→ReLU→Dropout→Linear(96,10)`) were updated by the optimizer.

**Why this is structurally the wrong choice for this exact task** (the conceptual core the report should build on): `PaperCRNN`'s backbone was trained, via the CTC loss, to solve character recognition — i.e. to predict *what word/letter is being written*, invariant to *who* is writing it or *how stylistically* they form their letters. A well-trained OCR backbone is explicitly supposed to be insensitive to exactly the signal a writer classifier needs: slant, letterform shape, proportion, and stroke style are, from the text-recognition task's point of view, *nuisance variation to be discarded*, not *target signal to be preserved*. Freezing that backbone and only training a small head on top of its output forces the writer-ID classifier to work entirely from whatever of that signal happened to survive as a side effect of OCR training — which, per the ink-weight discovery in this file's own module docstring, turned out to be mostly texture/ink-density artifacts rather than genuine geometric style, once the input was stroke-normalized to remove ink-weight entirely. In other words: on raw ink input the frozen-backbone/ink-weight failure mode was *masked* (the model could cheat using ink density, something `main_10authors`'s own fix later also had to confront conceptually by building this whole `shape` model), but once the input is stroke-normalized specifically to remove that cheat, a frozen OCR backbone has *nothing else useful left to report*, because the one thing it was never asked to preserve (letterform geometry) is the only thing left in the input.

**Measured effect reported in-code**: pooled validation accuracy with the frozen-backbone approach "looked fine" (95.3%), masking the real problem — this is itself a notable methodological point for the report (aggregate validation accuracy can look healthy while badly failing on specific, practically important cases). The real-photo confusion matrix told a different story: author `owen`'s lines were classified as `"yeukita"` 80% of the time (118/148), and `abhinav`'s lines were split between `"552"` and `"thiya"`. The docstring specifically rules out "the two styles just genuinely look alike" as an innocent explanation by comparing raw pixel-space distances: owen's own lines sit *closer* to each other (mean L2 distance 11118) than to yeukita's lines (mean L2 distance 15599) in the model's raw 64×640 input space — i.e. the raw images themselves are more self-similar within-author than cross-author, so the confusion is a property of what the frozen-backbone+shallow-head model was capable of learning, not an inherent ambiguity in the data. As corroborating evidence, the **ink classifier** (which fine-tunes its whole network on the same underlying crops, just not stroke-normalized) gets owen right 147/148 — strong evidence the fix (letting the backbone itself adapt) is what closes the gap, not some unrelated data-quality issue.

**The fix in one sentence** (as stated in-code): "the fix is to let this classifier's backbone learn from the normalized domain too, instead of relying on frozen OCR features it was never trained to need." Concretely: unfreeze every backbone parameter, keep the OCR weights only as a sensible *initialization* (a warm start that still saves training time and gives the model a head start on general stroke/edge detection, exactly as citied via arXiv:1712.07923), and let full backpropagation through the whole network, for all 20 epochs, adapt those convolutional filters specifically toward whatever features actually discriminate ten writers' stroke geometry — something a text-recognition objective alone never asked them to do.

**What "frozen" vs "fine-tuned" mean in parameter-update terms, for the report's design-and-implementation section**: "frozen" = `requires_grad=False` on a parameter tensor, meaning PyTorch's autograd engine still computes gradients *through* that tensor during backpropagation (if needed to reach earlier layers or in this case it's not needed further upstream, but regardless the point stands for layers before it) but never accumulates a `.grad` for that tensor itself, and the optimizer — correctly constructed here over only `[p for p in model.parameters() if p.requires_grad]` in the `original` subcommand's freeze path — never receives that parameter in its update step, so its value is mathematically frozen at whatever it was initialized to. "Fine-tuned" (here, "fully fine-tuned") = every parameter has `requires_grad=True` (the PyTorch default) and is included in the optimizer's parameter list, so ordinary gradient descent (`AdamW`'s moment-based update rule applied to `∂Loss/∂θ` for every `θ`) updates every weight, including the ones that started from a pretrained OCR checkpoint — the pretrained values only act as the *initial point* of gradient descent, not a fixed constraint.

**`main_shape()`**: simply calls `Train()` with its defaults (20 epochs, lr=1e-3, seed=0).

### Argument parsing / dispatch (`build_arg_parser`, `main`)

Three subparsers: `original`, `10authors` (positional optional `epochs` argument, default 20, matching the original `TrainAuthor10.py <epochs>` CLI), `shape`. Zero arguments dispatches to `original` by default, preserving the pre-merge standalone script's behaviour.

---

## 3. `Software/CNN/ExtractIAMLines.py`

### Role in the pipeline

This is the **line-segmentation and label-alignment utility** both training scripts above depend on (via `LineImageCache`/`IAMLineDatasetRaw` in `TrainText.py`, and transitively in `TrainAuthor.py`). It takes a full scanned/photographed IAM-style page image and (a) detects the printed-prompt vs. handwriting regions, (b) segments the handwriting region into individual line crops, and (c) — for the IAM "Sentence Database" form layout specifically — OCRs the printed prompt text and apportions it across the detected handwriting lines to produce automatically-generated, approximate ground-truth labels (used as a starting point, later hand/algorithmically corrected elsewhere in the project). Functional block boundary note: this file sits at the boundary between **Data Processing** and **Training & Classification** — it is documented here because it is a direct, load-bearing dependency of the training pipeline (`LineImageCache` calls `ExtractLinePatches` directly), but its internal image-processing algorithms (Otsu thresholding, projection-profile line splitting, periodicity-based splitting) are themselves classical computer-vision techniques, not neural-network code.

### `OtsuThreshold(grayImg)`

**Purpose**: Binarizes a grayscale image using Otsu's automatic thresholding method, producing a black/white (ink/background) mask.

**Inputs**: `grayImg`, 2-D numpy array, grayscale pixel values 0–255.

**Outputs**: `uint8` array, same shape, values `{0, 255}` — `255` where `grayImg < bestThresh` (i.e. **dark pixels, assumed ink**, become the "on" value — this is an *inverted* binarization: ink → white/255 in the output, consistent with the "ink = high value" convention used across this codebase).

**Inner workings and maths**: Computes a 256-bin histogram of the image. For each candidate threshold `t` from 0 to 255, incrementally maintains: `wB` = cumulative pixel count below `t` (background-class weight so far), `wF = total - wB` (foreground weight), `sumB` = cumulative intensity-weighted sum below `t`, from which the two classes' means are `mB = sumB/wB` and `mF = (sum1-sumB)/wF` (`sum1` is the whole image's weighted sum, precomputed once). The between-class variance is `varBetween = wB · wF · (mB - mF)²`. Otsu's method picks the threshold `t*` that **maximizes** this between-class variance — equivalently (by a standard identity) **minimizes** within-class variance, since total variance is fixed and splits additively into within+between components. The threshold with the highest `varBetween` seen so far is tracked incrementally in a single pass.

**Literature connection**: Otsu, N. (1979), *"A Threshold Selection Method from Gray-Level Histograms"* — this is a textbook, by-the-book from-scratch implementation of the canonical global image-thresholding algorithm.

### `LongestRun(boolRow)`

**Purpose**: Small helper — finds the length of the longest contiguous run of `True` values in a 1-D boolean array, used to detect long straight printed rule lines (which show up as one very long contiguous ink run across a row).

**Inner workings**: Finds indices where the boolean array is `True` (`inkCols`), finds gaps between consecutive true-indices greater than 1 (`np.diff(...) > 1`), uses those gap positions to delineate contiguous run boundaries (`starts`/`ends`), and returns the maximum `end - start + 1` over all runs.

### `ClusterRows(rows, maxGap=5)`

**Purpose**: Groups a sorted list of row indices into clusters where consecutive members are within `maxGap` of each other, replacing each cluster with its mean — used to collapse several adjacent pixel-rows that all belong to the same thick/anti-aliased printed rule line into a single representative row coordinate.

### `FindRuleLineClusters(binaryImg, rawGrayscaleArray=None)`

**Purpose**: Scans every row of the page for evidence of a long, mostly-continuous horizontal printed rule line (used to find the page's template boundaries — top of the handwriting-prompt box, bottom of the writing area, etc.).

**Inner workings**: For each row, if `rawGrayscaleArray` is given, thresholds that row directly at a fixed cutoff (`<225`, i.e. the raw pixel is dark) rather than relying on the already-Otsu-binarized image — the in-code comment explains some printed divider lines are too faint to survive Otsu's *global* threshold on some scans, so the raw grayscale is preferred when available, with the binary image kept only as a fallback. A row qualifies as a rule-line candidate if its longest contiguous ink run exceeds 25% of the image width **and** its total ink coverage exceeds 10% of the width (two independent conditions — guards against, respectively, a row with lots of scattered short ink marks but no one long line, and a row with one genuinely long but very thin/sparse run). Qualifying row indices are then clustered via `ClusterRows`.

### `DetectPageBounds(binaryImg, rawGrayscaleArray=None, return_confidence=False)`

**Purpose**: Finds the top (`topY`) and bottom (`botY`) y-coordinates bounding the handwriting-writing region of an IAM-style form page.

**Inner workings**: Gets rule-line row clusters via `FindRuleLineClusters`. Filters candidates to plausible fractional-height windows (`topY` must fall in `(0.12h, 0.40h)`; `botY` in `(0.55h, 0.92h)` — these windows encode an assumption about the IAM Sentence-Database form's layout proportions). If a real candidate is found in each window, `topY`/`botY` are set 8 pixels inside the detected rule line (a small safety margin); otherwise each falls back to a **fixed percentage of page height** (`0.12h`/`0.80h`) — a pure guess with no content-based justification, which is exactly the failure mode `PageBoundsAreUncertain` below exists to flag. With `return_confidence=True`, also returns two booleans indicating whether the top/bottom bound had to use this fallback guess.

### `HeaderGapIsSafe(binaryImg, topY, min_gap_px=10, max_ink_fraction=0.05)`

**Purpose**: An independent, content-based sanity check on `topY`: verifies the rows immediately *above* `topY` are mostly blank, as they should be if `topY` genuinely sits in the whitespace gap right after the printed header's last line of text (rather than on top of that text).

**Inner workings**: Extracts the `min_gap_px` rows immediately above `topY`, computes the fraction of ink pixels in each such row, and returns `True` only if **no** row in that zone exceeds `max_ink_fraction` (5%) ink coverage — i.e. the whole checked band must be near-blank.

### `PageBoundsAreUncertain(imgPath)`

**Purpose**: A public, callable per-page quality gate — returns `True` if this page's bounds detection is risky enough that auto-generated labels for it should **not** be trusted (and presumably skipped by whatever downstream script calls this, outside the scope of this file).

**Inner workings**: Loads and binarizes the page, runs `DetectPageBounds` with confidence reporting; returns `True` immediately if `topY` had to fall back to the fixed-percentage guess; otherwise also checks `HeaderGapIsSafe` and returns `True` if that check fails (i.e. even a "confidently detected" `topY` can still be flagged if the independent ink-based check disagrees). This two-pronged check is a deliberate defence against two *different* failure modes: "we couldn't find a rule line at all" vs. "we found *a* rule line candidate in the right height-window, but it isn't actually the boundary we think it is."

### `DetectHeaderTopY(binaryImg, rawGrayscaleArray, topY)`

**Purpose**: Finds the rule line bounding the *top* of the printed sentence-prompt box (separating it from the page's own title/header text above), used to crop OCR input tightly around just the prompt text.

**Inner workings**: Finds rule-line clusters in the zone between 4% of page height and `topY - 8`; if any exist, returns the highest (max) one plus 8px margin; otherwise falls back to a fixed `0.115h`. The in-code comment gives a concrete, verified motivation: a fixed-percentage fallback clips the first printed line on IAM form templates that place this box higher than others, confirmed to garble OCR specifically on several real `"c03"`-prefixed IAMpages671 pages where lines 3+ OCR'd correctly but lines 1–2 were consistently mangled — i.e. this function exists because a simpler fixed-percentage guess was empirically shown to fail on real data.

### `ExtractPrintedGroundTruth(rawGrayscaleArray, topY, headerTopY=None)`

**Purpose**: Crops the printed-prompt region (between the header and the handwriting area) and OCRs it with Tesseract to recover the expected printed words — later apportioned across the detected handwriting lines as an automatic initial transcript.

**Inner workings**: Crops `[y1:y2, x1:x2]` where `y1 = headerTopY` (or a fixed fallback), `y2 = topY - 8`, and `x1`/`x2` trim 6% off each horizontal edge; runs `pytesseract.image_to_string(..., config='--psm 6')` (`--psm 6`: Tesseract's "assume a single uniform block of text" page-segmentation mode); cleans the OCR'd text by collapsing newlines to spaces, regex-stripping the literal phrase `"Sentence Database"` and `NN-NNNa`-style page-ID codes, removing stray pipe/letter-`j` OCR artifacts; splits into a word list. Wrapped in a broad `try/except` that logs a warning and returns an empty list on any OCR failure, rather than propagating an exception.

**Literature connection**: Uses Tesseract (an existing, long-standing open-source OCR engine, originally developed at HP and now maintained by Google) as an off-the-shelf tool for ground-truth bootstrapping — not a component this project trains or re-implements itself.

### `ReadLabelLines(imgPath, labelPath=None)`

**Purpose**: Reads a page's (hand-corrected or auto-generated) per-line transcript file.

**Inputs**: `imgPath` (used to derive the default label path as `<stem>_labels.txt` if `labelPath` isn't given explicitly — needed for datasets like `PersonalDataset` that keep images and labels in separate `Pages`/`Labels` folders).

**Outputs**: `list[str]`, one line of text per non-blank line in the file.

**Inner workings**: Reads all non-blank, stripped lines; strips a leading `"<digit(s)>\t"` prefix from each (the label-file format is `"<line_number>\t<text>"`), so the numeric index never leaks into a training target as a literal digit character.

### `CountWordsFromInkColumns(inkCols, lineHeight, minGapFactor=0.6, minRunWidth=3)`

**Purpose**: Estimates how many space-separated words are physically present in one detected handwriting line, by analysing the actual ink layout rather than assuming a fixed character width or using the line's overall pixel width.

**Inputs**: `inkCols` (sorted array of column indices containing ink); `lineHeight` (used to scale the word-gap threshold — bigger handwriting has proportionally bigger inter-word gaps too); `minGapFactor` (the gap-width-to-line-height ratio above which a gap counts as a word boundary, default 0.6); `minRunWidth` (minimum run width in pixels to count as a real letter-run rather than noise, default 3).

**Inner workings**: Groups contiguous ink columns into runs (same gap-finding technique as `LongestRun`); discards runs narrower than `minRunWidth` (stray marks/noise); computes `wordGapThreshold = lineHeight * minGapFactor`; counts a new word every time the gap between consecutive surviving runs exceeds that threshold (starting the count at 1 for the first run). The module docstring explicitly motivates this over a pixel-width-proportional apportioning scheme (used by the sibling function `SplitWordsIntoLines`'s fallback): width-based guessing assumes roughly constant character width across the whole page, which real handwriting violates often enough to cause words to visibly drift onto the wrong line in the generated transcript.

### `SplitWordsIntoLines(words, lineWordCounts)`

**Purpose**: Distributes the OCR'd printed-prompt word list across the `N` detected handwriting lines in proportion to each line's estimated word count (from `CountWordsFromInkColumns`), producing the auto-generated per-line label text.

**Inner workings**: Computes `scale = numWords / totalEstimated` so the *scaled* line counts sum to exactly the known total word count (correcting for `CountWordsFromInkColumns`'s estimate being systematically a little over or under); hands out words to each line in order, using `round(lineWordCounts[i] * scale)` words per line (clamped to not exceed remaining words, and with the very last line simply taking whatever words remain — avoiding rounding-error drift accumulating to the end). Falls back to an even 1-word-per-line split if ink-based counting found nothing usable at all (e.g. extremely faint scan).

### `RemoveStraightFormLines(hwRegion)`

**Purpose**: Removes any long straight printed-rule-line pixels that leaked into the handwriting region crop (e.g. a faint continuation of a form's ruled boundary), so they don't get mistaken for handwriting ink during line splitting.

**Inner workings**: Reuses the same "long contiguous run + high total coverage" rule-line detection logic as `FindRuleLineClusters` (duplicated locally rather than refactored into a shared helper — a minor code-duplication note), and for any row meeting that criterion, zeroes out a small band (`row-2` to `row+3`) around it.

### `MergeUndersizedFragments(lines, minRelativeHeight=0.4)`

**Purpose**: Merges spuriously-small detected "line" segments (typically a descender tail, dangling punctuation mark, or stray ink speck that got split off as its own tiny region) back into whichever real neighbouring line is closer.

**Inputs**: `lines`, a list of `(startRow, endRow)` tuples.

**Inner workings**: Computes each segment's height, takes the **median** height across all segments as the reference "real line height" (robust to a few genuinely tiny fragments skewing a mean), sets `threshold = median * minRelativeHeight` (0.4 by default). Repeatedly (a `while changed` loop) finds any segment shorter than `threshold`, determines its nearer neighbour by whichever adjacent gap is smaller (or the only available neighbour if it's at either end of the list), and merges the two segments' row ranges together; repeats until no undersized segment remains or only one segment is left. The in-code comment gives the measured scale that justifies the specific threshold: real lines run roughly 100–130px tall on this dataset, while stray fragments run roughly 15–20px — a clear, well-separated size gap that height alone reliably distinguishes, regardless of how large a vertical gap happens to separate the fragment from its real line.

### `MergeOrSplitToExpectedLineCount(lines, rowInkCount, expectedLineCount)`

**Purpose**: When a ground-truth line **count** is already known (from a hand-verified label file), forces the detected segment count to match it exactly by iteratively merging the "cheapest" pair of adjacent segments (if too many were detected) or splitting the largest segment at its weakest ink point (if too few were detected).

**Inner workings — too many segments**: repeatedly computes, for every adjacent pair, a merge "cost" = `gap (in pixels) + mean ink density within the gap` (so a pair separated by a wider or inkier gap is considered a *worse*, i.e. less justified, merge candidate), and merges the pair with the **minimum** such cost — i.e. merges the two segments whose boundary looks least like a real line break.

**Inner workings — too few segments**: among segments taller than 42px (a minimum splittable size), picks the **largest**, and splits it at the row with the **minimum ink count** (`argmin`) within its interior (excluding an 18px margin at each end, to avoid creating a sliver) — i.e. splits at the most plausible "gap between two lines" point inside an oversized merged region. If no candidate is tall enough to safely split, the loop just stops (doesn't force an impossible split).

### Periodicity-based splitting — for densely-written / ruled-notebook pages

**`EstimateLinePitch(smoothedRowInk, minPitch=40, maxPitch=400)`**

**Purpose**: Estimates the expected vertical spacing ("pitch") between consecutive handwritten lines directly from the page's own ink-density signal, via autocorrelation — rather than assuming a fixed spacing.

**Inner workings**: Mean-centres the smoothed row-ink-density signal; computes its full autocorrelation (`np.correlate(sig, sig, mode='full')`) and keeps only the non-negative-lag half; restricts the search to lags between `minPitch` and `maxPitch`; returns the lag with the **maximum** autocorrelation in that window — i.e. the spacing at which the signal most strongly "repeats itself," a standard signal-processing technique for periodicity detection.

**Literature connection**: Autocorrelation-based periodicity estimation is a standard digital signal processing technique, not specific to any one paper.

**`SplitLinesByPeriodicity(binaryRegion, ruleLineHints=None, smoothWindow=9, minLineHeight=30)`**

**Purpose**: An alternative line-splitter for pages where lines sit close enough together that cursive ascenders/descenders physically bridge adjacent lines, leaving **no** genuinely blank row anywhere for the gap-based splitter to find (e.g. densely-written ruled notebook paper, as used for the project's own personal-author pages).

**Inner workings**: Computes the per-row ink count (`rowInk`), smooths it with a `smoothWindow`-wide moving-average box filter (`np.convolve`); estimates the expected line pitch via `EstimateLinePitch`; finds **local minima** of the smoothed signal (via `scipy.signal.find_peaks` on the *negated* signal) with a minimum inter-peak distance of `max(minLineHeight, 0.55*pitch)` and a minimum prominence scaled to the page's own peak ink density — i.e. it looks for the smoothed ink profile's *relative* low points (thinner-ink rows between lines, even if never literally zero), spaced out consistently with the estimated line pitch. If `ruleLineHints` (optional faint-ruled-line row positions) are supplied, any detected valley within a small snap-window of a hint is pulled exactly onto that hint — a light correction nudge, not a hard requirement (the function works from ink periodicity alone if no hints are given, since the module docstring notes ruled-line *pixel* detection on real photographs was found too unreliable to trust as the primary signal). The detected valleys become line-boundary cut points; any resulting zero-ink segment is dropped.

**Literature connection**: This is a project-specific heuristic combining standard signal-processing primitives (smoothing, peak-finding, autocorrelation); not a reproduction of a specific published line-segmentation algorithm.

### `ExtractLinePatches(imgPath, targetHeight=32, maxWidth=1024, expectedLineCount=None, labelLines=None, is_dataset=True)`

**Purpose**: The top-level orchestration function — given one page image, produces a list of segmented, resized line crops plus a matching list of label texts (either passed in directly, or auto-generated via OCR+apportioning), plus a visual debug preview image with bounding boxes drawn.

**Inputs**: `imgPath` (path to a page image); `targetHeight`/`maxWidth` (the fixed-height, max-width canvas each `processed_patch` is resized/padded onto — **note**: a *different*, older preprocessing convention from `TrainText.py`'s own `INPUT_HEIGHT/WIDTH=64/640` used for the actual CRNN training canvas; `LineImageCache` only keeps this function's `raw_crop` output, re-resizing separately via `resize_line_image_fixed`, specifically to avoid depending on this function's own resize convention); `expectedLineCount` (optional int, forces segment count via `MergeOrSplitToExpectedLineCount` when a hand-verified count is known); `labelLines` (optional list, used to populate each sample's `label_text` field directly when already known, bypassing the OCR/apportioning path); `is_dataset` (bool — selects between the IAM-form-specific pipeline and a generic non-IAM personal-page pipeline, see below).

**Inner workings, `is_dataset=True` path (IAM Sentence-Database form layout)**:
1. Loads the image, converts to grayscale, Otsu-binarizes.
2. `DetectPageBounds` → `topY`, `botY`; `DetectHeaderTopY` → `headerTopY`.
3. `ExtractPrintedGroundTruth` → OCR'd prompt word list.
4. Defines `handwritingStartY = topY + HANDWRITING_TOP_MARGIN(=15)` — a hard, always-on buffer below the detected boundary, specifically so a training line crop's top edge can never graze the last row or two of printed header text even if the rule-line detection is off by a few pixels (a documented deliberate trade: loses a negligible sliver of the true first handwritten line's very top, in exchange for structurally guaranteeing no header leakage).
5. Crops `hwRegion = binaryImg[handwritingStartY:botY, :]`; removes any residual straight printed rule lines (`RemoveStraightFormLines`); masks out tall vertical margin-rule columns (`colInkHeights > 0.40*regionH`) and a small horizontal margin at each side (7%/97% of page width).
6. Computes the row-ink-count profile; detects "has-ink" rows (`rowInkCount > max(10, 0.006*width)`); applies a 7-tap one-sided dilation (`smoothedInk`, a manual OR-based smoothing loop) to bridge small gaps; finds contiguous `True` runs via a diff-of-booleans trick (`np.diff` on the 0/1-cast, zero-padded signal) — each such run is a candidate line; drops any run shorter than 8px.
7. Merges descender fragments: any two consecutive candidate lines separated by less than 18px get merged (handles 'y'/'p'/'g' descender tails bridging into the next line's detected region).
8. Runs `MergeUndersizedFragments` then (if `expectedLineCount` given) `MergeOrSplitToExpectedLineCount`.

**Inner workings, `is_dataset=False` path (generic/personal pages, e.g. ruled notebook photographs)**: Skips all IAM-form-specific detection — uses almost the whole page (minus a 1%-height margin) as the writing region directly, and performs **no** printed-text OCR (there's no known transcript source on a personal page; `printedWords=[]`). Line splitting switches to `SplitLinesByPeriodicity` (the blank-gap method isn't safe to assume here) followed by `MergeUndersizedFragments`. The docstring explains concretely *why* the IAM-specific path would actively misfire on a non-IAM page, not just fail to help: `DetectPageBounds`'s rule-line search can lock onto arbitrary ordinary ruled-paper lines that happen to fall in the height-fraction windows it expects an IAM form's template rules to occupy, which would then (a) silently discard real handwriting lying outside that accidentally-chosen window, and (b) feed real handwriting pixels into the OCR call as if they were printed prompt text, producing garbage "ground truth."

**Final steps (both paths)**: Draws a debug preview (`drawCanvas`, an RGB copy with blue lines marking `topY`/`botY`/`handwritingStartY` and numbered red boxes around each detected line, in sync with the returned line-text index — useful for manually auditing segmentation quality against the generated labels). For each final `(s, e)` line segment: computes absolute pixel bounds with small padding margins (5px vertical, 8px horizontal, derived from the actual ink column extent within that band, not the raw segment bounds); crops the line from the **raw grayscale** (not the binary) image; resizes it (Lanczos) to `targetHeight`, preserving aspect ratio, capped at `maxWidth`; pads onto a white `maxWidth × targetHeight` canvas (**left-aligned**, no vertical centring here — a different, simpler padding convention than `TrainText.py`'s `resize_line_image_fixed`, which is why the cache keeps `raw_crop` separately and redoes the resize downstream with the training-time convention instead of trusting this function's own `processed_patch`). Appends a dict per line: `{line_idx, raw_crop, processed_patch, label_text}`.

**Outputs**: `(ExtractedLineSamples: list[dict], formattedTextLines: list[str], drawCanvas: PIL.Image)`. `formattedTextLines` is either the passed-in `labelLines` directly (if its length already matches the detected line count) or the result of `SplitWordsIntoLines` applied to the OCR'd prompt words and each line's estimated word count.

**Literature connection**: This whole function is a hand-engineered, rule-based page-layout and line-segmentation pipeline — a classical document-image-analysis approach (projection profiles, rule-line detection, periodicity analysis) rather than a learned segmentation model. Not attributable to one specific paper; it combines several standard, generic document-analysis techniques (horizontal projection profiling for line segmentation is itself a decades-old standard OCR preprocessing technique).

### `__main__` block

A standalone demo/test harness: processes every page image in `Data/Datasets/IAMpages10/150`, calling `ExtractLinePatches` with `expectedLineCount` derived from any existing hand-verified label file, saving a bounding-box preview image and per-line crop+label files to `handwriting_line_extraction_output/`, and reporting how many pages' detected line count matched the expected count. Useful for visually auditing segmentation quality but not itself part of the training pipeline.

---

## 4. `Software/CNN/authors_config.py`

### Role in the pipeline

A small, deliberately centralized configuration module — "single source of truth for the active 10-author set and the shared train/holdout split convention." Imported by `TrainText.py`, `TrainAuthor.py`, and (per its docstring) other scripts such as `BuildStyleProfile10Authors.py`. Its stated purpose is explicitly to prevent configuration drift: these values were previously duplicated independently across `TrainAuthor10.py`, `TrainTextJoint.py`, and `BuildStyleProfile10Authors.py`, and could silently fall out of sync, changing which lines a model was trained or evaluated on without anyone noticing.

### Module-level constants

- `DATASET_AUTHORS = ["150","151","152","153","384","551","552","588"]` — the 8 IAM dataset authors currently in active use (154/155 dropped for being redundant with 150/151/152's style cluster, per `TrainAuthor.py`'s docstring).
- `PERSONAL_AUTHORS = ["abhinav","dylan","owen","thiya","yeukita"]` — note this list has **5** entries in this shared config, while `TrainAuthor.py`'s own docstring and `main_10authors` specifically describe only **2** of them (`yeukita`, `dylan`) as part of the live 10-author writer-ID set; the other three (`abhinav`, `owen`, `thiya`) appear in this shared list presumably for use by other scripts (e.g. text-recognition personal fine-tuning, which the docstring says covers "every `authors_config.PERSONAL_AUTHORS` entry") and are *not* necessarily all included in the 10-author writer-ID classifier specifically — worth flagging as a point to double-check against `server.py`/the live checkpoint's actual `author_mapping` if precise author counts matter for the report.
- `VAL_FRACTION = 0.15`, `SPLIT_SEED = 0` — the shared per-page train/holdout split fraction and RNG seed used by every script that splits a personal author's photographed pages, guaranteeing the same physical lines are held out everywhere in the project rather than three different, inconsistent "holdout" definitions.
- `AUTHOR_CLASSIFIER_RUN_NAME = "author_classifier_10new"` — the checkpoint run-name `TrainAuthor.py 10authors` saves under; `server.py` derives its expected checkpoint filename from this same constant rather than the two being kept in sync by hand.

### `author_classifier_weights_filename()`

**Purpose**: Returns `f"{AUTHOR_CLASSIFIER_RUN_NAME}_weights.pt"` — the canonical live-checkpoint filename, computed once here so any caller needing it (e.g. `server.py`) never hardcodes it separately.

### `personal_author_pages(author)`

**Purpose**: A generator yielding `(img_path, label_path)` pairs for one personal author's photographed pages, reading directly from the single maintained `PersonalDataset/<Author>/{Pages,Labels}/` location.

**Inner workings**: Builds `folder = PERSONAL_DATASET_DIR / author.capitalize()`; globs `pagesDir.glob(f"{author.capitalize()}*.jpg")` sorted by filename; for each, yields the image path paired with the corresponding `<stem>_labels.txt` path in the sibling `Labels` folder. The in-code comment explains the design choice: reading directly from this one canonical location, rather than maintaining a `NOGIT/<author>` duplicate copy per script, avoids every script needing to keep a separate copy in sync, and avoids needing a `.gitignore` allowlist entry to get that duplicate data to a cloud training environment (e.g. Colab).

---

## 5. `Software/CNN/AuthorReproductionStuff/np_inference/weight_io.py`

### Role in the pipeline

This file implements a **from-scratch PyTorch checkpoint reader that never imports `torch`** — it parses the `.pt` file format directly via Python's standard `pickle`/`zipfile` modules and reconstructs tensor contents as plain numpy arrays. It is the foundation the rest of the `np_inference` package (`layers.py`, `text_model.py`, `author_model.py`) builds on, enabling a torch-free inference path for both the text recognizer and the writer classifier — useful for deployment contexts where installing the full PyTorch package is undesirable. Functional block: **Training & Classification** (specifically, the inference/deployment side of it).

### Background: the `.pt` file format, explained for the report

A PyTorch checkpoint saved via `torch.save(...)` is, by default, a **zip archive** (not a raw pickle stream) containing:
- One member, conventionally `<archive_name>/data.pkl` — a standard Python `pickle` byte stream encoding the *object graph* being saved (e.g. a `dict` mapping parameter names to tensor objects, or an `OrderedDict` state dict). Crucially, the actual tensor **data bytes are not embedded directly in this pickle stream**; each tensor is represented in the pickle as a *persistent reference* (an opaque identifier that pickle's protocol defers resolving to an external `persistent_load` hook rather than unpickling inline).
- One member per distinct underlying storage buffer, named `<archive_name>/data/<key>` — the tensor's raw bytes, stored flat, uncompressed (or possibly DEFLATE-compressed depending on zip settings), in whatever dtype the tensor was saved as (e.g. 4-byte little-endian floats for a `FloatStorage`).

This split exists because pickle alone is a poor format for large binary blobs (base64-style escaping inflates size and is slow); PyTorch's save format instead keeps the *metadata and structure* in the pickle stream and the *raw numeric payload* as separate, directly-addressable zip members, letting pickle's `persistent_load` mechanism splice the two back together during loading.

### `_STORAGE_DTYPES` (module-level dict)

Maps every `torch.*Storage` class name PyTorch might reference in the pickle (`FloatStorage`, `DoubleStorage`, `HalfStorage`, `LongStorage`, `IntStorage`, `ShortStorage`, `CharStorage`, `ByteStorage`, `BoolStorage`) to the equivalent numpy dtype string, all explicitly little-endian (`<f4`, `<i8`, etc.) — this is what lets the code reconstruct a tensor's numeric type without ever importing torch's own storage classes.

### `class _StorageStub`

**Purpose**: A lightweight stand-in object returned whenever the custom unpickler encounters a reference to a real `torch.FloatStorage` (etc.) class during unpickling — since the real torch class isn't available (and isn't wanted — the whole point is avoiding the torch dependency), this stub carries only the one piece of information actually needed: the corresponding numpy dtype.

**`__call__`**: a defensive no-op (`return self`) in case some pickle stream calls the storage "class" directly rather than only referencing it — rare, but kept harmless rather than letting it crash.

### `_rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad=False, backward_hooks=None, metadata=None)`

**Purpose**: Reimplements PyTorch's own internal `torch._utils._rebuild_tensor_v2` function — the function PyTorch's *own* unpickling machinery calls to reconstruct a tensor object from its underlying flat storage array plus view metadata (offset, logical shape, strides). By registering this **numpy** reimplementation in place of torch's real one (see `_TorchUnpickler.find_class` below), the exact same pickle byte stream that would normally produce a `torch.Tensor` instead produces a plain `numpy.ndarray` with identical numeric content and shape.

**Inputs**: `storage` (a flat 1-D numpy array — the decoded raw bytes for this tensor's underlying buffer, produced by `persistent_load` below); `storage_offset` (int, element offset into that flat buffer where this tensor's data begins — relevant when multiple tensors/views share one underlying storage, e.g. two tensors that are reshaped/sliced views of the same memory); `size` (tuple, the tensor's logical shape); `stride` (tuple, **in elements**, PyTorch's convention for how many storage elements to skip to advance one index along each dimension); `requires_grad`, `backward_hooks`, `metadata` (accepted for pickle-stream compatibility, unused — this reconstruction is inference-only, gradients are irrelevant).

**Inner workings and maths**: Converts `stride` from elements to bytes (`byte_strides = tuple(s * itemsize for s in stride)`) — numpy's `as_strided` needs byte strides, while PyTorch's metadata records element strides, so this unit conversion is essential and easy to get wrong. Handles the 0-dimensional (scalar tensor) case specially (`size == ()`) by slicing out exactly one element and reshaping to `()`. Otherwise, constructs a numpy *view* onto the storage buffer using `np.lib.stride_tricks.as_strided(storage[storage_offset:], shape=size, strides=byte_strides)` — this is exactly how PyTorch itself represents a tensor internally: a flat contiguous (or non-contiguous-but-strided) buffer plus a `(shape, stride, offset)` triple describing how to interpret it as an N-dimensional array, without copying data. The final `.copy()` then **materializes** that view into an independent, contiguous numpy array — necessary because `as_strided` views are not memory-safe to keep around indefinitely (they can alias memory in surprising ways, especially with non-standard strides), so copying once immediately after construction is the safe choice for a long-lived returned object.

**Literature connection**: Not a published algorithm — this is a direct reverse-engineering/reimplementation of PyTorch's internal serialization-format contract, documented in PyTorch's own (open) source code rather than any academic paper.

### `class _TorchUnpickler(pickle.Unpickler)`

**Purpose**: A custom subclass of Python's standard `pickle.Unpickler` that intercepts specific class references the pickle stream would normally resolve to real `torch.*` classes, substituting numpy-only equivalents, and implements `persistent_load` to splice in the actual raw tensor bytes from the zip archive.

**`__init__(self, file, zf, data_prefix)`**: stores a reference to the open `zipfile.ZipFile` (`zf`) and the archive's internal path prefix (`data_prefix`, e.g. `"archive_name/"`), both needed later by `persistent_load` to find the right zip member for a given tensor.

**`find_class(self, module, name)`**: This is pickle's standard extension point — whenever the pickle stream references a class by its `(module, name)` pair (e.g. `("torch._utils", "_rebuild_tensor_v2")`), the unpickler calls `find_class` to resolve what Python object that reference should actually produce. This override intercepts exactly three/four specific cases:
- `("torch._utils", "_rebuild_tensor_v2")` → returns the numpy reimplementation above, so every tensor-reconstruction call in the pickle stream transparently produces a numpy array instead of crashing on a missing `torch` import.
- `("torch", <StorageClassName>)` for any name in `_STORAGE_DTYPES` → returns a `_StorageStub` configured with the matching numpy dtype.
- `("collections", "OrderedDict")` → returns the real `collections.OrderedDict` (state dicts are commonly `OrderedDict`s; this one is safe and genuinely available, so it's passed through rather than stubbed).
- `("torch._utils", "_rebuild_parameter")` → returns a lambda that just returns `data` unchanged, discarding `requires_grad`/`backward_hooks` — an `nn.Parameter` is, for inference purposes, indistinguishable from its underlying tensor data, so no special reconstruction is needed.
- Anything else falls through to `super().find_class(module, name)` — ordinary pickle resolution, used for plain Python builtins, numpy's own pickle support, etc., which don't need any special handling.

**`persistent_load(self, pid)`**: This is pickle's other standard extension point, invoked whenever the stream encounters a *persistent id* reference (the mechanism PyTorch uses specifically so tensor storages are **not** embedded inline in the pickle bytes). `pid` is a 5-tuple `("storage", storage_type_stub, key, location, numel)` — `storage_type_stub` is the `_StorageStub` instance `find_class` already produced for this tensor's dtype; `key` identifies which zip member holds this storage's raw bytes; `location` (unused here — would normally indicate CPU vs. a specific GPU device, irrelevant for a numpy-only reconstruction); `numel` is the element count. The function reads the raw bytes directly from the zip archive (`zf.read(f"{data_prefix}data/{key}")`) and reinterprets them via `np.frombuffer(raw, dtype=dtype, count=numel)` — a **zero-copy reinterpretation** of the raw byte buffer as a typed numpy array (not a parse/convert operation; it's a direct reinterpretation of the underlying bytes according to the dtype's byte layout).

### `load_pt(pt_path)`

**Purpose**: Top-level entry point — given a `.pt` file path, returns the fully reconstructed Python object graph (typically a `dict` or `OrderedDict` of tensor-name → numpy-array, possibly nested inside an outer checkpoint dict) with every tensor as a numpy array.

**Inner workings**: Opens the `.pt` file as a `zipfile.ZipFile`; finds the one member ending in `data.pkl` (its exact folder prefix varies — commonly the file's own stem, but this code derives the prefix robustly from wherever `data.pkl` actually is, rather than assuming a fixed naming convention); constructs a `_TorchUnpickler` wrapping that pickle stream (with the open zip handle and derived prefix passed through, so `persistent_load` can reach the raw tensor bytes); calls `.load()`, which drives the whole interception/reconstruction process described above and returns the final Python object.

### `load_state_dict_numpy(pt_path)`

**Purpose**: A convenience wrapper — unwraps the common checkpoint convention used throughout this project (`{"model_state_dict": {...}, "author_mapping": {...}, ...}`) down to just the tensor-name → array mapping, since most callers only want the weights, not the surrounding training metadata.

**Inner workings**: Calls `load_pt`; if the result is a dict containing the key `"model_state_dict"`, returns that nested dict (cast to a plain `dict`, not an `OrderedDict`); otherwise returns the loaded object as-is (handles checkpoints saved as a bare state dict with no wrapping).

### `export_to_npz(pt_path, out_npz_path)`

**Purpose**: A one-off conversion utility — dumps a checkpoint's tensors to the standard `.npz` numpy archive format (plus a `.json` sidecar for any non-tensor metadata), letting downstream code skip re-parsing the custom pickle format on every load.

**Inner workings**: Loads via `load_pt`; separates `tensors` (the state dict) from `meta` (anything else in the top-level checkpoint dict, e.g. `author_mapping`, `best_val_author_acc`); `np.savez` writes all tensors keyed by their original parameter names; a small recursive `_jsonable` helper converts any numpy types in the metadata (arrays → lists, numpy scalars → Python scalars) so the metadata can be serialized as plain JSON.

---

## 6. `Software/CNN/AuthorReproductionStuff/np_inference/layers.py`

### Role in the pipeline

The **from-scratch numpy forward-pass primitive library** — every mathematical operation `PaperCRNNNumpy` and `AuthorClassifierCNNNumpy` (Sections 7–8) need to replicate the trained PyTorch models' inference behaviour exactly, using only numpy. This is the mathematically densest file in the training/classification block and is where the report's "Design and implementation" + "Literature study" sections should draw their deepest equations from. No training happens here — this is pure forward-pass (inference) computation, using weights already trained by `TrainText.py`/`TrainAuthor.py` and loaded via `weight_io.py`.

### `conv2d(x, weight, bias, stride=1, padding=1)`

**Purpose**: 2-D convolution, reimplemented via the classic **im2col + matrix multiplication** trick rather than a naive nested-loop sliding-window implementation.

**Inputs**: `x`, shape `(B, C, H, W)` — batch, input channels, height, width; `weight`, shape `(OutC, InC, kh, kw)` — matches PyTorch's `nn.Conv2d.weight` layout exactly; `bias`, shape `(OutC,)` or `None`; `stride` (int, default 1); `padding` (int, default 1 — zero-padding applied symmetrically on both height and width).

**Outputs**: `(B, OutC, outH, outW)` where `outH = (H + 2·padding − kh)/stride + 1` (and analogously for `outW`) — the standard convolution output-size formula.

**Inner workings and maths — the im2col trick explained**: Convolution computes, for every output position `(i,j)` and output channel `o`:
`out[b,o,i,j] = Σ_c Σ_u Σ_v  weight[o,c,u,v] · x_padded[b,c, i·stride+u, j·stride+v]  + bias[o]`

Computed directly, this is a 6-nested-loop operation. **im2col** restructures it into a single matrix multiplication by first materializing, for every output spatial position, the flattened patch of input values that convolution would dot-product against the kernel at that position. Concretely: `xp` is the zero-padded input; `np.lib.stride_tricks.as_strided` is used to construct a 6-D *view* `cols` of shape `(B, C, kh, kw, outH, outW)` over `xp`'s existing memory, with strides chosen so that `cols[b,c,u,v,i,j] == xp[b, c, i·stride+u, j·stride+v]` — i.e. this view, with **zero additional memory copy**, re-exposes every convolutional patch's pixels as if they were separately laid out. Reshaping this to `(B, C·kh·kw, outH·outW)` flattens each patch into one column vector (length `C·kh·kw`) per output position. The kernel weight tensor is reshaped to `(OutC, C·kh·kw)`. The convolution then reduces to a single matrix multiplication per batch element: `out[b] = weight_flat @ cols[b]`, of shape `(OutC, outH·outW)`, reshaped back to `(OutC, outH, outW)`. The in-code comment notes this plain `@` matmul (rather than an equivalent `einsum` expression) specifically dispatches to an optimized BLAS routine and was measured roughly 10× faster for numerically identical output (verified bit-for-bit equal to floating-point rounding noise, ~1e-13) — a concrete, measured engineering justification worth citing in the report as evidence of informed implementation choices, not guesswork. Batch is handled via a Python list comprehension over `B` (not vectorized across batch in one matmul), justified by the fact that in practice `B=1` for every actual inference call in this project (one image processed per forward pass), so this loop never meaningfully costs anything.

**Literature connection**: The im2col+GEMM (General Matrix Multiply) technique for implementing convolution is a standard, widely-used technique in deep-learning systems (its use is documented in, e.g., Chellapilla, Puri & Simard (2006), *"High Performance Convolutional Neural Networks for Document Processing"*, and it underlies most production CNN framework backends including cuDNN's GEMM-based convolution algorithms) — not attributable to this project as a novel contribution, but a textbook-standard technique correctly applied here.

### `batchnorm2d(x, weight, bias, running_mean, running_var, eps=1e-5)`

**Purpose**: Applies **evaluation-mode** (inference-time) batch normalization — using the *stored, fixed* running statistics computed during training, not statistics recomputed live from the current input batch.

**Inputs**: `x`, `(B,C,H,W)`; `weight` (γ, the learned per-channel scale, shape `(C,)`); `bias` (β, the learned per-channel shift, shape `(C,)`); `running_mean`, `running_var` (the per-channel statistics accumulated via exponential moving average over many training batches, shape `(C,)` each); `eps` (small constant preventing division by zero).

**Outputs**: `(B,C,H,W)`, same shape as input.

**Inner workings and maths**: This is the textbook batchnorm **inference-mode** affine transform:

`y = γ · (x − running_mean) / sqrt(running_var + eps) + β`

implemented here with a small algebraic refactor for efficiency: `scale = γ / sqrt(running_var + eps)` and `shift = β − running_mean · scale` are precomputed once per channel, so the full transform becomes the single fused affine `y = x·scale + shift`, broadcast across the batch/height/width dimensions via `reshape(1,-1,1,1)`. This refactor is mathematically identical to the direct formula (expand `γ·(x−μ)/σ + β = x·(γ/σ) + (β − μ·γ/σ)`) but avoids repeating the division/subtraction for every element — a standard optimization, not a change in the computed result.

**Why eval-mode matters, for the report**: during *training*, batchnorm instead computes `mean`/`var` **live from the current mini-batch** (and simultaneously updates `running_mean`/`running_var` via an exponential moving average, for later use at inference time) — this is the behaviour `nn.BatchNorm2d` exhibits when the enclosing module is in `.train()` mode. At inference time (and in this from-scratch numpy reimplementation, which is inference-only by construction), using the batch's own statistics would make the output depend on whatever else happens to be in the same batch — undesirable for deterministic, single-image inference — so PyTorch's `.eval()` mode (and this function, unconditionally) uses the *fixed*, pre-computed running statistics instead. This distinction is directly relevant to the `TrainAuthor.py` frozen-backbone discussion elsewhere in this document: calling `model.eval()` (as the old, buggy `shape`-classifier training loop did) also freezes every BatchNorm layer's running statistics in place for the whole training run, compounding the "nothing in the backbone adapts" problem beyond just the frozen conv weights.

**Literature connection**: Ioffe & Szegedy (2015), *"Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"* — this function implements exactly their published inference-time formula (their Algorithm 2, "Batch-Normalized Network, inference").

### `relu(x)`

**Purpose/maths**: `max(0, x)`, elementwise — the standard Rectified Linear Unit nonlinearity, applied via `np.maximum(x, 0)`.

**Literature connection**: ReLU as a standard activation function is generally attributed to Nair & Hinton (2010) for its popularization in deep learning, though the function itself long predates that specific paper; "standard CNN/activation technique" is the safe attribution here.

### `maxpool2d(x, kernel=2, stride=2)`

**Purpose**: 2×2 (by default) max-pooling with stride 2 — spatial downsampling by taking the maximum value in each non-overlapping window.

**Inputs**: `x`, `(B,C,H,W)`.

**Outputs**: `(B,C,H//stride, W//stride)`.

**Inner workings**: Crops `x` down to the largest region evenly divisible by `stride` on both axes (`xc`); uses `as_strided` again — the same technique as `conv2d`'s im2col, but here constructing a 6-D view `(B,C,outH,outW,kernel,kernel)` where the last two axes directly index each pooling window's contents (strides `(s[2]*stride, s[3]*stride)` step between windows, `(s[2], s[3])` step within a window); then `.max(axis=(4,5))` reduces each window to its single maximum value. Zero-copy view construction followed by a single vectorized reduction — same strided-view efficiency principle as `conv2d`.

**Literature connection**: Max-pooling is a standard CNN downsampling operation, used in essentially all CNN architectures since LeNet/AlexNet-era work; not attributable to a single specific paper for this generic usage.

### `adaptive_max_pool_height(x)`

**Purpose**: Reimplements `nn.AdaptiveMaxPool2d((1, None))` — collapses the entire height dimension to exactly 1 via a max operation, regardless of the current height, leaving width untouched. This is the operation that converts the CNN's final 2-D feature map into a width-indexed sequence of feature vectors, feeding the BiLSTM.

**Inner workings**: `x.max(axis=2, keepdims=True)` — straightforward max-reduction over the height axis (axis index 2 in `(B,C,H,W)`), with `keepdims=True` so the output shape is `(B,C,1,W)` rather than `(B,C,W)` (matching what PyTorch's `AdaptiveMaxPool2d` would return before the model code's own subsequent `.squeeze(2)`).

### `adaptive_avg_pool_width_1d(x)`

**Purpose**: Reimplements `nn.AdaptiveAvgPool1d(1)` — collapses the width dimension of a `(B,C,W)` tensor to exactly 1 via **averaging**. Used only by `AuthorClassifierCNN`'s width-pooling step (the text model instead feeds every width position into the LSTM individually, never collapsing width).

**Inner workings**: `x.mean(axis=2, keepdims=True)` → `(B,C,1)`.

### `linear(x, weight, bias)`

**Purpose**: Fully-connected (dense) layer, reimplementing `nn.Linear`.

**Inputs**: `weight`, shape `(out_features, in_features)` — matching PyTorch's own `nn.Linear.weight` storage layout exactly (not the mathematically more conventional `(in,out)` layout — this detail matters for correctly reading the raw state-dict tensors).

**Maths**: `out = x @ weight.T + bias` — i.e. `out[...,o] = Σ_i x[...,i]·weight[o,i] + bias[o]`, the standard affine transform of a dense layer.

### `sigmoid(x)`

**Maths**: `σ(x) = 1/(1+e^{-x})`, the standard logistic sigmoid, used by the LSTM gate equations below.

### `lstm_forward(x, weight_ih, weight_hh, bias_ih, bias_hh, hidden_size)`

**Purpose**: The from-scratch forward pass of **one direction, one layer** of an LSTM — the core recurrent computation underlying `nn.LSTM`, which `PaperCRNN`'s `sequence` module is built from.

**Inputs**: `x`, shape `(T, B, input_size)` — sequence length, batch, per-timestep feature dimension; `weight_ih`, shape `(4·hidden_size, input_size)`; `weight_hh`, shape `(4·hidden_size, hidden_size)`; `bias_ih`, `bias_hh`, each shape `(4·hidden_size,)`; `hidden_size` (int).

**Outputs**: `(outputs, (h, c))` — `outputs` shape `(T, B, hidden_size)` (the hidden state at every timestep); `(h, c)` the final hidden and cell states, each `(B, hidden_size)`.

**Inner workings and maths — full LSTM gate equations**: Initializes `h`, `c` to zero vectors of shape `(B, hidden_size)`. At each timestep `t`:

`gates = x[t] @ weight_ih.T + bias_ih + h @ weight_hh.T + bias_hh`  — shape `(B, 4·hidden_size)`, combining both the input-to-hidden and hidden-to-hidden affine contributions in one addition.

This `(4·hidden_size)`-wide vector is then split into four equal `hidden_size`-wide blocks, in the exact order **PyTorch's own `nn.LSTM` convention uses** (explicitly documented in the code comment, and essential for correctness — getting this order wrong would silently produce a numerically-plausible but functionally wrong network): **input, forget, cell(candidate), output**:

- `i_t = σ(gates[:, 0:H])` — the **input gate**: how much of the new candidate cell content to let into the cell state.
- `f_t = σ(gates[:, H:2H])` — the **forget gate**: how much of the previous cell state to retain.
- `g_t = tanh(gates[:, 2H:3H])` — the **candidate cell content** (sometimes written `\tilde{c}_t` in the literature; this code calls it `g`, matching PyTorch's own internal naming).
- `o_t = σ(gates[:, 3H:4H])` — the **output gate**: how much of the (squashed) cell state to expose as the hidden state.

Then the state updates:

`c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t`  (⊙ = elementwise product) — the cell state is a gated blend of its previous value and the new candidate content.

`h_t = o_t ⊙ tanh(c_t)` — the hidden state is a gated, squashed view of the cell state.

Both `h` and `c` are carried forward into the next iteration; `outputs[t] = h` is recorded at every step. The loop is a genuine Python `for t in range(T)` loop (not vectorized across time — LSTMs are inherently sequential since each timestep depends on the previous one's output).

**Literature connection**: This is a direct, by-the-equations implementation of **Hochreiter & Schmidhuber (1997)**, *"Long Short-Term Memory"* — specifically the now-standard formulation with input/forget/output gates and a separate cell state (the original 1997 paper's exact gating details were later refined slightly by subsequent work and by convention adopted in frameworks like PyTorch, but this is recognizably the canonical LSTM cell). The specific gate-ordering convention (`[input, forget, cell, output]` concatenated into one `4H`-wide weight matrix) is PyTorch's own documented internal convention, which this code must match exactly to correctly consume weights trained by PyTorch's `nn.LSTM`.

### `bidirectional_lstm_forward(x, params_fwd, params_bwd, hidden_size)`

**Purpose**: Combines two independent `lstm_forward` calls — one over the sequence in forward order, one over it reversed — to produce a bidirectional representation at every timestep.

**Inputs**: `x`, `(T,B,input_size)`; `params_fwd`/`params_bwd`, each a dict of the four weight/bias arrays `lstm_forward` needs (independent, separately-trained parameter sets — not the same weights reused); `hidden_size`.

**Outputs**: `(T, B, 2·hidden_size)` — forward and backward hidden sequences concatenated along the feature axis at every timestep.

**Inner workings**: `out_f, _ = lstm_forward(x, **params_fwd, ...)` runs the forward-direction LSTM over the sequence as given. `out_b, _ = lstm_forward(x[::-1], **params_bwd, ...)` runs a second, independently-weighted LSTM over the **time-reversed** input; its output is then reversed back (`out_b[::-1]`) so that `out_b[t]` ends up aligned with the same original timestep `t` as `out_f[t]`, despite having been computed by processing the sequence backward. The two aligned sequences are concatenated along the last axis: `np.concatenate([out_f, out_b], axis=2)`.

**Literature connection**: Schuster & Paliwal (1997), *"Bidirectional Recurrent Neural Networks"* — the foundational formulation of running two independent recurrent networks over a sequence in opposite directions and combining their outputs per-timestep, which this function implements directly (generalized here from their original RNN cell to the LSTM cell, as `PaperCRNN`'s `nn.LSTM(bidirectional=True)` does).

### `multilayer_bidirectional_lstm(x, layer_params, hidden_size)`

**Purpose**: Stacks multiple `bidirectional_lstm_forward` layers, matching `nn.LSTM(num_layers=2, bidirectional=True)`'s behaviour of feeding one layer's (bidirectional, concatenated) output as the next layer's input.

**Inputs**: `layer_params`, a list of `{"fwd": {...}, "bwd": {...}}` dicts, one per layer (for `PaperCRNN`, 2 entries, matching `lstm_layers=2`).

**Inner workings**: A simple sequential loop — `out = x`; for each layer's params, `out = bidirectional_lstm_forward(out, layer["fwd"], layer["bwd"], hidden_size)`. Note each layer after the first receives a `2·hidden_size`-wide input (the previous layer's concatenated bidirectional output) — this is why `text_model.py`'s `PaperCRNNNumpy.__init__` must fetch distinct weight tensors per layer index (`sequence.weight_ih_l0`, `sequence.weight_ih_l1`, etc.) from the checkpoint, each with the correct `input_size` for its position in the stack.

### `log_softmax(x, axis)`

**Purpose**: Numerically-stable log-softmax, matching `torch.nn.functional.log_softmax` — needed because `PaperCRNNNumpy`'s output must match `PaperCRNN.forward`'s own `log_softmax(logits, dim=2)` call exactly, since that's what `nn.CTCLoss` was trained against and what `decode_ctc` expects.

**Maths**: For input vector `x` along the given axis, the mathematically exact log-softmax is `log(softmax(x)_k) = x_k − log(Σ_j e^{x_j})`. Computed naively, `Σ_j e^{x_j}` can overflow for large `x_j`. The standard numerically-stable reformulation, implemented here: subtract the per-axis maximum first (`shifted = x − max(x)`, so the largest shifted value is exactly 0, making `e^{shifted}` safely bounded in `(0,1]`), then `lse = log(Σ_j e^{shifted_j})` (the "log-sum-exp" trick), and return `shifted − lse`. This is algebraically identical to the naive formula (the subtracted max cancels out exactly: `x_k − max − log(Σe^{x_j−max}) = x_k − max − (log Σe^{x_j} − max) = x_k − log Σe^{x_j}`) but avoids the overflow.

**Literature connection**: The log-sum-exp numerical stabilization trick is a standard numerical-computing technique, not attributable to a specific paper.

### `softmax(x, axis)`

**Maths**: Same max-subtraction stabilization, then `e^{shifted} / Σ_j e^{shifted_j}` — the standard softmax function, used (not log-softmax) wherever actual probability values (not log-probabilities) are needed, e.g. `AuthorClassifierCNNNumpy`'s classification output in `author_model.py`.

### `if __name__ == "__main__":` self-test block

A lightweight correctness/shape smoke-test: constructs small random tensors, runs each primitive above, and asserts the output shapes match expectations (e.g. `conv2d` output shape, `maxpool2d` halving spatial dims, `lstm_forward`/`bidirectional_lstm_forward`/`multilayer_bidirectional_lstm` producing the expected sequence shapes, `softmax` summing to 1 within tolerance). Not a numerical-correctness-against-PyTorch test (it doesn't compare against real torch output), just a shape/sanity check that would catch an obviously broken implementation.

---

## 7. `Software/CNN/AuthorReproductionStuff/np_inference/text_model.py`

### Role in the pipeline

Implements `PaperCRNNNumpy`, the numpy-only forward pass exactly mirroring `PaperCRNN.forward` (Section 1), using the checkpoint-loading machinery from `weight_io.py` and the primitives from `layers.py`. Also provides `ReadText`, the greedy-CTC-decode text-recognition entry point used wherever torch-free inference is needed, matching the behaviour of the torch-based `VerifyRewrite.ReadText` function (defined elsewhere, not in scope here) exactly. Functional block: **Training & Classification** (inference side).

### `resize_line_image_fixed` / `tensor_from_resized`

Byte-for-byte ports of the identically-named functions in `TrainText.py` (Section 1), re-documented there in full — repeated here only because this file is import-independent of `TrainText.py` (deliberately avoids any torch-importing module). `tensor_from_resized` differs only in its final line: it returns a plain numpy array shaped `(1,1,H,W)` (`arr[np.newaxis, np.newaxis, :, :]`) instead of a `torch.Tensor`.

### `decode_ctc(log_probs)`

Numpy port of `TrainText.py`'s `decode_ctc` — identical greedy-best-path-then-collapse-repeats-then-drop-blank algorithm (see Section 1's full explanation), operating on a numpy `(T,B,C)` array via `np.argmax` instead of `torch.argmax`.

### `_conv_block_forward(x, sd, prefix, n_layers)`

**Purpose**: Runs one `conv_block`'s worth of `(Conv→BN→ReLU) × n_layers` using raw weight-dict lookups, matching the exact submodule-index naming `nn.Sequential` assigns automatically (index `3i` = the `i`-th `Conv2d`, `3i+1` = the `i`-th `BatchNorm2d`; `3i+2`, the `ReLU`, has no learnable parameters and needs no lookup).

**Inputs**: `x` (numpy array); `sd` (the loaded state dict, numpy arrays); `prefix` (e.g. `"stage1"`); `n_layers` (2, 4, or 6, matching the three stages).

**Inner workings**: For `i` in `range(n_layers)`: looks up `sd[f"{prefix}.{3i}.weight"]`/`.bias` for the conv (calls `L.conv2d(..., padding=1)`), `sd[f"{prefix}.{3i+1}.weight"]`/`.bias`/`.running_mean`/`.running_var` for the batchnorm (calls `L.batchnorm2d`), then `L.relu(x)`. This exact indexing scheme is why `conv_block`'s structure in `TrainText.py` (`Conv, BN, ReLU` repeated, built via plain list concatenation into `nn.Sequential`) must never change without updating this function correspondingly — the two files are coupled through this positional-index convention, not through any explicit shared schema.

### `class PaperCRNNNumpy`

**`__init__(self, checkpoint_path=None, hidden_size=256)`**

**Purpose**: Loads a trained `PaperCRNN` checkpoint via `weight_io.load_state_dict_numpy` and pre-organizes the LSTM's per-layer, per-direction weight tensors into the nested structure `layers.multilayer_bidirectional_lstm` expects.

**Inner workings**: `self.checkpoint_path` defaults to `DEFAULT_TEXT_WEIGHTS`, itself resolved via the same checkpoint-preference fallback chain used elsewhere in the project (prefer `paper_cnn_bilstm_ctc_joint_best.pt`, then `_hf_best.pt`, else the plain `_best.pt` — mirroring `VerifyRewrite.py`'s own fallback chain, per the in-code comment, so numpy and torch inference always agree on which checkpoint is "the" current best). For each of the 2 LSTM layers, builds a `fwd` dict from keys `sequence.weight_ih_l{layer}` / `weight_hh_l{layer}` / `bias_ih_l{layer}` / `bias_hh_l{layer}` (PyTorch's naming convention for a forward-direction LSTM layer's parameters) and a `bwd` dict from the `_reverse`-suffixed counterparts (PyTorch's naming convention for the corresponding reverse-direction parameters within a `bidirectional=True` `nn.LSTM`) — this naming convention is exactly what lets this code correctly map the flat state-dict keys loaded by `weight_io.py` onto the nested `layer_params` structure `multilayer_bidirectional_lstm` consumes.

**`forward(self, x)`**

**Purpose**: The full numpy forward pass, computing the identical sequence of operations `PaperCRNN.forward` performs in PyTorch.

**Inputs**: `x`, `(B,1,H,W)` numpy array.

**Outputs**: `(T,B,num_classes)` numpy array of log-probabilities.

**Inner workings**: `_conv_block_forward(..., "stage1", 2)` → `L.maxpool2d(x,2,2)` → `_conv_block_forward(..., "stage2", 4)` → `L.maxpool2d(x,2,2)` → `_conv_block_forward(..., "stage3", 6)` → `L.adaptive_max_pool_height(x)` (→ `(B,128,1,W)`) → `x[:,:,0,:]` (→ `(B,128,W)`, numpy's direct-indexing equivalent of PyTorch's `.squeeze(2)`) → `np.transpose(x, (2,0,1))` (→ `(W,B,128)`, equivalent to `.permute(2,0,1)`) → `L.multilayer_bidirectional_lstm(...)` (→ `(W,B,512)`) → `L.linear(seq, sd["text_head.weight"], sd["text_head.bias"])` (→ `(W,B,num_classes)`) → `L.log_softmax(logits, axis=2)`. This is a line-for-line numpy mirror of `PaperCRNN.forward`'s PyTorch implementation (Section 1), using the primitives documented in Section 6.

### `ReadText(pil_image, model)`

**Purpose**: The single-call public entry point — takes a raw PIL line-crop image and a `PaperCRNNNumpy` instance, returns the decoded text string.

**Inner workings**: `resize_line_image_fixed` → `tensor_from_resized` → `model.forward` → `decode_ctc(...)[0]` (index `[0]` since this is always a batch-of-1 call). Explicitly documented as matching `VerifyRewrite.ReadText`'s (torch-based) behaviour, so swapping between the torch and numpy inference paths should be transparent to any caller.

### `levenshtein(a, b)`

Byte-for-byte port of `TrainText.py`'s function (Section 1) — identical Wagner–Fischer edit-distance algorithm.

### `CharAcc(pred, truth)`

**Purpose**: Character-accuracy metric (1 − CER, floored at 0), ported verbatim from `VerifyRewrite.py`.

**Inner workings**: Edge case `truth == ""`: returns 1.0 if `pred` is also empty, else 0.0 (an empty ground truth with any non-empty prediction is maximally wrong, not undefined). Otherwise `max(0, 1 − levenshtein(pred,truth)/len(truth))` — the floor at 0 handles the case where edit distance exceeds the ground-truth length (which can happen if the prediction inserts many extra characters), preventing a nonsensical negative "accuracy."

### `WordAcc(pred, truth)`

**Purpose**: Word-level accuracy via the fraction of ground-truth words recovered **in the correct relative order**, ported verbatim from `VerifyRewrite.py`.

**Inputs**: `pred`, `truth` — strings, split on whitespace into word lists `pw`, `tw`.

**Inner workings and maths**: Computes the **Longest Common Subsequence (LCS)** length between the two word sequences (case-insensitively, via `.lower()` comparison) using the standard O(m·n) dynamic-programming table: `dp[i+1][j+1] = dp[i][j]+1` if `tw[i].lower()==pw[j].lower()`, else `max(dp[i][j+1], dp[i+1][j])`. Returns `dp[m][n] / m` — the LCS length as a fraction of the *ground-truth* word count. Using LCS (rather than, say, exact set overlap) specifically credits words recovered in the right relative order while still tolerating insertions/deletions elsewhere in the sequence, a more linguistically meaningful metric than raw edit distance for judging readability of recognized text.

**Literature connection**: Longest Common Subsequence is a classical dynamic-programming algorithm (standard computer-science technique, not a handwriting-recognition-specific contribution); using word-level LCS as an accuracy proxy is a common, generic text-similarity evaluation technique.

---

## 8. `Software/CNN/AuthorReproductionStuff/np_inference/author_model.py`

### Role in the pipeline

Implements `AuthorClassifierCNNNumpy`, the numpy-only forward pass mirroring `AuthorClassifierCNN.forward` (Section 2), plus `ClassifyImage` (the single-call classification entry point, matching `EvaluateStyle.ClassifyImage`'s torch-based behaviour) and a numpy/non-torch port of `StrokeNormalize` (duplicating the logic documented in Section 2, here built on `ProfileIO`'s pure-numpy primitives instead of `BuildStyleProfile`/`SegmentPage`'s torch-adjacent module names, though the underlying algorithm is identical). Functional block: **Training & Classification** (inference side).

### `StrokeNormalize(gray, penWidthXh=PEN_WIDTH_XH)`

Functionally identical to `TrainAuthor.py`'s `StrokeNormalize` (fully documented in Section 2: binarize → measure x-height via `CoreBand` → skeletonize → re-dilate to a constant pen width → invert), explicitly noted in this file's own docstring as a direct port, built on `ProfileIO`'s numpy-only equivalents of `BinarizeLine`/`CoreBand`/`Skeletonize`/`Dilate` so this entire inference path never needs to import any torch-adjacent module. The constant `PEN_WIDTH_XH = 0.14` is kept in sync by convention (matching `TrainAuthor.py`'s module-level constant of the same name and value) rather than imported from a shared location — another small duplication point, consistent with the project's general pattern of accepting controlled duplication at inference-path boundaries to keep the numpy path import-independent of any torch-touching module.

### `_conv_block_forward(x, sd, prefix, n_layers)`

Identical in implementation to `text_model.py`'s function of the same name (Section 7) — duplicated rather than factored into a shared helper module, presumably to keep `text_model.py` and `author_model.py` each independently self-contained/importable without requiring the other.

### `class AuthorClassifierCNNNumpy`

**`__init__(self, checkpoint_path=None)`**

**Purpose**: Loads a trained `AuthorClassifierCNN` checkpoint, keeping both the state dict and (if present) the `author_mapping` dict (author-name → class-index) the training script saved alongside the weights.

**Inner workings**: Defaults `checkpoint_path` to `DEFAULT_AUTHOR_WEIGHTS`, resolved via a fallback chain mirroring `EvaluateStyle.py`'s own: prefer `author_shape_10new_weights.pt` (the shape classifier) if it exists, else `author_classifier_10_weights.pt`, else `author_fast_10_weights.pt`. Uses `W.load_pt` directly (not the `load_state_dict_numpy` convenience wrapper `text_model.py` uses) specifically so it can **also** retain the checkpoint's `author_mapping` dict, not just the tensor weights — `self.sd` is the unwrapped state dict, `self.author_mapping` is pulled from the raw loaded dict if present.

**`forward(self, x)`**

**Purpose**: Full numpy forward pass mirroring `AuthorClassifierCNN.forward`.

**Inputs**: `x`, `(B,1,H,W)`.

**Outputs**: `(B, num_authors)` raw (pre-softmax) logits.

**Inner workings**: Same three-stage conv backbone as the text model (`_conv_block_forward` + `maxpool2d` twice, matching `AuthorClassifierCNN`'s identical backbone shape) → `L.adaptive_max_pool_height` (→ `(B,128,1,W)`) → `x[:,:,0,:]` (→ `(B,128,W)`) → `L.adaptive_avg_pool_width_1d(x)[:,:,0]` (→ `(B,128)`, the author model's distinguishing **average**-over-width pooling, vs. the text model's per-position sequence output) → `L.linear(pooled, sd["author_head.0.weight"], sd["author_head.0.bias"])` → `L.relu` → (the `Dropout(0.2)` at `author_head.2` is explicitly skipped — correctly, since dropout is a training-only regularization technique with no effect, and no learnable parameters, at inference time) → `L.linear(h, sd["author_head.3.weight"], sd["author_head.3.bias"])` — the final author-logit layer.

### `ClassifyImage(pil_image, model, author_mapping=None, apply_stroke_normalize=True)`

**Purpose**: The single-call public classification entry point — takes a raw PIL line-crop image, returns the predicted author class index and the full probability vector, matching `EvaluateStyle.ClassifyImage`'s exact return signature so torch and numpy inference paths are interchangeable to any caller.

**Inputs**: `pil_image` (PIL image, raw line crop); `model` (an `AuthorClassifierCNNNumpy` instance); `author_mapping` (optional, accepted only so callers not wanting the module-level default can supply their own — not actually used inside this function, since resolving a predicted index back to an author-ID string is left to the caller, exactly as `EvaluateStyle.py` does outside its own `ClassifyImage`); `apply_stroke_normalize` (bool, default `True` — controls whether this call targets the shape classifier's stroke-normalized domain or a raw-ink classifier's native domain).

**Outputs**: `(predIdx: int, probs: np.ndarray)` — `probs` shape `(num_authors,)`, a genuine probability distribution (softmax-normalized, sums to 1), unlike `AuthorClassifierCNNNumpy.forward`'s raw logits.

**Inner workings**: If `apply_stroke_normalize`, runs `StrokeNormalize` on the grayscale-converted input first (falling back to the original image if normalization fails, e.g. no extractable ink/skeleton); resizes/tensorizes via the text-model module's shared `resize_line_image_fixed`/`tensor_from_resized` functions (imported directly from `text_model.py` — reused rather than duplicated a third time, unlike `StrokeNormalize` and `_conv_block_forward` above); runs `model.forward(t)` to get raw logits; applies `L.softmax(logits, axis=1)[0]` to convert to a probability vector for the single (batch-of-1) input; `predIdx = int(np.argmax(probs))`.

---

## Known limitations / honest caveats

**The frozen-backbone bug in `TrainAuthor.py`'s shape classifier — the project's clearest worked example of why task objectives shape which features a network learns.**

As documented at length in Section 2 above (`TrainAuthor.Train()`), the shape (stroke-normalized) writer-ID classifier originally trained only its small classification head (`author_head`) on top of a **frozen** backbone warm-started from the text-recognition model's trained weights. In precise terms: every convolutional backbone parameter had `requires_grad=False` set (via the `freeze_backbone()` method, defined but no longer called from this training path), so gradient descent never moved those weights away from whatever `PaperCRNN`'s CTC-based character-recognition training had converged them to; only `author_head`'s own small set of weights was updated, across 500 epochs at lr 3e-3.

**Why this fails, conceptually**: a well-trained OCR backbone's whole purpose is to extract features that let a decoder predict *which character is being written*, while being as **invariant as possible** to *who* wrote it and *how stylistically* — slant, proportion, letterform idiosyncrasy, and connection habits are, from a character-recognition loss's point of view, nuisance variation the network is explicitly rewarded for discarding. A writer-identification classifier needs **exactly the opposite**: it needs those stylistic degrees of freedom preserved and amplified, not suppressed. Freezing the OCR backbone and only training a shallow head on top of its output therefore asks the head to recover writer-identity signal from a representation that was never asked to retain it — it can only succeed insofar as some of that signal survives as an unintentional byproduct of the OCR objective, not because the backbone was ever optimized to preserve it.

**Measured symptom** (as recorded in the code's own comments, not invented for this document): pooled validation accuracy on the frozen-backbone shape classifier looked acceptable at 95.3%, but the real-photo confusion matrix told a different story — author `owen`'s lines were classified as `"yeukita"` 80% of the time (118 of 148 lines), and author `abhinav`'s lines were split between two other authors' classes (`"552"`/`"thiya"`). A raw-pixel-space distance check ruled out "the two writers' styles just look alike" as an innocent explanation: `owen`'s own lines sit closer to each other (mean L2 distance 11118) than to `yeukita`'s lines (mean L2 distance 15599) in the model's raw 64×640 input space — i.e. the confusion was a property of what the frozen-backbone, shallow-head model was *capable* of learning from stroke-normalized input, not an inherent ambiguity between the two writers' actual handwriting. As corroborating evidence, the sibling **ink** classifier — which fine-tunes its *whole* network on the same underlying line crops, just without stroke normalization — correctly classifies `owen`'s lines 147 times out of 148.

**The fix**: unfreeze the entire backbone (never call `freeze_backbone()`), keep the OCR-pretrained weights only as a sensible warm-start initialization (still saving real training time versus training from random weights on a fairly small 10-author dataset — the same transfer-learning logic cited to Nguyen et al., arXiv:1712.07923, elsewhere in the project), and let ordinary end-to-end backpropagation across the whole network, for 20 epochs, adapt the convolutional filters specifically toward whatever features actually discriminate ten writers' stroke geometry — something the original OCR training objective never asked them to preserve. This file's own training function and surrounding code, as read during the preparation of this document, confirm the current implementation performs this **full, unfrozen fine-tune**: `AdamW` is constructed over `model.parameters()` with no filtering, and `freeze_backbone()` is never invoked anywhere in the `shape` subcommand's call path. The code's own comments report the frozen-backbone run's pooled validation accuracy (95.3%) and the specific confusion-matrix failure counts quoted above as the measured before-state; no equivalent single aggregate "after" accuracy number for the fixed, fully-fine-tuned shape classifier is stated in the surrounding comments beyond the per-run `print` statements the training loop itself emits at runtime (`"Best val shape author-acc"` and per-author breakdown) — those actual numbers should be read from the training run's own console/log output when writing up final results, rather than guessed at here.

**Other documented, in-scope caveats worth carrying into the report:**

- **Greedy CTC decoding, not the source paper's best-performing decoder.** `TrainText.py`'s `decode_ctc` (and its numpy mirror in `text_model.py`) implements only best-path (greedy) CTC decoding. The Kizilirmak & Yanikoglu paper this architecture reproduces reports its *best* numbers using lexicon-constrained word-beam-search decoding, which requires external infrastructure (a beam-search decoder library plus corpora: Brown + WikiText2 + IAM train vocabulary) this project does not have. This is an honest, acknowledged gap, not an oversight — the module docstring names a concrete follow-up (e.g. `pyctcdecode`) as a reasonable future addition for anyone chasing the paper's exact CER/WER numbers.
- **Resolution deliberately halved from the paper's spec.** Input canvas is 64×640, not the paper's literal 100×960, specifically because 100×960 was measured at roughly 1 hour per epoch on CPU hardware — "halved to 64×640 (same ~1:10 aspect ratio, ~2.3× fewer pixels)" is an explicit, reasoned engineering tradeoff against real hardware constraints, not a silent deviation.
- **The `LSTM scheduler.step()` omission, since fixed.** `train()`'s in-code comment documents a real historical bug: `ReduceLROnPlateau` was constructed but its `.step(val_loss)` call was originally missing entirely, meaning the learning rate silently stayed fixed at its initial value for the whole run regardless of how long validation loss plateaued. This is now fixed (the call is present and conditioned correctly on `eval_every`).
- **Full-fine-tune vs. frozen-conv tradeoff, the mirror-image case in `TrainText.py`'s `personal` subcommand.** Separately from the `TrainAuthor.py` case above (where freezing was *wrong*), `train_personal`'s `freeze_conv` option documents the *opposite* lesson for a different task: naive full fine-tuning of the text-recognition CNN on a small personal-handwriting-only dataset was measured to cost ~10–11 percentage points of character accuracy on clean, general (Teklia) validation/test data after only 16 epochs — a textbook case of catastrophic forgetting when fine-tuning a general-purpose model on a narrow dataset. Freezing the backbone there is the *correct* choice, for the converse reason: a general-purpose OCR feature extractor doesn't need to change at all to recognize one new person's handwriting, it only needs its decision layers (LSTM + output head) to adapt. Reading these two cases together (`TrainAuthor.py`'s shape classifier needing *unfrozen* training; `TrainText.py`'s personal fine-tuning needing *frozen* conv layers) is a genuinely instructive contrast for the report's literature-study/discussion section: whether freezing a pretrained backbone helps or hurts depends entirely on whether the new task needs the *same* invariances the backbone was already trained to have, or needs signal the original training objective was actively trained to discard.
