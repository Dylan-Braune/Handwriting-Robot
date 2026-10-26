# Block B — The no-framework neural core: training and inference

Every layer, every gradient and both optimisers are written out in NumPy. There is no machine-learning framework anywhere in this block.

| File | Lines | Role |
|---|---|---|
| `Charset.py` | 7 | The 73-character label alphabet. One source of truth, imported by everything. |
| `AuthorsConfig.py` | 33 | The active writer set and the train/holdout split convention. |
| `NumpyNN.py` | 514 | **Forward *and* backward** for every layer, plus CTC loss, cross-entropy, RMSprop, AdamW, two schedulers, gradient clipping. |
| `GradCheckNumpyNN.py` | 509 | Numerical verification of all of the above against central differences. 78 configurations. |
| `TrainScratch.py` | 721 | The training program for both models. One file, shared backbone. |
| `TrainText.py` | 495 | Data pipeline for the recogniser: caching, augmentation, dataset assembly. |
| `TrainAuthor.py` | 112 | Data pipeline for writer ID, built on `TrainText`'s. |
| `Classify.py` | 340 | Inference-only NumPy forward passes for both models. What the live server runs. |
| `ConvertPtToNpz.py` | 121 | One-time: reads the old PyTorch `.pt` checkpoints without PyTorch, writes `.npz`. |
| `ExportTekliaLines.py` | 82 | One-time: fetches the pre-segmented Teklia/IAM-line set to plain PNG + label files. |

## B.0 Where the architecture comes from

The recogniser reproduces the architecture of **Kizilirmak & Yanikoglu, "CNN-BiLSTM model for English Handwriting Recognition", 2023 (arXiv:2307.00664)**, as recorded at `TrainText.py:1`. What the paper supplies is a described approach: a layer pattern, filter counts, a recurrent width, an augmentation table. It does not publish code. What this project supplies is the implementation — every layer's forward and backward pass, the input geometry, the data pipeline, the label cleaning, the staged training recipe and the verification harness.

The gradients are the standard published results for each layer type. The BatchNorm backward is the cs231n derivation, annotated as such in the source; the CTC gradient is Graves et al. (2006) equations 15–16; the LSTM recurrence is Hochreiter & Schmidhuber (1997). What this code does is write all of them out and evaluate them itself rather than calling a framework, and `GradCheckNumpyNN.py` confirms each one numerically against finite differences.

---

## B.1 `Charset.py` and `AuthorsConfig.py`

```python
CHARSET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:'\"!?()-"
CHAR_TO_IDX = {char: idx+1 for idx,char in enumerate(CHARSET)}
IDX_TO_CHAR = {idx: char for char,idx in CHAR_TO_IDX.items()}
```

73 characters: 52 letters, 10 digits, space and 10 punctuation marks. **`+1` on every index, because index 0 is reserved for the CTC blank.** The number of output classes is therefore `len(CHARSET)+1 = 74`, which is what `TrainScratch.TrainTextModel` passes as `numClasses`; every class count in the project is that expression rather than a literal.

Characters outside the set are dropped rather than mapped to an unknown token — `EncodeText` filters with `if c in CHAR_TO_IDX`, and `ExportTekliaLines` applies the same filter at export time, discarding the line entirely if nothing survives.

`AuthorsConfig.py` holds the writer set:

```python
DATASET_AUTHORS  = ["150","151","152","153","384","551","552","588"]   # 8 IAM writers
PERSONAL_AUTHORS = ["abhinav","dylan","owen","thiya","yeukita"]        # 5 personal writers
VAL_FRACTION = 0.15
SPLIT_SEED   = 0
```

Thirteen writers in total, and the number of classes is always computed as `len(DATASET_AUTHORS + PERSONAL_AUTHORS)` so it follows these lists automatically.

`SPLIT_SEED = 0` is the reason the file exists: a fixed seed means every script holds out *the same* lines, so a number measured by `Evaluate.py` is comparable with one measured during training. `VAL_FRACTION = 0.15` applies to the personal authors; the IAM path instead holds out one whole *page* per author (B.5.4), which is the stricter split.

---

## B.2 The architecture both models share

```
input (B,1,64,640)
  stage1:  2 × [conv3×3 pad1 → BatchNorm → ReLU]   1 → 32 channels
  maxpool 2×2                                        → (B,32,32,320)
  stage2:  4 × [conv3×3 pad1 → BatchNorm → ReLU]  32 → 64 channels
  maxpool 2×2                                        → (B,64,16,160)
  stage3:  6 × [conv3×3 pad1 → BatchNorm → ReLU]  64 → 128 channels
  adaptive max over height                           → (B,128,1,160)
```

Twelve convolutional layers, 2+4+6 per stage, 32→64→128 filters, with a 2×2 max-pool after stages 1 and 2 **only** — not after stage 3. That is the paper's pattern. Then the two models diverge:

- **Text:** squeeze the height axis, transpose to `(W,B,128)` so **width becomes the CTC time axis**, then a 2-layer bidirectional LSTM with hidden size 256 (no dropout, per the paper), then a linear head to 74 classes.
- **Writer ID:** additionally average-pool over *width*, giving one 128-vector per image, then `Linear(128→96) → ReLU → Dropout(0.2) → Linear(96→13)`.

The two pooling choices encode the difference between the tasks. Text recognition must preserve horizontal position — *where* each letter is — so width is kept and becomes time. Writer identification must be *invariant* to horizontal position, because a writer's identity does not depend on what they wrote or where on the line it falls, so width collapses to a mean. Max over height, mean over width: max because a stroke feature is either present or absent at some height, mean because a style statistic should be averaged over the whole line.

**`T = W/4 = 160`.** Two 2×2 pools quarter the width and there is no third pool, so 640/4 = 160 timesteps. This fixes the label-length cap:

```python
MAX_SAFE_LABEL_CHARS = int((INPUT_WIDTH//4)*0.75)   # = 120
```

CTC needs at least `2L+1` timesteps to emit `L` labels with separating blanks, making `L ≤ (T-1)/2 = 79` the hard feasibility limit. The looser 0.75·T = 120 cap is a data-quality filter: a "label" of 120+ characters is almost certainly a *collapsed* segmentation where several lines merged into one crop, not a real line.

**Input geometry is 64×640, halved from the paper's 100×960.** `TrainText.py:43` records the reason — CPU training at full size ran about an hour per epoch. The 2.3× reduction in pixels bought a tractable training time on the available hardware.

---

## B.3 `NumpyNN.py` — forward and backward for everything

Every forward returns `(out, cache)`; every backward takes `(dout, cache)` and returns the gradients the caller needs. Parameters live in plain dicts keyed *exactly* like a PyTorch `state_dict` (`"stage1.0.weight"`, `"sequence.weight_ih_l0_reverse"`, `"text_head.bias"`). That key compatibility has three consequences: a checkpoint trained by `TrainScratch.py` loads into `Classify.py` unmodified; an old converted PyTorch checkpoint and a new from-scratch one are interchangeable; and `LoadParamsInto` can warm-start one from the other by matching names and shapes.

### B.3.1 Convolution — im2col with stride tricks

```python
s = xp.strides
shape   = (B,C,kh,kw,outH,outW)
strides = (s[0],s[1],s[2],s[3],s[2],s[3])
cols = np.lib.stride_tricks.as_strided(xp,shape=shape,strides=strides)
out = (w@colsFlat).reshape(B,OutC,outH,outW)
```

Convolution is recast as one matrix multiply — the im2col/GEMM formulation (Chellapilla et al. 2006). `as_strided` builds a **view**, not a copy: the same padded buffer is reinterpreted as a 6-D array where `cols[b,c,i,j,y,x] = xp[b,c,y+i,x+j]`. The strides make that work, because advancing `i` and advancing `y` both step by one image row (`s[2]`), which is exactly the overlapping-window structure. Nothing is allocated for the windows.

Then `w @ colsFlat`, with `w` of shape `(OutC, C·kh·kw)` and `colsFlat` of shape `(B, C·kh·kw, outH·outW)`. NumPy broadcasts the 2-D operand across the batch dimension and dispatches a single batched BLAS GEMM rather than a per-sample Python loop.

The backward pass:

$$\frac{\partial L}{\partial W} = \sum_b \text{dout}_b\,\text{cols}_b^\top, \qquad \frac{\partial L}{\partial b} = \sum_{b,y,x} \text{dout}, \qquad \frac{\partial L}{\partial \text{cols}} = W^\top\,\text{dout}$$

then the **col2im scatter-add**, the one part that cannot be a reshape:

```python
for i in range(kh):
    for j in range(kw):
        dxp[:,:,i:i+outH,j:j+outW] += dcols[:,:,i,j,:,:]
```

Kernel windows overlap when `kh,kw > 1`, so each input pixel contributes to up to `kh·kw` output positions and its gradient is the *sum* over them. The loop runs over the 9 kernel offsets, not over pixels, so it stays vectorised. Writing it as an `as_strided` view with `+=` would silently drop contributions, because NumPy's buffered in-place add on an aliased view applies each overlapping write only once.

### B.3.2 BatchNorm — train mode

`BatchNorm2dForwardTrain` normalises by **batch** statistics over axes `(0,2,3)`, i.e. per channel across batch and both spatial axes, so `N = B·H·W` elements per channel.

```python
varUnbiased = varBiased*N/max(N-1,1)
runningMean[:] = (1-momentum)*runningMean + momentum*mu
runningVar[:]  = (1-momentum)*runningVar  + momentum*varUnbiased
```

The forward *normalises* with the biased variance but *accumulates* the unbiased estimate into the running statistic, matching PyTorch — which is what keeps inference-mode output consistent with training-mode output. The slice assignment `runningMean[:]` mutates in place, which is what makes these **buffers** rather than parameters: `TrainableParams` filters them out by key suffix so no optimiser touches them.

The backward is the standard closed form:

$$\frac{\partial L}{\partial x} = \frac{1}{N\sigma}\left(N\,\frac{\partial L}{\partial \hat x} - \sum \frac{\partial L}{\partial \hat x} - \hat x \sum \hat x \frac{\partial L}{\partial \hat x}\right)$$

The three terms are the direct path, the mean's dependence on `x`, and the variance's dependence on `x`. `eps = 1e-5` inside the square root, PyTorch's default, guards a constant channel.

### B.3.3 Pooling

`MaxPool2dForward` builds non-overlapping windows with `as_strided` and records the `argmax` within each. The backward scatters `dout` into the recorded positions with `np.put_along_axis` and then places blocks directly:

```python
dx[:,:,:outH*stride,:outW*stride] = dwindows.transpose(0,1,2,4,3,5).reshape(...)
```

Because `stride == kernel` here the windows tile rather than overlap, so unlike convolution this *is* a safe reshape. The transpose `(0,1,2,4,3,5)` interleaves window index and within-window index back into raster order. The slice `[:outH*stride, :outW*stride]` drops a ragged final row or column when the size is not divisible, a case the gradient check covers explicitly with a 7×7 input.

`AdaptiveMaxPoolHeightForward`/`Backward` route the gradient only to the arg-max row per `(batch, channel, column)`. `AdaptiveAvgPoolWidthForward` is a mean, so its backward is `np.repeat(dout, W, axis=2)/W` — each of the `W` positions receives an equal `1/W` share. Those two one-liners are the gradients of the two pooling choices that distinguish the models.

### B.3.4 Linear — the leading-dimension detail

```python
flatX    = x.reshape(-1, x.shape[-1])
flatDout = dout.reshape(-1, dout.shape[-1])
dw = flatDout.T @ flatX
db = flatDout.sum(axis=0)
dx = (dout @ weight).reshape(x.shape)
```

The text head is applied to a `(T,B,2H)` sequence, so `x` has *two* leading dimensions rather than one. Collapsing everything but the last axis lets one `Linear` serve both the `(B,in)` writer-ID head and the `(T,B,in)` text head, and it is correct because a linear layer treats every leading index independently, so the weight gradient is the sum over all of them. The check covers both 2-D and 3-D cases.

### B.3.5 Dropout — inverted

```python
mask = (rng.random(x.shape) >= p)/(1.0-p)
```

The `1/(1-p)` scale is applied at **training** time, so inference is a plain identity pass with no rescaling — which is why `Classify.AuthorClassifierCNNNumpy.forward` simply skips `author_head.2`. The mask is stored and reused by the backward.

### B.3.6 LSTM — forward and backpropagation through time

Gate blocks are ordered `[input, forget, cell, output]` within the `4H`-row weight matrices, PyTorch's convention, so a checkpoint trained here loads into `Classify.py` as-is.

Forward, per timestep:

$$\begin{aligned}
(i,f,g,o) &= \text{split}\big(x_t W_{ih}^\top + b_{ih} + h_{t-1} W_{hh}^\top + b_{hh}\big) \\
i,f,o &= \sigma(\cdot), \quad g = \tanh(\cdot) \\
c_t &= f \odot c_{t-1} + i \odot g, \qquad h_t = o \odot \tanh(c_t)
\end{aligned}$$

Both bias vectors are present and both are added — mathematically redundant, since their sum is one bias, but required for state-dict compatibility, and it is why `dBiasIh` and `dBiasHh` receive the same value in the backward.

The backward is the BPTT recurrence, walking `t` from `T-1` down:

```python
dh  = dOutputs[t] + dhNext          # gradient from above AND from the next step
do  = dh*tanhC;    doPre = do*o*(1-o)
dc  = dcNext + dh*o*(1-tanhC**2)
df  = dc*cPrev;    dfPre = df*f*(1-f)
di  = dc*g;        diPre = di*i*(1-i)
dg  = dc*i;        dgPre = dg*(1-g**2)
dcPrev = dc*f
```

The two lines carrying the recurrence are `dh = dOutputs[t] + dhNext` and `dc = dcNext + dh*o*(1-tanhC²)`: each hidden state influences the loss both through its own output and through every later timestep, so both paths are summed. The derivatives `σ' = σ(1-σ)` and `tanh' = 1-tanh²` are evaluated from the *cached activations* rather than recomputed, which is what the cache is for. `dcPrev = dc*f` is the constant-error-carousel path, and it is the reason an LSTM can propagate gradient across all 160 timesteps.

`BidirectionalLstmBackward` handles the direction bookkeeping: the backward direction ran on the time-reversed input, so its incoming gradient is reversed *before* `LstmBackward` and the resulting `dx` reversed *back* afterwards. The two directions' input gradients are then **summed**, because both read the same input sequence.

### B.3.7 `CtcLoss` — Graves et al. (2006) in log space

The most mathematically substantial function in the project.

The label sequence is blank-interleaved to `ext` of length `S = 2L+1`. Forward variables accumulate in log space:

$$\ln\alpha_t(s) = \ln p_t(\text{ext}[s]) + \text{logsumexp}\Big(\ln\alpha_{t-1}(s),\ \ln\alpha_{t-1}(s-1),\ [\ln\alpha_{t-1}(s-2)]\Big)$$

where the third term is admitted only when `ext[s] != 0` and `ext[s] != ext[s-2]` — no skipping into a blank, and two identical labels must be separated by one. Backward variables `β` satisfy the mirrored recursion. The total is

$$\ln p(y \mid x) = \text{logsumexp}\big(\ln\alpha_T(S{-}1),\ \ln\alpha_T(S{-}2)\big)$$

since a valid path may finish on the last label or on the trailing blank.

**The gradient.** Rather than backpropagating through a separate log-softmax, the occupation probabilities are formed directly:

```python
logGamma = logAlpha + logBeta - logP
occ[:,ext[s]] += gamma[:,s]          # for each s
dRawLogits[:,b,:] = probs[:,b,:] - occ
```

giving `∂L/∂(pre-softmax logits) = y − occupancy`, Graves equations 15–16. It is the same simplification as softmax-plus-cross-entropy's `pred − target`. This is why `TextModelForward` deliberately returns **raw logits, not log-probabilities**, and why `CtcLoss` does its own log-softmax internally — composing a separate log-softmax backward with this would be both slower and numerically worse.

`LogSumExp(a,b)` subtracts the max before exponentiating and wraps the arithmetic in `np.errstate(invalid="ignore")`, because `-inf − (-inf)` is evaluated eagerly by NumPy even where `np.where` discards it.

**The normalisation convention.** `return totalLoss/B, dRawLogits/B`. Each line contributes its whole `−ln p(y|x)` and the division is by **batch size only**, with no per-length term. A quoted loss from this project is therefore a mean **per line**, not per character: a long line counts for more than a short one, and the figure is not comparable with a per-character loss from elsewhere. Every line uses the full `T = 160` steps including padding.

The per-sample Python loop over the batch is unavoidable, because each sample has a different `S` and the lattices are ragged.

### B.3.8 Cross-entropy, optimisers, schedulers

`CrossEntropyLoss` uses the same combined-gradient trick: `dLogits = softmax(logits)`, then `−1` at the target index, then `/B`. The loss is the mean negative log-probability of the true class.

**`RMSprop`**, used for the text model. `sqAvg = α·sqAvg + (1-α)·g²`, then `p -= lr·g/(√sqAvg + ε)`, with `α = 0.99` and `ε = 1e-8` matching `torch.optim.RMSprop`'s defaults. Weight decay is added **to the gradient** (`g = g + wd·p`), which is the classic L2-via-gradient form — what `torch.optim.RMSprop` does.

**`AdamW`**, used for the writer-ID model. First and second moments with `β = (0.9, 0.999)`, bias-corrected by `1-β^t`, and weight decay **decoupled**: `p -= lr·(m̂/(√v̂+ε) + wd·p)`. The contrast with RMSprop above is exactly the distinction between L2 regularisation and weight decay, and the two classes implement both forms, differently and deliberately.

`ReduceLrOnPlateau` halves the learning rate after `patience` epochs without improvement, floored at `minLr`. The test is `badEpochs > patience`, so with `patience=5` it fires on the sixth bad epoch, matching PyTorch. `CosineAnnealingLr` sets `lr = ½·baseLr·(1+cos(π·epoch/tMax))`, decaying to zero at `tMax`.

`ClipGradNorm` computes the **combined** L2 norm across all gradients and scales every one by the *same* factor when it exceeds `maxNorm`. A global rescale preserves the gradient's direction, where per-tensor clipping would not. It returns the pre-clip norm so it can be logged.

---

## B.4 `GradCheckNumpyNN.py` — verifying the backward passes

Run standalone: `python GradCheckNumpyNN.py`. NumPy only.

### B.4.1 Method

For each operation a small random input and a **fixed random output gradient** `R` are built, defining the scalar `L = Σ(out ⊙ R)`. That construction makes `∂L/∂out = R` exactly, so `R` is precisely what is fed to the backward pass. For the two loss functions the scalar is the loss itself. The analytic gradient is compared element by element against the central difference

$$\frac{\partial L}{\partial\theta_i} \approx \frac{L(\theta + \varepsilon e_i) - L(\theta - \varepsilon e_i)}{2\varepsilon}, \qquad \varepsilon = 10^{-5}$$

Central rather than forward difference, because its truncation error is $O(\varepsilon^2)$ rather than $O(\varepsilon)$. **Everything runs in `float64`** — `NumpyNN.py` is dtype-agnostic and propagates whatever it is handed — because in `float32` the subtraction `L(θ+ε) − L(θ−ε)` would be swamped by rounding and `ε = 1e-5` would mean nothing. Up to 40 randomly chosen elements are probed per tensor, which samples the whole tensor while keeping a full run affordable.

Two metrics are reported: `relDiff = max|gₐ−gₙ| / max(1, max|gₐ|+max|gₙ|)` against a tolerance of `1e-6`, and `elemRel = max |gₐ−gₙ|/(|gₐ|+|gₙ|)` as a second opinion that does not shrink when the gradient is small.

Gradients are checked with respect to **inputs and every learnable parameter** — conv weight and bias, BatchNorm γ and β, linear weight and bias, and all four weight and bias blocks per direction per layer of the LSTM.

### B.4.2 Handling the non-smooth and stochastic operations

- **ReLU and max-pool** take inputs from a *permuted linspace* (`Distinct`), so all values are pairwise distinct by far more than `2ε`. Otherwise a perturbation could step an element across the kink at zero or flip a window's arg-max, and then the **numerical** gradient would be the wrong one rather than the analytic one.
- **Dropout** replaces the RNG with a `FixedRng` stub returning pre-drawn uniforms, so the mask is identical across every perturbed evaluation. It is also checked in eval mode, where it must be the identity.
- **BatchNorm** is checked in **train** mode, the only mode whose backward is ever used, and a fresh copy of the running statistics is passed on every call because the forward mutates them in place.
- `Check` asserts `np.shares_memory(flatTheta, theta)`, i.e. that `ravel()` returned a view so the in-place perturbation is actually visible to `lossFn`. A silent copy there would make every check trivially pass.
- `Fail()` records an operation that *errored out* as a failure with its reason, rather than omitting it, so a crash cannot be mistaken for a clean sheet.

### B.4.3 Results

```
configurations checked : 78
passed                 : 77
failed                 : 1
worst relDiff anywhere : 1.389e-10
total elements probed  : 2044
```

The worst disagreement anywhere is `1.4e-10` against a `1e-6` tolerance — four orders of margin — across 2,044 probed elements covering conv in five shape, padding and bias variants including 1×1 and a non-square 2×4 kernel; BatchNorm; ReLU; three max-pool configurations including a ragged one; both adaptive pools; linear in 2-D, 3-D and bias-free; dropout in both modes; LSTM; bidirectional LSTM; two-layer bidirectional LSTM; four CTC configurations; and two cross-entropy ones.

The single failure is the `target[]` CTC case — an *empty* label sequence. With `L=0` the extended sequence is `ext=[0]` so `S=1`, but

```python
prev2      = np.concatenate(([NEG,NEG], logAlpha[t-1,:-2]))
nextMinus2 = np.concatenate((nxt[2:],   [NEG,NEG]))
```

come out length 2 instead of length `S`, and the assignment `logAlpha[t] = combined + lp[t,ext]` raises `ValueError`. For every `S ≥ 3` the same expressions have length `S` and the gradient is correct, which is why all non-empty targets pass. It is a shape constraint on a degenerate input, and that input cannot arise from real data because `IAMLineDatasetRaw` skips any sample with `len(target)==0`.

The file separately records, without counting it as a configuration, that an over-long label (`2L+1 > T`) gives `loss = inf` and `nan` gradients — the correct answer for an infeasible alignment, and the reason such samples are filtered upstream by `MAX_SAFE_LABEL_CHARS`.

---

## B.5 `TrainText.py` — the data pipeline

No model code lives in this file. The PyTorch `PaperCRNN` class, `DecodeCtc` and the training loops moved out — the model to `Classify.py`'s NumPy version, training to `TrainScratch.py`. What remains is the pipeline, reused unchanged by the from-scratch trainer.

### B.5.1 Fixed-size fitting and tensor conversion

`ResizeLineImageFixed` is the same single-scale-factor fit as `SegmentLean.ToClassifierFormat`: uniform scaling, never independent width and height squashing, which measurably distorts stroke proportions on real crops. Left-aligned, vertically centred, white-padded.

```python
arr = np.array(pilImg, dtype=np.float32)/255.0
arr = 1.0 - arr          # ink becomes high
arr = (arr - 0.5)/0.5    # to roughly [-1,1]
```

Ink is **inverted to high values** so the informative signal is a large positive rather than a dip below a bright background, then centred on zero — which is what the Kaiming-initialised first conv layer expects and what makes BatchNorm's job easy from the first step.

`FrameXToPixel` is the exact inverse of that fit's x mapping:

```python
def FrameXToPixel(canvasX,rawW,origH):
    s = min(INPUT_HEIGHT/origH,INPUT_WIDTH/rawW)
    return float(np.clip(canvasX/s,0.0,float(rawW)))
```

Because the fit is a single uniform scale `s` with the image pasted at `x=0`, the forward map is `x_canvas = s·x_orig` and the inverse is a division. `BuildStyleProfile` uses it to turn a CTC frame index — a canvas-space position — into a column of the real line crop before cutting glyphs out of it, so every glyph boundary depends on it.

### B.5.2 Augmentation — the paper's Table 3 recipe without torchvision

All four transforms are PIL plus NumPy linear algebra.

```python
M = T2 @ R @ Sh @ T1          # translate to origin, shear, rotate, translate back
Minv = np.linalg.inv(M)       # PIL wants the inverse map
```

PIL's `Image.AFFINE` maps each *output* pixel back to its source, so it needs `M⁻¹`. The inverse is computed **numerically** with `np.linalg.inv` rather than derived algebraically, which keeps a sign error out of the one transform every augmented sample passes through.

`FindPerspectiveCoeffs` sets up the standard four-point-correspondence system

$$x' = \frac{ax+by+c}{gx+hy+1}, \qquad y' = \frac{dx+ey+f}{gx+hy+1}$$

linearised into two rows per correspondence and solved with `np.linalg.solve`.

| Transform | Parameters | Why those |
|---|---|---|
| `AugmentShear` | `k ~ U(-0.6, 0.6)`, angle `= atan(k)` | the paper's Table 3 form: sampling `k` then taking `atan` makes the *slope* uniform rather than the angle, which is the natural parameterisation of a shear |
| `AugmentRotate` | `± 2.5°` | segmentation already deskews a line crop, so this models residual error only |
| `AugmentPerspective` | corners jittered by `U(0.03, 0.08) × half-dimension` | a mild perspective jitter standing in for the paper's Luo et al. geometric transforms, whose implementation is not vendored here |
| `AugmentElastic` | `σ ∈ {3,4}`, `α ∈ {15,20}` | Simard et al. elastic distortion: a uniform random displacement field, Gaussian-smoothed with `σ` then scaled by `α`, applied by `map_coordinates` bilinear interpolation. σ sets how *local* the warp is, α how *strong* |

```python
def AugmentLineImage(pilImg):
    candidates = [fn for fn in AUGMENTATIONS if random.random() < 0.5]
    if not candidates: return pilImg
    return random.choice(candidates)(pilImg)
```

Each of the four is gated independently at p = 0.5, then **exactly one** of the survivors is applied. So no augmentation happens with probability `(1/2)⁴ = 1/16`, each transform's marginal probability is `15/16 × 1/4 ≈ 23%`, and transforms are never composed. That is the paper's "combined" recipe.

Elastic is called **after** resizing: on a raw high-DPI crop it allocates several full-resolution `float64` arrays per call, which exhausted memory under parallel data loading. Both `__getitem__` implementations resize first, then augment.

`scipy.ndimage` (`gaussian_filter`, `map_coordinates`) is used only here and only for elastic augmentation — an offline training-path dependency, not a runtime one.

### B.5.3 `LineImageCache` — what gets cached, and why that matters

Page segmentation is expensive, so results are cached. The important choice is *what* is cached:

> The cache holds **only** preprocessed line-crop PNG bytes, keyed by `"author/page.png"`. It is completely independent of label *text*. Labels are re-read fresh from `_labels.txt` on every run.

So correcting a label costs nothing — no re-segmentation. The cache only needs rebuilding if the segmentation code itself changes. Since label correction is iterative and segmentation takes minutes per page, this is the difference between a workable and an unworkable loop.

Crops are stored as **PNG bytes** rather than raw arrays, because mostly-white handwriting scans compress very well. Storage is sharded with a manifest:

- `manifest.pkl` records which pages are done and which shard files exist;
- each flush writes **only new** pages into a fresh `shard_NNNNNN.pkl` and never rewrites an old one;
- both writes go to a `.tmp` then `os.replace`, which is atomic, so an interrupt mid-write cannot corrupt the manifest;
- `SaveEveryNPages=10` bounds the work lost to a crash;
- `KeyboardInterrupt` is caught, pending pages flushed, and the exception re-raised, so Ctrl-C is a safe pause rather than a loss.

An unreadable cache triggers a rebuild rather than an exception — the case that motivated it being a cache written by the old `torch.save` format. A segmentation error on one page is caught and that page skipped with a message.

The cache is built with `expectedLineCount=None, labelLines=None`, so segmentation is never shaped by what the label file claims and a label cannot bias the crops it will later be matched against.

### B.5.4 `IAMLineDatasetRaw` — assembly and quality gates

Joins cached images with current labels, applying three filters that each print what they dropped:

1. **Line-count mismatch** → skip the whole page, naming the page and both counts. Pairing `n` images with `m` labels by index when `n ≠ m` would mislabel every line on the page.
2. **Empty target** → skip the line.
3. **Oversized target** (over 120 chars) → skip the line, printing the page, line index and first 80 characters, since the real diagnosis is a collapsed or merged segmentation on that page.

**The holdout split is per page:** `LabeledPages[-1]`, the last labelled page, is held out for each author with at least `MinPagesForHoldout=3` labelled pages. That is stricter than a random line split, because no line from a validation page was seen in training — which matters when lines from one page share paper, pen, lighting and that day's handwriting.

`ResolveDatasetRoot` handles dataset roots that nest the author folders one level down in `data/`.

`TrainEvalView` is a thin index wrapper whose job is to let the train and validation subsets toggle augmentation independently without deep-copying the dataset: augmentation on for train, off for validation.

`CollateFn` returns `(stacked images, flat concatenated targets, target lengths, texts)`. `TrainScratch.TextNumpyBatch` splits the flat array back into a per-sample list, because `NumpyNN.CtcLoss` takes ragged per-sample sequences.

### B.5.5 `TrainAuthor.py` — train/inference consistency

`AuthorLabeledView` surfaces an author index per sample — the base class always tracked `author_to_idx` but never exposed it — and **stroke-normalises every crop before training**:

```python
normImg = PIO.StrokeNormalize(np.array(gray))
if normImg is not None:
    gray = normImg
```

`Classify.ClassifyImage` stroke-normalises by default everywhere in this project, so training on raw ink and judging on stroke-normalised input would be a genuine train/inference mismatch. This makes the two consistent by construction, falling back to the raw crop only when `StrokeNormalize` found no usable ink. Together with A.2.4's argument — the gantry redraws with its own pen, so writer identity must be carried by geometry rather than pen character — this is why the writer-ID model sees skeletons rather than ink.

`AddPersonalSamples` brings the five personal writers in: segment each photo with **`SegmentLean`** (these are photographs, not IAM scans), read the hand-verified transcript, drop lines labelled with the literal string `"MESS"` (the manual marker for a line not worth training on), truncate to `min(len(crops), len(gt))`, shuffle with the shared `SPLIT_SEED`, and hold out `max(1, round(0.15·n))`. Samples are appended into `base.samples` in the same dict shape, so nothing downstream can distinguish them from IAM-derived samples.

---

## B.6 `TrainScratch.py` — the training program

One file trains both models, because everything through stage 3 is identical between them: `InitBackboneParams`, `ConvBlockForwardTrain`/`BackwardTrain`, `BackboneForwardTrain`/`BackwardTrain`, `EvalBackboneForward`, `IterateBatches` and the checkpoint helpers are written once and used twice.

### B.6.1 Initialisation

```python
def InitConv(rng,outCh,inCh,kh,kw):
    std = np.sqrt(2.0/(inCh*kh*kw))        # Kaiming normal, ReLU gain
def InitLinear(rng,outF,inF):
    bound = 1.0/np.sqrt(inF)               # PyTorch nn.Linear default
def InitLstmDirection(rng,inputSize,hiddenSize):
    bound = 1.0/np.sqrt(hiddenSize)        # PyTorch nn.LSTM default
```

Kaiming normal with the ReLU gain of 2 for convolutions: the factor 2 compensates for ReLU discarding half the signal, keeping forward activation variance stable through twelve layers. The uniform `±1/√fan` ranges reproduce PyTorch's `nn.Linear` and `nn.LSTM` defaults. BatchNorm starts at `γ=1, β=0, running_mean=0, running_var=1`. These approximate PyTorch's defaults — annealing over a real run washes out init differences, so what matters is the right general scale.

`InitLstmDirection` is called **separately for the forward and reverse direction** of each layer, so the two directions get independent draws rather than mirrored weights.

`lstmInputSizes = [128, HIDDEN_SIZE*2]`: layer 0 sees the backbone's 128 channels, layer 1 sees 512, because layer 0's bidirectional output concatenates two 256-wide hidden states.

### B.6.2 Training-mode versus eval-mode forward

Each model has two forward paths:

- `TextModelForward` / `AuthorModelForward` use `NumpyNN`'s train-mode layers — BatchNorm on **batch** statistics, dropout active, caches retained for the backward.
- `EvalTextForward` / `EvalAuthorForward` call `Classify.py`'s primitives directly — BatchNorm on **running** statistics, no dropout, no caches.

So validation is measured through the *same code* inference will run: `EvalBackboneForward` is literally `CLS.ConvBlockForward`. Both eval paths stop one step short of log-softmax so `NumpyNN.CtcLoss` can be handed raw logits.

`TrainableParams` filters `running_mean`/`running_var` out by key suffix, so the optimiser's parameter dict holds only true parameters while the buffers are updated in place by the forward.

### B.6.3 The two recipes

| | Text (`PaperCRNN`) | Writer ID (`AuthorClassifierCNN`) |
|---|---|---|
| Loss | `CtcLoss` | `CrossEntropyLoss` |
| Optimiser | RMSprop, `lr=1e-3`, `wd=1e-5` | AdamW, `lr=1e-3`, `wd=1e-4` |
| Schedule | `ReduceLrOnPlateau(0.5, patience=5, minLr=1e-5)` on **val loss** | `CosineAnnealingLr(tMax=epochs)` |
| Grad clip | none | `ClipGradNorm(5.0)` |
| Default epochs / batch | 50 / 16 | 20 / 16 |
| Seed | 7 | 0 (batching) and 1 (dropout, `seed+1`) |
| Selection metric | best validation **loss** | best validation **accuracy** |
| Dropout | none, per the paper | 0.2 in the head |

Two separate RNGs for the writer-ID model — `rngBatch` for shuffling, `rngDropout` for masks — so changing one does not perturb the other, which keeps a re-run reproducible in a way a single shared stream would not.

Weight decay is ten times higher for writer ID, which follows the task: 13 classes over a few thousand lines overfits far more readily than 74-class sequence transcription over tens of thousands.

Validation differs in kind. `ValidateTextModel` reports both mean CTC loss and character accuracy via `Levenshtein` — the accuracy is the number of interest, while the loss drives the scheduler and checkpoint selection. `ValidateAuthorModel` reports top-1 accuracy.

### B.6.4 Checkpointing, resume, warm start

```python
SaveCheckpoint(bestPath, params, extra={"__epoch__":..., "__best__":...})
SaveCheckpoint(ckptPath, params, extra={...})      # every epoch, regardless
```

Two files per run: `..._best.npz` written only on improvement, and `..._checkpoint.npz` written every epoch. Metadata travels in dunder keys inside the same `.npz`, and `Classify.LoadStateDictAny` strips anything starting with `__`, so one file serves both resumption and inference.

The writer-ID checkpoint also carries `__author_names__`, the author list sorted by index, because `.npz` cannot hold a dict; `LoadStateDictAny` rebuilds `{name: idx}` from it. The class-index mapping therefore travels with the weights, so a checkpoint always knows what its output indices mean.

`LoadParamsInto` copies a checkpoint's arrays into a freshly initialised dict **wherever name and shape both match**, returning the count loaded. One function serves `--resume` (same run after a crash, with epoch counter and best score restored) and `--warm-start` (initialise from a different trained checkpoint). The shape-match condition is what lets a warm start work across a changed output layer: a backbone transfers, a head with a different class count simply does not load, and the printed `loaded/total` says which happened.

`AppendLog` writes one `key=value` line per epoch next to the checkpoints, append-only.

### B.6.5 The two data sources

`--source pages` segments `IAMpages671` from scratch through `IAMLineDatasetRaw`. `--source teklia` reads the pre-segmented, pre-labelled Teklia/IAM-line export as a flat folder of PNG plus `_labels.txt` sidecars. `TekliaLineDataset.__getitem__` mirrors `TrainEvalView.__getitem__` call for call — same resize, same augment, same encode — so it is a drop-in alternative source for one unchanged training loop. The Teklia split needs no holdout logic, since HuggingFace already separates train and validation into folders.

---

## B.7 `Classify.py` — inference

Merged from what used to be a four-module `np_inference` package. No torch, and no pickle parsing either: every checkpoint is `.npz`, and `LoadStateDictAny` is a few lines.

The layer primitives are the forward halves of `NumpyNN`'s without the caches, with BatchNorm folded:

```python
scale = weight/np.sqrt(running_var+eps)
shift = bias - running_mean*scale
return x*scale.reshape(1,-1,1,1) + shift.reshape(1,-1,1,1)
```

At inference BatchNorm is an affine map with fixed coefficients, so the four stored tensors collapse into one multiply and one add per channel — identical to the training-mode formula evaluated with running statistics, and materially cheaper.

**Checkpoint precedence** is defined identically here and in `Evaluate.py`: `..._scratch_best.npz`, then `..._joint_best.npz`, then `..._hf_best.npz` for text; `..._10new_scratch_weights.npz`, then `..._shape_10new_weights.npz`, then `..._10new_weights.npz` for writer ID. The first existing file wins, so anything newly trained by `TrainScratch.py` is picked up everywhere with no explicit path argument.

`DecodeCtc` is greedy decoding: arg-max per timestep, collapse runs of the same token, drop the blank. Not beam search — the paper reports greedy, and this follows it.

`Levenshtein` is the two-row dynamic program, O(mn) time and O(min(m,n)) space, with `if len(a)<len(b): return Levenshtein(b,a)` so the shorter string drives the row width.

`CharAcc = max(0, 1 − edit_distance/len(truth))`. The floor at zero matters, because edit distance can exceed the reference length when the prediction is longer, which would otherwise give a negative accuracy.

`WordAcc` is **not** a per-position match. It is the longest common subsequence of word lists, case-insensitive, divided by the number of true words — the fraction of the intended words recovered **in order**. That is the right metric when a recogniser may insert or drop a word, since a positional comparison would mark every word after an insertion wrong.

`ClassifyImage` stroke-normalises by default before resizing, matching training, and returns `(predicted index, full softmax vector)` — the whole probability vector, so callers can threshold on confidence or sum probabilities across a page's lines. Callers resolve the index through the inverted `author_mapping` that travelled with the checkpoint.

On a labelled IAM line from author 150 the writer-ID classifier returns `150` at p = 1.000, and the `joint_best` text checkpoint reads four lines at 0.984 mean character accuracy.

---

## B.8 The two one-time tools

### B.8.1 `ConvertPtToNpz.py` — reading PyTorch checkpoints without PyTorch

A `.pt` file is a ZIP containing a pickle (`data.pkl`) plus raw tensor storage blobs. This reads them with a `pickle.Unpickler` subclass that intercepts the torch-specific classes:

| Intercepted | Replaced with |
|---|---|
| `torch._utils._rebuild_tensor_v2` | `RebuildTensor` |
| `torch.FloatStorage` and friends | `StorageStub(np.dtype('<f4'))` etc. |
| `torch._utils._rebuild_parameter` | `lambda data, requires_grad, hooks: data` |
| `collections.OrderedDict` | itself |

`persistent_load` is where tensor bytes arrive: the pickle refers to storages by key, and this reads `data/{key}` out of the ZIP and wraps it with `np.frombuffer` at the declared dtype and element count. `RebuildTensor` then applies the view:

```python
byte_strides = tuple(s*itemsize for s in stride)
view = np.lib.stride_tricks.as_strided(storage[storage_offset:], shape=tuple(size), strides=byte_strides)
return view.copy()
```

**PyTorch strides are in elements while `as_strided` wants bytes**, hence the multiply by `itemsize`. The `.copy()` materialises the view so nothing aliases the buffer afterwards. The explicit little-endian dtypes (`<f4`) are right because the format is little-endian regardless of host.

Output goes through `np.savez` in exactly `TrainScratch.SaveCheckpoint`'s layout — state-dict keys as-is, `author_mapping` flattened to `__author_names__` sorted by index — so a converted weight and a from-scratch one load through the same path. Because `find_class` whitelists, nothing outside the listed names is constructible from the archive.

### B.8.2 `ExportTekliaLines.py` — the only `datasets` dependency, quarantined

Downloads `Teklia/IAM-line` from HuggingFace once and writes it as plain PNG plus `_labels.txt` sidecars, the same shape every other dataset in the project uses. It applies the same filtering as the training path — `CHAR_TO_IDX` membership, drop empty, drop over `MAX_SAFE_LABEL_CHARS` — and reports the three counts.

This is the only place in the project that needs the `datasets` library. Nothing in training, classification or reproduction imports it, and once the files are on disk it is never touched again; the `ImportError` handler says so explicitly.
