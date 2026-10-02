"""From-scratch (numpy-only) forward pass for AuthorClassifierCNN
(TrainAuthor.py) + a ClassifyImage() matching EvaluateStyle.ClassifyImage's
behavior. No torch import in this file.

StrokeNormalize's binarize/skeletonize steps reuse ProfileIO's
BinarizeLine/Skeletonize/CoreBand and SegmentPage.Dilate -- both are
pure numpy with no torch dependency."""

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from . import layers as L
from . import weight_io as W

SCRIPT_DIR = Path(__file__).resolve().parent.parent  # .../AuthorReproductionStuff
CNN_DIR = SCRIPT_DIR.parent  # .../Software/CNN
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(CNN_DIR))

import SegmentPage as _RawOps  # noqa: E402  (pure numpy, see module docstring)
import ProfileIO as _SP  # noqa: E402  (BinarizeLine/Skeletonize/CoreBand, pure numpy)

from .text_model import resize_line_image_fixed, tensor_from_resized  # noqa: E402

PEN_WIDTH_XH = 0.14  # matches TrainAuthorShape.PEN_WIDTH_XH

# Mirrors EvaluateStyle.py's AUTHOR_WEIGHTS fallback chain exactly.
_SHAPE = CNN_DIR / "weights" / "author_shape_10new_weights.pt"
_W1 = CNN_DIR / "weights" / "author_classifier_10_weights.pt"
_W2 = CNN_DIR / "weights" / "author_fast_10_weights.pt"
DEFAULT_AUTHOR_WEIGHTS = _SHAPE if _SHAPE.exists() else (_W1 if _W1.exists() else _W2)


def StrokeNormalize(gray, penWidthXh=PEN_WIDTH_XH):
    """Ported from TrainAuthorShape.StrokeNormalize: real ink -> centreline
    -> re-inked at one constant width -> inverted."""
    ink = _SP.BinarizeLine(gray)
    if not ink.any():
        return None
    band = _SP.CoreBand(ink)
    xh = float(band[1] - band[0]) if band else 0.0
    if xh < 4:
        xh = max(8.0, ink.shape[0] / 3.0)
    w = int(np.clip(round(penWidthXh * xh), 2, 9))
    sk = _SP.Skeletonize(ink)
    if not sk.any():
        return None
    k = 2 * (w // 2) + 1
    out = _RawOps.Dilate(sk, k, k)
    return Image.fromarray(((~out) * 255).astype(np.uint8))


def _conv_block_forward(x, sd, prefix, n_layers):
    for i in range(n_layers):
        c = i * 3
        x = L.conv2d(x, sd[f"{prefix}.{c}.weight"], sd[f"{prefix}.{c}.bias"], padding=1)
        x = L.batchnorm2d(x, sd[f"{prefix}.{c+1}.weight"], sd[f"{prefix}.{c+1}.bias"],
                           sd[f"{prefix}.{c+1}.running_mean"], sd[f"{prefix}.{c+1}.running_var"])
        x = L.relu(x)
    return x


class AuthorClassifierCNNNumpy:
    def __init__(self, checkpoint_path=None):
        self.checkpoint_path = Path(checkpoint_path or DEFAULT_AUTHOR_WEIGHTS)
        raw = W.load_pt(self.checkpoint_path)
        self.sd = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
        self.author_mapping = raw.get("author_mapping") if isinstance(raw, dict) else None

    def forward(self, x):
        """x: (B,1,H,W) numpy array -> (B,num_authors) raw logits."""
        sd = self.sd
        x = _conv_block_forward(x, sd, "stage1", 2)
        x = L.maxpool2d(x, 2, 2)
        x = _conv_block_forward(x, sd, "stage2", 4)
        x = L.maxpool2d(x, 2, 2)
        x = _conv_block_forward(x, sd, "stage3", 6)
        x = L.adaptive_max_pool_height(x)              # (B, 128, 1, W)
        x = x[:, :, 0, :]                                # (B, 128, W)
        pooled = L.adaptive_avg_pool_width_1d(x)[:, :, 0]  # (B, 128)
        h = L.linear(pooled, sd["author_head.0.weight"], sd["author_head.0.bias"])
        h = L.relu(h)
        # author_head.2 is Dropout(0.2) -- inactive at inference, skipped.
        logits = L.linear(h, sd["author_head.3.weight"], sd["author_head.3.bias"])
        return logits


def ClassifyImage(pil_image, model, author_mapping=None, apply_stroke_normalize=True):
    """Matches EvaluateStyle.ClassifyImage's exact return signature:
    (predicted_index, probs) -- a class index into the model's output and
    the softmax probability vector, same as the torch version. Callers
    resolve predicted_index to an author-id string themselves via
    {v: k for k, v in author_mapping.items()}, same as EvaluateStyle.py
    does outside ClassifyImage. `author_mapping` is accepted here only so
    callers that don't want the module-level default can pass one in; it is
    unused when apply_stroke_normalize alone is all that's needed."""
    if apply_stroke_normalize:
        normImg = StrokeNormalize(np.array(pil_image.convert("L")))
        if normImg is not None:
            pil_image = normImg
    t = tensor_from_resized(resize_line_image_fixed(pil_image))
    logits = model.forward(t)
    probs = L.softmax(logits, axis=1)[0]
    predIdx = int(np.argmax(probs))
    return predIdx, probs
