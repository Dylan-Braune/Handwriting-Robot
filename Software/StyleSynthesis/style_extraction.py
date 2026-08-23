"""
style_extraction.py

Measures an author's style as a small set of interpretable geometric
parameters (slant angle, x-height, letter-spacing ratio) from their already-
segmented training pages, instead of storing per-character trajectory maps
verbatim. This is the "features per author, applied to an existing alphabet"
approach you described, and it's grounded in real precedent -- see the
research summary in this session (Kotani et al.'s Decoupled Style
Descriptors, ECCV 2020; "Encoding CNN Activations for Writer Recognition",
arXiv:1712.07923) for why separating "what the letters ARE" (base_alphabet.py)
from "HOW this author draws them" (this file) generalizes far better to
characters/words never seen in training than a lookup table does.

WHY GEOMETRIC FEATURES INSTEAD OF A LEARNED CNN EMBEDDING, FOR NOW: a learned
style embedding (the DSD/CNN-activation approach) needs many examples per
author to train reliably, and right now you have full segmentation+labels
for what looks like ONE author across the 3 NOGIT/NonDatasetImages pages.
Geometric features (slant, x-height, spacing) are measurable from a single
page with reasonable confidence and are individually interpretable, so you
can sanity-check each one against the source image by eye. Once you have
genuinely multiple authors' labelled pages, swap the "style vector" this
file produces for a learned embedding without changing anything downstream
(glyph_styler.py just consumes whatever style dict it's given).

WHY NOT PER-CHARACTER TRAJECTORY STORAGE: your proposal's FU2.5/2.6 sketch
that out, but it only reproduces text built from EXACTLY the characters/
groups seen in training. A slant+x-height+spacing deformation applied to
base_alphabet.py's Hershey skeletons works for ANY string, including words
never seen from that author -- directly what "very dynamic" in your last
message was asking for.

Only depends on NonDatasetSegmenterFP's PUBLIC ProcessPage() output (results
+ meta) -- does not read or duplicate its internals, so it stays decoupled
from that file (which you've asked not to have touched beyond the pixel-
completeness fix already made).
"""

import numpy as np
from PIL import Image


def _ink_mask_from_crop(rawCropRgbOrGray):
    """raw_crop from NonDatasetSegmenterFP results: white background (255),
    ink pixels are the original grayscale value (dark). Threshold at a fixed
    cutoff -- these crops are already clean (segmenter's own binarization
    decided what's ink), so this is just picking out the painted pixels vs
    the white canvas fill, not re-doing binarization."""
    arr = np.asarray(rawCropRgbOrGray)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    return arr < 250


def estimate_slant_deg(inkMask, angleRangeDeg=25, stepDeg=0.5):
    """Shear-search slant estimation: standard OCR preprocessing technique
    (used e.g. in classic slant-normalization for handwriting recognizers).
    Shears the ink mask by a range of candidate angles and picks the one
    that makes vertical strokes most vertical, measured as the angle that
    maximizes the peakiness (sum of squared column densities) of the
    vertical projection profile -- a shear that aligns ascenders/descenders
    into tight vertical columns spikes this metric; the wrong shear smears
    them across more columns."""
    ys, xs = np.nonzero(inkMask)
    if len(xs) < 20:
        return 0.0
    h, w = inkMask.shape
    bestAngle, bestScore = 0.0, -1.0

    for angleDeg in np.arange(-angleRangeDeg, angleRangeDeg + 1e-9, stepDeg):
        shear = np.tan(np.radians(angleDeg))
        # shear x by shear*y (y measured from vertical center so the shear pivots
        # around the mid-height of the line, not the top)
        shiftedX = xs - shear * (ys - h / 2.0)
        colCounts = np.bincount(np.clip(shiftedX.astype(int) - shiftedX.astype(int).min(), 0, None))
        score = float(np.sum(colCounts.astype(np.float64) ** 2))
        if score > bestScore:
            bestScore, bestAngle = score, angleDeg

    return bestAngle


def estimate_spacing_ratio(inkMask, xHeightPx):
    """Average blank-gap width between letters, expressed as a fraction of
    x-height (so it transfers to any font size). Finds gaps via the column
    projection's zero-runs, excluding an outlier-filtered word-gap (much
    wider than a letter-gap) so word spacing doesn't contaminate the
    letter-spacing estimate."""
    colHasInk = inkMask.any(axis=0)
    gapWidths = []
    run = 0
    for hasInk in colHasInk:
        if not hasInk:
            run += 1
        else:
            if run > 0:
                gapWidths.append(run)
            run = 0

    if not gapWidths:
        return 0.15  # no measurable gaps (single connected cursive blob) -- reasonable default

    gapWidths = np.array(gapWidths, dtype=np.float64)
    # word-gaps are typically >2.5x the median letter-gap -- drop them for THIS estimate
    median = np.median(gapWidths)
    letterGaps = gapWidths[gapWidths <= median * 2.5] if median > 0 else gapWidths
    if len(letterGaps) == 0:
        letterGaps = gapWidths
    return float(np.mean(letterGaps)) / max(1.0, xHeightPx)


class AuthorStyle:
    def __init__(self, slantDeg, xHeightPx, spacingRatio, nLinesUsed):
        self.slantDeg = slantDeg
        self.xHeightPx = xHeightPx
        self.spacingRatio = spacingRatio
        self.nLinesUsed = nLinesUsed

    def __repr__(self):
        return (f"AuthorStyle(slant={self.slantDeg:.1f}deg, "
                f"xHeight={self.xHeightPx:.1f}px, "
                f"spacingRatio={self.spacingRatio:.3f}, "
                f"from {self.nLinesUsed} line(s))")


def extract_author_style(results, meta, maxLines=None):
    """results/meta: exactly what NonDatasetSegmenterFP.ProcessPage(imgPath)
    returns. Averages slant + spacing across every TEXT line (skips MESS
    diagram blocks, which obviously aren't handwriting style signal), so one
    noisy line doesn't dominate. xHeight comes straight from meta['textH'] --
    the segmenter already computes a robust page-wide estimate of this, no
    reason to duplicate that logic here."""
    textResults = [r for r in results if r["tag"] == "TEXT"]
    if maxLines is not None:
        textResults = textResults[:maxLines]
    if not textResults:
        raise ValueError("No TEXT lines in results -- can't extract a style from a page with no handwriting on it.")

    xHeightPx = float(meta.get("textH", 20.0))

    slants, spacings = [], []
    for r in textResults:
        mask = _ink_mask_from_crop(r["raw_crop"])
        if mask.sum() < 20:
            continue
        slants.append(estimate_slant_deg(mask))
        spacings.append(estimate_spacing_ratio(mask, xHeightPx))

    if not slants:
        raise ValueError("Every TEXT line had too little ink to measure style from.")

    return AuthorStyle(
        slantDeg=float(np.median(slants)),
        xHeightPx=xHeightPx,
        spacingRatio=float(np.median(spacings)),
        nLinesUsed=len(slants),
    )


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "CNN"))
    import NonDatasetSegmenterFP as Segmenter

    imgPath = input("Path to a training page image (blank = ../CNN/NOGIT/NonDatasetImages/baseline_model.png): ").strip()
    if not imgPath:
        imgPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "CNN",
                                "NOGIT", "NonDatasetImages", "baseline_model.png")

    results, preview, meta = Segmenter.ProcessPage(imgPath)
    style = extract_author_style(results, meta)
    print(f"\nExtracted style: {style}")
    print("\nSanity check -- does this match what you see when you look at the page?")
    print(f"  Slant of {style.slantDeg:+.1f} degrees ({'right-leaning/italic' if style.slantDeg > 1 else 'left-leaning' if style.slantDeg < -1 else 'upright'})")
    print(f"  x-height ~{style.xHeightPx:.0f}px on this page's working resolution")
    print(f"  Letter spacing ~{style.spacingRatio:.2f}x the x-height between letters")
