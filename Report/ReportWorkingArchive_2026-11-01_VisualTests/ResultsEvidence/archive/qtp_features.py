"""
qtp_features.py -- ONE shared, deterministic, NON-NEURAL hand-crafted style
feature extractor, applied identically to real line images and to synthesised
line images (QTP1a).

Pipeline per line (grey uint8 image, ink dark):
  1. ProfileIO.BinarizeLine (Otsu + rule removal + small closing + speck removal)
  2. ProfileIO.CoreBand  -> x-height estimate xh (px)
  3. rescale the GREY image so that xh ~= TARGET_XH px (cv2.INTER_AREA) so that every
     line is measured at the same pixel scale (also keeps the pure-numpy Zhang-Suen
     skeletoniser fast), then repeat 1-2 on the rescaled image
  4. features (all normalised by the x-height of the rescaled image where a length):
       slant_deg        ProfileIO.EstimateSlantDeg (shear search, + = forward lean)
       stroke_w_xh      mean stroke width / xh  (2 * mean EDT value on skeleton)
       ink_density      ink pixels / (xh * ink-extent width)  [ink area per x-height of line]
       word_gap_xh      median interior column gap > 0.5 xh, in xh  (NaN if none)
       comps_per_word   8-connected components (max bbox side >= 0.4 xh, i.e. no dots)
                        divided by the number of words (= number of word gaps + 1)
       char_w_xh        ink-extent width / (number of characters in the text) / xh
       asc_xh           ascender reach above the x-height band / xh (1.5th percentile of ink rows)
       desc_xh          descender reach below baseline / xh (98.5th percentile of ink rows)
       asc_desc_ratio   asc / (asc + desc)
  Pen-weight features (stroke_w_xh, ink_density) are flagged PEN_FEATURES: the
  robot writes every author with ONE pen, so these are not a controllable style
  channel (see SynthesizeHandwriting.UNIFORM_INK).
"""
import numpy as np

import qtp_common as C  # noqa: F401  (sets sys.path)
import ProfileIO as PIO
import SegmentPage as F

TARGET_XH = 28.0
FEATURES = ["slant_deg", "stroke_w_xh", "ink_density", "word_gap_xh", "comps_per_word",
            "char_w_xh", "asc_xh", "desc_xh", "asc_desc_ratio"]
PEN_FEATURES = ["stroke_w_xh", "ink_density"]
SHAPE_FEATURES = [f for f in FEATURES if f not in PEN_FEATURES]


def _band(gray):
    ink = PIO.BinarizeLine(gray)
    if ink.sum() < 40:
        return None, None, None
    band = PIO.CoreBand(ink)
    if band is None:
        return None, None, None
    return ink, band, float(band[1] - band[0])


def _resize_gray(gray, scale):
    import cv2
    h, w = gray.shape
    nw, nh = max(8, int(round(w * scale))), max(8, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(gray, (nw, nh), interpolation=interp)


def line_features(gray, text=None):
    """gray: 2-D uint8 numpy (dark ink on light). text: ground-truth string of the line
    (needed only for char_w_xh). Returns dict feature->float, or None if unmeasurable."""
    from scipy import ndimage as ndi
    gray = np.asarray(gray)
    if gray.ndim == 3:
        gray = gray[..., 0]
    ink, band, xh = _band(gray)
    if ink is None or xh < 4:
        return None
    scale = TARGET_XH / xh
    if abs(scale - 1.0) > 0.05:
        gray = _resize_gray(gray, scale)
        ink, band, xh = _band(gray)
        if ink is None or xh < 4:
            return None
    top, base = band
    ys, xs = np.nonzero(ink)
    x0, x1 = xs.min(), xs.max()
    extent = float(x1 - x0 + 1)

    slant = PIO.EstimateSlantDeg(ink)

    sk = PIO.Skeletonize(ink)
    if sk.sum() < 10:
        return None
    edt = ndi.distance_transform_edt(ink)
    stroke_w = 2.0 * float(edt[sk].mean()) / xh

    ink_density = float(ink.sum()) / (xh * extent)

    col = ink.any(axis=0)
    gaps, run, started = [], 0, False
    for v in col:
        if v:
            if started and run:
                gaps.append(run)
            run, started = 0, True
        else:
            run += 1
    word_gaps = [g / xh for g in gaps if g / xh > 0.5]
    n_words = len(word_gaps) + 1
    word_gap = float(np.median(word_gaps)) if word_gaps else float("nan")

    labels, n = F.LabelComponents(ink, connectivity=8)
    n_big = 0
    if n:
        objs = ndi.find_objects(labels)
        for sl in objs:
            if sl is None:
                continue
            hh = sl[0].stop - sl[0].start
            ww = sl[1].stop - sl[1].start
            if max(hh, ww) >= 0.4 * xh:
                n_big += 1
    comps_per_word = n_big / n_words

    nchar = len(text.strip()) if text else None
    char_w = (extent / nchar / xh) if nchar else float("nan")

    rows = ys.astype(float)
    top_r = np.percentile(rows, 1.5)
    bot_r = np.percentile(rows, 98.5)
    asc = max(0.0, (top - top_r) / xh)
    desc = max(0.0, (bot_r - base) / xh)
    ratio = asc / (asc + desc) if (asc + desc) > 1e-6 else float("nan")

    return dict(slant_deg=float(slant), stroke_w_xh=stroke_w, ink_density=ink_density,
                word_gap_xh=word_gap, comps_per_word=float(comps_per_word),
                char_w_xh=float(char_w), asc_xh=float(asc), desc_xh=float(desc),
                asc_desc_ratio=float(ratio), xh_px=xh, n_words=n_words)


def mean_features(rows):
    """nan-mean of each feature over a list of line_features dicts."""
    out = {}
    for k in FEATURES:
        v = np.array([r[k] for r in rows if r is not None], float)
        out[k] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
    return out
