"""
METHOD 1 -- exemplar / fragment collage.

Stores real fragments the writer ACTUALLY wrote -- whole words, letter-pairs,
letters -- taken straight from their own pages, then builds any target
sentence by tiling those fragments and stitching the seams. Every stroke on
the output is the writer's own ink; nothing is invented.

Pipeline
  1. SegmentPage cuts each photographed page into line crops.
  2. Each crop is CTC-read and matched (monotonic alignment) to the right
     line of the hand transcript in pages/*_labels.txt -- robust to the
     segmenter emitting a few spurious / merged boxes.
  3. Each matched (crop, transcript) pair is CTC force-aligned to get
     per-character x-boundaries (snapped to thin+low ink = ligature crossings).
  4. Image crops are cut for every word and every 1..3-char run and stored.
  5. synthesize(text): greedy longest-match tiling from the library, scaled
     to one x-height, laid on a baseline, seam-stroke where ink heights
     disagree.

Trainable on ANY handwriting -- point PAGES_DIR at a different writer's pages.
Here it is pointed at the student's 3 lab-book pages.

    python method1_exemplar.py --build
    python method1_exemplar.py "the quick brown fox jumps over the lazy dog"
"""
import argparse
import pickle
import random
import sys
from collections import defaultdict

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import _env
from _env import PAGES_DIR, OUT_DIR, LIB_DIR, TEXT_WEIGHTS

from TrainText import (PaperCRNN, CHARSET, CHAR_TO_IDX, decode_ctc, levenshtein,
                       resize_line_image_fixed, tensor_from_resized,
                       frame_x_to_pixel, INPUT_WIDTH)
import SegmentPage as SEG
from ExtractIAMLines import ReadLabelLines
import BuildStyleProfile as SP
import RawImageOps as F

MAX_NGRAM = 3
MAX_PER_KEY = 8
MIN_NGRAM_INSTANCES = 2   # a 2-3 char run seen only ONCE has nothing to be
                          # cross-checked against (consensus needs >=3, and
                          # even a single check needs >=2) -- confirmed on
                          # "do" (1 instance, imperfectly-closed loop) that
                          # greedily preferring it over the well-attested
                          # single letters (8 instances each) makes "dog"
                          # render as "dcg". Below this count, fall back to
                          # single characters instead.
LIB_PATH = LIB_DIR / "method1_exemplar.pkl"

# every fragment is stored as a SKELETON TRAJECTORY (a thin centreline, not
# the raw ink pixels) so the render is one constant pen width throughout --
# see _crop_to_polys / render_polys below.
_UPSCALE = 4
_HOLE_CHECK_MAX_CHARS = 3   # loop-collapse ('o','a','e' eroding to a dot) is
                            # a single-letter phenomenon; checking it on long
                            # multi-word crops only adds cost, not accuracy


# ---------------------------------------------------------------------------
# recogniser
# ---------------------------------------------------------------------------
def load_model(device):
    m = PaperCRNN(num_classes=len(CHARSET) + 1).to(device)
    sd = torch.load(TEXT_WEIGHTS, map_location=device, weights_only=False)
    m.load_state_dict(sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd)
    m.eval()
    return m


def _as_line(pil, aspect=9.0):
    """Pad a word/n-gram crop with white to at least `aspect`:1 before
    reading -- PaperCRNN is trained on full LINES (~15:1) and badly
    misreads a near-square fragment crop once it's stretched to 640x64
    (confirmed earlier this session: the same fragment read at ~50% CER in
    isolation vs ~5% once padded to a line-like aspect)."""
    pil = pil.convert("L")
    w, h = pil.size
    target_w = max(w, int(h * aspect))
    if target_w == w:
        return pil
    out = Image.new("L", (target_w, h), 255)
    out.paste(pil, ((target_w - w) // 2, 0))
    return out


def ctc_read_batch(model, device, arrs, bs=32):
    out = []
    for i in range(0, len(arrs), bs):
        ts = torch.stack([tensor_from_resized(resize_line_image_fixed(
            _as_line(Image.fromarray(a)))) for a in arrs[i:i + bs]]).to(device)
        with torch.no_grad():
            out += decode_ctc(model(ts))
    return out


# ---------------------------------------------------------------------------
# CTC forced alignment (vectorised Viterbi over the blank-interleaved label)
# ---------------------------------------------------------------------------
def forced_align(log_probs, text):
    labels = [CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX]
    if not labels:
        return None
    ext = np.array([0] + sum(([l, 0] for l in labels), []), dtype=np.int64)
    S, T = len(ext), log_probs.shape[0]
    if T < len(labels):
        return None
    NEG = -1e30
    skip_ok = np.zeros(S, bool)
    for s in range(2, S):
        skip_ok[s] = ext[s] != 0 and ext[s] != ext[s - 2]
    dp = np.full((T, S), NEG)
    bp = np.zeros((T, S), np.int64)
    dp[0, 0] = log_probs[0, ext[0]]
    if S > 1:
        dp[0, 1] = log_probs[0, ext[1]]
    for t in range(1, T):
        prev = dp[t - 1]
        p1 = np.concatenate(([NEG], prev[:-1]))
        p2 = np.where(skip_ok, np.concatenate(([NEG, NEG], prev[:-2])), NEG)
        cand = np.stack([prev, p1, p2])
        best = np.argmax(cand, axis=0)
        dp[t] = cand[best, np.arange(S)] + log_probs[t, ext]
        bp[t] = best
    endS = S - 1 if (S == 1 or dp[T - 1, S - 1] >= dp[T - 1, S - 2]) else S - 2
    path = np.zeros(T, np.int64)
    s = endS
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= bp[t, s]
    kept = [c for c in text if c in CHAR_TO_IDX]
    spans = [((lambda ts: (int(ts[0]), int(ts[-1]) + 1) if len(ts) else None)
              (np.nonzero(path == 1 + 2 * ci)[0])) for ci in range(len(labels))]
    conf = float(np.mean(log_probs[np.arange(T), ext[path]]))
    return list(zip(kept, spans)), conf


def char_bounds(gray, text, model, device):
    rawW = gray.shape[1]
    with torch.no_grad():
        lp = model(tensor_from_resized(resize_line_image_fixed(
            Image.fromarray(gray))).unsqueeze(0).to(device))[:, 0, :].cpu().numpy()
    res = forced_align(lp, text)
    if res is None:
        return None, None
    align, conf = res
    T = lp.shape[0]
    ink = SP.BinarizeLine(gray)
    band = SP.CoreBand(ink)
    if band is None:
        return None, None
    top, base = band
    xh = float(base - top)
    if xh < 6:
        return None, None
    colProf = ink.sum(axis=0).astype(np.float64)
    k = max(3, int(round(xh * 0.12))) | 1
    colSm = np.convolve(colProf, np.ones(k) / k, mode="same")
    rows = np.arange(ink.shape[0])[:, None]
    hiY = np.where(ink, rows, ink.shape[0]).min(axis=0).astype(np.float64)
    topAbove = np.clip((base - hiY) / max(1.0, xh), 0.0, 3.0)
    pos = colSm[colSm > 0]
    thin = colSm / (np.percentile(pos, 60) if pos.size else 1.0)
    cutCost = thin + 1.15 * topAbove
    cutCost[colProf == 0] = 0.0
    snapR = max(3, int(round(xh * 0.5)))

    def snap(x):
        a = int(np.clip(round(x) - snapR, 0, rawW - 1))
        b = int(np.clip(round(x) + snapR + 1, 1, rawW))
        if b - a < 2:
            return float(x)
        off = np.abs(np.arange(a, b) - x) / max(1.0, snapR)
        return float(a + int(np.argmin(cutCost[a:b] + 0.6 * off)))

    # see TrainText.frame_x_to_pixel's docstring: resize_line_image_fixed
    # now fits-and-pads instead of stretching, so a canvas x-coordinate
    # (t/T*INPUT_WIDTH) no longer maps to original pixel x via a flat
    # "* rawW/T" -- it depends on how much of the canvas is real content.
    origH = gray.shape[0]
    xs = [(ch, None if sp is None else frame_x_to_pixel(sp[0] / T * INPUT_WIDTH, rawW, origH),
           None if sp is None else frame_x_to_pixel(sp[1] / T * INPUT_WIDTH, rawW, origH))
          for ch, sp in align]
    valid = [(i, c) for i, c in enumerate(xs) if c[1] is not None]
    if len(valid) < 2:
        return None, None
    centres = [0.5 * (c[1] + c[2]) for _, c in valid]
    out = []
    for k2, (i, (ch, x0, x1)) in enumerate(valid):
        c = centres[k2]
        left = x0 - 0.15 * xh if k2 == 0 else snap(0.5 * (centres[k2 - 1] + c))
        right = x1 + 0.15 * xh if k2 == len(valid) - 1 else snap(0.5 * (c + centres[k2 + 1]))
        out.append((ch, max(0.0, left), min(float(rawW), right)))
    return out, dict(xh=xh, base=float(base), slant=SP.EstimateSlantDeg(ink),
                     conf=conf, ink=ink)


def _edge_y(ink_slice, base, xh):
    def e(col):
        on = np.where(col)[0]
        return (base - on.mean()) / xh if len(on) else 0.5
    if ink_slice.shape[1] < 3:
        return 0.5, 0.5
    return e(ink_slice[:, :2].any(1)), e(ink_slice[:, -2:].any(1))


# ---------------------------------------------------------------------------
# ink crop -> skeleton trajectory (thinnest centreline, one pen width)
# ---------------------------------------------------------------------------
def _hole_px(mask):
    """Enclosed-background pixel count (RawImageOps.FillHoles, the same
    flood-fill hole-fill your own page-mask notes describe) -- used to check
    whether a letter's bowl/loop is still enclosed after thinning."""
    return int((F.FillHoles(mask) & ~mask).sum())


def _crop_to_polys(ink_crop, xh_px, n_chars):
    """Skeletonise + trace + Douglas-Peucker this ink crop. Returns a list
    of polylines in the CROP's OWN native pixel frame (same frame `ink_crop`
    is in), or None if the shape didn't survive thinning."""
    check_holes = n_chars <= _HOLE_CHECK_MAX_CHARS
    up = _UPSCALE if check_holes else 1
    big = np.kron(ink_crop, np.ones((up, up), dtype=ink_crop.dtype)) if up > 1 else ink_crop

    if check_holes:
        holeBefore = _hole_px(big)
    skel = SP.Skeletonize(big)
    if not skel.any():
        return None
    if check_holes and holeBefore > (0.12 * xh_px * up) ** 2:
        ring = F.Dilate(skel, 3, 3)
        if _hole_px(ring) < 0.35 * holeBefore:
            return None                          # the loop collapsed -- reject

    raw = SP.TracePolylines(skel)
    out = []
    for p in raw:
        if len(p) < 2:
            continue
        sp = SP.SimplifyPolyline(p, eps=max(0.6, 0.03 * xh_px * up))
        if len(sp) >= 2:
            out.append([(x / up, y / up) for x, y in sp])
    if not out:
        return None

    total_len = sum(float(np.hypot(*np.diff(np.asarray(s), axis=0).T).sum())
                    for s in out if len(s) > 1)
    xs = [x for s in out for x, _ in s]
    ys = [y for s in out for _, y in s]
    diag = float(np.hypot(max(xs) - min(xs), max(ys) - min(ys))) if xs else 0.0
    if total_len < 0.9 * diag:
        return None                              # collapsed some other way

    if len(out) > 1:                             # drop boundary-cut slivers
        lens = [float(np.hypot(*np.diff(np.asarray(s), axis=0).T).sum())
                if len(s) > 1 else 0.0 for s in out]
        main = max(range(len(out)), key=lambda i: lens[i])
        mx0 = min(pt[0] for pt in out[main]); mx1 = max(pt[0] for pt in out[main])
        kept = []
        for i, s in enumerate(out):
            if i == main or lens[i] > 0.25 * max(lens):
                kept.append(s); continue
            sx0 = min(pt[0] for pt in s); sx1 = max(pt[0] for pt in s)
            pad = 0.35 * max(1e-6, mx1 - mx0) / max(1, len(out))
            if sx1 >= mx0 - pad and sx0 <= mx1 + pad:
                kept.append(s)
        out = kept
    return out


_FONT_PATHS = [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
_font_cache, _anchor_cache = {}, {}


def _load_font(px):
    if px not in _font_cache:
        f = None
        for p in _FONT_PATHS:
            try:
                f = ImageFont.truetype(p, px); break
            except OSError:
                continue
        _font_cache[px] = f or ImageFont.load_default()
    return _font_cache[px]


def _anchor_ref_xh(px):
    """x-height of the anchor font at this size, measured off a real
    x-height-only glyph -- NOT the full ascender/descender bounding box
    (which is what a naive per-glyph bbox would give you, and badly
    mis-scales/mis-places anything with a descender, like 'j' or 'g')."""
    key = ("xh", px)
    if key not in _anchor_cache:
        font = _load_font(px)
        b = ImageDraw.Draw(Image.new("L", (4, 4))).textbbox((0, 0), "x", font=font)
        _anchor_cache[key] = float(b[3] - b[1]) or px * 0.5
    return _anchor_cache[key]


def _anchor_fragment(ch, px=96):
    """Plain single-stroke fallback for a character these 3 pages never
    gave us cleanly (numerals, punctuation, a rare letter) -- kept self
    contained to Method 1 (no Method-2/profile dependency) so every word is
    still spelled correctly. Thinned the same way, so it's the same
    constant pen width as every real fragment, just a different hand for
    that one glyph. Baseline comes from the font's own ascent metric, and
    scale from a shared x-height reference, so descenders ('j','g','y')
    and ascenders ('h','l','k') land at the right size and height instead
    of being measured off their own (much taller) bounding box."""
    if ch in _anchor_cache:
        return _anchor_cache[ch]
    font = _load_font(px)
    ascent, _descent = font.getmetrics()
    pad = px
    probe = ImageDraw.Draw(Image.new("L", (4, 4)))
    bbox = probe.textbbox((0, 0), ch, font=font)
    img = Image.new("L", (bbox[2] - bbox[0] + 2 * pad, bbox[3] - bbox[1] + 2 * pad), 255)
    ImageDraw.Draw(img).text((pad - bbox[0], pad), ch, fill=0, font=font)
    baseline_row = pad + ascent               # row of the baseline, pre-crop
    # NOTE: Image.getbbox() treats 0 (black) as background, so it's useless
    # on a black-ink-on-white canvas -- it would return the whole image.
    # Find the tight ink box in numpy instead.
    arr = np.array(img)
    ys, xs = np.where(arr < 128)
    result = None
    if len(ys):
        crop_top = int(ys.min())
        ink = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1] < 128
        polys = _crop_to_polys(ink, xh_px=_anchor_ref_xh(px), n_chars=1)
        if polys:
            xs = [x for p in polys for x, _ in p]
            xh_ref = _anchor_ref_xh(px)
            result = dict(polys=polys, xh=xh_ref, w_px=float(max(xs) - min(xs) + 2),
                         base_off=(baseline_row - crop_top) / xh_ref,
                         entry_y=0.0, exit_y=0.0, conf=0.0)
    _anchor_cache[ch] = result
    return result


# ---------------------------------------------------------------------------
# load pages -> (crop, transcript) pairs
# ---------------------------------------------------------------------------
def _sim(a, b):
    if not a and not b:
        return 1.0
    return 1.0 - levenshtein(a.lower(), b.lower()) / max(1, len(a), len(b))


def collect_pairs(model, device, verbose=True):
    """SegmentPage every page, CTC-read each crop, monotonically align the
    read strings to the transcript lines -> list of (gray_crop, text)."""
    pairs = []
    for page in sorted(PAGES_DIR.glob("page*.jpg")):
        results, _, _ = SEG.ProcessPage(str(page))
        crops = [r["raw_crop"] for r in results if r["tag"] == "TEXT"]
        gt = [g for g in ReadLabelLines(str(page)) if g.strip() and g.strip() != "MESS"]
        if not crops or not gt:
            continue
        reads = ctc_read_batch(model, device, crops)
        # monotonic DP alignment: match crops to gt lines allowing skips
        C, G = len(reads), len(gt)
        dp = np.full((C + 1, G + 1), -1e9)
        dp[0, 0] = 0
        bt = np.zeros((C + 1, G + 1), np.int8)
        SKIP = -0.35
        for i in range(C + 1):
            for j in range(G + 1):
                if i < C and dp[i, j] + SKIP > dp[i + 1, j]:
                    dp[i + 1, j] = dp[i, j] + SKIP; bt[i + 1, j] = 1        # skip crop
                if j < G and dp[i, j] + SKIP > dp[i, j + 1]:
                    dp[i, j + 1] = dp[i, j] + SKIP; bt[i, j + 1] = 2        # skip gt
                if i < C and j < G:
                    v = dp[i, j] + _sim(reads[i], gt[j])
                    if v > dp[i + 1, j + 1]:
                        dp[i + 1, j + 1] = v; bt[i + 1, j + 1] = 3          # match
        i, j, matched = C, G, []
        while i > 0 or j > 0:
            m = bt[i, j]
            if m == 3:
                matched.append((i - 1, j - 1)); i -= 1; j -= 1
            elif m == 1:
                i -= 1
            else:
                j -= 1
        matched.reverse()
        kept = 0
        for ci, gj in matched:
            if _sim(reads[ci], gt[gj]) < 0.45:
                continue
            g = np.array(Image.fromarray(crops[ci]).convert("L"))
            pairs.append((g, gt[gj]))
            kept += 1
        if verbose:
            print(f"  {page.name}: {len(crops)} crops, {len(gt)} labels -> {kept} pairs",
                  flush=True)
    return pairs


def _shape_grid(polys, nx=8, ny=10):
    """Coarse occupancy grid of a fragment's strokes over its own bounding
    box -- a cheap shape descriptor for comparing instances of the SAME
    key against each other (same idea as BuildStyleProfile's consensus
    filter: 'removes glyphs contaminated by a neighbouring letter')."""
    xs = [x for p in polys for x, _ in p]
    ys = [y for p in polys for _, y in p]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    w, h = max(1e-6, x1 - x0), max(1e-6, y1 - y0)
    grid = np.zeros((ny, nx))
    for p in polys:
        pts = np.asarray(p, np.float64)
        for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
            ns = max(2, int(np.hypot(bx - ax, by - ay)) + 1)
            for t in np.linspace(0, 1, ns):
                px, py = ax + t * (bx - ax), ay + t * (by - ay)
                gx = min(nx - 1, max(0, int((px - x0) / w * nx)))
                gy = min(ny - 1, max(0, int((py - y0) / h * ny)))
                grid[gy, gx] += 1
    v = grid.flatten()
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _consensus_rerank(frags, min_n=3, rel_cutoff=0.85, min_keep=2):
    """Confidence alone (the source LINE's alignment score) doesn't predict
    which INSTANCE of a repeated key is best formed -- checked directly:
    the top-confidence 's' among 8 real instances was a malformed outlier
    while several lower-confidence ones were clean. Instead, for any key
    with enough samples, rank instances by how close their shape sits to
    the consensus (mean) shape of all of this writer's own instances of
    that same key -- an odd one out is probably a bad segmentation cut,
    not a new letterform. Confidence still breaks ties / orders the small
    pools where a consensus isn't reliable.

    Ranking alone wasn't enough: a stray sliver of a neighbouring letter
    bleeding into a crop (checked directly on the 'la' key -- e.g. an extra
    hook or stem in front of the 'l') can still land a similarity score
    close to the clean instances, high enough to stay inside the top-`pool`
    synthesis samples a plain sort never removes. So also DROP any instance
    whose similarity falls more than `rel_cutoff` below this key's own best
    match (a relative, not absolute, bar -- some keys are naturally more
    variable than others) -- but never below `min_keep`, so a key never
    empties out."""
    for kk, lst in frags.items():
        if len(lst) < min_n:
            continue
        grids = [_shape_grid(f["polys"]) for f in lst]
        proto = np.mean(grids, axis=0)
        pn = np.linalg.norm(proto)
        if pn > 1e-9:
            proto = proto / pn
        sims = [float(np.dot(g, proto)) for g in grids]
        order = sorted(range(len(lst)), key=lambda i: -sims[i])
        best = sims[order[0]]
        keep = [i for i in order if best <= 0 or sims[i] >= rel_cutoff * best]
        if len(keep) < min_keep:
            keep = order[:min_keep]
        frags[kk] = [lst[i] for i in keep]


# ---------------------------------------------------------------------------
# build fragment library
# ---------------------------------------------------------------------------
def build_library(verbose=True):
    device = torch.device("cpu")
    model = load_model(device)
    pairs = collect_pairs(model, device, verbose)
    if not pairs:
        raise RuntimeError("no aligned lines -- check segmentation / labels")

    # Line-level quality gate BEFORE cutting any fragments (drop the bottom
    # quarter by forced-alignment confidence). Verifying individual tiny
    # fragments afterwards by reading them back in isolation was tried and
    # doesn't work: a whole-LINE CTC model forced to read a 2-4 letter crop
    # stretched to its 640x64 input sees a completely different character
    # density than any real line it was trained on, so it misreads good
    # fragments as often as bad ones. Filtering at the line level instead
    # keeps the same reader doing the job it's actually good at.
    aligned = []
    for gray, text in pairs:
        bounds, meta = char_bounds(gray, text, model, device)
        if bounds is not None:
            aligned.append((gray, text, bounds, meta))
    if not aligned:
        raise RuntimeError("no alignable lines")
    conf_cut = np.percentile([m["conf"] for _, _, _, m in aligned], 25)
    kept = [p for p in aligned if p[3]["conf"] >= conf_cut]
    if verbose:
        print(f"  [align] {len(kept)}/{len(aligned)} lines kept "
              f"(bottom 25% by alignment confidence dropped)", flush=True)

    frags = defaultdict(list)
    xhs, slants, gaps = [], [], []
    for gray, text, bounds, meta in kept:
        xh, base, ink = meta["xh"], meta["base"], meta["ink"]
        xhs.append(xh); slants.append(meta["slant"])
        n = len(bounds)
        words, i = [], 0
        while i < n:
            if bounds[i][0] == " ":
                i += 1; continue
            j = i
            while j < n and bounds[j][0] != " ":
                j += 1
            words.append((i, j))
            if j < n:
                gp = bounds[j][2] - bounds[j - 1][2]
                if 0 < gp < 6 * xh:
                    gaps.append(gp / xh)
            i = j

        def add(a, b):
            wi = a == 0 or bounds[a - 1][0] == " "
            wf = b == n or bounds[b][0] == " "
            L = max(0, int(round(bounds[a][1])) - (int(0.4 * xh) if wi else 2))
            R = min(ink.shape[1], int(round(bounds[b - 1][2])) + (int(0.35 * xh) if wf else 2))
            if R - L < 4:
                return
            cr = ink[:, L:R]
            ys = np.where(cr.any(1))[0]
            if len(ys) < 3:
                return
            y0 = max(0, ys[0] - 2)
            cr = cr[y0:ys[-1] + 3]
            txt = "".join(bounds[c][0] for c in range(a, b))
            polys = _crop_to_polys(cr, xh, n_chars=len(txt))
            if polys is None:
                return
            ein, eout = _edge_y(ink[:, L:R], base, xh)
            # advance width = the ink's own right edge, NOT the padded crop
            # width -- a fragment that was originally extracted as a whole
            # WORD (wi and wf padding both applied, e.g. a lone "b") stores
            # a crop much wider than its ink; using cr.shape[1] as the next
            # fragment's start-x left a visible dead-air gap when that same
            # fragment got reused mid-word (confirmed: "brown" -> "b  rown").
            right_edge = max((x for p in polys for x, _ in p), default=cr.shape[1])
            frags[txt].append(dict(polys=polys, xh=xh, text=txt,
                                   w_px=float(right_edge + 2),
                                   base_off=(base - y0) / xh,
                                   entry_y=ein, exit_y=eout, conf=meta["conf"]))

        for ws, we in words:
            if we - ws >= 2:
                add(ws, we)
            for Ln in (1, 2, 3):
                for a in range(ws, we - Ln + 1):
                    add(a, a + Ln)

    for kk in list(frags):
        frags[kk].sort(key=lambda f: -f["conf"])
    _consensus_rerank(frags)
    for kk in list(frags):
        frags[kk] = frags[kk][:MAX_PER_KEY]

    lib = dict(xh=float(np.median(xhs)) if xhs else 32.0,
               slant=float(np.median(slants)) if slants else 0.0,
               word_adv=float(np.median(gaps)) if gaps else 1.3,
               frags=dict(frags), n_lines=len(kept))
    with open(LIB_PATH, "wb") as f:
        pickle.dump(lib, f)
    print(f"[lib] {lib['n_lines']} lines, {len(lib['frags'])} fragment keys, "
          f"xh={lib['xh']:.0f} slant={lib['slant']:.1f} -> {LIB_PATH.name}")
    return lib


# ---------------------------------------------------------------------------
# synthesis
# ---------------------------------------------------------------------------
def _tile(word, frags, rng=None, pool=3):
    """Greedy longest-match. With rng=None, always the best-ranked (highest
    consensus-shape) instance of a key, so the word looks the same every
    time. With an rng, samples among the top `pool` real instances of that
    key instead of always the single best one -- these are all genuine ink
    the writer actually produced (frags[key] is consensus-ranked, so the
    pool never reaches down into the malformed tail), just different
    physical instances of the same letters/words across the 3 pages. Falls
    back to the built-in single-stroke font anchor for a character these
    pages never gave us cleanly, instead of leaving a blank gap -- every
    word stays correctly spelled."""
    out, i = [], 0
    while i < len(word):
        for Ln in range(min(MAX_NGRAM + 6, len(word) - i), 0, -1):
            key = word[i:i + Ln]
            if key in frags and (Ln == 1 or len(frags[key]) >= MIN_NGRAM_INSTANCES):
                lst = frags[key]
                if rng is not None and len(lst) > 1:
                    choice = lst[rng.randrange(min(pool, len(lst)))]
                else:
                    choice = lst[0]
                out.append(("real", choice)); i += Ln; break
        else:
            anchor = _anchor_fragment(word[i])
            out.append(("anchor", anchor) if anchor else ("skip", None))
            i += 1
    return out


def _seam_points(x0, y0, x1, y1, n=24):
    """Sampled points of a short quadratic sag from (x0,y0) to (x1,y1) --
    drawn with the same draw.line() call as every real stroke, so the seam
    is the same constant pen width, not a separate pixel-stamped blob."""
    ts = np.linspace(0, 1, max(2, n))
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1) + 0.12 * (x1 - x0)
    xs = (1 - ts) ** 2 * x0 + 2 * (1 - ts) * ts * cx + ts ** 2 * x1
    ys = (1 - ts) ** 2 * y0 + 2 * (1 - ts) * ts * cy + ts ** 2 * y1
    return list(zip(xs.tolist(), ys.tolist()))


def synthesize(text, lib=None, seed=0, xh=40, vary=False):
    """Every stroke drawn with the same draw.line(..., width=ink_w) call, so
    the whole render is ONE constant pen width -- real fragments,
    font-anchor fallbacks, and seam stitches alike. With vary=False
    (default) fragment choice is deterministic (always the best-ranked
    instance); with vary=True, `seed` also picks among the top few real
    instances of each repeated letter/word (see _tile), so different seeds
    give genuinely different renders built from the same 3 pages."""
    if lib is None:
        with open(LIB_PATH, "rb") as f:
            lib = pickle.load(f)
    rng = random.Random(seed)          # baseline wobble + (if vary) fragment choice
    tile_rng = rng if vary else None
    frags = lib["frags"]
    wg = int(np.clip(lib["word_adv"] * xh * 0.75, 0.45 * xh, 1.6 * xh))
    ink_w = int(np.clip(round(0.09 * xh), 2, 5))
    words = text.split()
    W = 48 + int(sum(len(w) for w in words) * xh * 1.2) + wg * len(words) + 300
    H = int(xh * 5)
    canvas = Image.new("L", (W, H), 255)
    draw = ImageDraw.Draw(canvas)
    bl, x = H // 2, 24
    n_anchor = n_skip = 0
    for word in words:
        prev = None
        for kind, fr in _tile(word, frags, rng=tile_rng):
            if kind == "skip":
                x += int(xh * 0.55); prev = None; n_skip += 1; continue
            n_anchor += kind == "anchor"
            s = xh / max(4.0, fr["xh"])
            y0 = bl - fr["base_off"] * xh + rng.uniform(-0.04, 0.04) * xh
            if prev is not None:
                ex, ey = prev
                ny = bl - fr["entry_y"] * xh
                if abs(ey - ny) > 0.16 * xh or (x - ex) > 0.10 * xh:
                    draw.line(_seam_points(ex, ey, x, ny), fill=0, width=ink_w)
            for poly in fr["polys"]:
                pts = [(x + px * s, y0 + py * s) for px, py in poly]
                if len(pts) >= 2:
                    draw.line(pts, fill=0, width=ink_w, joint="curve")
                else:
                    draw.ellipse([pts[0][0] - ink_w / 2, pts[0][1] - ink_w / 2,
                                 pts[0][0] + ink_w / 2, pts[0][1] + ink_w / 2], fill=0)
            x += fr["w_px"] * s
            prev = (x, bl - fr["exit_y"] * xh)
        x += wg
    cv = np.array(canvas)
    m = cv < 245
    if m.any():
        ys, xs = np.where(m)
        cv = cv[max(0, ys.min() - 8):ys.max() + 8, max(0, xs.min() - 8):xs.max() + 8]
    return Image.fromarray(cv).convert("L"), dict(n_anchor=n_anchor, n_skip=n_skip)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", default="the quick brown fox jumps over the lazy dog")
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()

    lib = build_library() if (args.build or not LIB_PATH.exists()) else \
        pickle.load(open(LIB_PATH, "rb"))
    im, stats = synthesize(args.text, lib)
    out = OUT_DIR / "method1_sample.png"
    im.save(out)
    print(f"\n  '{args.text}'\n  -> {out}  {im.size}  "
          f"({stats['n_anchor']} chars from the font-anchor fallback, "
          f"{stats['n_skip']} skipped)")


if __name__ == "__main__":
    main()
