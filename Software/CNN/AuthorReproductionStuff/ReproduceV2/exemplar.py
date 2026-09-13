"""
exemplar.py -- FIRST-PRINCIPLES handwriting reproduction by fragment collage.

The glyph-library synthesiser failed on cursive because it re-joined isolated
letters. Instead: copy fragments the author ACTUALLY wrote -- whole words,
then letter n-grams, then single letters -- so real ligatures survive, and
stitch them with short seam strokes only where a join falls.

  build_library(author non-holdout lines):
    CTC forced-align each transcript -> per-character x boundaries (snapped to
    ligature crossings) -> cut CLEANED image crops for every word and every
    1..3-char sub-run, tagged with baseline/x-height/ink entry+exit height.
    Long fragments are CTC-verified and dropped if unreadable.

  synthesize_line(text, library):
    greedy longest-match tiling of the target from library fragments ->
    scale each to a common x-height, lay on one baseline, seam-stroke where
    ink exit/entry heights disagree -> render.

No training, no GPU, every stroke is the author's own ink.
"""

import pickle
import random
import time
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

import _env  # noqa: F401
from _env import NOGIT_DIR, DATA_DIR, TEXT_WEIGHTS
from TrainText import (PaperCRNN, CHARSET, CHAR_TO_IDX, decode_ctc, levenshtein,
                       resize_line_image_fixed, tensor_from_resized, _decode_png,
                       IAMLineDatasetRaw)
import BuildStyleProfile as SP

LIB_DIR = NOGIT_DIR / "ReproduceV2" / "exemplar_lib"
MAX_NGRAM = 3
MAX_PER_KEY = 6
MAX_LINES = 130


# ---------------------------------------------------------------------------
# fast CTC forced alignment (vectorised Viterbi; own copy so BuildStyleProfile
# stays untouched)
# ---------------------------------------------------------------------------
def _forced_align(log_probs, text):
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
        p2 = np.concatenate(([NEG, NEG], prev[:-2]))
        p2 = np.where(skip_ok, p2, NEG)
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
    spans = []
    for ci in range(len(labels)):
        ts = np.nonzero(path == 1 + 2 * ci)[0]
        spans.append((int(ts[0]), int(ts[-1]) + 1) if len(ts) else None)
    conf = float(np.mean(log_probs[np.arange(T), ext[path]]))
    return list(zip(kept, spans)), conf


def _char_bounds(gray, text, model, device):
    rawW = gray.shape[1]
    with torch.no_grad():
        lp = model(tensor_from_resized(resize_line_image_fixed(
            Image.fromarray(gray))).unsqueeze(0).to(device))[:, 0, :].cpu().numpy()
    res = _forced_align(lp, text)
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

    xs = [(ch, None if sp is None else sp[0] / T * rawW,
           None if sp is None else sp[1] / T * rawW) for ch, sp in align]
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


def _read_batch(model, device, arrs, bs=64):
    out = []
    for i in range(0, len(arrs), bs):
        ts = torch.stack([tensor_from_resized(resize_line_image_fixed(
            Image.fromarray(a).convert("L"))) for a in arrs[i:i + bs]]).to(device)
        with torch.no_grad():
            out += decode_ctc(model(ts))
    return out


def _edge_y(ink_slice, base, xh):
    def e(col):
        on = np.where(col)[0]
        return (base - on.mean()) / xh if len(on) else 0.5
    if ink_slice.shape[1] < 3:
        return 0.5, 0.5
    return e(ink_slice[:, :2].any(1)), e(ink_slice[:, -2:].any(1))


# ---------------------------------------------------------------------------
def build_library(author_id, model, device):
    t0 = time.time()
    ds = IAMLineDatasetRaw(root_dir=str(DATA_DIR), cache_dir=str(SP.CACHE_DIR))
    lines = [s for s in ds.samples
             if s["page_key"].split("/")[0] == author_id and not s["is_holdout"]]
    random.Random(0).shuffle(lines)
    lines = lines[:MAX_LINES]

    ta = time.time()
    parsed = []
    for s in lines:
        g = np.array(_decode_png(s["image_png"]).convert("L"))
        b, m = _char_bounds(g, s["text"], model, device)
        if b is not None:
            parsed.append((b, m))
    if not parsed:
        raise RuntimeError(f"no alignable lines for {author_id}")
    print(f"  [align] {len(parsed)} lines in {time.time()-ta:.0f}s", flush=True)
    cc = np.percentile([m["conf"] for _, m in parsed], 30)
    parsed = [p for p in parsed if p[1]["conf"] >= cc]

    frags = defaultdict(list)
    xhs, slants, gaps = [], [], []
    for bounds, meta in parsed:
        xh, base, ink = meta["xh"], meta["base"], meta["ink"]
        xhs.append(xh); slants.append(meta["slant"])
        clean = 255 - ink.astype(np.uint8) * 255
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
            R = min(clean.shape[1], int(round(bounds[b - 1][2])) + (int(0.35 * xh) if wf else 2))
            if R - L < 4:
                return
            cr = clean[:, L:R]
            ys = np.where((cr < 128).any(1))[0]
            if len(ys) < 3:
                return
            y0 = max(0, ys[0] - 2)
            cr = cr[y0:ys[-1] + 3]
            ein, eout = _edge_y(ink[:, L:R], base, xh)
            txt = "".join(bounds[c][0] for c in range(a, b))
            frags[txt].append(dict(img=cr, xh=xh, text=txt,
                                   base_off=(base - y0) / xh, ink_h=len(ys),
                                   entry_y=ein, exit_y=eout, conf=meta["conf"]))

        for ws, we in words:
            if we - ws >= 2:
                add(ws, we)
            for Ln in (1, 2, 3):
                for a in range(ws, we - Ln + 1):
                    add(a, a + Ln)

    for kk in list(frags):
        frags[kk].sort(key=lambda f: -f["conf"])
        frags[kk] = frags[kk][:MAX_PER_KEY + 3]

    # verify only fragments >= 4 chars (whole words, mostly) -- these are the
    # ones a bad cut hurts most; 2/3-grams are cheap filler and trusted.
    tv = time.time()
    todo = [(kk, i) for kk in frags if len(kk) >= 4 for i in range(len(frags[kk]))]
    if todo:
        preds = _read_batch(model, device, [frags[kk][i]["img"] for kk, i in todo], bs=32)
        keep = defaultdict(list)
        for (kk, i), pr in zip(todo, preds):
            if levenshtein(pr.lower(), kk.lower()) <= (1 if len(kk) <= 5 else 2):
                keep[kk].append(frags[kk][i])
        for kk in list(frags):
            if len(kk) >= 4:
                frags[kk] = keep[kk][:MAX_PER_KEY] or frags[kk][:1]
    print(f"  [verify] {len(todo)} frags in {time.time()-tv:.0f}s", flush=True)
    for kk in list(frags):
        frags[kk] = frags[kk][:MAX_PER_KEY]

    return dict(author=author_id, xh=float(np.median(xhs)),
                slant=float(np.median(slants)),
                word_adv=float(np.median(gaps)) if gaps else 1.3,
                frags=dict(frags), n_lines=len(parsed),
                build_s=round(time.time() - t0, 1))


def library_for(author_id, model, device, rebuild=False):
    LIB_DIR.mkdir(parents=True, exist_ok=True)
    p = LIB_DIR / f"{author_id}.pkl"
    if p.exists() and not rebuild:
        with open(p, "rb") as f:
            return pickle.load(f)
    lib = build_library(author_id, model, device)
    with open(p, "wb") as f:
        pickle.dump(lib, f)
    return lib


# ---------------------------------------------------------------------------
# synthesis
# ---------------------------------------------------------------------------
def _lookup(key, frags):
    """exact key, else a case-folded variant (cap written as its lowercase,
    or vice versa) so a rare word-initial capital still gets real ink."""
    if key in frags:
        return frags[key], 1.0
    for alt, sc in ((key.lower(), 1.0), (key.capitalize(), 1.0),
                    (key.upper(), 1.0)):
        if alt != key and alt in frags:
            return frags[alt], (1.35 if key[:1].isupper() and alt[:1].islower() else 1.0)
    return None, 1.0


def _tile(word, frags, rng):
    """-> list of (frag_or_None, cap_scale). None = no ink, use archetype."""
    out, i = [], 0
    while i < len(word):
        hit = False
        for L in range(min(MAX_NGRAM + 5, len(word) - i), 0, -1):
            cands, sc = _lookup(word[i:i + L], frags)
            if cands:
                out.append((rng.choice(cands), sc))
                i += L
                hit = True
                break
        if not hit:
            out.append((None, 1.0))
            i += 1
    return out


def synthesize_line(text, lib, seed=0, xh=34, pad=24, missing=None):
    """missing: optional list, appended with every char that had no fragment."""
    rng = random.Random(seed)
    frags = lib["frags"]
    slant = lib["slant"]
    wg = int(np.clip(lib["word_adv"] * xh * 0.75, 0.45 * xh, 1.6 * xh))
    ink_w = int(np.clip(round(0.085 * xh), 2, 5))

    words = text.split()
    W = pad * 2 + int(sum(len(w) for w in words) * xh * 1.15) + wg * len(words) + 300
    H = int(xh * 5)
    cv = np.full((H, W), 255, np.uint8)
    bl = H // 2
    x = pad
    for wi, word in enumerate(words):
        prev_exit = None
        for ci, (fr, cap) in enumerate(_tile(word, frags, rng)):
            if fr is None:
                mc = word[ci] if ci < len(word) else "?"
                if missing is not None:
                    missing.append(mc)
                im = _arch_char(mc, xh, slant)
                base_off_xh = 0.78
                src_xh = xh
            else:
                im = fr["img"]
                src_xh = max(4.0, fr["xh"]) / cap
                base_off_xh = fr["base_off"]
            s = xh / src_xh
            s = min(s, xh * 2.6 / max(1, im.shape[0]))
            rim = np.array(Image.fromarray(im).resize(
                (max(1, int(im.shape[1] * s)), max(1, int(im.shape[0] * s))),
                Image.Resampling.LANCZOS))
            h, w = rim.shape
            y0 = int(bl - base_off_xh * xh + rng.uniform(-0.04, 0.04) * xh)
            if fr is not None and prev_exit is not None:
                ex, ey = prev_exit
                ny = bl - fr["entry_y"] * xh
                if abs(ey - ny) > 0.16 * xh or (x - ex) > 0.10 * xh:
                    _seam(cv, ex, ey, x, ny, ink_w)
            y1, x1 = min(H, y0 + h), min(W, x + w)
            if x1 > x and y1 > max(0, y0):
                yy0 = max(0, y0)
                reg = cv[yy0:y1, x:x1]
                np.minimum(reg, rim[yy0 - y0:y1 - y0, :x1 - x], out=reg)
            x = x1
            prev_exit = (x, bl - (fr["exit_y"] if fr is not None else 0.5) * xh)
        x += wg

    m = cv < 245
    if m.any():
        ys, xs = np.where(m)
        cv = cv[max(0, ys.min() - 8):ys.max() + 8, max(0, xs.min() - 8):xs.max() + 8]
    return Image.fromarray(cv).convert("L")


_ARCH_FONT = None


def _arch_char(ch, xh, slant):
    """last-resort ink for a character the author never wrote: a plain glyph,
    sheared to the author's slant. white bg / black ink array."""
    global _ARCH_FONT
    from PIL import ImageDraw, ImageFont
    if _ARCH_FONT is None:
        for p in (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\calibri.ttf",
                  "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
            try:
                _ARCH_FONT = ImageFont.truetype(p, 128)
                break
            except OSError:
                pass
        if _ARCH_FONT is None:
            _ARCH_FONT = ImageFont.load_default()
    pad = 140
    im = Image.new("L", (pad * 2, pad * 2), 255)
    ImageDraw.Draw(im).text((pad, pad), ch, fill=0, font=_ARCH_FONT)
    im = im.crop(im.getbbox() or (0, 0, 10, 10))
    target_h = int(xh * (1.5 if ch.isupper() or ch in "bdfhklt" else 1.0))
    s = target_h / max(1, im.height)
    im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))),
                   Image.Resampling.LANCZOS)
    k = np.tan(np.radians(np.clip(slant, -18, 18)))
    if abs(k) > 0.05:
        w, h = im.size
        im = im.transform((w + int(abs(k) * h), h), Image.Transform.AFFINE,
                          (1, -k, 0 if k < 0 else k * h, 0, 1, 0),
                          resample=Image.Resampling.BILINEAR, fillcolor=255)
    return np.array(im)


def _seam(cv, x0, y0, x1, y1, wd):
    x0, x1 = int(x0), int(x1)
    if x1 <= x0:
        return
    ts = np.linspace(0, 1, max(2, x1 - x0))
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1) + 0.12 * (x1 - x0)
    xs = (1 - ts) ** 2 * x0 + 2 * (1 - ts) * ts * cx + ts ** 2 * x1
    ys = (1 - ts) ** 2 * y0 + 2 * (1 - ts) * ts * cy + ts ** 2 * y1
    r = max(1, wd // 2)
    H, W = cv.shape
    for xx, yy in zip(xs, ys):
        xi, yi = int(round(xx)), int(round(yy))
        cv[max(0, yi - r):min(H, yi + r + 1), max(0, xi - r):min(W, xi + r + 1)] = 0


def _read_line(model, device, pil):
    t = tensor_from_resized(resize_line_image_fixed(pil.convert("L"))).unsqueeze(0).to(device)
    with torch.no_grad():
        return decode_ctc(model(t))[0]


def synthesize_best(text, lib, model, device, k=6, seed=0, xh=34):
    """best-of-k assemblies, ranked by how well the frozen reader reads the
    result back (the SynthesizeLegible pattern)."""
    cands = []
    for i in range(k):
        im = synthesize_line(text, lib, seed=seed + i, xh=xh)
        pred = _read_line(model, device, im)
        cer = levenshtein(pred.lower(), text.lower()) / max(1, len(text))
        cands.append((cer, im, pred))
    cands.sort(key=lambda c: c[0])
    return cands[0][1], cands[0][2], cands[0][0]


def load_text_model(device):
    m = PaperCRNN(num_classes=len(CHARSET) + 1).to(device)
    m.load_state_dict(torch.load(TEXT_WEIGHTS, map_location=device, weights_only=False))
    m.eval()
    return m


if __name__ == "__main__":
    import sys
    dev = torch.device("cpu")
    tm = load_text_model(dev)
    text = sys.argv[1] if len(sys.argv) > 1 else "the quick brown fox"
    author = sys.argv[2] if len(sys.argv) > 2 else "150"
    lib = library_for(author, tm, dev, rebuild="--rebuild" in sys.argv)
    print(f"[lib {author}] {lib['n_lines']} lines, {len(lib['frags'])} keys, "
          f"xh={lib['xh']:.0f} slant={lib['slant']:.1f} build={lib.get('build_s','?')}s")
    im = synthesize_line(text, lib, seed=0)
    out = _env.OUT_DIR / f"_exemplar_{author}.png"
    im.save(out)
    print("saved", out, im.size)
