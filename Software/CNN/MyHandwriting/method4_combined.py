"""
METHOD 4 -- combined trajectory library (Method 1's real strokes, backed up
by Method 2's per-letter model), rendered/exported at ONE constant pen width.

Why: Method 1 (raw pixel collage) carries the writer's actual textural
detail but has gaps -- letters/words they never wrote. Method 2 (denoised
skeleton + measured style) covers every letter but is a step removed from
the writer's literal ink. Method 4 gets both without their downsides:

  * every fragment -- whichever method it comes from -- is stored as a
    SKELETON TRAJECTORY (thinnest centreline of the stroke), not a pixel
    image. One pen width, drawn by `RenderTrajectory(uniformInk=True)` and
    exported straight to `WriteGCode.py` -- no variable-thickness ink left
    anywhere.
  * the primary source for any word/n-gram the writer actually wrote is
    Method 1's own skeletonised ink (`fragment_trajectories`, below) --
    their real letterforms and real joins.
  * anything missing (a letter they never wrote, a bad extraction) falls
    back to Method 2's already-built per-letter model (`method2_features`),
    which is itself a skeleton trajectory -- so the switch is invisible in
    the output: everything on the page is a centreline stroke either way.

RUNTIME COST: the CNN-BiLSTM-CTC classifier is used ONLY while building the
two libraries (forced alignment needs it to find character boundaries).
`synthesize()` never calls it -- pure dictionary lookup + geometry, so it is
fast and matches the proposal's split between FU2 (classification, trained
once) and FU3 (reproduction, reads the stored trajectory maps only).

    python method4_combined.py --build
    python method4_combined.py "the quick brown fox jumps over the lazy dog"
    python method4_combined.py "some text" --gcode          # also emit .gcode
"""
import argparse
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

import _env
from _env import OUT_DIR, LIB_DIR

from method1_exemplar import load_model, collect_pairs, char_bounds
import method2_features as M2
import BuildStyleProfile as SP
import RawImageOps as F
import SynthesizeHandwriting as SY

MAX_NGRAM = 3
MAX_PER_KEY = 6
LIB_PATH = LIB_DIR / "method4_fragments.pkl"
MM_PER_XH = 4.0                      # physical size of one x-height, mm


# ---------------------------------------------------------------------------
# fragment -> SKELETON TRAJECTORY (this is the "get the thinnest points" ask)
# ---------------------------------------------------------------------------
_UPSCALE = 4          # thinning a small letter (e.g. a fine-pen 'o', ~20px
                      # tall) at native resolution erodes the ring to a dot;
                      # upscale first (nearest-neighbour, so the binary edge
                      # stays crisp) so the stroke survives Zhang-Suen, then
                      # divide the traced coordinates back down.
_HOLE_CHECK_MAX_CHARS = 3   # the loop-collapse failure is a SINGLE-LETTER
                      # phenomenon ('o','a','e' eroding away). Upscaling +
                      # flood-fill-checking whole multi-word crops is the
                      # same fix applied where it can't be the problem, at
                      # far higher cost (Zhang-Suen and the hole flood-fill
                      # both scale with pixel count) -- so it's gated to
                      # short fragments only. Longer fragments are still
                      # protected, just by the cheaper checks below (length
                      # vs. bounding box, and the CTC read-back in the
                      # caller) rather than by the hole test.


def _hole_px(mask):
    """Enclosed-background pixel count (RawImageOps.FillHoles -- the same
    flood-fill hole-fill your own notes describe for page masks, reused here
    to check whether a letter's bowl/loop is still enclosed)."""
    return int((F.FillHoles(mask) & ~mask).sum())


def _crop_to_strokes_mm(ink_crop, xh_px, mm_per_xh, base_row_in_crop, n_chars=1):
    """Skeletonise + trace + Douglas-Peucker one ink crop, and convert every
    point from pixels to millimetres, y-up, baseline at y=0 -- the exact
    convention SynthesizeHandwriting.Trajectory already uses, so Method-1
    and Method-2 fragments become interchangeable."""
    check_holes = n_chars <= _HOLE_CHECK_MAX_CHARS
    up = _UPSCALE if check_holes else 1
    big = np.kron(ink_crop, np.ones((up, up), dtype=ink_crop.dtype)) if up > 1 else ink_crop

    if check_holes:
        holeBefore = _hole_px(big)

    skel = SP.Skeletonize(big)
    if not skel.any():
        return None

    if check_holes and holeBefore > (0.12 * xh_px * up) ** 2:
        # a bowl/loop ('o','a','e','d','b'...) must still be enclosed once
        # thinned to a ring -- a 1px-wide skeleton needs dilating back out
        # a little before the same hole test means anything
        ring = F.Dilate(skel, 3, 3)
        if _hole_px(ring) < 0.35 * holeBefore:
            return None                    # the loop collapsed -- reject

    polys = SP.TracePolylines(skel)
    scale = mm_per_xh / max(4.0, xh_px) / up
    out = []
    for p in polys:
        if len(p) < 2:
            continue
        sp = SP.SimplifyPolyline(p, eps=max(0.6, 0.03 * xh_px * up))
        if len(sp) < 2:
            continue
        out.append([(x * scale, (base_row_in_crop * up - y) * scale) for x, y in sp])
    if not out:
        return None
    # secondary net: a fragment whose skeleton is implausibly short for its
    # own bounding box collapsed some other way (not just a loop)
    total_len = sum(float(np.hypot(*np.diff(np.asarray(s), axis=0).T).sum())
                    for s in out if len(s) > 1)
    xs = [x for s in out for x, _ in s]
    ys = [y for s in out for _, y in s]
    diag = float(np.hypot(max(xs) - min(xs), max(ys) - min(ys))) if xs else 0.0
    if total_len < 0.9 * diag:
        return None
    # drop micro-strokes that sit off to one side, away from the main body
    # -- a boundary-cut sliver from the neighbouring letter, not real ink
    # (a genuine dot/accent sits close above its own letter, not off to a side)
    if len(out) > 1:
        lens = [float(np.hypot(*np.diff(np.asarray(s), axis=0).T).sum()) if len(s) > 1 else 0.0
                for s in out]
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
    out.sort(key=lambda s: min(pt[0] for pt in s))
    return out


def _stroke_bounds(strokes):
    xs = [x for s in strokes for x, _ in s]
    return min(xs), max(xs)


# ---------------------------------------------------------------------------
# build: Method-1-style word/n-gram fragments, stored as trajectories
# ---------------------------------------------------------------------------
def build_fragment_trajectories(verbose=True):
    device = torch.device("cpu")
    model = load_model(device)
    pairs = collect_pairs(model, device, verbose)
    if not pairs:
        raise RuntimeError("no aligned lines")

    # Line-level quality gate BEFORE cutting any fragments (this is the
    # proven filter, same as method1_exemplar/BuildStyleProfile): a line
    # the forced-aligner wasn't confident about has untrustworthy character
    # boundaries, so every fragment cut from it is suspect. Verifying each
    # tiny fragment afterwards by reading it back in isolation was tried and
    # doesn't work -- a whole-LINE CTC model forced to read a 2-3 letter
    # crop stretched to its 640x64 input has a totally different character
    # density than any real line it was trained on, so it misreads good
    # fragments as often as bad ones (confirmed: "in", cut correctly, read
    # back as garbage in >90% of instances). Filtering at the line level
    # instead keeps that same reader doing the job it's actually good at.
    parsed = []
    for gray, text in pairs:
        bounds, meta = char_bounds(gray, text, model, device)
        if bounds is not None:
            parsed.append((gray, text, bounds, meta))
    if not parsed:
        raise RuntimeError("no alignable lines")
    conf_cut = np.percentile([m["conf"] for _, _, _, m in parsed], 25)
    kept = [p for p in parsed if p[3]["conf"] >= conf_cut]
    if verbose:
        print(f"  [align] {len(kept)}/{len(parsed)} lines kept "
              f"(bottom 25% by alignment confidence dropped)", flush=True)

    frags = defaultdict(list)
    gaps = []
    for gray, text, bounds, meta in kept:
        xh, base, ink = meta["xh"], meta["base"], meta["ink"]
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
            crop = ink[:, L:R]
            ys = np.where(crop.any(1))[0]
            if len(ys) < 3:
                return
            y0 = max(0, ys[0] - 2)
            crop = crop[y0:ys[-1] + 3]
            txt = "".join(bounds[c][0] for c in range(a, b))
            strokes = _crop_to_strokes_mm(crop, xh, MM_PER_XH, base_row_in_crop=base - y0,
                                          n_chars=len(txt))
            if strokes is None:
                return
            x0, x1 = _stroke_bounds(strokes)
            strokes = [[(x - x0, y) for x, y in s] for s in strokes]
            frags[txt].append(dict(
                strokes=strokes, width_mm=x1 - x0,
                entry=strokes[0][0], exit=strokes[-1][-1], conf=meta["conf"]))

        for ws, we in words:
            if we - ws >= 2:
                add(ws, we)
            for Ln in (1, 2, 3):
                for a in range(ws, we - Ln + 1):
                    add(a, a + Ln)

    for kk in list(frags):
        frags[kk].sort(key=lambda f: -f["conf"])
        frags[kk] = frags[kk][:MAX_PER_KEY]
        if not frags[kk]:
            del frags[kk]

    lib = dict(frags=dict(frags), word_gap_xh=float(np.median(gaps)) if gaps else 1.3,
               mm_per_xh=MM_PER_XH)
    with open(LIB_PATH, "wb") as f:
        pickle.dump(lib, f)
    print(f"[method4] {len(lib['frags'])} fragment keys "
          f"(Method-1 source, now skeleton trajectories) -> {LIB_PATH.name}")
    return lib


# ---------------------------------------------------------------------------
# Method-2 backfill: one profile-driven trajectory per missing character
# ---------------------------------------------------------------------------
_char_cache = {}


def backfill_char_strokes(ch, profile, mm_per_xh=MM_PER_XH):
    if ch in _char_cache:
        return _char_cache[ch]
    traj = SY.SynthesizeText(ch, profile, mmPerXh=mm_per_xh, seed=0,
                             lineWidthMm=10_000.0,
                             legibility=profile.get("legibilityLambda", 0.6))
    strokes = [list(s) for s in traj.strokes if len(s) >= 2]
    if not strokes:
        _char_cache[ch] = None
        return None
    x0, x1 = _stroke_bounds(strokes)
    strokes = [[(x - x0, y) for x, y in s] for s in strokes]
    out = dict(strokes=strokes, width_mm=x1 - x0,
              entry=strokes[0][0], exit=strokes[-1][-1], conf=1.0, source="method2")
    _char_cache[ch] = out
    return out


# ---------------------------------------------------------------------------
# synthesis -- NO classifier calls, pure lookup + geometry
# ---------------------------------------------------------------------------
def _tile(word, frags):
    """Greedy longest-match. Deterministic: always the best-confidence
    instance of a key (frags[key] is sorted at build time) -- so the same
    word looks the same every time it's written, which matters more here
    than sampling variety."""
    out, i = [], 0
    while i < len(word):
        for Ln in range(min(MAX_NGRAM + 6, len(word) - i), 0, -1):
            key = word[i:i + Ln]
            if key in frags:
                out.append(("m1", frags[key][0])); i += Ln; break
        else:
            out.append(("m2", word[i])); i += 1
    return out


def synthesize(text, frag_lib=None, profile=None, seed=0, mm_per_xh=MM_PER_XH):
    if frag_lib is None:
        with open(LIB_PATH, "rb") as f:
            frag_lib = pickle.load(f)
    if profile is None:
        import json
        with open(LIB_DIR / "dylan.json", encoding="utf-8") as f:
            profile = json.load(f)
    rng = random.Random(seed)
    frags = frag_lib["frags"]
    word_gap = frag_lib["word_gap_xh"] * mm_per_xh * 0.8
    # a seam is only drawn when the ink genuinely looks like it touches --
    # this writer's connectedness is low (mostly print), so most letters
    # should get a natural pen-lift between them, not a stitched line
    seam_y_thresh_mm = 0.09 * mm_per_xh
    seam_x_thresh_mm = 0.16 * mm_per_xh

    strokes, x, n_m1, n_m2 = [], 0.0, 0, 0
    for word in text.split():
        prev_exit = None
        for kind, item in _tile(word, frags):
            frag = item if kind == "m1" else backfill_char_strokes(item, profile, mm_per_xh)
            if kind == "m1":
                n_m1 += 1
            else:
                n_m2 += 1
            if frag is None:
                x += 0.55 * mm_per_xh; prev_exit = None; continue
            dy = rng.uniform(-0.04, 0.04) * mm_per_xh
            placed = [[(px + x, py + dy) for px, py in s] for s in frag["strokes"]]
            ex, ey = frag["entry"][0] + x, frag["entry"][1] + dy
            if prev_exit is not None:
                pex, pey = prev_exit
                if abs(pey - ey) < seam_y_thresh_mm and 0 <= (ex - pex) < seam_x_thresh_mm:
                    strokes.append([(pex, pey), (ex, ey)])   # seam: still one pen-down stroke
            strokes.extend(placed)
            x += frag["width_mm"]
            prev_exit = placed[-1][-1]        # frag["exit"], translated by (x, dy)
        x += word_gap

    traj = SY.Trajectory(strokes, dict(mmPerXh=mm_per_xh, author="dylan_method4"))
    return traj, dict(n_m1=n_m1, n_m2=n_m2)


def render(traj, profile=None, px_per_mm=18.0):
    return SY.RenderTrajectory(traj, pxPerMm=px_per_mm, profile=profile, uniformInk=True)


def to_gcode(traj, out_path):
    import WriteGCode as GW
    cfg = GW.GantryConfig()
    return GW.WriteGcode(traj, cfg, str(out_path), title="method4")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?",
                    default="the quick brown fox jumps over the lazy dog")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--gcode", action="store_true")
    args = ap.parse_args()

    if args.build or not LIB_PATH.exists():
        build_fragment_trajectories()
    if not (LIB_DIR / "dylan.json").exists():
        M2.build_profile()

    traj, stats = synthesize(args.text)
    import json
    with open(LIB_DIR / "dylan.json", encoding="utf-8") as f:
        profile = json.load(f)
    img = render(traj, profile=profile)
    out = OUT_DIR / "method4_sample.png"
    img.save(out)
    print(f"\n  '{args.text}'")
    print(f"  {stats['n_m1']} chars from Method 1 (your real ink), "
          f"{stats['n_m2']} from Method 2 backfill")
    print(f"  -> {out}  {img.size}  (single pen width, uniformInk=True)")

    if args.gcode:
        gpath = OUT_DIR / "method4_sample.gcode"
        info = to_gcode(traj, gpath)
        print(f"  -> {gpath}  ({info['strokes']} strokes, {info['penPulses']} pen pulses)")


if __name__ == "__main__":
    main()
