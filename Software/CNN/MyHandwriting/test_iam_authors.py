"""Test Method 1's philosophy (real per-instance ink fragments, greedy
tiling, naive seam-stitch) on the 10 IAM authors already used elsewhere in
this project, and compare against Method 2 (BuildStyleProfile /
SynthesizeHandwriting -- feature/style-profile reproduction, connectedness-
aware) on the SAME sentence, scored by the SAME frozen reader.

Method 1 on these lab-book pages works at word/2-3-letter granularity
because real ligature-spanning crops are cheap to cut from a print writer's
pages. The 10 IAM authors are much more cursive; no whole-page raw images
+ transcripts are wired up here, but NOGIT/GlyphCache10/<id>.pkl already
holds exactly the same underlying primitive Method 1 needs -- real,
per-instance, forced-aligned, skeletonised letter strokes cut from actual
IAM lines for each author (built earlier for Method 2's style profiles).
So Method 1 is tested here at its most basic unit: single real letters,
greedily tiled, with a seam stitched between every pair (no stored
word/bigram fragments are available for these writers) -- the fairest
apples-to-apples way to ask "does Method 1's philosophy hold up on a
cursive hand, or does the feature/style-profile approach do better?"

    python test_iam_authors.py
    python test_iam_authors.py --authors 150,155 --text "some other sentence"
"""
import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import _env
from _env import NOGIT_DIR, OUT_DIR

from method1_exemplar import _consensus_rerank, synthesize as m1_synthesize
import SynthesizeHandwriting as SY
import VerifyRewrite as VR
import BuildStyleProfile as SP

# AuthorReproductionStuff/*.py still point at a pre-move NOGIT/ under their
# own folder (see memory: "AuthorReproductionStuff paths broken") -- patch
# just the paths this script needs rather than touching that shared module.
VR.TEXT_WEIGHTS = _env.TEXT_WEIGHTS
SP.PROFILE_DIR = SY.PROFILE_DIR = NOGIT_DIR / "StyleProfiles10"

GLYPH_CACHE = NOGIT_DIR / "GlyphCache10"
PXH = 100.0
TEXT = "the quick brown fox jumps over the lazy dog"


def build_frags_from_glyphcache(author_id, min_n=3, max_per_key=8):
    """Real per-instance single-letter fragments for one IAM author, in the
    exact schema method1_exemplar.synthesize() already consumes -- so
    Method 1's own tiler/seam-stitcher/consensus-reranker run completely
    unmodified, just fed a different (real) fragment source."""
    data = pickle.load(open(GLYPH_CACHE / f"{author_id}.pkl", "rb"))
    frags = defaultdict(list)
    slants, gaps = [], []
    for glyph_seq, meta in data:
        slants.append(meta.get("slant", 0.0))
        wg = meta.get("ref", {}).get("wordGap")
        if wg is not None and np.isfinite(wg) and wg > 0:
            gaps.append(wg)
        for g in glyph_seq:
            if not g or g.get("char") in (None, " ") or not g.get("strokes"):
                continue
            top = g.get("top", 1.0)
            polys = [[(x * PXH, (top - y) * PXH) for x, y in stroke]
                     for stroke in g["strokes"] if stroke]
            polys = [p for p in polys if p]
            if not polys:
                continue
            frags[g["char"]].append(dict(
                polys=polys, xh=PXH, text=g["char"],
                w_px=float(g.get("advance", g.get("width", 0.6)) * PXH),
                base_off=float(top),
                entry_y=float(g.get("entryY", 0.5)),
                exit_y=float(g.get("exitY", 0.5)),
                conf=float(meta.get("alignConf", 0.0))))
    for kk in list(frags):
        frags[kk].sort(key=lambda f: -f["conf"])
    _consensus_rerank(frags, min_n=min_n)
    for kk in list(frags):
        frags[kk] = frags[kk][:max_per_key]
    wa = float(np.median(gaps)) if gaps else 1.3
    if not np.isfinite(wa) or wa <= 0:
        wa = 1.3
    return dict(xh=PXH, slant=float(np.median(slants)) if slants else 0.0,
                word_adv=wa, frags=dict(frags), n_lines=len(data))


def _sheet_row(draw, y, label, im, font, label_w=70, xh_row=150):
    s = xh_row / max(1, im.height)
    im2 = im.resize((max(1, int(im.width * s)), xh_row), Image.LANCZOS)
    draw.text((6, y + xh_row // 2 - 8), label, fill=0, font=font)
    return im2, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--authors", default=None)
    ap.add_argument("--text", default=TEXT)
    args = ap.parse_args()

    ids = args.authors.split(",") if args.authors else \
        sorted(p.stem for p in GLYPH_CACHE.glob("*.pkl"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reader = VR.LoadTextModel(device)
    profiles = SY.LoadAllProfiles()

    try:
        label_font = ImageFont.truetype(r"C:\Windows\Fonts\consola.ttf", 15)
    except OSError:
        label_font = ImageFont.load_default()

    rows = []
    print(f"'{args.text}'\n")
    for aid in ids:
        if aid not in profiles:
            print(f"  {aid}: no StyleProfiles10 profile, skipping Method 2 side")
            continue
        lib = build_frags_from_glyphcache(aid)
        n_keys = len(lib["frags"])
        im1, stats1 = m1_synthesize(args.text, lib=lib, seed=0, xh=40)
        got1 = VR.ReadText(reader, im1, device)
        c1, w1 = VR.CharAcc(got1.lower(), args.text.lower()), VR.WordAcc(got1.lower(), args.text.lower())

        prof = profiles[aid]
        traj = SY.SynthesizeText(args.text, prof, mmPerXh=4.0, seed=0,
                                 lineWidthMm=10_000.0,
                                 legibility=prof.get("legibilityLambda", 0.0))
        im2 = SY.RenderTrajectory(traj, pxPerMm=18.0, profile=prof, uniformInk=True)
        got2 = VR.ReadText(reader, im2, device)
        c2, w2 = VR.CharAcc(got2.lower(), args.text.lower()), VR.WordAcc(got2.lower(), args.text.lower())

        winner = "Method1" if (c1, w1) > (c2, w2) else "Method2" if (c2, w2) > (c1, w1) else "tie"
        print(f"  {aid}  keys={n_keys:3d}  anchor={stats1['n_anchor']}  |  "
              f"M1 char {c1*100:5.1f}% word {w1*100:5.1f}% got={got1!r}  |  "
              f"M2 char {c2*100:5.1f}% word {w2*100:5.1f}% got={got2!r}  |  -> {winner}")
        rows.append((aid, im1, c1, w1, got1, im2, c2, w2, got2, winner))

    # comparison sheet
    label_w, xh_row, pad = 90, 90, 10
    scaled = []
    for aid, im1, c1, w1, got1, im2, c2, w2, got2, winner in rows:
        s1 = xh_row / im1.height
        s2 = xh_row / im2.height
        r1 = im1.resize((max(1, int(im1.width * s1)), xh_row), Image.LANCZOS)
        r2 = im2.resize((max(1, int(im2.width * s2)), xh_row), Image.LANCZOS)
        scaled.append((aid, r1, c1, w1, got1, r2, c2, w2, got2, winner))
    W = label_w + max((max(s[1].width, s[5].width) for s in scaled), default=400) + 20
    H = sum(2 * xh_row + 60 for _ in scaled) + pad
    sheet = Image.new("L", (W, max(60, H)), 255)
    d = ImageDraw.Draw(sheet)
    y = pad
    for aid, r1, c1, w1, got1, r2, c2, w2, got2, winner in scaled:
        d.text((6, y), f"{aid}", fill=0, font=label_font)
        d.text((label_w, y), f"M1 (char {c1*100:.0f}% word {w1*100:.0f}%): {got1}", fill=0, font=label_font)
        sheet.paste(r1, (label_w, y + 18))
        y += xh_row + 20
        d.text((label_w, y), f"M2 (char {c2*100:.0f}% word {w2*100:.0f}%): {got2}", fill=0, font=label_font)
        sheet.paste(r2, (label_w, y + 18))
        y += xh_row + 24
        d.text((6, y - 2), f"-> {winner}", fill=0, font=label_font)
        y += 16

    out = OUT_DIR / "iam_method1_vs_method2.png"
    sheet.save(out)
    print(f"\n-> {out}  {sheet.size}")

    n1 = sum(1 for r in rows if r[-1] == "Method1")
    n2 = sum(1 for r in rows if r[-1] == "Method2")
    nt = sum(1 for r in rows if r[-1] == "tie")
    print(f"\nwinners: Method1 {n1}, Method2 {n2}, tie {nt}  (of {len(rows)} authors)")


if __name__ == "__main__":
    main()
