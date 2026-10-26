"""Glyph contact sheets: every extracted variant an author has, per letter.

One page per author. Each row is one letter; the variants are ordered
best-first by priorD (distance from the cross-author shape prior), with:

  * priorD printed under each variant,
  * the pick probability SynthesizeHandwriting._GlyphSource actually uses
    printed next to it, so it is visible which variant gets drawn most,
  * variants past _ELIGIBLE_VARIANTS greyed out -- they are stored but can
    never be selected,
  * the author's own ideal letterform (idealMedoid, the anchor they are
    swapped TO) drawn first in blue for comparison.

    cd Software/CleanPipeline
    python NOGIT/experiments/cursive_anchor/glyph_pages.py
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, r"C:\Users\braun\OneDrive\Documents\GitHub\Handwriting-Robot\Software\CleanPipeline")
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import SynthesizeHandwriting as SY

OUT_DIR = Path(__file__).resolve().parent / "new_results" / "glyph_pages"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AUTHORS = ["yeukita", "abhinav"]
LETTERS = "abcdefghijklmnopqrstuvwxyz"

CELL_W, CELL_H = 86, 104   #one variant cell
PAD = 8
SCALE = 30.0   #px per x-height
BASE_Y = 70   #baseline inside a cell


def font(size, bold=False):
    try:
        return ImageFont.truetype(
            "C:/Windows/Fonts/consolab.ttf" if bold else "C:/Windows/Fonts/consola.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_glyph(d, strokes, ox, oy, colour, width=2):
    """glyph space (x right, y up, 1.0 = x-height) -> cell pixels"""
    for s in strokes:
        if len(s) < 2:
            continue
        pts = [(ox + x * SCALE, oy - y * SCALE) for (x, y) in s]
        d.line(pts, fill=colour, width=width, joint="curve")


#the selection weights _GlyphSource.pick() uses with no previous letter to
#join onto: only the first _ELIGIBLE_VARIANTS are candidates at all
def pick_probs(n):
    k = min(n, SY._ELIGIBLE_VARIANTS)
    w = [1.0 / (1.0 + 1.4 * i) for i in range(k)]
    tot = sum(w)
    return [wi / tot for wi in w] + [0.0] * (n - k)


profiles = SY.LoadAllProfiles()
for author in AUTHORS:
    prof = profiles[author]
    lib = prof["glyphs"]
    ideal = prof.get("idealMedoid", {}) or {}
    lam = SY.AutoLegibility(prof)

    rows = []
    for ch in LETTERS:
        if ch in lib and lib[ch]:
            rows.append((ch, sorted(lib[ch], key=lambda g: g.get("priorD", 0.5))))
    maxN = max(len(v) for _, v in rows)

    W = 120 + (maxN + 1) * (CELL_W + PAD) + PAD
    H = 64 + len(rows) * (CELL_H + PAD)
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    allD = [g.get("priorD", 0.5) for _, v in rows for g in v]
    d.text((PAD, 8), f"{author}   {len(allD)} glyph variants   "
                     f"median priorD {np.median(allD):.3f}   AutoLegibility lam={lam:.2f}",
           fill=(0, 0, 0), font=font(16, True))
    d.text((PAD, 30), "blue = the author's own ideal letterform (the anchor).  "
                      "then their real variants, best priorD first.  "
                      "p = how often synthesis picks it.  grey = stored but never eligible.",
           fill=(90, 90, 90), font=font(11))

    y = 56
    for ch, variants in rows:
        d.text((PAD, y + CELL_H // 2 - 10), f" {ch} ", fill=(0, 0, 0), font=font(22, True))
        d.text((PAD + 36, y + CELL_H // 2 - 4), f"n={len(variants)}", fill=(90, 90, 90), font=font(11))
        x = 120
        #the anchor this letter would be replaced BY
        ig = ideal.get(ch)
        if ig:
            d.rectangle([x, y, x + CELL_W, y + CELL_H], outline=(170, 190, 230))
            draw_glyph(d, ig["strokes"], x + 14, y + BASE_Y, (20, 60, 200))
            d.text((x + 4, y + CELL_H - 16), "ideal", fill=(20, 60, 200), font=font(10))
        x += CELL_W + PAD

        probs = pick_probs(len(variants))
        for i, (g, p) in enumerate(zip(variants, probs)):
            eligible = i < SY._ELIGIBLE_VARIANTS
            col = (0, 0, 0) if eligible else (185, 185, 185)
            dd = float(g.get("priorD", 0.5))
            d.rectangle([x, y, x + CELL_W, y + CELL_H],
                        outline=(225, 225, 225) if eligible else (240, 240, 240))
            #baseline and x-height guides, so a fragment is obvious
            d.line([(x + 4, y + BASE_Y), (x + CELL_W - 4, y + BASE_Y)], fill=(235, 235, 235))
            d.line([(x + 4, y + BASE_Y - SCALE), (x + CELL_W - 4, y + BASE_Y - SCALE)],
                   fill=(245, 245, 245))
            draw_glyph(d, g["strokes"], x + 14, y + BASE_Y, col)
            #colour the priorD by how clean it is
            dc = ((0, 130, 0) if dd <= SY._GLYPH_CLEAN
                  else (200, 0, 0) if dd >= SY._GLYPH_GARBAGE else (190, 110, 0))
            d.text((x + 4, y + CELL_H - 28), f"d={dd:.3f}", fill=dc if eligible else col,
                   font=font(11, True))
            if p > 0:
                d.text((x + 4, y + CELL_H - 15), f"p={p*100:.0f}%", fill=(0, 0, 170),
                       font=font(11, True))
            x += CELL_W + PAD
        y += CELL_H + PAD
        d.line([(0, y - PAD // 2), (W, y - PAD // 2)], fill=(238, 238, 238))

    out = OUT_DIR / f"glyphs_{author}.png"
    sheet.save(out)
    print(f"{author}: {len(rows)} letters, {len(allD)} variants -> {out}")

    #a compact summary of which letters are weakest, as text
    worst = sorted(rows, key=lambda r: np.median([g.get("priorD", 0.5) for g in r[1]]))
    print("   best letters :", " ".join(
        f"{c}={np.median([g.get('priorD',0.5) for g in v]):.2f}" for c, v in worst[:8]))
    print("   worst letters:", " ".join(
        f"{c}={np.median([g.get('priorD',0.5) for g in v]):.2f}" for c, v in worst[-8:]))
