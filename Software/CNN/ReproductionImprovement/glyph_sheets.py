"""Write one small contact sheet per letter of an author's stored glyphs, so
every stored glyph can be checked against the letter it is filed under.

    python glyph_sheets.py yeukita
    python glyph_sheets.py yeukita a b c   # only these
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths  # noqa: F401

import numpy as np
from PIL import Image, ImageDraw
import SynthesizeHandwriting as SY

CW, CH = 110, 150
SCALE = 58
NVAR = 6            # variants shown per sheet

author = sys.argv[1]
want = set(sys.argv[2:]) or None
prof = json.loads((SY.PROFILE_DIR / f"{author}.json").read_text(encoding="utf-8"))
lib = prof["glyphs"]
letters = [c for c in sorted(lib) if lib[c] and not c.isspace()]
if want:
    letters = [c for c in letters if c in want]

out = HERE / "out" / "sheets" / author
out.mkdir(parents=True, exist_ok=True)

for ch in letters:
    vs = lib[ch][:NVAR]
    n = len(vs)
    cols = max(1, n)
    img = Image.new("RGB", (cols * CW, CH + 20), (252, 252, 250))
    d = ImageDraw.Draw(img)
    pd = np.median([g.get("priorD", 0) for g in lib[ch]])
    d.text((4, 3), f"{ch!r}  {len(lib[ch])} variants  priorD {pd:.2f}",
           fill=(0, 0, 0))
    for i, g in enumerate(vs):
        cx, cy = i * CW, 20
        base = cy + 0.74 * CH
        xs = [p[0] for s in g["strokes"] for p in s]
        if not xs:
            continue
        ox = cx + 8 - min(xs) * SCALE
        d.rectangle([cx + 1, cy + 1, cx + CW - 2, cy + CH - 2], outline=(228, 228, 228))
        d.line([cx + 2, base, cx + CW - 3, base], fill=(170, 190, 235))
        d.line([cx + 2, base - SCALE, cx + CW - 3, base - SCALE], fill=(234, 234, 234))
        for si, s in enumerate(g["strokes"]):
            if len(s) < 2:
                continue
            q = [(ox + x * SCALE, base - y2 * SCALE) for (x, y2) in s]
            d.line(q, fill=(0, 0, 0) if si == 0 else (200, 60, 30), width=2)
    safe = "sp" if ch == " " else ("_" + hex(ord(ch))[2:] if not ch.isalnum() else ch)
    img.save(out / f"{safe}.png")

# one combined grid, letters stacked, to review in a single image
rows = []
for ch in letters:
    vs = lib[ch][:NVAR]
    r = Image.new("RGB", (NVAR * CW, 34), (255, 255, 255))
    dr = ImageDraw.Draw(r)
    dr.text((4, 10), f"{ch!r}", fill=(0, 0, 0))
    for i, g in enumerate(vs):
        cx = i * CW
        base = 28
        xs = [p[0] for s in g["strokes"] for p in s]
        if not xs:
            continue
        ox = cx + 22 - min(xs) * 40
        for si, s in enumerate(g["strokes"]):
            if len(s) < 2:
                continue
            q = [(ox + x * 40, base - y2 * 40) for (x, y2) in s]
            dr.line(q, fill=(0, 0, 0) if si == 0 else (200, 60, 30), width=2)
    rows.append(r)
if rows:
    grid = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows) + 8),
                     (255, 255, 255))
    y = 0
    for r in rows:
        grid.paste(r, (0, y))
        y += r.height + 8
    grid.save(out / "_all.png")
    print("wrote", out / "_all.png", grid.size)
print(f"{len(letters)} letter sheets -> {out}")
