"""Render one author's stored glyph variants as a contact sheet, so the
quality of what was extracted can be judged directly.

    python show_glyphs.py yeukita            # all letters, one row each
    python show_glyphs.py yeukita e a n      # only these letters
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths  # noqa: F401

import numpy as np
import SynthesizeHandwriting as SY
from PIL import Image, ImageDraw

CW, CH = 120, 170          # cell size, px
SCALE = 62                 # px per x-height


def main():
    a = sys.argv[1]
    want = set(sys.argv[2:]) or None
    prof = json.loads((SY.PROFILE_DIR / f"{a}.json").read_text(encoding="utf-8"))
    lib = prof["glyphs"]
    letters = [c for c in sorted(lib) if lib[c] and not c.isspace()]
    if want:
        letters = [c for c in letters if c in want]

    maxn = max(len(lib[c]) for c in letters)
    cols = min(maxn, 10)
    rows = sum((len(lib[c]) + cols - 1) // cols for c in letters)
    img = Image.new("RGB", (cols * CW, rows * CH + 22 * len(letters)), (252, 252, 250))
    d = ImageDraw.Draw(img)
    y = 0
    for c in letters:
        d.text((4, y + 4), f"{c!r}  ({len(lib[c])} variants, "
                           f"priorD {np.median([g.get('priorD',0) for g in lib[c]]):.2f})",
               fill=(0, 0, 0))
        y += 22
        for i, g in enumerate(lib[c]):
            cx = (i % cols) * CW
            cy = y + (i // cols) * CH
            base = cy + 0.72 * CH
            xs = [p[0] for s in g["strokes"] for p in s]
            if not xs:
                continue
            ox = cx + 8 - min(xs) * SCALE
            d.rectangle([cx + 1, cy + 1, cx + CW - 2, cy + CH - 2], outline=(225, 225, 225))
            d.line([cx + 2, base, cx + CW - 3, base], fill=(170, 190, 235))
            d.line([cx + 2, base - SCALE, cx + CW - 3, base - SCALE], fill=(232, 232, 232))
            for si, s in enumerate(g["strokes"]):
                if len(s) < 2:
                    continue
                q = [(ox + x * SCALE, base - y2 * SCALE) for (x, y2) in s]
                d.line(q, fill=(0, 0, 0) if si == 0 else (200, 60, 30), width=2)
        y += ((len(lib[c]) + cols - 1) // cols) * CH
    out = HERE / "out" / f"glyphs_{a}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print("wrote", out, img.size)
    print(f"{len(letters)} letters, {sum(len(lib[c]) for c in letters)} variants")


if __name__ == "__main__":
    main()
