"""Render a few letters' stored variants large enough to judge by eye."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths  # noqa: F401

from PIL import Image, ImageDraw
import SynthesizeHandwriting as SY

author = sys.argv[1]
letters = [a for a in sys.argv[2:] if not a.startswith("--")]
profile_arg = next((a.split("=", 1)[1] for a in sys.argv[2:]
                    if a.startswith("--profile=")), None)
if profile_arg:
    path = Path(profile_arg)
    if not path.is_absolute():
        path = HERE / path
    prof = json.loads(path.read_text(encoding="utf-8"))
else:
    prof = json.loads((SY.PROFILE_DIR / f"{author}.json").read_text(encoding="utf-8"))
lib = prof["glyphs"]
CW, CH, SC = 115, 150, 58

rows = []
for ch in letters:
    vs = lib.get(ch, [])[:6]
    r = Image.new("RGB", (CW * 6, CH + 22), (252, 252, 250))
    d = ImageDraw.Draw(r)
    d.text((4, 4), repr(ch) + "   " + str(len(lib.get(ch, []))) + " variants",
           fill=(0, 0, 0))
    for i, g in enumerate(vs):
        cx = i * CW
        base = 18 + 0.74 * CH
        xs = [p[0] for s in g["strokes"] for p in s]
        if not xs:
            continue
        ox = cx + 8 - min(xs) * SC
        d.rectangle([cx + 1, 19, cx + CW - 2, 20 + CH - 2], outline=(230, 230, 230))
        d.line([cx + 2, base, cx + CW - 3, base], fill=(170, 190, 235))
        d.line([cx + 2, base - SC, cx + CW - 3, base - SC], fill=(235, 235, 235))
        for si, s in enumerate(g["strokes"]):
            if len(s) < 2:
                continue
            q = [(ox + x * SC, base - y2 * SC) for (x, y2) in s]
            d.line(q, fill=(0, 0, 0) if si == 0 else (200, 60, 30), width=2)
    rows.append(r)

W = max(r.width for r in rows)
H = sum(r.height + 6 for r in rows)
img = Image.new("RGB", (W, H), (255, 255, 255))
y = 0
for r in rows:
    img.paste(r, (0, y))
    y += r.height + 6
out = HERE / "out" / "letters_check.png"
img.save(out)
print("wrote", out, img.size)
