"""Figure: why the normal (ink) classifier fails on robot-written text.

Puts, side by side for the three authors asked about:
  * a REAL line by that author
  * the line the pipeline draws for them (the G-code round trip)
  * their real line after StrokeNormalize (what the shape head sees)
with both heads' verdicts printed underneath.

Run from Software/CNN:
    python NOGIT/full_pipeline_run/make_gantry_figure.py
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

# path setup for the clean package layout
import sys as _sys
from pathlib import Path as _p
_root = _p(__file__).resolve().parent.parent
_cnn = _root / 'Software' / 'CNN'
_ar = _cnn / 'AuthorReproductionStuff'
_gantry = _root / 'Software' / 'GantryControl'
for _q in (_cnn, _ar, _ar / 'np_inference', _gantry):
    if str(_q) not in _sys.path:
        _sys.path.insert(0, str(_q))


THIS_DIR = Path(__file__).resolve().parent
CNN_DIR = _cnn
REPO = _root

from TrainText import IAMLineDatasetRaw, _decode_png  # noqa: E402
from TrainAuthor import StrokeNormalize  # noqa: E402
import numpy as np  # noqa: E402


CROPS = _cnn / "Crops"
E2E = THIS_DIR / "e2e_fast"
OUT = _root / "Verification" / "output" / "images"
DATA_DIR = _root / "Data" / "Datasets" / "IAMpages10"
CACHE = CNN_DIR / "NOGIT" / "line_cache_authors10"

FOCUS = ["abhinav", "owen", "yeukita"]
W = 1150


def band(img, label, h=74):
    im = img.convert("RGB")
    s = min(W / max(1, im.width), h / max(1, im.height))
    im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))),
                   Image.Resampling.LANCZOS)
    out = Image.new("RGB", (W, h + 20), "white")
    out.paste(im, (0, 20 + (h - im.height) // 2))
    ImageDraw.Draw(out).text((4, 4), label, fill=(0, 0, 170))
    return out


def real_line(author):
    """A real line by this author, from whichever source has one."""
    cap = author.capitalize()
    p = CROPS / cap
    if p.exists():
        for page in sorted(d for d in p.iterdir() if d.is_dir()):
            c = sorted(page.glob("[0-9][0-9].png"))
            if c:
                return Image.open(c[len(c) // 2]).convert("L")
    # IAM
    base = IAMLineDatasetRaw(root_dir=str(DATA_DIR), cache_dir=str(CACHE))
    for s in base.samples:
        if s["page_key"].split("/")[0] == author and 20 <= len(s["text"]) <= 60:
            return _decode_png(s["image_png"]).convert("L")
    return None


res = json.loads((THIS_DIR / "gantry_written_classify.json").read_text(encoding="utf-8"))
rows = res["rows"]


def verdict(author, source, cond):
    for r in rows:
        if r["author"] == author and r["source"] == source and r["condition"] == cond:
            return r
    return None


OUT.mkdir(parents=True, exist_ok=True)
for a in FOCUS:
    panels = []
    real = real_line(a)
    if real is not None:
        panels.append(band(real, f"REAL {a} line (this is what the ink head was trained on)"))
        n = StrokeNormalize(np.array(real))
        if n is not None:
            panels.append(band(n, "  same real line after StrokeNormalize (what the shape head sees)"))
    g = E2E / f"{a}_gcode.png"
    if g.exists():
        r = verdict(a, "e2e_fast", "through G-code")
        lbl = "ROBOT-WRITTEN (G-code round trip)"
        if r:
            lbl += (f"   ->  ink head: {r['ink_pred']} ({r['ink_conf']*100:.0f}%)"
                    f"   |  shape head: {r['shape_pred']} ({r['shape_conf']*100:.0f}%)")
        panels.append(band(Image.open(g), lbl))
        gn = StrokeNormalize(np.array(Image.open(g).convert("L")))
        if gn is not None:
            panels.append(band(gn, "  robot line after StrokeNormalize"))
    if not panels:
        continue
    G = 6
    H = sum(p.height for p in panels) + G * (len(panels) + 1)
    sheet = Image.new("RGB", (W + 2 * G, H), (236, 236, 236))
    y = G
    for p in panels:
        sheet.paste(p, (G, y))
        y += p.height + G
    o = OUT / f"gantry_vs_real_{a}.png"
    sheet.save(o)
    print(f"[{a}] {o}")
