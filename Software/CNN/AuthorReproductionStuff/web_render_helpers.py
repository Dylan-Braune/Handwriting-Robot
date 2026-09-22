"""
web_render_helpers.py -- shared image-generation helpers for server.py's
model/style info page: per-author glyph-grid images and real handwriting
sample crops. Pure rendering code, no Flask/HTTP here, so it's usable
standalone too (e.g. from a notebook) and easy to test in isolation.

Results are cached to disk under NOGIT/WebCache/ the first time each is
requested, since the glyph grids in particular are mildly expensive to
render and never change unless the profiles themselves change.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
NOGIT_DIR = SCRIPT_DIR.parent / "NOGIT"
CACHE_DIR = NOGIT_DIR / "WebCache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_FONT_CACHE = {}


def _font(size, bold=False):
    key = (size, bold)
    if key not in _FONT_CACHE:
        name = "consolab.ttf" if bold else "consola.ttf"
        try:
            _FONT_CACHE[key] = ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)
        except Exception:
            _FONT_CACHE[key] = ImageFont.load_default()
    return _FONT_CACHE[key]


def _render_one_glyph(g, cell=90, pad=8):
    strokes = g.get("strokes", [])
    im = Image.new("L", (cell, cell), 255)
    draw = ImageDraw.Draw(im)
    allpts = [p for s in strokes for p in s]
    if not allpts:
        return im
    xs = [p[0] for p in allpts]
    ys = [p[1] for p in allpts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    w = max(x1 - x0, 0.01)
    h = max(y1 - y0, 0.01)
    scale = (cell - 2 * pad) / max(w, h)

    def tx(p):
        return (pad + (p[0] - x0) * scale, cell - pad - (p[1] - y0) * scale)

    for s in strokes:
        pts = [tx(p) for p in s]
        if len(pts) >= 2:
            draw.line(pts, fill=0, width=2)
    return im


def glyph_grid_path(author):
    """Returns the path to a cached glyph-grid PNG for `author`, rendering
    it first if it doesn't exist yet. One row per character, one column
    per stored variant, matching the manual-inspection sheets used during
    development."""
    out_path = CACHE_DIR / f"glyphs_{author}.png"
    if out_path.exists():
        return out_path

    import SynthesizeHandwriting as SY
    profiles = SY.LoadAllProfiles()
    if author not in profiles:
        raise KeyError(f"No profile for author {author!r}")
    glyphs = profiles[author]["glyphs"]
    chars = sorted(glyphs.keys())
    max_variants = max(len(v) for v in glyphs.values())
    cell = 90
    label_w = 28
    W = label_w + max_variants * cell
    H = len(chars) * cell
    sheet = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(sheet)
    font = _font(13, bold=True)
    for ri, c in enumerate(chars):
        draw.text((4, ri * cell + cell // 2 - 8), repr(c)[1:-1], fill=(0, 0, 150), font=font)
        for ci, g in enumerate(glyphs[c]):
            im = _render_one_glyph(g, cell=cell)
            sheet.paste(im.convert("RGB"), (label_w + ci * cell, ri * cell))
    sheet.save(out_path)
    return out_path


def sample_crops(author, n=2):
    """Returns up to `n` (image_path, transcription) pairs of REAL,
    non-holdout handwriting for `author` -- one image per real line,
    cached to disk after first render. Used for the "what does this
    author's actual handwriting look like" gallery."""
    cached = sorted(CACHE_DIR.glob(f"sample_{author}_*.png"))
    meta_path = CACHE_DIR / f"sample_{author}_meta.txt"
    if len(cached) >= n and meta_path.exists():
        texts = meta_path.read_text(encoding="utf-8").splitlines()
        return list(zip(cached[:n], texts[:n]))

    import BuildStyleProfile as SP
    from TrainAuthor10 import add_personal_samples, DATASET_AUTHORS
    from TrainText import IAMLineDatasetRaw, _decode_png

    base = IAMLineDatasetRaw(root_dir=str(SP.DATA_DIR), cache_dir=str(SP.CACHE_DIR))
    if author in DATASET_AUTHORS:
        base.samples = [s for s in base.samples if s["page_key"].split("/")[0] == author]
    else:
        base.samples = []
        add_personal_samples(base)
        base.samples = [s for s in base.samples if s["page_key"].split("/")[0] == author]

    picked = []
    for s in base.samples:
        if s["is_holdout"]:
            continue
        if 20 <= len(s["text"]) <= 70:
            picked.append(s)
        if len(picked) >= n:
            break

    results = []
    texts = []
    for i, s in enumerate(picked):
        im = _decode_png(s["image_png"]).convert("L")
        path = CACHE_DIR / f"sample_{author}_{i}.png"
        im.save(path)
        results.append(path)
        texts.append(s["text"])
    meta_path.write_text("\n".join(texts), encoding="utf-8")
    return list(zip(results, texts))


def clear_cache(author=None):
    """Delete cached renders (call after rebuilding profiles/labels so
    stale images aren't served)."""
    pattern = f"*{author}*" if author else "*"
    for p in CACHE_DIR.glob(pattern):
        p.unlink()
