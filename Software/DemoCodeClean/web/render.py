#shared image-generation helpers for Server.py's model/style info page:
#per-author glyph-grid images and real handwriting sample crops. Pure
#rendering code, no Flask/HTTP here, so it's usable standalone too (e.g.
#from a notebook) and easy to test in isolation.
#
#results are cached to disk under NOGIT/WebCache/ the first time each is
#requested, since the glyph grids in particular are mildly expensive to
#render and never change unless the profiles themselves change.
import os
import sys
from pathlib import Path

sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))

SCRIPT_DIR = Path(__file__).resolve().parent
#this file lives in Software/CleanPipeline, a sibling of Software/CNN,
#where only the DATA (profiles, caches, weights) lives -- every module
#SampleCrops below needs is a sibling of this file
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import paths

from PIL import Image,ImageDraw,ImageFont

CACHE_DIR = paths.Cache/"WebCache"
CACHE_DIR.mkdir(parents=True,exist_ok=True)

_FONT_CACHE = {}


def _font(size,bold=False):
    key = (size,bold)
    if key not in _FONT_CACHE:
        name = "consolab.ttf" if bold else "consola.ttf"
        try:
            _FONT_CACHE[key] = ImageFont.truetype(f"C:/Windows/Fonts/{name}",size)
        except Exception:
            _FONT_CACHE[key] = ImageFont.load_default()
    return _FONT_CACHE[key]


#renders one stored glyph (list of strokes, each a list of (x,y) points in
#the normalized glyph frame) into a small square cell image, scaled to fit
def _RenderOneGlyph(g,cell=90,pad=8):
    strokes = g.get("strokes",[])
    im = Image.new("L",(cell,cell),255)
    draw = ImageDraw.Draw(im)
    allpts = [p for s in strokes for p in s]
    if not allpts:
        return im
    xs = [p[0] for p in allpts]
    ys = [p[1] for p in allpts]
    x0,x1,y0,y1 = min(xs),max(xs),min(ys),max(ys)
    w = max(x1-x0,0.01)
    h = max(y1-y0,0.01)
    scale = (cell-2*pad)/max(w,h)

    def tx(p):
        return (pad+(p[0]-x0)*scale,cell-pad-(p[1]-y0)*scale)

    for s in strokes:
        pts = [tx(p) for p in s]
        if len(pts)>=2:
            draw.line(pts,fill=0,width=2)
    return im


#returns the path to a cached glyph-grid PNG for `author`, rendering it
#first if it doesn't exist yet. One row per character, one column per
#stored variant, matching the manual-inspection sheets used during
#development.
def GlyphGridPath(author,profile):
    outPath = CACHE_DIR/f"glyphs_{author}.png"
    if outPath.exists():
        return outPath

    glyphs = profile.get("glyphs") or {}
    if not glyphs:
        raise KeyError(f"No glyphs in the profile for {author!r}")
    chars = sorted(glyphs.keys())
    maxVariants = max(len(v) for v in glyphs.values())
    cell = 90
    labelW = 28
    W = labelW+maxVariants*cell
    H = len(chars)*cell
    sheet = Image.new("RGB",(W,H),"white")
    draw = ImageDraw.Draw(sheet)
    font = _font(13,bold=True)
    for ri,c in enumerate(chars):
        draw.text((4,ri*cell+cell//2-8),repr(c)[1:-1],fill=(0,0,150),font=font)
        for ci,g in enumerate(glyphs[c]):
            im = _RenderOneGlyph(g,cell=cell)
            sheet.paste(im.convert("RGB"),(labelW+ci*cell,ri*cell))
    sheet.save(outPath)
    return outPath


#----------------------------------------------------------------------
#SampleCrops: up to n real line crops for one writer, with transcripts
#the stats page shows these beside the glyph grid, so a viewer can
#compare the writer's actual hand against what the machine draws.
#holdout lines are skipped, since those are the measurement set, and only
#mid length lines are picked because a two word line shows nothing
#----------------------------------------------------------------------
def SampleCrops(author,n=2):
    metaPath = CACHE_DIR/f"sample_{author}_meta.txt"
    cached = sorted(CACHE_DIR.glob(f"sample_{author}_*.png"))
    if len(cached)>=n and metaPath.exists():
        texts = metaPath.read_text(encoding="utf-8").splitlines()
        return list(zip(cached[:n],texts[:n]))

    import numpy as np
    from PIL import Image
    import train as TR

    base = TR.IAMLineDatasetRaw(rootDir=str(TR.DEFAULT_AUTHOR_DATA_DIR),
                                cacheDir=str(TR.DEFAULT_AUTHOR_CACHE_DIR))
    if author in paths.DatabaseWriters:
        base.samples = [s for s in base.samples
                        if s["page_key"].split("/")[0]==author]
    else:
        base.samples = []
        TR.AddPersonalSamples(base)
        base.samples = [s for s in base.samples
                        if s["page_key"].split("/")[0]==author]

    picked = []
    for s in base.samples:
        if s.get("is_holdout"):
            continue
        if 20<=len(s["text"])<=70:
            picked.append(s)
        if len(picked)>=n:
            break

    outPaths,texts = [],[]
    for i,s in enumerate(picked):
        arr = TR.DecodeCrop(s["imageBytes"])
        path = CACHE_DIR/f"sample_{author}_{i}.png"
        Image.fromarray(np.asarray(arr,dtype=np.uint8)).convert("L").save(path)
        outPaths.append(path)
        texts.append(s["text"])
    metaPath.write_text("\n".join(texts),encoding="utf-8")
    return list(zip(outPaths,texts))
