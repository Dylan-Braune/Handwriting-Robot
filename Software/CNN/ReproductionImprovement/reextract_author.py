"""Re-extract ONE author's glyphs with SegmentLean, into a TEST path.

yeukita's GlyphCache10/yeukita.pkl is from 2026/09/20 and was built from the old
SegmentPage crops, whose label pairing was unreliable. This rebuilds her raw
glyph library from the five pages using SegmentLean, and writes the cache and
profile into out/ so nothing live is overwritten.

    python reextract_author.py yeukita
    python reextract_author.py yeukita --write-live     # only once you have seen the audit
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths  # noqa: F401

import numpy as np

import BuildStyleProfile as BSP
import SegmentLean as SL
from authors_config import personal_author_pages, VAL_FRACTION
from ExtractIAMLines import ReadLabelLines
from np_inference.text_model import PaperCRNNNumpy

ap = argparse.ArgumentParser()
ap.add_argument("author")
ap.add_argument("--write-live", action="store_true",
                help="also write into NOGIT/GlyphCache10 and NOGIT/StyleProfiles10")
args = ap.parse_args()
author = args.author

out = HERE / "out" / "reextract" / author
out.mkdir(parents=True, exist_ok=True)


def personal_line_items(author):
    """TrainAuthor.add_personal_samples' segmentation, but keeping the page
    line order (no shuffle), which is the convention this cache uses."""
    import random
    rng = random.Random(0)
    items = []
    for img_path, label_path in personal_author_pages(author):
        lean, _ov, info = SL.SegmentLines(str(img_path))
        crops = [np.asarray(ln["image"]) for ln in lean]
        gt = [g for g in ReadLabelLines(str(img_path), str(label_path))
              if g.strip() != "MESS"]
        n = min(len(crops), len(gt))
        n_val = max(1, round(n * VAL_FRACTION))
        rows = list(zip(crops[:n], gt[:n]))
        rng.shuffle(rows)                       # same shuffle as the builder
        print(f"  {img_path.name}: {len(crops)} crops, {len(gt)} labels, "
              f"{n} paired, {n_val} held out  (pitch={info['pitch']})")
        for crop, text in rows[n_val:]:         # train rows only
            items.append((crop, text))
    return items


t0 = time.time()
items = personal_line_items(author)
print(f"\n{author}: {len(items)} training lines to extract "
      f"({time.time()-t0:.0f}s to segment)")

model = PaperCRNNNumpy(checkpoint_path=BSP.PERSONAL_WEIGHTS)
print(f"recognizer: {BSP.PERSONAL_WEIGHTS.name}")

parsed = []
for k, (gray, text) in enumerate(items):
    glyphs, stats = BSP.ExtractLineGlyphs(gray, text, model)
    if glyphs is None:
        continue
    stats = dict(stats)
    stats["ref"] = BSP.RenderRefStats(gray)
    parsed.append((glyphs, stats))
    if (k + 1) % 10 == 0:
        print(f"  {k+1}/{len(items)}  ({time.time()-t0:.0f}s)", flush=True)
print(f"{author}: {len(parsed)}/{len(items)} lines usable "
      f"({time.time()-t0:.0f}s)")

# raw library counts
lib = {}
for glyphs, _st in parsed:
    for g in glyphs:
        if g and g.get("char", " ") != " " and "strokes" in g:
            lib.setdefault(g["char"], []).append(g)
nv = sum(len(v) for v in lib.values())
print(f"raw glyphs: {len(lib)} letters, {nv} variants")
print(f"  i: {len(lib.get('i', []))}   l: {len(lib.get('l', []))}")

with open(out / f"{author}.pkl", "wb") as f:
    pickle.dump(parsed, f)
print(f"wrote {out / (author + '.pkl')}")

if args.write_live:
    import shutil
    live_pkl = BSP.RAW_DIR / f"{author}.pkl"
    shutil.copy2(out / f"{author}.pkl", live_pkl)
    print(f"WROTE LIVE: {live_pkl}")
    print("now run rebuild_author.py (or BuildAll10Authors) to rebuild the profile")
