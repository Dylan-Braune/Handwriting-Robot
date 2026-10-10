"""Show, per page, the SegmentLean crop paired with the label it will be
extracted against -- so a label/crop mismatch is visible before it poisons the
glyph library.

    python check_alignment.py yeukita
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths  # noqa: F401

import numpy as np
from PIL import Image, ImageDraw

import SegmentLean as SL
from authors_config import personal_author_pages
from ExtractIAMLines import ReadLabelLines

author = sys.argv[1]
out = HERE / "out" / "alignment" / author
out.mkdir(parents=True, exist_ok=True)

for img_path, label_path in personal_author_pages(author):
    lean, _ov, info = SL.SegmentLines(str(img_path))
    gt = [g for g in ReadLabelLines(str(img_path), str(label_path))
          if g.strip() != "MESS"]
    print(f"\n{img_path.name}: {len(lean)} crops, {len(gt)} labels, "
          f"{'MATCH' if len(lean) == len(gt) else 'MISMATCH'}")

    # contact sheet with each crop's label printed beside it
    rows = min(len(lean), len(gt))
    if rows == 0:
        continue
    W = max(np.asarray(ln["image"]).shape[1] for ln in lean[:rows]) + 420
    H = sum(max(28, np.asarray(lean[i]["image"]).shape[0]) + 6 for i in range(rows))
    sheet = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(sheet)
    y = 0
    for i in range(rows):
        im = lean[i]["image"]
        sheet.paste(im, (0, y))
        d.text((im.width + 8, y + max(2, im.height // 2 - 6)),
               f"{i:02d}: {gt[i][:58]}", fill=0)
        y += max(28, im.height) + 6
    if sheet.width > 2400:                     # keep it openable
        s = 2400 / sheet.width
        sheet = sheet.resize((2400, int(sheet.height * s)), Image.LANCZOS)
    p = out / f"{img_path.stem}_paired.png"
    sheet.save(p)
    print(f"  wrote {p}")
