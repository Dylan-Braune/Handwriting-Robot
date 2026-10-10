"""Compare the line crops yeukita's profile would get from the two segmenters.

The profile builder (build_personal_line_items) uses SegmentPage.ProcessPage and
keeps only `tag == "TEXT"` crops. Everywhere else -- TrainAuthor.py, TrainText.py
-- personal authors are segmented with SegmentLean.SegmentLines. So the glyph
library and the recognizer may be looking at different crops of the same page.

This writes both sets of crops and prints the line counts against the label
files, so the difference (and any label/crop misalignment) is visible.

    python cmp_segmenters.py yeukita
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths

import numpy as np
from PIL import Image

import SegmentPage as SP
import SegmentLean as SL
from authors_config import personal_author_pages

OUT = HERE / "out" / "segmenters"


def main():
    author = sys.argv[1] if len(sys.argv) > 1 else "yeukita"
    OUT.mkdir(parents=True, exist_ok=True)

    for img_path, label_path in personal_author_pages(author):
        gt = [g for g in label_path.read_text(encoding="utf-8").splitlines()
              if g.strip() and g.strip() != "MESS"]
        results, _prev, meta = SP.ProcessPage(str(img_path))
        spT = [r for r in results if r["tag"] == "TEXT"]
        lean, _ov, info = SL.SegmentLines(str(img_path))

        print(f"\n{img_path.name}")
        print(f"  labels            : {len(gt)}")
        print(f"  SegmentPage TEXT  : {len(spT)}  (of {len(results)} regions)")
        print(f"  SegmentLean lines : {len(lean)}   pitch={info['pitch']} "
              f"angle={info['angle']}")

        for name, crops in (("page", [r["raw_crop"] for r in spT]),
                            ("lean", [np.asarray(l["image"]) for l in lean])):
            d = OUT / author / img_path.stem
            d.mkdir(parents=True, exist_ok=True)
            for i, c in enumerate(crops):
                Image.fromarray(c).save(d / f"{name}_{i:02d}.png")

        # side-by-side contact sheet: SegmentPage on the left, SegmentLean right
        def sheet(crops, label):
            if not crops:
                return None, 0
            gap = 6
            W = max(c.shape[1] for c in crops)
            H = sum(c.shape[0] for c in crops) + gap * len(crops)
            im = Image.new("L", (W, H), 240)
            y = 0
            for c in crops:
                im.paste(Image.fromarray(c), (0, y))
                y += c.shape[0] + gap
            return im, H

        left, hl = sheet([r["raw_crop"] for r in spT], "sp")
        right, hr = sheet([np.asarray(l["image"]) for l in lean], "lean")
        if left and right:
            H = max(hl, hr)
            cmp_img = Image.new("L", (left.width + right.width + 20, H), 255)
            cmp_img.paste(left, (0, 0))
            cmp_img.paste(right, (left.width + 20, 0))
            if cmp_img.width > 4000:
                s = 4000 / cmp_img.width
                cmp_img = cmp_img.resize((4000, int(H * s)), Image.LANCZOS)
            p = OUT / f"cmp_{author}_{img_path.stem}.png"
            cmp_img.save(p)
            print(f"  wrote {p}")


if __name__ == "__main__":
    main()
