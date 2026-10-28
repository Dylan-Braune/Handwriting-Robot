"""Save the line crops a rendered page actually hands to the classifier.

    python results/dumpcrops.py best_glyph_n1
    python results/dumpcrops.py all13_sentences
    python results/dumpcrops.py best_glyph_n1 --only owen_short

Every accuracy number in these folders is read off crops, not off the page: the
page is segmented into lines, and each line crop is what the reader and the
namer see.  A page that looks right can still be cut into crops that are not,
and then the score is measuring the segmenter rather than the handwriting.  The
overlays already drawn beside each page show where the boxes landed, but a box
on a big page is hard to judge, and the crop is the thing the network was given.

For each page this writes one PNG per line, a contact sheet of all of them
stacked in reading order, and a line in SUMMARY.txt with the sizes and how
much ink each crop holds.  Crops that look wrong by measurement -- almost no
ink, or an aspect ratio far from a line of text -- are flagged there.
"""
import argparse
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np
from PIL import Image, ImageDraw
import segment as SEG

#---MIN_INK_FRAC: a crop holding less ink than this is probably not a line---
MIN_INK_FRAC = 0.005

#---MIN_ASPECT: a line of text is far wider than it is tall---
MIN_ASPECT = 2.0


def Sheet(crops, label, pad=8):
    """every crop of one page stacked in reading order, numbered"""
    if not crops:
        return None
    w = max(c.shape[1] for c in crops)+2*pad
    h = sum(c.shape[0]+22 for c in crops)+2*pad+18
    im = Image.new("L", (w, h), 230)
    d = ImageDraw.Draw(im)
    d.text((pad, 4), label, fill=0)
    y = 18+pad
    for i, c in enumerate(crops):
        im.paste(Image.fromarray(c), (pad, y+14))
        d.text((pad, y), "line %d   %d x %d" % (i, c.shape[1], c.shape[0]), fill=0)
        y += c.shape[0]+22
    return im


def OnePage(pagePath, outRoot):
    gray = np.asarray(Image.open(pagePath).convert("L"), dtype=np.uint8)
    lines, _dbg, info = SEG.SegmentLines(gray)
    outDir = outRoot/pagePath.stem.replace("_page", "")
    outDir.mkdir(parents=True, exist_ok=True)
    crops, rows = [], []
    for i, ln in enumerate(lines):
        c = np.asarray(ln["image"], dtype=np.uint8)
        Image.fromarray(c).save(outDir/("line%02d.png" % i))
        crops.append(c)
        ink = float((c < 128).mean())
        asp = c.shape[1]/max(1.0, c.shape[0])
        flags = []
        if ink < MIN_INK_FRAC:
            flags.append("almost no ink")
        if asp < MIN_ASPECT:
            flags.append("not line shaped")
        rows.append((i, c.shape[1], c.shape[0], ink, asp, ", ".join(flags)))
    sh = Sheet(crops, pagePath.stem)
    if sh is not None:
        sh.save(outDir/"_sheet.png")
    return outDir, rows, info


def Main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder", help="a folder under results/ holding *_page.png")
    ap.add_argument("--only", default=None,
                    help="substring: only pages whose name contains it")
    args = ap.parse_args()

    root = HERE/args.folder
    if not root.is_dir():
        raise SystemExit("no such folder: %s" % root)
    pages = sorted(p for p in root.glob("*_page.png")
                   if "segmented" not in p.name
                   and (args.only is None or args.only in p.name))
    if not pages:
        raise SystemExit("no page images in %s" % root)

    outRoot = root/"crops"
    outRoot.mkdir(parents=True, exist_ok=True)
    report = ["crops handed to the classifier, from %d page(s) in %s" % (len(pages), args.folder),
              "",
              "ink is the fraction of dark pixels, aspect is width over height",
              ""]
    nFlag = 0
    for p in pages:
        outDir, rows, info = OnePage(p, outRoot)
        report.append("%s  ->  %d line(s)   %s" % (p.stem, len(rows), outDir.name))
        for i, w, h, ink, asp, flags in rows:
            nFlag += 1 if flags else 0
            report.append("    line %d  %5d x %-4d  ink %.3f  aspect %6.1f  %s"
                          % (i, w, h, ink, asp, flags))
        print("%-44s %d line(s)%s" % (p.stem, len(rows),
                                      "  <-- %d flagged" % sum(1 for r in rows if r[5])
                                      if any(r[5] for r in rows) else ""), flush=True)
    report.append("")
    report.append("%d crop(s) flagged by measurement across all pages" % nFlag)
    (outRoot/"SUMMARY.txt").write_text("\n".join(report)+"\n", encoding="utf-8")
    print("\nwrote %s" % (outRoot/"SUMMARY.txt"))
    print("one folder per page under %s, each with its own _sheet.png" % outRoot)


if __name__ == "__main__":
    Main()
