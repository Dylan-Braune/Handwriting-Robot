"""Segmenter and classifier on the real photographed pages.

Everything else in results/ measures the system on its OWN output: a page is
synthesised, written to G-code, drawn back onto a canvas and then read.  That
tests reproduction.  It does not test the two parts that face real paper --
the segmenter, which has to find lines of ink in a photograph taken under a
lamp, and the ink classifier, which only means anything on a real photograph
because a drawn page carries one pen width everywhere.

This runs those two over the five photographed writers' own pages, against
the labels typed for them.

    python results/personal_pages/make.py
    python results/personal_pages/make.py --writer owen
    python results/personal_pages/make.py --overlays

Character accuracy here is the segmenter and the reader together: a line the
segmenter cuts badly is read badly, and the two cannot be separated from the
outside.  That is the honest figure for "read a page of handwriting", and it
is not comparable with the reproduction numbers in all13_sentences, which
start from clean synthetic ink.
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

import numpy as np
from PIL import Image, ImageDraw

import paths
import recognise as REC
import segment as SEG


def Save(img, path, tries=6):
    """OneDrive locks a file while it syncs; a save in that window dies."""
    for k in range(tries):
        try:
            img.save(path)
            return
        except OSError:
            if k == tries-1:
                raise
            time.sleep(0.5*(k+1))


def LabelText(labelPath):
    if not labelPath.exists():
        return None
    raw = labelPath.read_text(encoding="utf-8", errors="replace")
    return " ".join(raw.split())


def OnePage(imgPath, labelPath, reader, inkM, inkBy, overlays):
    page = np.asarray(Image.open(imgPath).convert("L"), dtype=np.uint8)
    t0 = time.perf_counter()
    lines, _dbg, _info = SEG.SegmentLines(page)
    segS = time.perf_counter()-t0
    if not lines:
        return dict(page=imgPath.stem, lines=0, acc=0.0, inkName="-", inkConf=0.0,
                    secs=segS, read="", truth=LabelText(labelPath) or "")

    texts, inkTot = [], None
    for ln in lines:
        crop = Image.fromarray(np.asarray(ln["image"])).convert("L")
        texts.append(REC.ReadText(crop, reader))
        #ink, not shape: these are real photographs, which is the only place
        #the ink model's evidence exists
        _k, pr = REC.ClassifyImage(crop, inkM, strokeNormalise=False)
        inkTot = np.asarray(pr, float) if inkTot is None else inkTot+pr

    got = " ".join(t for t in texts if t)
    truth = LabelText(labelPath)
    inkAvg = inkTot/max(1, len(lines))

    if overlays:
        ov = Image.fromarray(page).convert("RGB")
        dd = ImageDraw.Draw(ov)
        w = max(2, page.shape[1]//500)
        for ln in lines:
            b = ln["bbox"]
            dd.rectangle([b[0], b[1], b[2], b[3]], outline=(230, 30, 30), width=w)
        Save(ov, HERE/(imgPath.stem+"_segmented.png"))

    return dict(page=imgPath.stem, lines=len(lines),
                acc=(REC.CharAcc(got, truth) if truth else float("nan")),
                truthLines=(len(truth.split()) if truth else 0),
                inkName=inkBy[int(np.argmax(inkAvg))], inkConf=float(inkAvg.max()),
                secs=segS, read=got, truth=truth or "")


def Main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--writer", default=None, help="just this one")
    ap.add_argument("--overlays", action="store_true",
                    help="also write a page image with the line boxes drawn on")
    args = ap.parse_args()

    reader = REC.PaperCRNNNumpy(checkpointPath=REC.DEFAULT_TEXT_WEIGHTS)
    inkM = REC.AuthorClassifierCNNNumpy(checkpointPath=REC.DEFAULT_AUTHOR_WEIGHTS)
    inkBy = {v: k for k, v in inkM.authorMapping.items()}

    writers = [args.writer] if args.writer else list(paths.PersonalWriters)
    rows = []
    for w in writers:
        for imgPath, labelPath in paths.PersonalWriterPages(w):
            r = OnePage(imgPath, labelPath, reader, inkM, inkBy, args.overlays)
            r["writer"] = w
            r["named"] = (r["inkName"] == w)
            rows.append(r)
            print("%-9s %-14s %2d lines  char %5.1f%%  ink %-9s %3.0f%%  %s  %4.1fs"
                  % (w, r["page"], r["lines"], 100*r["acc"], r["inkName"],
                     100*r["inkConf"], "ok   " if r["named"] else "WRONG", r["secs"]))

    out = ["segmenter and ink classifier on the photographed pages",
           "  character accuracy is the segmenter and the reader together",
           "  the ink classifier is the right one here: these are real photographs",
           "",
           "%-9s %-14s %6s %8s %-10s %6s %7s" %
           ("writer", "page", "lines", "char %", "ink namer", "conf", "secs")]
    for r in rows:
        out.append("%-9s %-14s %6d %8.1f %-10s %5.0f%% %6.1f"
                   % (r["writer"], r["page"], r["lines"], 100*r["acc"],
                      r["inkName"], 100*r["inkConf"], r["secs"]))

    accs = np.array([r["acc"] for r in rows], float)
    good = accs[~np.isnan(accs)]
    named = sum(1 for r in rows if r["named"])
    out += ["", "per writer",
            "%-9s %6s %8s %8s %10s" % ("writer", "pages", "char %", "named", "secs/page")]
    for w in writers:
        sub = [r for r in rows if r["writer"] == w]
        if not sub:
            continue
        a = np.array([r["acc"] for r in sub], float)
        a = a[~np.isnan(a)]
        out.append("%-9s %6d %8.1f %5d/%-2d %10.1f"
                   % (w, len(sub), 100*a.mean() if a.size else float("nan"),
                      sum(1 for r in sub if r["named"]), len(sub),
                      float(np.mean([r["secs"] for r in sub]))))
    out += ["", "overall",
            "  pages            %d" % len(rows),
            "  character mean   %.1f%%" % (100*good.mean() if good.size else float("nan")),
            "  character median %.1f%%" % (100*np.median(good) if good.size else float("nan")),
            "  worst page       %.1f%%" % (100*good.min() if good.size else float("nan")),
            "  named right      %d/%d  %.0f%%" % (named, len(rows), 100*named/max(1, len(rows))),
            "  segment time     %.1f s per page" % float(np.mean([r["secs"] for r in rows])),
            "", "read back (first 90 chars)"]
    for r in rows:
        out.append("  %-9s %-14s %s" % (r["writer"], r["page"], r["read"][:90]))
        out.append("  %-9s %-14s %s" % ("", "(truth)", r["truth"][:90]))

    (HERE/"RESULTS.txt").write_text("\n".join(out)+"\n", encoding="utf-8")
    print("\nwrote", HERE/"RESULTS.txt")


if __name__ == "__main__":
    Main()
