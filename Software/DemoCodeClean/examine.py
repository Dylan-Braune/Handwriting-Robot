"""Segment one page and show what came out, without the web app.

    python3 examine.py <image> [--labels file] [--out DIR]
    python3 examine.py --writer dylan --page 1
    python3 examine.py --list

The website does this already, but a demonstration sometimes needs one page
looked at on its own: every line the segmenter found, written out as its own
image, the overlay showing where it cut, what the reader made of each line,
and which writer the namer votes for.

Nothing here is a second implementation.  The segmentation, the reading, the
per-line naming and the ink-weighted page vote are the same calls the server
makes, through the same worker pool, so a disagreement between this and the
website is a bug in one of them rather than two pipelines drifting apart.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE/"web"))

import paths
import recognise as REC
import segment as SEG
import lineworkers as LW


#---PageImage: the chosen page as grey, upright, from a path or the dataset---
def PageImage(args):
    if args.image:
        p = Path(args.image)
    else:
        w = args.writer.capitalize()
        folder = paths.PersonalPages/w
        pages = sorted((folder/"Pages").glob(w+"*.jpg"))   #that writer pages in name order
        if not pages:
            raise SystemExit("no pages on disk for %r" % args.writer)
        if not 1 <= args.page <= len(pages):
            raise SystemExit("%s has %d pages, asked for %d"
                             % (w, len(pages), args.page))
        p = pages[args.page-1]   #page 1 is the first on disk
    if not p.exists():
        raise SystemExit("no such page: %s" % p)
    img = ImageOps.exif_transpose(Image.open(p)).convert("L")   #the exif rotation is applied then the photo flattened to grey
    return p, np.asarray(img, dtype=np.uint8)


#---Truth: the transcript beside a dataset page, or one given explicitly---
def Truth(pagePath, labelArg):
    if labelArg:
        lab = Path(labelArg)
    else:
        lab = pagePath.parent.parent/"Labels"/(pagePath.stem+"_labels.txt")   #the dataset keeps transcripts in a sibling Labels folder
    if not lab.exists():
        return None
    import re
    lines = [re.sub(r"^\d+\t", "", l.strip())
             for l in lab.read_text(encoding="utf-8", errors="replace").splitlines()
             if l.strip()]
    return [l for l in lines if l != "MESS"]


#---Overlay: the page with a box drawn round every line the segmenter kept---
def Overlay(gray, lines, pxPerBox=3):
    rgb = np.repeat(np.asarray(gray, dtype=np.uint8)[:, :, None], 3, axis=2)   #the grey page copied into three colour channels
    H, W = gray.shape
    for n, ln in enumerate(lines):
        X0, Y0, X1, Y1 = ln["bbox"]
        X0, X1 = max(0, min(W-1, X0)), max(0, min(W-1, X1))   #the box corners held inside the page
        Y0, Y1 = max(0, min(H-1, Y0)), max(0, min(H-1, Y1))
        #blue, because the report's figures are black, silver, white and blue
        col = (31, 78, 160) if n % 2 == 0 else (120, 160, 210)    #alternating shades so neighbouring boxes read apart
        for t in range(pxPerBox):
            for y in (Y0+t, Y1-t):   # the top and bottom edges of the box
                if 0 <= y < H:
                    rgb[y, X0:X1+1] = col
            for x in (X0+t, X1-t):   #and the left and right ones
                if 0 <= x < W:
                    rgb[Y0:Y1+1, x] = col
    return rgb


def Main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("image", nargs="?", default=None,
                    help="a photograph of a page")
    ap.add_argument("--writer", default=None,
                    help="instead of a path, a writer in the dataset")
    ap.add_argument("--page", type=int, default=1,
                    help="which of that writer's pages (1 is the first)")
    ap.add_argument("--labels", default=None,
                    help="a transcript, one written line per text line")
    ap.add_argument("--out", default=None,
                    help="where the crops and the overlay go")
    ap.add_argument("--no-read", dest="read", action="store_false",
                    help="segment only, skipping the networks")
    ap.add_argument("--list", action="store_true",
                    help="list the pages on disk and stop")
    args = ap.parse_args()

    if args.list:
        for w in paths.PersonalWriters:
            folder = paths.PersonalPages/w.capitalize()
            pages = sorted((folder/"Pages").glob(w.capitalize()+"*.jpg"))
            print("%-9s %d pages: %s"
                  % (w, len(pages), ", ".join(p.stem for p in pages)))
        return
    if not args.image and not args.writer:
        raise SystemExit("give an image path, or --writer NAME [--page N], "
                         "or --list")

    pagePath, gray = PageImage(args)
    outDir = Path(args.out) if args.out else paths.Cache/"Examined"/pagePath.stem
    (outDir/"lines").mkdir(parents=True, exist_ok=True)
    print("page     %s  %dx%d" % (pagePath.name, gray.shape[1], gray.shape[0]))

    up, deg, s0, s90 = SEG.UprightPage(gray)   #the page turned a quarter turn when it reads better that way
    if deg:
        print("turned   %d deg (line score %.2f upright, %.2f sideways)"
              % (deg, s0, s90))
    lines, _dbg, info = SEG.SegmentLines(up)   #the same segmentation call the server makes
    print("segment  %d lines, skew %+.1f deg, line pitch %d px, "
          "paper coverage %.0f%%"
          % (len(lines), info["angle"], info["pitch"],
             100*info["paperCoverage"]))
    print("         %d straight pixels removed as printing, %d artefact pieces, "
          "%d crop(s) dropped as solid"
          % (info["removedStraightPixels"], info["artifactPieces"],
             info["dropped"]))

    Image.fromarray(Overlay(up, lines)).save(outDir/"overlay.png")
    for n, ln in enumerate(lines):
        Image.fromarray(np.asarray(ln["image"])).convert("L").save(
            outDir/"lines"/("line%02d.png" % (n+1)))

    if not args.read:
        print("\nwrote %s and %d line image(s) to %s"
              % ("overlay.png", len(lines), outDir))
        return

    reader = REC.PaperCRNNNumpy()
    namer = REC.AuthorClassifierCNNNumpy()
    byIndex = {v: k for k, v in namer.authorMapping.items()}   #class index back to the writer name
    #stroke normalised, as the namer was trained and as the server names
    LW.Configure(reader, namer, True)  #the same worker pool the server uses
    crops = [np.asarray(ln["image"]) for ln in lines]

    read = [t for _i, t, _s in sorted(LW.Map(LW.ReadOne,   #the lines are read in parallel then put back in order
                                             [(i, c, False) for i, c in enumerate(crops)]),
                                      key=lambda r: r[0])]
    perLine, avg, weights = LW.NameWriter(crops)   #a vote per line and an ink weighted page vote
    truth = Truth(pagePath, args.labels)

    print()
    hdr = "line  ink%%   writer      conf   %s" % ("text" if not truth else "acc   text")
    print(hdr)
    print("-"*len(hdr))
    accs = []
    for n, (txt, probs) in enumerate(zip(read, perLine)):
        who = byIndex[int(np.argmax(probs))]
        conf = 100*float(np.max(probs))
        inkPct = 100*float(weights[n])/float(max(weights.sum(), 1))   # this line share of the page ink is its share of the vote
        if truth and n < len(truth):
            a = 100*REC.CharAcc(txt, truth[n])    #the reading against the transcript line of the same index
            accs.append(a)
            print("%4d %5.1f  %-10s %5.1f%% %5.1f%%  %s"
                  % (n+1, inkPct, who, conf, a, txt[:46]))
        else:
            print("%4d %5.1f  %-10s %5.1f%%  %s"
                  % (n+1, inkPct, who, conf, txt[:52]))

    page = byIndex[int(np.argmax(avg))]   #the highest scoring writer over the whole page
    print()
    print("writer   %s at %.1f%% over %d lines"
          % (page, 100*float(np.max(avg)), len(crops)))
    agree = sum(1 for p in perLine if byIndex[int(np.argmax(p))] == page)    #how many single lines name that same writer
    print("         %d/%d lines name that writer on their own" % (agree, len(perLine)))
    if truth:
        if len(truth) != len(lines):
            print("NOTE     %d transcript lines against %d written lines, so the "
                  "accuracies above are paired BY INDEX and may be out of step"
                  % (len(truth), len(lines)))
        if accs:
            print("text     %.1f%% mean per line" % (sum(accs)/len(accs)))
            #the whole page as one string against the whole transcript as
            #one string.  the website reports a page figure too, from its
            #own line alignment, so the two are close but not the same
            #measurement and should not be quoted as though they were
            print("page     %.1f%% reading the page as one string"
                  % (100*REC.CharAcc(" ".join(read), " ".join(truth))))   # the whole page as one string against the whole transcript
    print()
    print("wrote overlay.png and %d line image(s) to %s" % (len(lines), outDir))


if __name__ == "__main__":
    Main()
