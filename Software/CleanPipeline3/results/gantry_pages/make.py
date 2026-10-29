"""The nine photographed gantry pages: the same text written nine ways.

These are photographs of pages the GANTRY wrote, not of a hand, so the two
author models mean different things here.  A plotted page carries one pen
width everywhere, which is the evidence the ink model reads, so the
stroke-normalised shape model is the one that means anything.  Both are
reported, as everywhere else, so the difference stays visible.

    python results/gantry_pages/make.py
    python results/gantry_pages/make.py --dir "<folder>" --truth "<text>"
"""
import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

import numpy as np
from PIL import Image, ImageDraw, ImageOps

import recognise as REC
import segment as SEG

SRC = Path(r"C:\Users\braun\OneDrive\Pictures\Camera Roll")
TRUTH = ("today i tested my gantry on my brother's handwriting, "
         "it hopefully works very well and gets the desired accuracy")


def Save(img, path, tries=6):
    for k in range(tries):
        try:
            img.save(path)
            return
        except OSError:
            if k == tries-1:
                raise
            time.sleep(0.5*(k+1))


def Main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(SRC))
    ap.add_argument("--truth", default=TRUTH)
    ap.add_argument("--count", type=int, default=9)
    args = ap.parse_args()

    files = sorted((p for p in Path(args.dir).glob("*.jpg")),
                   key=lambda p: p.stat().st_mtime, reverse=True)[:args.count]
    files = sorted(files, key=lambda p: p.name)
    if not files:
        sys.exit("no images found in %s" % args.dir)

    reader = REC.PaperCRNNNumpy(checkpointPath=REC.DEFAULT_TEXT_WEIGHTS)
    inkM = REC.AuthorClassifierCNNNumpy(checkpointPath=REC.DEFAULT_AUTHOR_WEIGHTS)
    shapeM = REC.AuthorClassifierCNNNumpy(checkpointPath=REC.SHAPE_AUTHOR_WEIGHTS)
    inkBy = {v: k for k, v in inkM.authorMapping.items()}
    shBy = {v: k for k, v in shapeM.authorMapping.items()}

    rows = []
    for f in files:
        #these came off a webcam and carry EXIF orientation 8.  PIL hands
        #back the RAW pixels, so the page is sideways to the segmenter while
        #every viewer shows it upright -- which cut ~40 vertical slices
        #through rotated text and scored 0 to 21 per cent
        page = np.asarray(ImageOps.exif_transpose(Image.open(f)).convert("L"),
                          dtype=np.uint8)
        t0 = time.perf_counter()
        lines, _dbg, info = SEG.SegmentLines(page)
        secs = time.perf_counter()-t0
        texts, inkTot, shTot = [], None, None
        for ln in lines:
            crop = Image.fromarray(np.asarray(ln["image"])).convert("L")
            texts.append(REC.ReadText(crop, reader))
            _k, pr = REC.ClassifyImage(crop, inkM, strokeNormalise=False)
            inkTot = np.asarray(pr, float) if inkTot is None else inkTot+pr
            _k, pr = REC.ClassifyImage(crop, shapeM)
            shTot = np.asarray(pr, float) if shTot is None else shTot+pr
        got = " ".join(t for t in texts if t)
        n = max(1, len(lines))
        ink = (inkTot/n) if inkTot is not None else np.zeros(len(inkBy))
        sh = (shTot/n) if shTot is not None else np.zeros(len(shBy))
        r = dict(name=f.stem, lines=len(lines), acc=REC.CharAcc(got, args.truth),
                 read=got, secs=secs, angle=info.get("angle"), pitch=info.get("pitch"),
                 ink=inkBy[int(np.argmax(ink))], inkConf=float(ink.max()),
                 shape=shBy[int(np.argmax(sh))], shapeConf=float(sh.max()))
        rows.append(r)
        print("%-28s %2d lines  char %5.1f%%  shape %-9s %3.0f%%  ink %-9s %3.0f%%  %4.1fs"
              % (r["name"], r["lines"], 100*r["acc"], r["shape"], 100*r["shapeConf"],
                 r["ink"], 100*r["inkConf"], r["secs"]))

        ov = Image.fromarray(page).convert("RGB")
        dd = ImageDraw.Draw(ov)
        wdt = max(2, page.shape[1]//400)
        for ln in lines:
            b = ln["bbox"]
            dd.rectangle([b[0], b[1], b[2], b[3]], outline=(230, 30, 30), width=wdt)
        Save(ov, HERE/(f.stem+"_segmented.png"))

    accs = np.array([r["acc"] for r in rows], float)
    out = ["the same page photographed nine times, different settings each time",
           "  truth: %s" % args.truth,
           "  these are GANTRY written pages: the shape namer is the meaningful one,",
           "  because a plotted page has one pen width everywhere and carries no ink evidence",
           "",
           "%-28s %6s %8s %-10s %6s %-10s %6s %7s %7s" %
           ("page", "lines", "char %", "shape namer", "conf", "ink namer", "conf",
            "angle", "pitch")]
    for r in rows:
        out.append("%-28s %6d %8.1f %-10s %5.0f%% %-10s %5.0f%% %7s %7s"
                   % (r["name"], r["lines"], 100*r["acc"], r["shape"], 100*r["shapeConf"],
                      r["ink"], 100*r["inkConf"],
                      ("%.2f" % r["angle"]) if r["angle"] is not None else "-",
                      ("%.0f" % r["pitch"]) if r["pitch"] is not None else "-"))
    out += ["",
            "overall",
            "  pages            %d" % len(rows),
            "  character mean   %.1f%%" % (100*accs.mean()),
            "  character best   %.1f%%  (%s)" % (100*accs.max(), rows[int(np.argmax(accs))]["name"]),
            "  character worst  %.1f%%  (%s)" % (100*accs.min(), rows[int(np.argmin(accs))]["name"]),
            "  spread           %.1f points" % (100*(accs.max()-accs.min())),
            "", "read back"]
    for r in rows:
        out.append("  %-28s %s" % (r["name"], r["read"]))

    (HERE/"RESULTS.txt").write_text("\n".join(out)+"\n", encoding="utf-8")
    print("\nwrote", HERE/"RESULTS.txt")


if __name__ == "__main__":
    Main()
