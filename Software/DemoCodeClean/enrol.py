"""Add a writer: build a glyph library from photographs of their pages.

    python3 enrol.py <name> page1.jpg page2.jpg ...
    python3 enrol.py <name> --folder /path/to/Pages
    python3 enrol.py <name> page.jpg --transcript lines.txt
    python3 enrol.py <name> --folder ... --dry-run

Every page needs its transcript: either a '<page>_labels.txt' beside it (or
in a sibling Labels folder, as the dataset keeps them), or, for a single
page, --transcript pointing at a file of its lines, one written line per
text line and in order.

This is the same build the website's Add author does -- the same alignment,
the same cross-writer letter prior, the same output -- so a profile made
here and one made there are interchangeable.  Use --dry-run first at a
demonstration: it reports which written line matched which transcript line
and stops, which is where a bad transcript shows up.

A new writer can be WRITTEN but not JUDGED.  The namer's classes are fixed
by its checkpoint, so until it is retrained this writer has no class in it.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import paths
import recognise as REC
import segment as SEG
import style as STY


#---Pages: the chosen images, in order, each with the transcript beside it---
def Pages(args):
    if args.folder:
        folder = Path(args.folder)
        files = sorted(p for p in folder.iterdir()
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if not files:
            raise SystemExit("no .jpg or .png pages in %s" % folder)
    else:
        files = [Path(p) for p in args.images]
    for p in files:
        if not p.exists():
            raise SystemExit("no such page: %s" % p)
    return files


#---TranscriptFor: the label file for one page, looked for where they live---
def TranscriptFor(page, single, args):
    if args.transcript:
        if not single:
            raise SystemExit("--transcript is for a single page; give each page "
                             "its own <page>_labels.txt when there are several")
        return Path(args.transcript)
    for cand in (page.with_name(page.stem+"_labels.txt"),
                 page.parent/"Labels"/(page.stem+"_labels.txt"),
                 page.parent.parent/"Labels"/(page.stem+"_labels.txt")):
        if cand.exists():
            return cand
    raise SystemExit("no transcript for %s -- put %s_labels.txt beside it or in "
                     "a Labels folder, or pass --transcript for a single page"
                     % (page.name, page.stem))


#---Lines: a transcript file as the written lines it holds---
def Lines(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    #the shipped label files carry a leading "7<tab>" index
    out = [re.sub(r"^\d+\t", "", l.strip()) for l in text.splitlines() if l.strip()]
    return [l for l in out if l != "MESS"]


#---StoredLibraries: every profile on disk, for the cross-writer letter prior---
#whole profiles, exactly as the server's LoadProfiles hands them over: the
#build takes the glyph dict out of each itself.  stripping them to their
#glyphs here instead left it asking a glyph dict for its "glyphs" key, which
#is not an error -- it just quietly produced an empty prior, so the filter
#that drops a writer's worst cuts had nothing to compare them against
def StoredLibraries():
    libs = {}
    for f in sorted(paths.Profiles.glob("*.json")):
        try:
            libs[f.stem] = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print("[enrol] skipping profile %s: %s" % (f.name, e))
    return libs


def Main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("name", help="the writer's name: letters, digits, - and _")
    ap.add_argument("images", nargs="*", help="photographs of their pages")
    ap.add_argument("--folder", default=None, help="a folder of pages instead")
    ap.add_argument("--transcript", default=None,
                    help="single page only: a file of its written lines")
    ap.add_argument("--out", default=None,
                    help="where the profile goes (default: the profile folder)")
    ap.add_argument("--dry-run", dest="dryRun", action="store_true",
                    help="report the line pairing and stop, writing nothing")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing profile for this name")
    args = ap.parse_args()

    raw = args.name.strip().lower()
    name = "".join(c for c in raw if c.isalnum() or c in "_-")
    if not name or name != raw:
        raise SystemExit("a writer's name may hold letters, digits, '-' and '_' only")
    if not args.images and not args.folder:
        raise SystemExit("give some page images, or --folder")

    outPath = Path(args.out) if args.out else paths.Profiles/("%s.json" % name)
    if outPath.exists() and not args.dryRun and not args.force:
        raise SystemExit("there is already a profile at %s -- pick another name, "
                         "or pass --force to replace it" % outPath)

    files = Pages(args)
    single = len(files) == 1
    reader = REC.PaperCRNNNumpy()
    print("[enrol] reader %s" % Path(reader.checkpointPath).name)

    pages = []
    for p in files:
        lab = TranscriptFor(p, single, args)
        gray = np.asarray(ImageOps.exif_transpose(Image.open(p)).convert("L"),
                          dtype=np.uint8)
        gray, _deg, _s0, _s90 = SEG.UprightPage(gray)
        truth = Lines(lab)
        print("[enrol] %s with %s (%d transcript lines)"
              % (p.name, lab.name, len(truth)))
        pages.append((gray, truth))

    if args.dryRun:
        #the pairing on its own, which is what goes wrong on a real page
        for i, (gray, truth) in enumerate(pages):
            lines, _dbg, _info = SEG.SegmentLines(gray)
            crops = [np.asarray(ln["image"]) for ln in lines]
            pairs, read = STY.PairCropsToTranscript(crops, truth, reader)
            print("\n--- page %d: %d written lines, %d transcript lines, "
                  "%d paired (floor %.2f)"
                  % (i+1, len(crops), len(truth), len(pairs), STY.PAIR_FLOOR))
            #in written order, so a crop the alignment skipped is seen where
            #it actually sits on the page
            byCrop = {c: (t, s) for c, t, s in pairs}
            for c in range(len(crops)):
                if c in byCrop:
                    t, s = byCrop[c]
                    print("  line %2d -> transcript %2d  %5.1f%%  %r"
                          % (c+1, t+1, 100*s, read[c][:40]))
                else:
                    print("  line %2d -> NOTHING            %r  <- no transcript "
                          "line matched this crop" % (c+1, read[c][:40]))
            missed = set(range(len(truth)))-{t for _c, t, _s in pairs}
            for t in sorted(missed):
                print("  transcript %2d matched no written line: %r"
                      % (t+1, truth[t][:40]))
        print("\nnothing was written (--dry-run)")
        return

    res = STY.BuildProfileFromPages(name, pages, outPath, reader,
                                    priorLibs=StoredLibraries(),
                                    inNamer=False)
    print()
    print("profile  %s" % outPath)
    print("letters  %d characters, %d variants, from %d usable lines"
          % (res["characters"], res["variants"], res["lines_used"]))
    print("hand     slant %+.1f deg, connectedness %.2f, x-height %.1f px"
          % (res["slant_deg"], res["connectedness"], res["x_height_px"]))
    print("namer    %s"
          % ("has a class for this writer" if res["in_namer_classes"] else
             "has NO class for this writer: they can be written, not judged, "
             "until it is retrained"))
    print()
    print("the website picks this up on its next start, and lists %s under "
          "the writers it can synthesise" % name)


if __name__ == "__main__":
    Main()
