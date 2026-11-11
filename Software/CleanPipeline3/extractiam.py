#u23539195 Dylan Braune.  cut an IAM form into its handwritten lines.
#
#   python extractiam.py                      every writer under IAMpages10
#   python extractiam.py --writers 384        one writer
#   python extractiam.py --writers 384 --dry-run
#
#WHAT THIS REPLACES.  the original Software/CNN/ExtractIAMLines.py did two
#jobs: it found the printed boundaries of the form, and it found the lines
#inside them with its own peak-finding segmenter.  the second job is now done
#far better by segment.py, which is the segmenter the rest of the system uses,
#so only the first job is kept here.
#
#the boundary code below is the original's, carried over because it is the
#part that works and the part segment.py does not do: segment.py is given a
#page of handwriting and finds its lines, but it has no notion of a printed
#prompt above the writing or a signature below it.  handed a whole IAM form it
#returns the handwriting correctly AND two extra bands of printed prompt and
#one of signature, measured on 384/j06-014: 15 crops for 12 written lines.
#cropping to the boundaries first removes all three.
#
#WHY THE CACHE NEEDED REBUILDING.  the crops in line_cache_authors10, written
#2026-10-24, contain merged lines: two written lines inside one crop.  writer
#384 carries 10 of the 13 merged crops in the whole 98 page corpus.  a merged
#crop desynchronises every label below it on that page, which is why 384 read
#at 60.78 per cent against 95 to 97 for the other database writers.  the
#current segment.py reproduces none of them: on all nine of 384's pages,
#cropped to the boundaries, it returns the right number of lines with zero
#merges.
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import paths
import segment as SEG


#======================================================================
#THE BOUNDARIES OF THE WRITING
#carried over from Software/CNN/ExtractIAMLines.py, which is otherwise
#retired.  an IAM form is: a printed heading, a rule, a printed copy of the
#prompt, a rule, the handwriting, a rule, "Name:" and a signature.  only the
#band between the second and third rules is wanted
#======================================================================
#---LongestRun: the longest unbroken run of ink in one row----------------
#a printed rule is one long run; a row of text is many short ones, so the
#longest run separates them where a simple ink count would not
def LongestRun(boolRow):
    inkCols = np.where(boolRow)[0]
    if len(inkCols) == 0:
        return 0
    gaps = np.where(np.diff(inkCols) > 1)[0]
    starts = np.concatenate(([0], gaps+1))
    ends = np.concatenate((gaps, [len(inkCols)-1]))
    return int(np.max(inkCols[ends]-inkCols[starts]+1))


#---ClusterRows: a rule is several adjacent rows, report it once----------
def ClusterRows(rows, maxGap=5):
    if not rows:
        return []
    clusters, group = [], [rows[0]]
    for i in range(1, len(rows)):
        if rows[i] <= rows[i-1]+maxGap:
            group.append(rows[i])
        else:
            clusters.append(int(np.mean(group)))
            group = [rows[i]]
    clusters.append(int(np.mean(group)))
    return clusters


#---FindRuleLineClusters: where the printed rules are---------------------
#the rules can be too faint to survive a global threshold, so the raw grey
#is used when it is available and the binary is only a fallback
def FindRuleLineClusters(binaryImg, rawGrayscaleArray=None):
    h, w = binaryImg.shape
    ruleRows = []
    src = rawGrayscaleArray if rawGrayscaleArray is not None else binaryImg
    for r in range(h):
        row = (src[r, :] < 225) if rawGrayscaleArray is not None else (src[r, :] > 0)
        if not np.any(row):
            continue
        if LongestRun(row) > w*0.25 and np.sum(row) > w*0.10:
            ruleRows.append(r)
    if not ruleRows:
        return []
    return ClusterRows(ruleRows, maxGap=8)


#---DetectPageBounds: the top and bottom of the handwriting---------------
#the fallbacks matter.  when no rule is found the boundary becomes a fixed
#fraction of the page height, which is a guess, and a guessed top can sit
#inside the printed prompt rather than below it.  the caller is told when
#that happened so it can refuse the page instead of labelling print as
#handwriting
def DetectPageBounds(binaryImg, rawGrayscaleArray=None, return_confidence=False):
    h, _w = binaryImg.shape
    clusters = FindRuleLineClusters(binaryImg, rawGrayscaleArray)
    topCandidates = [c for c in clusters if h*0.12 < c < h*0.40]
    botCandidates = [c for c in clusters if h*0.55 < c < h*0.92]
    topY = topCandidates[-1]+8 if topCandidates else int(h*0.12)
    botY = botCandidates[0]-8 if botCandidates else int(h*0.80)
    if return_confidence:
        return topY, botY, len(topCandidates) == 0, len(botCandidates) == 0
    return topY, botY


#---HeaderGapIsSafe: check the boundary against the ink, not the rules----
#independent of whether a rule was detected: the rows just above the top
#boundary should be nearly blank, because that is the gap after the printed
#prompt.  ink packed against the boundary means it is sitting on the print
def HeaderGapIsSafe(binaryImg, topY, minGapPx=10, maxInkFraction=0.05):
    zone = binaryImg[max(0, topY-minGapPx):topY, :]
    if zone.size == 0:
        return True
    return not np.any(np.mean(zone > 0, axis=1) > maxInkFraction)


#---Binarise: Otsu, for the boundary search only--------------------------
#segment.py does its own conditioning on the crop it is given; this is only
#to find the rules
def Binarise(gray):
    t = SEG.Otsu(gray)
    return ((gray < t).astype(np.uint8))*255


#======================================================================
#CUTTING ONE PAGE
#======================================================================
def ExtractPage(imgPath, verbose=True):
    """-> (crops, info).  crops is a list of uint8 line images, top to bottom.
    info carries the boundary decision so a caller can refuse a bad page."""
    gray = np.asarray(Image.open(imgPath).convert("L"), dtype=np.uint8)
    binary = Binarise(gray)
    topY, botY, topGuessed, botGuessed = DetectPageBounds(
        binary, gray, return_confidence=True)
    gapSafe = HeaderGapIsSafe(binary, topY)
    uncertain = topGuessed or not gapSafe

    body = np.ascontiguousarray(gray[topY:botY, :])
    try:
        lines, _dbg, seg = SEG.SegmentLines(body)
    except Exception as e:
        return [], {"error": str(e), "topY": topY, "botY": botY,
                    "uncertain": True, "topGuessed": topGuessed,
                    "gapSafe": gapSafe}

    crops = [np.asarray(l["image"], dtype=np.uint8) for l in lines]
    #a merged crop is about twice its page's median height.  segment.py does
    #not produce these on the current corpus, but the old cache was full of
    #them and an unreported one silently desynchronises every label below it
    merged = []
    if crops:
        hs = [c.shape[0] for c in crops]
        med = float(np.median(hs))
        merged = [i for i, h in enumerate(hs) if h > 1.75*med]

    info = {"topY": int(topY), "botY": int(botY), "topGuessed": bool(topGuessed),
            "botGuessed": bool(botGuessed), "gapSafe": bool(gapSafe),
            "uncertain": bool(uncertain), "merged": merged,
            "pitch": seg.get("pitch"), "angle": seg.get("angle"),
            "dropped": seg.get("dropped"), "nLines": len(crops)}
    if verbose:
        flags = []
        if topGuessed:
            flags.append("TOP RULE NOT FOUND, boundary guessed")
        if botGuessed:
            flags.append("bottom rule not found, boundary guessed")
        if not gapSafe:
            flags.append("INK ABOVE THE TOP BOUNDARY, it may sit on the print")
        if merged:
            flags.append("MERGED CROPS at %s" % merged)
        print("  %-16s rows %4d..%-4d  %2d lines  pitch %-4s  %s"
              % (Path(imgPath).name, topY, botY, len(crops), seg.get("pitch"),
                 "; ".join(flags) if flags else "ok"))
    return crops, info


#======================================================================
def Main():
    ap = argparse.ArgumentParser(
        description="cut IAM forms into handwritten line crops, using "
                    "segment.py for the lines and the original boundary "
                    "detection for the printed matter")
    ap.add_argument("--root", default=None,
                    help="the IAMpages10 folder (default: from paths.py)")
    ap.add_argument("--writers", nargs="+", default=None,
                    help="only these writer folders")
    ap.add_argument("--out", default=None,
                    help="write the crops here as PNG, one folder per page")
    ap.add_argument("--dry-run", dest="dry", action="store_true",
                    help="report what would be cut and write nothing")
    args = ap.parse_args()

    root = Path(args.root) if args.root else \
        paths.Data.parents[1]/"Data"/"Datasets"/"IAMpages10"
    if not root.exists():
        raise SystemExit("no IAM pages at %s" % root)
    writers = args.writers or sorted(p.name for p in root.iterdir() if p.is_dir())

    nPages = nLines = nMerged = nUncertain = nNoLabel = 0
    for w in writers:
        d = root/w
        if not d.is_dir():
            print("skipping %s: not a folder" % w)
            continue
        print("\n%s" % w)
        for png in sorted(d.glob("*.png")):
            lab = png.with_name(png.stem+"_labels.txt")
            crops, info = ExtractPage(png)
            nPages += 1
            nLines += len(crops)
            nMerged += len(info.get("merged", []))
            nUncertain += 1 if info.get("uncertain") else 0

            if not lab.exists():
                nNoLabel += 1
                print("       no %s, so this page cannot be used for training"
                      % lab.name)
            else:
                nLab = len([l for l in lab.read_text(encoding="utf-8").splitlines()
                            if l.strip()])
                if nLab != len(crops):
                    print("       LABEL MISMATCH: %d crops against %d label lines"
                          % (len(crops), nLab))

            if args.out and not args.dry:
                od = Path(args.out)/w/png.stem
                od.mkdir(parents=True, exist_ok=True)
                for i, c in enumerate(crops):
                    Image.fromarray(c).save(od/("line_%02d.png" % i))

    print("\n%d pages, %d line crops, %d merged crops, %d uncertain boundaries, "
          "%d pages with no label file" % (nPages, nLines, nMerged, nUncertain,
                                           nNoLabel))
    if args.dry:
        print("dry run: nothing was written")


if __name__ == "__main__":
    Main()
