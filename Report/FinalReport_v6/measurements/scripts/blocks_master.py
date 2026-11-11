#the master per-line analysis of the fourteen reproduced pages, split by the
#three blocks the test was designed around.
#
#   block A  the 378 character paragraph -- many lines, so it tests line
#            segmentation and general text accuracy together
#   block B  the pangram, every letter of the alphabet in one line -- tests
#            single line reading and single line writer identification
#   block C  a piece of the writer's OWN source text -- the comparison and
#            overfitting check, because this text is the only one of the three
#            that the writer themselves actually wrote
#
#every line is one sample.  the three blocks are scored separately because
#they answer different questions, and pooling them would hide the difference.
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
warnings.filterwarnings("ignore")

CP3 = Path(r"C:\Users\braun\OneDrive\Documents\GitHub\Handwriting-Robot\Software\CleanPipeline3")
sys.path.insert(0, str(CP3)); sys.path.insert(0, str(CP3/"web"))
import segment as SEG
import recognise as REC
import server as SRV

PAGES = CP3/"ReportResults"/"FullReproducedPages"
DL = Path(r"C:\Users\braun\Downloads")
OUT = Path(__file__).resolve().parent/"blocks"
OUT.mkdir(exist_ok=True)

#page -> writer, and which label line indices belong to which block.
#a None end means "to the end of the label"
LAYOUT = {
 "Author150":  ("150",     {"A": (0, 8),  "B": (8, 10),  "C": (10, None)}),
 "Author152":  ("152",     {"A": (0, 10), "B": (10, 12), "C": (12, None)}),
 "Author1531": ("153",     {"A": (0, 13), "B": (13, None)}),
 "Author1532": ("153",     {"C": (0, None)}),
 "Author3841": ("384",     {"A": (0, 11), "B": (11, None)}),
 "Author3842": ("384",     {"C": (0, None)}),
 "Author552":  ("552",     {"A": (0, 9),  "B": (9, 11),  "C": (11, None)}),
 "Author588":  ("588",     {"B": (0, 2),  "C": (2, 7),   "A": (7, None)}),
 "AuthorA1":   ("abhinav", {"A": (0, None)}),
 "AuthorA2":   ("abhinav", {"B": (0, 2),  "C": (2, None)}),
 "AuthorD":    ("dylan",   {"A": (0, 11), "B": (11, 13), "C": (13, None)}),
 "AuthorO":    ("owen",    {"A": (0, 9),  "B": (9, 11),  "C": (11, None)}),
 "AuthorT":    ("thiya",   {"A": (0, 11), "B": (11, 13), "C": (13, None)}),
 "AuthorY":    ("yeukita", {"A": (0, 9),  "B": (9, 10),  "C": (10, None)}),
}

CKPT = DL/"author_shape_11_pw170_weights.npz"
PW = 0.17
namer = REC.AuthorClassifierCNNNumpy(checkpointPath=str(CKPT))
byIdx = {v: k for k, v in namer.authorMapping.items()}
NAMES = [byIdx[i] for i in range(len(byIdx))]
print("namer %s, %d writers, pen width %.2f" % (CKPT.name, len(NAMES), PW))
print("reader %s\n" % REC.DEFAULT_TEXT_WEIGHTS.name)

rows = []
for stem in sorted(LAYOUT):
    writer, blocks = LAYOUT[stem]
    want = [l.rstrip("\r") for l in
            (PAGES/(stem+".txt")).read_text(encoding="utf-8").strip().split("\n")]
    #self-check: block B must actually be the pangram, or the layout is wrong
    if "B" in blocks:
        a, b = blocks["B"]
        txt = " ".join(want[a:b]).lower()
        if "quick brown fox" not in txt:
            print("  LAYOUT ERROR %s: block B is %r" % (stem, txt[:50]))
            continue

    gray = np.ascontiguousarray(np.rot90(
        np.asarray(Image.open(PAGES/(stem+".jpg")).convert("L"), dtype=np.uint8)))
    lines, _d, info = SEG.SegmentLines(gray)
    crops = [np.asarray(l["image"]) for l in lines]
    crops, read, _s, order, flip, _f, _r = SRV.ReadAllLines(crops)
    lines = [lines[i] for i in order]
    if flip:
        read = read[::-1]
        crops = crops[::-1]
        lines = lines[::-1]
    if len(read) != len(want):
        print("  %s: %d crops vs %d label lines, skipped" % (stem, len(read), len(want)))
        continue

    #which block each line index falls in
    blockOf = {}
    for tag, (a, b) in blocks.items():
        for i in range(a, b if b is not None else len(want)):
            blockOf[i] = tag

    for i, (r, w) in enumerate(zip(read, want)):
        n = REC.StrokeNormalize(crops[i], penWidthXh=PW)
        _k, p = REC.ClassifyImage(Image.fromarray(n if n is not None else crops[i])
                                  .convert("L"), namer, strokeNormalise=False)
        p = np.asarray(p, dtype=np.float64)
        rows.append({
            "page": stem, "writer": writer, "line": i,
            "block": blockOf.get(i, "?"),
            "read": r, "want": w,
            "char": 100*REC.CharAcc(r, w), "word": 100*REC.WordAcc(r, w),
            #raw counts, so the totals can be MICRO averaged: one sum of edit
            #distances over one sum of reference characters.  a mean of
            #per-line percentages lets a three character line weigh as much
            #as a sixty character one, which it should not
            "dist": REC.Levenshtein(r, w), "refChars": len(w),
            "wordsRight": sum(1 for a, b in zip(r.split(), w.split()) if a == b),
            "refWords": len(w.split()),
            "named": NAMES[int(np.argmax(p))],
            "conf": 100*float(p.max()),
            "pTrue": 100*float(p[NAMES.index(writer)]),
            "ink": float(lines[i].get("inkPct", 0.0)),
            "nChars": len(w),
        })
    print("  %-12s %-8s %2d lines  blocks %s"
          % (stem, writer, len(read),
             " ".join("%s=%d" % (t, sum(1 for v in blockOf.values() if v == t))
                      for t in "ABC" if t in blocks)), flush=True)

(OUT/"lines.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
print("\n%d line samples written" % len(rows))


def Wilson(k, n, z=1.96):
    """Wilson score interval: the right one for a proportion at small n,
    unlike the normal approximation which can run past 0 or 1"""
    if n == 0:
        return (0.0, 0.0)
    p = k/n
    d = 1+z*z/n
    c = (p + z*z/(2*n))/d
    h = z*((p*(1-p)/n + z*z/(4*n*n)) ** 0.5)/d
    return (100*max(0.0, c-h), 100*min(1.0, c+h))


print("\n\n" + "="*86)
print("BY BLOCK  (every line is one sample)")
print("="*86)
print("%-38s %-6s %-22s %-10s %s"
      % ("block", "lines", "writer id  [95% CI]", "char %", "word %"))
BL = {"A": "A  paragraph, many lines",
      "B": "B  pangram, one or two lines",
      "C": "C  the writer's own source text"}
for t in "ABC":
    sub = [r for r in rows if r["block"] == t]
    if not sub:
        continue
    k = sum(1 for r in sub if r["named"] == r["writer"])
    lo, hi = Wilson(k, len(sub))
    print("%-38s %-6d %-22s %-10s %s"
          % (BL[t], len(sub), "%d/%d = %.1f%%  [%.1f, %.1f]"
             % (k, len(sub), 100*k/len(sub), lo, hi),
             "%.1f" % np.mean([r["char"] for r in sub]),
             "%.1f" % np.mean([r["word"] for r in sub])))
allk = sum(1 for r in rows if r["named"] == r["writer"])
lo, hi = Wilson(allk, len(rows))
print("%-38s %-6d %-22s %-10s %s"
      % ("all three", len(rows), "%d/%d = %.1f%%  [%.1f, %.1f]"
         % (allk, len(rows), 100*allk/len(rows), lo, hi),
         "%.1f" % np.mean([r["char"] for r in rows]),
         "%.1f" % np.mean([r["word"] for r in rows])))

def Micro(sub):
    """one sum over one sum: total edit distance against total characters"""
    d = sum(r["dist"] for r in sub)
    n = sum(r["refChars"] for r in sub)
    wr = sum(r["wordsRight"] for r in sub)
    wn = sum(r["refWords"] for r in sub)
    return (100*max(0.0, 1.0-d/n) if n else 0.0,
            100*wr/wn if wn else 0.0, d, n, wr, wn)


print("\n\n" + "="*92)
print("MICRO totals: summed over every line, not a mean of line percentages")
print("="*92)
print("%-34s %-27s %-27s %s" % ("block", "characters", "words", "writer id"))
for t in "ABC":
    sub = [r for r in rows if r["block"] == t]
    if not sub:
        continue
    c, wd, d, n, wr, wn = Micro(sub)
    k = sum(1 for r in sub if r["named"] == r["writer"])
    lo, hi = Wilson(k, len(sub))
    print("%-34s %-27s %-27s %d/%d = %.1f%% [%.1f, %.1f]"
          % (BL[t], "%d of %d wrong = %.2f%%" % (d, n, c),
             "%d of %d right = %.2f%%" % (wr, wn, wd), k, len(sub),
             100*k/len(sub), lo, hi))
c, wd, d, n, wr, wn = Micro(rows)
k = sum(1 for r in rows if r["named"] == r["writer"])
lo, hi = Wilson(k, len(rows))
print("%-34s %-27s %-27s %d/%d = %.1f%% [%.1f, %.1f]"
      % ("ALL THREE", "%d of %d wrong = %.2f%%" % (d, n, c),
         "%d of %d right = %.2f%%" % (wr, wn, wd), k, len(rows),
         100*k/len(rows), lo, hi))
print("\nmacro (mean of per-line percentages): char %.2f%%, word %.2f%%"
      % (np.mean([r["char"] for r in rows]), np.mean([r["word"] for r in rows])))
print("the two differ because the macro mean over-weights short lines.")

print("\n\nBY WRITER AND BLOCK, writer identification per line")
print("%-9s %-14s %-14s %-14s %s" % ("writer", "A paragraph", "B pangram", "C own text", "all"))
for w in NAMES:
    cells = []
    for t in "ABC":
        sub = [r for r in rows if r["writer"] == w and r["block"] == t]
        cells.append("%d/%d" % (sum(1 for r in sub if r["named"] == w), len(sub))
                     if sub else "-")
    sub = [r for r in rows if r["writer"] == w]
    if not sub:
        continue
    print("%-9s %-14s %-14s %-14s %d/%d"
          % (w, cells[0], cells[1], cells[2],
             sum(1 for r in sub if r["named"] == w), len(sub)))

print("\n\nCHARACTER ACCURACY BY WRITER AND BLOCK")
print("%-9s %-10s %-10s %-10s %s" % ("writer", "A", "B", "C", "all"))
for w in NAMES:
    sub = [r for r in rows if r["writer"] == w]
    if not sub:
        continue
    cells = []
    for t in "ABC":
        s2 = [r for r in sub if r["block"] == t]
        cells.append("%.1f" % np.mean([r["char"] for r in s2]) if s2 else "-")
    print("%-9s %-10s %-10s %-10s %.1f"
          % (w, cells[0], cells[1], cells[2], np.mean([r["char"] for r in sub])))
