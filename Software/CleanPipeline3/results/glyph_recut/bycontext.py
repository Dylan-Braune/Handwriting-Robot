"""Does the letter before it change the glyph that comes out?  The decisive test.

The first probe compared the cut against the spans CtcForcedAlign reports, which
was the wrong reference: CTC is peaky, so a character's span covers the few
frames where the reader was most certain and not the letter's real extent.  A
cut sitting outside that span means nothing by itself.  What it did establish is
that the snapping is innocent: it moves the boundary by 0.00 x-heights in the
median, so wherever the boundary is wrong, it arrived wrong.

This measures the thing that actually matters instead.  Every glyph the cutter
produces is recorded with the character that preceded it in the line.  If 'h'
after 't' comes out systematically wider and with more strokes than 'h' after
anything else, then the 't' is being swallowed, and the context is the cause.
If 'h' is equally fat everywhere, the letter is simply being cut badly and the
't' is a red herring.
"""
import sys, pickle
from collections import defaultdict
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
import numpy as np
import paths, recognise as REC, segment as SEG, style as ST, train as TR

WRITER = sys.argv[1] if len(sys.argv) > 1 else "abhinav"
model = REC.PaperCRNNNumpy(checkpointPath=ST.PERSONAL_WEIGHTS)


def Width(g):
    pts = [p for s in g["strokes"] for p in s]
    return (max(p[0] for p in pts)-min(p[0] for p in pts)) if pts else 0.0


def Crossbar(g):
    """a short markedly horizontal stroke at or above x-height: only t and f have one"""
    for s in g["strokes"]:
        xs = [p[0] for p in s]; ys = [p[1] for p in s]
        dx, dy = max(xs)-min(xs), max(ys)-min(ys)
        if dx > 0.25 and dx > 2.5*dy and np.median(ys) > 0.6:
            return True
    return False


recs = []
for imgPath, labelPath in paths.PersonalWriterPages(WRITER):
    gray = TR.ReadImageFile(imgPath)
    lines, _ov, _info = SEG.SegmentLines(gray)
    texts = [g for g in TR.ReadLabelLines(str(imgPath), str(labelPath))
             if g.strip() != "MESS"]
    for i in range(min(len(lines), len(texts))):
        crop = np.asarray(lines[i]["image"], dtype=np.uint8)
        glyphs, stats = ST.ExtractLineGlyphs(crop, texts[i], model)
        if glyphs is None:
            continue
        for k, g in enumerate(glyphs):
            #a space is stored as char and width only, with no strokes at all
            if g is None or "strokes" not in g:
                continue
            prev = texts[i][k-1] if 0 < k < len(texts[i]) else " "
            recs.append(dict(ch=g["char"], prev=prev, w=Width(g),
                             nStrokes=len(g["strokes"]), bar=Crossbar(g),
                             page=imgPath.stem, line=i))
    print("  %s done, %d glyphs so far" % (imgPath.stem, len(recs)), flush=True)

if not recs:
    raise SystemExit("nothing extracted for %s" % WRITER)

with open(HERE/("bycontext_%s.pkl" % WRITER), "wb") as f:
    pickle.dump(recs, f)

base = np.median([r["w"] for r in recs if r["ch"] in "nou"]) or 1.0
out = []
out.append("%s: %d glyphs cut from the pages, n/o/u width %.2f x-heights" %
           (WRITER, len(recs), base))
out.append("")
out.append("every letter, split by the character before it (groups of 3 or more)")
out.append("%-5s %-6s %5s %9s %9s %9s" % ("char", "after", "n", "width/nou", "strokes", "crossbar"))
byCh = defaultdict(list)
for r in recs:
    byCh[r["ch"]].append(r)
for ch in sorted(byCh):
    if len(ch) != 1 or not ch.isalpha():
        continue
    groups = defaultdict(list)
    for r in byCh[ch]:
        groups[r["prev"]].append(r)
    big = [(len(v), k) for k, v in groups.items() if len(v) >= 3]
    if len(big) < 2:
        continue
    allW = np.median([r["w"] for r in byCh[ch]])/base
    out.append("%-5s %-6s %5d %9.2f %9.1f %8.0f%%" %
               (ch, "ANY", len(byCh[ch]), allW,
                np.median([r["nStrokes"] for r in byCh[ch]]),
                100.0*sum(r["bar"] for r in byCh[ch])/len(byCh[ch])))
    for n, prev in sorted(big, reverse=True)[:4]:
        v = groups[prev]
        out.append("%-5s %-6s %5d %9.2f %9.1f %8.0f%%" %
                   ("", repr(prev), n, np.median([r["w"] for r in v])/base,
                    np.median([r["nStrokes"] for r in v]),
                    100.0*sum(r["bar"] for r in v)/len(v)))

#----------------------------------------------------------------------
#the h case on its own, since that is the one that was spotted by eye
#----------------------------------------------------------------------
H = [r for r in recs if r["ch"] == "h"]
if H:
    aft = [r for r in H if r["prev"] == "t"]
    oth = [r for r in H if r["prev"] != "t"]
    out.append("")
    out.append("'h' on its own: %d cut, %d of them after a 't' (%.0f%%)" %
               (len(H), len(aft), 100.0*len(aft)/len(H)))
    for name, v in (("after 't'", aft), ("after anything else", oth)):
        if not v:
            continue
        out.append("  %-20s n%-4d width %.2f  strokes %.1f  crossbar %.0f%%" %
                   (name, len(v), np.median([r["w"] for r in v])/base,
                    np.median([r["nStrokes"] for r in v]),
                    100.0*sum(r["bar"] for r in v)/len(v)))
    if aft and oth:
        d = (np.median([r["w"] for r in aft])-np.median([r["w"] for r in oth]))/base
        out.append("")
        out.append("  difference in width: %+.2f x-heights" % d)
        if d > 0.25:
            out.append("  VERDICT: the preceding 't' IS the cause.  an 'h' after a 't' "
                       "comes out materially wider than the same writer's 'h' elsewhere.")
        else:
            out.append("  VERDICT: the 't' is not the cause.  'h' is cut just as wide "
                       "whatever precedes it, so the fault is in how 'h' itself is cut.")

text = "\n".join(out)
print(text)
(HERE/("bycontext_%s.txt" % WRITER)).write_text(text+"\n", encoding="utf-8")
print("\nwrote %s" % (HERE/("bycontext_%s.txt" % WRITER)))
