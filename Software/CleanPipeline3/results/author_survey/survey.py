"""Measure every writer in the 657-author set, cheaply, so a shortlist can be drawn.

Two sets are wanted: hands that are barely joined up and very legible, which
should reproduce well, and hands that are properly cursive but still readable,
which should land lower.  Deciding that by building a style profile for each
writer is not possible -- extraction is the slow path and there are 657 of them
-- so this measures the page image directly and uses no model at all.

What separates the two sets is how much of a word the pen draws without
lifting, and that is visible without reading anything: count the blobs of ink
and divide by the letters the transcript says are there.  A printed hand leaves
roughly one blob per letter; a cursive hand leaves about one per word.

The other measures are for telling writers APART rather than ranking them, so
that fifteen chosen hands do not all look alike: slant, letter size, how heavy
the pen is, how wide the letters sit, and how far they reach above and below.

Nothing here decides anything.  It writes one row per writer, and pick.py
chooses from those rows.
"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
import numpy as np
from PIL import Image
import segment as SEG

ROOT = (HERE.parents[3]/"Data"/"Datasets"/"IAMpages671"/"data")
PAGES_PER_WRITER = 2          #enough to be stable, cheap enough for 657 of them
MAX_SIDE = 1400               #work small: none of these measures need full detail


def Otsu(a):
    """the threshold that best splits ink from paper, by between-class variance"""
    hist = np.bincount(a.ravel(), minlength=256).astype(np.float64)
    tot = hist.sum()
    if tot <= 0:
        return 128
    w = np.cumsum(hist)
    m = np.cumsum(hist*np.arange(256))
    muT = m[-1]/tot          #the MEAN grey level, not the sum: with the sum
    #here the score peaks at the top of the range and calls the paper ink
    with np.errstate(invalid="ignore", divide="ignore"):
        between = (muT*w-m)**2/(w*(tot-w))
    between[~np.isfinite(between)] = 0.0
    return int(np.argmax(between))


#---RULE_COV: a printed rule covers at least this much of the row---
RULE_COV = 0.25

#---RULE_RUN: and does it in ONE unbroken run this wide---
#coverage alone is not enough: a heavily inked line of writing can cover a
#quarter of the row, and mistaking one for a rule cuts the band down to a
#sliver.  a rule is one continuous line across the page, writing is many
#short runs with gaps between the letters
RULE_RUN = 0.30


#---_LongestRun: the longest unbroken stretch of True in a row---
def _LongestRun(row):
    best = cur = 0
    for v in row:
        cur = cur+1 if v else 0
        if cur > best:
            best = cur
    return best


#----------------------------------------------------------------------
#RuleBand: the rows between two printed rules that hold the handwriting
#an IAM form is divided by three horizontal rules: under the heading,
#under the typeset prompt, and under the writing.  the writing is the
#BIGGEST gap between consecutive rules, which is what to look for -- some
#pages carry an extra rule, and assuming the last pair then returns a
#sliver of the page.  measuring a sliver while dividing by the whole
#transcript makes a plain hand look joined up, so this has to be right
#----------------------------------------------------------------------
def RuleBand(ink):
    W = ink.shape[1]
    cov = ink.sum(1)/max(1, W)
    groups = []
    for r in np.nonzero(cov > RULE_COV)[0]:
        if _LongestRun(ink[r]) < RULE_RUN*W:
            continue          #inked writing, not a rule
        if not groups or r-groups[-1] > max(4, ink.shape[0]//150):
            groups.append(int(r))
    if len(groups) < 2:
        return None
    pairs = [(groups[i+1]-groups[i], groups[i]+1, groups[i+1]) for i in range(len(groups)-1)]
    gap, y0, y1 = max(pairs)
    if gap < 0.25*ink.shape[0]:
        return None
    return y0, y1


#----------------------------------------------------------------------
#PageInk: the handwritten part of an IAM form, as an ink mask
#the form prints the prompt sentence at the top and the writer copies it
#underneath, so measuring the whole page would count the typeset text as
#if the writer had drawn it -- and typeset text is perfectly disconnected,
#which is exactly the thing being measured
#----------------------------------------------------------------------
def PageInk(path):
    im = Image.open(path).convert("L")
    k = max(im.size)/MAX_SIDE
    if k > 1:
        im = im.resize((int(im.width/k), int(im.height/k)), Image.BILINEAR)
    a = np.asarray(im, dtype=np.uint8)
    ink = a < Otsu(a)
    band = RuleBand(ink)
    if band is None:
        return ink
    y0, y1 = band
    out = np.zeros_like(ink)
    out[y0:y1] = ink[y0:y1]
    return out


def Transcript(labPath):
    try:
        txt = labPath.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return "", 0, 0
    lines = [l.strip() for l in txt.splitlines() if l.strip() and l.strip() != "MESS"]
    body = " ".join(lines)
    return body, len(body.replace(" ", "")), len(body.split())


#----------------------------------------------------------------------
#Measure: one writer, from a couple of their pages
#every number is a ratio or an angle so pages of different sizes and
#different amounts of text stay comparable
#----------------------------------------------------------------------
def Measure(writer, pages):
    comps = chars = words = 0
    hts, wds, inkFrac, slants = [], [], [], []
    nUsed = 0
    for imgPath in pages:
        labPath = imgPath.with_name(imgPath.stem+"_labels.txt")
        if not labPath.exists():
            continue
        _body, nCh, nWd = Transcript(labPath)
        if nCh < 40:
            continue
        ink = PageInk(imgPath)
        if not ink.any():
            continue
        lab, nc = SEG.LabelComponents(ink)
        if nc <= 0:
            continue
        ys, xs = np.nonzero(ink)
        pl = lab[ys, xs]
        area = np.bincount(pl, minlength=nc+1)
        top = np.full(nc+1, ink.shape[0]); bot = np.zeros(nc+1, int)
        lef = np.full(nc+1, ink.shape[1]); rig = np.zeros(nc+1, int)
        np.minimum.at(top, pl, ys); np.maximum.at(bot, pl, ys)
        np.minimum.at(lef, pl, xs); np.maximum.at(rig, pl, xs)
        h = bot-top+1
        w = rig-lef+1
        #blobs too small to be writing are specks, not letters or words
        sel = (area[1:] >= 8) & (h[1:] >= 3)
        if not sel.any():
            continue
        comps += int(sel.sum())
        chars += nCh
        words += nWd
        hts.append(float(np.median(h[1:][sel])))
        wds.append(float(np.median(w[1:][sel])))
        inkFrac.append(float(ink.mean()))
        slants.append(float(SEG.EstimateSlantDeg(ink)))
        nUsed += 1
    if not nUsed or not chars:
        return None
    medH = float(np.median(hts))
    return dict(writer=writer, pages=nUsed, chars=chars, words=words,
                compPerChar=comps/max(1, chars),
                compPerWord=comps/max(1, words),
                letterH=medH,
                aspect=float(np.median(wds))/max(1e-6, medH),
                inkFrac=float(np.median(inkFrac)),
                slant=float(np.median(slants)))


def Main():
    if not ROOT.is_dir():
        raise SystemExit("no author pages at %s" % ROOT)
    writers = sorted(d for d in ROOT.iterdir() if d.is_dir())
    print("%d writer folders under %s" % (len(writers), ROOT), flush=True)
    rows = []
    for i, d in enumerate(writers, 1):
        pages = sorted(d.glob("*.png"))[:PAGES_PER_WRITER]
        if not pages:
            continue
        try:
            r = Measure(d.name, pages)
        except Exception as e:
            print("  %s failed: %s" % (d.name, e), flush=True)
            continue
        if r:
            rows.append(r)
        if i % 50 == 0:
            print("  %d/%d measured, %d usable" % (i, len(writers), len(rows)), flush=True)
    out = HERE/"survey.json"
    out.write_text(json.dumps(rows), encoding="utf-8")
    print("\n%d writers measured -> %s" % (len(rows), out))

    cpc = np.array([r["compPerChar"] for r in rows])
    print("components per character: min %.2f  median %.2f  max %.2f"
          % (cpc.min(), np.median(cpc), cpc.max()))
    print("  (a printed hand leaves about one blob per letter, a cursive one far fewer)")
    for lo, hi, name in [(0.00, 0.25, "very cursive"), (0.25, 0.45, "cursive"),
                         (0.45, 0.70, "semi joined"), (0.70, 9.99, "printed")]:
        print("  %-14s %3d writers" % (name, int(((cpc >= lo) & (cpc < hi)).sum())))


if __name__ == "__main__":
    Main()
