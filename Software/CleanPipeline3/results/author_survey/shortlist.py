"""Read each candidate's real handwriting, so the two sets can be picked on evidence.

survey.py says how joined up a hand is and how big its letters are, which is
enough to split the pool into printed and cursive but says nothing about whether
a hand can be READ.  That is the number both sets are specified in, so it has to
be measured rather than guessed at.

Candidates are drawn from each pool by greedy max-min spacing in the style
features, not by taking the top of a ranking: fifteen hands that all score well
on one measure tend to be fifteen versions of the same hand, and the point of
the set is that its members do not look like each other.

Each candidate's own pages are segmented and read with the ordinary recogniser,
and scored against the transcript.  That is NOT the reproduction accuracy the
sets are specified in -- reproduction needs a style profile and a synthesised
page, which is the slow path -- but it is the best predictor available without
building one, and it is measured on the same reader the pipeline uses.

Caveat worth keeping in view: this reader was trained on IAM lines, and these
are IAM writers, so the absolute numbers are optimistic.  The RANKING between
writers is still usable, which is what the selection needs.
"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
import numpy as np
from PIL import Image
import paths, recognise as REC, segment as SEG
from survey import PageInk, Otsu, ROOT, Transcript, RULE_COV, RuleBand

N_CANDIDATES = 32          #per pool, cut to 15 after reading
PAGES_PER_CANDIDATE = 1
FEATURES = ["compPerChar", "letterH", "aspect", "inkFrac"]

reader = REC.PaperCRNNNumpy(checkpointPath=REC.DEFAULT_TEXT_WEIGHTS)


#----------------------------------------------------------------------
#Spread: pick n rows as far from each other as the pool allows
#greedy max-min on standardised features: start from the row nearest the
#middle, then repeatedly add whichever remaining row is furthest from
#everything already chosen.  picking the top n of a ranking instead would
#return n near-identical hands, which is the opposite of what is wanted
#----------------------------------------------------------------------
def Spread(rows, n):
    if len(rows) <= n:
        return list(rows)
    X = np.array([[r[f] for f in FEATURES] for r in rows], dtype=float)
    X = (X-X.mean(0))/np.where(X.std(0) > 1e-9, X.std(0), 1.0)
    chosen = [int(np.argmin(np.linalg.norm(X-X.mean(0), axis=1)))]
    d = np.linalg.norm(X-X[chosen[0]], axis=1)
    while len(chosen) < n:
        k = int(np.argmax(d))
        chosen.append(k)
        d = np.minimum(d, np.linalg.norm(X-X[k], axis=1))
    return [rows[i] for i in chosen]


#----------------------------------------------------------------------
#HandBand: the handwritten part of the form, in the ORIGINAL greyscale
#the rules are found on a small binary copy because that is cheap, but
#what comes back is a crop of the real page: the reader was trained on
#photographs, and handing it a hard black-and-white mask rebuilt from a
#downscaled copy costs it most of its accuracy
#----------------------------------------------------------------------
def HandBand(path):
    im = Image.open(path).convert("L")
    full = np.asarray(im, dtype=np.uint8)
    small = np.asarray(im.resize((im.width//2, im.height//2), Image.BILINEAR), dtype=np.uint8)
    band = RuleBand(small < Otsu(small))
    return full if band is None else full[2*band[0]:2*band[1]]


#----------------------------------------------------------------------
#ReadWriter: how well the reader does on this writer's own pages
#the handwriting band is cut out first, because the IAM form prints the
#prompt above it and the segmenter would otherwise hand the reader
#typeset text, which it reads perfectly and which is nobody's hand
#----------------------------------------------------------------------
def ReadWriter(writer, nPages=PAGES_PER_CANDIDATE):
    d = ROOT/writer
    accs, nLines = [], 0
    for imgPath in sorted(d.glob("*.png"))[:nPages]:
        labPath = imgPath.with_name(imgPath.stem+"_labels.txt")
        if not labPath.exists():
            continue
        body, nCh, _nW = Transcript(labPath)
        if nCh < 40:
            continue
        texts = [l.strip() for l in labPath.read_text(encoding="utf-8", errors="ignore").splitlines()
                 if l.strip() and l.strip() != "MESS"]
        gray = HandBand(imgPath)
        if gray is None:
            continue
        lines, _ov, _info = SEG.SegmentLines(gray)
        if not lines:
            continue
        got = []
        for ln in lines:
            crop = Image.fromarray(np.asarray(ln["image"], dtype=np.uint8)).convert("L")
            got.append(REC.ReadText(crop, reader))
        #if the segmenter and the transcript disagree about how many lines
        #are on the page, the crops and the labels are not the same lines and
        #the score would be comparing one writer's line against another's
        #sentence.  such a page is dropped rather than measured wrongly --
        #it usually means the form had no rules to find and the typeset
        #prompt is still in frame
        if abs(len(got)-len(texts)) > 1:
            continue
        n = min(len(got), len(texts))
        if n == 0:
            continue
        accs.append(REC.CharAcc(" ".join(got[:n]), " ".join(texts[:n])))
        nLines += n
    if not accs:
        return None
    return float(np.mean(accs)), nLines


def Main():
    rows = json.load(open(HERE/"survey.json", encoding="utf-8"))
    rows = [r for r in rows if r["chars"] >= 100 and r["letterH"] >= 4]
    print("%d writers with enough text to judge" % len(rows))

    pools = {
        "legible":  [r for r in rows if r["compPerChar"] >= 0.55],   #printed and semi joined
        "cursive":  [r for r in rows if r["compPerChar"] <  0.35],   #joined up
    }
    out = {}
    for name, pool in pools.items():
        cands = Spread(pool, N_CANDIDATES)
        print("\n%s pool: %d writers, %d candidates picked for spread"
              % (name, len(pool), len(cands)))
        scored = []
        for i, r in enumerate(cands, 1):
            res = ReadWriter(r["writer"])
            if res is None:
                print("   %-5s unreadable/no labels" % r["writer"], flush=True)
                continue
            acc, nl = res
            rec = dict(r); rec["readAcc"] = acc; rec["lines"] = nl
            scored.append(rec)
            print("   %2d/%d  %-5s cpc %.2f  letterH %5.1f  read %5.1f%%  (%d lines)"
                  % (i, len(cands), r["writer"], r["compPerChar"], r["letterH"],
                     100*acc, nl), flush=True)
        out[name] = scored
    (HERE/"shortlist.json").write_text(json.dumps(out), encoding="utf-8")
    print("\nwrote %s" % (HERE/"shortlist.json"))


if __name__ == "__main__":
    Main()
