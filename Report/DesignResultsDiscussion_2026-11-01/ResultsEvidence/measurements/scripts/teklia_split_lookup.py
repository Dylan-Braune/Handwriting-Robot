#which Teklia split holds the lines of each held-out page?
#
#the Teklia files are numbered 000000.png with no form or writer id, so the
#only handle is the transcript.  every Teklia label is indexed by a
#normalised form of its text; each held-out label line is then looked up.
#
#the distinction that matters: a line found in train/ was TRAINED ON by the
#reader, so measuring it is not a fair test.  a line found in validation/ was
#held out by Teklia's own split and never updated a weight.
import re
from collections import defaultdict
from pathlib import Path

TEK = Path(r"C:\Users\braun\OneDrive\Documents\GitHub\Handwriting-Robot\Data\Datasets\TekliaIamLines")
IAM = Path(r"C:\Users\braun\OneDrive\Documents\GitHub\Handwriting-Robot\Data\Datasets\IAMpages10")
WRITERS = ["150", "152", "153", "384", "552", "588"]


def Norm(s):
    #punctuation and spacing differ between the two transcriptions, so the
    #comparison is on letters and digits only, folded to lower case
    return re.sub(r"[^a-z0-9]", "", s.lower())


index = defaultdict(set)
for split in ("train", "validation"):
    n = 0
    for lab in sorted((TEK/split).glob("*_labels.txt")):
        t = Norm(lab.read_text(encoding="utf-8").strip())
        if len(t) >= 12:
            index[t].add(split)
            n += 1
    print("%-11s %d usable labels indexed" % (split, n))

print()
print("%-8s %-12s %-7s %s" % ("writer", "held-out page", "lines", "where each line is found"))
print("-"*74)
for w in WRITERS:
    d = IAM/w
    pages = sorted(d.glob("*.png"))
    labelled = [p for p in pages if p.with_name(p.stem+"_labels.txt").exists()]
    if not labelled:
        continue
    page = labelled[-1]
    want = [l for l in (page.with_name(page.stem+"_labels.txt"))
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    tally = defaultdict(int)
    for l in want:
        hit = index.get(Norm(l))
        tally["+".join(sorted(hit)) if hit else "not in Teklia"] += 1
    print("%-8s %-12s %-7d %s"
          % (w, page.stem, len(want),
             ",  ".join("%d %s" % (v, k) for k, v in sorted(tally.items()))))

#and the whole corpus, not only the held-out page, so the picture is complete
print()
print("%-8s %-7s %-14s %-14s %s"
      % ("writer", "lines", "in Teklia train", "in Teklia val", "in neither"))
print("-"*74)
for w in WRITERS:
    tr = va = no = tot = 0
    for lab in sorted((IAM/w).glob("*_labels.txt")):
        if lab.name.endswith(".orig"):
            continue
        for l in lab.read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            tot += 1
            hit = index.get(Norm(l)) or set()
            if "train" in hit:
                tr += 1
            elif "validation" in hit:
                va += 1
            else:
                no += 1
    print("%-8s %-7d %-14d %-14d %d" % (w, tot, tr, va, no))
