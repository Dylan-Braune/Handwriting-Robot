#the exact match found nothing for 552, 588 and 384.  exact matching is
#brittle: a single transcription difference makes a line vanish.  so this
#repeats the question loosely -- for every line of those three writers, what
#is the CLOSEST Teklia line, by normalised edit distance?
#
#if a writer is really in Teklia, its lines should find near-perfect twins.
#if it is genuinely absent, the best match should sit far away.  writers 150,
#152 and 153 are included as the positive control, since their lines were
#found exactly and must therefore score ~100 per cent.
import re
from pathlib import Path

TEK = Path(r"C:\Users\braun\OneDrive\Documents\GitHub\Handwriting-Robot\Data\Datasets\TekliaIamLines")
IAM = Path(r"C:\Users\braun\OneDrive\Documents\GitHub\Handwriting-Robot\Data\Datasets\IAMpages10")


def Norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def Lev(a, b):
    if a == b:
        return 0
    prev = list(range(len(b)+1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j]+1, cur[j-1]+1, prev[j-1]+(ca != cb)))
        prev = cur
    return prev[-1]


pool = {}
for split in ("train", "validation"):
    for lab in sorted((TEK/split).glob("*_labels.txt")):
        t = Norm(lab.read_text(encoding="utf-8").strip())
        if len(t) >= 12:
            pool.setdefault(t, split)
keys = list(pool.items())
print("%d distinct Teklia lines pooled\n" % len(keys))

print("%-8s %-7s %-22s %s"
      % ("writer", "lines", "mean best match", "best-match split of the closest"))
print("-"*70)
for w in ("150", "152", "153", "384", "552", "588"):
    scores, where = [], {"train": 0, "validation": 0}
    for lab in sorted((IAM/w).glob("*_labels.txt")):
        if lab.name.endswith(".orig"):
            continue
        for l in lab.read_text(encoding="utf-8").splitlines():
            q = Norm(l)
            if len(q) < 12:
                continue
            #a long line cannot be close to a much shorter one, so only
            #candidates of comparable length are scored
            lo, hi = len(q)*0.6, len(q)*1.6
            best, bestSplit = 0.0, None
            for t, split in keys:
                if not (lo <= len(t) <= hi):
                    continue
                s = 1.0-Lev(q, t)/max(len(q), len(t))
                if s > best:
                    best, bestSplit = s, split
            scores.append(best)
            if bestSplit:
                where[bestSplit] += 1
    print("%-8s %-7d %-22s train %d, val %d"
          % (w, len(scores), "%.1f %%" % (100*sum(scores)/len(scores)),
             where["train"], where["validation"]), flush=True)
