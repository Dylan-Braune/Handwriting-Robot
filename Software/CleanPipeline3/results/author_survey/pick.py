"""Choose the two sets from the scored candidates, spread across styles.

Set A wants hands that are barely joined up and read easily, so the demo has
something that reproduces well.  Set B wants hands that are properly cursive
but still readable, to land in the harder band.  Both want fifteen writers who
do not look like one another, so that a writer classifier trained on them has
something to tell apart.

Selection is greedy max-min on the standardised style features, taken only from
candidates that clear a read-accuracy floor.  Ranking by accuracy and taking the
top fifteen would return fifteen neat, similar hands; spacing them out costs a
few points of accuracy and buys separability, which is the whole point of the
set.

The personal writers are fixed by hand: four in the legible set, thiya in the
cursive one, as asked.
"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
import numpy as np
import paths

FEATURES = ["compPerChar", "letterH", "aspect", "inkFrac"]
LEGIBLE_FLOOR = 0.78       #read accuracy a legible-set writer must clear
CURSIVE_FLOOR = 0.70       #and a cursive-set one: below this they will not
CURSIVE_CEIL = 0.98        #reproduce in the band at all


def Spread(rows, n, seeds=()):
    """greedy max-min, with any seed rows already counted as chosen"""
    if not rows:
        return []
    X = np.array([[r[f] for f in FEATURES] for r in rows], dtype=float)
    mu, sd = X.mean(0), np.where(X.std(0) > 1e-9, X.std(0), 1.0)
    Z = (X-mu)/sd
    if seeds:
        S = (np.array([[s[f] for f in FEATURES] for s in seeds], dtype=float)-mu)/sd
        d = np.min(np.linalg.norm(Z[:, None, :]-S[None, :, :], axis=2), axis=1)
    else:
        d = np.linalg.norm(Z-Z.mean(0), axis=1)
    chosen = []
    while len(chosen) < min(n, len(rows)):
        k = int(np.argmax(d))
        chosen.append(rows[k])
        d = np.minimum(d, np.linalg.norm(Z-Z[k], axis=1))
        d[k] = -1.0
    return chosen


def PersonalRows(names):
    """the photographed writers, measured the same way where possible"""
    out = []
    surv = {r["writer"]: r for r in json.load(open(HERE/"survey.json", encoding="utf-8"))}
    for n in names:
        r = surv.get(n) or dict(writer=n, compPerChar=np.nan, letterH=np.nan,
                                aspect=np.nan, inkFrac=np.nan)
        r = dict(r); r["personal"] = True
        out.append(r)
    return out


def Main():
    sl = json.load(open(HERE/"shortlist.json", encoding="utf-8"))
    legible = [r for r in sl["legible"] if r["readAcc"] >= LEGIBLE_FLOOR]
    cursive = [r for r in sl["cursive"]
               if CURSIVE_FLOOR <= r["readAcc"] <= CURSIVE_CEIL]
    print("after the read floor: %d legible, %d cursive candidates"
          % (len(legible), len(cursive)))

    setA = Spread(legible, 11)
    setB = Spread(cursive, 14)

    lines = []
    lines.append("TWO WRITER SETS, chosen from %d measured IAM writers" % 623)
    lines.append("")
    lines.append("compPerChar is blobs of ink per character: about 1 for a printed")
    lines.append("hand, well under 0.5 for a joined-up one.  read %% is this project's")
    lines.append("recogniser on the writer's OWN handwriting, which predicts how well")
    lines.append("they will reproduce but is not the same measurement.")
    lines.append("")
    for name, rows, fixed, note in [
            ("SET A -- disconnected / semi joined, legible", setA,
             ["abhinav", "dylan", "owen", "yeukita"],
             "expected to reproduce high"),
            ("SET B -- cursive, still readable", setB, ["thiya"],
             "expected to land in the harder band")]:
        lines.append("="*66)
        lines.append("%s  (%s)" % (name, note))
        lines.append("="*66)
        lines.append("%-10s %12s %9s %8s %9s" % ("writer", "compPerChar", "letterH",
                                                 "aspect", "read %"))
        for f in fixed:
            lines.append("%-10s %12s %9s %8s %9s   (yours, fixed)" % (f, "-", "-", "-", "-"))
        for r in sorted(rows, key=lambda r: -r["readAcc"]):
            lines.append("%-10s %12.2f %9.1f %8.2f %8.1f%%"
                         % (r["writer"], r["compPerChar"], r["letterH"],
                            r["aspect"], 100*r["readAcc"]))
        lines.append("")
        acc = np.array([r["readAcc"] for r in rows])
        lines.append("  %d chosen + %d of yours = %d.  read accuracy mean %.1f%%, "
                     "range %.1f-%.1f%%" % (len(rows), len(fixed), len(rows)+len(fixed),
                                            100*acc.mean(), 100*acc.min(), 100*acc.max()))
        cpc = np.array([r["compPerChar"] for r in rows])
        lines.append("  compPerChar spans %.2f to %.2f" % (cpc.min(), cpc.max()))
        lines.append("")

    #how far apart the chosen hands sit, which is what makes them tellable apart
    for name, rows in (("set A", setA), ("set B", setB)):
        X = np.array([[r[f] for f in FEATURES] for r in rows], float)
        Z = (X-X.mean(0))/np.where(X.std(0) > 1e-9, X.std(0), 1.0)
        D = np.linalg.norm(Z[:, None, :]-Z[None, :, :], axis=2)
        np.fill_diagonal(D, np.inf)
        lines.append("%s: closest pair of styles sits %.2f standard deviations apart"
                     % (name, D.min()))
    text = "\n".join(lines)
    print(text)
    (HERE/"SETS.txt").write_text(text+"\n", encoding="utf-8")
    json.dump({"setA": [r["writer"] for r in setA],
               "setB": [r["writer"] for r in setB]},
              open(HERE/"sets.json", "w"), indent=1)
    print("\nwrote %s" % (HERE/"SETS.txt"))


if __name__ == "__main__":
    Main()
