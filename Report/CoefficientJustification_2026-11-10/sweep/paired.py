#u23539195 coefficient justification: is a synthesis difference real, or seed noise?
#the synthesis sweeps run the same twelve seeds at every value, so two settings
#can be compared seed by seed instead of only by their means.  this prints, for
#each value against the submitted one, the paired mean difference, how many of
#the twelve seeds moved which way, and a paired t statistic
import json
import sys
from pathlib import Path

import registry

RESULTS = Path(__file__).resolve().parent.parent/"results"


#---PairedT: the t statistic of the paired differences, and its degrees of freedom---
def PairedT(d):
    n = len(d)
    if n < 2:
        return None,None
    mean = sum(d)/n
    var = sum((x-mean)**2 for x in d)/(n-1)
    if var <= 0:
        return (float("inf") if mean else 0.0),n-1
    import math
    return mean/math.sqrt(var/n),n-1


#---Report: one coefficient's values compared with its submitted value---
def Report(key,metric="charAccAll"):
    p = RESULTS/("%s.json" % key)
    if not p.exists():
        return
    rec = json.loads(p.read_text(encoding="utf-8"))
    runs = [r for r in rec["runs"] if r.get("ok") and metric in (r["metrics"] or {})]
    if len(runs) < 2:
        return
    base = next((r for r in runs if r["chosen"]),None)
    if base is None:
        return
    b = base["metrics"][metric]
    print("== %s   (submitted %s, %d seeds, metric %s)"
          % (key,base["value"],len(b),metric))
    for r in runs:
        a = r["metrics"][metric]
        if len(a) != len(b):
            continue
        d = [x-y for x,y in zip(a,b)]
        up = sum(1 for x in d if x > 1e-9)
        dn = sum(1 for x in d if x < -1e-9)
        same = len(d)-up-dn
        t,df = PairedT(d)
        mean = sum(d)/len(d)
        tag = "  <- submitted" if r["chosen"] else ""
        if r["chosen"]:
            print("   %-8s  (reference, mean %.4f)%s"
                  % (r["value"],sum(b)/len(b),tag))
            continue
        verdict = ""
        if t is not None and t == t:
            #two-sided 5 % for 11 degrees of freedom is about 2.20
            verdict = ("different" if abs(t) > 2.201 else "not distinguishable")
        print("   %-8s  mean diff %+.4f   better/worse/same %2d/%2d/%2d   "
              "t=%s   %s"
              % (r["value"],mean,up,dn,same,
                 ("%+.2f" % t) if t is not None and abs(t) != float("inf") else "--",
                 verdict))
    print()


def Main():
    keys = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not keys:
        keys = [e["key"] for e in registry.All()
                if e.get("task") == "synth"]
    print("Paired comparison against the submitted value, over the same seeds.")
    print("A difference is called 'different' only when the paired t exceeds")
    print("the two-sided 5 % point for 11 degrees of freedom, 2.201.\n")
    for k in keys:
        Report(k)
    return 0


if __name__ == "__main__":
    sys.exit(Main())
