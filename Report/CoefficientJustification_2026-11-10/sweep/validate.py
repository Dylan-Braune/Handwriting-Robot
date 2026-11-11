#u23539195 coefficient justification: every patch must name exactly one place, and
#the submitted value must rewrite the source to itself, which is the proof that the
#middle panel of every figure IS the submitted system
import sys

import patch
import registry


def Main():
    bad = n = 0
    for e in registry.All():
        for v in e["values"]:
            subs = e["patches"](v)
            if not subs:
                print("EMPTY %s v=%s" % (e["key"],v)); bad += 1
            for (mod,old,new) in subs:
                n += 1
                t = patch.SourceOf(mod)
                if t.count(old) != 1:
                    print("BAD   %-24s v=%-8s occurs %d: %s"
                          % (e["key"],v,t.count(old),old[:60])); bad += 1
                elif old == new and v != e["default"]:
                    print("NOOP  %-24s v=%-8s rewrites to itself but is not the default"
                          % (e["key"],v)); bad += 1
        if e["default"] not in e["values"]:
            print("MISS  %s default %r not among its values" % (e["key"],e["default"]))
            bad += 1
    print("%d entries, %d patch applications, %d problem(s)"
          % (len(registry.All()),n,bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(Main())
