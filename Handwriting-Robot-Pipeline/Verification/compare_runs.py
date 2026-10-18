"""Compare the clean copy's measured metrics against the original pipeline's,
ignoring timing fields (which legitimately differ between runs)."""
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent
CLEAN = REPO / "Handwriting-Robot-Pipeline" / "Verification"
ORIG = REPO / "Software" / "CNN" / "NOGIT" / "full_pipeline_run"


def acc_from_csv(p):
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    n = len(rows)
    return dict(n=n,
                char=round(sum(float(r["char_acc"]) for r in rows) / n, 6),
                word=round(sum(float(r["word_acc"]) for r in rows) / n, 6))


def wr(p):
    d = json.loads(p.read_text(encoding="utf-8"))
    return dict(n=d["n_lines"],
                ink=round(d["overall"]["ink"]["acc"], 6),
                shape=round(d["overall"]["shape"]["acc"], 6))


def e2e(p):
    d = json.loads(p.read_text(encoding="utf-8"))
    m = d["mean"]
    return {k: round(v, 6) for k, v in m.items()}


print("=" * 78)
print("METRIC-LEVEL COMPARISON (timing fields ignored)")
print("=" * 78)

c = acc_from_csv(CLEAN / "text_accuracy_results.csv")
o = acc_from_csv(REPO / "Software" / "CNN" / "SegmentationBest" /
                 "text_accuracy_results.csv")
print(f"\ntext recognition, 669 real crops")
print(f"   clean    n={c['n']}  char={c['char']}  word={c['word']}")
print(f"   original n={o['n']}  char={o['char']}  word={o['word']}")
print(f"   -> {'MATCH' if c == o else 'DIFFER'}")

c = wr(CLEAN / "stage4_summary.json")
o = wr(ORIG / "stage4_summary.json")
print(f"\nwriter-ID, real lines")
print(f"   clean    n={c['n']}  ink={c['ink']}  shape={c['shape']}")
print(f"   original n={o['n']}  ink={o['ink']}  shape={o['shape']}")
print(f"   -> {'MATCH' if c == o else 'DIFFER'}")

c = e2e(CLEAN / "gcode_v2_summary.json")
o = e2e(ORIG / "gcode_v2_summary.json")
print(f"\nend-to-end through G-code, 13 writers")
for k in sorted(set(c) | set(o)):
    mark = "MATCH " if c.get(k) == o.get(k) else "DIFFER"
    print(f"   {k:<18} clean={c.get(k):<12} original={o.get(k):<12} {mark}")
