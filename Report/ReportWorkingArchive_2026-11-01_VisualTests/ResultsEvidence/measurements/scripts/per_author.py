#accuracy per writer, and accuracy against how connected each writer's hand is.
#
#CONNECTEDNESS is measured as pen lifts per character: the synthesiser's own
#G-code for the same 378 character paragraph is counted, and the number of
#M3/M5 pen pulses divided by the character count.  a joined cursive hand
#lifts the pen once every few characters; a printed hand lifts it for almost
#every letter.  the measure is a property of the writer's glyph library, not
#of the text, because the text is identical for all eleven.
#
#the five photographed contributors are reported by initial.
import json
import sys
from pathlib import Path

import numpy as np

SCR = Path(__file__).resolve().parent
CP3 = Path(r"C:\Users\braun\OneDrive\Documents\GitHub\Handwriting-Robot\Software\CleanPipeline3")
sys.path.insert(0, str(CP3))

rows = json.loads((SCR/"blocks"/"lines.json").read_text(encoding="utf-8"))
pred = {r["writer"]: r for r in
        json.loads((SCR/"timing"/"predicted.json").read_text(encoding="utf-8"))}

#the photographed contributors by initial; the database writers keep their
#IAM numbers, which are already anonymous
LABEL = {"abhinav": "A", "dylan": "B", "owen": "O", "thiya": "T",
         "yeukita": "Y"}
NCHARS = 378.0


def Disp(w):
    return LABEL.get(w, w)


def Wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k/n
    d = 1+z*z/n
    c = (p+z*z/(2*n))/d
    h = z*((p*(1-p)/n+z*z/(4*n*n))**0.5)/d
    return 100*p, 100*max(0, c-h), 100*min(1, c+h)


writers = sorted({r["writer"] for r in rows}, key=lambda w: Disp(w))
out = []
print("="*96)
print("ACCURACY PER WRITER  (every line is one sample)")
print("="*96)
print("%-8s %-6s %-24s %-10s %-10s %-9s %s"
      % ("writer", "lines", "writer id  [95% CI]", "char %", "word %",
         "lifts/ch", "pen-down mm"))
for w in writers:
    sub = [r for r in rows if r["writer"] == w]
    k = sum(1 for r in sub if r["named"] == w)
    p, lo, hi = Wilson(k, len(sub))
    d = sum(r["dist"] for r in sub); n = sum(r["refChars"] for r in sub)
    wr = sum(r["wordsRight"] for r in sub); wn = sum(r["refWords"] for r in sub)
    ch = 100*max(0.0, 1-d/n)
    wo = 100*wr/wn if wn else 0.0
    lifts = pred[w]["pulses"]/NCHARS
    out.append({"writer": w, "label": Disp(w), "n": len(sub), "idOk": k,
                "id": p, "lo": lo, "hi": hi, "char": ch, "word": wo,
                "lifts": lifts, "downMm": pred[w]["downMm"]})
    print("%-8s %-6d %-24s %-10.2f %-10.2f %-9.3f %.0f"
          % (Disp(w), len(sub), "%d/%d = %.1f%% [%.1f, %.1f]"
             % (k, len(sub), p, lo, hi), ch, wo, lifts, pred[w]["downMm"]))

print("\nA = Abhinav, B = Braune, O = Owen, T = Thiya, Y = Yeukita;")
print("the six numbered writers are the public database contributors.")

#---connectedness bands-------------------------------------------------
print("\n\n" + "="*96)
print("ACCURACY BY CONNECTEDNESS  (pen lifts per character, from the G-code)")
print("="*96)
srt = sorted(out, key=lambda r: r["lifts"])
print("%-8s %-10s %s" % ("writer", "lifts/ch", "hand"))
for r in srt:
    print("  %-6s %-10.3f %s" % (r["label"], r["lifts"],
                                 "joined / cursive" if r["lifts"] < 2.0 else
                                 ("mixed" if r["lifts"] < 2.7 else "printed / separated")))

BANDS = [("joined (under 2.0 lifts/char)", lambda x: x < 2.0),
         ("mixed (2.0 to 2.7)", lambda x: 2.0 <= x < 2.7),
         ("separated (2.7 and over)", lambda x: x >= 2.7)]
print("\n%-34s %-7s %-24s %-9s %s"
      % ("band", "writers", "writer id  [95% CI]", "char %", "word %"))
for name, test in BANDS:
    ws = [r["writer"] for r in out if test(r["lifts"])]
    sub = [r for r in rows if r["writer"] in ws]
    if not sub:
        continue
    k = sum(1 for r in sub if r["named"] == r["writer"])
    p, lo, hi = Wilson(k, len(sub))
    d = sum(r["dist"] for r in sub); n = sum(r["refChars"] for r in sub)
    wr = sum(r["wordsRight"] for r in sub); wn = sum(r["refWords"] for r in sub)
    print("%-34s %-7d %-24s %-9.2f %.2f"
          % (name, len(ws), "%d/%d = %.1f%% [%.1f, %.1f]" % (k, len(sub), p, lo, hi),
             100*max(0, 1-d/n), 100*wr/wn if wn else 0))
    print("%-34s %s" % ("", "  " + ", ".join(Disp(x) for x in sorted(ws, key=Disp))))

#---correlation---------------------------------------------------------
x = np.array([r["lifts"] for r in out])
for key, lab in (("char", "character accuracy"), ("word", "word accuracy"),
                 ("id", "writer identification")):
    y = np.array([r[key] for r in out])
    pear = float(np.corrcoef(x, y)[0, 1])
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    spear = float(np.corrcoef(rx, ry)[0, 1])
    print("\n%-24s vs lifts/char:  Pearson r = %+.3f   Spearman rho = %+.3f   (n = %d)"
          % (lab, pear, spear, len(x)))

(SCR/"blocks"/"per_author.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print("\nwrote", SCR/"blocks"/"per_author.json")
