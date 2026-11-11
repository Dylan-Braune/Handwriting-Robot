#every COMPLETE reproduction in the timing log, grouped by block length.
#
#a reproduction is asked for in two requests: one generates the page and one
#draws it.  the draw request carries the generation stages forward into its own
#total (timing.Run.Carry), so a block that contains a DRAWING stage is the
#complete reproduction and a generate-only block is a part of one.  counting
#both would count synthesis twice, so only blocks with a drawing stage are
#taken.
#
#the log is written in two layouts, one per server build:
#   "605.72 s total . 378 characters . 1 line(s)"          (browser panel)
#   "timing for reproduce: 1085.01 s total, 378 characters, 1 lines"  (console)
#both are read here, because the earlier Results draft quoted the drawing
#stage of the console layout as though it were the total and so understated
#the per-character figure.
#
#R3 allows 4 s per character.  the point of grouping by length is the worst
#case: the allowance scales with length while the best-of-N search is paid
#once per block, so the margin is thinnest on the shortest blocks.
import re
from collections import defaultdict
from pathlib import Path

SRC = Path(r"C:\Users\braun\Downloads\ALLTestData.txt")
lines = SRC.read_text(encoding="utf-8", errors="replace").splitlines()

PANEL = re.compile(r"^([\d.]+) s total . (\d+) characters . (\d+) line")
CONSOLE = re.compile(r"^\s*timing for (\w+): ([\d.]+) s total, (\d+) characters, (\d+) lines")

runs = []
for i, ln in enumerate(lines):
    m, kind = PANEL.match(ln), "panel"
    if m:
        total, chars = float(m.group(1)), int(m.group(2))
    else:
        m = CONSOLE.match(ln)
        if not m:
            continue
        kind, total, chars = m.group(1), float(m.group(2)), int(m.group(3))
    body = "\n".join(lines[i:i+16])
    if "drawing" not in body or chars == 0:
        continue
    syn = re.search(r"synthesis\s+([\d.]+)", body)
    drw = re.search(r"drawing\s+([\d.]+)", body)
    met = "MET" in body and "NOT MET" not in body
    runs.append({"srcLine": i+1, "kind": kind, "chars": chars, "total": total,
                 "perChar": total/chars, "met": met,
                 "synthesis": float(syn.group(1)) if syn else 0.0,
                 "drawing": float(drw.group(1)) if drw else 0.0})

#---two layouts, one run: the same reproduction is written to the console and
#to the browser panel, so a run that appears in both must be counted once.
#the pair is identified by its character count and its total to 0.1 s
seen, unique = set(), []
for r in runs:
    key = (r["chars"], round(r["total"], 1))
    if key in seen:
        continue
    seen.add(key)
    unique.append(r)
print("%d timing blocks with a drawing stage, %d distinct runs after removing "
      "the ones logged in both layouts" % (len(runs), len(unique)))
runs = unique

#---a job that was stopped part way is not a complete reproduction.  the
#gantry draws at a roughly constant rate, so a drawing stage far below the
#going rate did not finish.  these are identified and set aside rather than
#deleted, because they are in the log and their exclusion must be visible
STOPPED = 1.0
cut = [r for r in runs if r["chars"] and r["drawing"]/r["chars"] < STOPPED]
runs = [r for r in runs if r not in cut]
if cut:
    print("%d run(s) set aside as stopped before the page was finished "
          "(drawing below %.1f s per character):" % (len(cut), STOPPED))
    for r in cut:
        print("   log line %-6d %3d characters, drawing %.1f s = %.3f s/char"
              % (r["srcLine"], r["chars"], r["drawing"], r["drawing"]/r["chars"]))

print("\n%d complete reproductions, %d met their allowance\n"
      % (len(runs), sum(1 for r in runs if r["met"])))
print("%-8s %-6s %-10s %-10s %-11s %-11s %-11s %s"
      % ("chars", "runs", "allow s", "worst s", "worst s/ch", "% of allow",
         "synth s/ch", "draw s/ch"))
print("-"*94)
by = defaultdict(list)
for r in runs:
    by[r["chars"]].append(r)
for c in sorted(by):
    g = by[c]
    w = max(g, key=lambda r: r["perChar"])
    print("%-8d %-6d %-10.1f %-10.1f %-11.4f %-11.1f %-11.4f %.4f"
          % (c, len(g), 4*c, w["total"], w["perChar"], 100*w["perChar"]/4,
             w["synthesis"]/c, w["drawing"]/c))

print("\nevery complete reproduction, in log order:")
print("%-9s %-10s %-7s %-10s %-11s %-10s %s"
      % ("log line", "layout", "chars", "total s", "s/char", "% allow", "verdict"))
print("-"*78)
for r in runs:
    print("%-9d %-10s %-7d %-10.1f %-11.4f %-10.1f %s"
          % (r["srcLine"], r["kind"], r["chars"], r["total"], r["perChar"],
             100*r["perChar"]/4, "MET" if r["met"] else "NOT MET"))

worst = max(runs, key=lambda r: r["perChar"])
best = min(runs, key=lambda r: r["perChar"])
print("\nworst %.4f s/char on a %d-character block (%.1f %% of allowance, log line %d)"
      % (worst["perChar"], worst["chars"], 100*worst["perChar"]/4, worst["srcLine"]))
print("best  %.4f s/char on a %d-character block (%.1f %% of allowance)"
      % (best["perChar"], best["chars"], 100*best["perChar"]/4))

para = by.get(378, [])
if para:
    print("\nthe 378-character paragraph alone: %d runs, %.1f to %.1f s total, "
          "%.4f to %.4f s/char"
          % (len(para), min(r["total"] for r in para), max(r["total"] for r in para),
             min(r["perChar"] for r in para), max(r["perChar"] for r in para)))
    print("  spread between the slowest and the fastest: a factor of %.2f"
          % (max(r["perChar"] for r in para)/min(r["perChar"] for r in para)))

#---where the requirement would fail, from the two measured cost terms-----
short = by[min(by)]
ws = max(short, key=lambda r: r["perChar"])
synFixed, drawRate = ws["synthesis"], ws["drawing"]/ws["chars"]
if drawRate < 4:
    be = synFixed/(4-drawRate)
    print("\non the shortest block measured (%d characters) synthesis cost %.1f s "
          "for the block and drawing %.4f s per character." % (ws["chars"], synFixed, drawRate))
    print("treating the search as a fixed per-block cost, the allowance is "
          "exhausted at %.0f characters:" % be)
    print("   %.1f s + %.4f s/char x %.0f = %.1f s   against an allowance of %.1f s"
          % (synFixed, drawRate, be, synFixed+drawRate*be, 4*be))
    print("so a block shorter than about %.0f characters is expected to exceed "
          "R3.  the shortest measured is %d." % (be, ws["chars"]))
