#predict how long the 378 character paragraph takes to DRAW in each writer's
#style, so the nine measured paragraph times in ALLTestData.txt can be matched
#to the writer that produced them.
#
#the pen time is not a property of the text, it is a property of the hand: a
#looping cursive lays down far more millimetres of ink for the same words than
#an upright print does.  so the paragraph is regenerated for every writer, the
#pen-down and pen-up distances are summed out of the G-code, and the time is
#predicted from the two feed rates and the pen pulses.  ranking that against
#the measured times is evidence for the attribution rather than a guess.
#
#the synthesis is a best-of-N search and is seeded, so the absolute distance
#moves a little run to run; the RANK is what is used, not the absolute value.
import json
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
warnings.filterwarnings("ignore")

CP3 = Path(r"C:\Users\braun\OneDrive\Documents\GitHub\Handwriting-Robot\Software\CleanPipeline3")
sys.path.insert(0, str(CP3)); sys.path.insert(0, str(CP3/"web"))
import server as SRV

OUT = Path(__file__).resolve().parent/"timing"
OUT.mkdir(exist_ok=True)

PARA = ("This handwriting robot was designed to identify the words and writer of any "
        "piece of handwritten text, then reproduce new sentences in that same persons "
        "style using a gantry and pen. Over 13 different writers were used to train and "
        "test the system, each contributing several pages so the network could learn how "
        "they loop their letters, slant their lines and space their words.")

WRITERS = ["150", "152", "153", "384", "552", "588",
           "abhinav", "dylan", "owen", "thiya", "yeukita"]

#the feeds the log says were in force
F_DRAW = 800.0      #mm/min, pen down
F_TRAVEL = 4000.0   #mm/min, pen up
PEN_PULSE = 0.35    #s per M3 or M5

MOVE = re.compile(r"^(G0|G1)\b")
AXIS = re.compile(r"([XY])(-?\d+(?:\.\d+)?)")


def Measure(gcode):
    """pen-down mm, pen-up mm and pen pulses, straight out of the G-code"""
    down = up = 0.0
    pulses = 0
    x = y = 0.0
    pen = False
    for raw in gcode.splitlines():
        line = raw.split(";")[0].strip()
        if not line:
            continue
        if line.startswith("M3") or line.startswith("M5"):
            pen = line.startswith("M3")
            pulses += 1
            continue
        if not MOVE.match(line):
            continue
        nx, ny = x, y
        for ax, val in AXIS.findall(line):
            if ax == "X":
                nx = float(val)
            else:
                ny = float(val)
        d = ((nx-x)**2 + (ny-y)**2) ** 0.5
        if pen:
            down += d
        else:
            up += d
        x, y = nx, ny
    return down, up, pulses


c = SRV.app.test_client()
rows = []
for w in WRITERS:
    prof = w if w in SRV.PROFILES else (w+"_curated")
    if prof not in SRV.PROFILES:
        print("  %-9s no style profile, skipped" % w)
        continue
    t0 = time.time()
    r = c.post("/api/generate", json={"text": PARA, "author": prof, "nTries": 1})
    if r.status_code != 200:
        print("  %-9s generate failed: %s" % (w, (r.get_json() or {}).get("error")))
        continue
    jid = r.get_json()["job_id"] if "job_id" in r.get_json() else r.get_json().get("id")
    g = c.get("/api/gcode/%s" % jid)
    gcode = g.get_data(as_text=True)
    down, up, pulses = Measure(gcode)
    predDraw = down/F_DRAW*60.0 + up/F_TRAVEL*60.0 + pulses*PEN_PULSE
    rows.append({"writer": w, "profile": prof, "downMm": down, "upMm": up,
                 "pulses": pulses, "predDraw": predDraw,
                 "gcodeLines": len(gcode.splitlines()), "genS": time.time()-t0})
    print("  %-9s pen-down %7.0f mm   pen-up %7.0f mm   pulses %4d   "
          "predicted draw %6.1f s   (gen %.1f s)"
          % (w, down, up, pulses, predDraw, time.time()-t0), flush=True)

rows.sort(key=lambda r: -r["predDraw"])
print("\npredicted drawing time, longest first:")
for i, r in enumerate(rows, 1):
    print("  %2d. %-9s %7.1f s   (%.0f mm of ink)" % (i, r["writer"], r["predDraw"],
                                                      r["downMm"]))

MEASURED = [("session 9", 1022.8), ("session 8", 899.6), ("session 10", 839.3),
            ("session 13", 726.4), ("session 11", 663.0), ("session 12", 656.9),
            ("session 7", 651.2), ("session 3", 637.3), ("session 1", 497.5)]
print("\nmeasured drawing time, longest first (9 sessions with a drawn paragraph):")
for i, (s, t) in enumerate(MEASURED, 1):
    print("  %2d. %-11s %7.1f s" % (i, s, t))

(OUT/"predicted.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
print("\nwrote", OUT/"predicted.json")
