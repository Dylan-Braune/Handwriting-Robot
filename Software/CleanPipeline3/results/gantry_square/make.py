"""A 100 x 100 mm square, written as literal coordinates, for checking the machine.

Nothing here goes through ToMachine or the trajectory placement: the numbers in
the file are the numbers asked for, so if the drawn square measures something
else the error is the machine's, not the software's.

    python make.py [--usable-w 201.2] [--usable-h 260.0] [--size 100]
"""
import argparse, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
import reproduce as REP


def Square(usableW, usableH, size, out, feedDraw=None, feedTravel=None):
    cfg = REP.GantryConfig()
    feedDraw = feedDraw or cfg.drawFeedMmMin
    feedTravel = feedTravel or cfg.travelFeedMmMin

    #centred on round millimetres, which is easier to measure than a true centre
    x0 = round((usableW-size)/2.0)
    y0 = round((usableH-size)/2.0)
    x1, y1 = x0+size, y0+size
    corners = [(x0,y0),(x1,y0),(x1,y1),(x0,y1),(x0,y0)]

    L = []
    L.append("; %d x %d mm square for checking the gantry" % (size,size))
    L.append("; usable area taken as %.1f x %.1f mm (from the homing calibration)"
             % (usableW,usableH))
    L.append(";")
    L.append("; corners, in machine millimetres:")
    for (x,y) in corners[:4]:
        L.append(";   (%.1f, %.1f)" % (x,y))
    L.append(";")
    L.append("; what it should measure on the paper:")
    L.append(";   each side              %.1f mm" % size)
    L.append(";   diagonal corner to corner %.2f mm" % ((2*size**2)**0.5))
    L.append(";   margin left            %.1f mm" % x0)
    L.append(";   margin right           %.1f mm" % (usableW-x1))
    L.append(";   margin bottom          %.1f mm" % y0)
    L.append(";   margin top             %.1f mm" % (usableH-y1))
    L.append("; steps/mm X=%.3f Y=%.3f, so one step is %.4f mm"
             % (cfg.stepsPerMmX,cfg.stepsPerMmY,1.0/cfg.stepsPerMmX))
    L.append("; a side of %.1f mm is %.0f steps" % (size,size*cfg.stepsPerMmX))
    L.append("G21   ; millimetres")
    L.append("G90   ; absolute")
    pen = REP.NewPenState(cfg)
    L += REP.SetPenLines(pen,cfg,True)
    L.append("G0 F%.0f" % feedTravel)
    L.append("G0 X%.3f Y%.3f" % corners[0])
    L += REP.SetPenLines(pen,cfg,False)
    L.append("G1 F%.0f" % feedDraw)
    for (x,y) in corners[1:]:
        L.append("G1 X%.3f Y%.3f" % (x,y))
    L += REP.SetPenLines(pen,cfg,True)
    L.append("G0 X%.3f Y%.3f   ; park" % (cfg.originXmm,cfg.originYmm))
    L.append("M2")
    Path(out).write_text("\n".join(L)+"\n", encoding="utf-8")
    return dict(corners=corners[:4], size=size,
                marginLeft=x0, marginRight=usableW-x1,
                marginBottom=y0, marginTop=usableH-y1,
                diagonal=(2*size**2)**0.5, gcodeLines=len(L), penPulses=pen[1])


def Main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--usable-w", type=float, default=201.2, dest="w")
    ap.add_argument("--usable-h", type=float, default=260.0, dest="h")
    ap.add_argument("--size", type=float, default=100.0)
    ap.add_argument("-o","--out", default=str(HERE/"square_100mm.gcode"))
    a = ap.parse_args()
    r = Square(a.w, a.h, a.size, a.out)
    print("wrote %s" % a.out)
    print("  square   : %.0f x %.0f mm, corners %s" % (r["size"],r["size"],r["corners"]))
    print("  margins  : left %.1f, right %.1f, bottom %.1f, top %.1f mm"
          % (r["marginLeft"],r["marginRight"],r["marginBottom"],r["marginTop"]))
    print("  diagonal : %.2f mm" % r["diagonal"])
    print("  file     : %d lines, %d pen pulses" % (r["gcodeLines"],r["penPulses"]))


if __name__ == "__main__":
    Main()
