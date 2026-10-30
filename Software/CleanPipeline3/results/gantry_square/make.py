"""A 100 x 100 mm square, written as literal coordinates, for checking the machine.

Nothing here goes through ToMachine or the trajectory placement: the numbers in
the file are the numbers asked for, so if the drawn square measures something
else the error is the machine's, not the software's.

    python make.py [--usable-w 201.2] [--usable-h 260.0] [--size 100]
"""
import argparse, math, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
import reproduce as REP


def _Diagonals(size, skewDeg):
    """The two diagonals of a square of this side whose axes are skewDeg off
    square.  Equal at zero skew; the gap is what a caliper can actually see."""
    c = math.cos(math.radians(90.0-skewDeg))
    return (math.sqrt(2*size*size-2*size*size*c),
            math.sqrt(2*size*size+2*size*size*c))


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
    #NOT the steps/mm that will be used: Calibrate() recomputes it by homing,
    #as counted steps / X_SWITCH_TRAVEL_MM.  this is the planning default only
    L.append("; planning default steps/mm X=%.3f Y=%.3f (homing overrides this)"
             % (cfg.stepsPerMmX,cfg.stepsPerMmY))
    L.append("; travel constants in force: X=%.1f Y=%.1f mm"
             % (REP.X_SWITCH_TRAVEL_MM,REP.Y_SWITCH_TRAVEL_MM))
    dShort, dLong = _Diagonals(size, REP.SQUARENESS_ERROR_DEG)
    L.append(";   both diagonals should be %.2f mm.  they are the ONLY way to"
             % ((2*size**2)**0.5))
    L.append(";   see the %.1f deg axis skew, which leaves all four sides"
             % REP.SQUARENESS_ERROR_DEG)
    L.append(";   equal but splits the diagonals to %.2f and %.2f mm,"
             % (dShort,dLong))
    L.append(";   a difference of %.2f mm" % (dLong-dShort))
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
    #GantryConfig parks at originYmm, which defaults to the TOP of usable
    #travel.  driving hard onto the limit at the end of every run risks
    #tripping Y_MAX and firing the emergency stop, so stand 10 mm off it
    parkX, parkY = 10.0, usableH-10.0
    L.append("G0 X%.3f Y%.3f   ; park, kept 10 mm clear of the limits"
             % (parkX,parkY))
    L.append("M2")
    Path(out).write_text("\n".join(L)+"\n", encoding="utf-8")
    return dict(corners=corners[:4], size=size,
                marginLeft=x0, marginRight=usableW-x1,
                marginBottom=y0, marginTop=usableH-y1,
                diagonal=(2*size**2)**0.5, gcodeLines=len(L), penPulses=pen[1])


def Main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    #default to what the machine will actually believe at run time, so the
    #header and the margins cannot go stale against reproduce.py again
    ap.add_argument("--usable-w", type=float, dest="w",
                    default=REP.X_SWITCH_TRAVEL_MM-2*REP.EDGE_TOLERANCE_MM)
    ap.add_argument("--usable-h", type=float, dest="h",
                    default=REP.Y_SWITCH_TRAVEL_MM-2*REP.EDGE_TOLERANCE_MM)
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
