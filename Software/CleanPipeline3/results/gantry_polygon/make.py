"""A regular polygon of equal sides, as literal coordinates, for checking the machine.

Defaults to a pentagon of 80 mm sides, to match the triangle next door.  Nothing
here goes through ToMachine or the trajectory placement: the numbers in the file
are the numbers asked for, so if the drawn shape measures something else the
error is the machine's, not the software's.  Same arrangement as
gantry_square/make.py and gantry_triangle/make.py.

What this adds over the square and the triangle.  The square only ever drives
one axis at a time.  The triangle adds one diagonal, at 45 degrees, where X and
Y step the same number of times.  A pentagon has no two sides parallel and only
one of them axis-aligned, so every side is a different X:Y step ratio -- 1:3.08,
1:0.32 and so on.  That is where Bresenham's error term is actually exercised,
and it is the only one of the three shapes whose own geometry checks itself: all
five sides must measure the same and all five diagonals must measure the same,
so it can be judged without trusting a ruler's zero.

    python make.py [--sides 5] [--side 80] [--rotate 0]
"""
import argparse, math, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
import reproduce as REP

#---NAMES: what to call the file for the polygons anyone is likely to ask for---
NAMES = {3:"triangle", 4:"square", 5:"pentagon", 6:"hexagon", 7:"heptagon",
         8:"octagon", 9:"nonagon", 10:"decagon", 12:"dodecagon"}


#----------------------------------------------------------------------
#Vertices: the corners of a regular n-gon with a flat bottom edge
#the first vertex is the apex and the rest run anticlockwise.  starting the
#angles at 90 degrees puts one vertex straight up, which for an odd n leaves
#the bottom edge horizontal -- a pure X move, so there is one side on the
#page whose length depends on a single axis and can be checked on its own
#----------------------------------------------------------------------
def Vertices(n, side, rotateDeg, usableW, usableH):
    #circumradius and apothem from the side length
    R = side/(2.0*math.sin(math.pi/n))
    apothem = side/(2.0*math.tan(math.pi/n))
    raw = [(R*math.cos(math.radians(90.0+rotateDeg+k*360.0/n)),
            R*math.sin(math.radians(90.0+rotateDeg+k*360.0/n)))
           for k in range(n)]
    xs = [p[0] for p in raw]
    ys = [p[1] for p in raw]
    width, height = max(xs)-min(xs), max(ys)-min(ys)
    #centred on round millimetres, which is easier to measure than a true
    #centre.  the shape is placed by its BOUNDING BOX, not by its centroid:
    #a pentagon's centroid is not halfway up it
    x0 = round((usableW-width)/2.0)
    y0 = round((usableH-height)/2.0)
    pts = [(x0+(x-min(xs)), y0+(y-min(ys))) for (x,y) in raw]
    return pts, R, apothem, width, height, x0, y0


def Polygon(usableW, usableH, n, side, rotateDeg, out,
            feedDraw=None, feedTravel=None):
    cfg = REP.GantryConfig()
    feedDraw = feedDraw or cfg.drawFeedMmMin
    feedTravel = feedTravel or cfg.travelFeedMmMin

    pts, R, apothem, width, height, x0, y0 = Vertices(
        n, side, rotateDeg, usableW, usableH)
    closed = pts+[pts[0]]
    name = NAMES.get(n, "%d-gon" % n)

    #measured off the coordinates rather than worked out again from the side
    #length, so what is printed is what the file actually draws
    sides = [math.dist(closed[i], closed[i+1]) for i in range(n)]
    interior = 180.0-360.0/n
    #every distinct diagonal length, grouped.  for a pentagon there is one
    #length and five of them; for a hexagon there are two
    diags = {}
    for i in range(n):
        for j in range(i+2, n):
            if (i, j) == (0, n-1):
                continue   #that is a side, not a diagonal
            d = round(math.dist(pts[i], pts[j]), 3)
            diags[d] = diags.get(d, 0)+1

    L = []
    L.append("; regular %s, %.1f mm sides, for checking the gantry" % (name, side))
    L.append("; usable area taken as %.1f x %.1f mm (from the homing calibration)"
             % (usableW, usableH))
    L.append(";")
    L.append("; vertices, in machine millimetres (1 is the apex, then anticlockwise):")
    for k, (x, y) in enumerate(pts, 1):
        L.append(";   %2d  (%7.2f, %7.2f)" % (k, x, y))
    L.append(";")
    L.append("; what it should measure on the paper:")
    L.append(";   every side        %7.2f mm   (all %d the same)" % (side, n))
    L.append(";   every interior angle %6.2f deg" % interior)
    L.append(";   across the corners   %7.2f mm   (circumcircle diameter)" % (2*R))
    L.append(";   across the flats     %7.2f mm   (incircle diameter)" % (2*apothem))
    L.append(";   bounding box      %7.2f x %7.2f mm" % (width, height))
    for d in sorted(diags, reverse=True):
        L.append(";   diagonal          %7.2f mm   (x%d, all the same)"
                 % (d, diags[d]))
    L.append(";   margin left   %6.1f mm" % x0)
    L.append(";   margin right  %6.1f mm" % (usableW-(x0+width)))
    L.append(";   margin bottom %6.1f mm" % y0)
    L.append(";   margin top    %6.1f mm" % (usableH-(y0+height)))
    L.append(";")
    L.append("; steps/mm X=%.3f Y=%.3f, so one step is %.4f mm"
             % (cfg.stepsPerMmX, cfg.stepsPerMmY, 1.0/cfg.stepsPerMmX))
    L.append("; each side, as the step counts the two axes are given:")
    for i in range(n):
        dx = (closed[i+1][0]-closed[i][0])*cfg.stepsPerMmX
        dy = (closed[i+1][1]-closed[i][1])*cfg.stepsPerMmY
        ratio = ("Y only" if abs(dx) < 0.5 else
                 "X only" if abs(dy) < 0.5 else
                 "1 : %.2f" % abs(dy/dx))
        L.append(";   side %2d  dX %+8.0f  dY %+8.0f  steps, %s"
                 % (i+1, dx, dy, ratio))
    L.append(";")
    L.append("; this shape checks itself, which the square and the triangle")
    L.append("; cannot.  all %d sides must measure the same and all the" % n)
    L.append("; diagonals of a kind must measure the same, so a difference")
    L.append("; between them is a real fault and needs no trusted reference.")
    L.append("; a mismatch between the two steps/mm figures stretches the shape")
    L.append("; along one axis, so the sides stop being equal; the known %.1f"
             % REP.SQUARENESS_ERROR_DEG)
    L.append("; degree squareness fault in the rails shears it instead, which")
    L.append("; leaves the sides equal and makes the diagonals unequal.  the")
    L.append("; two faults are told apart by which of those two happens.")
    L.append("G21   ; millimetres")
    L.append("G90   ; absolute")
    pen = REP.NewPenState(cfg)
    L += REP.SetPenLines(pen, cfg, True)
    L.append("G0 F%.0f" % feedTravel)
    L.append("G0 X%.3f Y%.3f" % closed[0])
    L += REP.SetPenLines(pen, cfg, False)
    L.append("G1 F%.0f" % feedDraw)
    for (x, y) in closed[1:]:
        L.append("G1 X%.3f Y%.3f" % (x, y))
    L += REP.SetPenLines(pen, cfg, True)
    L.append("G0 X%.3f Y%.3f   ; park" % (cfg.originXmm, cfg.originYmm))
    L.append("M2")
    Path(out).write_text("\n".join(L)+"\n", encoding="utf-8")
    return dict(name=name, vertices=pts, sides=sides, interior=interior,
                circumDiameter=2*R, flatsDiameter=2*apothem,
                diagonals=diags, width=width, height=height,
                marginLeft=x0, marginRight=usableW-(x0+width),
                marginBottom=y0, marginTop=usableH-(y0+height),
                gcodeLines=len(L), penPulses=pen[1])


def Main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    #the defaults come from the homing calibration rather than being typed in
    #again, so they cannot go stale the way a copied pair of numbers does
    ap.add_argument("--usable-w", type=float,
                    default=REP.Calibration["usable_width_mm"], dest="w")
    ap.add_argument("--usable-h", type=float,
                    default=REP.Calibration["usable_height_mm"], dest="h")
    ap.add_argument("--sides", type=int, default=5)
    ap.add_argument("--side", type=float, default=80.0, dest="length")
    ap.add_argument("--rotate", type=float, default=0.0,
                    help="turn the whole shape, in degrees")
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()
    if a.sides < 3:
        ap.error("--sides must be at least 3")
    if a.length <= 0:
        ap.error("--side must be positive")
    out = a.out or str(HERE/("%s_%.0fmm.gcode"
                             % (NAMES.get(a.sides, "%dgon" % a.sides), a.length)))
    r = Polygon(a.w, a.h, a.sides, a.length, a.rotate, out)
    print("wrote %s" % out)
    print("  %-9s: %d sides of %.2f mm, interior angle %.2f deg"
          % (r["name"], a.sides, a.length, r["interior"]))
    print("  sides    : %s mm" % ["%.3f" % s for s in r["sides"]])
    print("  spread   : %.2f mm across the corners, %.2f across the flats"
          % (r["circumDiameter"], r["flatsDiameter"]))
    print("  diagonals: %s" % ["%.2f mm x%d" % (d, n)
                               for d, n in sorted(r["diagonals"].items(),
                                                  reverse=True)])
    print("  bbox     : %.2f x %.2f mm" % (r["width"], r["height"]))
    print("  margins  : left %.1f, right %.1f, bottom %.1f, top %.1f mm"
          % (r["marginLeft"], r["marginRight"], r["marginBottom"], r["marginTop"]))
    print("  file     : %d lines, %d pen pulses" % (r["gcodeLines"], r["penPulses"]))
    if min(r["marginLeft"], r["marginRight"],
           r["marginBottom"], r["marginTop"]) < 0:
        print("  WARNING  : this does not fit the usable area -- a negative margin")
        print("             means the runner will clamp it and the shape drawn")
        print("             will not be the shape measured")


if __name__ == "__main__":
    Main()
