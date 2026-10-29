"""An 80 mm base, 45 degree triangle, as literal coordinates, for checking the machine.

Nothing here goes through ToMachine or the trajectory placement: the numbers in
the file are the numbers asked for, so if the drawn triangle measures something
else the error is the machine's, not the software's.  Same arrangement as
gantry_square/make.py next door.

Why a triangle as well as a square.  A square only ever drives one axis at a
time, so it measures the two scales and nothing about how they combine.  The
hypotenuse here is driven at 45 degrees, where Bresenham steps X and Y the same
number of times, so a mismatch between the two steps/mm figures shows up
directly as the apex not landing where it should.  The square cannot see that
error at all.

    python make.py [--base 80] [--angle 45] [--shape right|iso]

--shape right (the default) is the 45 degree set square: the base runs along X,
the right angle is at its right-hand end, and the opposite side runs up Y, so
base and height are both 80 mm and each is measurable on its own axis.

--shape iso reads the base as the longest side instead: both base angles are 45
degrees, the apex is a right angle over the middle of the base, and the height
is 40 mm.
"""
import argparse, math, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
import reproduce as REP


#---Corners: the three vertices, in machine millimetres---
#returned closed (first point repeated last) so the drawing loop does not have
#to know it is drawing a closed shape
def Corners(usableW, usableH, base, angleDeg, shape):
    t = math.tan(math.radians(angleDeg))
    if shape == "iso":
        #base is the longest side, apex over its midpoint
        height = (base/2.0)*t
        width = base
    else:
        #right angle at the right-hand end of the base, opposite side up Y
        height = base*t
        width = base

    #centred on round millimetres, which is easier to measure than a true centre
    x0 = round((usableW-width)/2.0)
    y0 = round((usableH-height)/2.0)
    if shape == "iso":
        pts = [(x0,y0), (x0+base,y0), (x0+base/2.0,y0+height)]
    else:
        pts = [(x0,y0), (x0+base,y0), (x0+base,y0+height)]
    return pts+[pts[0]], height, width, x0, y0


def Triangle(usableW, usableH, base, angleDeg, shape, out,
             feedDraw=None, feedTravel=None):
    cfg = REP.GantryConfig()
    feedDraw = feedDraw or cfg.drawFeedMmMin
    feedTravel = feedTravel or cfg.travelFeedMmMin

    closed, height, width, x0, y0 = Corners(usableW, usableH, base, angleDeg, shape)
    pts = closed[:3]
    #side lengths, measured off the coordinates rather than worked out again
    #from the angle, so what is printed is what the file actually draws
    sides = [math.dist(closed[i], closed[i+1]) for i in range(3)]
    #interior angles from the side lengths, by the cosine rule.  printing the
    #angles the FILE describes rather than the ones asked for is the point: a
    #rounded vertex would show here as an angle that is not 45
    angles = []
    for i in range(3):
        #the angle AT vertex i sits between the two sides that meet there,
        #sides[i] and sides[i-1], and is opposite the third one.  getting this
        #rotation wrong prints the right angle against the wrong vertex, which
        #is worse than not printing it at all when someone is holding a
        #protractor against the paper
        adj1, adj2, opp = sides[i], sides[(i+2)%3], sides[(i+1)%3]
        angles.append(math.degrees(math.acos(max(-1.0, min(1.0,
            (adj1*adj1+adj2*adj2-opp*opp)/(2*adj1*adj2))))))

    L = []
    L.append("; %.0f mm base, %.0f degree triangle for checking the gantry (%s)"
             % (base, angleDeg, shape))
    L.append("; usable area taken as %.1f x %.1f mm (from the homing calibration)"
             % (usableW, usableH))
    L.append(";")
    L.append("; vertices, in machine millimetres:")
    for (x,y) in pts:
        L.append(";   (%.2f, %.2f)" % (x,y))
    L.append(";")
    L.append("; what it should measure on the paper:")
    L.append(";   base      %7.2f mm   (along X, vertex 1 to 2)" % sides[0])
    L.append(";   height    %7.2f mm   (along Y, above the base)" % height)
    for i in range(3):
        L.append(";   side %d    %7.2f mm   angle at vertex %d  %6.2f deg"
                 % (i+1, sides[i], i+1, angles[i]))
    L.append(";   margin left   %6.1f mm" % x0)
    L.append(";   margin right  %6.1f mm" % (usableW-(x0+width)))
    L.append(";   margin bottom %6.1f mm" % y0)
    L.append(";   margin top    %6.1f mm" % (usableH-(y0+height)))
    L.append(";")
    L.append("; steps/mm X=%.3f Y=%.3f, so one step is %.4f mm"
             % (cfg.stepsPerMmX, cfg.stepsPerMmY, 1.0/cfg.stepsPerMmX))
    L.append("; the base of %.1f mm is %.0f X steps, the height %.0f Y steps"
             % (base, base*cfg.stepsPerMmX, height*cfg.stepsPerMmY))
    L.append(";")
    L.append("; the sloped side is the measurement a square cannot make.  at 45")
    L.append("; degrees Bresenham steps X and Y the same number of times, so if")
    L.append("; the two steps/mm are not equal the apex misses by the difference:")
    L.append("; 1 percent out between the axes puts the apex %.2f mm off."
             % (0.01*height))
    L.append("; about %.2f mm of any error is the known %.1f degree squareness"
             % (width*math.tan(math.radians(REP.SQUARENESS_ERROR_DEG)),
                REP.SQUARENESS_ERROR_DEG))
    L.append("; fault in the rails, which is mechanical and not a calibration")
    L.append("; number -- see SQUARENESS_ERROR_DEG in reproduce.py.")
    L.append("G21   ; millimetres")
    L.append("G90   ; absolute")
    pen = REP.NewPenState(cfg)
    L += REP.SetPenLines(pen, cfg, True)
    L.append("G0 F%.0f" % feedTravel)
    L.append("G0 X%.3f Y%.3f" % closed[0])
    L += REP.SetPenLines(pen, cfg, False)
    L.append("G1 F%.0f" % feedDraw)
    for (x,y) in closed[1:]:
        L.append("G1 X%.3f Y%.3f" % (x,y))
    L += REP.SetPenLines(pen, cfg, True)
    L.append("G0 X%.3f Y%.3f   ; park" % (cfg.originXmm, cfg.originYmm))
    L.append("M2")
    Path(out).write_text("\n".join(L)+"\n", encoding="utf-8")
    return dict(vertices=pts, base=sides[0], height=height, sides=sides,
                angles=angles, marginLeft=x0, marginRight=usableW-(x0+width),
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
    ap.add_argument("--base", type=float, default=80.0)
    ap.add_argument("--angle", type=float, default=45.0)
    ap.add_argument("--shape", choices=("right","iso"), default="right")
    ap.add_argument("-o","--out", default=None)
    a = ap.parse_args()
    out = a.out or str(HERE/("triangle_%.0fmm_%.0fdeg%s.gcode"
                             % (a.base, a.angle,
                                "" if a.shape=="right" else "_"+a.shape)))
    if a.angle<=0 or a.angle>=90:
        ap.error("--angle must be between 0 and 90 degrees exclusive")
    r = Triangle(a.w, a.h, a.base, a.angle, a.shape, out)
    tooWide = r["marginLeft"]<0 or r["marginRight"]<0
    tooTall = r["marginBottom"]<0 or r["marginTop"]<0
    print("wrote %s" % out)
    print("  triangle : base %.2f mm, height %.2f mm, %s"
          % (r["base"], r["height"], a.shape))
    print("  vertices : %s" % [("%.2f,%.2f"%p) for p in r["vertices"]])
    print("  sides    : %s mm" % ["%.2f"%s for s in r["sides"]])
    print("  angles   : %s deg" % ["%.2f"%v for v in r["angles"]])
    print("  margins  : left %.1f, right %.1f, bottom %.1f, top %.1f mm"
          % (r["marginLeft"],r["marginRight"],r["marginBottom"],r["marginTop"]))
    print("  file     : %d lines, %d pen pulses" % (r["gcodeLines"],r["penPulses"]))
    if tooWide or tooTall:
        print("  WARNING  : this does not fit the usable area -- a negative margin")
        print("             means the runner will clamp it and the shape drawn")
        print("             will not be the shape measured")


if __name__ == "__main__":
    Main()
