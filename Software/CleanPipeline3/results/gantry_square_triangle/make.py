#u23539195 Dylan Braune.  the 100 mm square and the 45 degree isosceles
#triangle on one sheet, drawn in one run.
#
#   python make.py
#
#the two were separate tests and each costs a sheet, a homing cycle and a pen
#change to run.  they measure different faults and neither replaces the other:
#the square gives the two scale factors and, through its diagonals, the axis
#skew; the triangle gives the sloped side, which is the one measurement a
#square cannot make, because at 45 degrees Bresenham steps X and Y the same
#number of times and any difference between the two steps/mm moves the apex.
#
#drawing both together also adds a measurement neither has on its own.  the
#triangle sits inside the square, so the gap from each triangle vertex to the
#square side beside it is a fixed number that can be checked with the same
#callipers, and those gaps are only right if BOTH figures came out right in
#the same run.  two separately drawn sheets cannot show that.
#
#geometry, unchanged from the two originals:
#   square    (45,76) (145,76) (145,176) (45,176)
#   triangle  (55,105) (135,105) (95,145)
#the triangle is wholly inside the square, so the pen never crosses a line it
#has already drawn and no join is ambiguous when it is measured.
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

import reproduce as REP

OUT = HERE/"square_triangle_100mm_80mm.gcode"

#---the two figures, as the separate makers place them---
SQUARE = [(45.0, 76.0), (145.0, 76.0), (145.0, 176.0), (45.0, 176.0)]
TRIANGLE = [(55.0, 105.0), (135.0, 105.0), (95.0, 145.0)]

TRAVEL_F = 2400     #pen up, as both originals use
DRAW_F = 900        #pen down
PEN_WAIT = 0.350    #the 90 deg pulse of the one-way gear motor


def Dist(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5


def Closed(pts):
    return list(pts)+[pts[0]]


def Figure(lines, pts, name):
    """one closed outline: travel to the start, pen down, draw, pen up"""
    path = Closed(pts)
    lines.append(";")
    lines.append("; --- %s ---" % name)
    lines.append("G0 F%d" % TRAVEL_F)
    lines.append("G0 X%.3f Y%.3f" % path[0])
    lines.append("M3   ; pen DOWN (one 90deg pulse of the one-way gear motor)")
    lines.append("G4 P%.3f   ; wait for the 90deg rotation to finish" % PEN_WAIT)
    lines.append("G1 F%d" % DRAW_F)
    for p in path[1:]:
        lines.append("G1 X%.3f Y%.3f" % p)
    lines.append("M5   ; pen UP (one 90deg pulse of the one-way gear motor)")
    lines.append("G4 P%.3f   ; wait for the 90deg rotation to finish" % PEN_WAIT)


def Main():
    usableW = REP.X_SWITCH_TRAVEL_MM - 2*REP.EDGE_TOLERANCE_MM
    usableH = REP.Y_SWITCH_TRAVEL_MM - 2*REP.EDGE_TOLERANCE_MM

    #the gaps between the two figures.  these are what a combined sheet can be
    #measured for and two separate sheets cannot
    gapBottom = TRIANGLE[0][1]-SQUARE[0][1]
    gapLeft = TRIANGLE[0][0]-SQUARE[0][0]
    gapRight = SQUARE[1][0]-TRIANGLE[1][0]
    gapApexTop = SQUARE[2][1]-TRIANGLE[2][1]

    L = []
    L.append("; 100 mm square with the 45 degree isosceles triangle inside it")
    L.append("; both figures in one run, so the gaps between them can be measured too")
    L.append("; usable area taken as %.1f x %.1f mm (from the homing calibration)"
             % (usableW, usableH))
    L.append(";")
    L.append("; SQUARE, in machine millimetres:")
    for p in SQUARE:
        L.append(";   (%.2f, %.2f)" % p)
    L.append(";   each side              100.00 mm")
    L.append(";   diagonal corner to corner %.2f mm" % Dist(SQUARE[0], SQUARE[2]))
    L.append(";   both diagonals should be %.2f mm.  they are the ONLY way to"
             % Dist(SQUARE[0], SQUARE[2]))
    L.append(";   see the 1.0 deg axis skew, which leaves all four sides equal")
    L.append(";   but splits the diagonals to 140.18 and 142.65 mm, 2.47 mm apart")
    L.append(";")
    L.append("; TRIANGLE, in machine millimetres:")
    for p in TRIANGLE:
        L.append(";   (%.2f, %.2f)" % p)
    L.append(";   base        %.2f mm   (along X)" % Dist(TRIANGLE[0], TRIANGLE[1]))
    L.append(";   height      %.2f mm   (along Y)" % (TRIANGLE[2][1]-TRIANGLE[0][1]))
    L.append(";   side 2      %.2f mm   at 45 deg" % Dist(TRIANGLE[1], TRIANGLE[2]))
    L.append(";   side 3      %.2f mm   at 45 deg" % Dist(TRIANGLE[2], TRIANGLE[0]))
    L.append(";   the sloped sides are the measurement a square cannot make: at")
    L.append(";   45 deg Bresenham steps X and Y the same number of times, so a")
    L.append(";   difference between the two steps/mm moves the apex sideways.")
    L.append(";   1 percent out between the axes puts the apex 0.40 mm off, and")
    L.append(";   about 1.40 mm of any error is the known 1.0 deg squareness")
    L.append(";   fault in the rails -- see SQUARENESS_ERROR_DEG in reproduce.py")
    L.append(";")
    L.append("; GAPS between the two, which only a combined sheet can give:")
    L.append(";   square bottom to triangle base      %.2f mm" % gapBottom)
    L.append(";   square left to triangle vertex 1    %.2f mm" % gapLeft)
    L.append(";   square right to triangle vertex 2   %.2f mm" % gapRight)
    L.append(";   triangle apex to square top         %.2f mm" % gapApexTop)
    L.append(";   the base must also be parallel to the square's bottom side:")
    L.append(";   any wedge between them is the axis skew shown directly")
    L.append(";")
    L.append("; travel constants in force: X=%.1f Y=%.1f mm"
             % (REP.X_SWITCH_TRAVEL_MM, REP.Y_SWITCH_TRAVEL_MM))
    L.append("; planning default steps/mm X=80.000 Y=80.000 (homing overrides this)")
    L.append("G21   ; millimetres")
    L.append("G90   ; absolute")

    #square first, then the triangle inside it, so the pen never has to cross
    #a line it has already laid down
    Figure(L, SQUARE, "square, 100 x 100 mm")
    Figure(L, TRIANGLE, "triangle, 80 mm base, 45 deg")

    L.append(";")
    L.append("G0 F%d" % TRAVEL_F)
    L.append("G0 X10.000 Y%.3f   ; park, kept 10 mm clear of the limits"
             % (usableH-10.0))
    L.append("M2")

    OUT.write_text("\n".join(L)+"\n", encoding="utf-8")
    print("wrote %s (%d lines)" % (OUT.name, len(L)))
    for p in SQUARE+TRIANGLE:
        if not (0 <= p[0] <= usableW and 0 <= p[1] <= usableH):
            print("  WARNING: (%.1f, %.1f) is outside the usable area" % p)
    print("  square   X %.0f..%.0f  Y %.0f..%.0f"
          % (SQUARE[0][0], SQUARE[1][0], SQUARE[0][1], SQUARE[2][1]))
    print("  triangle X %.0f..%.0f  Y %.0f..%.0f"
          % (TRIANGLE[0][0], TRIANGLE[1][0], TRIANGLE[0][1], TRIANGLE[2][1]))
    print("  gaps: bottom %.1f  left %.1f  right %.1f  apex-to-top %.1f mm"
          % (gapBottom, gapLeft, gapRight, gapApexTop))


if __name__ == "__main__":
    Main()
