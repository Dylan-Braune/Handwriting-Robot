#the gantry geometry measurements of the CALIBRATED machine.
#
#an earlier set of side lengths was taken while the travel constants were
#still being tuned.  those measured a machine that no longer exists and are
#not reported: the calibration that followed is part of the design, and
#qualifying the system against its own uncalibrated state would say nothing
#about the system being submitted.  the calibration procedure itself is
#described in the design section; what is qualified here is its outcome.
#
#the square gives the two scale factors: its horizontal sides measure the X
#axis and its vertical sides the Y axis.  it CANNOT see a squareness error,
#because a sheared square still has four equal sides.
#
#the triangle can.  it is isosceles by construction, so its two sloped sides
#are equal only if the axes are perpendicular.  if Y leans by phi the apex
#shifts sideways by d = h*tan(phi) and the two sloped sides split, which makes
#the difference between them a direct measurement of the skew.
import math

TARGET = 100.0
SQUARE = {"left": 100.1, "bottom": 100.3, "right": 99.9, "top": 100.2}
#from the G-code: bottom (45,76)->(145,76) and top run along X;
#left (45,76)->(45,176) and right run along Y
AXIS = {"bottom": "X", "top": "X", "left": "Y", "right": "Y"}

print("="*70)
print("SQUARE, nominal %.1f mm a side, calibrated machine, ONE specimen" % TARGET)
print("="*70)
print("%-10s %-7s %-10s %s" % ("side", "axis", "measured", "error"))
for s in ("bottom", "top", "left", "right"):
    print("%-10s %-7s %-10.1f %+.1f mm" % (s, AXIS[s], SQUARE[s], SQUARE[s]-TARGET))

err = [v-TARGET for v in SQUARE.values()]
ax = {}
for s, v in SQUARE.items():
    ax.setdefault(AXIS[s], []).append(v)
print("\n  mean signed error    %+.3f mm" % (sum(err)/len(err)))
print("  mean absolute error   %.3f mm" % (sum(abs(e) for e in err)/len(err)))
print("  worst side            %.1f mm" % max(abs(e) for e in err))
print("  range over the 4 sides %.1f mm   <- within ONE square, not repeatability"
      % (max(err)-min(err)))
for a in ("X", "Y"):
    m = sum(ax[a])/len(ax[a])
    print("  %s axis scale         %.2f mm = %+.3f %%  (from %s)"
          % (a, m, 100*(m-TARGET)/TARGET,
             " and ".join("%.1f" % v for v in ax[a])))

#---the triangle------------------------------------------------------------
BASE_N, H_N = 80.0, 40.0
SIDE_N = math.hypot(BASE_N/2, H_N)
base, right, left = 80.3, 56.9, 56.2

print("\n\n" + "="*70)
print("TRIANGLE, nominal base %.1f, height %.1f, sloped sides %.2f mm"
      % (BASE_N, H_N, SIDE_N))
print("="*70)
print("  base        %.1f mm   %+.1f mm   %+.3f %%"
      % (base, base-BASE_N, 100*(base-BASE_N)/BASE_N))
print("  right side  %.1f mm   %+.2f mm" % (right, right-SIDE_N))
print("  left side   %.1f mm   %+.2f mm" % (left, left-SIDE_N))
print("  sloped sides differ by %.2f mm, though they are equal by construction"
      % abs(right-left))

#   left^2  = (base/2 + d)^2 + h^2 ,  right^2 = (base/2 - d)^2 + h^2
#   =>  d = (left^2 - right^2) / (2*base)
d = (left**2 - right**2)/(2*base)
phi = math.degrees(math.atan2(d, H_N))
pl, pr = math.hypot(BASE_N/2 + d, H_N), math.hypot(BASE_N/2 - d, H_N)
print("\n  apex offset        d   = %+.3f mm" % d)
print("  squareness error   phi = %+.3f degrees" % phi)
print("\n  one free parameter, two measurements, so the fit is testable:")
print("    left  predicted %.2f   measured %.1f   residual %+.2f mm" % (pl, left, pl-left))
print("    right predicted %.2f   measured %.1f   residual %+.2f mm" % (pr, right, pr-right))

print("\n  the X scale from the square (%+.3f %%) predicts a base of %.2f mm;"
      % (100*(sum(ax['X'])/2-TARGET)/TARGET, BASE_N*(sum(ax['X'])/2)/TARGET))
print("  the base measured %.1f mm, so the two figures agree to %.2f mm."
      % (base, abs(base-BASE_N*(sum(ax['X'])/2)/TARGET)))

d1 = math.hypot(100.0 + 100.0*math.tan(math.radians(abs(phi))), 100.0)
d2 = math.hypot(100.0 - 100.0*math.tan(math.radians(abs(phi))), 100.0)
print("\n  the same skew would split the square's diagonals to %.2f and %.2f mm"
      % (d1, d2))
print("  against %.2f nominal, a %.2f mm difference -- an independent check"
      % (math.hypot(100, 100), abs(d1-d2)))

print("\n\n" + "="*70)
print("WHAT THESE TWO SPECIMENS DO AND DO NOT SUPPORT")
print("="*70)
print("  n = 1 square and n = 1 triangle.  that is enough to state the")
print("  measured scale errors and to solve the skew, because the skew fit")
print("  is over-determined and its residuals are 0.02 mm.")
print("  it is NOT enough for repeatability, a standard deviation, or a")
print("  tolerance claim: the 0.4 mm range above is the spread between the")
print("  four sides of one square, which is form error plus reading error,")
print("  not the variation between repeated draws.  repeated specimens are")
print("  needed before any 'the gantry holds X mm' claim can be made.")
