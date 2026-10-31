; 100 mm square with the 45 degree isosceles triangle inside it
; both figures in one run, so the gaps between them can be measured too
; usable area taken as 190.6 x 251.3 mm (from the homing calibration)
;
; SQUARE, in machine millimetres:
;   (45.00, 76.00)
;   (145.00, 76.00)
;   (145.00, 176.00)
;   (45.00, 176.00)
;   each side              100.00 mm
;   diagonal corner to corner 141.42 mm
;   both diagonals should be 141.42 mm.  they are the ONLY way to
;   see the 1.0 deg axis skew, which leaves all four sides equal
;   but splits the diagonals to 140.18 and 142.65 mm, 2.47 mm apart
;
; TRIANGLE, in machine millimetres:
;   (55.00, 105.00)
;   (135.00, 105.00)
;   (95.00, 145.00)
;   base        80.00 mm   (along X)
;   height      40.00 mm   (along Y)
;   side 2      56.57 mm   at 45 deg
;   side 3      56.57 mm   at 45 deg
;   the sloped sides are the measurement a square cannot make: at
;   45 deg Bresenham steps X and Y the same number of times, so a
;   difference between the two steps/mm moves the apex sideways.
;   1 percent out between the axes puts the apex 0.40 mm off, and
;   about 1.40 mm of any error is the known 1.0 deg squareness
;   fault in the rails -- see SQUARENESS_ERROR_DEG in reproduce.py
;
; GAPS between the two, which only a combined sheet can give:
;   square bottom to triangle base      29.00 mm
;   square left to triangle vertex 1    10.00 mm
;   square right to triangle vertex 2   10.00 mm
;   triangle apex to square top         31.00 mm
;   the base must also be parallel to the square's bottom side:
;   any wedge between them is the axis skew shown directly
;
; travel constants in force: X=200.6 Y=261.3 mm
; planning default steps/mm X=80.000 Y=80.000 (homing overrides this)
G21   ; millimetres
G90   ; absolute
;
; --- square, 100 x 100 mm ---
G0 F2400
G0 X45.000 Y76.000
M3   ; pen DOWN (one 90deg pulse of the one-way gear motor)
G4 P0.350   ; wait for the 90deg rotation to finish
G1 F900
G1 X145.000 Y76.000
G1 X145.000 Y176.000
G1 X45.000 Y176.000
G1 X45.000 Y76.000
M5   ; pen UP (one 90deg pulse of the one-way gear motor)
G4 P0.350   ; wait for the 90deg rotation to finish
;
; --- triangle, 80 mm base, 45 deg ---
G0 F2400
G0 X55.000 Y105.000
M3   ; pen DOWN (one 90deg pulse of the one-way gear motor)
G4 P0.350   ; wait for the 90deg rotation to finish
G1 F900
G1 X135.000 Y105.000
G1 X95.000 Y145.000
G1 X55.000 Y105.000
M5   ; pen UP (one 90deg pulse of the one-way gear motor)
G4 P0.350   ; wait for the 90deg rotation to finish
;
G0 F2400
G0 X10.000 Y241.300   ; park, kept 10 mm clear of the limits
M2
