; regular pentagon, 80.0 mm sides, for checking the gantry
; usable area taken as 189.2 x 250.3 mm (from the homing calibration)
;
; vertices, in machine millimetres (1 is the apex, then anticlockwise):
;    1  (  94.72,  187.11)
;    2  (  30.00,  140.08)
;    3  (  54.72,   64.00)
;    4  ( 134.72,   64.00)
;    5  ( 159.44,  140.08)
;
; what it should measure on the paper:
;   every side          80.00 mm   (all 5 the same)
;   every interior angle 108.00 deg
;   across the corners    136.10 mm   (circumcircle diameter)
;   across the flats      110.11 mm   (incircle diameter)
;   bounding box       129.44 x  123.11 mm
;   diagonal           129.44 mm   (x5, all the same)
;   margin left     30.0 mm
;   margin right    29.8 mm
;   margin bottom   64.0 mm
;   margin top      63.2 mm
;
; steps/mm X=80.000 Y=80.000, so one step is 0.0125 mm
; each side, as the step counts the two axes are given:
;   side  1  dX    -5178  dY    -3762  steps, 1 : 0.73
;   side  2  dX    +1978  dY    -6087  steps, 1 : 3.08
;   side  3  dX    +6400  dY       -0  steps, X only
;   side  4  dX    +1978  dY    +6087  steps, 1 : 3.08
;   side  5  dX    -5178  dY    +3762  steps, 1 : 0.73
;
; this shape checks itself, which the square and the triangle
; cannot.  all 5 sides must measure the same and all the
; diagonals of a kind must measure the same, so a difference
; between them is a real fault and needs no trusted reference.
; a mismatch between the two steps/mm figures stretches the shape
; along one axis, so the sides stop being equal; the known 1.0
; degree squareness fault in the rails shears it instead, which
; leaves the sides equal and makes the diagonals unequal.  the
; two faults are told apart by which of those two happens.
G21   ; millimetres
G90   ; absolute
G0 F2400
G0 X94.721 Y187.107
M3   ; pen DOWN (one 90deg pulse of the one-way gear motor)
G4 P0.350   ; wait for the 90deg rotation to finish
G1 F900
G1 X30.000 Y140.085
G1 X54.721 Y64.000
G1 X134.721 Y64.000
G1 X159.443 Y140.085
G1 X94.721 Y187.107
M5   ; pen UP (one 90deg pulse of the one-way gear motor)
G4 P0.350   ; wait for the 90deg rotation to finish
G0 X0.000 Y250.300   ; park
M2
