; 100 x 100 mm square for checking the gantry
; usable area taken as 190.6 x 251.3 mm (from the homing calibration)
;
; corners, in machine millimetres:
;   (45.0, 76.0)
;   (145.0, 76.0)
;   (145.0, 176.0)
;   (45.0, 176.0)
;
; what it should measure on the paper:
;   each side              100.0 mm
;   diagonal corner to corner 141.42 mm
;   margin left            45.0 mm
;   margin right           45.6 mm
;   margin bottom          76.0 mm
;   margin top             75.3 mm
; planning default steps/mm X=80.000 Y=80.000 (homing overrides this)
; travel constants in force: X=200.6 Y=261.3 mm
;   both diagonals should be 141.42 mm.  they are the ONLY way to
;   see the 1.0 deg axis skew, which leaves all four sides
;   equal but splits the diagonals to 140.18 and 142.65 mm,
;   a difference of 2.47 mm
G21   ; millimetres
G90   ; absolute
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
G0 X10.000 Y241.300   ; park, kept 10 mm clear of the limits
M2
