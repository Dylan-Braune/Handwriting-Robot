; 100 x 100 mm square for checking the gantry
; usable area taken as 194.0 x 252.0 mm (from the homing calibration)
;
; corners, in machine millimetres:
;   (47.0, 76.0)
;   (147.0, 76.0)
;   (147.0, 176.0)
;   (47.0, 176.0)
;
; what it should measure on the paper:
;   each side              100.0 mm
;   diagonal corner to corner 141.42 mm
;   margin left            47.0 mm
;   margin right           47.0 mm
;   margin bottom          76.0 mm
;   margin top             76.0 mm
; steps/mm X=80.000 Y=80.000, so one step is 0.0125 mm
; a side of 100.0 mm is 8000 steps
G21   ; millimetres
G90   ; absolute
G0 F2400
G0 X47.000 Y76.000
M3   ; pen DOWN (one 90deg pulse of the one-way gear motor)
G4 P0.350   ; wait for the 90deg rotation to finish
G1 F900
G1 X147.000 Y76.000
G1 X147.000 Y176.000
G1 X47.000 Y176.000
G1 X47.000 Y76.000
M5   ; pen UP (one 90deg pulse of the one-way gear motor)
G4 P0.350   ; wait for the 90deg rotation to finish
G0 X10.000 Y180.000   ; park
M2
