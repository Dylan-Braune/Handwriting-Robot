; 80 mm base, 45 degree triangle for checking the gantry (iso)
; usable area taken as 189.2 x 250.3 mm (from the homing calibration)
;
; vertices, in machine millimetres:
;   (55.00, 105.00)
;   (135.00, 105.00)
;   (95.00, 145.00)
;
; what it should measure on the paper:
;   base        80.00 mm   (along X, vertex 1 to 2)
;   height      40.00 mm   (along Y, above the base)
;   side 1      80.00 mm   angle at vertex 1   45.00 deg
;   side 2      56.57 mm   angle at vertex 2   45.00 deg
;   side 3      56.57 mm   angle at vertex 3   90.00 deg
;   margin left     55.0 mm
;   margin right    54.2 mm
;   margin bottom  105.0 mm
;   margin top     105.3 mm
;
; steps/mm X=80.000 Y=80.000, so one step is 0.0125 mm
; the base of 80.0 mm is 6400 X steps, the height 3200 Y steps
;
; the sloped side is the measurement a square cannot make.  at 45
; degrees Bresenham steps X and Y the same number of times, so if
; the two steps/mm are not equal the apex misses by the difference:
; 1 percent out between the axes puts the apex 0.40 mm off.
; about 1.40 mm of any error is the known 1.0 degree squareness
; fault in the rails, which is mechanical and not a calibration
; number -- see SQUARENESS_ERROR_DEG in reproduce.py.
G21   ; millimetres
G90   ; absolute
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
G0 X0.000 Y250.300   ; park
M2
