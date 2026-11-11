; Handwriting-Robot G-code
; author=fixture strokes=2 scale=1.000
; steps/mm  X=80.000  Y=80.000 (3200 microsteps/rev)
; pen: one-way gear motor, M5=up M3=down, each = one 90deg pulse
G21   ; millimetres
G90   ; absolute
G0 F2400
G0 F2400 X15.000 Y227.221
M3   ; pen DOWN (one 90deg pulse of the one-way gear motor)
G4 P0.350   ; wait for the 90deg rotation to finish
G1 F900
G1 X19.327 Y227.221
G1 X22.654 Y230.438
M5   ; pen UP (one 90deg pulse of the one-way gear motor)
G4 P0.350   ; wait for the 90deg rotation to finish
G0 F2400 X29.123 Y229.653
M3   ; pen DOWN (one 90deg pulse of the one-way gear motor)
G4 P0.350   ; wait for the 90deg rotation to finish
G1 F900
G1 X31.000 Y232.900
M5   ; pen UP (one 90deg pulse of the one-way gear motor)
G4 P0.350   ; wait for the 90deg rotation to finish
G0 F2400 X0.000 Y251.300   ; park
M2
