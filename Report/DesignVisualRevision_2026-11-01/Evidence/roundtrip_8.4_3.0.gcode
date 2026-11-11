; Handwriting-Robot G-code
; author=fixture strokes=2 scale=1.000
; steps/mm  X=80.000  Y=80.000 (3200 microsteps/rev)
; pen: one-way gear motor, M5=up M3=down, each = one 90deg pulse
G21   ; millimetres
G90   ; absolute
G0 F2400
G0 F2400 X15.000 Y215.863
M3   ; pen DOWN (one 90deg pulse of the one-way gear motor)
G4 P0.350   ; wait for the 90deg rotation to finish
G1 F900
G1 X27.981 Y215.863
G1 X37.963 Y225.513
M5   ; pen UP (one 90deg pulse of the one-way gear motor)
G4 P0.350   ; wait for the 90deg rotation to finish
G0 F2400 X57.370 Y223.160
M3   ; pen DOWN (one 90deg pulse of the one-way gear motor)
G4 P0.350   ; wait for the 90deg rotation to finish
G1 F900
G1 X63.000 Y232.900
M5   ; pen UP (one 90deg pulse of the one-way gear motor)
G4 P0.350   ; wait for the 90deg rotation to finish
G0 F2400 X0.000 Y251.300   ; park
M2
