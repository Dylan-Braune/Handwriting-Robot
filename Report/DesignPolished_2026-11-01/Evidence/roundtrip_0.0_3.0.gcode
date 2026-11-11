; Handwriting-Robot G-code
; author=fixture strokes=2 scale=1.000
; steps/mm  X=80.000  Y=80.000 (3200 microsteps/rev)
; pen: one-way gear motor, M5=up M3=down, each = one 90deg pulse
G21   ; millimetres
G90   ; absolute
G0 F2400
G0 F2400 X15.000 Y224.263
M3   ; pen DOWN (one 90deg pulse of the one-way gear motor)
G4 P0.350   ; wait for the 90deg rotation to finish
G1 F900
G1 X27.981 Y224.263
G1 X37.963 Y233.913
M5   ; pen UP (one 90deg pulse of the one-way gear motor)
G4 P0.350   ; wait for the 90deg rotation to finish
G0 F2400 X57.370 Y231.560
M3   ; pen DOWN (one 90deg pulse of the one-way gear motor)
G4 P0.350   ; wait for the 90deg rotation to finish
G1 F900
G1 X63.000 Y241.300
M5   ; pen UP (one 90deg pulse of the one-way gear motor)
G4 P0.350   ; wait for the 90deg rotation to finish
G0 F2400 X0.000 Y251.300   ; park
M2
