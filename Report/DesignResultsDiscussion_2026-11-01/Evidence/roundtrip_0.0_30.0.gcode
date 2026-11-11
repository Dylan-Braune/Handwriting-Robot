; Handwriting-Robot G-code
; author=fixture strokes=2 scale=0.335
; steps/mm  X=80.000  Y=80.000 (3200 microsteps/rev)
; pen: one-way gear motor, M5=up M3=down, each = one 90deg pulse
G21   ; millimetres
G90   ; absolute
G0 F2400
G0 F2400 X15.000 Y184.298
M3   ; pen DOWN (one 90deg pulse of the one-way gear motor)
G4 P0.350   ; wait for the 90deg rotation to finish
G1 F900
G1 X58.433 Y184.298
G1 X91.830 Y216.586
M5   ; pen UP (one 90deg pulse of the one-way gear motor)
G4 P0.350   ; wait for the 90deg rotation to finish
G0 F2400 X156.764 Y208.710
M3   ; pen DOWN (one 90deg pulse of the one-way gear motor)
G4 P0.350   ; wait for the 90deg rotation to finish
G1 F900
G1 X175.600 Y241.300
M5   ; pen UP (one 90deg pulse of the one-way gear motor)
G4 P0.350   ; wait for the 90deg rotation to finish
G0 F2400 X0.000 Y251.300   ; park
M2
