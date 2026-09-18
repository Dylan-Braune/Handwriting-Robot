; 20mm diameter circle (10mm radius), true G2 arc command -- the standard
; way to express a circle in G-code (single command, not a polyline).
; Centre placed at (10, 10) so the whole circle stays in positive
; coordinates -- travels to the start point (20, 10) with the pen up,
; puts the pen down, draws the full circle in one G2 (start point == end
; point is the standard convention for a full circle), pen back up, done.
G21           ; millimetres
G90           ; absolute positioning
M5            ; pen up
G0 X20 Y10    ; travel to the start point on the circle (3 o'clock position)
M3            ; pen down
G2 X20 Y10 I-10 J0 F600   ; full circle: centre is 10mm to the left (I-10 J0)
M5            ; pen up
G0 X0 Y0      ; park
M2            ; program end
