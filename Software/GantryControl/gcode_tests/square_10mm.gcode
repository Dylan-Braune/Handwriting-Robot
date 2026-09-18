; 10mm square, plain G0/G1 -- the simplest possible test (no arcs)
G21
G90
M5
G0 X0 Y0
M3
G1 X10 Y0 F600
G1 X10 Y10
G1 X0 Y10
G1 X0 Y0
M5
G0 X0 Y0
M2
