#!/bin/sh
# Build libaccel.so for Linux / the ODROID N2+ (ARM64), using the system's
# own native gcc. This is the command that actually matters for deployment
# -- run it ON the ODROID after copying accel.c over.
#
# Pure C99, no platform-specific intrinsics, no SIMD -- the exact same
# accel.c that build_windows.bat compiles into libaccel.dll.

set -e
cd "$(dirname "$0")"
gcc -O3 -shared -fPIC -o libaccel.so accel.c -lm
echo "Built $(pwd)/libaccel.so"
