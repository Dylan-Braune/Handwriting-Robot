#!/bin/sh
# Build the Linux accel.so from accel.c. This is the script that runs ON
# THE ODROID N2+ itself (native gcc, ARM64) -- do not cross-compile.
# Run from anywhere; builds into this same directory.

set -e
HERE="$(cd "$(dirname "$0")" && pwd)"

gcc -O3 -shared -fPIC -o "$HERE/libaccel.so" "$HERE/accel.c" -lm

echo "Built $HERE/libaccel.so"
