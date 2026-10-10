"""Crop a strip out of a rendered sentence so a specific word can be inspected
against the text it was meant to be.

    python zoom_render.py yeukita_goodmorning_gcode.png 620 900
"""
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
p = Path(sys.argv[1])
if not p.is_absolute():
    p = HERE / "out" / "sentences" / p
x0, x1 = int(sys.argv[2]), int(sys.argv[3])
scale = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0
im = Image.open(p)
c = im.crop((x0, 0, x1, im.height))
c = c.resize((int(c.width * scale), int(c.height * scale)), Image.LANCZOS)
out = p.with_name(p.stem + f"_z{x0}_{x1}.png")
c.save(out)
print("wrote", out, c.size)
