"""Segments every PersonalDataset page with this folder's own SegmentLean.py
copy and writes the crops here (SegmentationBest/Crops/), generated fresh on
first run -- self-contained, nothing read from or written to NOGIT.
Run from anywhere: python SegmentationBest/run_full_test.py
"""
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
CNN_DIR = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
from SegmentLean import SegmentLines

AUTHORS = ["Abhinav", "Dylan", "Owen", "Thiya", "Yeukita"]
OUT_ROOT = THIS_DIR / "Crops"

pages = []
for author in AUTHORS:
    pdir = CNN_DIR / "PersonalDataset" / author / "Pages"
    for p in sorted(pdir.glob(f"{author}*.jpg")):
        pages.append((author, p))

print(f"Found {len(pages)} pages across {len(AUTHORS)} authors")

t_start = time.time()
total_lines = 0
for author, path in pages:
    out = OUT_ROOT / author / path.stem
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    lines, ov, info = SegmentLines(str(path))
    (out / "classifier").mkdir(exist_ok=True)
    for ln in lines:
        ln["image"].save(out / f"{ln['order']:02d}.png")
        ln["clf"].save(out / "classifier" / f"{ln['order']:02d}.png")
    ov.save(out / "_cuts_overlay.jpg", quality=88)
    total_lines += info["nLines"]
    print(f"{author}/{path.stem}: {info['nLines']} lines")

elapsed = time.time() - t_start
print()
print(f"Total pages: {len(pages)}, total lines: {total_lines}")
print(f"Total time: {elapsed:.2f}s, average per page: {elapsed/len(pages):.2f}s")
print(f"Crops written to: {OUT_ROOT}")
