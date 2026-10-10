"""Rebuild ONE author's style profile with the ReproductionImprovement builder.

Backs up both files it will overwrite (the raw glyph cache and the profile
JSON) into out/backup/<author>_<stamp>/ first, so the previous state can be put
back exactly.

    python rebuild_author.py yeukita
"""
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths

import BuildStyleProfile as BSP
import authors_config as AC

author = sys.argv[1] if len(sys.argv) > 1 else "yeukita"
raw = BSP.RAW_DIR / f"{author}.pkl"
prof = paths.CNN / "NOGIT" / "StyleProfiles10" / f"{author}.json"

stamp = time.strftime("%Y%m%d_%H%M%S")
bdir = HERE / "out" / "backup" / f"{author}_{stamp}"
bdir.mkdir(parents=True, exist_ok=True)
for p in (raw, prof):
    if p.exists():
        shutil.copy2(p, bdir / p.name)
        print(f"backed up {p}  ->  {bdir / p.name}")

# Only this author, so nothing else is re-segmented or overwritten.
AC.DATASET_AUTHORS = []
AC.PERSONAL_AUTHORS = [author]

print(f"\nrebuilding {author} (segmentation: SegmentLean) ...")
BSP.BuildAll10Authors()

new = paths.CNN / "NOGIT" / "StyleProfiles10" / f"{author}.json"
if new.exists() and prof.exists():
    import json
    a = json.loads(prof.read_text(encoding="utf-8"))
    b = json.loads(new.read_text(encoding="utf-8"))
    print(f"\nbefore: {len(a['glyphs'])} letters, "
          f"{sum(len(v) for v in a['glyphs'].values())} variants, "
          f"conn={a['connectedness']}")
    print(f"after : {len(b['glyphs'])} letters, "
          f"{sum(len(v) for v in b['glyphs'].values())} variants, "
          f"conn={b['connectedness']}")
    print(f"backup of the before state: {bdir}")
