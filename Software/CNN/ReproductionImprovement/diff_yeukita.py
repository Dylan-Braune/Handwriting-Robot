"""One-off: report exactly how the live yeukita profile differs from the backup
rebuild_author.py took before rebuilding."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIVE = HERE.parent / "NOGIT" / "StyleProfiles10" / "yeukita.json"
BK = HERE / "out" / "backup" / "yeukita_20261009_141147" / "yeukita.json"

a = json.loads(LIVE.read_text(encoding="utf-8"))
b = json.loads(BK.read_text(encoding="utf-8"))

print("before (backup) bytes:", BK.stat().st_size, " mtime:", BK.stat().st_mtime)
print("live            bytes:", LIVE.stat().st_size, " mtime:", LIVE.stat().st_mtime)
print("byte-identical:", LIVE.read_bytes() == BK.read_bytes())
print()
for k in ("nLines", "connectedness", "slantDeg", "xHeightPx", "ascender",
          "descender", "wordSpaceXh", "strokeWidthXh"):
    print(f"  {k:16} before={b.get(k)!r:>10}  live={a.get(k)!r:>10}")
print(f"  {'glyph letters':16} before={len(b['glyphs']):>10}  live={len(a['glyphs']):>10}")
va = sum(len(v) for v in a["glyphs"].values())
vb = sum(len(v) for v in b["glyphs"].values())
print(f"  {'glyph variants':16} before={vb:>10}  live={va:>10}")
print(f"  {'idealGlyphs':16} before={len(b.get('idealGlyphs', {})):>10}  "
      f"live={len(a.get('idealGlyphs', {})):>10}")
