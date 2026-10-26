"""Which figures exist on disk but are not yet referenced by any report source,
and which measured results still have no figure?

Run from Report/FinalReport_v2:
    python figure_audit.py
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG = HERE / "Figures"
TEX = list(HERE.rglob("*.tex"))

referenced = set()
for t in TEX:
    s = t.read_text(encoding="utf-8", errors="replace")
    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", s):
        referenced.add(Path(m.group(1)).name)

existing = sorted(p.name for p in FIG.glob("*.png"))

print(f"{len(existing)} figures on disk, {len(referenced)} referenced\n")
print("=== NOT REFERENCED (available but unused) ===")
for n in existing:
    if n not in referenced:
        kb = (FIG / n).stat().st_size // 1024
        print(f"  {kb:>5} KB  {n}")

print("\n=== REFERENCED ===")
for n in existing:
    if n in referenced:
        print(f"  {n}")

print("\n=== referenced but MISSING from disk ===")
for n in sorted(referenced):
    if not (FIG / n).exists():
        print(f"  {n}")
