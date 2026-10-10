"""Build a profile from an already-extracted raw cache, writing to a test path.

Used to evaluate a re-extraction without touching NOGIT. Reuses the project's
own BuildAuthorProfile / BuildLetterPrior unchanged.

    python build_from_raw.py yeukita out/reextract/yeukita/yeukita.pkl
"""
import json
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths  # noqa: F401

import BuildStyleProfile as BSP

author = sys.argv[1]
pkl = HERE / sys.argv[2]
outdir = HERE / "out" / "profiles_test"
outdir.mkdir(parents=True, exist_ok=True)

with open(pkl, "rb") as f:
    parsed = pickle.load(f)

lib = {}
for glyphs, _st in parsed:
    for g in glyphs:
        if g and g.get("char", " ") != " " and "strokes" in g:
            lib.setdefault(g["char"], []).append(g)
prior = BSP.BuildLetterPrior({author: lib})
print(f"letter prior from {author}'s own glyphs: {len(prior)} characters")

refs = [st["ref"] for _g, st in parsed if "ref" in st]
prof = BSP.BuildAuthorProfile(author, parsed, refs=refs, prior=prior)

n_old = None
old = paths.CNN / "NOGIT" / "StyleProfiles10" / f"{author}.json"
if old.exists():
    n_old = sum(len(v) for v in
                json.loads(old.read_text(encoding="utf-8"))["glyphs"].values())

with open(outdir / f"{author}.json", "w", encoding="utf-8") as f:
    json.dump(prof, f)
nv = sum(len(v) for v in prof["glyphs"].values())
print(f"\nnew profile: {len(prof['glyphs'])} letters, {nv} variants"
      + (f"   (old: {n_old} variants)" if n_old else ""))
print(f"  i: {len(prof['glyphs'].get('i', []))} variants "
      f"(old: {len(json.loads(old.read_text(encoding='utf-8'))['glyphs'].get('i', [])) if old.exists() else '?'})")
print(f"  l: {len(prof['glyphs'].get('l', []))}   "
      f"a: {len(prof['glyphs'].get('a', []))}   o: {len(prof['glyphs'].get('o', []))}")
print(f"wrote {outdir / (author + '.json')}")
