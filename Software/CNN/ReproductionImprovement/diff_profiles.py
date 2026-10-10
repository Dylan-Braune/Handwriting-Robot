"""Report how the rebuilt profiles differ from the git-tracked FinalPipeline
snapshot (the only committed copy of the 8 dataset-author profiles).

The live NOGIT/StyleProfiles10 is gitignored and was never committed, so for the
personal authors there is no recoverable "before". For the 8 dataset authors,
FinalPipeline/Software/CNN/NOGIT/StyleProfiles10 IS tracked, so it can at least
show whether the rebuild produced something materially different.
"""
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
LIVE = HERE.parent / "NOGIT" / "StyleProfiles10"

print(f"{'profile':>12} {'tracked?':>9} {'nLines':>14} {'variants':>14} {'conn':>14}")
for a in ("150", "151", "152", "153", "384", "551", "552", "588",
          "yeukita", "dylan", "owen", "thiya", "abhinav"):
    rel = f"FinalPipeline/Software/CNN/NOGIT/StyleProfiles10/{a}.json"
    out = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=REPO,
                         capture_output=True)
    live = json.loads((LIVE / f"{a}.json").read_text(encoding="utf-8"))
    lv = sum(len(v) for v in live["glyphs"].values())
    if out.returncode != 0:
        print(f"{a:>12} {'no':>9} {'':>14} {lv:>14} {live['connectedness']:>14}")
        continue
    old = json.loads(out.stdout.decode("utf-8"))
    ov = sum(len(v) for v in old["glyphs"].values())
    same = (old.get("nLines") == live.get("nLines") and ov == lv
            and old.get("connectedness") == live.get("connectedness"))
    print(f"{a:>12} {'yes':>9} "
          f"{str(old.get('nLines'))+' -> '+str(live.get('nLines')):>14} "
          f"{str(ov)+' -> '+str(lv):>14} "
          f"{str(old.get('connectedness'))+' -> '+str(live.get('connectedness')):>14}"
          + ("   IDENTICAL" if same else "   differs"))
