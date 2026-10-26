import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = HERE / "qtp_data"


def show(name, maxdepth=3):
    p = D / name
    if not p.exists():
        print(f"  MISSING {name}")
        return
    d = json.loads(p.read_text(encoding="utf-8"))

    def walk(o, depth=0, key=""):
        pad = "  " * (depth + 1)
        if depth > maxdepth:
            return
        if isinstance(o, dict):
            for k, v in list(o.items())[:14]:
                if isinstance(v, (dict, list)):
                    print(f"{pad}{k}: {type(v).__name__}[{len(v)}]")
                    walk(v, depth + 1, k)
                else:
                    print(f"{pad}{k} = {v}")
        elif isinstance(o, list):
            print(f"{pad}[{len(o)} items] first: {str(o[0])[:120] if o else ''}")

    print(f"\n=== {name}")
    walk(d)


for n in ("qtp1a_results.json", "qtp1b_results.json", "qtp1c_results.json",
          "qtp2_results.json", "qtp3_results.json", "qtp4_results.json",
          "qtp5_results.json", "qtp6_results.json", "qtp_fieldstats.json"):
    show(n)
