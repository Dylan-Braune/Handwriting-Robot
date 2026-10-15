import json
from pathlib import Path

D = Path(__file__).resolve().parent / "qtp_data"


def get(name):
    p = D / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


d = get("qtp1a_results.json")
if d:
    print("=" * 88)
    print("QTP1a  STYLE CORRELATION (real held-out vs synthesised), 10 writers")
    print("=" * 88)
    cc = d.get("combined_correlation", {})
    for group, gv in cc.items():
        print(f"\n  {group}:")
        for feat_set, fs in gv.items():
            print(f"    {feat_set}:")
            for mode, mv in fs.items():
                if isinstance(mv, dict):
                    print(f"      {mode}: " + ", ".join(
                        f"{k}={v}" for k, v in list(mv.items())[:6]))
                else:
                    print(f"      {mode} = {mv}")

    print("\n  per-feature Pearson r (real_held vs synth, across writers):")
    for fk, fv in d.get("per_feature", {}).get("real_held", {}).items():
        r = fv.get("pearson_r")
        sp = fv.get("spearman")
        rm = fv.get("real_mean")
        sm = fv.get("synth_mean")
        print(f"    {fk:<18} r={r:+.3f}  rho={sp:+.3f}  real_mean={rm:.3f}  "
              f"synth_mean={sm:.3f}")

    nn = d.get("nn_writer_id")
    if nn:
        print("\n  nearest-neighbour writer retrieval:")
        print("   ", json.dumps(nn)[:600])

for n in ("qtp1b_results.json", "qtp1c_results.json"):
    d = get(n)
    if not d:
        continue
    print("\n" + "=" * 88)
    print(f"{n}")
    print("=" * 88)


    def walk(o, depth=0):
        pad = "  " * (depth + 1)
        if depth > 2:
            return
        if isinstance(o, dict):
            for k, v in list(o.items())[:20]:
                if isinstance(v, (dict, list)):
                    print(f"{pad}{k}: {type(v).__name__}[{len(v)}]")
                    walk(v, depth + 1)
                else:
                    print(f"{pad}{k} = {v}")
        elif isinstance(o, list) and o:
            print(f"{pad}[{len(o)}] first = {str(o[0])[:110]}")

    walk(d)
