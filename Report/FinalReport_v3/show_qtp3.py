import json
from pathlib import Path

D = Path(__file__).resolve().parent / "qtp_data"


def get(n):
    p = D / n
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


d = get("qtp1c_results.json")
print("=" * 90)
print("QTP1c  CHARACTER ACCURACY WITH AN INDEPENDENT JUDGE")
print("=" * 90)
if d:
    for judge, jv in d.items():
        if judge == "meta":
            print(f"\nmeta: production={jv.get('production_ckpt')}  "
                  f"independent={jv.get('heldout_ckpt')}  N={jv.get('nTries_in_synthesis')}")
            print(f"      {jv.get('note')}")
            continue
        print(f"\njudge = {judge}")
        for cond, cv in jv.items():
            if isinstance(cv, dict) and "overall" in cv:
                o = cv["overall"]
                print(f"  {cond}: n={o.get('n')}  char_mean={o.get('char_acc_mean'):.4f}  "
                      f"char_micro={o.get('char_acc_micro'):.4f}  "
                      f"word_mean={o.get('word_acc_mean'):.4f}")
            elif not isinstance(cv, dict):
                print(f"  {cond} = {cv}")

for n in ("qtp3_results.json", "qtp4_results.json", "qtp5_results.json"):
    d = get(n)
    if not d:
        print(f"\nMISSING {n}")
        continue
    print("\n" + "=" * 90)
    print(n)
    print("=" * 90)


    def walk(o, depth=0, maxd=2):
        pad = "  " * (depth + 1)
        if depth > maxd:
            return
        if isinstance(o, dict):
            for k, v in list(o.items())[:18]:
                if isinstance(v, (dict, list)):
                    print(f"{pad}{k}: {type(v).__name__}[{len(v)}]")
                    walk(v, depth + 1, maxd)
                else:
                    print(f"{pad}{k} = {v}")
        elif isinstance(o, list) and o:
            print(f"{pad}[{len(o)}] first={str(o[0])[:130]}")

    walk(d)
