"""Try several synthesis settings in ONE pass (models loaded once).

Each setting is applied, every author is scored on both judges, and the
table is printed at the end so the settings are directly comparable on the
same sentences and seeds.

    python m2_sweep.py --vars 1,2,3
    python m2_sweep.py --lig on,off
"""
import argparse

import numpy as np
import torch

import _env

import SynthesizeHandwriting as SY
import VerifyRewrite as VR
import VerifyShapeStyle as VS
import TrainAuthorShape as SH
from EvaluateLegibility import NOVEL_CORPUS


def run_config(profiles, corpus, reader, shapeModel, i2a, device, nSeeds=1):
    rows = {}
    for a in sorted(profiles):
        prof = profiles[a]
        c = w = idOk = n = 0.0
        for si, text in enumerate(corpus):
            for seed in range(nSeeds):
                traj = SY.SynthesizeText(text, prof, mmPerXh=4.0,
                                         seed=1009 * si + seed,
                                         lineWidthMm=10_000.0, legibility=0.0)
                img = SY.RenderTrajectory(traj, pxPerMm=18.0, profile=prof,
                                          uniformInk=True)
                got = VR.ReadText(reader, img, device)
                c += VR.CharAcc(got.lower().strip(), text.lower().strip())
                w += VR.WordAcc(got.lower(), text.lower())
                norm = SH.StrokeNormalize(np.array(img.convert("L")))
                if norm is not None:
                    idOk += (i2a[VS.Classify(shapeModel, norm, device)] == a)
                n += 1
        rows[a] = dict(char=c / n, word=w / n, wid=idOk / n)
    return rows


def main(varCounts, nSent, nSeeds, ligModes, joinCaps=(1.0,), gaps=(0.05,),
         ranks=("selfD",)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reader = VR.LoadTextModel(device)
    shapeModel, _m, i2a = VS.LoadShapeModel(device)
    profiles = SY.LoadAllProfiles()
    corpus = NOVEL_CORPUS[:nSent]
    for p in profiles.values():            # remember the shipped ranking
        for vs in p["glyphs"].values():
            for g in vs:
                g["selfD0"] = g.get("selfD", g.get("priorD", 0.5))

    results = {}
    # `rank` picks which score orders each character's stored variants.
    # "priorD" is the ORIGINAL behaviour (distance to the cross-author
    # prototype); "selfD" is the consensus change. Both are read from the
    # same profiles, so the two are measured on identical sentences, seeds
    # and glyph cache -- the comparison the separate runs could not make.
    for rank in ranks:
        for p in profiles.values():
            for vs in p["glyphs"].values():
                for g in vs:
                    if rank == "priorD":
                        g["selfD"] = g.get("priorD", 0.5)
                    elif "selfD0" in g:
                        g["selfD"] = g["selfD0"]
                vs.sort(key=lambda g: g.get("selfD", 0.5))
        for lig in ligModes:
            saved = {a: p.get("ligSag") for a, p in profiles.items()}
            if lig == "off":
                for p in profiles.values():
                    p["ligSag"] = None
            for v in varCounts:
                for jc in joinCaps:
                    for gp in gaps:
                        SY._ELIGIBLE_VARIANTS = v
                        SY.JOIN_PROB_CAP = jc
                        SY.COLLISION_MIN_GAP_XH = gp
                        SY._STYLE_CAL.clear()   # calibration depends on these
                        rows = run_config(profiles, corpus, reader, shapeModel,
                                          i2a, device, nSeeds)
                        key = f"{rank} lig={lig} v={v} jc={jc} gap={gp}"
                        results[key] = rows
                        wid = float(np.mean([r["wid"] for r in rows.values()]))
                        ch = float(np.mean([r["char"] for r in rows.values()]))
                        nPass = sum(1 for r in rows.values()
                                    if r["char"] >= 0.85 and r["wid"] >= 0.85)
                        nId = sum(1 for r in rows.values() if r["wid"] >= 0.85)
                        print(f"  {key:>34} | writer-ID {100*wid:5.1f}%  "
                              f"text {100*ch:5.1f}% | id>=85%: {nId}/10  "
                              f"both>=85%: {nPass}/10", flush=True)
            for a, p in profiles.items():
                p["ligSag"] = saved[a]

    print("\nper-author writer-ID:")
    hdr = "  author | " + " | ".join(f"{k:>16}" for k in results)
    print(hdr)
    for a in sorted(next(iter(results.values()))):
        print(f"  {a:>6} | " + " | ".join(
            f"{100*results[k][a]['wid']:15.1f}%" for k in results))
    print("\nper-author text(char):")
    for a in sorted(next(iter(results.values()))):
        print(f"  {a:>6} | " + " | ".join(
            f"{100*results[k][a]['char']:15.1f}%" for k in results))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vars", default="2")
    ap.add_argument("--lig", default="on")
    ap.add_argument("--sentences", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--joincap", default="1.0")
    ap.add_argument("--gap", default="0.05")
    ap.add_argument("--rank", default="selfD", help="selfD,priorD")
    args = ap.parse_args()
    main([int(x) for x in args.vars.split(",")], args.sentences, args.seeds,
         args.lig.split(","), [float(x) for x in args.joincap.split(",")],
         [float(x) for x in args.gap.split(",")], args.rank.split(","))
