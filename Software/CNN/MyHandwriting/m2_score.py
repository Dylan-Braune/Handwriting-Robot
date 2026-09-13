"""Fast per-author scoreboard for Method 2: text legibility AND writer-ID,
the two numbers that both have to clear 85%.

Loads every model once and reuses it, so a tuning loop costs one pass. The
judges are the project's own frozen classifiers used ONLY as judges -- the
handwriting itself is still produced by geometry (stored trajectories +
style measurements), never by a generative model.

    python m2_score.py                    # all authors, default corpus
    python m2_score.py --sentences 8 --seeds 2
    python m2_score.py --lam 0.4          # force a legibility blend
"""
import argparse
import json
import sys

import numpy as np
import torch

import _env
from _env import NOGIT_DIR, OUT_DIR

import SynthesizeHandwriting as SY
import BuildStyleProfile as SP
import VerifyRewrite as VR
import VerifyShapeStyle as VS
import TrainAuthorShape as SH
from EvaluateLegibility import NOVEL_CORPUS


def score(authors=None, nSent=8, nSeeds=1, lam=None, perAuthorLam=False,
          mmPerXh=4.0, pxPerMm=18.0, verbose=True, saveJson=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reader = VR.LoadTextModel(device)
    shapeModel, _mapping, i2a = VS.LoadShapeModel(device)
    profiles = SY.LoadAllProfiles()
    if authors:
        profiles = {a: profiles[a] for a in authors if a in profiles}
    corpus = NOVEL_CORPUS[:nSent]

    rows = {}
    for a in sorted(profiles):
        prof = profiles[a]
        L = (prof.get("legibilityLambda", 0.0) if perAuthorLam
             else (lam if lam is not None else 0.0))
        c = w = idOk = n = 0.0
        for si, text in enumerate(corpus):
            for seed in range(nSeeds):
                traj = SY.SynthesizeText(text, prof, mmPerXh=mmPerXh,
                                         seed=1009 * si + seed,
                                         lineWidthMm=10_000.0, legibility=L)
                img = SY.RenderTrajectory(traj, pxPerMm=pxPerMm, profile=prof,
                                          uniformInk=True)
                got = VR.ReadText(reader, img, device)
                c += VR.CharAcc(got.lower().strip(), text.lower().strip())
                w += VR.WordAcc(got.lower(), text.lower())
                norm = SH.StrokeNormalize(np.array(img.convert("L")))
                if norm is not None:
                    idOk += (i2a[VS.Classify(shapeModel, norm, device)] == a)
                n += 1
        rows[a] = dict(char=c / n, word=w / n, wid=idOk / n, lam=L)
        if verbose:
            ok = "OK " if (rows[a]["char"] >= 0.85 and rows[a]["wid"] >= 0.85) else "   "
            print(f"  {ok}{a}  lam {L:.2f} | text(char) {100*rows[a]['char']:5.1f}%  "
                  f"word {100*rows[a]['word']:5.1f}%  |  writer-ID {100*rows[a]['wid']:5.1f}%",
                  flush=True)

    agg = {k: float(np.mean([r[k] for r in rows.values()])) for k in ("char", "word", "wid")}
    nPass = sum(1 for r in rows.values() if r["char"] >= 0.85 and r["wid"] >= 0.85)
    if verbose:
        print(f"\n  mean: text {100*agg['char']:.1f}%  word {100*agg['word']:.1f}%  "
              f"writer-ID {100*agg['wid']:.1f}%")
        print(f"  authors clearing BOTH 85%: {nPass}/{len(rows)}")
        bad = [(a, r) for a, r in rows.items() if r["char"] < 0.85 or r["wid"] < 0.85]
        if bad:
            print("  failing:", ", ".join(
                f"{a}(text {100*r['char']:.0f}%, id {100*r['wid']:.0f}%)" for a, r in bad))
    if saveJson:
        with open(saveJson, "w", encoding="utf-8") as f:
            json.dump(dict(perAuthor=rows, aggregate=agg, nPass=nPass), f, indent=2)
    return rows, agg, nPass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--authors", default=None)
    ap.add_argument("--sentences", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--per-author-lam", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    score(authors=args.authors.split(",") if args.authors else None,
          nSent=args.sentences, nSeeds=args.seeds, lam=args.lam,
          perAuthorLam=args.per_author_lam, saveJson=args.json)
