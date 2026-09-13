"""What do the two judges score on the authors' REAL handwriting?

This is the ceiling any synthesis can be measured against. If the shape-only
writer-ID model only reaches N% on a writer's genuine held-out lines, then
"synthesis scores N%" means synthesis is indistinguishable from their real
hand -- not that it failed. Same for the text recogniser.
"""
import argparse
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

import _env

import BuildStyleProfile as SP
import VerifyRewrite as VR
import VerifyShapeStyle as VS
import TrainAuthorShape as SH
from TrainText import IAMLineDatasetRaw, _decode_png


def main(maxPerAuthor=12, holdoutOnly=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reader = VR.LoadTextModel(device)
    shapeModel, _m, i2a = VS.LoadShapeModel(device)

    base = IAMLineDatasetRaw(root_dir=str(SP.DATA_DIR), cache_dir=str(SP.CACHE_DIR))
    byAuthor = defaultdict(list)
    for s in base.samples:
        if holdoutOnly and not s["is_holdout"]:
            continue
        byAuthor[s["page_key"].split("/")[0]].append(s)

    print(f"real-handwriting ceiling ({'holdout' if holdoutOnly else 'all'} lines)\n")
    rows = {}
    for a in sorted(byAuthor):
        c = idOk = n = 0.0
        for s in byAuthor[a][:maxPerAuthor]:
            img = _decode_png(s["image_png"]).convert("L")
            got = VR.ReadText(reader, img, device)
            c += VR.CharAcc(got.lower().strip(), s["text"].lower().strip())
            norm = SH.StrokeNormalize(np.array(img))
            if norm is not None:
                idOk += (i2a[VS.Classify(shapeModel, norm, device)] == a)
            n += 1
        if n:
            rows[a] = (c / n, idOk / n, int(n))
            print(f"  {a}  n={int(n):3d} | text(char) {100*c/n:5.1f}%  "
                  f"writer-ID {100*idOk/n:5.1f}%")
    if rows:
        print(f"\n  mean: text {100*np.mean([v[0] for v in rows.values()]):.1f}%  "
              f"writer-ID {100*np.mean([v[1] for v in rows.values()]):.1f}%")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--all", action="store_true", help="include non-holdout lines")
    args = ap.parse_args()
    main(maxPerAuthor=args.n, holdoutOnly=not args.all)
