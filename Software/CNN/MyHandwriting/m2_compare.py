"""Real line vs synthesized line, per author, with what each judge said.

Both rows are put through the writer-ID judge's own preprocessing
(StrokeNormalize: centreline re-inked at one constant pen width) so the
comparison shows what the judge actually sees, not what a human sees.
"""
import argparse

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import _env
from _env import OUT_DIR

import SynthesizeHandwriting as SY
import BuildStyleProfile as SP
import VerifyRewrite as VR
import VerifyShapeStyle as VS
import TrainAuthorShape as SH
from TrainText import IAMLineDatasetRaw, _decode_png

TEXT = "the quick brown fox jumps over the lazy dog"


def main(authors=None, text=TEXT, normalized=False, out="m2_real_vs_synth.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reader = VR.LoadTextModel(device)
    shapeModel, _m, i2a = VS.LoadShapeModel(device)
    profiles = SY.LoadAllProfiles()
    if authors:
        profiles = {a: profiles[a] for a in authors if a in profiles}

    base = IAMLineDatasetRaw(root_dir=str(SP.DATA_DIR), cache_dir=str(SP.CACHE_DIR))
    realBy = {}
    for s in base.samples:
        a = s["page_key"].split("/")[0]
        realBy.setdefault(a, s)

    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\consola.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    rowH, pad, labelW = 80, 8, 110
    panels = []
    for a in sorted(profiles):
        prof = profiles[a]
        traj = SY.SynthesizeText(text, prof, mmPerXh=4.0, seed=0,
                                 lineWidthMm=10_000.0,
                                 legibility=prof.get("legibilityLambda", 0.0))
        synth = SY.RenderTrajectory(traj, pxPerMm=18.0, profile=prof, uniformInk=True)
        real = _decode_png(realBy[a]["image_png"]).convert("L") if a in realBy else None

        def judge(img):
            got = VR.ReadText(reader, img, device)
            norm = SH.StrokeNormalize(np.array(img.convert("L")))
            who = i2a[VS.Classify(shapeModel, norm, device)] if norm is not None else "?"
            return got, who, (norm if normalized else img)

        sGot, sWho, sImg = judge(synth)
        panels.append((a, "synth", sImg, sGot, sWho))
        if real is not None:
            rGot, rWho, rImg = judge(real)
            panels.append((a, "real ", rImg, rGot, rWho))

    scaled = []
    for a, kind, im, got, who in panels:
        im = im.convert("L")
        s = rowH / max(1, im.height)
        scaled.append((a, kind, im.resize((max(1, int(im.width * s)), rowH),
                                          Image.LANCZOS), got, who))
    W = labelW + max(p[2].width for p in scaled) + 20
    H = sum(rowH + 20 for _ in scaled) + pad
    sheet = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(sheet)
    y = pad
    for a, kind, im, got, who in scaled:
        mark = "OK" if who == a else "->" + who
        d.text((4, y + 2), f"{a} {kind}", fill=0, font=font)
        d.text((labelW, y), f"ID:{mark}  read:{got[:80]}", fill=0, font=font)
        sheet.paste(im, (labelW, y + 16))
        y += rowH + 20
    path = OUT_DIR / out
    sheet.save(path)
    print(f"-> {path}  {sheet.size}")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--authors", default=None)
    ap.add_argument("--text", default=TEXT)
    ap.add_argument("--normalized", action="store_true",
                    help="show what the writer-ID judge sees, not the raw render")
    ap.add_argument("--out", default="m2_real_vs_synth.png")
    args = ap.parse_args()
    main(authors=args.authors.split(",") if args.authors else None,
         text=args.text, normalized=args.normalized, out=args.out)
