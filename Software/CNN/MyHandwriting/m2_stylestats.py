"""Which measured style statistic does synthesis fail to reproduce?

For each author, measure the SAME geometric statistics on (a) their real
lines and (b) a synthesized line, using the same measuring code, and print
the gap. A writer-ID failure has to show up as a divergence in something
measurable -- this says which one, instead of guessing from the picture.
"""
import argparse

import numpy as np
import torch
from PIL import Image

import _env

import SynthesizeHandwriting as SY
import BuildStyleProfile as SP
import VerifyShapeStyle as VS
import TrainAuthorShape as SH
from TrainText import IAMLineDatasetRaw, _decode_png

SAMPLE = ("the quick brown fox jumps over lazy dogs and by half past "
          "eight it kept flying quietly")


def measure(gray, text):
    """Same measurements RenderRefStats makes, on any grayscale line."""
    ink = SP.BinarizeLine(gray)
    band = SP.CoreBand(ink)
    if band is None or not ink.any():
        return None
    top, base = band
    xh = float(base - top)
    if xh < 4:
        return None
    ys = np.nonzero(ink.any(axis=1))[0]
    col = ink.any(axis=0)
    xs = np.nonzero(col)[0]
    gaps, run = [], 0
    for v in col:
        if v:
            if run:
                gaps.append(run)
            run = 0
        else:
            run += 1
    wide = [g / xh for g in gaps if g / xh > 0.5]
    # pen lifts: separate ink runs per character, a real per-writer signature
    runs = 0
    prev = False
    for v in col:
        if v and not prev:
            runs += 1
        prev = v
    return dict(
        slant=float(SP.EstimateSlantDeg(ink)),
        asc=float(base - ys.min()) / xh,
        desc=float(base - ys.max()) / xh,
        pitch=float(xs.max() - xs.min() + 1) / xh / max(1, len(text)),
        wordGap=float(np.median(wide)) if wide else np.nan,
        heightXh=float(ys.max() - ys.min() + 1) / xh,
        runsPerChar=runs / max(1, len(text)),
    )


def main(authors=None, nReal=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    profiles = SY.LoadAllProfiles()
    if authors:
        profiles = {a: profiles[a] for a in authors if a in profiles}
    base = IAMLineDatasetRaw(root_dir=str(SP.DATA_DIR), cache_dir=str(SP.CACHE_DIR))
    realBy = {}
    for s in base.samples:
        realBy.setdefault(s["page_key"].split("/")[0], []).append(s)

    keys = ["slant", "asc", "desc", "pitch", "wordGap", "heightXh", "runsPerChar"]
    print(f"{'author':>6} {'stat':>12} {'real':>9} {'synth':>9} {'gap':>9}")
    print("-" * 50)
    for a in sorted(profiles):
        prof = profiles[a]
        rms = []
        for s in realBy.get(a, [])[:nReal]:
            m = measure(np.array(_decode_png(s["image_png"]).convert("L")), s["text"])
            if m:
                rms.append(m)
        if not rms:
            continue
        real = {k: float(np.nanmedian([m[k] for m in rms])) for k in keys}
        traj = SY.SynthesizeText(SAMPLE, prof, mmPerXh=4.0, seed=0,
                                 lineWidthMm=10_000.0,
                                 legibility=prof.get("legibilityLambda", 0.0))
        img = SY.RenderTrajectory(traj, pxPerMm=18.0, profile=prof, uniformInk=True)
        syn = measure(np.array(img.convert("L")), SAMPLE)
        if syn is None:
            continue
        for k in keys:
            gap = syn[k] - real[k]
            flag = ""
            denom = max(1e-6, abs(real[k]))
            if k == "slant":
                flag = " <<<" if abs(gap) > 6 else ""
            elif abs(gap) / denom > 0.20:
                flag = " <<<"
            print(f"{a:>6} {k:>12} {real[k]:9.3f} {syn[k]:9.3f} {gap:+9.3f}{flag}")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--authors", default=None)
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    main(authors=args.authors.split(",") if args.authors else None, nReal=args.n)
