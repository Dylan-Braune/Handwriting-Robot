"""Render each author's Method 2 output to its own image, at a size meant to
be READ rather than scored. No classifier involved."""
import argparse

import numpy as np

import _env
from _env import OUT_DIR

import SynthesizeHandwriting as SY

TEXT = "the quick brown fox jumps over the lazy dog"


def main(text=TEXT, authors=None, pxPerMm=26.0, seed=0):
    profiles = SY.LoadAllProfiles()
    if authors:
        profiles = {a: profiles[a] for a in authors if a in profiles}
    for a in sorted(profiles):
        prof = profiles[a]
        traj = SY.SynthesizeText(text, prof, mmPerXh=4.0, seed=seed,
                                 lineWidthMm=10_000.0,
                                 legibility=prof.get("legibilityLambda", 0.0))
        img = SY.RenderTrajectory(traj, pxPerMm=pxPerMm, profile=prof,
                                  uniformInk=True)
        p = OUT_DIR / f"read_{a}.png"
        img.save(p)
        print(f"{a}: {img.size}  -> {p.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=TEXT)
    ap.add_argument("--authors", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(text=args.text,
         authors=args.authors.split(",") if args.authors else None,
         seed=args.seed)
