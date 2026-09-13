"""
METHOD 2 -- first-principles feature extraction + parametric synthesis.

This is the route that matches the project proposal's FU 2.5 ("character
image -> trajectory map"): instead of pasting pixel crops (Method 1), it
skeletonises every letter the writer made into an ordered pen path, keeps
several denoised variants per letter, and measures the writer's style
numbers (slant, x-height, stroke width, ascender/descender reach, word
spacing, connectedness). New text is then DRAWN from those paths, sized and
slanted to the writer, with a plain single-stroke fallback for any letter
the writer never wrote.

Pipeline (all your own code -- Skeletonize / TracePolylines / DouglasPeucker
/ CTC forced-align live in BuildStyleProfile.py):
  collect_pairs (Method 1)  ->  ExtractAuthorRaw  ->  BuildAuthorProfile
  ->  SynthesizeHandwriting.SynthesizeLegible  ->  pen trajectory + render

Trainable on any writer.  --build extracts a profile for whichever pages
sit in pages/ ; here that is the student's own hand.  Pass --iam to also
build the 10 IAM authors so the same method demonstrably writes their
styles too.

    python method2_features.py --build
    python method2_features.py "some new sentence"
    python method2_features.py --iam            # + build IAM-10 profiles
"""
import argparse
import json

import numpy as np
import torch

import _env
from _env import OUT_DIR, LIB_DIR

from method1_exemplar import load_model, collect_pairs
import BuildStyleProfile as SP

PROFILE_JSON = LIB_DIR / "method2_dylan.json"
SP.PROFILE_DIR = LIB_DIR                          # LoadProfile() looks here now
GLYPH_CACHE = _env.CNN_DIR / "NOGIT" / "GlyphCache10"


def _rawlib(parsed):
    lib = {}
    for glyphs, _st in parsed:
        for g in glyphs:
            if g and g.get("char", " ") != " " and "strokes" in g:
                lib.setdefault(g["char"], []).append(g)
    return lib


def letter_prior(my_parsed):
    """What each letter looks like, pooled across the 10 IAM hands + this
    writer -- only used to REJECT this writer's own worst-cut variants,
    never to substitute another hand's shape (same as BuildStyleProfile)."""
    import pickle
    libs = {}
    for p in sorted(GLYPH_CACHE.glob("*.pkl")):
        try:
            with open(p, "rb") as f:
                libs[p.stem] = _rawlib(pickle.load(f))
        except Exception:
            pass
    libs["_self"] = _rawlib(my_parsed)
    prior = SP.BuildLetterPrior(libs, minSamples=8)
    return prior


def build_profile(author_id="dylan", pairs=None, verbose=True):
    device = torch.device("cpu")
    model = load_model(device)
    if pairs is None:
        pairs = collect_pairs(model, device, verbose)
    if not pairs:
        raise RuntimeError("no aligned lines")
    parsed = SP.ExtractAuthorRaw(author_id, pairs, model, device)
    if not parsed:
        raise RuntimeError("ExtractLineGlyphs returned nothing -- alignment too weak")
    refs = [st["ref"] for _, st in parsed if "ref" in st]
    prior = letter_prior(parsed)
    prof = SP.BuildAuthorProfile(author_id, parsed, refs=refs, prior=prior,
                                 verbose=verbose)
    if prof is None:
        raise RuntimeError("BuildAuthorProfile failed")
    out = LIB_DIR / f"{author_id}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(prof, f)
    nvar = sum(len(v) for v in prof.get("glyphs", {}).values())
    print(f"[profile {author_id}] {len(parsed)} lines, "
          f"{len(prof.get('glyphs', {}))} letters, {nvar} glyph variants | "
          f"slant {prof['slantDeg']:+.1f} deg  x-h {prof['xHeightPx']:.0f}px  "
          f"conn {prof['connectedness']:.2f}  wordgap {prof['wordSpaceXh']:.2f}xh")
    return prof


def synthesize(text, profile=None, author_id="dylan", seed=7, legible=True):
    import SynthesizeHandwriting as SY
    if profile is None:
        with open(LIB_DIR / f"{author_id}.json", encoding="utf-8") as f:
            profile = json.load(f)
    if legible and hasattr(SY, "SynthesizeLegible"):
        traj = SY.SynthesizeLegible(text, profile, nTries=6, mmPerXh=4.0,
                                    lineWidthMm=10_000.0)
    else:
        traj = SY.SynthesizeText(text, profile, mmPerXh=4.0, seed=seed,
                                 lineWidthMm=10_000.0,
                                 legibility=profile.get("legibilityLambda", 0.6))
    img = SY.RenderTrajectory(traj, pxPerMm=18.0, profile=profile)
    return img, traj


def build_iam(verbose=True):
    """Show the SAME method builds the 10 dataset authors' styles."""
    device = torch.device("cpu")
    model = load_model(device)
    from TrainText import IAMLineDatasetRaw, _decode_png
    ds = IAMLineDatasetRaw(root_dir=str(SP.DATA_DIR), cache_dir=str(SP.CACHE_DIR))
    from _env import CNN_DIR
    authors = sorted({s["page_key"].split("/")[0] for s in ds.samples})
    for a in authors:
        pairs = [(np.array(_decode_png(s["image_png"]).convert("L")), s["text"])
                 for s in ds.samples
                 if s["page_key"].split("/")[0] == a and not s["is_holdout"]][:120]
        try:
            build_profile(a, pairs, verbose=False)
        except Exception as e:
            print(f"  {a}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?",
                    default="the quick brown fox jumps over the lazy dog")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--iam", action="store_true")
    args = ap.parse_args()

    if args.iam:
        build_iam()
    prof = build_profile() if (args.build or args.iam
                               or not (LIB_DIR / "dylan.json").exists()) else None
    img, _ = synthesize(args.text, prof)
    out = OUT_DIR / "method2_sample.png"
    img.save(out)
    print(f"\n  '{args.text}'\n  -> {out}  {img.size}")


if __name__ == "__main__":
    main()
