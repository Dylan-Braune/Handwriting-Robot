"""
match_author_style.py -- "which of the 10 authors is this closest to?",
decided on measured geometry alone.

A deliberately transparent second opinion on style accuracy. Where the CNN
judges are a black box (and the original writer-ID model turned out to be
reading ink weight rather than the hand), this one compares a short vector
of human-meaningful measurements and picks the nearest author:

    slant, word gap, ascender reach, descender reach, line height,
    letter pitch, letter width, and a cursive-ness proxy (how many
    separate ink pieces per unit of writing -- a joined hand leaves few)

Every one of those is a property the gantry actually reproduces, and NONE
of them depends on how thick the pen is, so the score means the same thing
on paper as it does on screen.

Each author's reference vector is the median over their real TRAINING
lines; a sample is assigned to the author whose vector is nearest in units
of across-author spread (z-scored, so no single feature dominates).

Run:
    python match_author_style.py
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fp_ops as F
import style_profile as SP
from train_paper_cnn_bilstm_ctc import IAMLineDatasetRaw, _decode_png

SCRIPT_DIR = Path(__file__).resolve().parent

FEATURES = ("slant", "wordGap", "asc", "desc", "height", "pitch",
            "compW", "pieces")


def MeasureGeometry(gray):
    """The style measurements, all independent of stroke thickness."""
    ink = SP.BinarizeLine(gray)
    if ink.sum() < 40:
        return None
    band = SP.CoreBand(ink)
    if band is None:
        return None
    top, base = band
    xh = float(base - top)
    if xh < 4:
        return None
    ys = np.nonzero(ink.any(axis=1))[0]

    col = ink.any(axis=0)
    gaps, run = [], 0
    for v in col:
        if v:
            if run:
                gaps.append(run)
            run = 0
        else:
            run += 1
    wide = [g / xh for g in gaps if g / xh > 0.5]

    labels, n = F.LabelComponents(ink, connectivity=8)
    widths, cxs = [], []
    if n:
        stats = F.ComponentStatsFromLabels(labels, n)
        for st in stats:
            if st is None or st["area"] < max(8, 0.02 * xh * xh):
                continue
            widths.append(st["w"] / xh)
            cxs.append(st["cx"])
    cxs.sort()
    pitch = float(np.median(np.diff(cxs)) / xh) if len(cxs) > 2 else np.nan
    spanXh = (ink.any(axis=0).sum()) / xh

    return dict(
        slant=float(SP.EstimateSlantDeg(ink)),
        wordGap=float(np.median(wide)) if wide else np.nan,
        asc=float(base - ys.min()) / xh,
        desc=float(base - ys.max()) / xh,
        height=float(ys.max() - ys.min() + 1) / xh,
        pitch=pitch,
        compW=float(np.median(widths)) if widths else np.nan,
        # separate ink pieces per x-height of writing: a joined (cursive)
        # hand leaves very few, a print hand leaves many
        pieces=float(len(widths)) / max(1.0, spanXh),
    )


def Vec(m):
    return np.array([m[k] for k in FEATURES], dtype=np.float64)


class AuthorMatcher:
    def __init__(self, refs):
        """refs: {author: [measurement dicts from their real lines]}"""
        self.authors = sorted(refs)
        rows = []
        for a in self.authors:
            V = np.array([Vec(m) for m in refs[a]], dtype=np.float64)
            rows.append(np.nanmedian(V, axis=0))
        self.M = np.vstack(rows)
        # spread ACROSS authors: features that separate authors weigh more
        self.scale = np.nanstd(self.M, axis=0)
        self.scale[~np.isfinite(self.scale) | (self.scale < 1e-6)] = 1.0

    def Nearest(self, m):
        v = Vec(m)
        d = (self.M - v[None, :]) / self.scale[None, :]
        d = np.where(np.isfinite(d), d, 0.0)
        return self.authors[int(np.argmin(np.sqrt((d ** 2).mean(axis=1))))]

    def Ranked(self, m):
        v = Vec(m)
        d = (self.M - v[None, :]) / self.scale[None, :]
        d = np.where(np.isfinite(d), d, 0.0)
        order = np.argsort(np.sqrt((d ** 2).mean(axis=1)))
        return [(self.authors[i], float(np.sqrt((d[i] ** 2).mean())))
                for i in order]


def BuildFromRealPages(maxTrain=None):
    """Reference vectors from each author's real TRAINING lines, and the
    real HELD-OUT lines kept back to measure the matcher's own ceiling."""
    base = IAMLineDatasetRaw(root_dir=str(SP.DATA_DIR), cache_dir=str(SP.CACHE_DIR))
    refs, held = {}, {}
    for s in base.samples:
        a = s["page_key"].split("/")[0]
        m = MeasureGeometry(np.array(_decode_png(s["image_png"]).convert("L")))
        if m is None:
            continue
        (held if s["is_holdout"] else refs).setdefault(a, []).append(m)
    if maxTrain:
        refs = {a: v[:maxTrain] for a, v in refs.items()}
    return AuthorMatcher(refs), held


# ---------------------------------------------------------------------------
# Scoring synthesized handwriting with this matcher
# ---------------------------------------------------------------------------
def ScoreSynthesis(nSeeds=2, mmPerXh=4.0, pxPerMm=18.0, throughGcode=True):
    """How often is synthesized author X matched back to author X, on NOVEL
    sentences, judged on geometry alone?"""
    import gcode_writer as GW
    import synthesize_handwriting as SY
    import verify_end_to_end as V

    matcher, held = BuildFromRealPages()
    profiles = SY.LoadAllProfiles()
    cfg = GW.GantryConfig()
    tmp = SCRIPT_DIR / "NOGIT" / "EndToEnd" / "_match.gcode"
    tmp.parent.mkdir(parents=True, exist_ok=True)

    per, perG = {}, {}
    for a in sorted(profiles):
        prof = profiles[a]
        ok = okG = tot = 0
        for si, text in enumerate(V.NOVEL_SENTENCES):
            for seed in range(nSeeds):
                traj = SY.SynthesizeText(text, prof, mmPerXh=mmPerXh,
                                         seed=100 * si + seed,
                                         lineWidthMm=10_000.0)
                img = SY.RenderTrajectory(traj, pxPerMm=pxPerMm, profile=prof)
                m = MeasureGeometry(np.array(img))
                if m:
                    ok += (matcher.Nearest(m) == a)
                if throughGcode:
                    g = V.GcodeRoundTrip(traj, cfg, str(tmp), prof)
                    if g is not None:
                        mg = MeasureGeometry(np.array(g))
                        if mg:
                            okG += (matcher.Nearest(mg) == a)
                tot += 1
        per[a] = ok / max(1, tot)
        perG[a] = okG / max(1, tot)
        print("  %s: geometry-match %5.1f%%   (through G-code %5.1f%%)"
              % (a, 100 * per[a], 100 * perG[a]))

    rok = sum(sum(1 for m in v if matcher.Nearest(m) == a)
              for a, v in held.items())
    rtot = sum(len(v) for v in held.values())
    print("\nsynthesized : %.1f%%   (through G-code %.1f%%)"
          % (100 * np.mean(list(per.values())), 100 * np.mean(list(perG.values()))))
    print("CEILING, same matcher on REAL held-out handwriting: %.1f%%"
          % (100 * rok / max(1, rtot)))
    return per, perG


def CeilingReport():
    matcher, held = BuildFromRealPages()
    print("=== geometry-only nearest-author matching ===")
    print("features: %s\n" % ", ".join(FEATURES))
    ok = tot = 0
    for a in sorted(held):
        c = sum(1 for m in held[a] if matcher.Nearest(m) == a)
        ok += c
        tot += len(held[a])
        print("  %s: %5.1f%%  (%d/%d) on the author's own REAL held-out lines"
              % (a, 100 * c / max(1, len(held[a])), c, len(held[a])))
    print("\nCEILING on real held-out handwriting: %.1f%%  (%d lines, chance 10%%)"
          % (100 * ok / max(1, tot), tot))


if __name__ == "__main__":
    CeilingReport()
    print("")
    print("=== synthesized handwriting, NOVEL sentences ===")
    ScoreSynthesis()
