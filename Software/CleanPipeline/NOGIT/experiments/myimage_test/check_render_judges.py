"""Does swapping PIL for MyImage change what the judges see?

The unit test showed the new rasteriser differs from PIL's by about 1.3% of
pixels, which is inherent: a distance field and a polygon fill are not the
same algorithm.  The question that matters is whether that moves the two
scores the best-of-N search is driven by.

Method: drive the REAL production path end to end with SynthesizeJointBestOf,
once with the new rasteriser and once with _Draw monkeypatched back to the
old PIL code.  Nothing about the render parameters is guessed here, both runs
go through RenderTrajectory exactly as the server does, and the same seed is
used so the trajectories are identical.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIPE = HERE.parent.parent.parent
sys.path.insert(0, str(PIPE))

import json                                                   # noqa: E402
import numpy as np                                            # noqa: E402
from PIL import Image, ImageDraw, ImageFilter                 # noqa: E402

import ProfileIO                                              # noqa: E402
import SynthesizeHandwriting as SH                            # noqa: E402
from Classify import (PaperCRNNNumpy, AuthorClassifierCNNNumpy,
                      ReadText, CharAcc, ClassifyImage)        # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

#must stay on ONE line.  the trajectory wraps at 180 mm, and at 4 mm per
#x-height that is about 40 characters, so a longer text renders as two
#stacked lines and the recogniser, which expects a single line, reads the
#pair as noise.  that is a property of the harness and not of the renderer
TEXTS = ["the gantry writes this",
         "handwriting is copied",
         "measure before opinion"]
WRITERS = ["yeukita", "abhinav", "thiya", "153", "151"]
NTRIES = 6

_MyDraw = SH._Draw


def PilDraw(traj, pxPerMm, padMm, strokePx, paper=255, ink=0, softPx=0.0):
    """The exact _Draw body that was in the file before the migration."""
    x0, y0, x1, y1 = traj.Bounds()
    W = max(8, int((x1 - x0 + 2 * padMm) * pxPerMm))
    H = max(8, int((y1 - y0 + 2 * padMm) * pxPerMm))
    img = Image.new('L', (W, H), int(round(paper)))
    d = ImageDraw.Draw(img)
    fill = int(round(ink))
    for s in traj.strokes:
        pts = [((x - x0 + padMm) * pxPerMm, (y1 - y + padMm) * pxPerMm)
               for (x, y) in s]
        if len(pts) >= 2:
            d.line(pts, fill=fill, width=strokePx, joint='curve')
            r = strokePx / 2.0
            if r >= 1.0:
                for (px, py) in (pts[0], pts[-1]):
                    d.ellipse([px - r, py - r, px + r, py + r], fill=fill)
        else:
            d.point(pts, fill=fill)
    if softPx > 0.05:
        img = img.filter(ImageFilter.GaussianBlur(radius=softPx))
    return img


def main():
    reader = PaperCRNNNumpy()
    authorModel = AuthorClassifierCNNNumpy()
    mapping = authorModel.author_mapping

    rows = []
    print("%-9s %-4s | %-22s | %-22s" % ("writer", "txt",
                                         "PIL  char / writer", "mine char / writer"))
    print("-" * 74)
    for wname in WRITERS:
        fp = ProfileIO.PROFILE_DIR / ("%s.json" % wname)
        if not fp.exists():
            print("skip %s: no profile file" % wname)
            continue
        prof = json.load(open(fp, encoding="utf-8"))
        for ti, text in enumerate(TEXTS):
            out = {}
            for tag, drawfn in (("pil", PilDraw), ("mine", _MyDraw)):
                SH._Draw = drawfn
                traj = SH.SynthesizeJointBestOf(
                    wname, text, prof, nTries=NTRIES, seed=11,
                    reader=reader, authorModel=authorModel,
                    authorMapping=mapping)
                #render the winner through the production path, which is the
                #same call the search used internally to score it
                img = SH.RenderTrajectory(traj, pxPerMm=18.0, profile=prof,
                                          uniformInk=True)
                acc = CharAcc(ReadText(img, reader), text.replace("\n", " "))
                _i, probs = ClassifyImage(img, authorModel)
                out[tag] = (acc, float(probs[mapping[wname]]),
                            float((np.asarray(img) < 128).mean()))
            SH._Draw = _MyDraw
            rows.append((out["pil"], out["mine"]))
            print("%-9s %-4d | %8.1f%% /%8.1f%%  | %8.1f%% /%8.1f%%"
                  % (wname, ti,
                     100 * out["pil"][0], 100 * out["pil"][1],
                     100 * out["mine"][0], 100 * out["mine"][1]))

    if rows:
        p = np.array([r[0] for r in rows])
        m = np.array([r[1] for r in rows])
        print()
        print("lines compared             : %d (best-of-%d each, same seed)"
              % (len(rows), NTRIES))
        print("char accuracy   PIL -> mine: %.2f%% -> %.2f%%  (change %+.2f pts)"
              % (100 * p[:, 0].mean(), 100 * m[:, 0].mean(),
                 100 * (m[:, 0] - p[:, 0]).mean()))
        print("writer prob     PIL -> mine: %.2f%% -> %.2f%%  (change %+.2f pts)"
              % (100 * p[:, 1].mean(), 100 * m[:, 1].mean(),
                 100 * (m[:, 1] - p[:, 1]).mean()))
        print("ink fraction    PIL -> mine: %.4f -> %.4f  (relative %+.1f%%)"
              % (p[:, 2].mean(), m[:, 2].mean(),
                 100 * (m[:, 2].mean() / max(p[:, 2].mean(), 1e-9) - 1)))
        print()
        print("writer identified correctly: PIL %d/%d, mine %d/%d"
              % (int((p[:, 1] > 0.5).sum()), len(rows),
                 int((m[:, 1] > 0.5).sum()), len(rows)))


if __name__ == "__main__":
    main()
