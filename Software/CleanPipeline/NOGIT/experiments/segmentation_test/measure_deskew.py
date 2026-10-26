"""Validate the deskew stage against a known injected rotation.

The page is rotated by a known angle, the segmenter is run on the rotated
page, and the angle it chooses is compared with the negative of the injected
one.  This is a direct test of the projection objective of report
Equation (seg-skew), because the correct answer is known exactly.

Only angles inside the search range of +/-4 degrees are injected; the search
step is 0.5 degrees, so the best achievable error is 0.25 degrees.

Writes new_results/deskew_results.json and prints a table.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIPE = HERE.parent.parent.parent
sys.path.insert(0, str(PIPE))

import numpy as np                                            # noqa: E402
from PIL import Image                                         # noqa: E402

import AuthorsConfig as AC                                    # noqa: E402
import SegmentLean                                            # noqa: E402

OUT_DIR = HERE / "new_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TMP = OUT_DIR / "tmp_rotated.png"

INJECT = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
STEP = 0.5

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def ChosenAngle(path):
    """Run only the deskew search of SegmentLines and return its angle.

    This mirrors the deployed code exactly: the same Halve, the same
    InkMask and the same SkewScore the module itself uses.
    """
    import MyImage
    full = Image.open(path).convert("L")
    halfA = MyImage.Halve(np.asarray(full, dtype=np.uint8))
    mask, _ = SegmentLean.InkMask(halfA.astype(np.float64))
    q = MyImage.Halve(mask.astype(np.uint8) * 255) > 127
    return float(max(np.arange(-4, 4.01, STEP),
                     key=lambda a: SegmentLean.SkewScore(q, a)))


def main():
    # one page per personal writer, so the result is not one hand's property
    pages = []
    for author in AC.PERSONAL_AUTHORS:
        for imgPath, _lab in AC.PersonalAuthorPages(author):
            pages.append((author, imgPath))
            break

    rows = []
    print("%-10s %-16s %8s %8s %8s %8s" %
          ("writer", "page", "inject", "native", "expect", "chosen"))
    for author, imgPath in pages:
        native = ChosenAngle(str(imgPath))
        src = Image.open(imgPath).convert("L")
        for inj in INJECT:
            # rotate the page by the injected angle, white fill as the camera sees paper
            src.rotate(inj, Image.BILINEAR, fillcolor=255).save(TMP)
            chosen = ChosenAngle(str(TMP))
            # the segmenter must undo the injected tilt on top of the native one
            expect = native - inj
            err = chosen - expect
            rows.append(dict(author=author, page=imgPath.name, injected=inj,
                             native=native, expected=expect, chosen=chosen,
                             error=round(err, 3)))
            print("%-10s %-16s %+8.1f %+8.1f %+8.1f %+8.1f   err %+.1f"
                  % (author, imgPath.name, inj, native, expect, chosen, err))

    errs = [abs(r["error"]) for r in rows]
    within = [e for e in errs if e <= STEP]
    print()
    print("trials                    : %d (%d pages x %d angles)"
          % (len(rows), len(pages), len(INJECT)))
    print("mean absolute error       : %.2f degrees" % (sum(errs) / len(errs)))
    print("maximum absolute error    : %.2f degrees" % max(errs))
    print("within one search step    : %d of %d (%.0f%%)"
          % (len(within), len(errs), 100.0 * len(within) / len(errs)))
    print("search step / best possible: %.1f / %.2f degrees" % (STEP, STEP / 2))

    (OUT_DIR / "deskew_results.json").write_text(
        json.dumps(dict(step=STEP, rows=rows), indent=1), encoding="utf-8")
    if TMP.exists():
        TMP.unlink()
    print("\nwrote %s" % (OUT_DIR / "deskew_results.json"))


if __name__ == "__main__":
    main()
