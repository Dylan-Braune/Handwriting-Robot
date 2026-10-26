"""Compare every MyImage function against the PIL call it replaces.

Uses a real page crop and a deliberately ASYMMETRIC test pattern, because a
symmetric pattern hides a mirrored or transposed result -- that is exactly
how an inverted Rotate slipped through an earlier check.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIPE = HERE.parent.parent.parent
sys.path.insert(0, str(PIPE))

import numpy as np                                            # noqa: E402
from PIL import Image, ImageDraw, ImageFilter                 # noqa: E402

import MyImage                                                # noqa: E402
import AuthorsConfig as AC                                    # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

FAILED = []


def report(name, mine, theirs, tol, note=""):
    a = np.asarray(mine, dtype=np.float64)
    b = np.asarray(theirs, dtype=np.float64)
    if a.shape != b.shape:
        print("%-34s SHAPE MISMATCH %s vs %s" % (name, a.shape, b.shape))
        FAILED.append(name)
        return
    diff = np.abs(a - b)
    mx, mean = diff.max(), diff.mean()
    inkA = float((a < 128).mean())
    inkB = float((b < 128).mean())
    ok = mx <= tol
    print("%-34s max %7.2f  mean %6.3f  ink %.4f/%.4f  %s %s"
          % (name, mx, mean, inkA, inkB, "ok" if ok else "FAIL", note))
    if not ok:
        FAILED.append(name)


def asymmetric(h=97, w=263):
    """A pattern with no symmetry of any kind."""
    im = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(im)
    d.rectangle([3, 3, 40, 12], fill=0)          # bar in the top-left only
    d.ellipse([200, 60, 250, 90], fill=40)       # blob bottom-right only
    d.line([(10, 90), (120, 20)], fill=90, width=3)
    d.text((60, 40), "Fq7", fill=0)
    a = np.asarray(im, dtype=np.uint8).copy()
    a[0, :] = 10                                  # distinct first row
    a[:, 0] = 60                                  # distinct first column
    return a


def main():
    pat = asymmetric()
    pilPat = Image.fromarray(pat)

    # a real crop as well, so the test is not only synthetic
    page = None
    for a in AC.PERSONAL_AUTHORS:
        for p, _ in AC.PersonalAuthorPages(a):
            page = np.asarray(Image.open(p).convert("L"), dtype=np.uint8)
            break
        break
    real = page[400:900, 300:1500]
    pilReal = Image.fromarray(real)

    print("== Resize, shrink (the case that matters: crops are shrunk)")
    for (nw, nh) in [(64, 24), (131, 48), (900, 300)]:
        report("Resize linear %dx%d" % (nw, nh),
               MyImage.Resize(real, nw, nh, 'linear'),
               pilReal.resize((nw, nh), Image.BILINEAR), 64)

    print("\n== Resize, grow")
    report("Resize linear grow",
           MyImage.Resize(pat, 400, 150, 'linear'),
           pilPat.resize((400, 150), Image.BILINEAR), 8)

    print("\n== Halve (box), the 2:1 page reduction")
    report("Halve vs PIL BOX",
           MyImage.Halve(real),
           pilReal.resize((real.shape[1] // 2, real.shape[0] // 2), Image.BOX), 1)

    print("\n== Rotate (asymmetric pattern, catches a sign flip)")
    for ang in (-3.0, -0.5, 1.5, 4.0):
        report("Rotate %+0.1f deg" % ang,
               MyImage.Rotate(pat, ang, fill=255),
               pilPat.rotate(ang, Image.BILINEAR, fillcolor=255), 2)

    print("\n== Warp, affine (shear and rotate augmentation)")
    import math
    for angleDeg, shearDeg in [(0.0, 20.0), (2.5, 0.0), (-2.0, -15.0)]:
        w0, h0 = pilPat.size
        cx, cy = w0 / 2.0, h0 / 2.0
        ang, sh = math.radians(angleDeg), math.radians(shearDeg)
        R = np.array([[math.cos(ang), -math.sin(ang), 0],
                      [math.sin(ang), math.cos(ang), 0], [0, 0, 1]])
        Sh = np.array([[1, math.tan(sh), 0], [0, 1, 0], [0, 0, 1]])
        T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
        T2 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
        Minv = np.linalg.inv(T2 @ R @ Sh @ T1)
        co = (Minv[0, 0], Minv[0, 1], Minv[0, 2],
              Minv[1, 0], Minv[1, 1], Minv[1, 2])
        report("Warp affine a=%+.1f s=%+.1f" % (angleDeg, shearDeg),
               MyImage.Warp(pat, co, fill=255),
               pilPat.transform(pilPat.size, Image.AFFINE, co,
                                resample=Image.BILINEAR, fillcolor=255), 2)

    print("\n== Warp, perspective")
    w0, h0 = pilPat.size
    start = [(0, 0), (w0, 0), (w0, h0), (0, h0)]
    end = [(6, 4), (w0 - 9, 2), (w0 - 3, h0 - 7), (5, h0 - 2)]
    mat = []
    for (x, y), (X, Y) in zip(end, start):
        mat.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
        mat.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
    co8 = tuple(np.linalg.solve(np.array(mat, dtype=np.float64),
                                np.array(start, dtype=np.float64).reshape(8)))
    report("Warp perspective",
           MyImage.Warp(pat, co8, fill=255),
           pilPat.transform(pilPat.size, Image.PERSPECTIVE, co8,
                            resample=Image.BILINEAR, fillcolor=255), 3)

    print("\n== Blur (PIL's radius is the gaussian sigma)")
    for sig in (0.8, 2.0, 4.0):
        report("Blur sigma %.1f" % sig,
               MyImage.Blur(real, sig),
               pilReal.filter(ImageFilter.GaussianBlur(radius=sig)), 12)

    print("\n== DrawPolyline vs ImageDraw.line + round caps")
    strokes = [[(12.0, 20.0), (60.0, 55.0), (130.0, 18.0), (190.0, 70.0)],
               [(20.0, 80.0), (240.0, 88.0)],
               [(100.0, 10.0), (100.0, 90.0)]]
    for wpx in (2, 3, 5):
        mineImg = np.full((100, 260), 255, np.uint8)
        for s in strokes:
            MyImage.DrawPolyline(mineImg, s, 0, wpx)
        ref = Image.new("L", (260, 100), 255)
        dr = ImageDraw.Draw(ref)
        for s in strokes:
            dr.line(s, fill=0, width=wpx, joint='curve')
            rr = wpx / 2.0
            if rr >= 1.0:
                for (px, py) in (s[0], s[-1]):
                    dr.ellipse([px - rr, py - rr, px + rr, py + rr], fill=0)
        mi = float((np.asarray(mineImg) < 128).mean())
        ri = float((np.asarray(ref) < 128).mean())
        dis = float(np.mean((np.asarray(mineImg) < 128)
                            != (np.asarray(ref) < 128)))
        print("%-34s ink %.4f/%.4f  pixels differing %.3f%%  %s"
              % ("DrawPolyline width %d" % wpx, mi, ri, 100 * dis,
                 "ok" if dis < 0.02 else "CHECK"))

    print()
    if FAILED:
        print("FAILED: %s" % ", ".join(FAILED))
    else:
        print("all numeric checks within tolerance")


if __name__ == "__main__":
    main()
