"""Hybrid: the old SegmentPage.py front end (page mask, lighting, deskew, edge/rule/speckle removal,
mess detection) + the lean line finder (SegmentLean.py): spacing P -> least-ink boundaries ->
pieces owned by lines -> loose letter parts follow their letter body -> crops.

Usage: python SegmentHybrid.py page.jpg outdir
"""
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import SegmentPage as SP
from SegmentLean import _any, _box, _label, _pitch, _seam


def _front_end(path):
    """SegmentPage.ProcessPage up to the cleaned ink image (no recovery pass, no line grouping)."""
    rgb = SP.LoadImage(path)
    pageMask = SP.DetectPageMask(rgb)
    gray = SP.RgbToGray(rgb)
    illum = SP.CorrectIllumination(gray)
    ink0 = SP.BinarizeInk(illum, pageMask) & ~SP.RedInkMask(rgb)
    ink0 = SP.RemoveSpeckles(ink0)
    angle = SP.EstimateSkew(ink0, searchRange=8.0)
    applied = 0.0
    for rng in (None, 3.0):
        a = angle if rng is None else SP.EstimateSkew(ink0, searchRange=rng)
        if abs(a) < 0.15:
            break
        rgb = SP.Rotate(rgb, -a, fill=255)
        pageMask = SP.Rotate(pageMask.astype(np.uint8), -a, isMask=True).astype(bool)
        gray = SP.RgbToGray(rgb)
        illum = SP.CorrectIllumination(gray)
        ink0 = SP.BinarizeInk(illum, pageMask) & ~SP.RedInkMask(rgb)
        ink0 = SP.RemoveSpeckles(ink0)
        applied += a
    _, rc0 = SP.ComponentStats(ink0)
    rH0 = SP.EstimateTextHeight(rc0)
    illumC = SP.CorrectIllumination(SP._InkGray(rgb))
    ink0 = SP.FaintFilterRuleAware(ink0, illum, rH0)
    ink0 = SP.RemoveEdgeComponentsWide(ink0, pageMask, rH0)
    _, roughComps = SP.ComponentStats(ink0)
    roughH = SP.EstimateTextHeight(roughComps)
    ink0 = SP.RemoveOffPageColumns(ink0, roughH, alsoClip=ink0)
    ink, rm1 = SP.RemoveRuleLines(ink0, roughH, illum=illum)
    ink, rm2 = SP.RemoveRuleLines(ink, roughH, illum=illum)
    ink = SP.RemoveSpeckles(ink, minSize=12)
    ink = SP.StripSparseRuleNetworks(ink, roughH)
    ink = SP.RemoveSpeckles(ink, minSize=12)
    return ink, np.minimum(illum, illumC), float(applied)


def _drop_mess(ink):
    """Remove diagram/scribble pieces the way SegmentPage.GroupLines decides them."""
    labels, comps = SP.ComponentStats(ink)
    textH = SP.EstimateTextHeight(comps)
    for c in comps:
        c["mess"] = SP.ScoreComponentMess(c, labels, textH)
    cand = [c for c in comps if c["mess"] >= 0.5 and c["area"] > textH * textH * 0.8 and c["h"] > textH * 1.2]

    def veto(c):                                           # a big heading letter among normal text is not mess
        neigh = []
        for o in comps:
            if o is c or o["mess"] >= 0.5 or o["h"] < textH * 0.35 or o["h"] > textH * 2.4 or o["area"] < textH * textH * 0.2:
                continue
            if c["x"] <= o["cx"] <= c["x"] + c["w"]:
                continue
            if max(0, min(c["y"] + c["h"], o["y"] + o["h"]) - max(c["y"], o["y"])) < 0.6 * o["h"]:
                continue
            if max(o["x"] - (c["x"] + c["w"]), c["x"] - (o["x"] + o["w"])) < textH * 6.0:
                neigh.append(o)
        return len(neigh) >= 3 and c["h"] <= 1.9 * float(np.percentile([o["h"] for o in neigh], 75))

    mess = [c for c in cand if c["mess"] >= 0.85 or not veto(c)]
    ink = ink.copy()
    for c in mess:
        ink[labels == c["id"]] = False
    return ink, len(mess)


def SegmentLines(path):
    t0 = time.time()
    ink, render, angle = _front_end(path)
    ink, nMess = _drop_mess(ink)
    Hf, Wf = ink.shape
    H, W = Hf // 2, Wf // 2
    mask = ink[:H * 2, :W * 2].reshape(H, 2, W, 2).mean((1, 3)) >= 0.25            # half resolution
    f = Wf / W

    rc = max(1, int(0.1 * _pitch(mask)[0]))
    dil = _any(_any(mask, rc, 0), rc, 1)
    closed = ~_any(_any(~dil, rc, 0), rc, 1)                     # tiny pen-lift gaps joined
    lab, nc = _label(closed)

    # 3. line spacing and centres
    P, sm = _pitch(mask)
    peaks = [i for i in range(1, H - 1) if sm[i] >= sm[i - 1] and sm[i] > sm[i + 1] and sm[i] > 0.06 * sm.max()]
    centres = []
    for c in sorted(peaks, key=lambda i: -sm[i]):
        if all(abs(c - o) >= 0.6 * P for o in centres):
            centres.append(c)
    centres.sort()

    # 4. rough boundaries: least-ink path between neighbouring centres
    cuts = [np.full(W, max(0, centres[0] - int(0.75 * P)))]
    m = (_box(mask.astype(np.float64), 1) > 0).astype(np.float64)
    for c0, c1 in zip(centres, centres[1:]):
        yy = np.arange(c0, c1 + 1)[:, None]
        mid, half_w = (c0 + c1) / 2.0, max(1.0, (c1 - c0) / 2.0)
        cuts.append(c0 + _seam(4.0 * m[c0:c1 + 1] + 0.05 * ((yy - mid) / half_w) ** 2))
    cuts.append(np.full(W, min(H - 1, centres[-1] + int(0.75 * P))))
    L = len(cuts) - 1
    cutsArr = np.array(cuts)

    # 5. pieces (tiny pen-lift gaps joined), owned by the line holding most of their middle band
    ys, xs = np.nonzero(mask)
    pl = lab[ys, xs]
    Y = np.arange(H)[:, None]
    region = sum((Y >= cuts[n][None, :]).astype(np.int16) for n in range(1, L)) if L > 1 else np.zeros((H, W), np.int16)
    rp = region[ys, xs]
    ok = (ys >= cutsArr[0][xs]) & (ys < cutsArr[-1][xs])
    area = np.bincount(pl, minlength=nc + 1).astype(float)
    fr = (ys - cutsArr[rp, xs]) / np.maximum(cutsArr[rp + 1, xs] - cutsArr[rp, xs], 1)
    core = ok & (fr >= 0.25) & (fr < 0.75)
    cnt = np.bincount(pl[ok] * L + rp[ok], minlength=(nc + 1) * L).reshape(nc + 1, L)
    ccnt = np.bincount(pl[core] * L + rp[core], minlength=(nc + 1) * L).reshape(nc + 1, L)
    owner = np.where(ccnt.sum(1) > 0, ccnt.argmax(1), cnt.argmax(1))
    owner[area == 0] = -1
    top, bot = np.full(nc + 1, H), np.zeros(nc + 1, int)
    np.minimum.at(top, pl, ys)
    np.maximum.at(bot, pl, ys)
    xmin, xmax = np.full(nc + 1, W), np.zeros(nc + 1, int)
    np.minimum.at(xmin, pl, xs)
    np.maximum.at(xmax, pl, xs)
    srt = np.sort(ccnt, 1)
    split = (srt[:, -2] >= np.maximum(0.25 * srt[:, -1], 0.1 * P * P)) if L > 1 else np.zeros(nc + 1, bool)
    # a loose part of a letter (i-dot, g/y tail, quote mark) hangs outside its line's middle band:
    # give it to the nearest real letter body overlapping it in x, above or below, within a short gap
    coreFrac = ccnt.sum(1) / np.maximum(area, 1)
    loose = (area > 0) & (area < 0.4 * P * P) & (coreFrac < 0.4) & ~split
    body = (area >= 0.065 * P * P) & ~loose & (owner >= 0)
    nLoose = nMoved = 0
    for l in np.nonzero(loose)[0]:
        nLoose += 1
        xa, xb = max(0, xmin[l] - 3), xmax[l] + 4
        for gap in range(1, int(0.35 * P) + 1):
            hit = []
            for yy in (bot[l] + gap, top[l] - gap):                   # below first, then above
                if 0 <= yy < H:
                    row = lab[yy, xa:xb]
                    hit += [v for v in np.unique(row[row > 0]) if v != l and body[v]]
            if hit:
                nMoved += owner[hit[0]] != owner[l]
                owner[l] = owner[hit[0]]
                break
    own = np.full((H, W), -1, np.int16)
    own[ys, xs] = np.where(split[pl], rp, owner[pl])
    own[ys[~ok], xs[~ok]] = -1

    # crops from the full-resolution, illumination-corrected page
    norm = np.clip(render, 0, 255)
    perLine = np.bincount(own[own >= 0], minlength=L)
    minInk = max(60, 0.12 * np.median(perLine[perLine > 0]))
    lines = []
    for n in range(L):
        ink_n = own == n
        yy, xx = np.nonzero(ink_n)
        if len(yy) < minInk or yy.max() - yy.min() < 0.45 * P:
            continue
        pad = int(0.15 * P)
        y0, y1 = max(0, yy.min() - pad), min(H, yy.max() + 1 + pad)
        x0, x1 = max(0, xx.min() - pad), min(W, xx.max() + 1 + pad)
        Y0, Y1, X0, X1 = int(y0 * f), min(Hf, int(y1 * f)), int(x0 * f), min(Wf, int(x1 * f))
        nz = _box(ink_n[y0:y1, x0:x1].astype(np.float64), 3) > 0
        nz = np.asarray(Image.fromarray(nz.astype(np.uint8) * 255).resize((X1 - X0, Y1 - Y0))) > 127
        crop = np.where(nz, norm[Y0:Y1, X0:X1], 255).astype(np.uint8)
        lines.append(dict(order=len(lines), image=Image.fromarray(crop), bbox=(X0, Y0, X1, Y1)))

    ov = Image.fromarray(np.clip(norm[::2, ::2], 0, 255).astype(np.uint8)).convert("RGB")
    d = ImageDraw.Draw(ov)
    for n, cu in enumerate(cuts):
        d.line(list(zip(range(W), cu.tolist())), fill=(255, 0, 0) if n % 2 else (0, 140, 255), width=2)
    return lines, ov, dict(skew=round(angle, 2), pitch=int(P), messPiecesRemoved=nMess, nLines=len(lines),
                           seconds=round(time.time() - t0, 1))


if __name__ == "__main__":
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    lines, ov, info = SegmentLines(src)
    for ln in lines:
        ln["image"].save(out / f"{ln['order']:02d}.png")
    ov.save(out / "_cuts_overlay.jpg", quality=88)
    print(info)
