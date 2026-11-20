#why does the segmenter not return every written line on 588's two pages?
#
#SegmentLines finds L line bands, then builds a crop for each one and keeps
#the crop only if _BuildLineCrop returns something.  a band whose ink falls
#below minInk is dropped silently, and minInk is
#       minInk = max(60, 0.12 * median(ink per line))
#so a short closing line can fall under the floor set by its own page's
#longer lines.
#
#this repeats the same steps with the rejection made visible: every band is
#reported with its ink count, the threshold it was judged against, and
#whether it survived.  every band is also written out as an image, INCLUDING
#the rejected ones, by asking for the crop again with the floor removed.
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
warnings.filterwarnings("ignore")

CP3 = Path(r"C:\Users\braun\OneDrive\Documents\GitHub\Handwriting-Robot\Software\CleanPipeline3")
sys.path.insert(0, str(CP3))
import extractiam as EX
import segment as SEG

SRC = CP3.parent.parent/"Data"/"Datasets"/"IAMpages10"/"588"
OUT = Path(__file__).resolve().parent/"dropped588"
OUT.mkdir(parents=True, exist_ok=True)

for page in ("n06-082", "n06-133"):
    png = SRC/(page+".png")
    lab = png.with_name(page+"_labels.txt")
    want = [l for l in lab.read_text(encoding="utf-8").splitlines() if l.strip()]

    #crop to the handwriting the same way extractiam does
    gray = np.asarray(Image.open(png).convert("L"), dtype=np.uint8)
    binary = EX.Binarise(gray)
    topY, botY = EX.DetectPageBounds(binary, gray)
    body = np.ascontiguousarray(gray[topY:botY, :])

    #---the steps SegmentLines takes, repeated so the rejection is visible---
    fullA, halfA, mask, ratio, best = SEG.Deskew(body)
    g = halfA.astype(np.float64)
    H, W = mask.shape
    mask, paper, P, _pr, _rm = SEG.RemovePrintedStructure(mask, g)
    mask, labIm, nc, area, _na = SEG.DropArtifactPieces(mask, ratio, P, H, W)
    P, centres = SEG.LineCentres(mask, H, None)
    cuts, cutsArr, L = SEG.SeamCuts(mask, centres, P, H, W)
    own = SEG.AssignOwnership(mask, labIm, nc, cuts, cutsArr, L, centres, P, H, W)

    r = max(8, min(40, round(H/41)))
    bgFull = SEG.Resize(SEG.BoxMean(g, r).astype(np.float32),
                        fullA.shape[1], fullA.shape[0], "linear")
    norm = np.clip(fullA.astype(np.float32)/np.maximum(bgFull, 1.0), 0, 1)*255
    f = fullA.shape[1]/W

    perLine = np.bincount(own[own >= 0], minlength=L)
    med = float(np.median(perLine[perLine > 0]))
    minInk = max(60, 0.12*med)

    print("\n%s" % page)
    print("  %d bands found, %d label lines" % (L, len(want)))
    print("  median ink per line %.0f px, so minInk = max(60, 0.12 x %.0f) = %.0f"
          % (med, med, minInk))
    print("  %-5s %-10s %-9s %-8s %s" % ("band", "ink px", "vs minInk", "kept", "label"))
    kept = []
    for n in range(L):
        #ask twice: once as the pipeline does, once with the floor removed so
        #a rejected band can still be written out and looked at
        crop = anyCrop = None
        try:
            crop = SEG._BuildLineCrop(n, own, norm, P, f, H, W, minInk)
        except ValueError:
            pass
        try:
            anyCrop = SEG._BuildLineCrop(n, own, norm, P, f, H, W, 0)
        except ValueError:
            #no ink was assigned to this band at all, so there is nothing to
            #cut out: the failure is in ownership, not in the ink threshold
            anyCrop = None
        ok = crop is not None
        if ok:
            kept.append(n)
        print("  %-5d %-10d %-9s %-8s %s"
              % (n, perLine[n], "%.2fx" % (perLine[n]/minInk),
                 "yes" if ok else ("NO - NO INK OWNED" if perLine[n] == 0
                                   else "NO - below minInk"),
                 want[len(kept)-1][:46] if ok and len(kept) <= len(want) else ""))
        if anyCrop is not None:
            a = np.asarray(anyCrop["image"], dtype=np.uint8)
            tag = "band%02d_%s" % (n, "kept" if ok else "DROPPED")
            Image.fromarray(a).save(OUT/("%s_%s.png" % (page, tag)))

    #one sheet with every band in order, the dropped ones marked
    imgs = []
    for n in range(L):
        p = OUT/("%s_band%02d_%s.png" % (page, n, "kept" if n in kept else "DROPPED"))
        if p.exists():
            imgs.append((n, Image.open(p).convert("L"), n in kept, perLine[n]))
    if imgs:
        LAB = 26
        w = max(i.width for _n, i, _k, _v in imgs)
        h = sum(i.height+LAB+4 for _n, i, _k, _v in imgs)+8
        sheet = Image.new("RGB", (max(w, 1100)+16, h), (255, 255, 255))
        d = ImageDraw.Draw(sheet)
        y = 6
        for n, im, ok, ink in imgs:
            d.text((8, y), "band %02d   ink %d px   %s"
                   % (n, ink, "kept" if ok else "DROPPED, below minInk %.0f" % minInk),
                   fill=(18, 58, 107) if ok else (170, 0, 0))
            y += LAB
            sheet.paste(im.convert("RGB"), (8, y))
            y += im.height+4
        sheet.save(OUT/("%s_all_bands.png" % page))
    print("  wrote %d band images and %s_all_bands.png" % (len(imgs), page))

print("\nall in", OUT)
