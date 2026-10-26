"""Text accuracy on REAL line crops: my resize against PIL's.

This is the measurement the report's 98.61% figure comes from, so it is the
one that settles whether removing the imaging library cost any accuracy.

Same crops, same recogniser, same ground-truth labels.  The ONLY thing that
differs between the two runs is how the crop is scaled onto the 64x640
classifier input: MyImage.FitFixedCanvas, or PIL's resize + paste.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIPE = HERE.parent.parent.parent
sys.path.insert(0, str(PIPE))

import numpy as np                                            # noqa: E402
from PIL import Image                                         # noqa: E402

import Classify                                               # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

CNN = PIPE.parent / "CNN"
CROPS = CNN / "SegmentationBest" / "Crops"
LABELS = CNN / "PersonalDataset"
AUTHORS = ["Abhinav", "Dylan", "Owen", "Thiya", "Yeukita"]

H, W = Classify.INPUT_HEIGHT, Classify.INPUT_WIDTH


def PilFit(pil_img, height=H, width=W, pad_value=255):
    """The code that was in ResizeLineImageFixed before the migration."""
    img = pil_img.convert("L")
    w, h = img.size
    if w <= 0 or h <= 0:
        return Image.new("L", (width, height), pad_value)
    s = min(height / h, width / w)
    newW, newH = max(1, round(w * s)), max(1, round(h * s))
    resized = img.resize((newW, newH), Image.BILINEAR)
    canvas = Image.new("L", (width, height), pad_value)
    canvas.paste(resized, (0, max(0, (height - newH) // 2)))
    return canvas


def LoadLabels(author):
    """page stem -> list of transcription lines"""
    out = {}
    d = LABELS / author / "Labels"
    for f in sorted(d.glob("*_labels.txt")):
        stem = f.name.replace("_labels.txt", "")
        out[stem] = [ln for ln in
                     f.read_text(encoding="utf-8", errors="replace").splitlines()
                     if ln.strip()]
    return out


def main():
    reader = Classify.PaperCRNNNumpy()
    realFit = Classify.ResizeLineImageFixed

    results = {}
    for tag in ("mine", "pil"):
        if tag == "pil":
            Classify.ResizeLineImageFixed = PilFit
        else:
            Classify.ResizeLineImageFixed = realFit

        charSum = wordSum = n = 0
        perAuthor = {}
        for author in AUTHORS:
            labels = LoadLabels(author)
            aC = aW = aN = 0
            for pageDir in sorted((CROPS / author).iterdir()):
                if not pageDir.is_dir():
                    continue
                truth = labels.get(pageDir.name)
                if not truth:
                    continue
                crops = sorted(pageDir.glob("*.png"))
                # only score pages where the crop count matches the label
                # count, which is how the 669-crop figure was built: a page
                # that segmented into a different number of lines has no
                # reliable line-to-label correspondence
                if len(crops) != len(truth):
                    continue
                for cp, want in zip(crops, truth):
                    img = Image.open(cp)
                    got = Classify.ReadText(img, reader)
                    aC += Classify.CharAcc(got, want)
                    aW += Classify.WordAcc(got, want)
                    aN += 1
            if aN:
                perAuthor[author] = (100 * aC / aN, 100 * aW / aN, aN)
                charSum += aC
                wordSum += aW
                n += aN
        results[tag] = (100 * charSum / max(n, 1), 100 * wordSum / max(n, 1),
                        n, perAuthor)

    Classify.ResizeLineImageFixed = realFit

    print("crops scored: %d (pages whose crop count matches the label count)"
          % results["mine"][2])
    print()
    print("%-10s %18s %18s" % ("writer", "char  PIL / mine", "word  PIL / mine"))
    print("-" * 50)
    for a in AUTHORS:
        if a not in results["mine"][3]:
            continue
        cM, wM, nA = results["mine"][3][a]
        cP, wP, _ = results["pil"][3][a]
        print("%-10s %8.2f%% /%8.2f%% %8.2f%% /%8.2f%%   (%d)"
              % (a, cP, cM, wP, wM, nA))
    print("-" * 50)
    cM, wM, n, _ = results["mine"]
    cP, wP, _, _ = results["pil"]
    print("%-10s %8.2f%% /%8.2f%% %8.2f%% /%8.2f%%   (%d)"
          % ("OVERALL", cP, cM, wP, wM, n))
    print()
    print("char accuracy change, mine minus PIL: %+.3f points" % (cM - cP))
    print("word accuracy change, mine minus PIL: %+.3f points" % (wM - wP))


if __name__ == "__main__":
    main()
