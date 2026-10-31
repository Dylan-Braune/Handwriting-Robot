#u23539195 Dylan Braune.  stroke normalise every crop ONCE, per pen width.
#
#   python strokenorm_prepare.py
#
#AuthorLabeledView calls StrokeNormalize inside __getitem__ (train.py:1328),
#so every crop is skeletonised again on every epoch.  measured on this corpus,
#per epoch of 1031 train and 155 held-out crops:
#
#   network forward/backward   245.5 s   58%
#   StrokeNormalize            158.7 s   37%
#   decode + resize             28.0 s    7%
#   augment                      3.9 s    1%
#   modelled total             436.1 s   (the measured epoch was 426 s)
#
#the result depends only on the crop and the pen width, never on the epoch, so
#fifty epochs repeat the same skeletonisation fifty times for nothing.  it is
#also CPU work that no gpu can touch, which makes it a hard floor: without
#this step a gpu run still waits 186.7 s an epoch on skeletonising and
#resizing, so it could never beat 2.3x however fast the card is.
#
#this does it once per width and writes the finished 1xHxW crops to an .npz.
#training then only has to augment and batch them, which is the 3.8 ms part.
#the augmentation still runs per epoch, as it must -- it is random, and
#freezing it would hand the network the same distorted crop every time.
#
#the .npz also makes the sweep portable: it carries the crops, the labels and
#the held-out split, so Colab needs this file and the repo, and none of the
#NOGIT caches.
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import paths
import recognise as REC
import train as TR

LEVELS = [0.06, 0.09, 0.11, 0.17]
DROP = {"151", "551"}
OUT_DIR = paths.Cache/"StrokeNormSweep"


#----------------------------------------------------------------------
#the cache holds crops in TWO formats and train.DecodeCrop only reads one.
#IAMLineDatasetRaw stores raw PNG bytes (train.py:1253) while
#AddPersonalSamples stores zlib compressed .npy (train.py:1401), and
#DecodeCrop only zlib-decompresses, so every IAM crop raises
#   zlib.error: Error -3 while decompressing data: incorrect header check
#read whichever it is.  train.py itself is left alone; it needs the same
#change before its own namer training can touch IAM data.
#----------------------------------------------------------------------
_Original = TR.DecodeCrop


def DecodeCropEither(raw):
    try:
        return _Original(raw)
    except Exception:
        from PIL import Image
        import io as _io
        return np.asarray(Image.open(_io.BytesIO(raw)).convert("L"), dtype=np.uint8)


TR.DecodeCrop = DecodeCropEither


def BuildDataset():
    base = TR.IAMLineDatasetRaw(rootDir=str(TR.DEFAULT_AUTHOR_DATA_DIR),
                                cacheDir=str(TR.DEFAULT_AUTHOR_CACHE_DIR))
    keepDb = [w for w in paths.DatabaseWriters if w not in DROP]
    before = len(base.samples)
    base.samples = [s for s in base.samples
                    if s["page_key"].split("/")[0] in keepDb]
    print("[IAM]      kept %d/%d for %s" % (len(base.samples), before, keepDb),
          flush=True)
    TR.AddPersonalSamples(base)
    keep = set(keepDb)|{w for w in paths.PersonalWriters if w not in DROP}
    base.samples = [s for s in base.samples if s["page_key"].split("/")[0] in keep]
    base.author_folders = sorted(keep)
    base.author_to_idx = {a: i for i, a in enumerate(base.author_folders)}
    print("[Authors]  %d: %s" % (len(base.author_folders), base.author_folders),
          flush=True)
    print("[Samples]  %d lines" % len(base.samples), flush=True)
    return base


def Main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = BuildDataset()
    trainIdx, valIdx = base.HoldoutSplitIndices()
    if not valIdx:
        raise RuntimeError("No holdout lines found.")
    print("[Split]    %d train, %d held out\n" % (len(trainIdx), len(valIdx)),
          flush=True)

    #decode once, outside the per-width loop: the raw crop does not depend on
    #the pen width and decoding it four times would be four times the cost
    order = list(trainIdx)+list(valIdx)
    isVal = np.array([0]*len(trainIdx)+[1]*len(valIdx), dtype=np.uint8)
    print("[Decode]   %d crops ..." % len(order), end="", flush=True)
    t0 = time.time()
    raw = [np.asarray(TR.DecodeCrop(base.samples[i]["imageBytes"]), dtype=np.uint8)
           for i in order]
    labels = np.array([base.author_to_idx[base.samples[i]["page_key"].split("/")[0]]
                       for i in order], dtype=np.int64)
    keys = np.array([base.samples[i]["page_key"] for i in order])
    print(" %.1f s" % (time.time()-t0), flush=True)

    for pw in LEVELS:
        t0 = time.time()
        out = np.zeros((len(raw), REC.INPUT_HEIGHT, REC.INPUT_WIDTH), dtype=np.uint8)
        fellBack = 0
        for n, g in enumerate(raw):
            norm = REC.StrokeNormalize(g, penWidthXh=pw)
            if norm is None:
                fellBack += 1       #no usable ink, same fallback the view uses
                norm = g
            out[n] = np.asarray(REC.ResizeLineImageFixed(norm), dtype=np.uint8)
            if (n+1) % 200 == 0:
                print("  [pw %.2f] %d/%d" % (pw, n+1, len(raw)), flush=True)

        path = OUT_DIR/("strokenorm_data_pw%03d.npz" % round(pw*1000))
        np.savez_compressed(path, images=out, labels=labels, isVal=isVal,
                            keys=keys, authorNames=np.array(base.author_folders),
                            penWidth=np.array(pw))
        mb = path.stat().st_size/1e6
        print("[pw %.2f]  %d crops, %d fell back to the raw crop, %.1f min, "
              "%.1f MB -> %s" % (pw, len(raw), fellBack, (time.time()-t0)/60,
                                 mb, path.name), flush=True)

    print("\nupload these four .npz to Colab (or Drive) together with the repo.")
    print("they carry the crops, the labels, the writer names and the split,")
    print("so Colab needs no NOGIT cache and no dataset.")


if __name__ == "__main__":
    Main()
