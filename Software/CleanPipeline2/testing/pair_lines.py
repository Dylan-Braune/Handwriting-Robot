"""Pair cached line images with their transcript, for the database writers.

The line images sit in NOGIT/line_cache_authors10 keyed by page.  The text of
each line is recoverable from NOGIT/GlyphCache10, because every cached line
records the character at each cut.  What is NOT recorded is which image goes
with which text, and the two caches are in different orders.

So the pairing is done by content: read every image with the recogniser, then
greedily match each reading to its closest cached text.  The reader is around
98% accurate on these writers, so the match is unambiguous, and the match
score is reported so a bad pairing cannot pass silently.

Holdout pages are skipped, so nothing here was used to measure anything.
"""
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CP2 = HERE.parent
sys.path.insert(0, str(CP2))

import numpy as np                                              # noqa: E402
from PIL import Image                                            # noqa: E402

import paths                                                     # noqa: E402
import recognise as rec                                          # noqa: E402

SHARDS = paths.Cache / "line_cache_authors10" / "shards"
GLYPHS = paths.Cache / "line_cache_authors10"
CACHE = paths.Cache / "GlyphCache10"
HOLDOUT = paths.Cache / "holdout_test_pages"
MIN_MATCH = 0.75          # a reading must look this much like its text to count


def HoldoutPages(writer):
    """page stems this writer has a holdout page for, which are skipped"""
    return {p.name.split("_", 1)[1].replace(".png", "")
            for p in HOLDOUT.glob("%s_*.png" % writer)}


def CachedTexts(writer):
    """the transcript of every cached line, read back from its cut characters"""
    p = CACHE / ("%s.pkl" % writer)
    if not p.exists():
        return []
    with open(p, "rb") as f:
        parsed = pickle.load(f)
    out = []
    for _glyphs, st in parsed:
        cuts = st.get("cuts") or []
        t = "".join(c[1] for c in cuts).strip()
        if len(t) >= 8:
            out.append(t)
    return out


def LineImages(writer, skip):
    """every cached line image for this writer, excluding holdout pages"""
    out = []
    import io
    for f in sorted(SHARDS.glob("*.pkl")):
        with open(f, "rb") as fh:
            shard = pickle.load(fh)
        for key, pngs in shard.items():
            if not key.startswith(writer + "/"):
                continue
            stem = key.split("/")[1].replace(".png", "")
            if stem in skip:
                continue
            for b in pngs:
                g = np.asarray(Image.open(io.BytesIO(b)).convert("L"),
                               dtype=np.uint8)
                out.append((stem, g))
    return out


def Items(writer, reader, verbose=True):
    """(gray,text) per line, paired by what the reader sees"""
    skip = HoldoutPages(writer)
    imgs = LineImages(writer, skip)
    texts = CachedTexts(writer)
    if not imgs or not texts:
        return [], dict(images=len(imgs), texts=len(texts), matched=0)
    #read every image once, then match readings to texts
    reads = [rec.ReadText(g, reader) for _s, g in imgs]
    pairs = []
    for i, r in enumerate(reads):
        for j, t in enumerate(texts):
            pairs.append((rec.CharAcc(r, t), i, j))
    pairs.sort(reverse=True)
    usedI, usedJ, items, scores = set(), set(), [], []
    for s, i, j in pairs:
        if s < MIN_MATCH or i in usedI or j in usedJ:
            continue
        usedI.add(i)
        usedJ.add(j)
        items.append((imgs[i][1], texts[j]))
        scores.append(s)
    info = dict(images=len(imgs), texts=len(texts), matched=len(items),
                skippedPages=sorted(skip),
                matchMean=round(float(np.mean(scores)), 4) if scores else 0.0,
                matchMin=round(float(np.min(scores)), 4) if scores else 0.0)
    if verbose:
        print("    %-8s %3d images, %3d texts, matched %3d  "
              "(match score mean %.3f, worst %.3f)  holdout skipped %s"
              % (writer, info["images"], info["texts"], info["matched"],
                 info["matchMean"], info["matchMin"],
                 ",".join(info["skippedPages"]) or "none"))
    return items, info


if __name__ == "__main__":
    reader = rec.PaperCRNNNumpy(
        checkpointPath=rec.WEIGHTS_DIR / "paper_cnn_bilstm_ctc_joint_best.npz")
    print("pairing check, database writers:")
    for w in paths.DatabaseWriters:
        Items(w, reader)
