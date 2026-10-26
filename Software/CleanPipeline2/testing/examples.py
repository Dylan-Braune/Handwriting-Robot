"""Example images with their expected and recognised text, per writer.

Companion to full_results.py.  That one gives the aggregate numbers, this one
gives the pictures behind them so the tables can be checked by eye.  Writes
testing/results/EXAMPLES.md plus the png files under results/images/

  crops   real line crops, what the transcript says vs what each checkpoint read
  synth   synthesised lines, the target text vs what the reader got back
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CP2 = HERE.parent
sys.path.insert(0, str(CP2))   #every pipeline file now sits in CleanPipeline2

import numpy as np                                            # noqa: E402
from PIL import Image                                         # noqa: E402

import paths                                                  # noqa: E402
import recognise as rec                                        # noqa: E402
import synthesise as syn                                       # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

OUT = HERE / "results"
IMG = OUT / "images"
(IMG / "crops").mkdir(parents=True, exist_ok=True)
(IMG / "synth").mkdir(parents=True, exist_ok=True)
CROPS = paths.Data / "SegmentationBest" / "Crops"

N_CROPS = 5                                   # example crops per photographed writer
SENTENCE = "the gantry writes this"
NTRIES = 6
CKPTS = [("scratch", "paper_cnn_bilstm_ctc_scratch_best.npz"),
         ("joint", "paper_cnn_bilstm_ctc_joint_best.npz")]


def Labels(writer):
    out = {}
    d = paths.PersonalPages / writer.capitalize() / "Labels"
    if not d.exists():
        return out
    for f in sorted(d.glob("*_labels.txt")):
        out[f.name.replace("_labels.txt", "")] = [
            l for l in f.read_text(encoding="utf-8", errors="replace").splitlines()
            if l.strip()]
    return out


def Save(arr, path, maxWide=900):
    """write a grey array as png, shrunk only if it is wider than the page"""
    im = Image.fromarray(np.asarray(arr, dtype=np.uint8))
    if im.width > maxWide:
        im = im.resize((maxWide, max(1, round(im.height * maxWide / im.width))),
                       Image.LANCZOS)
    im.save(path)


def CropExamples():
    print("\n[crops] real line crops, expected vs read")
    readers = {k: rec.PaperCRNNNumpy(checkpointPath=rec.WEIGHTS_DIR / f)
               for k, f in CKPTS}
    inkModel = rec.AuthorClassifierCNNNumpy(
        checkpointPath=rec.WEIGHTS_DIR / "author_classifier_10new_weights.npz")
    shapeModel = rec.AuthorClassifierCNNNumpy(
        checkpointPath=rec.WEIGHTS_DIR / "author_shape_10new_weights.npz")
    inkInv = {v: k for k, v in inkModel.authorMapping.items()}
    shapeInv = {v: k for k, v in shapeModel.authorMapping.items()}

    rows = []
    for w in paths.PersonalWriters:
        labs = Labels(w)
        folder = CROPS / w.capitalize()
        if not folder.exists():
            continue
        picked = None
        for pageDir in sorted(p for p in folder.iterdir() if p.is_dir()):
            truth = labs.get(pageDir.name)
            crops = sorted(pageDir.glob("*.png"))
            if truth and len(crops) == len(truth):
                picked = (pageDir, crops, truth)
                break                          # first page that pairs cleanly
        if not picked:
            continue
        pageDir, crops, truth = picked
        # spread the picks across the page instead of taking the top few
        idx = np.linspace(0, len(crops) - 1, min(N_CROPS, len(crops))).astype(int)
        for j, i in enumerate(idx, 1):
            arr = np.asarray(Image.open(crops[i]).convert("L"), dtype=np.uint8)
            want = truth[i]
            rel = "images/crops/%s_%d.png" % (w, j)
            Save(arr, IMG / "crops" / ("%s_%d.png" % (w, j)))
            got = {k: rec.ReadText(arr, readers[k]) for k, _f in CKPTS}
            ink, pInk = rec.ClassifyImage(arr, inkModel, strokeNormalise=False)
            shape, pShape = rec.ClassifyImage(arr, shapeModel, strokeNormalise=True)
            rows.append(dict(
                writer=w, page=pageDir.name, line=int(i) + 1, image=rel,
                expected=want,
                gotJoint=got["joint"], gotScratch=got["scratch"],
                charJoint=round(100 * rec.CharAcc(got["joint"], want), 1),
                charScratch=round(100 * rec.CharAcc(got["scratch"], want), 1),
                wordJoint=round(100 * rec.WordAcc(got["joint"], want), 1),
                inkPred=inkInv.get(ink, str(ink)),
                inkConf=round(100 * float(pInk[ink]), 1),
                shapePred=shapeInv.get(shape, str(shape)),
                shapeConf=round(100 * float(pShape[shape]), 1)))
            print("    %-9s line %2d  char %5.1f%%  ink %-8s %5.1f%%  shape %-8s %5.1f%%"
                  % (w, int(i) + 1, rows[-1]["charJoint"], rows[-1]["inkPred"],
                     rows[-1]["inkConf"], rows[-1]["shapePred"],
                     rows[-1]["shapeConf"]))
    return rows


def SynthExamples():
    print("\n[synth] synthesised lines, target vs read back")
    reader = rec.PaperCRNNNumpy(
        checkpointPath=rec.WEIGHTS_DIR / "paper_cnn_bilstm_ctc_joint_best.npz")
    shapeModel = rec.AuthorClassifierCNNNumpy(
        checkpointPath=rec.WEIGHTS_DIR / "author_shape_10new_weights.npz")
    smap = shapeModel.authorMapping
    sInv = {v: k for k, v in smap.items()}

    rows = []
    for w in paths.AllWriters:
        fp = paths.Profiles / ("%s.json" % w)
        if not fp.exists() or w not in smap:
            continue
        prof = json.load(open(fp, encoding="utf-8"))
        t0 = time.time()
        traj = syn.SynthesizeJointBestOf(
            w, SENTENCE, prof, nTries=NTRIES, seed=11,
            reader=reader, authorModel=shapeModel, authorMapping=smap)
        secs = time.time() - t0
        img = syn.RenderTrajectory(traj, pxPerMm=18.0, profile=prof, uniformInk=True)
        got = rec.ReadText(img, reader)
        pred, probs = rec.ClassifyImage(img, shapeModel)
        rel = "images/synth/%s.png" % w
        Save(img, IMG / "synth" / ("%s.png" % w))
        rows.append(dict(
            writer=w, image=rel, target=SENTENCE, got=got,
            charAcc=round(100 * rec.CharAcc(got, SENTENCE), 1),
            wordAcc=round(100 * rec.WordAcc(got, SENTENCE), 1),
            predWriter=sInv.get(pred, str(pred)),
            trueConf=round(100 * float(probs[smap[w]]), 1),
            predConf=round(100 * float(probs[pred]), 1),
            strokes=len(traj.strokes), seconds=round(secs, 1)))
        print("    %-9s char %5.1f%%  pred %-8s %5.1f%%  (true %5.1f%%)  %4.1f s"
              % (w, rows[-1]["charAcc"], rows[-1]["predWriter"],
                 rows[-1]["predConf"], rows[-1]["trueConf"], secs))
    return rows


def Write(cropRows, synthRows):
    md = ["# Example images, with the expected text beside what was read\n",
          "Companion to `RESULTS.md`.  Same weights, nothing retrained.  Every "
          "image below is a real file under `results/images/`, so each number "
          "can be checked by eye.\n"]

    md.append("\n## 1. Real line crops\n")
    md.append("One page per writer, %d lines spread down it.  `expected` is the "
              "transcript typed when the page was collected.  Two readers are "
              "shown: *joint* is the checkpoint the report figures were taken "
              "on, *scratch* is the one the loader currently picks first.\n"
              % N_CROPS)
    for w in paths.PersonalWriters:
        rs = [r for r in cropRows if r["writer"] == w]
        if not rs:
            continue
        md.append("\n### %s, page %s\n" % (w, rs[0]["page"]))
        for r in rs:
            md.append("![%s line %d](%s)\n" % (w, r["line"], r["image"]))
            md.append("| | text | char %% |")
            md.append("|---|---|---|")
            md.append("| expected | `%s` | |" % r["expected"])
            md.append("| read, joint | `%s` | %s |" % (r["gotJoint"], r["charJoint"]))
            md.append("| read, scratch | `%s` | %s |" % (r["gotScratch"], r["charScratch"]))
            md.append("")
            md.append("writer id on this crop: ink says **%s** at %s%%, shape says "
                      "**%s** at %s%%\n"
                      % (r["inkPred"], r["inkConf"], r["shapePred"], r["shapeConf"]))

    md.append("\n## 2. Real crops, all examples in one table\n")
    md.append("| writer | line | expected | read, joint | char % | read, scratch | char % | ink | shape |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for r in cropRows:
        md.append("| %s | %d | `%s` | `%s` | %s | `%s` | %s | %s %s%% | %s %s%% |"
                  % (r["writer"], r["line"], r["expected"], r["gotJoint"],
                     r["charJoint"], r["gotScratch"], r["charScratch"],
                     r["inkPred"], r["inkConf"], r["shapePred"], r["shapeConf"]))

    md.append("\n## 3. Synthesised lines\n")
    md.append("Target text `%s`, best of %d, one seed, rendered at 18 px/mm then "
              "read back.  The same two networks score the search, so these are "
              "not an independent test of it.\n" % (SENTENCE, NTRIES))
    for r in synthRows:
        md.append("\n### %s\n" % r["writer"])
        md.append("![synth %s](%s)\n" % (r["writer"], r["image"]))
        md.append("| | |")
        md.append("|---|---|")
        md.append("| target | `%s` |" % r["target"])
        md.append("| read back | `%s` |" % r["got"])
        md.append("| char accuracy | %s%% |" % r["charAcc"])
        md.append("| word accuracy | %s%% |" % r["wordAcc"])
        md.append("| writer predicted | %s at %s%% |" % (r["predWriter"], r["predConf"]))
        md.append("| confidence in the true writer | %s%% |" % r["trueConf"])
        md.append("| strokes / time | %d / %s s |" % (r["strokes"], r["seconds"]))
        md.append("")

    md.append("\n## 4. Synthesised lines, all in one table\n")
    md.append("| writer | target | read back | char % | word % | predicted | conf % | true conf % | strokes | s |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in synthRows:
        md.append("| %s | `%s` | `%s` | %s | %s | %s | %s | %s | %d | %s |"
                  % (r["writer"], r["target"], r["got"], r["charAcc"], r["wordAcc"],
                     r["predWriter"], r["predConf"], r["trueConf"], r["strokes"],
                     r["seconds"]))

    (OUT / "EXAMPLES.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (OUT / "examples.json").write_text(
        json.dumps(dict(crops=cropRows, synth=synthRows), indent=1), encoding="utf-8")
    print("\nwrote %s" % (OUT / "EXAMPLES.md"))
    print("wrote %s" % (OUT / "examples.json"))


if __name__ == "__main__":
    cache = OUT / "_crops_cache.json"
    if cache.exists():
        c = json.load(open(cache, encoding="utf-8"))   #skip the reread if already done
        print("\n[crops] reusing %d cached rows" % len(c))
    else:
        c = CropExamples()
        cache.write_text(json.dumps(c, indent=1), encoding="utf-8")
    s = SynthExamples()
    Write(c, s)
