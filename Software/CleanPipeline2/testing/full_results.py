"""Per writer results for every stage of the pipeline.

Uses the EXISTING trained weights, nothing is retrained here.  Writes
testing/results/full_results.json and testing/results/RESULTS.md

Stages measured
  1 segmentation      detected vs hand typed line count, time, pitch, angle
  2 recognition       character and word accuracy on real line crops
  3 writer id, ink    per writer accuracy on real crops
  4 writer id, shape  the same crops after stroke normalising
  5 style profile     what was extracted, and how trustworthy it is
  6 synthesis         what the two judges say about synthesised output
  7 g-code round trip geometry in vs geometry read back out
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
import segment                                                # noqa: E402
import recognise as rec                                       # noqa: E402
import synthesise as syn                                      # noqa: E402
import reproduce as rep                                       # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
CNN = paths.Data
CROPS = CNN / "SegmentationBest" / "Crops"

SENTENCES = ["the gantry writes this", "measure before opinion"]
NTRIES = 6
R = {}          # writer -> {stage: {...}}


def Row(w):
    return R.setdefault(w, {})


def Labels(writer):
    """page stem -> list of transcription lines, for a photographed writer"""
    out = {}
    d = paths.PersonalPages / writer.capitalize() / "Labels"
    if not d.exists():
        return out
    for f in sorted(d.glob("*_labels.txt")):
        out[f.name.replace("_labels.txt", "")] = [
            l for l in f.read_text(encoding="utf-8", errors="replace").splitlines()
            if l.strip()]
    return out


# ---------------------------------------------------------------- stage 1
def Stage1Segmentation():
    print("\n[1] segmentation, photographed writers only (needs pages + labels)")
    for w in paths.PersonalWriters:
        labs = Labels(w)
        pages = det = truth = 0
        secs, pitches, angles = [], [], []
        for img, _lab in paths.PersonalWriterPages(w):
            t = labs.get(img.stem)
            if not t:
                continue
            arr = np.asarray(Image.open(img).convert("L"), dtype=np.uint8)
            t0 = time.time()
            lines, _dbg, info = segment.SegmentLines(arr)
            secs.append(time.time() - t0)
            pages += 1
            det += info["nLines"]
            truth += len(t)
            pitches.append(info["pitch"])
            angles.append(info["angle"])
        if not pages:
            continue
        Row(w)["segmentation"] = dict(
            pages=pages, linesDetected=det, linesTrue=truth,
            exact=det == truth,
            secondsMean=round(float(np.mean(secs)), 2),
            secondsMin=round(float(np.min(secs)), 2),
            secondsMax=round(float(np.max(secs)), 2),
            pitchMin=int(min(pitches)), pitchMax=int(max(pitches)),
            angleMin=float(min(angles)), angleMax=float(max(angles)))
        print("    %-9s %2d pages  %3d/%3d lines  %.1f-%.1f s  pitch %d-%d  angle %+.1f..%+.1f"
              % (w, pages, det, truth, min(secs), max(secs),
                 min(pitches), max(pitches), min(angles), max(angles)))


# ------------------------------------------------------------- stages 2-4
TEXT_CKPTS = [("scratch","paper_cnn_bilstm_ctc_scratch_best.npz"),
              ("joint","paper_cnn_bilstm_ctc_joint_best.npz")]


def Stages234():
    print("\n[2,3,4] recognition and writer id, on real line crops")
    readers = {k:rec.PaperCRNNNumpy(checkpointPath=rec.WEIGHTS_DIR/f)
               for k,f in TEXT_CKPTS}
    inkModel = rec.AuthorClassifierCNNNumpy(
        checkpointPath=rec.WEIGHTS_DIR / "author_classifier_10new_weights.npz")
    shapeModel = rec.AuthorClassifierCNNNumpy(
        checkpointPath=rec.WEIGHTS_DIR / "author_shape_10new_weights.npz")
    inkMap, shapeMap = inkModel.authorMapping, shapeModel.authorMapping

    for w in paths.PersonalWriters:
        labs = Labels(w)
        folder = CROPS / w.capitalize()
        if not folder.exists():
            continue
        acc = {k:[0.0,0.0] for k,_ in TEXT_CKPTS}   #char and word sums per checkpoint
        n = 0
        inkOk = shapeOk = 0
        tRec = []
        for pageDir in sorted(folder.iterdir()):
            if not pageDir.is_dir():
                continue
            truth = labs.get(pageDir.name)
            crops = sorted(pageDir.glob("*.png"))
            if not truth or len(crops) != len(truth):
                continue   # no reliable line to label pairing on this page
            for cp, want in zip(crops, truth):
                a = np.asarray(Image.open(cp).convert("L"), dtype=np.uint8)
                t0 = time.time()
                for k,_f in TEXT_CKPTS:
                    got = rec.ReadText(a, readers[k])
                    acc[k][0] += rec.CharAcc(got, want)
                    acc[k][1] += rec.WordAcc(got, want)
                tRec.append((time.time() - t0) / max(1, len(want)))
                n += 1
                if w in inkMap:
                    i, _p = rec.ClassifyImage(a, inkModel, strokeNormalise=False)
                    inkOk += (i == inkMap[w])
                if w in shapeMap:
                    i, _p = rec.ClassifyImage(a, shapeModel, strokeNormalise=True)
                    shapeOk += (i == shapeMap[w])
        if not n:
            continue
        Row(w)["recognition"] = dict(
            crops=n,
            charAccScratch=round(100 * acc["scratch"][0] / n, 2),
            wordAccScratch=round(100 * acc["scratch"][1] / n, 2),
            charAccJoint=round(100 * acc["joint"][0] / n, 2),
            wordAccJoint=round(100 * acc["joint"][1] / n, 2),
            msPerChar=round(1000 * float(np.mean(tRec)), 2))
        Row(w)["writerInk"] = dict(crops=n, acc=round(100 * inkOk / n, 2))
        Row(w)["writerShape"] = dict(crops=n, acc=round(100 * shapeOk / n, 2))
        print("    %-9s %3d crops  scratch %6.2f%%  joint %6.2f%%  ink %6.2f%%  shape %6.2f%%"
              % (w, n, 100 * acc["scratch"][0] / n, 100 * acc["joint"][0] / n,
                 100 * inkOk / n, 100 * shapeOk / n))


# ---------------------------------------------------------------- stage 5
def Stage5Profiles():
    print("\n[5] style profiles, all 13 stored writers")
    for w in paths.AllWriters:
        fp = paths.Profiles / ("%s.json" % w)
        if not fp.exists():
            print("    %-9s no profile file" % w)
            continue
        p = json.load(open(fp, encoding="utf-8"))
        g = p.get("glyphs", {})
        ds = [float(v.get("priorD", 0.5)) for lst in g.values() for v in lst]
        Row(w)["profile"] = dict(
            characters=len(g), variants=sum(len(v) for v in g.values()),
            prototypes=len(p.get("idealGlyphs", {})),
            medoids=len(p.get("idealMedoid", {})),
            slantDeg=p.get("slantDeg"), xHeightPx=p.get("xHeightPx"),
            ascender=p.get("ascender"), descender=p.get("descender"),
            wordSpaceXh=p.get("wordSpaceXh"), connectedness=p.get("connectedness"),
            strokeWidthXh=p.get("strokeWidthXh"),
            medianPriorD=round(float(np.median(ds)), 4) if ds else None,
            autoLegibility=round(float(syn.AutoLegibility(p)), 3),
            storedLegibility=p.get("legibilityLambda"))
        print("    %-9s %2d chars %4d variants  slant %6.1f  xh %3s  conn %.3f  medPriorD %s  lam %.2f"
              % (w, len(g), sum(len(v) for v in g.values()),
                 p.get("slantDeg", 0), p.get("xHeightPx"), p.get("connectedness", 0),
                 ("%.3f" % np.median(ds)) if ds else "-",
                 syn.AutoLegibility(p)))


# -------------------------------------------------------------- stage 6,7
def Stages67():
    print("\n[6,7] synthesis judged by both networks, then the g-code round trip")
    reader = rec.PaperCRNNNumpy(
        checkpointPath=rec.WEIGHTS_DIR / "paper_cnn_bilstm_ctc_joint_best.npz")
    shapeModel = rec.AuthorClassifierCNNNumpy(
        checkpointPath=rec.WEIGHTS_DIR / "author_shape_10new_weights.npz")
    smap = shapeModel.authorMapping
    cfg = rep.GantryConfig()

    for w in paths.AllWriters:
        fp = paths.Profiles / ("%s.json" % w)
        if not fp.exists() or w not in smap:
            continue
        prof = json.load(open(fp, encoding="utf-8"))
        accs, confs, ious, eMeans, eMaxes, secs = [], [], [], [], [], []
        for text in SENTENCES:
            t0 = time.time()
            traj = syn.SynthesizeJointBestOf(
                w, text, prof, nTries=NTRIES, seed=11,
                reader=reader, authorModel=shapeModel, authorMapping=smap)
            secs.append(time.time() - t0)
            img = syn.RenderTrajectory(traj, pxPerMm=18.0, profile=prof, uniformInk=True)
            accs.append(rec.CharAcc(rec.ReadText(img, reader), text))
            _i, probs = rec.ClassifyImage(img, shapeModel)
            confs.append(float(probs[smap[w]]))
            # g-code round trip: write it, read it back with the other parser
            gp = OUT / ("_tmp_%s.gcode" % w)
            rep.WriteGcode(traj, cfg, str(gp), title=w)
            back = rep.ParseGcode(str(gp))
            placed, _scale = rep.ToMachine(traj, cfg)
            iou, em, ex = CompareGeometry(placed, back)
            ious.append(iou); eMeans.append(em); eMaxes.append(ex)
            gp.unlink(missing_ok=True)
        Row(w)["synthesis"] = dict(
            sentences=len(SENTENCES), nTries=NTRIES,
            charAcc=round(100 * float(np.mean(accs)), 2),
            writerConf=round(100 * float(np.mean(confs)), 2),
            identified=bool(np.mean(confs) > 0.5),
            secondsMean=round(float(np.mean(secs)), 1))
        Row(w)["gcode"] = dict(
            iou=round(float(np.mean(ious)), 4),
            errMeanMm=round(float(np.mean(eMeans)), 4),
            errMaxMm=round(float(np.max(eMaxes)), 4))
        print("    %-9s char %6.2f%%  writerConf %6.2f%%  %5.1f s   gcode IoU %.4f  mean %.3f mm  max %.3f mm"
              % (w, 100 * np.mean(accs), 100 * np.mean(confs), np.mean(secs),
                 np.mean(ious), np.mean(eMeans), np.max(eMaxes)))


def CompareGeometry(strokesA, strokesB, pxPerMm=8.0):
    """IoU of the two drawings plus the vertex distance, A is the truth"""
    A = [np.asarray(s, dtype=np.float64) for s in strokesA if len(s) >= 2]
    B = [np.asarray(s, dtype=np.float64) for s in strokesB if len(s) >= 2]
    if not A or not B:
        return 0.0, float("nan"), float("nan")
    allPts = np.concatenate(A + B)
    x0, y0 = allPts.min(0)
    x1, y1 = allPts.max(0)
    W = max(8, int((x1 - x0) * pxPerMm) + 4)
    H = max(8, int((y1 - y0) * pxPerMm) + 4)

    def Raster(ss):
        m = np.full((H, W), 255, np.uint8)
        for s in ss:
            pts = [(float((x - x0) * pxPerMm) + 2, float((y - y0) * pxPerMm) + 2)
                   for x, y in s]
            segment.DrawPolyline(m, pts, 0, 2)
        return m < 128

    ma, mb = Raster(A), Raster(B)
    inter = int((ma & mb).sum())
    union = int((ma | mb).sum())
    iou = inter / max(1, union)
    # nearest vertex distance, B against A, subsampled so the search is cheap
    pa = np.concatenate(A)
    pb = np.concatenate(B)
    if len(pa) > 4000:
        pa = pa[np.linspace(0, len(pa) - 1, 4000).astype(int)]
    if len(pb) > 3000:
        pb = pb[np.linspace(0, len(pb) - 1, 3000).astype(int)]
    d = np.hypot(pb[:, None, 0] - pa[None, :, 0], pb[:, None, 1] - pa[None, :, 1]).min(1)
    return iou, float(d.mean()), float(d.max())


def WriteDocument():
    md = []
    md.append("# Handwriting robot: per writer results, every pipeline stage\n")
    md.append("Produced by `testing/full_results.py` on the existing trained "
              "weights; nothing was retrained.\n")
    md.append("Writers 150 to 588 come from the public database, the five named "
              "writers were photographed for this project.\n")

    def table(title, note, cols, pick):
        md.append("\n## %s\n" % title)
        if note:
            md.append("%s\n" % note)
        md.append("| writer | " + " | ".join(c[0] for c in cols) + " |")
        md.append("|---" * (len(cols) + 1) + "|")
        for w in paths.AllWriters:
            d = R.get(w, {}).get(pick)
            if not d:
                continue
            md.append("| %s | " % w + " | ".join(c[1](d) for c in cols) + " |")

    f = lambda k, p="": (lambda d: ("-" if d.get(k) is None else
                                    ("%s%s" % (d[k], p))))
    table("1. Segmentation",
          "Ground truth is the transcript typed when the page was collected, so "
          "the line count is exact. Photographed writers only.",
          [("pages", f("pages")), ("lines found", f("linesDetected")),
           ("lines true", f("linesTrue")),
           ("exact", lambda d: "yes" if d["exact"] else "NO"),
           ("s/page mean", f("secondsMean")), ("s min", f("secondsMin")),
           ("s max", f("secondsMax")),
           ("pitch px", lambda d: "%d-%d" % (d["pitchMin"], d["pitchMax"])),
           ("skew deg", lambda d: "%+.1f..%+.1f" % (d["angleMin"], d["angleMax"]))],
          "segmentation")
    table("2. Text recognition on real crops",
          "Crops whose count matches the transcript, so each line pairs with "
          "its label. Measured with BOTH text checkpoints: scratch is the one "
          "the loader picks by default, joint is the one the report figures "
          "were taken on.",
          [("crops", f("crops")),
           ("char % scratch", f("charAccScratch")),
           ("word % scratch", f("wordAccScratch")),
           ("char % joint", f("charAccJoint")),
           ("word % joint", f("wordAccJoint")),
           ("ms/char", f("msPerChar"))], "recognition")
    table("3. Writer identification, ink classifier",
          "Raw grey crops, which is what the live server uses on a photograph.",
          [("crops", f("crops")), ("accuracy %", f("acc"))], "writerInk")
    table("4. Writer identification, shape classifier",
          "The same crops after stroke normalising, which is the judge used on "
          "drawn output.",
          [("crops", f("crops")), ("accuracy %", f("acc"))], "writerShape")
    table("5. Style profiles",
          "What the builder extracted per writer, and how far the library is "
          "trusted. lam is what AutoLegibility derives from the median prior "
          "distance.",
          [("chars", f("characters")), ("variants", f("variants")),
           ("prototypes", f("prototypes")), ("slant deg", f("slantDeg")),
           ("x-height px", f("xHeightPx")), ("ascender", f("ascender")),
           ("descender", f("descender")), ("word gap", f("wordSpaceXh")),
           ("connectedness", f("connectedness")),
           ("median priorD", f("medianPriorD")), ("lam", f("autoLegibility")),
           ("stored lam", f("storedLegibility"))], "profile")
    table("6. Synthesis, judged by both networks",
          "Best of %d, two sentences, one seed. Both judges also drive the "
          "search, so these figures are not independent of it." % NTRIES,
          [("char %", f("charAcc")), ("writer conf %", f("writerConf")),
           ("identified", lambda d: "yes" if d["identified"] else "no"),
           ("s/line", f("secondsMean"))], "synthesis")
    table("7. G-code round trip",
          "Geometry written out, then read back by a separate parser and "
          "compared. This measures rounding, stroke reordering and clamping "
          "only, not the machine.",
          [("IoU", f("iou")), ("mean err mm", f("errMeanMm")),
           ("max err mm", f("errMaxMm"))], "gcode")

    (OUT / "RESULTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (OUT / "full_results.json").write_text(json.dumps(R, indent=1), encoding="utf-8")
    print("\nwrote %s" % (OUT / "RESULTS.md"))
    print("wrote %s" % (OUT / "full_results.json"))


if __name__ == "__main__":
    Stage1Segmentation()
    Stages234()
    Stage5Profiles()
    Stages67()
    WriteDocument()
