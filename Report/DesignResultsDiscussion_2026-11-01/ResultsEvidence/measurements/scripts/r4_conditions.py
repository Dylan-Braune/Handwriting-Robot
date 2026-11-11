#under what conditions can R4 be met?
#
#R4 allows 2 + 0.025x seconds for a page of x characters.  on the ODROID N2+
#the deployed configuration takes 46 to 55 s against allowances of 14.5 and
#26.6 s.  the measured split there is 32.4 per cent image work and 67.6 per
#cent networks, and the allowance has only one term that grows with the page,
#so the two halves must be judged against different terms:
#
#   the NETWORKS are the only stages whose cost grows with the character
#   count, so the 0.025 s/char term is theirs.  on target they cost 0.0394 to
#   0.0602 s/char, which is 1.6 to 2.4 times that term.
#
#   the IMAGE stages are per-page work, so they must fit the fixed 2 s term.
#   on target they cost 16.1 to 16.8 s, which is 8.1 to 8.4 times it.
#
#this measures the two changes that could close those two gaps, so that the
#projection in the report rests on measured factors rather than on assertion:
#
#   change 1, the library route: the same weights in the torch modules of
#       trainlibrary.py, every crop of a page in one batched forward, instead
#       of the hand written numpy forward one crop at a time.
#   change 2, decimation: the image stages run on the page at half linear
#       resolution.  the filters and the connected component labelling scale
#       with the pixel count, so a factor of 2 on each axis is a factor of 4
#       in work.  the line count found is recorded at both resolutions,
#       because a speed-up that loses lines is not a speed-up.
#
#nothing here is a change to the deployed pipeline.  it measures what two
#candidate changes would buy, on the same pages, with the same weights.
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
warnings.filterwarnings("ignore")

CP3 = Path(r"C:\Users\braun\OneDrive\Documents\GitHub\Handwriting-Robot\Software\CleanPipeline3")
sys.path.insert(0, str(CP3))
import recognise as REC
import segment as SEG
import timing as TIMING
import trainlibrary as TL
import torch

DL = Path(r"C:\Users\braun\Downloads")
PAGES = CP3/"ReportResults"/"FullReproducedPages"
OUT = Path(__file__).resolve().parent/"r4_conditions"
OUT.mkdir(exist_ok=True)
PW = 0.17
SCALES = (1.0, 0.5)

torch.set_grad_enabled(False)
readerPath = DL/"paper_cnn_bilstm_ctc_scratch_best.npz"
namerPath = DL/"author_shape_11_pw170_weights.npz"
readerNp = REC.PaperCRNNNumpy(checkpointPath=str(readerPath))
namerNp = REC.AuthorClassifierCNNNumpy(checkpointPath=str(namerPath))
byIdx = {v: k for k, v in namerNp.authorMapping.items()}
NAMES = [byIdx[i] for i in range(len(byIdx))]
with np.load(str(readerPath), allow_pickle=False) as z:
    rp = {k: z[k] for k in z.files if not k.startswith("__")}
with np.load(str(namerPath), allow_pickle=True) as z:
    npp = {k: z[k] for k in z.files
           if not k.startswith("__") and z[k].dtype != object}
readerT = TL.ReaderTorch(numClasses=readerNp.sd["text_head.bias"].shape[0]).eval()
TL.SeedFromNumpy(readerT, rp)
namerT = TL.NamerTorch(numAuthors=len(NAMES)).eval()
TL.SeedFromNumpy(namerT, npp)
print("torch %s, %d threads" % (torch.__version__, torch.get_num_threads()))
print("reader %s\nnamer  %s\n" % (readerPath.name, namerPath.name))


def Batch(crops, norm):
    ts = []
    for c in crops:
        img = c
        if norm:
            n = REC.StrokeNormalize(c, penWidthXh=PW)
            if n is not None:
                img = n
        t = REC.TensorFromResized(REC.ResizeLineImageFixed(
            Image.fromarray(img).convert("L")))
        ts.append(np.asarray(t)[0])
    return np.ascontiguousarray(np.stack(ts, axis=0), dtype=np.float32)


rows = []
print("%-22s %-6s %-7s %-10s %-10s %-9s %-9s %s"
      % ("page", "scale", "lines", "image s", "net np s", "net pt s",
         "chars", "text vs full-res"))
print("-"*100)

for png in sorted(PAGES.glob("*.jpg")):
    full = np.asarray(Image.open(png).convert("L"), dtype=np.uint8)
    ref = None
    for sc in SCALES:
        if sc == 1.0:
            gray = full
        else:
            im = Image.fromarray(full)
            gray = np.asarray(im.resize((int(im.width*sc), int(im.height*sc)),
                                        Image.LANCZOS).convert("L"),
                              dtype=np.uint8)
        run = TIMING.Run("classify")
        with run.Step("loading", chars=0):
            SEG.UprightPage(gray)
        #at reduced resolution the segmenter can find no line centres at all
        #and then raises instead of returning nothing, so the failure is
        #caught and recorded: a decimation that crashes the segmenter is a
        #result about decimation, not a reason to abandon the measurement
        try:
            lines, _dbg, _info = SEG.SegmentLines(gray, timer=run)
        except Exception as exc:
            print("%-22s %-6.2f SEGMENTER FAILED: %s: %s"
                  % (png.stem[:22], sc, type(exc).__name__, exc), flush=True)
            rows.append({"page": png.stem, "scale": sc, "failed":
                         "%s: %s" % (type(exc).__name__, exc)})
            continue
        crops = [np.asarray(ln["image"]) for ln in lines]
        image = sum(s["seconds"] for s in run.stages)
        if not crops:
            print("%-22s %-6.2f no lines" % (png.stem[:22], sc))
            rows.append({"page": png.stem, "scale": sc, "failed": "no lines"})
            continue

        #---the networks, both routes, on the crops this resolution produced--
        t0 = time.perf_counter()
        textsNp = [REC.ReadText(Image.fromarray(c).convert("L"), readerNp)
                   for c in crops]
        for c in crops:
            REC.ClassifyImage(Image.fromarray(c).convert("L"), namerNp,
                              strokeNormalise=True)
        netNp = time.perf_counter()-t0

        t0 = time.perf_counter()
        lp = torch.log_softmax(readerT(torch.from_numpy(
            Batch(crops, norm=False))), dim=2).numpy()
        textsT = [REC.DecodeCtc(np.ascontiguousarray(lp[:, i:i+1, :]))[0]
                  for i in range(len(crops))]
        namerT(torch.from_numpy(Batch(crops, norm=True)))
        netPt = time.perf_counter()-t0

        chars = sum(len(t) for t in textsT)
        joined = " ".join(textsT)
        if ref is None:
            ref = joined
            acc = 100.0
        else:
            acc = 100*REC.CharAcc(joined, ref)
        rows.append({"page": png.stem, "scale": sc, "lines": len(crops),
                     "image": image, "netNp": netNp, "netPt": netPt,
                     "chars": chars, "accVsFull": acc,
                     "stages": {s["stage"]: s["seconds"] for s in run.stages}})
        print("%-22s %-6.2f %-7d %-10.2f %-10.2f %-9.2f %-9d %.1f %%"
              % (png.stem[:22], sc, len(crops), image, netNp, netPt, chars, acc),
              flush=True)

#---the two factors, measured-------------------------------------------------
print("\n" + "="*100)
ok = [r for r in rows if "failed" not in r]
failed = [r for r in rows if "failed" in r]
byScale = {sc: [r for r in ok if r["scale"] == sc] for sc in SCALES}
if failed:
    print("\n%d page/scale combination(s) the segmenter could not handle:"
          % len(failed))
    for r in failed:
        print("   %-22s scale %.2f  %s" % (r["page"], r["scale"], r["failed"]))
f = byScale[1.0]
h = byScale[0.5]
print("%d pages at full resolution, %d at half" % (len(f), len(h)))

netNp = sum(r["netNp"] for r in f)
netPt = sum(r["netPt"] for r in f)
print("\nchange 1, the library route, at full resolution:")
print("   networks, numpy one crop at a time  %8.1f s" % netNp)
print("   networks, torch batched per page    %8.1f s" % netPt)
print("   measured factor                     %8.2f x" % (netNp/netPt))

if h:
    paired = {r["page"]: r for r in h}
    ff = [r for r in f if r["page"] in paired]
    imF = sum(r["image"] for r in ff)
    imH = sum(paired[r["page"]]["image"] for r in ff)
    print("\nchange 2, half linear resolution, image stages only:")
    print("   image stages, full resolution       %8.1f s" % imF)
    print("   image stages, half resolution       %8.1f s" % imH)
    print("   measured factor                     %8.2f x" % (imF/imH))
    lf = sum(r["lines"] for r in ff)
    lh = sum(paired[r["page"]]["lines"] for r in ff)
    print("   lines found: %d at full, %d at half" % (lf, lh))
    same = sum(1 for r in ff if paired[r["page"]]["lines"] == r["lines"])
    print("   pages whose line count is unchanged: %d of %d" % (same, len(ff)))
    accs = [paired[r["page"]]["accVsFull"] for r in ff]
    print("   text at half resolution against the full-resolution reading: "
          "%.1f %% mean, %.1f %% worst" % (float(np.mean(accs)), min(accs)))

#---applying the measured factors to the ON-TARGET measurement---------------
#the two ODROID runs, from the server log: (characters, image s, network s)
TARGET = [(500, 16.14, 30.12), (982, 16.81, 38.67)]
netFac = netNp/netPt
imFac = (imF/imH) if h else 1.0
print("\n" + "="*100)
print("projection onto the ODROID N2+, applying the two measured factors to "
      "the measured on-target stage times")
print("%-8s %-9s %-11s %-11s %-11s %-9s %s"
      % ("chars", "allow s", "image s", "network s", "total s", "over s", "verdict"))
print("-"*78)
for chars, img, net in TARGET:
    allow = 2+0.025*chars
    i2, n2 = img/imFac, net/netFac
    t2 = i2+n2
    print("%-8d %-9.2f %-11.2f %-11.2f %-11.2f %-9.2f %s"
          % (chars, allow, i2, n2, t2, t2-allow,
             "MET" if t2 <= allow else "EXCEEDED"))
    print("         %-9s %-11s %-11s"
          % ("", "was %.2f" % img, "was %.2f" % net))

print("\nand the per-character network cost against the 0.025 s/char term:")
for chars, _img, net in TARGET:
    print("   %4d characters: %.4f s/char now, %.4f s/char with the library "
          "route (%.1f x the term)"
          % (chars, net/chars, net/netFac/chars, (net/netFac/chars)/0.025))

print("\nwhat the fixed 2 s term still demands of the image stages:")
for chars, img, _net in TARGET:
    print("   %4d characters: %.2f s now, %.2f s decimated, against a 2 s "
          "term: a further %.1f x is needed"
          % (chars, img, img/imFac, (img/imFac)/2))

(OUT/"conditions.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
print("\nwrote", OUT/"conditions.json")
