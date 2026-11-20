#u23539195 Dylan Braune.  does a thinner stroke normalisation name writers better?
#
#   python strokenorm_train.py --epochs 50 --data /content/strokenorm
#
#the shape namer redraws every crop at one constant pen width before it looks
#at it, so that it cannot cheat by recognising how hard a pen pressed.  how
#WIDE that constant pen should be was never measured; 0.14 x-heights is simply
#what was first tried.  a thinner pen keeps the small bends and junctions a
#fat pen fills in, so it ought to help.  this trains the same namer once per
#width, changing nothing else, and reports the held-out accuracy of each.
#
#it reads the .npz files strokenorm_prepare.py writes, so the skeletonising is
#already done and an epoch is only augment, batch and backprop.  it uses the
#gpu when there is one.  the model and the augmentation come from the repo, so
#the recipe is the project's own and nothing is reimplemented here.
#
#the two speedups are separate and both are needed.  of a 426 s cpu epoch the
#network is 245.5 s and StrokeNormalize 158.7 s: the gpu only attacks the
#first, and skeletonising is numpy on the cpu that no card can touch.  caching
#it in prepare is what removes the 186.7 s floor that would otherwise cap a
#gpu run at 2.3x.  together they are roughly 17x.
#
#--batch 8 is what the project's recipe uses and what the cpu wanted; a gpu is
#badly underfed by it and --batch 32 is much faster.  that is a change to the
#recipe, but it applies equally to all four widths, so the comparison between
#them still holds -- say so in the write-up if it is used.
#
#---WHY THESE FOUR WIDTHS---------------------------------------------------
#the pen width is quantised.  recognise.StrokeNormalize does
#       w = clip(round(penWidthXh*xh),2,9) ;  k = 2*(w//2)+1
#so only kernels 3,5,7,9 are reachable -- w is floored at 2, which makes k=3
#the THINNEST pen that exists, not k=1.  measured over 400 real crops the
#x-height runs 13 to 226 px, median 39, and every setting at or below 0.06
#lands on k=3 for 98 per cent of crops, so four settings below 0.06 would
#train four identical classifiers.  these four are spaced off that
#measurement so each is genuinely different:
#
#   0.06  mean k 3.06   k3 98%                     thinnest pen that exists
#   0.09  mean k 4.14   k3 46%  k5 53%             thinner
#   0.11  mean k 4.88   k3 18%  k5 72%  k7  9%     thinner
#   0.17  mean k ~7.1   between the measured 6.75 at 0.16 and 7.53 at 0.18
#                                                  slightly thicker
#
#the current 0.14 (mean k 5.99) is not trained, by request.  0.11 and 0.17
#bracket it, so the trend still answers the question: if accuracy falls as the
#pen thickens, thinner wins.  prepare and pass 0.14 as well for the direct
#comparison.
#
#writers 151 and 551 are dropped, so this trains on 11 writers and its
#accuracy is NOT comparable with the shipped 13-class namer: fewer classes is
#an easier problem.  compare these four only with each other.
#
#---STOPPING A LEVEL EARLY--------------------------------------------------
#the best weights are written the moment a level beats its own previous best,
#so whatever has been reached is already on disk and stopping loses nothing.
#
#   ctrl-C once        end this width now, save, move to the next
#   ctrl-C twice       (within 3 seconds) abandon the whole sweep
#   touch skip.flag    same as one ctrl-C, for a notebook or background run
#   touch stop.flag    same as two
#
#the flags live in the output directory and are removed once acted on.  a
#width that was cut short is marked in the results and in the final table,
#because a width given 12 epochs cannot be ranked against one given 50.
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import train as TR               #AugmentLineImage, TensorFromResized
import trainlibrary as TL        #NamerTorch, SaveNpz, SeedFromNumpy

LEVELS = [0.06, 0.09, 0.11, 0.17]
CURRENT = 0.14


class AbortSweep(Exception):
    pass


#---NpzView: the prepared crops, augmented per epoch exactly as the repo does---
class NpzView:
    def __init__(self, images, labels, keys, isTrain):
        self.images, self.labels, self.keys = images, labels, keys
        self.isTrain = isTrain

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = self.images[i]
        if self.isTrain:
            img = TR.AugmentLineImage(img)
        return TR.TensorFromResized(img), int(self.labels[i]), str(self.keys[i])


def CheckFlags(outDir):
    stop, skip = outDir/"stop.flag", outDir/"skip.flag"
    if stop.exists():
        stop.unlink()
        raise AbortSweep("stop.flag")
    if skip.exists():
        skip.unlink()
        return True
    return False


def TrainOne(pw, dataDir, outDir, args):
    path = dataDir/("strokenorm_data_pw%03d.npz" % round(pw*1000))
    if not path.exists():
        print("  missing %s -- run strokenorm_prepare.py first" % path.name)
        return None
    z = np.load(path, allow_pickle=False)
    images, labels, isVal = z["images"], z["labels"], z["isVal"].astype(bool)
    keys, names = z["keys"], [str(a) for a in z["authorNames"]]
    trainSet = NpzView(images[~isVal], labels[~isVal], keys[~isVal], True)
    valSet = NpzView(images[isVal], labels[isVal], keys[isVal], False)
    print("  %d train, %d held out, %d writers"
          % (len(trainSet), len(valSet), len(names)), flush=True)

    #seeded from the same numpy initialiser every run, so the only difference
    #between the widths is the pen width itself
    model = TL.NamerTorch(len(names))
    TL.SeedFromNumpy(model, TR.InitAuthorModelParams(args.seed, len(names)))
    model.to(TL.DEVICE)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    lossFn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(args.seed)

    outPath = outDir/("author_shape_11_pw%03d_weights.npz" % round(pw*1000))
    best, bestEpoch, history = 0.0, 0, []
    cutShort, lastInt = None, [0.0]

    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        total, nB = 0.0, 0
        try:
            for batch in TR.IterateBatches(trainSet, args.batch, rng, True):
                imgs, labs, _k = TR.AuthorCollateFn(batch)
                logits = model(torch.from_numpy(imgs).to(TL.DEVICE))
                loss = lossFn(logits, torch.from_numpy(labs).to(TL.DEVICE))
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                total += float(loss.item())
                nB += 1
            scheduler.step()
            acc = TL.ScoreNamer(model, valSet, args.batch)
        except KeyboardInterrupt:
            now = time.time()
            if now-lastInt[0] < 3.0:
                raise AbortSweep("second ctrl-C")
            lastInt[0] = now
            print("\n  [pw %.2f] interrupted in epoch %d. best %.2f%% (epoch %d) "
                  "is already saved to %s\n  moving on -- ctrl-C again within 3s "
                  "to abandon the sweep"
                  % (pw, epoch, 100*best, bestEpoch, outPath.name), flush=True)
            cutShort = epoch
            break

        saved = ""
        if acc > best:
            best, bestEpoch = acc, epoch
            TL.SaveNpz(outPath, model, {"__author_names__": np.array(names),
                                        "__epoch__": np.array(epoch),
                                        "__best__": np.array(best),
                                        "__pen_width_xh__": np.array(pw)})
            saved = " [best]"
        history.append({"epoch": epoch, "loss": total/max(1, nB), "val": acc,
                        "min": (time.time()-t0)/60})
        print("  [pw %.2f] epoch %03d | loss %.4f | val %.2f%% | best %.2f%% | "
              "%.1f min%s" % (pw, epoch, total/max(1, nB), 100*acc, 100*best,
                              (time.time()-t0)/60, saved), flush=True)

        if CheckFlags(outDir):
            print("  [pw %.2f] skip.flag seen, stopping at epoch %d, best %.2f%% "
                  "already saved" % (pw, epoch, 100*best), flush=True)
            cutShort = epoch
            break

    return {"penWidth": pw, "best": best, "bestEpoch": bestEpoch,
            "weights": str(outPath), "history": history,
            "epochsRun": cutShort if cutShort else args.epochs,
            "cutShort": cutShort is not None}


def Main():
    ap = argparse.ArgumentParser(description="stroke-normalisation width sweep")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--levels", type=float, nargs="+", default=LEVELS)
    ap.add_argument("--data", default=None,
                    help="where the strokenorm_data_pw*.npz live "
                         "(default: beside the weights, as prepare wrote them)")
    ap.add_argument("--out", default=None, help="where to write the weights")
    args = ap.parse_args()

    import paths
    dataDir = Path(args.data) if args.data else paths.Cache/"StrokeNormSweep"
    outDir = Path(args.out) if args.out else dataDir
    outDir.mkdir(parents=True, exist_ok=True)
    print("[data]   %s" % dataDir)
    print("[out]    %s" % outDir)
    print("[device] %s" % TL.DEVICE)
    print("[stop]   ctrl-C once skips a width, twice abandons the sweep; or")
    print("         drop skip.flag / stop.flag into the output directory\n",
          flush=True)

    results, aborted = [], None
    for pw in args.levels:
        note = ("the current setting" if abs(pw-CURRENT) < 1e-9 else
                "thinner than current" if pw < CURRENT else "thicker than current")
        print("\n" + "="*70)
        print("pen width %.2f x-heights   (%s)" % (pw, note))
        print("="*70, flush=True)
        try:
            r = TrainOne(pw, dataDir, outDir, args)
            if r:
                results.append(r)
        except AbortSweep as e:
            print("\n[abort]  %s -- every best so far is on disk" % e, flush=True)
            aborted = str(e)
            break
        finally:
            (outDir/"sweep_results.json").write_text(
                json.dumps(results, indent=1), encoding="utf-8")

    if not results:
        print("\nnothing completed.")
        return

    print("\n\n" + "="*70)
    print("STROKE NORMALISATION SWEEP, 11 writers")
    print("="*70)
    print("%-12s %-10s %-10s %-9s %s"
          % ("penWidthXh", "best val", "at epoch", "epochs", "vs thinnest"))
    thin = min(results, key=lambda r: r["penWidth"])
    for r in sorted(results, key=lambda r: r["penWidth"]):
        d = "" if r is thin else "  %+.1f pp" % (100*(r["best"]-thin["best"]))
        mark = ("%d*" % r["epochsRun"]) if r["cutShort"] else ("%d" % r["epochsRun"])
        print("%-12.2f %-10.4f %-10d %-9s%s"
              % (r["penWidth"], r["best"], r["bestEpoch"], mark, d))

    runs = {r["epochsRun"] for r in results}
    if any(r["cutShort"] for r in results) or len(runs) > 1:
        print("\n*  cut short.  these widths did NOT get equal training, so the")
        print("   ranking above is not a fair comparison -- one that ran %d"
              % min(runs))
        print("   epochs cannot be judged against one that ran %d.  treat this as"
              % max(runs))
        print("   a progress report and level the epoch counts before concluding.")
    else:
        ranked = sorted(results, key=lambda r: -r["best"])
        print("\nbest: %.2f x-heights at %.2f%%, all widths on %d epochs"
              % (ranked[0]["penWidth"], 100*ranked[0]["best"], args.epochs))
        thinner = [r for r in results if r["penWidth"] < CURRENT]
        thicker = [r for r in results if r["penWidth"] > CURRENT]
        if thinner and thicker:
            mt = float(np.mean([r["best"] for r in thinner]))
            mk = float(np.mean([r["best"] for r in thicker]))
            print("mean of the %d thinner %.2f%%, the %d thicker %.2f%% (%+.1f pp)"
                  " -- thinner %s"
                  % (len(thinner), 100*mt, len(thicker), 100*mk, 100*(mt-mk),
                     "looks better" if mt > mk else "does NOT look better"))

    if aborted:
        print("\nabandoned (%s); %d of %d widths done."
              % (aborted, len(results), len(args.levels)))
    print("\nweights and sweep_results.json in %s" % outDir)
    print("\nNOTE: 11 classes, NOT comparable with the shipped 13-class namer.")


if __name__ == "__main__":
    Main()
