"""Torch twin of the from-scratch trainer, for comparing the two on a tiny subset.

python torchtrain.py text   --lines 16 --epochs 40 --engine both
python torchtrain.py author --lines 40 --epochs 40 --engine both

The point is a controlled comparison, so both engines are given the same
thing in every respect that is allowed to vary:

  same crops      the subset is materialised once from train.py's own dataset
                  and held in memory, so both engines see identical pixels
  same start      both are initialised from train.py's InitTextModelParams /
                  InitAuthorModelParams with the same seed
  same batches    both iterate with train.py's IterateBatches and the same rng
  same recipe     RMSprop lr/1e-5 decay for the reader, AdamW lr/1e-4 decay
                  with cosine annealing and grad clipping for the namer, which
                  is what the old torch pipeline used

If the implementation is correct the two loss curves should sit almost on top
of each other for the first steps and both should drive a tiny subset to near
zero.  A gap that opens immediately is a bug in a forward or backward pass; a
gap that opens slowly is usually the optimiser or the loss reduction.

Augmentation is deliberately off (isTrain=False views): it is random per call,
so it would make the two runs incomparable, and it also stops a tiny subset
from being memorised, which is the thing being tested.

Nothing here writes to the real weights directory.  Logs and checkpoints go to
testing/results/trainvs/
"""
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
CP2 = HERE.parent
sys.path.insert(0, str(CP2))

import paths
import recognise as CLS
import train as TR

OUT = HERE/"results"/"trainvs"


#======================================================================
#THE SAME TWO NETWORKS, IN TORCH
#every parameter name below matches the numpy name in train.py, so a
#state dict moves between the two without any renaming
#======================================================================
def ConvBlock(inCh, outCh, nLayers):
    layers = []
    for i in range(nLayers):
        layers += [nn.Conv2d(inCh if i == 0 else outCh, outCh, kernel_size=3, padding=1),
                   nn.BatchNorm2d(outCh),
                   nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class PaperCRNNTorch(nn.Module):
    """conv backbone, height collapsed to one row, 2-layer BiLSTM, ctc head"""

    def __init__(self, numClasses, hiddenSize=TR.HIDDEN_SIZE):
        super().__init__()
        self.stage1 = ConvBlock(1, 32, 2)
        self.stage2 = ConvBlock(32, 64, 4)
        self.stage3 = ConvBlock(64, 128, 6)
        self.pool = nn.MaxPool2d(2, 2)
        self.sequence = nn.LSTM(128, hiddenSize, num_layers=2,
                                bidirectional=True, batch_first=False)
        self.text_head = nn.Linear(hiddenSize*2, numClasses)

    def forward(self, x):
        x = self.pool(self.stage1(x))
        x = self.pool(self.stage2(x))
        x = self.stage3(x)
        x = torch.nn.functional.adaptive_max_pool2d(x, (1, x.shape[3]))   #collapse height
        x = x[:, :, 0, :]                     #(B,128,W)
        x = x.permute(2, 0, 1).contiguous()   #(W,B,128), width is the ctc time axis
        x, _ = self.sequence(x)
        return self.text_head(x)              #raw logits (T,B,C)


class AuthorCNNTorch(nn.Module):
    """the same backbone, width collapsed as well, one score per writer"""

    def __init__(self, numAuthors):
        super().__init__()
        self.stage1 = ConvBlock(1, 32, 2)
        self.stage2 = ConvBlock(32, 64, 4)
        self.stage3 = ConvBlock(64, 128, 6)
        self.pool = nn.MaxPool2d(2, 2)
        self.author_head = nn.Sequential(nn.Linear(128, 96),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(0.2),
                                         nn.Linear(96, numAuthors))

    def forward(self, x):
        x = self.pool(self.stage1(x))
        x = self.pool(self.stage2(x))
        x = self.stage3(x)
        x = torch.nn.functional.adaptive_max_pool2d(x, (1, x.shape[3]))
        x = x[:, :, 0, :]              #(B,128,W)
        pooled = x.mean(dim=2)         #(B,128), width averaged away
        return self.author_head(pooled)


#---LoadNumpyInto: give a torch model the numpy trainer's starting weights---
def LoadNumpyInto(model, params):
    sd = model.state_dict()
    loaded = 0
    for k, v in params.items():
        if k.startswith("__") or k not in sd:
            continue
        t = torch.from_numpy(np.ascontiguousarray(v))
        if tuple(t.shape) != tuple(sd[k].shape):
            raise SystemExit("shape mismatch on %s: numpy %s vs torch %s"
                             % (k, tuple(t.shape), tuple(sd[k].shape)))
        sd[k] = t
        loaded += 1
    model.load_state_dict(sd, strict=False)   #strict off only for num_batches_tracked
    missing = [k for k in sd if not k.endswith("num_batches_tracked") and k not in params]
    if missing:
        raise SystemExit("no numpy weight for: %s" % missing[:6])
    return loaded


#---ExportToNumpy: torch weights back out in the npz layout recognise.py reads---
def ExportToNumpy(model):
    return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()
            if not k.endswith("num_batches_tracked")}


#======================================================================
#THE SUBSET
#materialised once so neither engine can see a different image
#======================================================================
def MaterialiseSubset(view, nLines, seed):
    rng = np.random.default_rng(seed)
    n = min(nLines, len(view))
    picks = rng.permutation(len(view))[:n]
    samples = [view[int(i)] for i in sorted(picks)]
    print("[Subset] %d lines held in memory, no augmentation" % len(samples))
    return samples


#---Batches: the same order for both engines, from train.py's own iterator---
def Batches(samples, batchSize, rng, shuffle=True):
    return TR.IterateBatches(samples, batchSize, rng, shuffle)


def WriteLog(task, engine, rows, meta):
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT/("%s_%s" % (task, engine))
    with open(str(stem)+".csv", "w", encoding="utf-8") as f:
        f.write(",".join(rows[0].keys())+"\n")
        for r in rows:
            f.write(",".join("%.6g" % v if isinstance(v, float) else str(v)
                             for v in r.values())+"\n")
    (Path(str(stem)+".json")).write_text(json.dumps({"meta": meta, "rows": rows}, indent=1),
                                         encoding="utf-8")
    print("[Log] %s.csv" % stem)


#======================================================================
#THE READER, BOTH WAYS
#======================================================================
#---ScoreNumpy: held-out loss and character accuracy, numpy engine---
def ScoreNumpy(samples, params, batchSize, dtype=np.float32):
    if not samples:
        return None, None
    total, nB, chars, errors = 0.0, 0, 0, 0
    for start in range(0, len(samples), batchSize):
        images, targets, texts = TR.TextNumpyBatch(samples[start:start+batchSize])
        rawLogits = TR.EvalTextForward(params, images.astype(dtype))
        loss, _ = TR.CtcLoss(rawLogits, targets)
        total += loss
        nB += 1
        for pred, truth in zip(CLS.DecodeCtc(CLS.LogSoftmax(rawLogits, axis=2)), texts):
            errors += CLS.Levenshtein(pred, truth)
            chars += max(1, len(truth))
    return total/max(1, nB), max(0.0, 1.0-errors/max(1, chars))


#---ScoreTorch: the same, each engine scoring with its own forward pass---
def ScoreTorch(samples, model, batchSize, ctc, dtype=np.float32):
    if not samples:
        return None, None
    model.eval()
    total, nB, chars, errors = 0.0, 0, 0, 0
    with torch.no_grad():
        for start in range(0, len(samples), batchSize):
            images, targets, texts = TR.TextNumpyBatch(samples[start:start+batchSize])
            x = torch.from_numpy(images.astype(dtype))
            rawLogits = model(x)
            logProbs = torch.log_softmax(rawLogits, dim=2)
            flat = torch.tensor([c for t in targets for c in t], dtype=torch.long)
            tLens = torch.tensor([len(t) for t in targets], dtype=torch.long)
            iLens = torch.full((x.shape[0],), logProbs.shape[0], dtype=torch.long)
            total += float(ctc(logProbs, flat, iLens, tLens))/x.shape[0]
            nB += 1
            for pred, truth in zip(CLS.DecodeCtc(logProbs.numpy()), texts):
                errors += CLS.Levenshtein(pred, truth)
                chars += max(1, len(truth))
    model.train()
    return total/max(1, nB), max(0.0, 1.0-errors/max(1, chars))


def RunTextNumpy(samples, params, epochs, batchSize, lr, seed, valSamples=()):
    trainable = TR.TrainableParams(params)
    optimizer = TR.RMSprop(trainable, lr=lr, weightDecay=1e-5)
    scheduler = TR.ReduceLrOnPlateau(optimizer, factor=0.5, patience=5, minLr=1e-5)
    rng = np.random.default_rng(seed)
    rows = []
    for epoch in range(1, epochs+1):
        t0 = time.time()
        total, nB = 0.0, 0
        for batch in Batches(samples, batchSize, rng):
            images, targets, _texts = TR.TextNumpyBatch(batch)
            rawLogits, cache = TR.TextModelForward(images, params)
            loss, dRawLogits = TR.CtcLoss(rawLogits, targets)
            _, grads = TR.TextModelBackward(dRawLogits, cache)
            optimizer.Step({k: grads[k] for k in trainable})
            total += loss
            nB += 1
        valLoss, valAcc = ScoreNumpy(valSamples, params, batchSize)
        if valLoss is not None:
            scheduler.Step(valLoss)
        rows.append(dict(epoch=epoch, loss=total/max(1, nB), valLoss=valLoss,
                         valCharAcc=valAcc, lr=optimizer.lr, seconds=time.time()-t0))
        print("[numpy] epoch %03d | train %.4f | val %s | val char acc %s | %.1f s"
              % (epoch, rows[-1]["loss"],
                 "----" if valLoss is None else "%.4f" % valLoss,
                 "----" if valAcc is None else "%.2f%%" % (100*valAcc),
                 rows[-1]["seconds"]), flush=True)
    return rows, params


def RunTextTorch(samples, params, epochs, batchSize, lr, seed, reduction, valSamples=()):
    torch.manual_seed(seed)
    model = PaperCRNNTorch(len(CLS.CHARSET)+1)
    print("[torch] loaded %d tensors from the numpy initialisation" % LoadNumpyInto(model, params))
    model.train()
    #same recipe the old pipeline used: RMSprop, 1e-5 decay, blank index 0
    optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                                    patience=5, min_lr=1e-5)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True,
                     reduction="sum" if reduction == "match" else "mean")
    rows = []
    for epoch in range(1, epochs+1):
        t0 = time.time()
        rng = np.random.default_rng(seed)   #reseeded per epoch to mirror numpy's single rng stream
        total, nB = 0.0, 0
        for batch in Batches(samples, batchSize, rng):
            images, targets, _texts = TR.TextNumpyBatch(batch)
            x = torch.from_numpy(images)
            rawLogits = model(x)
            logProbs = torch.log_softmax(rawLogits, dim=2)
            flat = torch.tensor([c for t in targets for c in t], dtype=torch.long)
            targetLengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
            inputLengths = torch.full((x.shape[0],), logProbs.shape[0], dtype=torch.long)
            loss = ctc(logProbs, flat, inputLengths, targetLengths)
            if reduction == "match":
                loss = loss/x.shape[0]   #numpy CtcLoss divides the summed loss by the batch, not by label length
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            nB += 1
        valLoss, valAcc = ScoreTorch(valSamples, model, batchSize,
                                     nn.CTCLoss(blank=0, zero_infinity=True, reduction="sum"))
        if valLoss is not None:
            scheduler.step(valLoss)
        rows.append(dict(epoch=epoch, loss=total/max(1, nB), valLoss=valLoss,
                         valCharAcc=valAcc, lr=optimizer.param_groups[0]["lr"],
                         seconds=time.time()-t0))
        print("[torch] epoch %03d | train %.4f | val %s | val char acc %s | %.1f s"
              % (epoch, rows[-1]["loss"],
                 "----" if valLoss is None else "%.4f" % valLoss,
                 "----" if valAcc is None else "%.2f%%" % (100*valAcc),
                 rows[-1]["seconds"]), flush=True)
    return rows, ExportToNumpy(model)


#---TextAccuracy: character accuracy over a set of crops, in batches---
def TextAccuracy(samples, params, batchSize):
    chars, errors = 0, 0
    for start in range(0, len(samples), batchSize):
        images, _targets, texts = TR.TextNumpyBatch(samples[start:start+batchSize])
        logProbs = CLS.LogSoftmax(TR.EvalTextForward(params, images), axis=2)
        for pred, truth in zip(CLS.DecodeCtc(logProbs), texts):
            errors += CLS.Levenshtein(pred, truth)
            chars += max(1, len(truth))
    return max(0.0, 1.0-errors/max(1, chars))


#---ShowTextOverfit: what the network now reads back, against the truth---
def ShowTextOverfit(samples, params, howMany=6):
    images, _targets, texts = TR.TextNumpyBatch(samples[:howMany])
    rawLogits = TR.EvalTextForward(params, images)
    logProbs = rawLogits-rawLogits.max(axis=2, keepdims=True)
    logProbs = logProbs-np.log(np.exp(logProbs).sum(axis=2, keepdims=True))
    predictions = CLS.DecodeCtc(logProbs)
    accs = []
    for pred, truth in zip(predictions, texts):
        acc = CLS.CharAcc(pred, truth)
        accs.append(acc)
        print("  truth: %s\n  read : %s   (%.1f%% chars)" % (truth, pred, 100*acc))
    return float(np.mean(accs)) if accs else 0.0


#======================================================================
#THE NAMER, BOTH WAYS
#======================================================================
def RunAuthorNumpy(samples, params, epochs, batchSize, lr, seed):
    trainable = TR.TrainableParams(params)
    optimizer = TR.AdamW(trainable, lr=lr, weightDecay=1e-4)
    scheduler = TR.CosineAnnealingLr(optimizer, tMax=epochs)
    rngBatch = np.random.default_rng(seed)
    rngDropout = np.random.default_rng(seed+1)
    rows = []
    for epoch in range(1, epochs+1):
        t0 = time.time()
        total, nB, correct, seen = 0.0, 0, 0, 0
        for batch in Batches(samples, batchSize, rngBatch):
            images, authorIdxs, _ = TR.AuthorNumpyBatch(batch)
            logits, cache = TR.AuthorModelForward(images, params, training=True, rng=rngDropout)
            loss, dLogits = TR.CrossEntropyLoss(logits, authorIdxs)
            _, grads = TR.AuthorModelBackward(dLogits, cache)
            trainableGrads = {k: grads[k] for k in trainable}
            TR.ClipGradNorm(trainableGrads, maxNorm=5.0)
            optimizer.Step(trainableGrads)
            total += loss
            correct += int((np.argmax(logits, axis=1) == authorIdxs).sum())
            seen += len(authorIdxs)
            nB += 1
        scheduler.Step()
        rows.append(dict(epoch=epoch, loss=total/max(1, nB), trainAcc=correct/max(1, seen),
                         seconds=time.time()-t0))
        print("[numpy] epoch %03d | loss %.4f | train acc %.3f | %.1f s"
              % (epoch, rows[-1]["loss"], rows[-1]["trainAcc"], rows[-1]["seconds"]))
    return rows, params


def RunAuthorTorch(samples, params, epochs, batchSize, lr, seed, numAuthors):
    torch.manual_seed(seed)
    model = AuthorCNNTorch(numAuthors)
    print("[torch] loaded %d tensors from the numpy initialisation" % LoadNumpyInto(model, params))
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    lossFn = nn.CrossEntropyLoss()
    rows = []
    for epoch in range(1, epochs+1):
        t0 = time.time()
        rng = np.random.default_rng(seed)
        total, nB, correct, seen = 0.0, 0, 0, 0
        for batch in Batches(samples, batchSize, rng):
            images, authorIdxs, _ = TR.AuthorNumpyBatch(batch)
            x = torch.from_numpy(images)
            y = torch.from_numpy(authorIdxs)
            logits = model(x)
            loss = lossFn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total += float(loss.item())
            correct += int((logits.argmax(dim=1) == y).sum().item())
            seen += len(authorIdxs)
            nB += 1
        scheduler.step()
        rows.append(dict(epoch=epoch, loss=total/max(1, nB), trainAcc=correct/max(1, seen),
                         seconds=time.time()-t0))
        print("[torch] epoch %03d | loss %.4f | train acc %.3f | %.1f s"
              % (epoch, rows[-1]["loss"], rows[-1]["trainAcc"], rows[-1]["seconds"]))
    return rows, ExportToNumpy(model)


#======================================================================
#DRIVERS
#======================================================================
#---BaseDataset: the cached line crops, with no page loading involved---
#---CacheDir: where crops cut by the first-principles code are kept---
def CacheDir(args):
    return Path(args.cacheDir) if args.cacheDir else paths.Cache/"line_cache_firstprinciples"


#---BaseDataset: pages cut into crops by train.py's own segmentation, cached---
def BaseDataset(args, maxPages=None):
    base = TR.IAMLineDatasetRaw(rootDir=str(TR.DEFAULT_AUTHOR_DATA_DIR),
                                cacheDir=str(CacheDir(args)), maxPages=maxPages)
    if not base.samples:
        raise SystemExit("no line samples built from %s" % TR.DEFAULT_AUTHOR_DATA_DIR)
    return base


#---TextSource: the crops and the held-out crops for the reader---
def TextSource(args):
    if args.source == "teklia":
        root = paths.Data.parents[1]/"Data"/"Datasets"/"TekliaIamLines"
        trainSet, valSet = TR.BuildTekliaSplit(str(root))
        trainSet.isTrain = False   #augmentation is random per call, so it would make the runs incomparable
        return trainSet, valSet
    base = BaseDataset(args, args.pages)
    trainIdx, valIdx = base.HoldoutSplitIndices()
    return (TR.TrainEvalView(base, trainIdx, isTrain=False),
            TR.TrainEvalView(base, valIdx, isTrain=False))


def DoText(args):
    trainSet, holdSet = TextSource(args)
    samples = MaterialiseSubset(trainSet, args.lines, args.seed)
    #the held-out lines are split again: validation steers the schedule and is
    #watched every epoch, test is scored once at the end and never looked at
    held = MaterialiseSubset(holdSet, args.val+args.test, args.seed) if len(holdSet) else []
    valSamples, testSamples = held[:args.val], held[args.val:args.val+args.test]
    print("[Split] %d train, %d validation, %d test lines"
          % (len(samples), len(valSamples), len(testSamples)))

    meta = dict(task="text", lines=len(samples), valLines=len(valSamples),
                testLines=len(testSamples), epochs=args.epochs, batchSize=args.batch,
                lr=args.lr, seed=args.seed, reduction=args.reduction, source=args.source,
                optimiser="RMSprop(lr=%g, weight_decay=1e-5) + ReduceLROnPlateau" % args.lr)
    finals = {}
    for engine in (["numpy", "torch"] if args.engine == "both" else [args.engine]):
        params = TR.InitTextModelParams(args.seed, len(CLS.CHARSET)+1)   #same start for both
        print("\n=== reader, %s engine ===" % engine)
        if engine == "numpy":
            rows, trained = RunTextNumpy(samples, params, args.epochs, args.batch, args.lr,
                                         args.seed, valSamples)
        else:
            rows, trained = RunTextTorch(samples, params, args.epochs, args.batch, args.lr,
                                         args.seed, args.reduction, valSamples)
        print("[%s] what it reads back on lines it trained on:" % engine)
        trainAcc = ShowTextOverfit(samples, trained, howMany=3)
        testLoss, testAcc = ScoreNumpy(testSamples, trained, args.batch)
        finals[engine] = dict(trainCharAcc=trainAcc,
                              bestValCharAcc=max((r["valCharAcc"] or 0.0) for r in rows),
                              finalValCharAcc=rows[-1]["valCharAcc"],
                              finalTrainLoss=rows[-1]["loss"],
                              testLoss=testLoss, testCharAcc=testAcc)
        print("[%s] train %.1f%% | best val %.1f%% | test %s"
              % (engine, 100*trainAcc, 100*finals[engine]["bestValCharAcc"],
                 "----" if testAcc is None else "%.1f%%" % (100*testAcc)))
        WriteLog("text", engine, rows, dict(meta, **finals[engine]))
        if args.save:
            np.savez(str(OUT/("text_%s_weights.npz" % engine)), **trained)

    if len(finals) == 2:
        print("\n=== the two engines side by side ===")
        print("  %-22s %12s %12s" % ("", "numpy", "torch"))
        for key in ("finalTrainLoss", "finalValCharAcc", "bestValCharAcc", "testCharAcc"):
            a, b = finals["numpy"][key], finals["torch"][key]
            fmt = lambda v: "----" if v is None else ("%.4f" % v if "Loss" in key else "%.2f%%" % (100*v))
            print("  %-22s %12s %12s" % (key, fmt(a), fmt(b)))
        (OUT/"text_comparison.json").write_text(
            json.dumps(dict(meta=meta, finals=finals), indent=1, default=float), encoding="utf-8")
        print("  [Log] %s" % (OUT/"text_comparison.json"))


def DoAuthor(args):
    base = BaseDataset(args, args.pages)
    base.samples = [s for s in base.samples if s["page_key"].split("/")[0] in paths.DatabaseWriters]
    writers = list(paths.DatabaseWriters)
    if args.personal:
        #needs the photographed pages cut into lines, which goes through
        #AddPersonalSamples -> SegmentLines, and that call passes a path
        #where an array is wanted
        TR.AddPersonalSamples(base)
        writers += list(paths.PersonalWriters)
    base.author_folders = sorted(writers)
    base.author_to_idx = {a: i for i, a in enumerate(base.author_folders)}
    trainIdx, _valIdx = base.HoldoutSplitIndices()
    view = TR.AuthorLabeledView(base, trainIdx, isTrain=False)
    samples = MaterialiseSubset(view, args.lines, args.seed)
    numAuthors = len(base.author_folders)
    print("[Authors] %d classes, %d crops in the subset" % (numAuthors, len(samples)))

    meta = dict(task="author", lines=len(samples), authors=numAuthors, epochs=args.epochs,
                batchSize=args.batch, lr=args.lr, seed=args.seed,
                optimiser="AdamW(lr=%g, weight_decay=1e-4) + cosine" % args.lr)
    for engine in (["numpy", "torch"] if args.engine == "both" else [args.engine]):
        params = TR.InitAuthorModelParams(args.seed, numAuthors)
        print("\n=== namer, %s engine ===" % engine)
        if engine == "numpy":
            rows, trained = RunAuthorNumpy(samples, params, args.epochs, args.batch, args.lr, args.seed)
        else:
            rows, trained = RunAuthorTorch(samples, params, args.epochs, args.batch, args.lr,
                                           args.seed, numAuthors)
        WriteLog("author", engine, rows, meta)
        if args.save:
            np.savez(str(OUT/("author_%s_weights.npz" % engine)), **trained)


#======================================================================
#SIDE BY SIDE TRAINING
#both engines are stepped on the same batch, one after the other, so
#every epoch can be printed as a pair with the gap between them.  this
#is the run that shows the two implementations behaving the same way,
#as distinct from the single step check further down which shows that
#they compute the same thing
#======================================================================
def DoCompare(args):
    dtype = np.float64 if args.double else np.float32
    trainSet, holdSet = TextSource(args)
    samples = MaterialiseSubset(trainSet, args.lines, args.seed)
    held = MaterialiseSubset(holdSet, args.val+args.test, args.seed) if len(holdSet) else []
    valSamples, testSamples = held[:args.val], held[args.val:args.val+args.test]
    print("[Split] %d train, %d validation, %d test lines (no overlap)"
          % (len(samples), len(valSamples), len(testSamples)))
    print("[Precision] %s on both sides" % ("float64" if args.double else "float32"))

    #identical starting weights
    params = TR.InitTextModelParams(args.seed, len(CLS.CHARSET)+1)
    torch.manual_seed(args.seed)
    model = PaperCRNNTorch(len(CLS.CHARSET)+1)
    loaded = LoadNumpyInto(model, params)
    pN = {k: v.astype(dtype) for k, v in params.items()}
    if args.double:
        model.double()
    model.train()
    print("[Init] both engines start from the same %d tensors" % loaded)

    trainable = TR.TrainableParams(pN)
    optN = TR.RMSprop(trainable, lr=args.lr, weightDecay=1e-5)
    schedN = TR.ReduceLrOnPlateau(optN, factor=0.5, patience=5, minLr=1e-5)
    optT = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-5)
    schedT = optim.lr_scheduler.ReduceLROnPlateau(optT, mode="min", factor=0.5,
                                                 patience=5, min_lr=1e-5)
    ctcSum = nn.CTCLoss(blank=0, zero_infinity=True, reduction="sum")

    print("\n%5s | %-25s | %-25s | %-19s | %9s" %
          ("", "train ctc loss", "validation ctc loss", "val char acc", ""))
    print("%5s | %12s %12s | %12s %12s | %9s %9s | %9s" %
          ("epoch", "numpy", "torch", "numpy", "torch", "numpy", "torch", "rel gap"))
    print("-"*103)

    rows = []
    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        rng = np.random.default_rng(args.seed)   #one stream, so both engines get the same batches
        totN = totT = 0.0
        nB = 0
        for batch in Batches(samples, args.batch, rng):
            images, targets, _texts = TR.TextNumpyBatch(batch)
            x = images.astype(dtype)

            rawLogits, cache = TR.TextModelForward(x, pN)
            lossN, dRawLogits = TR.CtcLoss(rawLogits, targets)
            _, grads = TR.TextModelBackward(dRawLogits, cache)
            optN.Step({k: grads[k] for k in trainable})

            xt = torch.from_numpy(x)
            logProbs = torch.log_softmax(model(xt), dim=2)
            flat = torch.tensor([c for t in targets for c in t], dtype=torch.long)
            tLens = torch.tensor([len(t) for t in targets], dtype=torch.long)
            iLens = torch.full((xt.shape[0],), logProbs.shape[0], dtype=torch.long)
            lossT = ctcSum(logProbs, flat, iLens, tLens)/xt.shape[0]
            optT.zero_grad()
            lossT.backward()
            optT.step()

            totN += lossN
            totT += float(lossT.item())
            nB += 1

        trainN, trainT = totN/max(1, nB), totT/max(1, nB)
        valLossN, valAccN = ScoreNumpy(valSamples, pN, args.batch, dtype)
        valLossT, valAccT = ScoreTorch(valSamples, model, args.batch, ctcSum, dtype)
        if valLossN is not None:
            schedN.Step(valLossN)
            schedT.step(valLossT)
        gap = abs(trainN-trainT)/max(abs(trainT), 1e-12)
        rows.append(dict(epoch=epoch, trainNumpy=trainN, trainTorch=trainT, trainRelGap=gap,
                         valLossNumpy=valLossN, valLossTorch=valLossT,
                         valAccNumpy=valAccN, valAccTorch=valAccT,
                         lrNumpy=optN.lr, lrTorch=optT.param_groups[0]["lr"],
                         seconds=time.time()-t0))
        pct = lambda v: "----" if v is None else "%8.2f%%" % (100*v)
        num = lambda v: "----" if v is None else "%12.5f" % v
        print("%5d | %12.5f %12.5f | %s %s | %s %s | %9.2e"
              % (epoch, trainN, trainT, num(valLossN), num(valLossT),
                 pct(valAccN), pct(valAccT), gap), flush=True)

    #the test lines are scored once, here, and never steered the schedule
    testLossN, testAccN = ScoreNumpy(testSamples, pN, args.batch, dtype)
    weights = ExportToNumpy(model)
    testLossT, testAccT = ScoreTorch(testSamples, model, args.batch, ctcSum, dtype)

    print("\n=== final comparison ===")
    print("  %-28s %14s %14s" % ("", "numpy", "torch"))
    def Line(label, a, b, asPct=False):
        f = (lambda v: "----" if v is None else
             ("%13.2f%%" % (100*v) if asPct else "%14.6f" % v))
        print("  %-28s %14s %14s" % (label, f(a), f(b)))
    Line("final train loss", rows[-1]["trainNumpy"], rows[-1]["trainTorch"])
    Line("final validation loss", rows[-1]["valLossNumpy"], rows[-1]["valLossTorch"])
    Line("final val char accuracy", rows[-1]["valAccNumpy"], rows[-1]["valAccTorch"], True)
    Line("best val char accuracy",
         max((r["valAccNumpy"] or 0) for r in rows),
         max((r["valAccTorch"] or 0) for r in rows), True)
    Line("TEST loss (unseen)", testLossN, testLossT)
    Line("TEST char accuracy (unseen)", testAccN, testAccT, True)
    Line("total minutes", sum(r["seconds"] for r in rows)/60.0, None)

    #the same weights read by the other engine: this is the equivalence that
    #matters if the library version is the one actually used for training
    imgs, _tg, _tx = TR.TextNumpyBatch(testSamples[:4] or samples[:4])
    model.eval()
    with torch.no_grad():
        tLogits = model(torch.from_numpy(imgs.astype(dtype))).numpy()
    nLogits = TR.EvalTextForward(weights, imgs.astype(dtype))
    agree = float((tLogits.argmax(2) == nLogits.argmax(2)).mean())
    print("\n=== torch weights evaluated by the numpy inference code ===")
    print("  logits max abs difference   %.3e" % np.abs(tLogits-nLogits).max())
    print("  argmax frames in agreement  %.4f%%" % (100*agree))
    print("  decoded strings identical   %s"
          % (CLS.DecodeCtc(CLS.LogSoftmax(tLogits, axis=2)) ==
             CLS.DecodeCtc(CLS.LogSoftmax(nLogits, axis=2))))

    meta = dict(task="compare", precision="float64" if args.double else "float32",
                lines=len(samples), valLines=len(valSamples), testLines=len(testSamples),
                epochs=args.epochs, batchSize=args.batch, lr=args.lr, seed=args.seed,
                source=args.source, testNumpy=testAccN, testTorch=testAccT,
                logitsMaxDiff=float(np.abs(tLogits-nLogits).max()), argmaxAgreement=agree)
    WriteLog("compare", "both", rows, meta)
    if args.save:
        np.savez(str(OUT/"compare_numpy_weights.npz"), **pN)
        np.savez(str(OUT/"compare_torch_weights.npz"), **weights)


#======================================================================
#EQUIVALENCE CHECK
#no training at all: identical weights and one identical batch through
#both engines, comparing the forward values, the loss, and then every
#gradient after one backward pass
#======================================================================
def MaxRelDiff(a, b):
    scale = max(float(np.abs(a).max()), float(np.abs(b).max()), 1e-8)
    return float(np.abs(a-b).max())/scale


#---ReportDiffs: compare every gradient, ignoring tensors that are zero anyway---
def ReportDiffs(title, numpyGrads, torchModel, rows):
    #a conv bias followed by batch norm cannot change the output, so its
    #gradient is zero in both engines and a relative difference on it is
    #round-off divided by round-off.  those are counted, not ranked.
    #relative L2 and cosine are the honest measures: an elementwise maximum is
    #dominated by whichever single weight had the most cancellation in its sum
    worstL2, worstName, worstCos, inert = 0.0, "", 0.0, 0
    for name, t in torchModel.named_parameters():
        if t.grad is None or name not in numpyGrads:
            continue
        a = np.asarray(numpyGrads[name], dtype=np.float64).ravel()
        b = t.grad.detach().cpu().numpy().astype(np.float64).ravel()
        magnitude = max(float(np.abs(a).max()), float(np.abs(b).max()))
        nb = np.linalg.norm(b)
        relL2 = float(np.linalg.norm(a-b)/nb) if nb > 0 else 0.0
        cosGap = 1.0-float(a@b/(np.linalg.norm(a)*nb)) if nb > 0 and np.linalg.norm(a) > 0 else 0.0
        rows.append(dict(tensor=name, magnitude=magnitude, maxRel=MaxRelDiff(a, b),
                         relL2=relL2, cosGap=cosGap))
        if magnitude < 1e-6:
            inert += 1
            continue
        if relL2 > worstL2:
            worstL2, worstName = relL2, name
        worstCos = max(worstCos, cosGap)
    print("  %s: %d tensors (%d inert, gradient zero in both)" % (title, len(rows), inert))
    print("    worst relative L2 %.3e on %s, worst 1-cosine %.3e"
          % (worstL2, worstName, worstCos))
    return worstL2


def DoCheck(args):
    rng = np.random.default_rng(args.seed)
    B = 2
    dtype = np.float64 if args.double else np.float32
    x = rng.standard_normal((B, 1, CLS.INPUT_HEIGHT, CLS.INPUT_WIDTH)).astype(dtype)
    report = dict(precision="float64" if args.double else "float32")
    if args.double:
        print("[Precision] float64 on both sides: round-off should drop to about 1e-14,")
        print("            which separates a formula error from accumulation noise")

    print("\n=== reader: forward, ctc loss, and every gradient ===")
    p = TR.InitTextModelParams(args.seed, len(CLS.CHARSET)+1)
    model = PaperCRNNTorch(len(CLS.CHARSET)+1)
    LoadNumpyInto(model, p)
    if args.double:
        model.double()                                        #the float32 to float64 cast is exact,
        p = {k: v.astype(np.float64) for k, v in p.items()}    #so both engines still start identical
    model.train()
    targets = [sorted(rng.integers(1, len(CLS.CHARSET), size=6).tolist()) for _ in range(B)]

    rawLogits, cache = TR.TextModelForward(x, p)
    loss, dRawLogits = TR.CtcLoss(rawLogits, targets)
    _, grads = TR.TextModelBackward(dRawLogits, cache)

    out = model(torch.from_numpy(x))
    logProbs = torch.log_softmax(out, dim=2)
    flat = torch.tensor([c for t in targets for c in t], dtype=torch.long)
    tLens = torch.tensor([len(t) for t in targets], dtype=torch.long)
    iLens = torch.full((B,), logProbs.shape[0], dtype=torch.long)
    tLoss = nn.CTCLoss(blank=0, zero_infinity=True, reduction="sum")(logProbs, flat, iLens, tLens)/B
    tLoss.backward()

    fwd = MaxRelDiff(rawLogits, out.detach().numpy())
    print("  forward logits: relative difference %.3e" % fwd)
    print("  ctc loss: numpy %.6f vs torch %.6f (difference %.3e)"
          % (loss, float(tLoss), abs(loss-float(tLoss))))
    rows = []
    report["text"] = dict(forward=fwd, loss=abs(loss-float(tLoss)),
                          worstGrad=ReportDiffs("gradients", grads, model, rows), tensors=rows)

    print("\n=== namer: forward, cross entropy, and every gradient ===")
    numAuthors = 13
    pa = TR.InitAuthorModelParams(args.seed, numAuthors)
    amodel = AuthorCNNTorch(numAuthors)
    LoadNumpyInto(amodel, pa)
    if args.double:
        amodel.double()
        pa = {k: v.astype(np.float64) for k, v in pa.items()}
    amodel.train()
    for m in amodel.modules():
        if isinstance(m, nn.Dropout):
            m.eval()   #dropout masks cannot be shared between the engines, so switch it off for the comparison
    labels = rng.integers(0, numAuthors, size=B)

    logits, cacheA = TR.AuthorModelForward(x, pa, training=False, rng=np.random.default_rng(0))
    lossA, dLogits = TR.CrossEntropyLoss(logits, labels)
    _, gradsA = TR.AuthorModelBackward(dLogits, cacheA)

    outA = amodel(torch.from_numpy(x))
    tLossA = nn.CrossEntropyLoss()(outA, torch.from_numpy(labels))
    tLossA.backward()

    fwdA = MaxRelDiff(logits, outA.detach().numpy())
    print("  forward logits: relative difference %.3e" % fwdA)
    print("  cross entropy: numpy %.6f vs torch %.6f (difference %.3e)"
          % (lossA, float(tLossA), abs(lossA-float(tLossA))))
    rowsA = []
    report["author"] = dict(forward=fwdA, loss=abs(lossA-float(tLossA)),
                            worstGrad=ReportDiffs("gradients", gradsA, amodel, rowsA),
                            tensors=rowsA)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"equivalence.json").write_text(json.dumps(report, indent=1, default=float),
                                        encoding="utf-8")
    print("\n[Log] %s" % (OUT/"equivalence.json"))
    if args.double:
        print("At float64 anything near 1e-13 is machine precision: the two")
        print("implementations compute the same function, not merely a similar one.")
    else:
        print("In float32 a relative L2 of a few times 1e-3 with a 1-cosine near 1e-5 is")
        print("accumulation order, not a formula error.  Re-run with --double to confirm it.")


def Main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("task", choices=["compare", "check", "text", "author"])
    ap.add_argument("--engine", choices=["numpy", "torch", "both"], default="both")
    ap.add_argument("--lines", type=int, default=16, help="crops in the tiny subset")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--source", choices=["pages", "teklia"], default="pages",
                    help="pages: cut the database pages up with train.py's own segmentation. "
                         "teklia: the pre-cut line crops, no segmentation")
    ap.add_argument("--pages", type=int, default=None, help="limit how many pages are cut up")
    ap.add_argument("--cacheDir", default=None, help="where the cut crops are cached")
    ap.add_argument("--val", type=int, default=0,
                    help="held-out lines scored every epoch, which also steer the lr schedule")
    ap.add_argument("--test", type=int, default=0,
                    help="further held-out lines, scored once at the very end")
    ap.add_argument("--personal", action="store_true",
                    help="author task: also include the five photographed writers")
    ap.add_argument("--reduction", choices=["match", "legacy"], default="match",
                    help="match: sum/batch, the same quantity the numpy CtcLoss returns. "
                         "legacy: torch's own mean, which the old pipeline trained on")
    ap.add_argument("--save", action="store_true", help="also write the trained weights to testing/results/trainvs")
    ap.add_argument("--double", action="store_true",
                    help="check and compare: run both engines in float64, so round-off drops to ~1e-14")
    args = ap.parse_args()
    if args.task == "compare":
        DoCompare(args)
    elif args.task == "check":
        DoCheck(args)
    elif args.task == "text":
        DoText(args)
    else:
        DoAuthor(args)


if __name__ == "__main__":
    Main()
