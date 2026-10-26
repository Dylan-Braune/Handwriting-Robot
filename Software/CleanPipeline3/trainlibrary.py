"""The same two networks trained with a library, for when time is short.

    python trainlibrary.py text   --epochs 40 --batch 8
    python trainlibrary.py author --epochs 40 --batch 8

train.py trains these networks from first principles, with every forward and
backward pass written out in numpy.  This file builds the identical
architecture in torch and trains it with the same recipe, which runs several
times faster because the library's convolutions are compiled and threaded.

The two are interchangeable.  Every parameter name here matches the name
train.py gives it, so a checkpoint written by either one loads into the other
and into recognise.py without renaming anything.  The weights are saved as
.npz in exactly the layout recognise.py reads, so a model trained here runs on
the robot through the from-first-principles inference code.

Both paths share train.py's data pipeline, so the crops, the transcripts and
the held-out split are the same either way.

Recipe, unchanged from the project's own:
    reader   RMSprop, lr 1e-3, weight decay 1e-5, ReduceLROnPlateau 0.5/5
    namer    AdamW, lr 1e-3, weight decay 1e-4, cosine annealing, clip 5.0
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import paths
import recognise as CLS
import train as TR


#======================================================================
#THE NETWORKS
#the module names below are chosen so that model.state_dict() comes out
#with the same keys train.py uses: stage1.0.weight, sequence.weight_ih_l0,
#text_head.bias and so on
#======================================================================
def ConvBlock(inCh, outCh, nLayers):
    layers = []
    for i in range(nLayers):
        layers += [nn.Conv2d(inCh if i == 0 else outCh, outCh, kernel_size=3, padding=1),
                   nn.BatchNorm2d(outCh),
                   nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class ReaderTorch(nn.Module):
    """twelve convolutions, height collapsed to one row, two bidirectional
    lstm layers, one score per character per column"""

    def __init__(self, numClasses, hiddenSize=TR.HIDDEN_SIZE):
        super().__init__()
        self.stage1 = ConvBlock(1, 32, 2)
        self.stage2 = ConvBlock(32, 64, 4)
        self.stage3 = ConvBlock(64, 128, 6)
        self.pool = nn.MaxPool2d(2, 2)
        self.sequence = nn.LSTM(128, hiddenSize, num_layers=2, bidirectional=True)
        self.text_head = nn.Linear(hiddenSize*2, numClasses)

    def forward(self, x):
        x = self.pool(self.stage1(x))
        x = self.pool(self.stage2(x))
        x = self.stage3(x)
        x = torch.nn.functional.adaptive_max_pool2d(x, (1, x.shape[3]))
        x = x[:, :, 0, :].permute(2, 0, 1).contiguous()   #(W,B,128), width is the ctc time axis
        x, _ = self.sequence(x)
        return self.text_head(x)


class NamerTorch(nn.Module):
    """the same backbone with the width averaged away too, one score per writer"""

    def __init__(self, numAuthors):
        super().__init__()
        self.stage1 = ConvBlock(1, 32, 2)
        self.stage2 = ConvBlock(32, 64, 4)
        self.stage3 = ConvBlock(64, 128, 6)
        self.pool = nn.MaxPool2d(2, 2)
        self.author_head = nn.Sequential(nn.Linear(128, 96), nn.ReLU(inplace=True),
                                         nn.Dropout(0.2), nn.Linear(96, numAuthors))

    def forward(self, x):
        x = self.pool(self.stage1(x))
        x = self.pool(self.stage2(x))
        x = self.stage3(x)
        x = torch.nn.functional.adaptive_max_pool2d(x, (1, x.shape[3]))
        return self.author_head(x[:, :, 0, :].mean(dim=2))


#---SeedFromNumpy: start from the same weights train.py would have started from---
def SeedFromNumpy(model, params):
    sd = model.state_dict()
    for k, v in params.items():
        if k in sd:
            sd[k] = torch.from_numpy(np.ascontiguousarray(v))
    model.load_state_dict(sd, strict=False)   #strict off only for num_batches_tracked


#---SaveNpz: weights in the layout recognise.py reads---
def SaveNpz(path, model, extra=None):
    payload = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()
               if not k.endswith("num_batches_tracked")}
    if extra:
        payload.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), **payload)
    print("[saved] %s" % path)


#======================================================================
#THE READER
#======================================================================
def TrainReader(args):
    if args.source == "teklia":
        root = paths.Data.parents[1]/"Data"/"Datasets"/"TekliaIamLines"
        trainSet, valSet = TR.BuildTekliaSplit(str(root))
    else:
        trainSet, valSet = TR.BuildPagesSplit(args.pages, str(TR.DEFAULT_TEXT_DATA_DIR))
    print("[split] %d train lines, %d held out" % (len(trainSet), len(valSet)))

    model = ReaderTorch(len(CLS.CHARSET)+1)
    SeedFromNumpy(model, TR.InitTextModelParams(args.seed, len(CLS.CHARSET)+1))
    model.train()
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                    factor=0.5, patience=5, min_lr=1e-5)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    best = np.inf
    outPath = paths.Weights/"paper_cnn_bilstm_ctc_library_best.npz"
    rng = np.random.default_rng(args.seed)

    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        total, nB = 0.0, 0
        for batch in TR.IterateBatches(trainSet, args.batch, rng, True):
            images, flat, lengths, _texts = TR.TextCollateFn(batch)
            x = torch.from_numpy(images)
            logProbs = torch.log_softmax(model(x), dim=2)
            iLens = torch.full((x.shape[0],), logProbs.shape[0], dtype=torch.long)
            loss = ctc(logProbs, torch.from_numpy(flat), iLens, torch.from_numpy(lengths))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            nB += 1
        trainLoss = total/max(1, nB)
        valLoss, valAcc = ScoreReader(model, valSet, args.batch, ctc)
        scheduler.step(valLoss)
        saved = ""
        if valLoss < best:
            best = valLoss
            SaveNpz(outPath, model, {"__epoch__": np.array(epoch), "__best__": np.array(best)})
            saved = " [best]"
        print("[reader] epoch %03d | train %.4f | val %.4f | val char acc %.2f%% | %.1f min%s"
              % (epoch, trainLoss, valLoss, 100*valAcc, (time.time()-t0)/60, saved), flush=True)
    return best


def ScoreReader(model, valSet, batchSize, ctc):
    if len(valSet) == 0:
        return float("inf"), 0.0
    model.eval()
    total, nB, chars, errors = 0.0, 0, 0, 0
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for batch in TR.IterateBatches(valSet, batchSize, rng, False):
            images, flat, lengths, texts = TR.TextCollateFn(batch)
            x = torch.from_numpy(images)
            logProbs = torch.log_softmax(model(x), dim=2)
            iLens = torch.full((x.shape[0],), logProbs.shape[0], dtype=torch.long)
            total += float(ctc(logProbs, torch.from_numpy(flat), iLens,
                               torch.from_numpy(lengths)))
            nB += 1
            for pred, truth in zip(CLS.DecodeCtc(logProbs.numpy()), texts):
                errors += CLS.Levenshtein(pred, truth)
                chars += max(1, len(truth))
    model.train()
    return total/max(1, nB), max(0.0, 1.0-errors/max(1, chars))


#======================================================================
#THE NAMER
#trained twice in the project: once on raw grey crops, giving the ink
#model, and once on stroke normalised crops, giving the shape model that
#judges drawn output
#======================================================================
def TrainNamer(args):
    base = TR.IAMLineDatasetRaw(rootDir=str(TR.DEFAULT_AUTHOR_DATA_DIR),
                                cacheDir=str(TR.DEFAULT_AUTHOR_CACHE_DIR))
    before = len(base.samples)
    base.samples = [s for s in base.samples
                    if s["page_key"].split("/")[0] in paths.DatabaseWriters]
    print("[filter] kept %d of %d database samples" % (len(base.samples), before))
    TR.AddPersonalSamples(base)
    base.author_folders = sorted(paths.DatabaseWriters+paths.PersonalWriters)
    base.author_to_idx = {a: i for i, a in enumerate(base.author_folders)}
    trainIdx, valIdx = base.HoldoutSplitIndices()
    trainSet = TR.AuthorLabeledView(base, trainIdx, isTrain=True)
    valSet = TR.AuthorLabeledView(base, valIdx, isTrain=False)
    print("[split] %d train crops, %d held out, %d writers"
          % (len(trainIdx), len(valIdx), len(base.author_folders)))

    numAuthors = len(base.author_folders)
    model = NamerTorch(numAuthors)
    SeedFromNumpy(model, TR.InitAuthorModelParams(args.seed, numAuthors))
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    lossFn = nn.CrossEntropyLoss()
    best = 0.0
    outPath = paths.Weights/"author_classifier_library_weights.npz"
    rng = np.random.default_rng(args.seed)

    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        total, nB = 0.0, 0
        for batch in TR.IterateBatches(trainSet, args.batch, rng, True):
            images, labels, _keys = TR.AuthorCollateFn(batch)
            logits = model(torch.from_numpy(images))
            loss = lossFn(logits, torch.from_numpy(labels))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total += float(loss.item())
            nB += 1
        scheduler.step()
        acc = ScoreNamer(model, valSet, args.batch)
        saved = ""
        if acc > best:
            best = acc
            SaveNpz(outPath, model, {"__author_names__": np.array(base.author_folders),
                                     "__epoch__": np.array(epoch), "__best__": np.array(best)})
            saved = " [best]"
        print("[namer] epoch %03d | loss %.4f | val writer acc %.2f%% | %.1f min%s"
              % (epoch, total/max(1, nB), 100*acc, (time.time()-t0)/60, saved), flush=True)
    return best


def ScoreNamer(model, valSet, batchSize):
    if len(valSet) == 0:
        return 0.0
    model.eval()
    correct, seen = 0, 0
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for batch in TR.IterateBatches(valSet, batchSize, rng, False):
            images, labels, _keys = TR.AuthorCollateFn(batch)
            preds = model(torch.from_numpy(images)).argmax(dim=1).numpy()
            correct += int((preds == labels).sum())
            seen += len(labels)
    model.train()
    return correct/max(1, seen)


def Main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("task", choices=["text", "author"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--source", choices=["pages", "teklia"], default="pages",
                    help="reader only: cut the pages up, or use the pre-cut line crops")
    ap.add_argument("--pages", type=int, default=None, help="limit how many pages are cut up")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    if args.task == "text":
        TrainReader(args)
    else:
        TrainNamer(args)


if __name__ == "__main__":
    Main()
