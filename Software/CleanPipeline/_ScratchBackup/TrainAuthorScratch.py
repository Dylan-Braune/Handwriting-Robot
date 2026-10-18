#trains AuthorClassifierCNN (same conv backbone as the text model + pooling
#head) with zero ml libraries -- only NumpyNN.py's gradient-checked
#primitives + numpy/PIL. simplified single-path version of TrainAuthor.py's
#multi-subcommand design (original/10authors/shape): this trains the current
#10authors set (AuthorsConfig.DATASET_AUTHORS + PERSONAL_AUTHORS), matching
#TrainAuthor.py's Main10Authors behavior (minus the text-model backbone warm
#start, which would need a text checkpoint already trained from scratch here).
#data loading (IAMLineDatasetRaw, AuthorLabeledView, AddPersonalSamples,
#CollateFn) is reused as-is from TrainText.py/TrainAuthor.py -- see
#TrainTextScratch.py's module comment for why only a .numpy() call is needed
#at the collate boundary instead of a rewrite.
#checkpoints are plain .npz (np.savez), keyed exactly like AuthorClassifierCNN's
#state_dict (author_head.0/.3) so Classify.py's existing from-scratch
#inference loads them unmodified (see the npz branch in Classify.LoadStateDictAny).
#usage: python TrainAuthorScratch.py [--epochs N] [--batch-size N]
import argparse
import time
from pathlib import Path

import numpy as np

import NumpyNN as NN
import Classify as CLS
from TrainAuthor import IAMLineDatasetRaw,AuthorLabeledView,CollateFn,AddPersonalSamples
from AuthorsConfig import DATASET_AUTHORS,PERSONAL_AUTHORS

SCRIPT_DIR = Path(__file__).resolve().parent
CNN_DATA_DIR = SCRIPT_DIR.parent/"CNN"
DEFAULT_DATA_DIR = SCRIPT_DIR.parents[1]/"Data"/"Datasets"/"IAMpages10"
DEFAULT_CACHE_DIR = CNN_DATA_DIR/"NOGIT"/"line_cache_authors10"
WEIGHTS_DIR = CNN_DATA_DIR/"weights"


#=== parameter init -- see TrainTextScratch.py's InitConv/InitBatchNorm/
#InitLinear for the full reasoning, duplicated here since this file stays
#standalone (the two scripts share no third module on purpose) ===

def InitConv(rng,outCh,inCh,kh,kw):
    fanIn = inCh*kh*kw
    std = np.sqrt(2.0/fanIn)
    w = rng.normal(0.0,std,size=(outCh,inCh,kh,kw)).astype(np.float32)
    b = np.zeros(outCh,dtype=np.float32)
    return w,b


def InitBatchNorm(ch):
    return (np.ones(ch,dtype=np.float32),np.zeros(ch,dtype=np.float32),
            np.zeros(ch,dtype=np.float32),np.ones(ch,dtype=np.float32))


def InitLinear(rng,outF,inF):
    bound = 1.0/np.sqrt(inF)
    w = rng.uniform(-bound,bound,size=(outF,inF)).astype(np.float32)
    b = rng.uniform(-bound,bound,size=(outF,)).astype(np.float32)
    return w,b


def InitConvBlock(rng,params,prefix,inCh,outCh,nLayers):
    ch = inCh
    for i in range(nLayers):
        c = i*3
        w,b = InitConv(rng,outCh,ch,3,3)
        params[f"{prefix}.{c}.weight"] = w
        params[f"{prefix}.{c}.bias"] = b
        gamma,beta,rm,rv = InitBatchNorm(outCh)
        params[f"{prefix}.{c+1}.weight"] = gamma
        params[f"{prefix}.{c+1}.bias"] = beta
        params[f"{prefix}.{c+1}.running_mean"] = rm
        params[f"{prefix}.{c+1}.running_var"] = rv
        ch = outCh


#builds every AuthorClassifierCNN parameter, keyed exactly like its own
#state_dict (stage1/2/3 backbone, author_head.0 + author_head.3 -- indices
#1/2 of the Sequential are relu/dropout, no params)
def InitAuthorModelParams(seed,numAuthors):
    rng = np.random.default_rng(seed)
    params = {}
    InitConvBlock(rng,params,"stage1",1,32,2)
    InitConvBlock(rng,params,"stage2",32,64,4)
    InitConvBlock(rng,params,"stage3",64,128,6)
    w0,b0 = InitLinear(rng,96,128)
    params["author_head.0.weight"] = w0
    params["author_head.0.bias"] = b0
    w3,b3 = InitLinear(rng,numAuthors,96)
    params["author_head.3.weight"] = w3
    params["author_head.3.bias"] = b3
    return params


def TrainableParams(params):
    return {k:v for k,v in params.items() if not k.endswith("running_mean") and not k.endswith("running_var")}


#=== forward/backward through one conv stage (identical to TrainTextScratch.py) ===

def ConvBlockForwardTrain(x,params,prefix,nLayers):
    caches = []
    for i in range(nLayers):
        c = i*3
        x,convCache = NN.Conv2dForward(x,params[f"{prefix}.{c}.weight"],params[f"{prefix}.{c}.bias"],padding=1)
        x,bnCache = NN.BatchNorm2dForwardTrain(
            x,params[f"{prefix}.{c+1}.weight"],params[f"{prefix}.{c+1}.bias"],
            params[f"{prefix}.{c+1}.running_mean"],params[f"{prefix}.{c+1}.running_var"],
        )
        x,reluCache = NN.ReluForward(x)
        caches.append((convCache,bnCache,reluCache))
    return x,caches


def ConvBlockBackwardTrain(dout,caches,prefix,nLayers):
    grads = {}
    for i in reversed(range(nLayers)):
        c = i*3
        convCache,bnCache,reluCache = caches[i]
        dout = NN.ReluBackward(dout,reluCache)
        dout,dgamma,dbeta = NN.BatchNorm2dBackward(dout,bnCache)
        dout,dw,db = NN.Conv2dBackward(dout,convCache)
        grads[f"{prefix}.{c}.weight"] = dw
        grads[f"{prefix}.{c}.bias"] = db
        grads[f"{prefix}.{c+1}.weight"] = dgamma
        grads[f"{prefix}.{c+1}.bias"] = dbeta
    return dout,grads


#=== full model forward/backward (training mode) ===

#x:(B,1,H,W) -> logits:(B,NumAuthors)
def AuthorModelForward(x,params,training,rng):
    out,c1 = ConvBlockForwardTrain(x,params,"stage1",2)
    out,poolCache1 = NN.MaxPool2dForward(out,2,2)
    out,c2 = ConvBlockForwardTrain(out,params,"stage2",4)
    out,poolCache2 = NN.MaxPool2dForward(out,2,2)
    out,c3 = ConvBlockForwardTrain(out,params,"stage3",6)

    pooledH,hpCache = NN.AdaptiveMaxPoolHeightForward(out)   #(B,128,1,W)
    squeezed = pooledH[:,:,0,:]                               #(B,128,W)
    pooledW,apCache = NN.AdaptiveAvgPoolWidthForward(squeezed)  #(B,128,1), width collapsed too (unlike the text model)
    pooled = pooledW[:,:,0]                                    #(B,128)

    h,lin1Cache = NN.LinearForward(pooled,params["author_head.0.weight"],params["author_head.0.bias"])
    h,reluCache = NN.ReluForward(h)
    h,dropCache = NN.DropoutForward(h,0.2,training,rng)
    logits,lin2Cache = NN.LinearForward(h,params["author_head.3.weight"],params["author_head.3.bias"])

    cache = dict(c1=c1,c2=c2,c3=c3,poolCache1=poolCache1,poolCache2=poolCache2,hpCache=hpCache,apCache=apCache,
                 lin1Cache=lin1Cache,reluCache=reluCache,dropCache=dropCache,lin2Cache=lin2Cache)
    return logits,cache


def AuthorModelBackward(dLogits,cache):
    grads = {}
    dh,dw2,db2 = NN.LinearBackward(dLogits,cache["lin2Cache"])
    grads["author_head.3.weight"] = dw2
    grads["author_head.3.bias"] = db2

    dh = NN.DropoutBackward(dh,cache["dropCache"])
    dh = NN.ReluBackward(dh,cache["reluCache"])
    dpooled,dw1,db1 = NN.LinearBackward(dh,cache["lin1Cache"])
    grads["author_head.0.weight"] = dw1
    grads["author_head.0.bias"] = db1

    dPooledW = dpooled[:,:,None]                                   #(B,128,1)
    dSqueezed = NN.AdaptiveAvgPoolWidthBackward(dPooledW,cache["apCache"])  #(B,128,W)
    dPooledH = dSqueezed[:,:,None,:]                               #(B,128,1,W)
    dout = NN.AdaptiveMaxPoolHeightBackward(dPooledH,cache["hpCache"])

    dout,g3 = ConvBlockBackwardTrain(dout,cache["c3"],"stage3",6)
    grads.update(g3)
    dout = NN.MaxPool2dBackward(dout,cache["poolCache2"])
    dout,g2 = ConvBlockBackwardTrain(dout,cache["c2"],"stage2",4)
    grads.update(g2)
    dout = NN.MaxPool2dBackward(dout,cache["poolCache1"])
    dout,g1 = ConvBlockBackwardTrain(dout,cache["c1"],"stage1",2)
    grads.update(g1)
    return dout,grads


#=== eval-mode forward (running-stat batchnorm, no dropout) -- reuses
#Classify.py's own tested primitives directly, same as
#AuthorClassifierCNNNumpy.forward ===

def EvalAuthorForward(params,x):
    sd = params
    x = CLS.ConvBlockForward(x,sd,"stage1",2)
    x = CLS.MaxPool2d(x,2,2)
    x = CLS.ConvBlockForward(x,sd,"stage2",4)
    x = CLS.MaxPool2d(x,2,2)
    x = CLS.ConvBlockForward(x,sd,"stage3",6)
    x = CLS.AdaptiveMaxPoolHeight(x)
    x = x[:,:,0,:]
    pooled = CLS.AdaptiveAvgPoolWidth(x)[:,:,0]
    h = CLS.Relu(CLS.Linear(pooled,sd["author_head.0.weight"],sd["author_head.0.bias"]))
    return CLS.Linear(h,sd["author_head.3.weight"],sd["author_head.3.bias"])


#=== manual batching, replacing torch.utils.data.DataLoader ===

def IterateBatches(dataset,batchSize,rng,shuffle):
    n = len(dataset)
    order = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0,n,batchSize):
        idxs = order[start:start+batchSize]
        yield [dataset[int(i)] for i in idxs]


def NumpyBatch(batch):
    images_t,authorIdxs_t,pageKeys = CollateFn(batch)
    return images_t.numpy(),authorIdxs_t.numpy(),pageKeys


def ValidateAuthorModel(valSet,params,batchSize,rng):
    if len(valSet)==0:
        return 0.0
    correct,total = 0,0
    for batch in IterateBatches(valSet,batchSize,rng,shuffle=False):
        images,authorIdxs,_ = NumpyBatch(batch)
        logits = EvalAuthorForward(params,images)
        preds = np.argmax(logits,axis=1)
        correct+=int((preds==authorIdxs).sum())
        total+=len(authorIdxs)
    return correct/max(1,total)


#npz can't hold a python dict directly -- stash the author_mapping as a
#plain array of names sorted by index (Classify.LoadStateDictAny rebuilds
#the {name: idx} dict on load; see its module comment there)
def SaveCheckpoint(path,params,authorToIdx):
    path.parent.mkdir(parents=True,exist_ok=True)
    payload = dict(params)
    names = sorted(authorToIdx,key=lambda a:authorToIdx[a])
    payload["__author_names__"] = np.array(names)
    np.savez(str(path),**payload)


#=== training loop: AdamW lr=1e-3 wd=1e-4, CosineAnnealingLR, grad-clip at
#norm 5.0 -- same recipe as TrainAuthor.py's Main10Authors(), minus the
#text-model backbone warm start (no from-scratch text checkpoint to warm-
#start from yet) and the original/shape subcommands (simplified single
#path, see module comment) ===

def TrainAuthorModel(numEpochs,batchSize,lr,seed):
    base = IAMLineDatasetRaw(rootDir=str(DEFAULT_DATA_DIR),cacheDir=str(DEFAULT_CACHE_DIR))
    before = len(base.samples)
    base.samples = [s for s in base.samples if s["page_key"].split("/")[0] in DATASET_AUTHORS]
    print(f"[Filter] kept {len(base.samples)}/{before} IAM samples for {DATASET_AUTHORS}")

    AddPersonalSamples(base)
    base.author_folders = sorted(DATASET_AUTHORS+PERSONAL_AUTHORS)
    base.author_to_idx = {a:i for i,a in enumerate(base.author_folders)}
    print(f"[Authors] {len(base.author_folders)}: {base.author_folders}")

    trainIdx,valIdx = base.HoldoutSplitIndices()
    if not valIdx:
        raise RuntimeError("No holdout lines found.")
    trainSet = AuthorLabeledView(base,trainIdx,isTrain=True)
    valSet = AuthorLabeledView(base,valIdx,isTrain=False)
    print(f"[Split] Train lines: {len(trainIdx)} | Held-out val lines: {len(valIdx)}")

    numAuthors = len(base.author_folders)
    params = InitAuthorModelParams(seed,numAuthors)
    trainable = TrainableParams(params)
    optimizer = NN.AdamW(trainable,lr=lr,weightDecay=1e-4)
    scheduler = NN.CosineAnnealingLr(optimizer,tMax=numEpochs)

    rngBatch = np.random.default_rng(seed)
    rngDropout = np.random.default_rng(seed+1)
    bestAcc = 0.0
    outPath = WEIGHTS_DIR/"author_classifier_10new_scratch_weights.npz"

    for epoch in range(1,numEpochs+1):
        epochStart = time.time()
        epochLoss,nBatches = 0.0,0
        for batch in IterateBatches(trainSet,batchSize,rngBatch,shuffle=True):
            images,authorIdxs,_ = NumpyBatch(batch)
            logits,cache = AuthorModelForward(images,params,training=True,rng=rngDropout)
            loss,dLogits = NN.CrossEntropyLoss(logits,authorIdxs)
            _,grads = AuthorModelBackward(dLogits,cache)
            trainableGrads = {k:grads[k] for k in trainable}
            NN.ClipGradNorm(trainableGrads,maxNorm=5.0)
            optimizer.Step(trainableGrads)
            epochLoss+=loss
            nBatches+=1
        scheduler.Step()
        trainLoss = epochLoss/max(1,nBatches)

        valAcc = ValidateAuthorModel(valSet,params,batchSize,rngBatch)
        savedMsg = ""
        if valAcc>bestAcc:
            bestAcc = valAcc
            SaveCheckpoint(outPath,params,base.author_to_idx)
            savedMsg = " [SAVED]"

        print(f"Epoch {epoch:03d}/{numEpochs} | loss {trainLoss:.3f} | val author-acc {valAcc:.3f} "
              f"(best {bestAcc:.3f}) | lr {optimizer.lr:.6f} | {(time.time()-epochStart)/60:.1f} min{savedMsg}")

    print(f"\nBest val author-accuracy: {bestAcc:.3f}")
    print(f"Weights saved to: {outPath}")
    return bestAcc


#=== cheap sanity check: synthetic data, no dataset needed -- confirms the
#model builds, forward produces the right shape, and a full training step
#(forward+backward+clip+optimizer.Step) runs without crashing ===

def SelfTest():
    rng = np.random.default_rng(0)
    numAuthors = 5
    params = InitAuthorModelParams(seed=0,numAuthors=numAuthors)
    trainable = TrainableParams(params)
    optimizer = NN.AdamW(trainable,lr=1e-3,weightDecay=1e-4)

    B,H,W = 3,64,640
    x = rng.standard_normal((B,1,H,W)).astype(np.float32)
    authorIdxs = rng.integers(0,numAuthors,size=B)
    dropoutRng = np.random.default_rng(1)

    logits,cache = AuthorModelForward(x,params,training=True,rng=dropoutRng)
    assert logits.shape==(B,numAuthors),logits.shape

    loss,dLogits = NN.CrossEntropyLoss(logits,authorIdxs)
    assert np.isfinite(loss)
    _,grads = AuthorModelBackward(dLogits,cache)
    trainableGrads = {k:grads[k] for k in trainable}
    NN.ClipGradNorm(trainableGrads,maxNorm=5.0)
    optimizer.Step(trainableGrads)

    print(f"[selftest] author model forward+backward+step ok | logits shape {logits.shape} | loss {loss:.4f}")
    return loss


def Main():
    parser = argparse.ArgumentParser(description="Train AuthorClassifierCNN from scratch (NumpyNN only, no torch).")
    parser.add_argument("--epochs",type=int,default=20)
    parser.add_argument("--batch-size",type=int,default=16)
    parser.add_argument("--lr",type=float,default=1e-3)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--selftest",action="store_true",help="run the synthetic smoke test and exit")
    args = parser.parse_args()

    if args.selftest:
        SelfTest()
        return

    print(f"\n--- Training author_classifier_10new_scratch (lr={args.lr}, batch_size={args.batch_size}, "
          f"max_epochs={args.epochs}) ---")
    TrainAuthorModel(args.epochs,args.batch_size,args.lr,args.seed)


if __name__=="__main__":
    Main()
