#trains PaperCRNN (conv backbone + bidirectional lstm + ctc) with zero ml
#libraries -- only NumpyNN.py's gradient-checked primitives + numpy/PIL.
#simplified single-path version of TrainText.py's multi-subcommand design
#(base/hf/joint/personal): this trains on IAMpages671 only, matching
#TrainText.py's original `base` behavior. data loading (IAMLineDatasetRaw,
#TrainEvalView, CollateFn, augmentation) is reused as-is from TrainText.py --
#that pipeline was already pure PIL/numpy except for the final torch.stack/
#torch.cat in CollateFn, so batches are converted back to numpy with one
#.numpy() call each rather than reimplementing collation.
#checkpoints are plain .npz (np.savez), keyed exactly like a pytorch
#state_dict so Classify.py's existing from-scratch inference loads them
#unmodified (see the npz branch added to Classify.LoadStateDictAny).
#usage: python TrainTextScratch.py [--epochs N] [--batch-size N] [--max-pages N]
import argparse
import sys
import time
from pathlib import Path

import numpy as np

import NumpyNN as NN
import Classify as CLS
from Charset import CHARSET
from TrainText import IAMLineDatasetRaw,TrainEvalView,CollateFn

SCRIPT_DIR = Path(__file__).resolve().parent
CNN_DATA_DIR = SCRIPT_DIR.parent/"CNN"
DEFAULT_DATA_DIR = SCRIPT_DIR.parents[1]/"Data"/"Datasets"/"IAMpages671"
DEFAULT_CACHE_DIR = CNN_DATA_DIR/"NOGIT"/"line_image_cache"
WEIGHTS_DIR = CNN_DATA_DIR/"weights"
HIDDEN_SIZE = 256


#=== parameter init, approximating pytorch's own defaults (annealing over a
#real training run washes out init differences, so exactness doesn't matter
#here -- just the right general shape) ===

def InitConv(rng,outCh,inCh,kh,kw):
    fanIn = inCh*kh*kw
    std = np.sqrt(2.0/fanIn)   #kaiming normal, matches nn.Conv2d's relu-gain default
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


def InitLstmDirection(rng,inputSize,hiddenSize):
    bound = 1.0/np.sqrt(hiddenSize)   #matches nn.LSTM's own uniform init range
    wIh = rng.uniform(-bound,bound,size=(4*hiddenSize,inputSize)).astype(np.float32)
    wHh = rng.uniform(-bound,bound,size=(4*hiddenSize,hiddenSize)).astype(np.float32)
    bIh = rng.uniform(-bound,bound,size=(4*hiddenSize,)).astype(np.float32)
    bHh = rng.uniform(-bound,bound,size=(4*hiddenSize,)).astype(np.float32)
    return wIh,wHh,bIh,bHh


#one conv stage: nLayers x (conv3x3,pad1 -> batchnorm -> relu). conv at flat
#index i*3, bn at i*3+1 -- matches Classify.ConvBlockForward's key layout
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


#builds every PaperCRNN parameter, keyed exactly like PaperCRNN's own
#state_dict (stage1/2/3, sequence.*_l{layer}[_reverse], text_head)
def InitTextModelParams(seed,numClasses):
    rng = np.random.default_rng(seed)
    params = {}
    InitConvBlock(rng,params,"stage1",1,32,2)
    InitConvBlock(rng,params,"stage2",32,64,4)
    InitConvBlock(rng,params,"stage3",64,128,6)

    lstmInputSizes = [128,HIDDEN_SIZE*2]   #layer0 sees the backbone's 128 channels, layer1 sees 2*hidden
    for l,inSize in enumerate(lstmInputSizes):
        wIh,wHh,bIh,bHh = InitLstmDirection(rng,inSize,HIDDEN_SIZE)
        params[f"sequence.weight_ih_l{l}"] = wIh
        params[f"sequence.weight_hh_l{l}"] = wHh
        params[f"sequence.bias_ih_l{l}"] = bIh
        params[f"sequence.bias_hh_l{l}"] = bHh
        wIhR,wHhR,bIhR,bHhR = InitLstmDirection(rng,inSize,HIDDEN_SIZE)
        params[f"sequence.weight_ih_l{l}_reverse"] = wIhR
        params[f"sequence.weight_hh_l{l}_reverse"] = wHhR
        params[f"sequence.bias_ih_l{l}_reverse"] = bIhR
        params[f"sequence.bias_hh_l{l}_reverse"] = bHhR

    w,b = InitLinear(rng,numClasses,HIDDEN_SIZE*2)
    params["text_head.weight"] = w
    params["text_head.bias"] = b
    return params


#optimizer only ever sees trainable arrays -- batchnorm running stats are
#buffers, updated in place by BatchNorm2dForwardTrain itself, never by a gradient
def TrainableParams(params):
    return {k:v for k,v in params.items() if not k.endswith("running_mean") and not k.endswith("running_var")}


#=== forward/backward through one conv stage, using NumpyNN's primitives ===

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


#=== full model forward/backward (training mode: batch-stat batchnorm) ===

#x:(B,1,H,W) -> rawLogits:(T,B,NumClasses) where T=W/4 (two 2x2 maxpools).
#NOT log-softmaxed on purpose -- NumpyNN.CtcLoss does log_softmax internally
#and returns the gradient straight w.r.t. these raw logits
def TextModelForward(x,params):
    x,c1 = ConvBlockForwardTrain(x,params,"stage1",2)
    x,poolCache1 = NN.MaxPool2dForward(x,2,2)
    x,c2 = ConvBlockForwardTrain(x,params,"stage2",4)
    x,poolCache2 = NN.MaxPool2dForward(x,2,2)
    x,c3 = ConvBlockForwardTrain(x,params,"stage3",6)

    pooled,hpCache = NN.AdaptiveMaxPoolHeightForward(x)   #(B,128,1,W)
    B,C,_,W = pooled.shape
    squeezed = pooled[:,:,0,:]                            #(B,128,W)
    seqIn = np.transpose(squeezed,(2,0,1))                #(W,B,128), width is the ctc time axis

    layerParams = []
    for l in range(2):
        fwd = dict(weightIh=params[f"sequence.weight_ih_l{l}"],weightHh=params[f"sequence.weight_hh_l{l}"],
                   biasIh=params[f"sequence.bias_ih_l{l}"],biasHh=params[f"sequence.bias_hh_l{l}"])
        bwd = dict(weightIh=params[f"sequence.weight_ih_l{l}_reverse"],weightHh=params[f"sequence.weight_hh_l{l}_reverse"],
                   biasIh=params[f"sequence.bias_ih_l{l}_reverse"],biasHh=params[f"sequence.bias_hh_l{l}_reverse"])
        layerParams.append({"fwd":fwd,"bwd":bwd})
    seqOut,lstmCaches = NN.MultilayerBidirectionalLstmForwardTrain(seqIn,layerParams,HIDDEN_SIZE)

    rawLogits,linCache = NN.LinearForward(seqOut,params["text_head.weight"],params["text_head.bias"])
    cache = dict(c1=c1,c2=c2,c3=c3,poolCache1=poolCache1,poolCache2=poolCache2,hpCache=hpCache,
                 lstmCaches=lstmCaches,linCache=linCache,bchw=(B,C,W))
    return rawLogits,cache


def TextModelBackward(dRawLogits,cache):
    grads = {}
    dSeqOut,dw,db = NN.LinearBackward(dRawLogits,cache["linCache"])
    grads["text_head.weight"] = dw
    grads["text_head.bias"] = db

    dSeqIn,gradsPerLayer = NN.MultilayerBidirectionalLstmBackward(dSeqOut,cache["lstmCaches"])
    for l,(gradsFwd,gradsBwd) in enumerate(gradsPerLayer):
        grads[f"sequence.weight_ih_l{l}"] = gradsFwd["weightIh"]
        grads[f"sequence.weight_hh_l{l}"] = gradsFwd["weightHh"]
        grads[f"sequence.bias_ih_l{l}"] = gradsFwd["biasIh"]
        grads[f"sequence.bias_hh_l{l}"] = gradsFwd["biasHh"]
        grads[f"sequence.weight_ih_l{l}_reverse"] = gradsBwd["weightIh"]
        grads[f"sequence.weight_hh_l{l}_reverse"] = gradsBwd["weightHh"]
        grads[f"sequence.bias_ih_l{l}_reverse"] = gradsBwd["biasIh"]
        grads[f"sequence.bias_hh_l{l}_reverse"] = gradsBwd["biasHh"]

    B,C,W = cache["bchw"]
    dSqueezed = np.transpose(dSeqIn,(1,2,0))   #(W,B,128) -> (B,128,W)
    dPooled = dSqueezed[:,:,None,:]            #(B,128,1,W)
    dx = NN.AdaptiveMaxPoolHeightBackward(dPooled,cache["hpCache"])

    dx,g3 = ConvBlockBackwardTrain(dx,cache["c3"],"stage3",6)
    grads.update(g3)
    dx = NN.MaxPool2dBackward(dx,cache["poolCache2"])
    dx,g2 = ConvBlockBackwardTrain(dx,cache["c2"],"stage2",4)
    grads.update(g2)
    dx = NN.MaxPool2dBackward(dx,cache["poolCache1"])
    dx,g1 = ConvBlockBackwardTrain(dx,cache["c1"],"stage1",2)
    grads.update(g1)
    return dx,grads


#=== eval-mode forward (running-stat batchnorm, matches real inference) --
#reuses Classify.py's own tested primitives directly, stopping one step
#before its log_softmax so NumpyNN.CtcLoss can be fed raw logits ===

def BuildLstmLayerParams(sd):
    layerParams = []
    for l in range(2):
        fwd = dict(weight_ih=sd[f"sequence.weight_ih_l{l}"],weight_hh=sd[f"sequence.weight_hh_l{l}"],
                   bias_ih=sd[f"sequence.bias_ih_l{l}"],bias_hh=sd[f"sequence.bias_hh_l{l}"])
        bwd = dict(weight_ih=sd[f"sequence.weight_ih_l{l}_reverse"],weight_hh=sd[f"sequence.weight_hh_l{l}_reverse"],
                   bias_ih=sd[f"sequence.bias_ih_l{l}_reverse"],bias_hh=sd[f"sequence.bias_hh_l{l}_reverse"])
        layerParams.append({"fwd":fwd,"bwd":bwd})
    return layerParams


def EvalForwardRawLogits(params,x):
    sd = params
    x = CLS.ConvBlockForward(x,sd,"stage1",2)
    x = CLS.MaxPool2d(x,2,2)
    x = CLS.ConvBlockForward(x,sd,"stage2",4)
    x = CLS.MaxPool2d(x,2,2)
    x = CLS.ConvBlockForward(x,sd,"stage3",6)
    x = CLS.AdaptiveMaxPoolHeight(x)
    x = x[:,:,0,:]
    x = np.transpose(x,(2,0,1))
    seq = CLS.MultilayerBidirectionalLstm(x,BuildLstmLayerParams(sd),HIDDEN_SIZE)
    return CLS.Linear(seq,sd["text_head.weight"],sd["text_head.bias"])


#=== manual batching, replacing torch.utils.data.DataLoader ===

def IterateBatches(dataset,batchSize,rng,shuffle):
    n = len(dataset)
    order = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0,n,batchSize):
        idxs = order[start:start+batchSize]
        yield [dataset[int(i)] for i in idxs]


#CollateFn still returns torch tensors (the dataset's own TensorFromResized
#builds them) -- converting at this one boundary with .numpy() is far less
#code than a from-scratch collate/augmentation rewrite
def NumpyBatch(batch):
    images_t,flatTargets_t,targetLengths_t,texts = CollateFn(batch)
    images = images_t.numpy()
    flatTargets = flatTargets_t.numpy()
    targetLengths = targetLengths_t.numpy()
    targets,offset = [],0
    for length in targetLengths:
        length = int(length)
        targets.append(flatTargets[offset:offset+length].tolist())
        offset+=length
    return images,targets,texts


def ValidateTextModel(valSet,params,batchSize,rng):
    if len(valSet)==0:
        return 0.0,0.0
    totalLoss,nBatches = 0.0,0
    totalChars,totalCharErrors = 0,0
    for batch in IterateBatches(valSet,batchSize,rng,shuffle=False):
        images,targets,texts = NumpyBatch(batch)
        rawLogits = EvalForwardRawLogits(params,images)
        loss,_ = NN.CtcLoss(rawLogits,targets)
        totalLoss+=loss
        nBatches+=1
        logProbs = CLS.LogSoftmax(rawLogits,axis=2)
        preds = CLS.DecodeCtc(logProbs)
        for pred,truth in zip(preds,texts):
            totalChars+=len(truth)
            totalCharErrors+=CLS.Levenshtein(pred,truth)
    avgLoss = totalLoss/max(1,nBatches)
    charAcc = 1.0-totalCharErrors/max(1,totalChars)
    return avgLoss,charAcc


def SaveCheckpoint(path,params):
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez(str(path),**params)


#=== training loop: RMSprop lr=1e-3 wd=1e-5, ReduceLROnPlateau on val loss --
#same recipe as TrainText.py's Train(), minus warm restarts/resume/HF/joint/
#personal variants (simplified single path, see module comment) ===

def TrainTextModel(numEpochs,batchSize,lr,seed,maxPages,dataDir):
    dataset = IAMLineDatasetRaw(rootDir=dataDir,cacheDir=DEFAULT_CACHE_DIR,maxPages=maxPages)
    if len(dataset)<4:
        raise RuntimeError("Not enough line samples to train.")
    trainIdx,testIdx = dataset.HoldoutSplitIndices()
    if not testIdx:
        raise RuntimeError("No held-out lines found.")
    trainSet = TrainEvalView(dataset,trainIdx,isTrain=True)
    valSet = TrainEvalView(dataset,testIdx,isTrain=False)
    print(f"[Split] Train lines: {len(trainSet)} | Held-out val lines: {len(valSet)}")

    numClasses = len(CHARSET)+1
    params = InitTextModelParams(seed,numClasses)
    trainable = TrainableParams(params)
    optimizer = NN.RMSprop(trainable,lr=lr,weightDecay=1e-5)
    scheduler = NN.ReduceLrOnPlateau(optimizer,factor=0.5,patience=5,minLr=1e-5)

    rng = np.random.default_rng(seed)
    bestValLoss = np.inf
    bestPath = WEIGHTS_DIR/"paper_cnn_bilstm_ctc_scratch_best.npz"
    ckptPath = WEIGHTS_DIR/"paper_cnn_bilstm_ctc_scratch_checkpoint.npz"

    for epoch in range(1,numEpochs+1):
        epochStart = time.time()
        epochLoss,nBatches = 0.0,0
        for batch in IterateBatches(trainSet,batchSize,rng,shuffle=True):
            images,targets,_texts = NumpyBatch(batch)
            rawLogits,cache = TextModelForward(images,params)
            loss,dRawLogits = NN.CtcLoss(rawLogits,targets)
            _,grads = TextModelBackward(dRawLogits,cache)
            trainableGrads = {k:grads[k] for k in trainable}
            optimizer.Step(trainableGrads)
            epochLoss+=loss
            nBatches+=1
            if nBatches==1 or nBatches%5==0:
                print(f"\r[TrainTextScratch] Epoch {epoch:03d} | Batch {nBatches:04d} | "
                      f"Running loss {epochLoss/nBatches:.3f}",end="",flush=True)
        print()
        trainLoss = epochLoss/max(1,nBatches)

        valLoss,valCharAcc = ValidateTextModel(valSet,params,batchSize,rng)
        scheduler.Step(valLoss)
        improved = valLoss<bestValLoss
        savedMsg = ""
        if improved:
            bestValLoss = valLoss
            SaveCheckpoint(bestPath,params)
            savedMsg = " [BEST SAVED]"
        SaveCheckpoint(ckptPath,params)

        print(f"[TrainTextScratch] Epoch {epoch:03d}/{numEpochs} | Train loss {trainLoss:.3f} | "
              f"Val loss {valLoss:.3f} (best {bestValLoss:.3f}) | Val char-acc {valCharAcc:.2%} | "
              f"lr {optimizer.lr:.6f} | {(time.time()-epochStart)/60:.1f} min{savedMsg}")

    return bestValLoss


#=== cheap sanity check: synthetic data, no dataset needed -- confirms the
#model builds, forward produces the right shape, and a full training step
#(forward+backward+optimizer.Step) runs without crashing. not a convergence test ===

def SelfTest():
    rng = np.random.default_rng(0)
    numClasses = len(CHARSET)+1
    params = InitTextModelParams(seed=0,numClasses=numClasses)
    trainable = TrainableParams(params)
    optimizer = NN.RMSprop(trainable,lr=1e-3,weightDecay=1e-5)

    B,H,W = 2,64,640
    x = rng.standard_normal((B,1,H,W)).astype(np.float32)
    targets = [[1,2,3],[4,5]]

    rawLogits,cache = TextModelForward(x,params)
    expectedT = W//4
    assert rawLogits.shape==(expectedT,B,numClasses),rawLogits.shape

    loss,dRawLogits = NN.CtcLoss(rawLogits,targets)
    assert np.isfinite(loss)
    _,grads = TextModelBackward(dRawLogits,cache)
    trainableGrads = {k:grads[k] for k in trainable}
    optimizer.Step(trainableGrads)

    print(f"[selftest] text model forward+backward+step ok | logits shape {rawLogits.shape} | loss {loss:.4f}")
    return loss


def Main():
    parser = argparse.ArgumentParser(description="Train PaperCRNN from scratch (NumpyNN only, no torch).")
    parser.add_argument("--epochs",type=int,default=50)
    parser.add_argument("--batch-size",type=int,default=16)
    parser.add_argument("--lr",type=float,default=1e-3)
    parser.add_argument("--seed",type=int,default=7)
    parser.add_argument("--max-pages",type=int,default=None)
    parser.add_argument("--data-dir",type=str,default=None)
    parser.add_argument("--selftest",action="store_true",help="run the synthetic smoke test and exit")
    args = parser.parse_args()

    if args.selftest:
        SelfTest()
        return

    dataDir = Path(args.data_dir) if args.data_dir else DEFAULT_DATA_DIR
    print(f"\n--- Training paper_cnn_bilstm_ctc_scratch (lr={args.lr}, batch_size={args.batch_size}, "
          f"max_epochs={args.epochs}, data={dataDir}) ---")
    best = TrainTextModel(args.epochs,args.batch_size,args.lr,args.seed,args.max_pages,dataDir)
    print(f"\nDone. Best validation loss: {best:.4f}")


if __name__=="__main__":
    Main()
