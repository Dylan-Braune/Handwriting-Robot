#trains both PaperCRNN (text) and AuthorClassifierCNN (writer-id) with zero ml
#libraries -- only NumpyNN.py's gradient-checked primitives + numpy/PIL. one
#file instead of two: both models share the identical conv backbone
#(stage1->pool1->stage2->pool2->stage3) and all of its init/forward/backward/
#batching code, so that part is written once below and used by both trainers.
#simplified single-path versions of TrainText.py/TrainAuthor.py's multi-
#subcommand designs -- text trains on IAMpages671 only (matching the `base`
#path), author trains the current DATASET_AUTHORS+PERSONAL_AUTHORS set
#(matching `10authors`, minus the text-model backbone warm start, which
#would need a text checkpoint already trained from scratch here).
#data loading (IAMLineDatasetRaw, TrainEvalView/AuthorLabeledView,
#AddPersonalSamples, CollateFn) is reused as-is from TrainText.py/
#TrainAuthor.py -- that pipeline is pure PIL/numpy throughout, CollateFn's
#own output included (no torch anywhere in this project's training code now).
#checkpoints are plain .npz (np.savez), keyed exactly like a pytorch
#state_dict so Classify.py's existing from-scratch inference loads them
#unmodified (see the npz branch in Classify.LoadStateDictAny).
#usage: python TrainScratch.py text [--epochs N] [--batch-size N] [--max-pages N]
#       python TrainScratch.py author [--epochs N] [--batch-size N]
#       python TrainScratch.py {text,author} --selftest
import argparse
import time
from pathlib import Path

import numpy as np

import NumpyNN as NN
import Classify as CLS
from Charset import CHARSET
from TrainText import IAMLineDatasetRaw,TrainEvalView,CollateFn as TextCollateFn
from TrainAuthor import AuthorLabeledView,CollateFn as AuthorCollateFn,AddPersonalSamples
from AuthorsConfig import DATASET_AUTHORS,PERSONAL_AUTHORS

SCRIPT_DIR = Path(__file__).resolve().parent
CNN_DATA_DIR = SCRIPT_DIR.parent/"CNN"
WEIGHTS_DIR = CNN_DATA_DIR/"weights"
HIDDEN_SIZE = 256   #lstm hidden size, text model only

DEFAULT_TEXT_DATA_DIR = SCRIPT_DIR.parents[1]/"Data"/"Datasets"/"IAMpages671"
DEFAULT_TEXT_CACHE_DIR = CNN_DATA_DIR/"NOGIT"/"line_image_cache"
DEFAULT_AUTHOR_DATA_DIR = SCRIPT_DIR.parents[1]/"Data"/"Datasets"/"IAMpages10"
DEFAULT_AUTHOR_CACHE_DIR = CNN_DATA_DIR/"NOGIT"/"line_cache_authors10"


#=== parameter init, approximating pytorch's own defaults (annealing over a
#real training run washes out init differences, so exactness doesn't matter
#here -- just the right general shape). shared by both models ===

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


#builds the shared backbone's parameters (stage1 1->32x2, stage2 32->64x4,
#stage3 64->128x6) -- both models use exactly this, called once by each
#model's own init function below
def InitBackboneParams(rng,params):
    InitConvBlock(rng,params,"stage1",1,32,2)
    InitConvBlock(rng,params,"stage2",32,64,4)
    InitConvBlock(rng,params,"stage3",64,128,6)


#optimizer only ever sees trainable arrays -- batchnorm running stats are
#buffers, updated in place by BatchNorm2dForwardTrain itself, never by a gradient
def TrainableParams(params):
    return {k:v for k,v in params.items() if not k.endswith("running_mean") and not k.endswith("running_var")}


#=== shared conv backbone forward/backward (training mode), used by both
#models -- everything through stage3 is byte-identical between them ===

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


def BackboneForwardTrain(x,params):
    x,c1 = ConvBlockForwardTrain(x,params,"stage1",2)
    x,poolCache1 = NN.MaxPool2dForward(x,2,2)
    x,c2 = ConvBlockForwardTrain(x,params,"stage2",4)
    x,poolCache2 = NN.MaxPool2dForward(x,2,2)
    x,c3 = ConvBlockForwardTrain(x,params,"stage3",6)
    cache = dict(c1=c1,c2=c2,c3=c3,poolCache1=poolCache1,poolCache2=poolCache2)
    return x,cache


def BackboneBackwardTrain(dx,cache):
    dx,g3 = ConvBlockBackwardTrain(dx,cache["c3"],"stage3",6)
    dx = NN.MaxPool2dBackward(dx,cache["poolCache2"])
    dx,g2 = ConvBlockBackwardTrain(dx,cache["c2"],"stage2",4)
    dx = NN.MaxPool2dBackward(dx,cache["poolCache1"])
    dx,g1 = ConvBlockBackwardTrain(dx,cache["c1"],"stage1",2)
    grads = {}
    grads.update(g1); grads.update(g2); grads.update(g3)
    return dx,grads


#eval-mode backbone (running-stat batchnorm), reusing Classify.py's own
#tested primitives directly -- shared by both models' eval-mode forward below
def EvalBackboneForward(params,x):
    sd = params
    x = CLS.ConvBlockForward(x,sd,"stage1",2)
    x = CLS.MaxPool2d(x,2,2)
    x = CLS.ConvBlockForward(x,sd,"stage2",4)
    x = CLS.MaxPool2d(x,2,2)
    x = CLS.ConvBlockForward(x,sd,"stage3",6)
    return x


#=== manual batching, replacing torch.utils.data.DataLoader. shared by both ===

def IterateBatches(dataset,batchSize,rng,shuffle):
    n = len(dataset)
    order = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0,n,batchSize):
        idxs = order[start:start+batchSize]
        yield [dataset[int(i)] for i in idxs]


def SaveCheckpoint(path,params,extra=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    payload = dict(params)
    if extra:
        payload.update(extra)
    np.savez(str(path),**payload)


#######################################################################
# text model (PaperCRNN): backbone -> bidirectional lstm -> ctc head
#######################################################################

#builds every PaperCRNN parameter, keyed exactly like PaperCRNN's own
#state_dict (stage1/2/3, sequence.*_l{layer}[_reverse], text_head)
def InitTextModelParams(seed,numClasses):
    rng = np.random.default_rng(seed)
    params = {}
    InitBackboneParams(rng,params)

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


#x:(B,1,H,W) -> rawLogits:(T,B,NumClasses) where T=W/4 (two 2x2 maxpools).
#NOT log-softmaxed on purpose -- NumpyNN.CtcLoss does log_softmax internally
#and returns the gradient straight w.r.t. these raw logits
def TextModelForward(x,params):
    backboneOut,backboneCache = BackboneForwardTrain(x,params)
    pooled,hpCache = NN.AdaptiveMaxPoolHeightForward(backboneOut)   #(B,128,1,W)
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
    cache = dict(backboneCache=backboneCache,hpCache=hpCache,lstmCaches=lstmCaches,linCache=linCache,bchw=(B,C,W))
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
    dBackboneOut = NN.AdaptiveMaxPoolHeightBackward(dPooled,cache["hpCache"])
    dx,backboneGrads = BackboneBackwardTrain(dBackboneOut,cache["backboneCache"])
    grads.update(backboneGrads)
    return dx,grads


def BuildLstmLayerParams(sd):
    layerParams = []
    for l in range(2):
        fwd = dict(weight_ih=sd[f"sequence.weight_ih_l{l}"],weight_hh=sd[f"sequence.weight_hh_l{l}"],
                   bias_ih=sd[f"sequence.bias_ih_l{l}"],bias_hh=sd[f"sequence.bias_hh_l{l}"])
        bwd = dict(weight_ih=sd[f"sequence.weight_ih_l{l}_reverse"],weight_hh=sd[f"sequence.weight_hh_l{l}_reverse"],
                   bias_ih=sd[f"sequence.bias_ih_l{l}_reverse"],bias_hh=sd[f"sequence.bias_hh_l{l}_reverse"])
        layerParams.append({"fwd":fwd,"bwd":bwd})
    return layerParams


#eval-mode forward (running-stat batchnorm, matches real inference), stopping
#one step before log_softmax so NumpyNN.CtcLoss can be fed raw logits
def EvalTextForward(params,x):
    sd = params
    x = EvalBackboneForward(sd,x)
    x = CLS.AdaptiveMaxPoolHeight(x)
    x = x[:,:,0,:]
    x = np.transpose(x,(2,0,1))
    seq = CLS.MultilayerBidirectionalLstm(x,BuildLstmLayerParams(sd),HIDDEN_SIZE)
    return CLS.Linear(seq,sd["text_head.weight"],sd["text_head.bias"])


#TextCollateFn's flat targets array needs splitting back into one list per
#sample, since NumpyNN.CtcLoss wants a per-sample target sequence, not the
#single flattened array the collate step produces
def TextNumpyBatch(batch):
    images,flatTargets,targetLengths,texts = TextCollateFn(batch)
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
        images,targets,texts = TextNumpyBatch(batch)
        rawLogits = EvalTextForward(params,images)
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


#training loop: RMSprop lr=1e-3 wd=1e-5, ReduceLROnPlateau on val loss --
#same recipe as TrainText.py's Train(), minus warm restarts/resume/HF/joint/
#personal variants (simplified single path, see module comment)
def TrainTextModel(numEpochs,batchSize,lr,seed,maxPages,dataDir):
    dataset = IAMLineDatasetRaw(rootDir=dataDir,cacheDir=DEFAULT_TEXT_CACHE_DIR,maxPages=maxPages)
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
            images,targets,_texts = TextNumpyBatch(batch)
            rawLogits,cache = TextModelForward(images,params)
            loss,dRawLogits = NN.CtcLoss(rawLogits,targets)
            _,grads = TextModelBackward(dRawLogits,cache)
            trainableGrads = {k:grads[k] for k in trainable}
            optimizer.Step(trainableGrads)
            epochLoss+=loss
            nBatches+=1
            if nBatches==1 or nBatches%5==0:
                print(f"\r[TrainText] Epoch {epoch:03d} | Batch {nBatches:04d} | "
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

        print(f"[TrainText] Epoch {epoch:03d}/{numEpochs} | Train loss {trainLoss:.3f} | "
              f"Val loss {valLoss:.3f} (best {bestValLoss:.3f}) | Val char-acc {valCharAcc:.2%} | "
              f"lr {optimizer.lr:.6f} | {(time.time()-epochStart)/60:.1f} min{savedMsg}")

    return bestValLoss


#cheap sanity check: synthetic data, no dataset needed -- confirms the model
#builds, forward produces the right shape, and a full training step
#(forward+backward+optimizer.Step) runs without crashing. not a convergence test
def TextSelfTest():
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


#######################################################################
# author model (AuthorClassifierCNN): backbone -> pooling -> classify head
#######################################################################

#builds every AuthorClassifierCNN parameter, keyed exactly like its own
#state_dict (stage1/2/3 backbone, author_head.0 + author_head.3 -- indices
#1/2 of the Sequential are relu/dropout, no params)
def InitAuthorModelParams(seed,numAuthors):
    rng = np.random.default_rng(seed)
    params = {}
    InitBackboneParams(rng,params)
    w0,b0 = InitLinear(rng,96,128)
    params["author_head.0.weight"] = w0
    params["author_head.0.bias"] = b0
    w3,b3 = InitLinear(rng,numAuthors,96)
    params["author_head.3.weight"] = w3
    params["author_head.3.bias"] = b3
    return params


#x:(B,1,H,W) -> logits:(B,NumAuthors)
def AuthorModelForward(x,params,training,rng):
    backboneOut,backboneCache = BackboneForwardTrain(x,params)
    pooledH,hpCache = NN.AdaptiveMaxPoolHeightForward(backboneOut)   #(B,128,1,W)
    squeezed = pooledH[:,:,0,:]                               #(B,128,W)
    pooledW,apCache = NN.AdaptiveAvgPoolWidthForward(squeezed)  #(B,128,1), width collapsed too (unlike the text model)
    pooled = pooledW[:,:,0]                                    #(B,128)

    h,lin1Cache = NN.LinearForward(pooled,params["author_head.0.weight"],params["author_head.0.bias"])
    h,reluCache = NN.ReluForward(h)
    h,dropCache = NN.DropoutForward(h,0.2,training,rng)
    logits,lin2Cache = NN.LinearForward(h,params["author_head.3.weight"],params["author_head.3.bias"])

    cache = dict(backboneCache=backboneCache,hpCache=hpCache,apCache=apCache,
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
    dBackboneOut = NN.AdaptiveMaxPoolHeightBackward(dPooledH,cache["hpCache"])
    dx,backboneGrads = BackboneBackwardTrain(dBackboneOut,cache["backboneCache"])
    grads.update(backboneGrads)
    return dx,grads


#eval-mode forward (running-stat batchnorm, no dropout), same as
#AuthorClassifierCNNNumpy.forward in Classify.py
def EvalAuthorForward(params,x):
    sd = params
    x = EvalBackboneForward(sd,x)
    x = CLS.AdaptiveMaxPoolHeight(x)
    x = x[:,:,0,:]
    pooled = CLS.AdaptiveAvgPoolWidth(x)[:,:,0]
    h = CLS.Relu(CLS.Linear(pooled,sd["author_head.0.weight"],sd["author_head.0.bias"]))
    return CLS.Linear(h,sd["author_head.3.weight"],sd["author_head.3.bias"])


def AuthorNumpyBatch(batch):
    return AuthorCollateFn(batch)


def ValidateAuthorModel(valSet,params,batchSize,rng):
    if len(valSet)==0:
        return 0.0
    correct,total = 0,0
    for batch in IterateBatches(valSet,batchSize,rng,shuffle=False):
        images,authorIdxs,_ = AuthorNumpyBatch(batch)
        logits = EvalAuthorForward(params,images)
        preds = np.argmax(logits,axis=1)
        correct+=int((preds==authorIdxs).sum())
        total+=len(authorIdxs)
    return correct/max(1,total)


#training loop: AdamW lr=1e-3 wd=1e-4, CosineAnnealingLR, grad-clip at norm
#5.0 -- same recipe as TrainAuthor.py's Main10Authors(), minus the text-model
#backbone warm start (no from-scratch text checkpoint to warm-start from yet)
#and the original/shape subcommands (simplified single path, see module comment)
def TrainAuthorModel(numEpochs,batchSize,lr,seed):
    base = IAMLineDatasetRaw(rootDir=str(DEFAULT_AUTHOR_DATA_DIR),cacheDir=str(DEFAULT_AUTHOR_CACHE_DIR))
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
            images,authorIdxs,_ = AuthorNumpyBatch(batch)
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
            SaveCheckpoint(outPath,params,extra={"__author_names__":np.array(base.author_folders)})
            savedMsg = " [SAVED]"

        print(f"[TrainAuthor] Epoch {epoch:03d}/{numEpochs} | loss {trainLoss:.3f} | val author-acc {valAcc:.3f} "
              f"(best {bestAcc:.3f}) | lr {optimizer.lr:.6f} | {(time.time()-epochStart)/60:.1f} min{savedMsg}")

    print(f"\nBest val author-accuracy: {bestAcc:.3f}")
    print(f"Weights saved to: {outPath}")
    return bestAcc


#cheap sanity check: synthetic data, no dataset needed -- confirms the model
#builds, forward produces the right shape, and a full training step
#(forward+backward+clip+optimizer.Step) runs without crashing
def AuthorSelfTest():
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


#######################################################################
# cli: python TrainScratch.py {text,author} [options]
#######################################################################

def Main():
    parser = argparse.ArgumentParser(description="Train PaperCRNN or AuthorClassifierCNN from scratch (NumpyNN only, no torch).")
    sub = parser.add_subparsers(dest="model",required=True)

    textP = sub.add_parser("text")
    textP.add_argument("--epochs",type=int,default=50)
    textP.add_argument("--batch-size",type=int,default=16)
    textP.add_argument("--lr",type=float,default=1e-3)
    textP.add_argument("--seed",type=int,default=7)
    textP.add_argument("--max-pages",type=int,default=None)
    textP.add_argument("--data-dir",type=str,default=None)
    textP.add_argument("--selftest",action="store_true",help="run the synthetic smoke test and exit")

    authorP = sub.add_parser("author")
    authorP.add_argument("--epochs",type=int,default=20)
    authorP.add_argument("--batch-size",type=int,default=16)
    authorP.add_argument("--lr",type=float,default=1e-3)
    authorP.add_argument("--seed",type=int,default=0)
    authorP.add_argument("--selftest",action="store_true",help="run the synthetic smoke test and exit")

    args = parser.parse_args()

    if args.model=="text":
        if args.selftest:
            TextSelfTest()
            return
        dataDir = Path(args.data_dir) if args.data_dir else DEFAULT_TEXT_DATA_DIR
        print(f"\n--- Training paper_cnn_bilstm_ctc_scratch (lr={args.lr}, batch_size={args.batch_size}, "
              f"max_epochs={args.epochs}, data={dataDir}) ---")
        best = TrainTextModel(args.epochs,args.batch_size,args.lr,args.seed,args.max_pages,dataDir)
        print(f"\nDone. Best validation loss: {best:.4f}")
    else:
        if args.selftest:
            AuthorSelfTest()
            return
        print(f"\n--- Training author_classifier_10new_scratch (lr={args.lr}, batch_size={args.batch_size}, "
              f"max_epochs={args.epochs}) ---")
        TrainAuthorModel(args.epochs,args.batch_size,args.lr,args.seed)


if __name__=="__main__":
    Main()
