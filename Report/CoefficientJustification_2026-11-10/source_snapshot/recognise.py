#u23539195 D.R. Braune.  reading a line of handwriting, and saying who wrote it
#two trained networks live here, both run forward only
#  the reader   convolutions, then a bidirectional lstm, then ctc decoding
#  the namer    a small convolution net that scores a crop for every writer
import numpy as np
from pathlib import Path

import paths
from segment import (BinarizeLine,CoreBand,Dilate,FitFixedCanvas,Skeletonize) #my own first principles segment lines

#every character the reader can emit, slot 0 is the CTC blank
CHARSET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:'\"!?()-"

#character to slot number, and back
CHAR_TO_IDX = {c:i+1 for i,c in enumerate(CHARSET)}
IDX_TO_CHAR = {i:c for c,i in CHAR_TO_IDX.items()}

WEIGHTS_DIR = paths.Weights

#the crop size both networks take
INPUT_HEIGHT = 64
INPUT_WIDTH = 640

#the reader, written by train.py
DEFAULT_TEXT_WEIGHTS = WEIGHTS_DIR/"paper_cnn_bilstm_ctc_scratch_best.npz"

#the writer namer, written by train.py
DEFAULT_AUTHOR_WEIGHTS = WEIGHTS_DIR/"author_shape_11w_pw170_scratch_weights.npz"

#---LoadStateDictAny: reads an .npz checkpoint into a dict of arrays, with its writer names---
def LoadStateDictAny(checkpointPath):
    with np.load(checkpointPath,allow_pickle=False) as npz:
        sd = {k: npz[k] for k in npz.files if not k.startswith("__")}
        authorMapping = None
        if "__author_names__" in npz.files:
            names = [str(n) for n in npz["__author_names__"]]
            authorMapping = {name: idx for idx,name in enumerate(names)}
    return sd,authorMapping

#---Conv2d: slides learned filters over the image by turning each patch into a column---
def Conv2d(x,weight,bias,stride=1,padding=1):
    B,C,H,W = x.shape
    OutC,InC,kh,kw = weight.shape
    xp = np.pad(x,((0,0),(0,0),(padding,padding),(padding,padding)))
    Hp,Wp = xp.shape[2],xp.shape[3]
    outH = (Hp-kh)//stride+1
    outW = (Wp-kw)//stride+1

    s = xp.strides   #a strided view holding every patch, (B, C, kh, kw, outH, outW)
    shape = (B,C,kh,kw,outH,outW)
    strides = (s[0],s[1],s[2],s[3],s[2]*stride,s[3]*stride)
    cols = np.lib.stride_tricks.as_strided(xp,shape=shape,strides=strides)
    cols = cols.reshape(B,C*kh*kw,outH*outW)

    w = weight.reshape(OutC,C*kh*kw)
    out = np.stack([w@cols[b] for b in range(B)],axis=0).reshape(B,OutC,outH,outW)
    if bias is not None:
        out = out+bias.reshape(1,OutC,1,1)
    return out

#BatchNorm2d: rescales each channel by the mean and variance recorded during training---
def BatchNorm2d(x,weight,bias,runningMean,runningVar,eps=1e-5):
    scale = weight/np.sqrt(runningVar+eps)
    shift = bias-runningMean*scale
    return x*scale.reshape(1,-1,1,1)+shift.reshape(1,-1,1,1)

#-Relu: keeps positives, sets negatives to zero
def Relu(x):
    return np.maximum(x,0)

#Sigmoid: squashes any number into 0 to 1
def Sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

#---MaxPool2d: keeps the largest value in each window, halving height and width---
def MaxPool2d(x,kernel=2,stride=2):
    B,C,H,W = x.shape
    outH,outW = H//stride,W//stride
    xc = x[:,:,:outH*stride,:outW*stride]   #drop a leftover odd row or column
    st = xc.strides
    windows = np.lib.stride_tricks.as_strided(
        xc,shape=(B,C,outH,outW,kernel,kernel),
        strides=(st[0],st[1],st[2]*stride,st[3]*stride,st[2],st[3]))
    return windows.max(axis=(4,5))

#---AdaptiveMaxPoolHeight: collapses the height to one row, keeping the largest value per column---
def AdaptiveMaxPoolHeight(x):
    return x.max(axis=2,keepdims=True)

#---AdaptiveAvgPoolWidth: averages along the width axis---
def AdaptiveAvgPoolWidth(x):
    return x.mean(axis=2,keepdims=True)

#---Linear: weights times input, plus bias---
def Linear(x,weight,bias):
    out = x@weight.T
    return out if bias is None else out+bias

#---LogSoftmax: scores to log probabilities, with the maximum subtracted first---
def LogSoftmax(x,axis):
    m = np.max(x,axis=axis,keepdims=True)
    shifted = x-m
    lse = np.log(np.sum(np.exp(shifted),axis=axis,keepdims=True))
    return shifted-lse

#---Softmax: scores to probabilities that sum to one---
def Softmax(x,axis):
    m = np.max(x,axis=axis,keepdims=True)
    e = np.exp(x-m)
    return e/np.sum(e,axis=axis,keepdims=True)

#---ConvBlockForward: the repeated convolution, batch-norm, relu block of the backbone---
def ConvBlockForward(x,sd,prefix,nLayers):
    for i in range(nLayers):
        c = i*3
        x = Conv2d(x,sd[f"{prefix}.{c}.weight"],sd[f"{prefix}.{c}.bias"],padding=1)
        x = BatchNorm2d(x,sd[f"{prefix}.{c+1}.weight"],sd[f"{prefix}.{c+1}.bias"],
                         sd[f"{prefix}.{c+1}.running_mean"],sd[f"{prefix}.{c+1}.running_var"])
        x = Relu(x)
    return x

#---LstmForward: reads a sequence one column at a time through four gates, in one direction---
def LstmForward(x,weightIh,weightHh,biasIh,biasHh,hiddenSize):
    T,B,_ = x.shape
    h = np.zeros((B,hiddenSize),dtype=x.dtype)
    c = np.zeros((B,hiddenSize),dtype=x.dtype)
    H = hiddenSize
    outputs = np.empty((T,B,H),dtype=x.dtype)

    for t in range(T):
        gates = x[t]@weightIh.T+biasIh+h@weightHh.T+biasHh
        i = Sigmoid(gates[:,0*H:1*H])
        f = Sigmoid(gates[:,1*H:2*H])
        g = np.tanh(gates[:,2*H:3*H])
        o = Sigmoid(gates[:,3*H:4*H])
        c = f*c+i*g
        h = o*np.tanh(c)
        outputs[t] = h

    return outputs,(h,c)

#---BidirectionalLstmForward: runs the sequence both ways and concatenates the two hidden states---
def BidirectionalLstmForward(x,paramsFwd,paramsBwd,hiddenSize):
    outF,_ = LstmForward(x,**paramsFwd,hiddenSize=hiddenSize)
    outB,_ = LstmForward(x[::-1],**paramsBwd,hiddenSize=hiddenSize)
    outB = outB[::-1]
    return np.concatenate([outF,outB],axis=2)

#---MultilayerBidirectionalLstm: stacks those layers, each reading the output of the one below---
def MultilayerBidirectionalLstm(x,layerParams,hiddenSize):
    out = x
    for layer in layerParams:
        out = BidirectionalLstmForward(out,layer["fwd"],layer["bwd"],hiddenSize)
    return out

#---ResizeLineImageFixed: scales a crop onto the trained canvas by one factor for both axes---
def ResizeLineImageFixed(img,height=INPUT_HEIGHT,width=INPUT_WIDTH,padValue=255):
    return FitFixedCanvas(np.asarray(img,dtype=np.uint8),height,width,padValue)

#---TensorFromResized: grey image to the normalised float array the networks take---
def TensorFromResized(img):
    arr = np.array(img,dtype=np.float32)/255.0
    arr = 1.0-arr
    arr = (arr-0.5)/0.5
    return arr[np.newaxis,np.newaxis,:,:]

#the pen width stroke normalising redraws at, in x-heights
PenWidthXh = 0.17

#---StrokeNormalize: thins the ink to its centreline and redraws it at one fixed pen width---
def StrokeNormalize(gray,penWidthXh=PenWidthXh):
    ink = BinarizeLine(gray)
    if not ink.any():
        return None
    band = CoreBand(ink)
    xh = float(band[1]-band[0]) if band else 0.0
    if xh<4:
        xh = max(8.0,ink.shape[0]/3.0)   #no usable band, take a third of the crop
    sk = Skeletonize(ink)
    if not sk.any():
        return None
    w = int(np.clip(round(penWidthXh*xh),2,9))   #width as a fraction of x-height
    k = 2*(w//2)+1   #odd, so the pen sits centred on the centreline
    return ((~Dilate(sk,k,k))*255).astype(np.uint8)   #re-ink, then invert to ink on white

#---DecodeCtc: takes the best character per column, collapses repeats and drops blanks---
def DecodeCtc(logProbs):
    bestPath = np.argmax(logProbs,axis=2)
    decoded = []
    for batchIdx in range(bestPath.shape[1]):
        prev,chars = None,[]
        for token in bestPath[:,batchIdx]:
            token = int(token)
            if token!=0 and token!=prev:
                chars.append(IDX_TO_CHAR.get(token,""))
            prev = token
        decoded.append("".join(chars))
    return decoded

#----------------------------------------------------------------------
#CtcForcedAlign: walks the one path through the scores that spells a known text, and
#returns the column span of every character plus the path's mean score
#----------------------------------------------------------------------
def CtcForcedAlign(logProbs,text):
    labels = [CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX]
    if not labels:
        return None
    ext = [0]
    for l in labels:
        ext+=[l,0]
    S,T = len(ext),logProbs.shape[0]
    if T<len(labels):
        return None
    NEG = -1e30
    dp = np.full((T,S),NEG)
    bp = np.zeros((T,S),np.int32)
    dp[0,0] = logProbs[0,ext[0]]
    if S>1:
        dp[0,1] = logProbs[0,ext[1]]
    for t in range(1,T):
        emit = logProbs[t,ext]
        stay = dp[t-1]
        prev1 = np.concatenate(([NEG],dp[t-1,:-1]))
        prev2 = np.concatenate(([NEG,NEG],dp[t-1,:-2]))
        for s in range(S):   #a skip is allowed only onto a non-blank differing two back
            if s>=2 and (ext[s]==0 or ext[s]==ext[s-2]):
                prev2[s] = NEG
        cand = np.stack([stay,prev1,prev2])
        best = np.argmax(cand,axis=0)
        dp[t] = cand[best,np.arange(S)]+emit
        bp[t] = best
    endS = S-1 if S==1 else (S-1 if dp[T-1,S-1]>=dp[T-1,S-2] else S-2)
    path = np.zeros(T,np.int32)
    s = endS
    for t in range(T-1,-1,-1):
        path[t] = s
        s-=bp[t,s]
    spans = []
    for ci in range(len(labels)):
        sIdx = 1+2*ci
        ts = np.nonzero(path==sIdx)[0]
        if len(ts)==0:
            spans.append(None)
        else:
            spans.append((int(ts[0]),int(ts[-1])+1))
    kept = [c for c in text if c in CHAR_TO_IDX]
    conf = float(np.mean([logProbs[t,ext[path[t]]] for t in range(T)]))
    return list(zip(kept,spans)),conf

#---FrameXToPixel: undoes the resize and padding, mapping a canvas column back to a crop pixel---
def FrameXToPixel(canvasX,rawW,origH):
    s = min(INPUT_HEIGHT/origH,INPUT_WIDTH/rawW)
    return float(np.clip(canvasX/s,0.0,float(rawW)))

#----------------------------------------------------------------------
#PaperCRNNNumpy: the reader, the convolution backbone, the LSTM stack and a score per
#character per column
#----------------------------------------------------------------------
class PaperCRNNNumpy:
    def __init__(self,checkpointPath=None,hiddenSize=256):
        self.checkpointPath = Path(checkpointPath or DEFAULT_TEXT_WEIGHTS)
        self.sd,_ = LoadStateDictAny(self.checkpointPath)
        self.hiddenSize = hiddenSize

        self.lstmLayerParams = []
        for l in range(2):
            def Get(name,layer=l):
                return self.sd[f"sequence.{name}_l{layer}"]

            fwd = dict(weightIh=Get("weight_ih"),weightHh=Get("weight_hh"),
                       biasIh=Get("bias_ih"),biasHh=Get("bias_hh"))
            bwd = dict(weightIh=self.sd[f"sequence.weight_ih_l{l}_reverse"],
                       weightHh=self.sd[f"sequence.weight_hh_l{l}_reverse"],
                       biasIh=self.sd[f"sequence.bias_ih_l{l}_reverse"],
                       biasHh=self.sd[f"sequence.bias_hh_l{l}_reverse"])
            self.lstmLayerParams.append({"fwd":fwd,"bwd":bwd})

    #(B,1,H,W) in, (T,B,numClasses) log probabilities out
    def Forward(self,x):
        sd = self.sd
        x = ConvBlockForward(x,sd,"stage1",2)
        x = MaxPool2d(x,2,2)
        x = ConvBlockForward(x,sd,"stage2",4)
        x = MaxPool2d(x,2,2)
        x = ConvBlockForward(x,sd,"stage3",6)
        x = AdaptiveMaxPoolHeight(x)
        x = x[:,:,0,:]
        x = np.transpose(x,(2,0,1))   #width becomes the CTC time axis
        seq = MultilayerBidirectionalLstm(x,self.lstmLayerParams,self.hiddenSize)
        logits = Linear(seq,sd["text_head.weight"],sd["text_head.bias"])
        return LogSoftmax(logits,axis=2)

#---ReadTextScored: reads a crop and returns the text with the mean best-column log probability---
def ReadTextScored(image,model):
    t = TensorFromResized(ResizeLineImageFixed(image))
    logProbs = model.Forward(t)
    best = np.max(logProbs[:,0,:],axis=1)
    return DecodeCtc(logProbs)[0],float(np.mean(best))

#---ReadText: crop in, string out---
def ReadText(image,model):
    return ReadTextScored(image,model)[0]

#---Levenshtein: edit distance, counting an insert, a delete or a substitution as one---
def Levenshtein(a,b):
    if len(a)<len(b):
        return Levenshtein(b,a)
    if len(b)==0:
        return len(a)
    previous = list(range(len(b)+1))
    for ca in a:
        current = [previous[0]+1]
        for j,cb in enumerate(b):
            current.append(min(current[j]+1,previous[j+1]+1,previous[j]+(ca!=cb)))
        previous = current
    return previous[-1]

#---CharAcc: fraction of characters correct, from that edit distance---
def CharAcc(pred,truth):
    if not truth:
        return 1.0 if not pred else 0.0
    return max(0.0,1.0-Levenshtein(pred,truth)/len(truth))

#---AuthorClassifierCNNNumpy: the writer namer, the same backbone, no recurrence, one score per writer---
class AuthorClassifierCNNNumpy:
    def __init__(self,checkpointPath=None):
        self.checkpointPath = Path(checkpointPath or DEFAULT_AUTHOR_WEIGHTS)
        self.sd,self.authorMapping = LoadStateDictAny(self.checkpointPath)

    #(B,1,H,W) in, (B,numAuthors) raw scores out
    def Forward(self,x):
        sd = self.sd
        x = ConvBlockForward(x,sd,"stage1",2)
        x = MaxPool2d(x,2,2)
        x = ConvBlockForward(x,sd,"stage2",4)
        x = MaxPool2d(x,2,2)
        x = ConvBlockForward(x,sd,"stage3",6)
        x = AdaptiveMaxPoolHeight(x)
        x = x[:,:,0,:]
        pooled = AdaptiveAvgPoolWidth(x)[:,:,0]
        h = Linear(pooled,sd["author_head.0.weight"],sd["author_head.0.bias"])
        h = Relu(h)
        #author_head.2 is dropout, inactive at inference
        logits = Linear(h,sd["author_head.3.weight"],sd["author_head.3.bias"])
        return logits

# ClassifyImage: crop in, writer index and the probability over writers out
def ClassifyImage(image,model,authorMapping=None,strokeNormalise=True):
    if strokeNormalise:
        normImg = StrokeNormalize(np.asarray(image,dtype=np.uint8))
        if normImg is not None:
            image = normImg
    t = TensorFromResized(ResizeLineImageFixed(image))
    logits = model.Forward(t)
    probs = Softmax(logits,axis=1)[0]
    predIdx = int(np.argmax(probs))
    return predIdx,probs
