#u23539195 Dylan Braune.  reading a line of handwriting, and saying who wrote
#it.  FU 3, FU 4, R3, R4
#
#two trained networks live here, both run forward only, in numpy:
#  the reader   convolutions, then a bidirectional lstm, then CTC decoding
#  the namer    a small convolution net that scores a crop for every writer
#
#training lives in train.py, the image toolkit in segment.py
import numpy as np
from pathlib import Path

import paths
from segment import (BinarizeLine,CoreBand,Dilate,FitFixedCanvas,Skeletonize)


#======================================================================
#WHAT THE READER CAN SPELL
#the alphabet, and the two lookups between a letter and its slot number
#======================================================================
#---CHARSET: every character the reader can output, slot 0 is the CTC blank---
CHARSET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:'\"!?()-"

#---CHAR_TO_IDX: letter to slot number, for making labels---
CHAR_TO_IDX = {c:i+1 for i,c in enumerate(CHARSET)}

#---IDX_TO_CHAR: slot number back to letter, for reading output---
IDX_TO_CHAR = {i:c for c,i in CHAR_TO_IDX.items()}



#======================================================================
#WHERE THE TRAINED WEIGHTS LIVE
#file names only, no loading yet, so a caller can point somewhere else
#======================================================================
#---WEIGHTS_DIR: the folder holding every trained .npz---
WEIGHTS_DIR = paths.Weights

#---INPUT_HEIGHT: the exact crop height both networks were trained on---
INPUT_HEIGHT = 64   #the fixed canvas both models were trained on

#---INPUT_WIDTH: and the width, crops are fitted to this---
INPUT_WIDTH = 640

#----------------------------------------------------------------------
#_TEXT_CANDIDATES: reader checkpoints, best first
#the first one that exists on disk is the one that gets used
#----------------------------------------------------------------------
_TEXT_CANDIDATES = ["paper_cnn_bilstm_ctc_scratch_best.npz","paper_cnn_bilstm_ctc_joint_best.npz",
                    "paper_cnn_bilstm_ctc_hf_best.npz"]

#---DEFAULT_TEXT_WEIGHTS: the reader checkpoint picked from that list---
DEFAULT_TEXT_WEIGHTS = next((WEIGHTS_DIR/n for n in _TEXT_CANDIDATES if (WEIGHTS_DIR/n).exists()),
                            WEIGHTS_DIR/_TEXT_CANDIDATES[0])

#---_AUTHOR_NPZ: the writer naming checkpoints to look for---
_AUTHOR_NPZ = WEIGHTS_DIR/"author_classifier_10new_scratch_weights.npz"

#---_SHAPE: shape based weights, trained on stroke normalised crops---
_SHAPE = WEIGHTS_DIR/"author_shape_10new_weights.npz"

#---_W1: ink based weights, trained on raw grey crops---
_W1 = WEIGHTS_DIR/"author_classifier_10new_weights.npz"

#---DEFAULT_AUTHOR_WEIGHTS: whichever writer checkpoint was found---
DEFAULT_AUTHOR_WEIGHTS = _AUTHOR_NPZ if _AUTHOR_NPZ.exists() else (_SHAPE if _SHAPE.exists() else _W1)



#======================================================================
#LOADING A CHECKPOINT
#one loader, because the files were saved by several different scripts
#======================================================================
#----------------------------------------------------------------------
#LoadStateDictAny: open a saved checkpoint whatever its layout
#the npz files came from different training runs with different key
#names, so this tries each known layout and returns a plain dict of
#arrays.  keeps the rest of the file from caring
#----------------------------------------------------------------------
def LoadStateDictAny(checkpointPath):
    with np.load(checkpointPath,allow_pickle=False) as npz:
        sd = {k: npz[k] for k in npz.files if not k.startswith("__")}
        authorMapping = None
        if "__author_names__" in npz.files:
            names = [str(n) for n in npz["__author_names__"]]
            authorMapping = {name: idx for idx,name in enumerate(names)}
    return sd,authorMapping



#======================================================================
#THE PIECES A NETWORK IS BUILT FROM, FORWARD ONLY
#each one takes arrays in and gives arrays out.  no classes, no framework
#======================================================================
#----------------------------------------------------------------------
#Conv2d: slide learned filters over the image
#turns each patch into a column and multiplies once, which is the same
#answer as sliding a window but far quicker.  this is what finds strokes,
#corners and loops
#----------------------------------------------------------------------
def Conv2d(x,weight,bias,stride=1,padding=1):
    B,C,H,W = x.shape
    OutC,InC,kh,kw = weight.shape
    xp = np.pad(x,((0,0),(0,0),(padding,padding),(padding,padding)))
    Hp,Wp = xp.shape[2],xp.shape[3]
    outH = (Hp-kh)//stride+1
    outW = (Wp-kw)//stride+1

    s = xp.strides   #im2col via as_strided: (B,C,kh,kw,outH,outW)
    shape = (B,C,kh,kw,outH,outW)
    strides = (s[0],s[1],s[2],s[3],s[2]*stride,s[3]*stride)
    cols = np.lib.stride_tricks.as_strided(xp,shape=shape,strides=strides)
    cols = cols.reshape(B,C*kh*kw,outH*outW)

    w = weight.reshape(OutC,C*kh*kw)
    #plain matmul, which numpy hands to blas
    out = np.stack([w@cols[b] for b in range(B)],axis=0).reshape(B,OutC,outH,outW)
    if bias is not None:
        out = out+bias.reshape(1,OutC,1,1)
    return out

#----------------------------------------------------------------------
#BatchNorm2d: put each channel back on a standard scale
#subtracts the mean and divides by the spread that was recorded during
#training, then applies the learned scale and shift.  keeps the numbers
#in a range the next layer was trained for
#----------------------------------------------------------------------
def BatchNorm2d(x,weight,bias,runningMean,runningVar,eps=1e-5):
    scale = weight/np.sqrt(runningVar+eps)
    shift = bias-runningMean*scale
    return x*scale.reshape(1,-1,1,1)+shift.reshape(1,-1,1,1)   #one scale and shift per CHANNEL, broadcast over B,H,W

#---Relu: keep positives, flatten negatives to zero---
def Relu(x):
    return np.maximum(x,0)   #negatives to zero, that is the whole layer

#---Sigmoid: squash any number into 0 to 1, used by the lstm gates---
def Sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

#---MaxPool2d: keep the strongest value in each small window, shrinking the image---
def MaxPool2d(x,kernel=2,stride=2):
    B,C,H,W = x.shape
    outH,outW = H//stride,W//stride
    xc = x[:,:,:outH*stride,:outW*stride]   #drop a leftover odd row/column
    st = xc.strides
    windows = np.lib.stride_tricks.as_strided(
        xc,shape=(B,C,outH,outW,kernel,kernel),
        strides=(st[0],st[1],st[2]*stride,st[3]*stride,st[2],st[3]))
    return windows.max(axis=(4,5))

#---AdaptiveMaxPoolHeight: squash the height to one row, keeping the strongest---
def AdaptiveMaxPoolHeight(x):
    return x.max(axis=2,keepdims=True)

#---AdaptiveAvgPoolWidth: average the width down to a fixed number of columns---
def AdaptiveAvgPoolWidth(x):
    return x.mean(axis=2,keepdims=True)

#---Linear: the plain weights times input plus bias layer---
def Linear(x,weight,bias):
    out = x@weight.T
    return out if bias is None else out+bias

#----------------------------------------------------------------------
#LogSoftmax: turn scores into log probabilities
#subtracts the largest score first so nothing overflows, which matters
#because CTC adds thousands of these together
#----------------------------------------------------------------------
def LogSoftmax(x,axis):
    m = np.max(x,axis=axis,keepdims=True)
    shifted = x-m
    lse = np.log(np.sum(np.exp(shifted),axis=axis,keepdims=True))   #the max is already subtracted, so exp cannot overflow
    return shifted-lse

#---Softmax: turn scores into probabilities that add to one---
def Softmax(x,axis):
    m = np.max(x,axis=axis,keepdims=True)
    e = np.exp(x-m)
    return e/np.sum(e,axis=axis,keepdims=True)

#----------------------------------------------------------------------
#ConvBlockForward: the repeated convolution, normalise, relu
#the backbone is this block several times over, so it is written once
#----------------------------------------------------------------------
def ConvBlockForward(x,sd,prefix,nLayers):
    for i in range(nLayers):
        c = i*3
        x = Conv2d(x,sd[f"{prefix}.{c}.weight"],sd[f"{prefix}.{c}.bias"],padding=1)
        x = BatchNorm2d(x,sd[f"{prefix}.{c+1}.weight"],sd[f"{prefix}.{c+1}.bias"],
                         sd[f"{prefix}.{c+1}.running_mean"],sd[f"{prefix}.{c+1}.running_var"])
        x = Relu(x)
    return x



#======================================================================
#THE MEMORY PART OF THE READER
#an lstm reads the line left to right, carrying what it has seen so far
#======================================================================
#----------------------------------------------------------------------
#LstmForward: read a sequence one step at a time, one direction
#at each column four gates decide what to forget, what to take in and
#what to pass on.  that memory is what lets the reader use the letters
#around a smudge to work out what the smudge was
#----------------------------------------------------------------------
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

#----------------------------------------------------------------------
#BidirectionalLstmForward: read it forwards and backwards
#runs the same thing right to left as well and sticks the two together,
#so every column knows about the letters on both sides of it
#----------------------------------------------------------------------
def BidirectionalLstmForward(x,paramsFwd,paramsBwd,hiddenSize):
    outF,_ = LstmForward(x,**paramsFwd,hiddenSize=hiddenSize)
    outB,_ = LstmForward(x[::-1],**paramsBwd,hiddenSize=hiddenSize)
    outB = outB[::-1]
    return np.concatenate([outF,outB],axis=2)

#---MultilayerBidirectionalLstm: stack those, each layer reading the one below---
def MultilayerBidirectionalLstm(x,layerParams,hiddenSize):
    out = x
    for layer in layerParams:
        out = BidirectionalLstmForward(out,layer["fwd"],layer["bwd"],hiddenSize)
    return out



#======================================================================
#GETTING A CROP READY FOR THE NETWORK
#the networks only accept one exact size, so every crop is fitted to it
#======================================================================
#----------------------------------------------------------------------
#ResizeLineImageFixed: scale a line crop to the trained size
#one scale factor for both axes so letters keep their proportions,
#then pad with paper to fill the canvas
#----------------------------------------------------------------------
def ResizeLineImageFixed(img,height=INPUT_HEIGHT,width=INPUT_WIDTH,padValue=255):
    return FitFixedCanvas(np.asarray(img,dtype=np.uint8),height,width,padValue)

#---TensorFromResized: grey image to the float array shape the network wants---
def TensorFromResized(img):
    arr = np.array(img,dtype=np.float32)/255.0
    arr = 1.0-arr
    arr = (arr-0.5)/0.5
    return arr[np.newaxis,np.newaxis,:,:]

#---PenWidthXh: the pen width stroke normalising redraws at, in x-heights---
PenWidthXh = 0.14   #pen width the stroke normaliser redraws at, in x-heights

#----------------------------------------------------------------------
#StrokeNormalize: redraw the writing at one pen width
#thins the ink to its centre line then draws it again at a fixed
#width.  that throws away how hard the pen pressed while keeping the
#shape, so the shape based namer cannot cheat by recognising ink
#----------------------------------------------------------------------
def StrokeNormalize(gray,penWidthXh=PenWidthXh):
    ink = BinarizeLine(gray)
    if not ink.any():
        return None
    band = CoreBand(ink)
    xh = float(band[1]-band[0]) if band else 0.0
    if xh<4:
        xh = max(8.0,ink.shape[0]/3.0)   #no usable band, fall back to a third of the crop
    sk = Skeletonize(ink)
    if not sk.any():
        return None
    w = int(np.clip(round(penWidthXh*xh),2,9))   #width is a fraction of x-height, so resolution free
    k = 2*(w//2)+1   #force odd so the pen sits centred on the centreline
    return ((~Dilate(sk,k,k))*255).astype(np.uint8)   #re-ink, then invert to ink-on-white



#======================================================================
#TURNING NETWORK OUTPUT BACK INTO WORDS
#the reader emits a score per character per column, which has to be decoded
#======================================================================
#----------------------------------------------------------------------
#DecodeCtc: column scores to a string
#takes the best character in each column, then collapses repeats and
#drops the blanks.  the blank is what lets the network say 'still the
#same letter' versus 'a new letter', so 'bb' survives but 'b b' does not
#----------------------------------------------------------------------
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
#CtcForcedAlign: given the text, find WHERE each letter is
#the text is already known, so instead of decoding it walks the one path
#through the scores that spells it, keeping the best running score at
#each step.  that gives the column span of every character, which is
#how the glyph extractor knows where to cut
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
        for s in range(S):   #skip only onto a non blank that differs two back, standard ctc topology
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
    conf = float(np.mean([logProbs[t,ext[path[t]]] for t in range(T)]))   #mean per frame log prob along the path
    return list(zip(kept,spans)),conf

#----------------------------------------------------------------------
#FrameXToPixel: a column in network output back to a pixel in the crop
#undoes the resize and the padding that got the crop onto the canvas,
#so an alignment span can be turned into a cut position
#----------------------------------------------------------------------
def FrameXToPixel(canvasX,rawW,origH):
    s = min(INPUT_HEIGHT/origH,INPUT_WIDTH/rawW)
    return float(np.clip(canvasX/s,0.0,float(rawW)))



#======================================================================
#THE READER
#the whole text network, and how its answer is scored
#======================================================================
#----------------------------------------------------------------------
#PaperCRNNNumpy: the reader, loaded and ready to run
#holds the weights and runs the layers in order: convolutions to find
#stroke shapes, then the bidirectional lstm to read along the line,
#then a score per character per column
#----------------------------------------------------------------------
class PaperCRNNNumpy:
    def __init__(self,checkpointPath=None,hiddenSize=256):
        self.checkpointPath = Path(checkpointPath or DEFAULT_TEXT_WEIGHTS)
        self.sd,_ = LoadStateDictAny(self.checkpointPath)
        self.hiddenSize = hiddenSize

        self.lstmLayerParams = []
        for l in range(2):
            def get(name,layer=l):
                return self.sd[f"sequence.{name}_l{layer}"]

            fwd = dict(weightIh=get("weight_ih"),weightHh=get("weight_hh"),
                       biasIh=get("bias_ih"),biasHh=get("bias_hh"))
            bwd = dict(weightIh=self.sd[f"sequence.weight_ih_l{l}_reverse"],
                       weightHh=self.sd[f"sequence.weight_hh_l{l}_reverse"],
                       biasIh=self.sd[f"sequence.bias_ih_l{l}_reverse"],
                       biasHh=self.sd[f"sequence.bias_hh_l{l}_reverse"])
            self.lstmLayerParams.append({"fwd":fwd,"bwd":bwd})

    #x: (B,1,H,W) numpy array -> (T,B,numClasses) log-probs
    def forward(self,x):
        sd = self.sd
        x = ConvBlockForward(x,sd,"stage1",2)
        x = MaxPool2d(x,2,2)
        x = ConvBlockForward(x,sd,"stage2",4)
        x = MaxPool2d(x,2,2)
        x = ConvBlockForward(x,sd,"stage3",6)
        x = AdaptiveMaxPoolHeight(x)   #(B,128,1,W)
        x = x[:,:,0,:]   #(B,128,W)
        x = np.transpose(x,(2,0,1))   #(W,B,128), width becomes the ctc time axis
        seq = MultilayerBidirectionalLstm(x,self.lstmLayerParams,self.hiddenSize)
        logits = Linear(seq,sd["text_head.weight"],sd["text_head.bias"])
        return LogSoftmax(logits,axis=2)

#---ReadText: crop in, string out, the whole reader in one call---
def ReadText(image,model):
    t = TensorFromResized(ResizeLineImageFixed(image))
    logProbs = model.forward(t)
    return DecodeCtc(logProbs)[0]

#----------------------------------------------------------------------
#Levenshtein: how many edits turn one string into another
#fills a small table where each cell is the cheapest way to match the
#first few letters of each string, counting an insert, a delete or a
#swap as one
#----------------------------------------------------------------------
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

#---CharAcc: fraction of characters right, from the edit distance---
def CharAcc(pred,truth):
    if not truth:
        return 1.0 if not pred else 0.0
    return max(0.0,1.0-Levenshtein(pred,truth)/len(truth))

#---WordAcc: fraction of whole words that came back exactly right---
def WordAcc(pred,truth):
    tw,pw = truth.split(),pred.split()
    if not tw:
        return 1.0
    m,n = len(tw),len(pw)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            dp[i+1][j+1] = (dp[i][j]+1 if tw[i].lower()==pw[j].lower()
                            else max(dp[i][j+1],dp[i+1][j]))
    return dp[m][n]/m



#======================================================================
#THE WRITER NAMER
#a smaller network that answers 'whose handwriting is this'
#======================================================================
#----------------------------------------------------------------------
#AuthorClassifierCNNNumpy: the writer naming network
#same style of backbone as the reader but with no memory part, and it
#ends in one score per writer instead of per character.  it also carries
#the mapping from writer name to output slot that it was trained with
#----------------------------------------------------------------------
class AuthorClassifierCNNNumpy:
    def __init__(self,checkpointPath=None):
        self.checkpointPath = Path(checkpointPath or DEFAULT_AUTHOR_WEIGHTS)
        self.sd,self.authorMapping = LoadStateDictAny(self.checkpointPath)

    #x: (B,1,H,W) numpy array -> (B,numAuthors) raw logits
    def forward(self,x):
        sd = self.sd
        x = ConvBlockForward(x,sd,"stage1",2)
        x = MaxPool2d(x,2,2)
        x = ConvBlockForward(x,sd,"stage2",4)
        x = MaxPool2d(x,2,2)
        x = ConvBlockForward(x,sd,"stage3",6)
        x = AdaptiveMaxPoolHeight(x)   #(B,128,1,W)
        x = x[:,:,0,:]   #(B,128,W)
        pooled = AdaptiveAvgPoolWidth(x)[:,:,0]   #(B,128)
        h = Linear(pooled,sd["author_head.0.weight"],sd["author_head.0.bias"])
        h = Relu(h)
        #authorHead.2 is dropout, inactive at inference, skipped
        logits = Linear(h,sd["author_head.3.weight"],sd["author_head.3.bias"])
        return logits

#----------------------------------------------------------------------
#ClassifyImage: crop in, writer and confidences out
#optionally stroke normalises first, which is what makes the shape based
#model judge letter shapes instead of ink darkness
#----------------------------------------------------------------------
def ClassifyImage(image,model,authorMapping=None,strokeNormalise=True):
    if strokeNormalise:
        #redraw at one constant pen width first, see StrokeNormalize
        normImg = StrokeNormalize(np.asarray(image,dtype=np.uint8))
        if normImg is not None:
            image = normImg
    t = TensorFromResized(ResizeLineImageFixed(image))
    logits = model.forward(t)
    probs = Softmax(logits,axis=1)[0]
    predIdx = int(np.argmax(probs))
    return predIdx,probs
