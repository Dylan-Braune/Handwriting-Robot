#u23539195 D.R. Braune.  reading a line of handwriting, and saying who wrote it
#two trained networks live here, both run forward only
#  the reader   convolutions, then a bidirectional lstm, then ctc decoding
#  the namer    a small convolution net that scores a crop for every writer
import numpy as np
from pathlib import Path

import paths
from segment import (BinarizeLine,CoreBand,Dilate,FitFixedCanvas,Skeletonize) #my own first principles segment lines

#every character the reader can emit, slot 0 is the CTC blank
CHARSET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:'\"!?()-"   # 74 characters so with the blank the reader scores 75 classes

#character to slot number, and back
CHAR_TO_IDX = {c:i+1 for i,c in enumerate(CHARSET)}   #numbering starts at 1 which leaves slot 0 for the blank
IDX_TO_CHAR = {i:c for c,i in CHAR_TO_IDX.items()}

WEIGHTS_DIR = paths.Weights

#the crop size both networks take
INPUT_HEIGHT = 64
INPUT_WIDTH = 640   #640 wide turns into 160 ctc columns after the two poolings

#the reader, written by train.py
DEFAULT_TEXT_WEIGHTS = WEIGHTS_DIR/"paper_cnn_bilstm_ctc_scratch_best.npz"

#the writer namer, written by train.py
DEFAULT_AUTHOR_WEIGHTS = WEIGHTS_DIR/"author_shape_11w_pw170_scratch_weights.npz"

#---LoadStateDictAny: reads an .npz checkpoint into a dict of arrays, with its writer names---
def LoadStateDictAny(checkpointPath):
    with np.load(checkpointPath,allow_pickle=False) as npz:
        sd = {k: npz[k] for k in npz.files if not k.startswith("__")}   #keys starting with __ hold metadata rather than weights
        authorMapping = None
        if "__author_names__" in npz.files:
            names = [str(n) for n in npz["__author_names__"]]
            authorMapping = {name: idx for idx,name in enumerate(names)}   #writer name to the class index the namer was trained with
    return sd,authorMapping

#---Conv2d: slides learned filters over the image by turning each patch into a column---
def Conv2d(x,weight,bias,stride=1,padding=1):
    B,C,H,W = x.shape
    OutC,InC,kh,kw = weight.shape
    xp = np.pad(x,((0,0),(0,0),(padding,padding),(padding,padding)))   # a one pixel border of zeros holds the output size steady
    Hp,Wp = xp.shape[2],xp.shape[3]
    outH = (Hp-kh)//stride+1   #padded height less the kernel gives the output height
    outW = (Wp-kw)//stride+1

    s = xp.strides   #a strided view holding every patch, (B, C, kh, kw, outH, outW)
    shape = (B,C,kh,kw,outH,outW)
    strides = (s[0],s[1],s[2],s[3],s[2]*stride,s[3]*stride)   #one step along the kernel and one along the output both move one row
    cols = np.lib.stride_tricks.as_strided(xp,shape=shape,strides=strides)   # a view of every patch with no data copied
    cols = cols.reshape(B,C*kh*kw,outH*outW)   #each patch flattened into a column of C times kh times kw

    w = weight.reshape(OutC,C*kh*kw)    #each filter flattened into a row of that same length
    out = np.stack([w@cols[b] for b in range(B)],axis=0).reshape(B,OutC,outH,outW)    # one matrix multiply puts every filter at every position
    if bias is not None:
        out = out+bias.reshape(1,OutC,1,1)   #one bias per filter broadcast over the whole map
    return out

#BatchNorm2d: rescales each channel by the mean and variance recorded during training---
def BatchNorm2d(x,weight,bias,runningMean,runningVar,eps=1e-5):
    scale = weight/np.sqrt(runningVar+eps)   #gamma over the stored standard deviation
    shift = bias-runningMean*scale   # beta less the scaled running mean
    return x*scale.reshape(1,-1,1,1)+shift.reshape(1,-1,1,1)   #one multiply and one add per channel

#-Relu: keeps positives, sets negatives to zero
def Relu(x):
    return np.maximum(x,0)    #the larger of x and zero

#Sigmoid: squashes any number into 0 to 1
def Sigmoid(x):
    return 1.0/(1.0+np.exp(-x))  #1 over 1 plus e to the minus x lands between 0 and 1

#---MaxPool2d: keeps the largest value in each window, halving height and width---
def MaxPool2d(x,kernel=2,stride=2):
    B,C,H,W = x.shape
    outH,outW = H//stride,W//stride   #a 2x2 window halves both axes
    xc = x[:,:,:outH*stride,:outW*stride]   #drop a leftover odd row or column
    st = xc.strides
    windows = np.lib.stride_tricks.as_strided(
        xc,shape=(B,C,outH,outW,kernel,kernel),
        strides=(st[0],st[1],st[2]*stride,st[3]*stride,st[2],st[3]))   #two strides step the window and two step inside it
    return windows.max(axis=(4,5))  # the largest of the four values in each window

#---AdaptiveMaxPoolHeight: collapses the height to one row, keeping the largest value per column---
def AdaptiveMaxPoolHeight(x):
    return x.max(axis=2,keepdims=True)    # one value per column the strongest over all 16 rows

#---AdaptiveAvgPoolWidth: averages along the width axis---
def AdaptiveAvgPoolWidth(x):
    return x.mean(axis=2,keepdims=True)   # the mean along width leaves one number per channel

#---Linear: weights times input, plus bias---
def Linear(x,weight,bias):
    out = x@weight.T   # x times the weight matrix transposed
    return out if bias is None else out+bias

#---LogSoftmax: scores to log probabilities, with the maximum subtracted first---
def LogSoftmax(x,axis):
    m = np.max(x,axis=axis,keepdims=True)   #the maximum comes off first so exp stays in range
    shifted = x-m
    lse = np.log(np.sum(np.exp(shifted),axis=axis,keepdims=True))   #log of the summed exponentials
    return shifted-lse   # the shifted score less that log sum is the log probability

#---Softmax: scores to probabilities that sum to one---
def Softmax(x,axis):
    m = np.max(x,axis=axis,keepdims=True)
    e = np.exp(x-m)
    return e/np.sum(e,axis=axis,keepdims=True)   #each exponential over their total so the row sums to one

#----------------------------------------------------------------------
#FoldBatchNorm: rewrite the convolutions so the batch norms vanish
#at inference a batch norm is a fixed affine scale per channel, so it can be
#multiplied into the convolution above it: Wf = W*s, bf = (b-mean)*s+shift
#with s = weight/sqrt(var+eps).  the block then runs one operation instead
#of two and the batch-norm tensors are dropped from the dictionary
#----------------------------------------------------------------------
def FoldBatchNorm(sd,prefixes=("stage1","stage2","stage3")):
    out = dict(sd)
    for prefix in prefixes:
        i = 0
        while f"{prefix}.{i}.weight" in out:
            bn = f"{prefix}.{i+1}"
            if f"{bn}.running_var" not in out:
                i += 3
                continue
            scale = out[f"{bn}.weight"]/np.sqrt(out[f"{bn}.running_var"]+1e-5)   # s is gamma over the root of the running variance
            w,b = out[f"{prefix}.{i}.weight"],out[f"{prefix}.{i}.bias"]
            out[f"{prefix}.{i}.weight"] = w*scale.reshape(-1,1,1,1)   #every filter scaled by its own channel factor
            out[f"{prefix}.{i}.bias"] = (b-out[f"{bn}.running_mean"])*scale+out[f"{bn}.bias"]   #the conv bias less the running mean scaled then offset by beta
            for k in ("weight","bias","running_mean","running_var"):
                out.pop(f"{bn}.{k}",None)
            i += 3
    return out

#---AsFloat32: every stored tensor in single precision, which halves the traffic---
def AsFloat32(sd):
    return {k:(v.astype(np.float32) if hasattr(v,"dtype") and v.dtype==np.float64 else v)   #float64 tensors drop to float32 and halve the traffic
            for k,v in sd.items()}

#----------------------------------------------------------------------
#ConvBnReluFolded: one 3x3 convolution, its folded batch norm and the relu
#in a single pass.  the three stages above each walked the whole feature map
#and allocated a fresh array doing it; here the patch matrix is multiplied
#straight into a preallocated buffer and the bias and relu are applied in
#place, so one array is touched instead of four
#----------------------------------------------------------------------
def ConvBnReluFolded(x,weight,bias):
    B,C,H,W = x.shape
    OutC = weight.shape[0]
    xp = np.zeros((B,C,H+2,W+2),dtype=x.dtype)   #zeros+slice, not np.pad
    xp[:,:,1:H+1,1:W+1] = x   #the image sits in the middle of the zero buffer
    st = xp.strides
    cols = np.lib.stride_tricks.as_strided(
        xp,shape=(B,C,3,3,H,W),
        strides=(st[0],st[1],st[2],st[3],st[2],st[3])).reshape(B,C*9,H*W)   #the same patch view flattened to C times 9 rows
    w2d = weight.reshape(OutC,C*9)
    out = np.empty((B,OutC,H*W),dtype=x.dtype)
    for b in range(B):
        np.matmul(w2d,cols[b],out=out[b])   #the product lands straight in the output buffer
    out += bias.reshape(1,OutC,1)  #bias added in place
    np.maximum(out,0,out=out)  #relu in place on the same buffer
    return out.reshape(B,OutC,H,W)

#---ConvBlockForward: the repeated convolution, batch-norm, relu block of the backbone---
def ConvBlockForward(x,sd,prefix,nLayers):
    for i in range(nLayers):
        c = i*3
        if f"{prefix}.{c+1}.running_var" not in sd:   #FoldBatchNorm has run
            x = ConvBnReluFolded(x,sd[f"{prefix}.{c}.weight"],sd[f"{prefix}.{c}.bias"])
            continue
        x = Conv2d(x,sd[f"{prefix}.{c}.weight"],sd[f"{prefix}.{c}.bias"],padding=1)
        x = BatchNorm2d(x,sd[f"{prefix}.{c+1}.weight"],sd[f"{prefix}.{c+1}.bias"],
                         sd[f"{prefix}.{c+1}.running_mean"],sd[f"{prefix}.{c+1}.running_var"])
        x = Relu(x)
    return x

#---LstmForward: reads a sequence one column at a time through four gates, in one direction---
_LSTM_PACKS = {}   #packed weights, keyed by the identity of the stored tensor

#---_PackLstm: the gate rows reordered to i,f,o,g so one sigmoid covers three---
def _PackLstm(weightIh,weightHh,biasIh,biasHh,H):
    order = np.r_[np.arange(0,2*H),np.arange(3*H,4*H),np.arange(2*H,3*H)]   # the gate rows move from i f g o into i f o g
    return dict(WihT=np.ascontiguousarray(weightIh[order].T),   #transposed once here so each step is a plain multiply
                WhhT=np.ascontiguousarray(weightHh[order].T),
                bias=(biasIh+biasHh)[order])   #the two bias vectors add into one

def LstmForward(x,weightIh,weightHh,biasIh,biasHh,hiddenSize):
    T,B,_ = x.shape
    H = hiddenSize
    key = id(weightIh)
    pk = _LSTM_PACKS.get(key)
    if pk is None:
        pk = _LSTM_PACKS[key] = _PackLstm(weightIh,weightHh,biasIh,biasHh,H)
    h = np.zeros((B,H),dtype=x.dtype)
    c = np.zeros((B,H),dtype=x.dtype)
    outputs = np.empty((T,B,H),dtype=x.dtype)
    #the input projection does not depend on the recurrence, so every
    #timestep is done in one matrix multiply before the loop starts
    pre = (x.reshape(T*B,-1)@pk["WihT"]).reshape(T,B,4*H)   #every timestep projected in one multiply ahead of the loop
    pre += pk["bias"]
    g = np.empty((B,4*H),dtype=x.dtype)
    for t in range(T):
        np.matmul(h,pk["WhhT"],out=g)   #the recurrent half is the only part inside the loop
        g += pre[t]   #gate pre activations for this column
        sig = g[:,:3*H]   #i, f and o are adjacent after the reorder
        np.negative(sig,out=sig); np.exp(sig,out=sig)   # exp of minus the scores in place
        sig += 1.0; np.reciprocal(sig,out=sig)   #1 over 1 plus that gives the three sigmoids at once
        gg = np.tanh(g[:,3*H:])   #the candidate runs through tanh
        c *= g[:,H:2*H]   #the forget gate scales the old memory
        c += g[:,:H]*gg   # the input gate adds the new candidate
        np.tanh(c,out=outputs[t])   #memory squashed through tanh
        outputs[t] *= g[:,2*H:3*H]   #the output gate sets how much of it shows
        h = outputs[t]
    return outputs,(h,c)

#---BidirectionalLstmForward: runs the sequence both ways and concatenates the two hidden states---
def BidirectionalLstmForward(x,paramsFwd,paramsBwd,hiddenSize):
    outF,_ = LstmForward(x,**paramsFwd,hiddenSize=hiddenSize)
    outB,_ = LstmForward(x[::-1],**paramsBwd,hiddenSize=hiddenSize)   #the second direction reads the columns backwards
    outB = outB[::-1]   #flipped back so column t lines up in both halves
    return np.concatenate([outF,outB],axis=2)   #256 forward and 256 backward make 512 per column

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
    arr = np.array(img,dtype=np.float32)/255.0   #0 to 255 becomes 0 to 1
    arr = 1.0-arr  #inverted so ink reads high and paper low
    arr = (arr-0.5)/0.5  # shifted again to minus 1 for paper and plus 1 for ink
    return arr[np.newaxis,np.newaxis,:,:]   #one image one channel is the shape both networks take

#the pen width stroke normalising redraws at, in x-heights
PenWidthXh = 0.17    # pen width as a fraction of the x-height

#---StrokeNormalize: thins the ink to its centreline and redraws it at one fixed pen width---
def StrokeNormalize(gray,penWidthXh=PenWidthXh):
    ink = BinarizeLine(gray)
    if not ink.any():
        return None
    band = CoreBand(ink)
    xh = float(band[1]-band[0]) if band else 0.0   # baseline less band top is the x-height in pixels
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
    bestPath = np.argmax(logProbs,axis=2)   # the highest scoring class in every column
    decoded = []
    for batchIdx in range(bestPath.shape[1]):
        prev,chars = None,[]
        for token in bestPath[:,batchIdx]:
            token = int(token)
            if token!=0 and token!=prev:   #repeats collapse and the blank at slot 0 falls away
                chars.append(IDX_TO_CHAR.get(token,""))
            prev = token
        decoded.append("".join(chars))
    return decoded

#----------------------------------------------------------------------
#CtcForcedAlign: walks the one path through the scores that spells a known text, and
#returns the column span of every character plus the path's mean score
#----------------------------------------------------------------------
def CtcForcedAlign(logProbs,text):
    labels = [CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX]   #the known text as class numbers
    if not labels:
        return None
    ext = [0]
    for l in labels:
        ext+=[l,0]   #a blank goes between and around every character
    S,T = len(ext),logProbs.shape[0]    #S is twice the characters plus 1 and T is the column count
    if T<len(labels):
        return None
    NEG = -1e30   #stands in for log of zero on a barred move
    dp = np.full((T,S),NEG)
    bp = np.zeros((T,S),np.int32)
    dp[0,0] = logProbs[0,ext[0]]   #column 0 may start on the leading blank
    if S>1:
        dp[0,1] = logProbs[0,ext[1]]   #or straight on the first character
    for t in range(1,T):
        emit = logProbs[t,ext]   #the score of emitting each extended symbol now
        stay = dp[t-1]   #holding on the same row
        prev1 = np.concatenate(([NEG],dp[t-1,:-1]))   #stepping down one row
        prev2 = np.concatenate(([NEG,NEG],dp[t-1,:-2]))    # skipping down two rows across a blank
        for s in range(S):   #a skip is allowed only onto a non-blank differing two back
            if s>=2 and (ext[s]==0 or ext[s]==ext[s-2]):   #the skip is barred onto a blank and between two equal characters
                prev2[s] = NEG
        cand = np.stack([stay,prev1,prev2])
        best = np.argmax(cand,axis=0)
        dp[t] = cand[best,np.arange(S)]+emit    #best predecessor plus this emission is the viterbi step
        bp[t] = best   #0 1 or 2 records how far the path stepped
    endS = S-1 if S==1 else (S-1 if dp[T-1,S-1]>=dp[T-1,S-2] else S-2)  # the path finishes on the last blank or the last character
    path = np.zeros(T,np.int32)
    s = endS
    for t in range(T-1,-1,-1):
        path[t] = s
        s-=bp[t,s]   #backtracking takes the recorded step back off
    spans = []
    for ci in range(len(labels)):
        sIdx = 1+2*ci
        ts = np.nonzero(path==sIdx)[0]   #odd rows hold characters so row 1+2i is character i
        if len(ts)==0:
            spans.append(None)
        else:
            spans.append((int(ts[0]),int(ts[-1])+1))   #first and last column the path spent on that character
    kept = [c for c in text if c in CHAR_TO_IDX]
    conf = float(np.mean([logProbs[t,ext[path[t]]] for t in range(T)]))  #mean log probability along the chosen path
    return list(zip(kept,spans)),conf

#---FrameXToPixel: undoes the resize and padding, mapping a canvas column back to a crop pixel---
def FrameXToPixel(canvasX,rawW,origH):
    s = min(INPUT_HEIGHT/origH,INPUT_WIDTH/rawW)    # the same single scale the canvas fit used
    return float(np.clip(canvasX/s,0.0,float(rawW)))   #dividing by it carries a canvas column back to a crop pixel

#----------------------------------------------------------------------
#PaperCRNNNumpy: the reader, the convolution backbone, the LSTM stack and a score per
#character per column
#----------------------------------------------------------------------
class PaperCRNNNumpy:
    def __init__(self,checkpointPath=None,hiddenSize=256):
        self.checkpointPath = Path(checkpointPath or DEFAULT_TEXT_WEIGHTS)
        self.sd,_ = LoadStateDictAny(self.checkpointPath)
        self.sd = AsFloat32(FoldBatchNorm(self.sd))   #batch norms folded into the convolutions then cast to float32
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
        x = x[:,:,0,:]    # the height axis is now 1 and drops away
        x = np.transpose(x,(2,0,1))   #width becomes the CTC time axis
        seq = MultilayerBidirectionalLstm(x,self.lstmLayerParams,self.hiddenSize)
        logits = Linear(seq,sd["text_head.weight"],sd["text_head.bias"])   #512 features into one score per character per column
        return LogSoftmax(logits,axis=2)   # log probabilities across the 75 classes

#---ReadTextScored: reads a crop and returns the text with the mean best-column log probability---
def ReadTextScored(image,model):
    t = TensorFromResized(ResizeLineImageFixed(image))
    logProbs = model.Forward(t)
    best = np.max(logProbs[:,0,:],axis=1)    #the best score in each column
    return DecodeCtc(logProbs)[0],float(np.mean(best))   #their mean is the confidence of the reading

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
            current.append(min(current[j]+1,previous[j+1]+1,previous[j]+(ca!=cb)))   #insert delete or substitute whichever costs least
        previous = current
    return previous[-1]

#---CharAcc: fraction of characters correct, from that edit distance---
def CharAcc(pred,truth):
    if not truth:
        return 1.0 if not pred else 0.0
    return max(0.0,1.0-Levenshtein(pred,truth)/len(truth))    #edits over length taken off 1 gives character accuracy

#---AuthorClassifierCNNNumpy: the writer namer, the same backbone, no recurrence, one score per writer---
class AuthorClassifierCNNNumpy:
    def __init__(self,checkpointPath=None):
        self.checkpointPath = Path(checkpointPath or DEFAULT_AUTHOR_WEIGHTS)
        self.sd,self.authorMapping = LoadStateDictAny(self.checkpointPath)
        self.sd = AsFloat32(FoldBatchNorm(self.sd))

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
        pooled = AdaptiveAvgPoolWidth(x)[:,:,0]   #averaging the 160 columns leaves 128 numbers per line
        h = Linear(pooled,sd["author_head.0.weight"],sd["author_head.0.bias"])   #128 features into 96 hidden
        h = Relu(h)
        #author_head.2 is dropout, inactive at inference
        logits = Linear(h,sd["author_head.3.weight"],sd["author_head.3.bias"])   #96 hidden into one score per writer
        return logits

# ClassifyImage: crop in, writer index and the probability over writers out
def ClassifyImage(image,model,authorMapping=None,strokeNormalise=True):
    if strokeNormalise:
        normImg = StrokeNormalize(np.asarray(image,dtype=np.uint8))   # the crop redrawn at one pen width before it is scored
        if normImg is not None:
            image = normImg
    t = TensorFromResized(ResizeLineImageFixed(image))
    logits = model.Forward(t)
    probs = Softmax(logits,axis=1)[0]   #writer scores turned into probabilities summing to one
    predIdx = int(np.argmax(probs))   #the highest scoring writer
    return predIdx,probs
