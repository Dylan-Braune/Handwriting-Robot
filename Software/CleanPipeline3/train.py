#u23539195 Dylan Braune.  training the two networks.  FU 3, FU 4
#
#this file holds everything that only training needs:
#  the backward pass of every layer, worked out by hand and checked against
#  numerically estimated gradients
#  the losses, the optimisers and the learning rate schedules
#  the datasets, the augmentation, and the two training runs themselves
#
#recognise.py runs the trained networks forward; nothing here is needed to
#USE a checkpoint, only to make one
import io
import hashlib
import math
import os
import pickle
import random
import re
import sys
import time
import zlib
from pathlib import Path

import numpy as np

#run as a script, python does not always put this file's own folder on the
#import path, so the sibling modules below are found explicitly
sys.path.insert(0,str(Path(__file__).resolve().parent))

import paths
import recognise as CLS   #the eval-only forward passes below run on recognise's versions
from recognise import (CHARSET,CHAR_TO_IDX,INPUT_WIDTH,
                       Sigmoid,LoadStateDictAny,DecodeCtc,Levenshtein,
                       CharAcc,StrokeNormalize,ResizeLineImageFixed,
                       TensorFromResized)
import segment as SL   #AddPersonalSamples cuts the photographed pages into lines with it
from segment import (Elastic,Otsu,Resize,RowRunLength,Warp)


#======================================================================
#THE LAYERS, BACKWARD AS WELL AS FORWARD
#each forward keeps what the backward needs, and each backward returns the
#gradient for its input and for its own weights.  hand derived, then checked
#against numerical gradients, which agreed to about 1e-11
#======================================================================
#----------------------------------------------------------------------
#Conv2dForward: convolution, keeping what backward needs
#same patches-into-columns trick as the forward only version, but it also
#returns the column matrix, because backward needs it again
#----------------------------------------------------------------------
def Conv2dForward(x,weight,bias,padding=1):
    B,C,H,W = x.shape
    OutC,InC,kh,kw = weight.shape
    xp = np.pad(x,((0,0),(0,0),(padding,padding),(padding,padding)))
    Hp,Wp = xp.shape[2],xp.shape[3]
    outH = Hp-kh+1
    outW = Wp-kw+1

    s = xp.strides
    shape = (B,C,kh,kw,outH,outW)
    strides = (s[0],s[1],s[2],s[3],s[2],s[3])
    #as_strided makes a VIEW holding every 3x3 patch without copying any
    #data, by lying about the strides.  that view is what im2col needs
    cols = np.lib.stride_tricks.as_strided(xp,shape=shape,strides=strides)
    colsFlat = cols.reshape(B,C*kh*kw,outH*outW)

    w = weight.reshape(OutC,C*kh*kw)
    out = (w@colsFlat).reshape(B,OutC,outH,outW)   #w(OutC,K) broadcasts over colsFlat's batch dim -> one batched BLAS gemm, not a per-sample python loop
    if bias is not None:
        out = out+bias.reshape(1,OutC,1,1)
    cache = (x.shape,weight,colsFlat,padding,kh,kw,outH,outW)
    return out,cache

#----------------------------------------------------------------------
#Conv2dBackward: gradients for a convolution
#the weight gradient is the output gradient times those same columns,
#and the input gradient is the output gradient times the weights, folded
#back from columns into image shape
#----------------------------------------------------------------------
def Conv2dBackward(dout,cache):
    xShape,weight,colsFlat,padding,kh,kw,outH,outW = cache
    B,C,H,W = xShape
    OutC = weight.shape[0]
    doutFlat = dout.reshape(B,OutC,outH*outW)

    dw = (doutFlat@colsFlat.transpose(0,2,1)).sum(axis=0).reshape(weight.shape)   #batched matmul, summed over the batch
    db = dout.sum(axis=(0,2,3))

    w = weight.reshape(OutC,C*kh*kw)
    dcolsFlat = w.T@doutFlat   #w.T(K,OutC) broadcasts over doutFlat's batch dim, same trick as forward
    dcols = dcolsFlat.reshape(B,C,kh,kw,outH,outW)

    Hp,Wp = H+2*padding,W+2*padding
    dxp = np.zeros((B,C,Hp,Wp),dtype=dout.dtype)
    for i in range(kh):
        for j in range(kw):
            dxp[:,:,i:i+outH,j:j+outW]+=dcols[:,:,i,j,:,:]
    dx = dxp[:,:,padding:padding+H,padding:padding+W]
    return dx,dw,db

#----------------------------------------------------------------------
#BatchNorm2dForwardTrain: normalise using THIS batch
#training uses the batch's own mean and spread, not the stored ones,
#and updates the stored ones as it goes so inference has them later
#----------------------------------------------------------------------
def BatchNorm2dForwardTrain(x,gamma,beta,runningMean,runningVar,momentum=0.1,eps=1e-5):
    N = x.shape[0]*x.shape[2]*x.shape[3]
    mu = x.mean(axis=(0,2,3))
    xc = x-mu.reshape(1,-1,1,1)
    varBiased = (xc**2).mean(axis=(0,2,3))
    std = np.sqrt(varBiased+eps)
    xhat = xc/std.reshape(1,-1,1,1)
    out = gamma.reshape(1,-1,1,1)*xhat+beta.reshape(1,-1,1,1)   #one scale and shift per CHANNEL, broadcast over B,H,W

    varUnbiased = varBiased*N/max(N-1,1)   #pytorch updates running_var with the unbiased estimate
    runningMean[:] = (1-momentum)*runningMean+momentum*mu
    runningVar[:] = (1-momentum)*runningVar+momentum*varUnbiased

    cache = (xhat,std,gamma,N)
    return out,cache

#----------------------------------------------------------------------
#BatchNorm2dBackward: gradients through the normalising
#awkward because the mean and spread depend on every item in the batch,
#so changing one input changes the whole batch's normalisation
#----------------------------------------------------------------------
def BatchNorm2dBackward(dout,cache):
    xhat,std,gamma,N = cache
    dgamma = (dout*xhat).sum(axis=(0,2,3))
    dbeta = dout.sum(axis=(0,2,3))
    dxhat = dout*gamma.reshape(1,-1,1,1)
    #standard batchnorm backward (cs231n derivation), N = elements per channel
    dx = (1.0/(N*std.reshape(1,-1,1,1)))*(
        N*dxhat
        -dxhat.sum(axis=(0,2,3)).reshape(1,-1,1,1)
        -xhat*(dxhat*xhat).sum(axis=(0,2,3)).reshape(1,-1,1,1)
    )
    return dx,dgamma,dbeta

#---ReluForward: keep positives, remember which ones they were---
def ReluForward(x):
    out = np.maximum(x,0)
    return out,(x>0)

#---ReluBackward: gradient passes only where the input was positive---
def ReluBackward(dout,cache):
    mask = cache
    return dout*mask

#---MaxPool2dForward: keep the biggest in each window, remember where it was---
def MaxPool2dForward(x,kernel=2,stride=2):
    B,C,H,W = x.shape
    outH,outW = H//stride,W//stride
    xc = x[:,:,:outH*stride,:outW*stride]
    s = xc.strides
    shape = (B,C,outH,outW,kernel,kernel)
    strides = (s[0],s[1],s[2]*stride,s[3]*stride,s[2],s[3])
    #a no copy view of every pooling window, so one max does the whole image
    windows = np.lib.stride_tricks.as_strided(xc,shape=shape,strides=strides)
    flat = windows.reshape(B,C,outH,outW,kernel*kernel)
    argmax = flat.argmax(axis=4)
    out = np.take_along_axis(flat,argmax[...,None],axis=4)[...,0]
    cache = (x.shape,argmax,kernel,stride,outH,outW)
    return out,cache

#---MaxPool2dBackward: gradient goes back only to the pixel that won---
def MaxPool2dBackward(dout,cache):
    xShape,argmax,kernel,stride,outH,outW = cache
    B,C,H,W = xShape
    dflat = np.zeros((B,C,outH,outW,kernel*kernel),dtype=dout.dtype)
    np.put_along_axis(dflat,argmax[...,None],dout[...,None],axis=4)
    dwindows = dflat.reshape(B,C,outH,outW,kernel,kernel)
    dx = np.zeros(xShape,dtype=dout.dtype)
    #non overlapping windows (stride==kernel): direct block placement, no accumulation needed
    dx[:,:,:outH*stride,:outW*stride] = dwindows.transpose(0,1,2,4,3,5).reshape(B,C,outH*stride,outW*stride)
    return dx

#---AdaptiveMaxPoolHeightForward: squash the height to one row---
def AdaptiveMaxPoolHeightForward(x):
    argmax = x.argmax(axis=2)   #(B,C,W)
    out = np.take_along_axis(x,argmax[:,:,None,:],axis=2)
    cache = (x.shape,argmax)
    return out,cache

#---AdaptiveMaxPoolHeightBackward: gradient back to the winning row---
def AdaptiveMaxPoolHeightBackward(dout,cache):
    xShape,argmax = cache
    dx = np.zeros(xShape,dtype=dout.dtype)
    np.put_along_axis(dx,argmax[:,:,None,:],dout,axis=2)
    return dx

#---AdaptiveAvgPoolWidthForward: average the width to a fixed size---
def AdaptiveAvgPoolWidthForward(x):
    out = x.mean(axis=2,keepdims=True)
    return out,x.shape[2]

#---AdaptiveAvgPoolWidthBackward: spread the gradient evenly back over the width---
def AdaptiveAvgPoolWidthBackward(dout,cache):
    W = cache
    return np.repeat(dout,W,axis=2)/W

#---LinearForward: weights times input plus bias---
def LinearForward(x,weight,bias):
    out = x@weight.T
    if bias is not None:
        out = out+bias
    return out,(x,weight)

#---LinearBackward: gradients for input, weights and bias---
def LinearBackward(dout,cache):
    x,weight = cache
    #x can be (...,inFeatures) with leading dims beyond batch (e.g. (T,B,in)
    #for the lstm->linear text head) -- flatten everything but the last axis
    flatX = x.reshape(-1,x.shape[-1])
    flatDout = dout.reshape(-1,dout.shape[-1])
    dw = flatDout.T@flatX
    db = flatDout.sum(axis=0)
    dx = (dout@weight).reshape(x.shape)
    return dx,dw,db

#----------------------------------------------------------------------
#DropoutForward: randomly zero some values while training
#stops the network leaning on any one feature.  the survivors are
#scaled up so the overall level stays the same
#----------------------------------------------------------------------
def DropoutForward(x,p,training,rng):
    if not training or p<=0:
        return x,None
    mask = (rng.random(x.shape)>=p)/(1.0-p)
    return x*mask,mask

#---DropoutBackward: gradient follows the same mask---
def DropoutBackward(dout,cache):
    mask = cache
    if mask is None:
        return dout
    return dout*mask

#---ConvBlockForwardTrain: the convolution, normalise, relu block with its saved values---
def ConvBlockForwardTrain(x,params,prefix,nLayers):
    caches = []
    for i in range(nLayers):
        c = i*3
        x,convCache = Conv2dForward(x,params[f"{prefix}.{c}.weight"],params[f"{prefix}.{c}.bias"],padding=1)
        x,bnCache = BatchNorm2dForwardTrain(
            x,params[f"{prefix}.{c+1}.weight"],params[f"{prefix}.{c+1}.bias"],
            params[f"{prefix}.{c+1}.running_mean"],params[f"{prefix}.{c+1}.running_var"],
        )
        x,reluCache = ReluForward(x)
        caches.append((convCache,bnCache,reluCache))
    return x,caches

#---ConvBlockBackwardTrain: and its gradients, in reverse order---
def ConvBlockBackwardTrain(dout,caches,prefix,nLayers):
    grads = {}
    for i in reversed(range(nLayers)):
        c = i*3
        convCache,bnCache,reluCache = caches[i]
        dout = ReluBackward(dout,reluCache)
        dout,dgamma,dbeta = BatchNorm2dBackward(dout,bnCache)
        dout,dw,db = Conv2dBackward(dout,convCache)
        grads[f"{prefix}.{c}.weight"] = dw
        grads[f"{prefix}.{c}.bias"] = db
        grads[f"{prefix}.{c+1}.weight"] = dgamma
        grads[f"{prefix}.{c+1}.bias"] = dbeta
    return dout,grads



#======================================================================
#THE MEMORY LAYER, BACKWARD AS WELL
#the lstm is the fiddliest part, because gradient flows back through time
#======================================================================
#----------------------------------------------------------------------
#LstmForwardTrain: run the lstm, keeping every gate value
#backward needs what each gate did at each step, so all of it is kept
#----------------------------------------------------------------------
def LstmForwardTrain(x,weightIh,weightHh,biasIh,biasHh,hiddenSize):
    T,B,_ = x.shape
    H = hiddenSize
    hPrev = np.zeros((B,H),dtype=x.dtype)
    cPrev = np.zeros((B,H),dtype=x.dtype)
    outputs = np.empty((T,B,H),dtype=x.dtype)
    stepCache = []

    for t in range(T):
        gates = x[t]@weightIh.T+biasIh+hPrev@weightHh.T+biasHh
        i = Sigmoid(gates[:,0*H:1*H])
        f = Sigmoid(gates[:,1*H:2*H])
        g = np.tanh(gates[:,2*H:3*H])
        o = Sigmoid(gates[:,3*H:4*H])
        cNew = f*cPrev+i*g
        tanhC = np.tanh(cNew)
        hNew = o*tanhC
        outputs[t] = hNew
        stepCache.append((x[t],hPrev,cPrev,i,f,g,o,tanhC))
        hPrev,cPrev = hNew,cNew

    cache = (stepCache,weightIh,weightHh,H)
    return outputs,cache

#----------------------------------------------------------------------
#LstmBackward: gradients through the lstm, walking back in time
#starts at the last column and works backwards, carrying the gradient of
#the memory as well as the output, because each step fed the next
#----------------------------------------------------------------------
def LstmBackward(dOutputs,cache):
    stepCache,weightIh,weightHh,H = cache
    T = len(stepCache)
    B = dOutputs.shape[1]
    inputSize = weightIh.shape[1]

    dWeightIh = np.zeros_like(weightIh)
    dWeightHh = np.zeros_like(weightHh)
    dBiasIh = np.zeros(4*H,dtype=dOutputs.dtype)
    dBiasHh = np.zeros(4*H,dtype=dOutputs.dtype)
    dX = np.empty((T,B,inputSize),dtype=dOutputs.dtype)

    dhNext = np.zeros((B,H),dtype=dOutputs.dtype)
    dcNext = np.zeros((B,H),dtype=dOutputs.dtype)

    for t in range(T-1,-1,-1):
        xt,hPrev,cPrev,i,f,g,o,tanhC = stepCache[t]
        dh = dOutputs[t]+dhNext
        do = dh*tanhC
        doPre = do*o*(1-o)
        dc = dcNext+dh*o*(1-tanhC**2)
        df = dc*cPrev
        dfPre = df*f*(1-f)
        di = dc*g
        diPre = di*i*(1-i)
        dg = dc*i
        dgPre = dg*(1-g**2)
        dcPrev = dc*f

        dGates = np.concatenate([diPre,dfPre,dgPre,doPre],axis=1)   #(B,4H)
        dWeightIh+=dGates.T@xt
        dWeightHh+=dGates.T@hPrev
        dBiasIh+=dGates.sum(axis=0)
        dBiasHh+=dGates.sum(axis=0)
        dX[t] = dGates@weightIh
        dhNext = dGates@weightHh
        dcNext = dcPrev

    return dX,dWeightIh,dWeightHh,dBiasIh,dBiasHh

#---BidirectionalLstmForwardTrain: both directions, keeping both sets of values---
def BidirectionalLstmForwardTrain(x,paramsFwd,paramsBwd,hiddenSize):
    outF,cacheF = LstmForwardTrain(x,**paramsFwd,hiddenSize=hiddenSize)
    outB,cacheB = LstmForwardTrain(x[::-1],**paramsBwd,hiddenSize=hiddenSize)
    outB = outB[::-1]
    out = np.concatenate([outF,outB],axis=2)
    return out,(cacheF,cacheB,hiddenSize)

#---BidirectionalLstmBackward: split the gradient and send each half back its own way---
def BidirectionalLstmBackward(dout,cache):
    cacheF,cacheB,hiddenSize = cache
    H = hiddenSize
    dOutF = dout[:,:,:H]
    dOutB = dout[:,:,H:2*H]

    dxF,dWihF,dWhhF,dbihF,dbhhF = LstmBackward(dOutF,cacheF)
    #the backward direction ran on the time reversed input, so its own output
    #gradient must also be time reversed before feeding LstmBackward, and the
    #resulting dx reversed back afterwards
    dxB,dWihB,dWhhB,dbihB,dbhhB = LstmBackward(dOutB[::-1],cacheB)
    dxB = dxB[::-1]

    dx = dxF+dxB
    gradsFwd = dict(weightIh=dWihF,weightHh=dWhhF,biasIh=dbihF,biasHh=dbhhF)
    gradsBwd = dict(weightIh=dWihB,weightHh=dWhhB,biasIh=dbihB,biasHh=dbhhB)
    return dx,gradsFwd,gradsBwd

#---MultilayerBidirectionalLstmForwardTrain: stacked layers, values kept per layer---
def MultilayerBidirectionalLstmForwardTrain(x,layerParams,hiddenSize):
    out = x
    caches = []
    for layer in layerParams:
        out,cache = BidirectionalLstmForwardTrain(out,layer["fwd"],layer["bwd"],hiddenSize)
        caches.append(cache)
    return out,caches

#---MultilayerBidirectionalLstmBackward: gradients down through the stack---
def MultilayerBidirectionalLstmBackward(dout,caches):
    dx = dout
    gradsPerLayer = []
    for cache in reversed(caches):
        dx,gradsFwd,gradsBwd = BidirectionalLstmBackward(dx,cache)
        gradsPerLayer.append((gradsFwd,gradsBwd))
    gradsPerLayer.reverse()
    return dx,gradsPerLayer



#======================================================================
#THE LOSSES
#what the network is actually being pushed towards
#======================================================================
#---LogSumExp: add numbers that are stored as logs, without overflowing---
def LogSumExp(a,b):
    m = np.maximum(a,b)
    #-inf + -inf would nan in the subtraction below (even though it's masked
    #out by np.where afterward, numpy still evaluates it eagerly and warns) --
    #errstate suppresses that specific, already-handled warning
    with np.errstate(invalid="ignore"):
        #log(e^a+e^b) = m+log(e^(a-m)+e^(b-m)) with m the larger, so nothing
        #overflows.  both -inf is a forbidden transition and must stay -inf
        out = np.where(np.isneginf(m),-np.inf,m+np.log(np.exp(a-m)+np.exp(b-m)))
    return out

#----------------------------------------------------------------------
#CtcLoss: scoring a read line when you do not know where letters are
#the label says WHAT was written but not WHERE, so CTC adds up the
#probability of every possible alignment of that text to the columns.
#done with a forward and a backward sweep in logs, which is also where
#the gradient comes from
#----------------------------------------------------------------------
def CtcLoss(rawLogits,targets):
    T,B,C = rawLogits.shape
    m = rawLogits.max(axis=2,keepdims=True)
    logProbs = rawLogits-m-np.log(np.exp(rawLogits-m).sum(axis=2,keepdims=True))   #log_softmax
    probs = np.exp(logProbs)

    dRawLogits = np.zeros_like(rawLogits)
    totalLoss = 0.0
    NEG = -np.inf

    for b in range(B):
        labels = targets[b]
        ext = [0]
        for l in labels:
            ext+=[l,0]
        S = len(ext)
        lp = logProbs[:,b,:]   #(T,C)

        logAlpha = np.full((T,S),NEG)
        logAlpha[0,0] = lp[0,ext[0]]
        if S>1:
            logAlpha[0,1] = lp[0,ext[1]]
        for t in range(1,T):
            stay = logAlpha[t-1]
            prev1 = np.concatenate(([NEG],logAlpha[t-1,:-1]))
            prev2 = np.concatenate(([NEG,NEG],logAlpha[t-1,:-2]))
            for s in range(2,S):
                if ext[s]==0 or ext[s]==ext[s-2]:
                    prev2[s] = NEG   #skip transition forbidden: blank, or repeats the label 2 back
            combined = LogSumExp(LogSumExp(stay,prev1),prev2)
            logAlpha[t] = combined+lp[t,ext]

        logBeta = np.full((T,S),NEG)
        logBeta[T-1,S-1] = 0.0
        if S>1:
            logBeta[T-1,S-2] = 0.0
        for t in range(T-2,-1,-1):
            nxt = logBeta[t+1]+lp[t+1,ext]
            stay = nxt
            nextMinus1 = np.concatenate((nxt[1:],[NEG]))
            nextMinus2 = np.concatenate((nxt[2:],[NEG,NEG]))
            for s in range(S-2):
                if ext[s]==0 or ext[s]==ext[s+2]:
                    nextMinus2[s] = NEG
            logBeta[t] = LogSumExp(LogSumExp(stay,nextMinus1),nextMinus2)

        logP = LogSumExp(logAlpha[T-1,S-1],logAlpha[T-1,S-2] if S>1 else NEG)
        totalLoss+=-logP

        #occupation: for each timestep/class, sum of alpha*beta over every
        #extended-sequence position s that emits that class, normalised by
        #the total path probability -- this is exactly Graves eq 15/16
        logGamma = logAlpha+logBeta-logP   #(T,S)
        gamma = np.exp(logGamma)
        occ = np.zeros((T,C))
        for s in range(S):
            occ[:,ext[s]]+=gamma[:,s]
        dRawLogits[:,b,:] = probs[:,b,:]-occ

    return totalLoss/B,dRawLogits/B

#---CrossEntropyLoss: the plain 'which class is it' loss, for the writer namer---
def CrossEntropyLoss(logits,targetIdx):
    B = logits.shape[0]
    m = logits.max(axis=1,keepdims=True)
    logProbs = logits-m-np.log(np.exp(logits-m).sum(axis=1,keepdims=True))
    loss = -logProbs[np.arange(B),targetIdx].mean()
    probs = np.exp(logProbs)
    dLogits = probs.copy()
    dLogits[np.arange(B),targetIdx]-=1.0
    dLogits/=B
    return loss,dLogits



#======================================================================
#THE OPTIMISERS AND SCHEDULES
#how a gradient becomes a weight update, and how the step size changes
#======================================================================
#----------------------------------------------------------------------
#RMSprop: scale each weight's step by how noisy its gradient has been
#keeps a running average of the squared gradient and divides by it, so a
#weight with wildly varying gradient takes smaller steps
#----------------------------------------------------------------------
class RMSprop:
    def __init__(self,params,lr=1e-3,alpha=0.99,eps=1e-8,weightDecay=0.0):
        self.params = params   #dict name -> array (updated in place)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weightDecay = weightDecay
        self.sqAvg = {k: np.zeros_like(v) for k,v in params.items()}

    def Step(self,grads):
        for k,p in self.params.items():
            g = grads[k]
            if self.weightDecay>0:
                g = g+self.weightDecay*p
            self.sqAvg[k] = self.alpha*self.sqAvg[k]+(1-self.alpha)*(g**2)
            p-=self.lr*g/(np.sqrt(self.sqAvg[k])+self.eps)

#----------------------------------------------------------------------
#AdamW: momentum and that scaling together, with honest weight decay
#keeps running averages of the gradient and its square, corrects them at
#the start when they are still cold, and shrinks the weights separately
#from the gradient step
#----------------------------------------------------------------------
class AdamW:
    def __init__(self,params,lr=1e-3,betas=(0.9,0.999),eps=1e-8,weightDecay=0.0):
        self.params = params
        self.lr = lr
        self.beta1,self.beta2 = betas
        self.eps = eps
        self.weightDecay = weightDecay
        self.m = {k: np.zeros_like(v) for k,v in params.items()}
        self.v = {k: np.zeros_like(v) for k,v in params.items()}
        self.t = 0

    def Step(self,grads):
        self.t+=1
        for k,p in self.params.items():
            g = grads[k]
            self.m[k] = self.beta1*self.m[k]+(1-self.beta1)*g
            self.v[k] = self.beta2*self.v[k]+(1-self.beta2)*(g**2)
            mHat = self.m[k]/(1-self.beta1**self.t)
            vHat = self.v[k]/(1-self.beta2**self.t)
            p-=self.lr*(mHat/(np.sqrt(vHat)+self.eps)+self.weightDecay*p)

#---ReduceLrOnPlateau: cut the step size when validation stops improving---
class ReduceLrOnPlateau:
    def __init__(self,optimizer,factor=0.5,patience=5,minLr=1e-5):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.minLr = minLr
        self.best = np.inf
        self.badEpochs = 0

    def Step(self,metric):
        if metric<self.best:
            self.best = metric
            self.badEpochs = 0
        else:
            self.badEpochs+=1
            if self.badEpochs>self.patience:
                self.optimizer.lr = max(self.minLr,self.optimizer.lr*self.factor)
                self.badEpochs = 0

#---CosineAnnealingLr: ease the step size down a cosine curve over the run---
class CosineAnnealingLr:
    def __init__(self,optimizer,tMax,baseLr=None):
        self.optimizer = optimizer
        self.tMax = tMax
        self.baseLr = baseLr if baseLr is not None else optimizer.lr
        self.epoch = 0

    def Step(self):
        self.epoch+=1
        self.optimizer.lr = 0.5*self.baseLr*(1+np.cos(np.pi*self.epoch/self.tMax))

#----------------------------------------------------------------------
#ClipGradNorm: cap how big one update may be
#if the whole gradient is longer than a limit it is scaled back.  stops
#one bad batch from wrecking weights that were training fine
#----------------------------------------------------------------------
def ClipGradNorm(grads,maxNorm=5.0):
    totalNormSq = sum(float((g**2).sum()) for g in grads.values())
    totalNorm = np.sqrt(totalNormSq)
    if totalNorm>maxNorm:
        scale = maxNorm/(totalNorm+1e-6)
        for k in grads:
            grads[k] = grads[k]*scale
    return totalNorm



#======================================================================
#READING AN IMAGE FILE
#decoding a jpeg or png is file format i/o, the same category as the
#camera interface and the pin interface, so it goes through the library
#the rest of the project already reads frames with.  everything after
#this point sees a plain grey array, which is what segment.SegmentLines
#and ExtractLinePatches want
#======================================================================
#---ReadImageFile: a page on disk as a grey array---
def ReadImageFile(path):
    import cv2
    arr = cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise ValueError("could not read %s (missing, or an unsupported format)" % path)
    return arr


#======================================================================
#GETTING THE DATA READY
#reading pages, finding their lines, and caching the result
#======================================================================
#---EncodeCrop: pack a crop into bytes for the cache---
def EncodeCrop(arr):
    buf = io.BytesIO()
    np.save(buf,np.asarray(arr,dtype=np.uint8),allow_pickle=False)
    return zlib.compress(buf.getvalue(),6)   #6 is the usual size/speed middle

#---DecodeCrop: and unpack it again---
def DecodeCrop(raw):
    return np.load(io.BytesIO(zlib.decompress(raw)),allow_pickle=False)

#---ReadLabelLines: read a page's typed transcript---
def ReadLabelLines(imgPath,labelPath=None):
    p = Path(labelPath) if labelPath else Path(imgPath).with_name(Path(imgPath).stem+"_labels.txt")
    if not p.exists():
        return []
    raw = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [re.sub(r"^\d+\t","",l) for l in raw]   #drop the leading "3<tab>" index

#---ClusterRows: group row numbers that sit together---
def ClusterRows(rows,maxGap=5):
    rows = np.asarray(sorted(rows),dtype=np.int64)
    if rows.size==0:
        return []
    cut = np.nonzero(np.diff(rows)>maxGap)[0]+1
    return [int(g.mean()) for g in np.split(rows,cut) if g.size]

#---FindRuleLineClusters: spot the printed ruling on a form---
def FindRuleLineClusters(binaryImg,rawGrayscaleArray=None):
    h,w = binaryImg.shape
    row = (rawGrayscaleArray<225) if rawGrayscaleArray is not None else (binaryImg>0)
    longest = RowRunLength(row).max(axis=1)     #longest run in each row
    filled = row.sum(axis=1)                       #how much ink in each row
    ruleRows = np.nonzero((longest>w*0.25)&(filled>w*0.10))[0]
    return ClusterRows(ruleRows.tolist())

#---RemoveStraightFormLines: rub that ruling out---
def RemoveStraightFormLines(hwRegion):
    cleaned = hwRegion.copy()
    h,w = cleaned.shape
    row = cleaned>0
    longest = RowRunLength(row).max(axis=1)
    hit = np.nonzero((longest>w*0.25)&(row.sum(axis=1)>w*0.10))[0]
    for r in hit:
        cleaned[max(0,r-2):min(h,r+3),:] = 0
    return cleaned

#---DetectPageBounds: find the sheet inside the photograph---
def DetectPageBounds(binaryImg,rawGrayscaleArray=None):
    h,_w = binaryImg.shape
    clusters = FindRuleLineClusters(binaryImg,rawGrayscaleArray)

    topCandidates = [c for c in clusters if h*0.12<c<h*0.40]
    botCandidates = [c for c in clusters if h*0.55<c<h*0.92]

    topY = topCandidates[-1]+8 if topCandidates else int(h*0.12)
    botY = botCandidates[0]-8 if botCandidates else int(h*0.80)
    return topY,botY

#---MergeUndersizedFragments: join line pieces too small to stand alone---
def MergeUndersizedFragments(lines,minRelativeHeight=0.4):
    if len(lines)<=1:
        return lines

    heights = [e-s for s,e in lines]
    medianHeight = sorted(heights)[len(heights)//2]
    threshold = medianHeight*minRelativeHeight

    merged = list(lines)
    changed = True
    while changed and len(merged)>1:
        changed = False
        for i,(s,e) in enumerate(merged):
            if (e-s)>=threshold:
                continue
            if i==0:
                neighbor = 1
            elif i==len(merged)-1:
                neighbor = i-1
            else:
                gapPrev = merged[i][0]-merged[i-1][1]
                gapNext = merged[i+1][0]-merged[i][1]
                neighbor = i-1 if gapPrev<=gapNext else i+1

            lo,hi = min(i,neighbor),max(i,neighbor)
            combined = (min(merged[lo][0],merged[hi][0]),max(merged[lo][1],merged[hi][1]))
            merged = merged[:lo]+[combined]+merged[hi+1:]
            changed = True
            break

    return merged

#----------------------------------------------------------------------
#MergeOrSplitToExpectedLineCount: make the count match the transcript
#the transcript says how many lines there should be, so if the detector
#found a different number the nearest merge or split is applied
#----------------------------------------------------------------------
def MergeOrSplitToExpectedLineCount(lines,rowInkCount,expectedLineCount):
    if expectedLineCount is None or expectedLineCount<=0 or len(lines)==0:
        return lines

    adjusted = list(lines)

    while len(adjusted)>expectedLineCount:
        gapScores = []
        for i in range(len(adjusted)-1):
            prevS,prevE = adjusted[i]
            currS,currE = adjusted[i+1]
            gap = currS-prevE
            gapInk = np.mean(rowInkCount[prevE:currS]) if currS>prevE else 0
            gapScores.append((gap+gapInk,i))

        _,mergeIdx = min(gapScores)
        adjusted[mergeIdx] = (adjusted[mergeIdx][0],adjusted[mergeIdx+1][1])
        del adjusted[mergeIdx+1]

    while len(adjusted)<expectedLineCount:
        splittable = [(e-s,i,s,e) for i,(s,e) in enumerate(adjusted) if (e-s)>=42]
        if not splittable:
            break

        _,splitIdx,s,e = max(splittable)
        lo,hi = s+18,e-18
        if hi>lo:
            splitY = lo+int(np.argmin(rowInkCount[lo:hi]))
        else:
            splitY = (s+e)//2
        adjusted[splitIdx:splitIdx+1] = [(s,splitY),(splitY,e)]

    return adjusted

#---EstimateLinePitch: the spacing between lines on this page---
def EstimateLinePitch(smoothedRowInk,minPitch=40,maxPitch=400):
    sig = smoothedRowInk-smoothedRowInk.mean()
    autocorr = np.correlate(sig,sig,mode='full')[len(sig)-1:]
    maxPitch = min(maxPitch,len(autocorr))
    if maxPitch<=minPitch:
        return minPitch
    search = autocorr[minPitch:maxPitch]
    return minPitch+int(np.argmax(search))

#---SplitLinesByPeriodicity: split a block that is really several evenly spaced lines---
def SplitLinesByPeriodicity(binaryRegion,ruleLineHints=None,smoothWindow=9,minLineHeight=30):
    rowInk = np.sum(binaryRegion>0,axis=1).astype(np.float32)
    if rowInk.max()<=0:
        return []

    kernel = np.ones(smoothWindow)/smoothWindow
    smoothed = np.convolve(rowInk,kernel,mode='same')

    pitch = EstimateLinePitch(smoothed)
    minDistance = max(minLineHeight,int(pitch*0.55))
    #my own peak finder, prominence rejects a ripple on a big peak's flank
    peaks = FindPeaks(-smoothed,distance=minDistance,
                         prominence=max(1.0,rowInk.max()*0.03))

    if ruleLineHints:
        snapWindow = max(6,int(pitch*0.12))
        peaks = list(peaks)
        for i,p in enumerate(peaks):
            nearby = [r for r in ruleLineHints if abs(r-p)<=snapWindow]
            if nearby:
                peaks[i] = min(nearby,key=lambda r: abs(r-p))
        peaks = np.array(sorted(set(peaks)))

    boundaries = [0]+list(peaks)+[len(rowInk)]
    lines = [(boundaries[i],boundaries[i+1]) for i in range(len(boundaries)-1)]
    #drop segments with no ink at all
    lines = [(s,e) for s,e in lines if np.sum(rowInk[s:e])>0]
    return lines

#---ExtractLinePatches: cut the page into one image per line---
def ExtractLinePatches(page,targetHeight=32,maxWidth=1024,expectedLineCount=None,labelLines=None,
                        isDataset=True):
    #page is a grey array.  loading it is the caller's job, see testing/
    RawGrayscaleArray = np.asarray(page,dtype=np.uint8)
    if RawGrayscaleArray.ndim==3:
        RawGrayscaleArray = np.rint(RawGrayscaleArray[...,:3].mean(2)).astype(np.uint8)
    h,w = RawGrayscaleArray.shape

    BinaryInvertedImage = ((RawGrayscaleArray<Otsu(RawGrayscaleArray)).astype(np.uint8)*255)

    if not isDataset:
        #no IAM-style printed prompt box to bound against -- use almost the
        #whole page, trimming a small margin for scan/photo edge artifacts
        margin = int(h*0.01)
        topY,botY = margin,h-margin
    else:
        topY,botY = DetectPageBounds(BinaryInvertedImage,RawGrayscaleArray)

    #handwriting extraction starts this many px below topY, a hard buffer
    #so a rule-line detection off by a few px can't pull header pixels into
    #a line crop (at the negligible cost of a sliver off the true first line)
    HANDWRITING_TOP_MARGIN = 15
    handwritingStartY = min(botY,topY+HANDWRITING_TOP_MARGIN)

    hwRegion = BinaryInvertedImage[handwritingStartY:botY,:].copy()
    regionH,regionW = hwRegion.shape

    hwRegion = RemoveStraightFormLines(hwRegion)

    #mask left vertical margin lines and edge artifacts
    colInkHeights = np.sum(hwRegion>0,axis=0)
    vertLineCols = np.where(colInkHeights>regionH*0.40)[0]
    hwRegion[:,vertLineCols] = 0
    hwRegion[:,:int(w*0.07)] = 0
    hwRegion[:,int(w*0.97):] = 0

    rowInkCount = np.sum(hwRegion>0,axis=1)

    if not isDataset:
        #densely-written page -- periodicity splitter, blank rows aren't safe to assume
        mergedLines = SplitLinesByPeriodicity(hwRegion)
        mergedLines = MergeUndersizedFragments(mergedLines)
    else:
        hasInk = rowInkCount>max(10,int(w*0.006))

        pad1D = np.pad(hasInk,(3,3),mode='constant')
        smoothedInk = np.zeros_like(hasInk)
        for dy in range(7):
            smoothedInk = np.logical_or(smoothedInk,pad1D[dy:dy+len(hasInk)])

        #pad a zero each end then diff, so a run start shows as +1 and an end
        #as -1 even when the run touches the first or last row
        diff = np.diff(np.concatenate(([0],smoothedInk.astype(np.int8),[0])))
        lineStarts,lineEnds = np.where(diff==1)[0],np.where(diff==-1)[0]

        rawLines = [(s,e) for s,e in zip(lineStarts,lineEnds) if (e-s)>=8]

        #merge descender fragments ('y','p','g')
        mergedLines = []
        for line in rawLines:
            if not mergedLines:
                mergedLines.append(line)
            else:
                prevS,prevE = mergedLines[-1]
                currS,currE = line
                if (currS-prevE)<18:
                    mergedLines[-1] = (prevS,currE)
                else:
                    mergedLines.append(line)

        mergedLines = MergeUndersizedFragments(mergedLines)
        mergedLines = MergeOrSplitToExpectedLineCount(mergedLines,rowInkCount,expectedLineCount)

    #the cut positions go back raw so testing/ can draw a debug overlay with
    #a library if it wants one, nothing here draws anything
    cutRows = dict(topY=topY,handwritingStartY=handwritingStartY,botY=botY)

    ExtractedLineSamples = []

    for idx,(s,e) in enumerate(mergedLines):
        #offset from handwritingStartY (not topY), since hwRegion/s/e start there
        absY1 = handwritingStartY+max(0,s-5)
        absY2 = handwritingStartY+min(regionH,e+5)
        cropH = absY2-absY1

        lineBinary = hwRegion[max(0,s-3):min(regionH,e+3),:]
        colSum = np.sum(lineBinary>0,axis=0)
        inkCols = np.where(colSum>0)[0]
        if len(inkCols)==0: continue

        absX1 = max(0,inkCols[0]-8)
        absX2 = min(w,inkCols[-1]+8)
        cropW = absX2-absX1

        #numbered red box on the preview, matching this line's returned index

        rawLineCrop = RawGrayscaleArray[absY1:absY2,absX1:absX2]
        scaleRatio = targetHeight/float(cropH)
        newWidth = min(maxWidth,max(1,int(cropW*scaleRatio)))

        #linear not lanczos: lanczos overshoots at a hard ink edge and leaves
        #a pale halo beside every stroke, which then reads as ink to the
        #threshold further down the pipeline
        resized = Resize(rawLineCrop,newWidth,targetHeight,'linear')

        #pad to a fixed width with white, left aligned.  one array fill and
        #one slice assignment, no canvas object needed
        paddedCanvas = np.full((targetHeight,maxWidth),255,np.uint8)
        paddedCanvas[:,:newWidth] = resized

        ExtractedLineSamples.append({
            'line_idx': idx,
            'raw_crop': rawLineCrop,
            'processed_patch': paddedCanvas,
            'label_text': labelLines[idx] if labelLines is not None and idx<len(labelLines) else ""
        })

    #the transcript is returned alongside the crops only when the two agree on
    #line count, so a caller can never pair n crops with a different number of
    #label lines by index
    formattedTextLines = (labelLines if labelLines is not None
                          and len(labelLines)==len(ExtractedLineSamples)
                          else [""]*len(ExtractedLineSamples))

    #cutRows replaces the old debug canvas, testing/ draws it if wanted
    return ExtractedLineSamples,formattedTextLines,cutRows

#---EncodeText: a transcript line into label numbers---
def EncodeText(text):
    return np.array([CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX],dtype=np.int64)

#----------------------------------------------------------------------
#FindPeaks: peaks in a profile, with a prominence test
#a peak counts only if you have to come down by a certain amount before
#you can climb higher, which is what stops a bumpy slope being read as
#a dozen separate lines
#----------------------------------------------------------------------
def FindPeaks(sm,floorFrac=0.0,minGap=0,prominence=None,distance=None):
    sm = np.asarray(sm,dtype=np.float64)
    floor = floorFrac*sm.max() if floorFrac>0 else -np.inf   #must not default to 0, valleys are negative
    peaks = [i for i in range(1,len(sm)-1)
             if sm[i]>=sm[i-1] and sm[i]>sm[i+1] and sm[i]>floor]   #>= then >, so a flat top counts once

    if prominence is not None:
        keep = []
        for i in peaks:
            j,lo = i-1,sm[i]
            while j>=0 and sm[j]<=sm[i]:   #walk left until the profile rises above this peak
                lo = min(lo,sm[j]); j-=1
            leftLow = lo
            j,lo = i+1,sm[i]
            while j<len(sm) and sm[j]<=sm[i]:   #and the same to the right
                lo = min(lo,sm[j]); j+=1
            if sm[i]-max(leftLow,lo)>=prominence:   #prominence = peak - the higher of the two lows
                keep.append(i)
        peaks = keep   #this is what rejects a ripple riding on a big peak's flank

    gap = distance if distance is not None else minGap
    if not gap or gap<=0:
        return peaks
    kept = []
    for i in sorted(peaks,key=lambda j:-sm[j]):   #tallest first
        if all(abs(i-j)>=gap for j in kept):   #refuse any peak too close to one already kept
            kept.append(i)
    return sorted(kept)



#======================================================================
#AUGMENTATION
#the same handwriting, plausibly written again, so the network generalises
#======================================================================
#---MyAffine: shift, rotate, scale and shear in one go---
def MyAffine(img,angleDeg=0.0,shearDeg=0.0,fillValue=255):
    a = np.asarray(img,dtype=np.uint8)
    h,w = a.shape
    cx,cy = w/2.0,h/2.0
    ang,sh = np.radians(angleDeg),np.radians(shearDeg)
    R = np.array([[np.cos(ang),-np.sin(ang),0],[np.sin(ang),np.cos(ang),0],[0,0,1]])   #rotation, homogeneous 3x3
    Sh = np.array([[1,np.tan(sh),0],[0,1,0],[0,0,1]])
    T1 = np.array([[1,0,-cx],[0,1,-cy],[0,0,1]])
    T2 = np.array([[1,0,cx],[0,1,cy],[0,0,1]])
    Mi = np.linalg.inv(T2@R@Sh@T1)
    return Warp(a,(Mi[0,0],Mi[0,1],Mi[0,2],Mi[1,0],Mi[1,1],Mi[1,2]),fill=fillValue)

#---AugmentShear: lean the writing a bit more or less---
def AugmentShear(img):
    k = random.uniform(-0.6,0.6)
    shearDeg = math.degrees(math.atan(k))
    return MyAffine(img,angleDeg=0.0,shearDeg=shearDeg)

#---AugmentRotate: tilt the line slightly---
def AugmentRotate(img):
    angle = random.uniform(-2.5,2.5)
    return MyAffine(img,angleDeg=angle,shearDeg=0.0)

#----------------------------------------------------------------------
#FindPerspectiveCoeffs: the eight numbers for a corner to corner map
#solves a small linear system so the four corners land where asked,
#which is what a photograph taken at an angle does to a page
#----------------------------------------------------------------------
def FindPerspectiveCoeffs(startpoints,endpoints):
    matrix = []
    for (x,y),(X,Y) in zip(endpoints,startpoints):
        matrix.append([x,y,1,0,0,0,-X*x,-X*y])
        matrix.append([0,0,0,x,y,1,-Y*x,-Y*y])
    A = np.array(matrix,dtype=np.float64)
    B = np.array(startpoints,dtype=np.float64).reshape(8)
    return tuple(np.linalg.solve(A,B))

#---AugmentPerspective: apply that, as if photographed off square---
def AugmentPerspective(img):
    #arrays are (height,width), not (width,height)
    h,w = np.asarray(img).shape[:2]
    DistortionScale = random.uniform(0.03,0.08)
    startpoints,endpoints = PerspectiveJitterPoints(w,h,DistortionScale)
    coeffs = FindPerspectiveCoeffs(startpoints,endpoints)
    #same Warp, 8 coeffs instead of 6 so it divides by a varying third row
    #8 coefficients, so Warp divides by the varying third row
    return Warp(np.asarray(img,dtype=np.uint8),coeffs,fill=255)

#---PerspectiveJitterPoints: pick slightly moved corners to map to---
def PerspectiveJitterPoints(width,height,DistortionScale):
    halfW,halfH = width//2,height//2
    topleft = (0,0)
    topright = (width-1,0)
    botright = (width-1,height-1)
    botleft = (0,height-1)
    startpoints = [topleft,topright,botright,botleft]

    def Jitter(x,y,maxDx,maxDy):
        return (
            int(x+random.uniform(-maxDx,maxDx)),
            int(y+random.uniform(-maxDy,maxDy)),
        )

    maxDx = DistortionScale*halfW
    maxDy = DistortionScale*halfH
    endpoints = [
        Jitter(*topleft,maxDx,maxDy),
        Jitter(*topright,maxDx,maxDy),
        Jitter(*botright,maxDx,maxDy),
        Jitter(*botleft,maxDx,maxDy),
    ]
    return startpoints,endpoints

#---AugmentElastic: bend the strokes locally---
def AugmentElastic(img):
    sigma = random.choice([3,4])
    alpha = random.choice([15,20])
    rng = np.random.default_rng(random.randrange(1<<30))
    return Elastic(np.asarray(img,dtype=np.uint8),sigma,alpha,rng)

#---AugmentLineImage: pick some of the above at random for one crop---
def AugmentLineImage(pilImg):
    candidates = [fn for fn in AUGMENTATIONS if random.random()<0.5]
    if not candidates:
        return pilImg
    return random.choice(candidates)(pilImg)

#---AUGMENTATIONS: which augmentations are on, and how strong---
AUGMENTATIONS = [AugmentShear,AugmentRotate,AugmentElastic,AugmentPerspective]



#======================================================================
#THE DATASETS
#where crops and labels come from, and how they are batched
#======================================================================
#---LineImageCache: keep extracted line crops on disk so a rerun is quick---
class LineImageCache:
    def __init__(self,rootDir,cacheDir="line_image_cache",ForceRebuild=False,
                 SaveEveryNPages=10,maxPages=None):
        self.cacheDir = Path(cacheDir)
        self.shardsDir = self.cacheDir/"shards"
        self.manifestPath = self.cacheDir/"manifest.pkl"
        self.shardsDir.mkdir(parents=True,exist_ok=True)

        #page_key ("author/page.png") -> list of PNG bytes, one per detected line
        self.pages = {}
        ProcessedPages = set()
        ShardFiles = []

        if self.manifestPath.exists() and not ForceRebuild:
            print(f"[ImageCache] Loading manifest from: {self.manifestPath}")
            try:
                with open(self.manifestPath,"rb") as f:
                    manifest = pickle.load(f)
                ProcessedPages = set(manifest.get("processed_pages",[]))
                ShardFiles = manifest.get("shard_files",[])
                for shardName in ShardFiles:
                    shardPath = self.shardsDir/shardName
                    if shardPath.exists():
                        with open(shardPath,"rb") as f:
                            self.pages.update(pickle.load(f))
                print(f"[ImageCache] Loaded images for {len(self.pages)} page(s) from {len(ShardFiles)} shard file(s).")
            except Exception as e:
                #an old format cache or a corrupt one, either way rebuilding is
                #the only safe recovery, so do not propagate this
                print(f"[ImageCache] Existing cache unreadable ({e}) -- rebuilding from scratch.")
                ProcessedPages,ShardFiles,self.pages = set(),[],{}

        def SaveManifest():
            tmpPath = self.manifestPath.with_suffix(".tmp")
            with open(tmpPath,"wb") as f:
                pickle.dump({"processed_pages": list(ProcessedPages),"shard_files": ShardFiles},f)
            os.replace(tmpPath,self.manifestPath)

        #writes only newly-collected pages into their own shard file -- previously
        #written shards are never touched again
        def FlushPending(PendingPages):
            if not PendingPages:
                return
            shardName = f"shard_{len(ShardFiles):06d}.pkl"
            shardPath = self.shardsDir/shardName
            tmpPath = shardPath.with_suffix(".tmp")
            with open(tmpPath,"wb") as f:
                pickle.dump(PendingPages,f)
            os.replace(tmpPath,shardPath)
            ShardFiles.append(shardName)
            SaveManifest()
            print(f"\n[ImageCache Checkpoint] Flushed {len(PendingPages)} new page(s) to {shardName} "
                  f"({len(self.pages)} pages cached total).")

        AuthorFolders = sorted(p.name for p in rootDir.iterdir() if p.is_dir())
        allPages = []
        for authorId in AuthorFolders:
            authorDir = rootDir/authorId
            for imgPath in sorted(authorDir.glob("*.png")):
                allPages.append((authorId,imgPath))

        if maxPages is not None and maxPages>0:
            allPages = allPages[:maxPages]

        TotalFiles = len(allPages)
        PageCounter = 0
        PendingPages = {}
        try:
            for idx,(authorId,imgPath) in enumerate(allPages,1):
                pageKey = f"{authorId}/{imgPath.name}"
                if pageKey in ProcessedPages:
                    continue

                pct = (idx/TotalFiles)*100.0 if TotalFiles>0 else 0.0
                print(f"\r[ImageCache Preprocessing] Progress: {pct:5.1f}% ({idx}/{TotalFiles} pages)",end="",flush=True)

                try:
                    #expectedLineCount/labelLines=None: segmentation stands on its
                    #own, never shaped by what a particular label file says
                    LineSamples,_,_ = ExtractLinePatches(
                        ReadImageFile(imgPath),
                        targetHeight=32,
                        maxWidth=1024,
                        expectedLineCount=None,
                        labelLines=None,
                    )
                except Exception as e:
                    print(f"\n[ImageCache] Skipping {pageKey} (segmentation error: {e})")
                    LineSamples = []

                pngList = []
                for sample in LineSamples:
                    pilLine = np.asarray(sample["raw_crop"],dtype=np.uint8)
                    pngList.append(EncodeCrop(pilLine))

                self.pages[pageKey] = pngList
                PendingPages[pageKey] = pngList
                ProcessedPages.add(pageKey)
                PageCounter+=1
                if PageCounter%SaveEveryNPages==0:
                    FlushPending(PendingPages)
                    PendingPages = {}

            if PageCounter>0:
                print()
        except KeyboardInterrupt:
            print("\n[Paused] Interrupted! Flushing pending page images before exiting.")
            FlushPending(PendingPages)
            raise

        #catches whatever's left from the last incomplete save_every_n_pages batch
        if PendingPages:
            FlushPending(PendingPages)

        print(f"[ImageCache] Total pages with cached images: {len(self.pages)}")

    def get(self,pageKey):
        return self.pages.get(pageKey)

#----------------------------------------------------------------------
#IAMLineDatasetRaw: the public database, line by line
#walks the pages, pairs each line with its transcript, and marks which
#ones are held out.  held out lines never reach training, so a score on
#them means something
#----------------------------------------------------------------------
class IAMLineDatasetRaw:
    def __init__(
        self,
        rootDir,
        MinPagesForHoldout=3,
        cacheDir="line_image_cache",
        ForceRebuild=False,
        SaveEveryNPages=10,
        maxPages=None,
        isTrain=True,
    ):
        self.samples = []
        self.isTrain = isTrain

        rootDir = self.ResolveDatasetRoot(Path(rootDir))

        ImageCache = LineImageCache(
            rootDir,
            cacheDir=cacheDir,
            ForceRebuild=ForceRebuild,
            SaveEveryNPages=SaveEveryNPages,
            maxPages=maxPages,
        )

        self.authorFolders = sorted(p.name for p in rootDir.iterdir() if p.is_dir())
        self.authorToIdx = {author: idx for idx,author in enumerate(self.authorFolders)}

        MismatchedPages = 0
        OversizedLabels = 0
        for authorId in self.authorFolders:
            authorDir = rootDir/authorId
            AuthorPages = sorted(authorDir.glob("*.png"))
            if not AuthorPages:
                continue
            LabeledPages = [p for p in AuthorPages if ReadLabelLines(str(p))]
            HoldoutPageName = LabeledPages[-1].name if len(LabeledPages)>=MinPagesForHoldout else None

            for imgPath in AuthorPages:
                pageKey = f"{authorId}/{imgPath.name}"
                LabelLines = ReadLabelLines(str(imgPath))
                if not LabelLines:
                    continue

                pngList = ImageCache.get(pageKey)
                if pngList is None:
                    continue   #not in the image cache (e.g. maxPages truncation)

                if len(pngList)!=len(LabelLines):
                    MismatchedPages+=1
                    print(f"[Dataset] Skipping {pageKey}: cached image has {len(pngList)} line(s) but "
                          f"the current label has {len(LabelLines)} line(s) -- segmentation and label "
                          f"disagree on line count for this page. This page's images need re-cropping "
                          f"(delete/rebuild line_image_cache/ for it) before it can be used.")
                    continue

                isHoldout = imgPath.name==HoldoutPageName
                for lineIdx in range(len(LabelLines)):
                    text = LabelLines[lineIdx]
                    target = EncodeText(text)
                    if len(target)==0:
                        continue
                    #this long is two merged lines, and ctc has no room for it
                    #anyway, so skip it loudly
                    if len(target)>MAX_SAFE_LABEL_CHARS:
                        OversizedLabels+=1
                        print(f"[Dataset] Skipping oversized line label ({len(target)} chars, cap is "
                              f"{MAX_SAFE_LABEL_CHARS}) at {pageKey} line {lineIdx} -- likely merged/"
                              f"collapsed line segmentation on this page, not a real single line: "
                              f"{text[:80]!r}...")
                        continue
                    self.samples.append({
                        "page_key": pageKey,
                        "line_idx": lineIdx,
                        "imageBytes": pngList[lineIdx],
                        "target": target,
                        "text": text,
                        "is_holdout": isHoldout,
                    })

        if MismatchedPages:
            print(f"[Dataset] {MismatchedPages} page(s) skipped due to image/label line-count mismatch.")
        if OversizedLabels:
            print(f"[Dataset] {OversizedLabels} line(s) skipped as oversized (likely collapsed/merged "
                  f"line segmentation -- see warnings above for which pages).")
        print(f"[Dataset] Total line samples: {len(self.samples)}")

    #some dataset roots nest the real author folders one level down in "data/"
    @staticmethod
    def ResolveDatasetRoot(rootDir):
        dataDir = rootDir/"data"
        return dataDir if dataDir.exists() and any(p.is_dir() for p in dataDir.iterdir()) else rootDir

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,idx):
        item = self.samples[idx]
        pilImg = ResizeLineImageFixed(DecodeCrop(item["imageBytes"]))
        if self.isTrain:
            pilImg = AugmentLineImage(pilImg)
        imgTensor = TensorFromResized(pilImg)
        return imgTensor,item["target"],item["text"]

    def HoldoutSplitIndices(self):
        trainIdx = [i for i,s in enumerate(self.samples) if not s["is_holdout"]]
        testIdx = [i for i,s in enumerate(self.samples) if s["is_holdout"]]
        return trainIdx,testIdx

#---TrainEvalView: the same dataset with augmentation on or off---
class TrainEvalView:
    def __init__(self,BaseDataset,indices,isTrain):
        self.baseDataset = BaseDataset
        self.indices = indices
        self.isTrain = isTrain

    def __len__(self):
        return len(self.indices)

    def __getitem__(self,i):
        item = self.baseDataset.samples[self.indices[i]]
        pilImg = ResizeLineImageFixed(DecodeCrop(item["imageBytes"]))
        if self.isTrain:
            pilImg = AugmentLineImage(pilImg)
        imgTensor = TensorFromResized(pilImg)
        return imgTensor,item["target"],item["text"]

#---CollateFn: pad a batch of different width crops to one size---
def TextCollateFn(batch):
    images,targets,texts = zip(*batch)
    TargetLengths = np.array([len(t) for t in targets],dtype=np.int64)
    FlatTargets = np.concatenate(targets)
    #each crop arrives shaped (1,1,H,W) for single crop inference, so they are
    #joined along that first axis rather than stacked behind a new one
    return np.concatenate(images,0),FlatTargets,TargetLengths,list(texts)

#---AuthorLabeledView: the same crops labelled by writer instead of by text---
class AuthorLabeledView:
    def __init__(self,base: IAMLineDatasetRaw,indices,isTrain):
        self.base = base
        self.indices = indices
        self.isTrain = isTrain

    def __len__(self):
        return len(self.indices)

    def __getitem__(self,i):
        item = self.base.samples[self.indices[i]]
        gray = DecodeCrop(item["imageBytes"])
        normImg = StrokeNormalize(np.array(gray))
        if normImg is not None:
            gray = normImg   #falls back to the raw crop only if stroke-normalize
                              #found no usable ink (empty/near-blank line)
        pilImg = ResizeLineImageFixed(gray)
        if self.isTrain:
            pilImg = AugmentLineImage(pilImg)
        imgTensor = TensorFromResized(pilImg)
        authorId = item["page_key"].split("/")[0]
        authorIdx = self.base.author_to_idx[authorId]
        return imgTensor,authorIdx,item["page_key"]

#---CollateFn (writer version): batch crops for the writer namer---
def AuthorCollateFn(batch):
    images,authorIdxs,pageKeys = zip(*batch)
    return np.concatenate(images,0),np.array(authorIdxs,dtype=np.int64),list(pageKeys)

#---AddPersonalSamples: mix the photographed writers in with the public ones---
def AddPersonalSamples(base):
    #cutting these pages up costs a couple of minutes, and the result only
    #changes when a page or its transcript changes, so it is kept on disk and
    #keyed by the modification times of every page and label file involved
    cachePath = paths.Cache/"personal_lines_cache.pkl"
    #keyed on the file NAME and size, plus a digest of the transcript, rather
    #than on absolute paths and modification times.  a path differs between
    #machines and a modification time changes whenever the files are copied or
    #synced, either of which would throw a perfectly good cache away
    stamp = []
    for author in paths.PersonalWriters:
        for imgPath,labelPath in paths.PersonalWriterPages(author):
            imgSize = imgPath.stat().st_size if imgPath.exists() else 0
            digest = (hashlib.sha1(labelPath.read_bytes()).hexdigest()
                      if labelPath.exists() else "")
            stamp.append((imgPath.name,imgSize,digest))
    key = tuple(sorted(stamp))

    if cachePath.exists():
        try:
            with open(cachePath,"rb") as f:
                cached = pickle.load(f)
            if cached.get("key")==key:
                base.samples.extend(cached["samples"])
                print(f"[Personal] {len(cached['samples'])} line samples read from the cache",
                      flush=True)
                return
            print("[Personal] pages changed since the cache was written, cutting again",flush=True)
        except Exception as e:
            print(f"[Personal] cache unreadable ({e}), cutting again",flush=True)

    added = []
    rng = random.Random(paths.SplitSeed)
    nAdded = 0
    print(f"[Personal] cutting pages for {paths.PersonalWriters} into lines",flush=True)
    for author in paths.PersonalWriters:
        for imgPath,labelPath in paths.PersonalWriterPages(author):
            print(f"[Personal] {imgPath.name} ...",end="",flush=True)
            leanLines,_ov,_info = SL.SegmentLines(ReadImageFile(imgPath))
            crops = [np.asarray(ln["image"]) for ln in leanLines]
            gt = ReadLabelLines(str(imgPath),str(labelPath))
            gt = [g for g in gt if g.strip()!="MESS"]
            n = min(len(crops),len(gt))
            rows = list(zip(range(n),crops[:n],gt[:n]))
            rng.shuffle(rows)
            nVal = max(1,round(len(rows)*paths.ValFraction))
            for rank,(origIdx,crop,text) in enumerate(rows):
                target = EncodeText(text)
                if len(target)==0:
                    continue
                isHoldout = rank<nVal
                pageKey = f"{author}/{imgPath.stem}_{origIdx:02d}.png"
                sample = {
                    "page_key": pageKey,
                    "line_idx": origIdx,
                    "imageBytes": EncodeCrop(crop),
                    "target": target,
                    "text": text,
                    "is_holdout": isHoldout,
                }
                base.samples.append(sample)
                added.append(sample)
                nAdded+=1
    print(f"[Personal] added {nAdded} line samples from {len(paths.PersonalWriters)} writers")

    cachePath.parent.mkdir(parents=True,exist_ok=True)
    tmp = cachePath.with_suffix(".tmp")
    with open(tmp,"wb") as f:                       #written aside then moved, so an
        pickle.dump({"key":key,"samples":added},f)  #interrupted write cannot leave a
    tmp.replace(cachePath)                          #half finished cache behind
    print(f"[Personal] cached to {cachePath.name} for next time",flush=True)

#---TekliaLineDataset: the second public source, same interface---
class TekliaLineDataset:
    def __init__(self,rootDir,isTrain):
        rootDir = Path(rootDir)
        self.isTrain = isTrain
        self.items = []
        #pairing thousands of files with their labels takes a minute or so and
        #used to say nothing at all, which is indistinguishable from a hang
        print(f"[Teklia] indexing {rootDir.name} ...",end="",flush=True)
        for imgPath in sorted(rootDir.glob("*.png")):
            labelPath = imgPath.with_name(imgPath.stem+"_labels.txt")
            if labelPath.exists():
                text = labelPath.read_text(encoding="utf-8").strip()
                if text:
                    self.items.append((imgPath,text))
        print(f" {len(self.items)} lines",flush=True)
        if not self.items:
            raise RuntimeError(f"No line images found under {rootDir} -- run "
                                f"ExportTekliaLines.py first.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self,i):
        imgPath,text = self.items[i]
        pilImg = ResizeLineImageFixed(ReadImageFile(imgPath))
        if self.isTrain:
            pilImg = AugmentLineImage(pilImg)
        imgTensor = TensorFromResized(pilImg)
        return imgTensor,EncodeText(text),text

#---MAX_SAFE_LABEL_CHARS: longest label CTC can fit in the available columns---
MAX_SAFE_LABEL_CHARS = int((INPUT_WIDTH//4)*0.75)

#---HIDDEN_SIZE: how much memory the lstm carries---
HIDDEN_SIZE = 256            #lstm width per direction

#---WEIGHTS_DIR: where checkpoints are written---
WEIGHTS_DIR = paths.Weights

#---DEFAULT_TEXT_CACHE_DIR: cache for the reader's crops---
DEFAULT_TEXT_CACHE_DIR = paths.Cache/"line_image_cache"

#---DEFAULT_AUTHOR_CACHE_DIR: cache for the namer's crops---
DEFAULT_AUTHOR_CACHE_DIR = paths.Cache/"line_cache_authors10"

#---DEFAULT_AUTHOR_DATA_DIR: where the namer's pages live---
DEFAULT_AUTHOR_DATA_DIR = paths.Data.parents[1]/"Data"/"Datasets"/"IAMpages10"

#---DEFAULT_TEXT_DATA_DIR: where the reader's pages live---
DEFAULT_TEXT_DATA_DIR = DEFAULT_AUTHOR_DATA_DIR



#======================================================================
#STARTING WEIGHTS, AND MOVING THEM AROUND
#how a fresh network is initialised, saved, and loaded again
#======================================================================
#----------------------------------------------------------------------
#InitConv: random starting weights for a convolution
#scaled by how many inputs feed each output, so the signal neither dies
#nor explodes as it passes through the layers
#----------------------------------------------------------------------
def InitConv(rng,outCh,inCh,kh,kw):
    fanIn = inCh*kh*kw
    std = np.sqrt(2.0/fanIn)   #kaiming normal, matches nn.Conv2d's relu-gain default
    w = rng.normal(0.0,std,size=(outCh,inCh,kh,kw)).astype(np.float32)
    b = np.zeros(outCh,dtype=np.float32)
    return w,b

#---InitBatchNorm: scale at one, shift at zero, running stats empty---
def InitBatchNorm(ch):
    return (np.ones(ch,dtype=np.float32),np.zeros(ch,dtype=np.float32),
            np.zeros(ch,dtype=np.float32),np.ones(ch,dtype=np.float32))

#---InitLinear: starting weights for a plain layer---
def InitLinear(rng,outF,inF):
    bound = 1.0/np.sqrt(inF)
    w = rng.uniform(-bound,bound,size=(outF,inF)).astype(np.float32)
    b = rng.uniform(-bound,bound,size=(outF,)).astype(np.float32)
    return w,b

#---InitLstmDirection: starting weights for one lstm direction---
def InitLstmDirection(rng,inputSize,hiddenSize):
    bound = 1.0/np.sqrt(hiddenSize)   #matches nn.LSTM's own uniform init range
    wIh = rng.uniform(-bound,bound,size=(4*hiddenSize,inputSize)).astype(np.float32)
    wHh = rng.uniform(-bound,bound,size=(4*hiddenSize,hiddenSize)).astype(np.float32)
    bIh = rng.uniform(-bound,bound,size=(4*hiddenSize,)).astype(np.float32)
    bHh = rng.uniform(-bound,bound,size=(4*hiddenSize,)).astype(np.float32)
    return wIh,wHh,bIh,bHh

#---InitConvBlock: one convolution block's weights---
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

#---InitBackboneParams: the whole shared backbone---
def InitBackboneParams(rng,params):
    InitConvBlock(rng,params,"stage1",1,32,2)
    InitConvBlock(rng,params,"stage2",32,64,4)
    InitConvBlock(rng,params,"stage3",64,128,6)

#---TrainableParams: every array the optimiser is allowed to touch---
def TrainableParams(params):
    return {k:v for k,v in params.items() if not k.endswith("running_mean") and not k.endswith("running_var")}

#---SaveCheckpoint: write the weights and the run state to an npz---
def SaveCheckpoint(path,params,extra=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    payload = dict(params)
    if extra:
        payload.update(extra)
    np.savez(str(path),**payload)

#---LoadCheckpointMeta: read back what a checkpoint says about itself---
def LoadCheckpointMeta(path):
    with np.load(path,allow_pickle=False) as npz:
        epoch = int(npz["__epoch__"]) if "__epoch__" in npz.files else 0
        best = float(npz["__best__"]) if "__best__" in npz.files else None
    return epoch,best

#---LoadParamsInto: load saved weights into a built network---
def LoadParamsInto(params,checkpointPath):
    sd,_ = CLS.LoadStateDictAny(checkpointPath)
    loaded = 0
    for k in params:
        if k in sd and sd[k].shape==params[k].shape:
            params[k] = sd[k].astype(np.float32)
            loaded+=1
    return loaded

#---AppendLog: add one line to the training log---
def AppendLog(path,line):
    path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,"a",encoding="utf-8") as f:
        f.write(line+"\n")



#======================================================================
#THE SHARED BACKBONE
#both networks start with the same stack of convolutions
#======================================================================
#---BackboneForwardTrain: run the backbone, keeping what backward needs---
def BackboneForwardTrain(x,params):
    x,c1 = ConvBlockForwardTrain(x,params,"stage1",2)
    x,poolCache1 = MaxPool2dForward(x,2,2)
    x,c2 = ConvBlockForwardTrain(x,params,"stage2",4)
    x,poolCache2 = MaxPool2dForward(x,2,2)
    x,c3 = ConvBlockForwardTrain(x,params,"stage3",6)
    cache = dict(c1=c1,c2=c2,c3=c3,poolCache1=poolCache1,poolCache2=poolCache2)
    return x,cache

#---BackboneBackwardTrain: gradients back down through it---
def BackboneBackwardTrain(dx,cache):
    dx,g3 = ConvBlockBackwardTrain(dx,cache["c3"],"stage3",6)
    dx = MaxPool2dBackward(dx,cache["poolCache2"])
    dx,g2 = ConvBlockBackwardTrain(dx,cache["c2"],"stage2",4)
    dx = MaxPool2dBackward(dx,cache["poolCache1"])
    dx,g1 = ConvBlockBackwardTrain(dx,cache["c1"],"stage1",2)
    grads = {}
    grads.update(g1); grads.update(g2); grads.update(g3)
    return dx,grads

#---EvalBackboneForward: the same forward, no saved values, for scoring---
def EvalBackboneForward(params,x):
    sd = params
    x = CLS.ConvBlockForward(x,sd,"stage1",2)
    x = CLS.MaxPool2d(x,2,2)
    x = CLS.ConvBlockForward(x,sd,"stage2",4)
    x = CLS.MaxPool2d(x,2,2)
    x = CLS.ConvBlockForward(x,sd,"stage3",6)
    return x

#---IterateBatches: hand out shuffled batches for one pass over the data---
def IterateBatches(dataset,batchSize,rng,shuffle):
    n = len(dataset)
    order = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0,n,batchSize):
        idxs = order[start:start+batchSize]
        yield [dataset[int(i)] for i in idxs]



#======================================================================
#TRAINING THE READER
#the convolution, lstm and CTC network that turns a crop into text
#======================================================================
#---InitTextModelParams: build a fresh reader---
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

#---BuildLstmLayerParams: assemble the lstm stack's weights---
def BuildLstmLayerParams(sd):
    layerParams = []
    for l in range(2):
        fwd = dict(weightIh=sd[f"sequence.weight_ih_l{l}"],weightHh=sd[f"sequence.weight_hh_l{l}"],
                   biasIh=sd[f"sequence.bias_ih_l{l}"],biasHh=sd[f"sequence.bias_hh_l{l}"])
        bwd = dict(weightIh=sd[f"sequence.weight_ih_l{l}_reverse"],weightHh=sd[f"sequence.weight_hh_l{l}_reverse"],
                   biasIh=sd[f"sequence.bias_ih_l{l}_reverse"],biasHh=sd[f"sequence.bias_hh_l{l}_reverse"])
        layerParams.append({"fwd":fwd,"bwd":bwd})
    return layerParams

#---TextModelForward: backbone, lstm, scores per column---
def TextModelForward(x,params):
    backboneOut,backboneCache = BackboneForwardTrain(x,params)
    pooled,hpCache = AdaptiveMaxPoolHeightForward(backboneOut)   #(B,128,1,W)
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
    seqOut,lstmCaches = MultilayerBidirectionalLstmForwardTrain(seqIn,layerParams,HIDDEN_SIZE)

    rawLogits,linCache = LinearForward(seqOut,params["text_head.weight"],params["text_head.bias"])
    cache = dict(backboneCache=backboneCache,hpCache=hpCache,lstmCaches=lstmCaches,linCache=linCache,bchw=(B,C,W))
    return rawLogits,cache

#---TextModelBackward: and all the way back again---
def TextModelBackward(dRawLogits,cache):
    grads = {}
    dSeqOut,dw,db = LinearBackward(dRawLogits,cache["linCache"])
    grads["text_head.weight"] = dw
    grads["text_head.bias"] = db

    dSeqIn,gradsPerLayer = MultilayerBidirectionalLstmBackward(dSeqOut,cache["lstmCaches"])
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
    dBackboneOut = AdaptiveMaxPoolHeightBackward(dPooled,cache["hpCache"])
    dx,backboneGrads = BackboneBackwardTrain(dBackboneOut,cache["backboneCache"])
    grads.update(backboneGrads)
    return dx,grads

#---EvalTextForward: forward only, for validation---
def EvalTextForward(params,x):
    sd = params
    x = EvalBackboneForward(sd,x)
    x = CLS.AdaptiveMaxPoolHeight(x)
    x = x[:,:,0,:]
    x = np.transpose(x,(2,0,1))
    seq = CLS.MultilayerBidirectionalLstm(x,BuildLstmLayerParams(sd),HIDDEN_SIZE)
    return CLS.Linear(seq,sd["text_head.weight"],sd["text_head.bias"])

#---TextNumpyBatch: a batch ready for the reader---
def TextNumpyBatch(batch):
    images,flatTargets,targetLengths,texts = TextCollateFn(batch)
    targets,offset = [],0
    for length in targetLengths:
        length = int(length)
        targets.append(flatTargets[offset:offset+length].tolist())
        offset+=length
    return images,targets,texts

#----------------------------------------------------------------------
#ValidateTextModel: score the reader on held out lines
#decodes each held out crop and compares with its transcript, giving
#character and word accuracy on lines training never saw
#----------------------------------------------------------------------
def ValidateTextModel(valSet,params,batchSize,rng):
    if len(valSet)==0:
        return 0.0,0.0
    totalLoss,nBatches = 0.0,0
    totalChars,totalCharErrors = 0,0
    for batch in IterateBatches(valSet,batchSize,rng,shuffle=False):
        images,targets,texts = TextNumpyBatch(batch)
        rawLogits = EvalTextForward(params,images)
        loss,_ = CtcLoss(rawLogits,targets)
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

#---BuildPagesSplit: split the pages into training and held out---
def BuildPagesSplit(maxPages,dataDir):
    dataset = IAMLineDatasetRaw(rootDir=dataDir,cacheDir=DEFAULT_TEXT_CACHE_DIR,maxPages=maxPages)
    if len(dataset)<4:
        raise RuntimeError("Not enough line samples to train.")
    trainIdx,testIdx = dataset.HoldoutSplitIndices()
    if not testIdx:
        raise RuntimeError("No held-out lines found.")
    return TrainEvalView(dataset,trainIdx,isTrain=True),TrainEvalView(dataset,testIdx,isTrain=False)

#---BuildTekliaSplit: the same for the second source---
def BuildTekliaSplit(dataDir):
    trainSet = TekliaLineDataset(Path(dataDir)/"train",isTrain=True)
    valSet = TekliaLineDataset(Path(dataDir)/"validation",isTrain=False)
    return trainSet,valSet


#----------------------------------------------------------------------
#PersonalLineView: the photographed writers' lines, for the reader
#AddPersonalSamples cuts those pages into lines and records a transcript
#for each, which is everything the reader needs.  it is normally called
#for the namer, so this wraps the same samples in the reader's shape
#----------------------------------------------------------------------
class PersonalLineView:
    def __init__(self,samples,isTrain):
        self.samples = samples
        self.isTrain = isTrain

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,i):
        item = self.samples[i]
        pilImg = ResizeLineImageFixed(DecodeCrop(item["imageBytes"]))
        if self.isTrain:
            pilImg = AugmentLineImage(pilImg)
        return TensorFromResized(pilImg),item["target"],item["text"]


#---CollectPersonalLines: every photographed line, split train and held out---
def CollectPersonalLines():
    holder = type("Holder",(),{})()
    holder.samples = []
    AddPersonalSamples(holder)
    train = [s for s in holder.samples if not s.get("is_holdout")]
    held = [s for s in holder.samples if s.get("is_holdout")]
    return train,held


#----------------------------------------------------------------------
#BuildJointSplit: Teklia plus the photographed writers
#this is the mix the deployed reader was trained on.  the public set
#supplies the volume a CTC network needs, the photographed pages supply
#the hands the robot is actually asked to imitate, and without them the
#reader is weakest on exactly the writing that matters here
#----------------------------------------------------------------------
def BuildJointSplit(dataDir):
    tekTrain,tekVal = BuildTekliaSplit(dataDir)
    perTrain,perHeld = CollectPersonalLines()
    print(f"[Mix] {len(tekTrain)} Teklia + {len(perTrain)} personal train lines, "
          f"{len(tekVal)} Teklia + {len(perHeld)} personal held out")
    return tekTrain,PersonalLineView(perTrain,isTrain=True),tekVal,PersonalLineView(perHeld,isTrain=False)


#----------------------------------------------------------------------
#IterateMixedBatches: batches drawn from two sets at a fixed ratio
#every batch holds the same proportion of photographed lines, rather than
#the two sets being concatenated and shuffled.  concatenating would let
#whole batches contain no personal lines at all, and with batch
#normalising that changes what the layer sees, not just the order
#----------------------------------------------------------------------
def IterateMixedBatches(mainSet,extraSet,batchSize,fraction,rng,shuffle=True):
    nExtra = min(len(extraSet),max(1,int(round(batchSize*fraction)))) if len(extraSet) else 0
    nMain = max(1,batchSize-nExtra)
    mainOrder = rng.permutation(len(mainSet)) if shuffle else np.arange(len(mainSet))
    extraOrder = rng.permutation(len(extraSet)) if (shuffle and nExtra) else np.arange(len(extraSet))
    extraAt = 0
    for start in range(0,len(mainSet),nMain):
        batch = [mainSet[int(i)] for i in mainOrder[start:start+nMain]]
        for _ in range(nExtra):
            if extraAt>=len(extraOrder):          #the smaller set is reused as often as needed
                extraOrder = rng.permutation(len(extraSet))
                extraAt = 0
            batch.append(extraSet[int(extraOrder[extraAt])])
            extraAt += 1
        yield batch

#----------------------------------------------------------------------
#TrainTextModel: the reader's training run
#batches, forward, CTC loss, backward, clip, step, validate, save when
#it improves.  this is the loop that produced the checkpoints the
#results were measured on
#----------------------------------------------------------------------
def TrainTextModel(numEpochs,batchSize,lr,seed,maxPages,dataDir,source="pages",resume=False,
                   warmStart=None,personalFraction=0.20,tag=None):
    personalSet = personalVal = None
    if source=="joint":
        trainSet,personalSet,valSet,personalVal = BuildJointSplit(dataDir)
    elif source=="teklia":
        trainSet,valSet = BuildTekliaSplit(dataDir)
    else:
        trainSet,valSet = BuildPagesSplit(maxPages,dataDir)
    print(f"[Split] Train lines: {len(trainSet)} | Held-out val lines: {len(valSet)}")

    numClasses = len(CHARSET)+1
    params = InitTextModelParams(seed,numClasses)
    trainable = TrainableParams(params)
    optimizer = RMSprop(trainable,lr=lr,weightDecay=1e-5)
    scheduler = ReduceLrOnPlateau(optimizer,factor=0.5,patience=5,minLr=1e-5)

    rng = np.random.default_rng(seed)
    bestValLoss = np.inf
    bestPath = WEIGHTS_DIR/"paper_cnn_bilstm_ctc_scratch_best.npz"
    ckptPath = WEIGHTS_DIR/"paper_cnn_bilstm_ctc_scratch_checkpoint.npz"
    logPath = WEIGHTS_DIR/"paper_cnn_bilstm_ctc_scratch_log.txt"

    startEpoch = 1
    if warmStart:
        loaded = LoadParamsInto(params,warmStart)
        print(f"[WarmStart] loaded {loaded}/{len(params)} tensors from {warmStart}")
    if resume and ckptPath.exists():
        loaded = LoadParamsInto(params,ckptPath)
        prevEpoch,prevBest = LoadCheckpointMeta(ckptPath)
        startEpoch = prevEpoch+1
        if prevBest is not None:
            bestValLoss = prevBest
        print(f"[Resume] loaded {loaded}/{len(params)} tensors from {ckptPath}, "
              f"continuing at epoch {startEpoch} (best val loss so far {bestValLoss:.4f})")

    for epoch in range(startEpoch,startEpoch+numEpochs):
        epochStart = time.time()
        epochLoss,nBatches = 0.0,0
        batches = (IterateMixedBatches(trainSet,personalSet,batchSize,personalFraction,rng)
                   if personalSet is not None
                   else IterateBatches(trainSet,batchSize,rng,shuffle=True))
        for batch in batches:
            images,targets,_texts = TextNumpyBatch(batch)
            rawLogits,cache = TextModelForward(images,params)
            loss,dRawLogits = CtcLoss(rawLogits,targets)
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
        if personalVal is not None and len(personalVal):
            #the public held-out set says whether it reads handwriting; this one
            #says whether it reads THESE writers, which is what the robot needs
            _pLoss,pAcc = ValidateTextModel(personalVal,params,batchSize,rng)
            print(f"[TrainText] held-out personal char-acc {pAcc:.2%}")
        scheduler.Step(valLoss)
        improved = valLoss<bestValLoss
        savedMsg = ""
        if improved:
            bestValLoss = valLoss
            SaveCheckpoint(bestPath,params,extra={"__epoch__":np.array(epoch),"__best__":np.array(bestValLoss)})
            savedMsg = " [BEST SAVED]"
        SaveCheckpoint(ckptPath,params,extra={"__epoch__":np.array(epoch),"__best__":np.array(bestValLoss)})

        elapsedMin = (time.time()-epochStart)/60
        print(f"[TrainText] Epoch {epoch:03d} | Train loss {trainLoss:.3f} | "
              f"Val loss {valLoss:.3f} (best {bestValLoss:.3f}) | Val char-acc {valCharAcc:.2%} | "
              f"lr {optimizer.lr:.6f} | {elapsedMin:.1f} min{savedMsg}")
        AppendLog(logPath,f"epoch={epoch} train_loss={trainLoss:.4f} val_loss={valLoss:.4f} "
                          f"best_val_loss={bestValLoss:.4f} val_char_acc={valCharAcc:.4f} "
                          f"lr={optimizer.lr:.6f} minutes={elapsedMin:.1f}{savedMsg}")

    return bestValLoss



#======================================================================
#TRAINING THE WRITER NAMER
#the smaller network that answers whose handwriting a crop is
#======================================================================
#---InitAuthorModelParams: build a fresh namer---
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

#---AuthorModelForward: backbone, pooling, one score per writer---
def AuthorModelForward(x,params,training,rng):
    backboneOut,backboneCache = BackboneForwardTrain(x,params)
    pooledH,hpCache = AdaptiveMaxPoolHeightForward(backboneOut)   #(B,128,1,W)
    squeezed = pooledH[:,:,0,:]                               #(B,128,W)
    pooledW,apCache = AdaptiveAvgPoolWidthForward(squeezed)  #(B,128,1), width collapsed too (unlike the text model)
    pooled = pooledW[:,:,0]                                    #(B,128)

    h,lin1Cache = LinearForward(pooled,params["author_head.0.weight"],params["author_head.0.bias"])
    h,reluCache = ReluForward(h)
    h,dropCache = DropoutForward(h,0.2,training,rng)
    logits,lin2Cache = LinearForward(h,params["author_head.3.weight"],params["author_head.3.bias"])

    cache = dict(backboneCache=backboneCache,hpCache=hpCache,apCache=apCache,
                 lin1Cache=lin1Cache,reluCache=reluCache,dropCache=dropCache,lin2Cache=lin2Cache)
    return logits,cache

#---AuthorModelBackward: and back again---
def AuthorModelBackward(dLogits,cache):
    grads = {}
    dh,dw2,db2 = LinearBackward(dLogits,cache["lin2Cache"])
    grads["author_head.3.weight"] = dw2
    grads["author_head.3.bias"] = db2

    dh = DropoutBackward(dh,cache["dropCache"])
    dh = ReluBackward(dh,cache["reluCache"])
    dpooled,dw1,db1 = LinearBackward(dh,cache["lin1Cache"])
    grads["author_head.0.weight"] = dw1
    grads["author_head.0.bias"] = db1

    dPooledW = dpooled[:,:,None]                                   #(B,128,1)
    dSqueezed = AdaptiveAvgPoolWidthBackward(dPooledW,cache["apCache"])  #(B,128,W)
    dPooledH = dSqueezed[:,:,None,:]                               #(B,128,1,W)
    dBackboneOut = AdaptiveMaxPoolHeightBackward(dPooledH,cache["hpCache"])
    dx,backboneGrads = BackboneBackwardTrain(dBackboneOut,cache["backboneCache"])
    grads.update(backboneGrads)
    return dx,grads

#---EvalAuthorForward: forward only, for validation---
def EvalAuthorForward(params,x):
    sd = params
    x = EvalBackboneForward(sd,x)
    x = CLS.AdaptiveMaxPoolHeight(x)
    x = x[:,:,0,:]
    pooled = CLS.AdaptiveAvgPoolWidth(x)[:,:,0]
    h = CLS.Relu(CLS.Linear(pooled,sd["author_head.0.weight"],sd["author_head.0.bias"]))
    return CLS.Linear(h,sd["author_head.3.weight"],sd["author_head.3.bias"])

#---AuthorNumpyBatch: a batch ready for the namer---
def AuthorNumpyBatch(batch):
    return AuthorCollateFn(batch)

#---ValidateAuthorModel: accuracy on held out crops---
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

#----------------------------------------------------------------------
#TrainAuthorModel: the namer's training run
#same shape of loop, cross entropy instead of CTC.  trained twice, once
#on raw grey crops and once on stroke normalised ones, which is where the
#ink based and shape based checkpoints come from
#----------------------------------------------------------------------
def TrainAuthorModel(numEpochs,batchSize,lr,seed,resume=False,warmStart=None):
    base = IAMLineDatasetRaw(rootDir=str(DEFAULT_AUTHOR_DATA_DIR),cacheDir=str(DEFAULT_AUTHOR_CACHE_DIR))
    before = len(base.samples)
    base.samples = [s for s in base.samples if s["page_key"].split("/")[0] in paths.DatabaseWriters]
    print(f"[Filter] kept {len(base.samples)}/{before} IAM samples for {paths.DatabaseWriters}")

    AddPersonalSamples(base)
    base.author_folders = sorted(paths.DatabaseWriters+paths.PersonalWriters)
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
    optimizer = AdamW(trainable,lr=lr,weightDecay=1e-4)
    scheduler = CosineAnnealingLr(optimizer,tMax=numEpochs)

    rngBatch = np.random.default_rng(seed)
    rngDropout = np.random.default_rng(seed+1)
    bestAcc = 0.0
    outPath = WEIGHTS_DIR/"author_classifier_10new_scratch_weights.npz"
    ckptPath = WEIGHTS_DIR/"author_classifier_10new_scratch_checkpoint.npz"
    logPath = WEIGHTS_DIR/"author_classifier_10new_scratch_log.txt"

    startEpoch = 1
    if warmStart:
        loaded = LoadParamsInto(params,warmStart)
        print(f"[WarmStart] loaded {loaded}/{len(params)} tensors from {warmStart}")
    if resume and ckptPath.exists():
        loaded = LoadParamsInto(params,ckptPath)
        prevEpoch,prevBest = LoadCheckpointMeta(ckptPath)
        startEpoch = prevEpoch+1
        if prevBest is not None:
            bestAcc = prevBest
        print(f"[Resume] loaded {loaded}/{len(params)} tensors from {ckptPath}, "
              f"continuing at epoch {startEpoch} (best val acc so far {bestAcc:.3f})")

    for epoch in range(startEpoch,startEpoch+numEpochs):
        epochStart = time.time()
        epochLoss,nBatches = 0.0,0
        for batch in IterateBatches(trainSet,batchSize,rngBatch,shuffle=True):
            images,authorIdxs,_ = AuthorNumpyBatch(batch)
            logits,cache = AuthorModelForward(images,params,training=True,rng=rngDropout)
            loss,dLogits = CrossEntropyLoss(logits,authorIdxs)
            _,grads = AuthorModelBackward(dLogits,cache)
            trainableGrads = {k:grads[k] for k in trainable}
            ClipGradNorm(trainableGrads,maxNorm=5.0)
            optimizer.Step(trainableGrads)
            epochLoss+=loss
            nBatches+=1
        scheduler.Step()
        trainLoss = epochLoss/max(1,nBatches)

        valAcc = ValidateAuthorModel(valSet,params,batchSize,rngBatch)
        savedMsg = ""
        if valAcc>bestAcc:
            bestAcc = valAcc
            SaveCheckpoint(outPath,params,extra={"__author_names__":np.array(base.author_folders),
                                                  "__epoch__":np.array(epoch),"__best__":np.array(bestAcc)})
            savedMsg = " [SAVED]"
        SaveCheckpoint(ckptPath,params,extra={"__author_names__":np.array(base.author_folders),
                                               "__epoch__":np.array(epoch),"__best__":np.array(bestAcc)})

        elapsedMin = (time.time()-epochStart)/60
        print(f"[TrainAuthor] Epoch {epoch:03d} | loss {trainLoss:.3f} | val author-acc {valAcc:.3f} "
              f"(best {bestAcc:.3f}) | lr {optimizer.lr:.6f} | {elapsedMin:.1f} min{savedMsg}")
        AppendLog(logPath,f"epoch={epoch} train_loss={trainLoss:.4f} val_acc={valAcc:.4f} "
                          f"best_val_acc={bestAcc:.4f} lr={optimizer.lr:.6f} minutes={elapsedMin:.1f}{savedMsg}")

    print(f"\nBest val author-accuracy: {bestAcc:.3f}")
    print(f"Weights saved to: {outPath}")
    return bestAcc


#======================================================================
#RUNNING A TRAINING JOB
#python train.py text   --epochs 40
#python train.py author --epochs 40
#
#--resume continues from this module's own checkpoint, and --warm-start
#begins from any checkpoint on disk.  neither restores the OPTIMISER, only
#the weights: RMSprop's running average of the squared gradient starts at
#zero again, and because it divides the step by the square root of that
#average, the first few steps after a resume are much larger than they
#would have been.  a lower learning rate for the first epochs avoids
#undoing the progress that was being resumed
#======================================================================
def Main():
    import argparse
    ap = argparse.ArgumentParser(description="train the reader or the namer from first principles")
    ap.add_argument("task",choices=["text","author"])
    ap.add_argument("--epochs",type=int,default=40)
    ap.add_argument("--batch",type=int,default=8)
    ap.add_argument("--lr",type=float,default=1e-3)
    ap.add_argument("--seed",type=int,default=7)
    ap.add_argument("--source",choices=["joint","teklia","pages"],default="joint",
                    help="joint: the pre-cut Teklia lines plus the photographed writers, "
                         "which is what the deployed reader was trained on. "
                         "teklia: public lines only. pages: segment the database pages here")
    ap.add_argument("--personal-fraction",dest="personalFraction",type=float,default=0.20,
                    help="joint only: share of every batch drawn from photographed lines")
    ap.add_argument("--pages",type=int,default=None,help="limit how many pages are cut up")
    ap.add_argument("--data",default=None,help="override where the pages or crops live")
    ap.add_argument("--resume",action="store_true",
                    help="continue from this module's own checkpoint")
    ap.add_argument("--warm-start",dest="warmStart",default=None,
                    help="start from a named checkpoint, e.g. an earlier scratch run")
    args = ap.parse_args()

    warm = Path(args.warmStart) if args.warmStart else None
    if warm and not warm.exists():
        candidate = WEIGHTS_DIR/warm.name
        if candidate.exists():
            warm = candidate
        else:
            raise SystemExit("no checkpoint at %s" % args.warmStart)

    if args.task=="text":
        if args.data:
            dataDir = args.data
        elif args.source in ("teklia","joint"):
            dataDir = str(paths.Data.parents[1]/"Data"/"Datasets"/"TekliaIamLines")
        else:
            dataDir = str(DEFAULT_TEXT_DATA_DIR)
        TrainTextModel(args.epochs,args.batch,args.lr,args.seed,args.pages,dataDir,
                       source=args.source,resume=args.resume,warmStart=warm,
                       personalFraction=args.personalFraction)
    else:
        TrainAuthorModel(args.epochs,args.batch,args.lr,args.seed,
                         resume=args.resume,warmStart=warm)


if __name__ == "__main__":
    Main()
