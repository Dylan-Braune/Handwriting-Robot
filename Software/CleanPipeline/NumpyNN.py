#hand written forward AND backward passes for every layer type the text and
#author classifiers use (conv2d, batchnorm2d, relu, maxpool2d, adaptive max/avg
#pool, linear, dropout, bidirectional multilayer lstm, ctc loss, cross entropy
#loss) plus the two optimizers (rmsprop, adamw) and two schedulers (reduce on
#plateau, cosine annealing) the original training recipe uses. no ml library
#anywhere, only numpy. every forward returns (out,cache); every backward takes
#(dout,cache) and returns the gradients needed by the caller. parameters are
#kept in plain dicts keyed exactly like a pytorch state_dict ("stage1.0.weight"
#etc.) so a checkpoint saved from here loads straight into Classify.py's
#existing from-scratch inference code with no changes there at all
import numpy as np


#=== conv2d ===

#x:(B,C,H,W), weight:(OutC,InC,kh,kw). im2col+matmul, same as Classify.py's
#inference version but this one also returns what backward needs
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
    cols = np.lib.stride_tricks.as_strided(xp,shape=shape,strides=strides)
    colsFlat = cols.reshape(B,C*kh*kw,outH*outW)

    w = weight.reshape(OutC,C*kh*kw)
    out = (w@colsFlat).reshape(B,OutC,outH,outW)   #w(OutC,K) broadcasts over colsFlat's batch dim -> one batched BLAS gemm, not a per-sample python loop
    if bias is not None:
        out = out+bias.reshape(1,OutC,1,1)
    cache = (x.shape,weight,colsFlat,padding,kh,kw,outH,outW)
    return out,cache


#dw via dout@cols.T, dx via col2im scatter add of weight.T@dout (kernel windows
#overlap when kh,kw>1 so this must accumulate, not just reshape)
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


#=== batchnorm2d (training mode: batch statistics, not running stats) ===

def BatchNorm2dForwardTrain(x,gamma,beta,runningMean,runningVar,momentum=0.1,eps=1e-5):
    N = x.shape[0]*x.shape[2]*x.shape[3]
    mu = x.mean(axis=(0,2,3))
    xc = x-mu.reshape(1,-1,1,1)
    varBiased = (xc**2).mean(axis=(0,2,3))
    std = np.sqrt(varBiased+eps)
    xhat = xc/std.reshape(1,-1,1,1)
    out = gamma.reshape(1,-1,1,1)*xhat+beta.reshape(1,-1,1,1)

    varUnbiased = varBiased*N/max(N-1,1)   #pytorch updates running_var with the unbiased estimate
    runningMean[:] = (1-momentum)*runningMean+momentum*mu
    runningVar[:] = (1-momentum)*runningVar+momentum*varUnbiased

    cache = (xhat,std,gamma,N)
    return out,cache


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


#=== relu ===

def ReluForward(x):
    out = np.maximum(x,0)
    return out,(x>0)


def ReluBackward(dout,cache):
    mask = cache
    return dout*mask


#=== maxpool2d, kernel==stride (non overlapping, so no scatter add needed) ===

def MaxPool2dForward(x,kernel=2,stride=2):
    B,C,H,W = x.shape
    outH,outW = H//stride,W//stride
    xc = x[:,:,:outH*stride,:outW*stride]
    s = xc.strides
    shape = (B,C,outH,outW,kernel,kernel)
    strides = (s[0],s[1],s[2]*stride,s[3]*stride,s[2],s[3])
    windows = np.lib.stride_tricks.as_strided(xc,shape=shape,strides=strides)
    flat = windows.reshape(B,C,outH,outW,kernel*kernel)
    argmax = flat.argmax(axis=4)
    out = np.take_along_axis(flat,argmax[...,None],axis=4)[...,0]
    cache = (x.shape,argmax,kernel,stride,outH,outW)
    return out,cache


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


#=== adaptive max pool over height: (B,C,H,W) -> (B,C,1,W) ===

def AdaptiveMaxPoolHeightForward(x):
    argmax = x.argmax(axis=2)   #(B,C,W)
    out = np.take_along_axis(x,argmax[:,:,None,:],axis=2)
    cache = (x.shape,argmax)
    return out,cache


def AdaptiveMaxPoolHeightBackward(dout,cache):
    xShape,argmax = cache
    dx = np.zeros(xShape,dtype=dout.dtype)
    np.put_along_axis(dx,argmax[:,:,None,:],dout,axis=2)
    return dx


#=== adaptive avg pool over width: (B,C,W) -> (B,C,1) ===

def AdaptiveAvgPoolWidthForward(x):
    out = x.mean(axis=2,keepdims=True)
    return out,x.shape[2]


def AdaptiveAvgPoolWidthBackward(dout,cache):
    W = cache
    return np.repeat(dout,W,axis=2)/W


#=== linear ===

def LinearForward(x,weight,bias):
    out = x@weight.T
    if bias is not None:
        out = out+bias
    return out,(x,weight)


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


#=== dropout (inverted, so no rescaling needed at inference) ===

def DropoutForward(x,p,training,rng):
    if not training or p<=0:
        return x,None
    mask = (rng.random(x.shape)>=p)/(1.0-p)
    return x*mask,mask


def DropoutBackward(dout,cache):
    mask = cache
    if mask is None:
        return dout
    return dout*mask


#=== lstm, one direction one layer. gate blocks ordered [input,forget,cell,
#output] in weight_ih/weight_hh, matching pytorch's convention (and therefore
#Classify.py's inference code, so a checkpoint trained here loads there as-is)

def Sigmoid(x):
    return 1.0/(1.0+np.exp(-x))


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


#backprop through time: standard lstm gate gradient recurrence
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


#x:(T,B,input_size) -> (T,B,2*hidden_size)
def BidirectionalLstmForwardTrain(x,paramsFwd,paramsBwd,hiddenSize):
    outF,cacheF = LstmForwardTrain(x,**paramsFwd,hiddenSize=hiddenSize)
    outB,cacheB = LstmForwardTrain(x[::-1],**paramsBwd,hiddenSize=hiddenSize)
    outB = outB[::-1]
    out = np.concatenate([outF,outB],axis=2)
    return out,(cacheF,cacheB,hiddenSize)


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


#layerParams: list of {"fwd": {...}, "bwd": {...}} weight dicts, one per layer
def MultilayerBidirectionalLstmForwardTrain(x,layerParams,hiddenSize):
    out = x
    caches = []
    for layer in layerParams:
        out,cache = BidirectionalLstmForwardTrain(out,layer["fwd"],layer["bwd"],hiddenSize)
        caches.append(cache)
    return out,caches


def MultilayerBidirectionalLstmBackward(dout,caches):
    dx = dout
    gradsPerLayer = []
    for cache in reversed(caches):
        dx,gradsFwd,gradsBwd = BidirectionalLstmBackward(dx,cache)
        gradsPerLayer.append((gradsFwd,gradsBwd))
    gradsPerLayer.reverse()
    return dx,gradsPerLayer


#=== ctc loss: graves et al. 2006, forward-backward (alpha/beta) in log space
#for numerical stability, then the closed form gradient w.r.t. the PRE-softmax
#logits (y - occupation/total), which skips needing a separate log_softmax
#backward step -- same simplification as softmax+cross-entropy's "pred-target" ===

def LogSumExp(a,b):
    m = np.maximum(a,b)
    #-inf + -inf would nan in the subtraction below (even though it's masked
    #out by np.where afterward, numpy still evaluates it eagerly and warns) --
    #errstate suppresses that specific, already-handled warning
    with np.errstate(invalid="ignore"):
        out = np.where(np.isneginf(m),-np.inf,m+np.log(np.exp(a-m)+np.exp(b-m)))
    return out


#rawLogits:(T,B,C). targets: list of length-B int label-index sequences
#(indices into the model's own class space, blank is index 0). returns
#(meanLoss, dRawLogits) -- one sample at a time internally, python loop over
#the batch, since each sample's label length differs
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


#=== cross entropy loss (author classifier): combined softmax+nll gradient ===

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


#=== optimizers, matching the exact recipes TrainText.py/TrainAuthor.py use ===

#rmsprop: matches torch.optim.RMSprop defaults (alpha=0.99,eps=1e-8) + decoupled
#weight decay applied the way torch's RMSprop applies it (added to the grad,
#not decoupled like adamw -- torch.optim.RMSprop's weight_decay is the
#classic L2-via-gradient form, not the "W" decoupled form)
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


#adamw: decoupled weight decay (subtracted directly from the parameter, not
#added to the gradient before the moment estimates), matching torch.optim.AdamW
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


#reduce-lr-on-plateau: halves lr after `patience` epochs with no improvement
#in the tracked (validation) loss, floored at minLr. matches
#torch.optim.lr_scheduler.ReduceLROnPlateau(mode="min",factor=0.5,...)
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


#cosine annealing: matches torch.optim.lr_scheduler.CosineAnnealingLR(eta_min=0)
class CosineAnnealingLr:
    def __init__(self,optimizer,tMax,baseLr=None):
        self.optimizer = optimizer
        self.tMax = tMax
        self.baseLr = baseLr if baseLr is not None else optimizer.lr
        self.epoch = 0

    def Step(self):
        self.epoch+=1
        self.optimizer.lr = 0.5*self.baseLr*(1+np.cos(np.pi*self.epoch/self.tMax))


#matches torch.nn.utils.clip_grad_norm_: scales every gradient down together
#(one shared scale factor from the combined l2 norm) if that norm exceeds maxNorm
def ClipGradNorm(grads,maxNorm=5.0):
    totalNormSq = sum(float((g**2).sum()) for g in grads.values())
    totalNorm = np.sqrt(totalNormSq)
    if totalNorm>maxNorm:
        scale = maxNorm/(totalNorm+1e-6)
        for k in grads:
            grads[k] = grads[k]*scale
    return totalNorm
