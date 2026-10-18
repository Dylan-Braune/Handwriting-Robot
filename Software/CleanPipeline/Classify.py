#from scratch numpy only inference for both classifiers: the text recognizer
#(conv backbone + bidirectional lstm + ctc) and the writer-id classifier (same
#conv backbone + pooling head). no torch import anywhere in this file -- .pt
#checkpoints are read directly as a zip of pickled metadata + raw tensor bytes.
#merged from the old np_inference package (weight_io+layers+text_model+author_model)
import pickle
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from Charset import CHARSET,CHAR_TO_IDX,IDX_TO_CHAR
import ProfileIO as PIO

#this file lives in Software/CleanPipeline, a sibling of Software/CNN where
#the actual trained weights still live -- data is not duplicated
CNN_DATA_DIR = Path(__file__).resolve().parent.parent/"CNN"
WEIGHTS_DIR = CNN_DATA_DIR/"weights"

INPUT_HEIGHT = 64
INPUT_WIDTH = 640
PEN_WIDTH_XH = 0.14

#prefers a from-scratch .npz checkpoint (TrainScratch.py's output) over the
#old torch-trained .pt chain -- same precedence rule in both places that
#pick a default checkpoint (here, and Evaluate.py's own AUTHOR_WEIGHTS/
#TEXT_WEIGHTS/SHAPE_WEIGHTS), so anything that trains with TrainScratch.py
#gets picked up everywhere a model is loaded with no explicit path argument
DEFAULT_TEXT_WEIGHTS = WEIGHTS_DIR/"paper_cnn_bilstm_ctc_best.pt"
for _name in ("paper_cnn_bilstm_ctc_joint_best.pt","paper_cnn_bilstm_ctc_hf_best.pt"):
    _candidate = WEIGHTS_DIR/_name
    if _candidate.exists():
        DEFAULT_TEXT_WEIGHTS = _candidate
        break
_TEXT_NPZ = WEIGHTS_DIR/"paper_cnn_bilstm_ctc_scratch_best.npz"
if _TEXT_NPZ.exists():
    DEFAULT_TEXT_WEIGHTS = _TEXT_NPZ

_AUTHOR_NPZ = WEIGHTS_DIR/"author_classifier_10new_scratch_weights.npz"
_SHAPE = WEIGHTS_DIR/"author_shape_10new_weights.pt"
_W1 = WEIGHTS_DIR/"author_classifier_10_weights.pt"
_W2 = WEIGHTS_DIR/"author_fast_10_weights.pt"
DEFAULT_AUTHOR_WEIGHTS = _AUTHOR_NPZ if _AUTHOR_NPZ.exists() else (
    _SHAPE if _SHAPE.exists() else (_W1 if _W1.exists() else _W2))

STORAGE_DTYPES = {
    "FloatStorage": np.dtype("<f4"),
    "DoubleStorage": np.dtype("<f8"),
    "HalfStorage": np.dtype("<f2"),
    "LongStorage": np.dtype("<i8"),
    "IntStorage": np.dtype("<i4"),
    "ShortStorage": np.dtype("<i2"),
    "CharStorage": np.dtype("<i1"),
    "ByteStorage": np.dtype("<u1"),
    "BoolStorage": np.dtype("?"),
}


#=== checkpoint loading: reads a .pt file (zip of pickled graph + raw tensor
#bytes) straight into numpy arrays, without ever importing torch ===

#placeholder for a torch storage class, real bytes arrive later via persistent_load
class StorageStub:
    def __init__(self,dtype):
        self.dtype = dtype

    def __call__(self,*args,**kwargs):
        return self


#rebuilds a numpy array from a flat storage array and its view info. pytorch
#strides are in elements, as_strided needs strides in bytes
def RebuildTensor(storage,storage_offset,size,stride,requires_grad=False,backward_hooks=None,metadata=None):
    itemsize = storage.dtype.itemsize
    byte_strides = tuple(s*itemsize for s in stride)
    if len(size)==0:
        return storage[storage_offset:storage_offset+1].reshape(()).copy()
    view = np.lib.stride_tricks.as_strided(storage[storage_offset:],shape=tuple(size),strides=byte_strides)
    return view.copy()


#pickle unpickler that intercepts torch specific classes and resolves everything else normally
class TorchUnpickler(pickle.Unpickler):
    def __init__(self,file,zf,data_prefix):
        super().__init__(file)
        self._zf = zf
        self._data_prefix = data_prefix

    def find_class(self,module,name):
        if module=="torch._utils" and name=="_rebuild_tensor_v2":
            return RebuildTensor
        if module=="torch" and name in STORAGE_DTYPES:
            return StorageStub(STORAGE_DTYPES[name])
        if module=="collections" and name=="OrderedDict":
            import collections
            return collections.OrderedDict
        if module=="torch._utils" and name=="_rebuild_parameter":
            return lambda data,requires_grad,backward_hooks: data
        return super().find_class(module,name)

    def persistent_load(self,pid):
        tag,storage_type,key,location,numel = pid   #format: ('storage', type_stub, key, location, numel)
        assert tag=="storage"
        dtype = storage_type.dtype
        raw = self._zf.read(f"{self._data_prefix}data/{key}")
        arr = np.frombuffer(raw,dtype=dtype,count=numel)
        return arr


#unpickles a .pt checkpoint's full object graph, tensors as numpy arrays
def LoadCheckpoint(pt_path):
    pt_path = str(pt_path)
    import io
    with zipfile.ZipFile(pt_path,"r") as zf:
        names = zf.namelist()
        pkl_name = next(n for n in names if n.endswith("data.pkl"))
        prefix = pkl_name[:-len("data.pkl")]   #archive folder name varies, derive it from data.pkl's own path
        with zf.open(pkl_name) as f:
            buf = io.BytesIO(f.read())
            unpickler = TorchUnpickler(buf,zf,prefix)
            return unpickler.load()


#thin wrapper: unwraps 'model_state_dict' if the checkpoint is a dict wrapping the state dict
def LoadStateDict(pt_path):
    obj = LoadCheckpoint(pt_path)
    if isinstance(obj,dict) and "model_state_dict" in obj:
        obj = obj["model_state_dict"]
    return dict(obj)


#=== .npz loading path (new) -- for checkpoints produced by TrainTextScratch.py/
#TrainAuthorScratch.py. kept entirely separate from the .pt pickle path above,
#which is untouched and stays the fallback for already-existing .pt checkpoints.
#npz can't hold a python dict, so a saved author_mapping is stashed as a plain
#"__author_names__" string array (names sorted by index) and rebuilt into a
#{name: idx} dict here ===
def LoadStateDictAny(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix==".npz":
        with np.load(checkpoint_path,allow_pickle=False) as npz:
            sd = {k: npz[k] for k in npz.files if k!="__author_names__"}
            author_mapping = None
            if "__author_names__" in npz.files:
                names = [str(n) for n in npz["__author_names__"]]
                author_mapping = {name: idx for idx,name in enumerate(names)}
        return sd,author_mapping
    obj = LoadCheckpoint(checkpoint_path)
    author_mapping = obj.get("author_mapping") if isinstance(obj,dict) else None
    if isinstance(obj,dict) and "model_state_dict" in obj:
        obj = obj["model_state_dict"]
    return dict(obj),author_mapping


#dumps a checkpoint's tensors to .npz (+json sidecar for other metadata), so
#downstream code can skip re-parsing pickle. convenience only, not on the live path
def ExportToNpz(pt_path,out_npz_path):
    import json
    obj = LoadCheckpoint(pt_path)
    if isinstance(obj,dict) and "model_state_dict" in obj:
        tensors = obj["model_state_dict"]
        meta = {k:v for k,v in obj.items() if k!="model_state_dict"}
    else:
        tensors = obj
        meta = {}
    out_npz_path = Path(out_npz_path)
    np.savez(out_npz_path,**{k:v for k,v in tensors.items()})

    def _jsonable(v):
        if isinstance(v,dict):
            return {str(k):_jsonable(x) for k,x in v.items()}
        if isinstance(v,np.ndarray):
            return v.tolist()
        if isinstance(v,(np.integer,np.floating)):
            return v.item()
        return v

    if meta:
        with open(out_npz_path.with_suffix(".json"),"w") as f:
            json.dump(_jsonable(meta),f,indent=2)


#=== numpy layer primitives, shared by both models ===

#x: (B,C,H,W), weight: (OutC,InC,kh,kw). im2col + matmul
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
    #plain matmul dispatches to blas, ~10x faster than einsum for identical output here
    out = np.stack([w@cols[b] for b in range(B)],axis=0).reshape(B,OutC,outH,outW)
    if bias is not None:
        out = out+bias.reshape(1,OutC,1,1)
    return out


#eval mode batchnorm affine from stored running mean/var
def BatchNorm2d(x,weight,bias,running_mean,running_var,eps=1e-5):
    scale = weight/np.sqrt(running_var+eps)
    shift = bias-running_mean*scale
    return x*scale.reshape(1,-1,1,1)+shift.reshape(1,-1,1,1)


def Relu(x):
    return np.maximum(x,0)


def MaxPool2d(x,kernel=2,stride=2):
    B,C,H,W = x.shape
    outH,outW = H//stride,W//stride
    xc = x[:,:,:outH*stride,:outW*stride]
    s = xc.strides
    shape = (B,C,outH,outW,kernel,kernel)
    strides = (s[0],s[1],s[2]*stride,s[3]*stride,s[2],s[3])
    windows = np.lib.stride_tricks.as_strided(xc,shape=shape,strides=strides)
    return windows.max(axis=(4,5))


#(B,C,H,W) -> (B,C,1,W), max over H
def AdaptiveMaxPoolHeight(x):
    return x.max(axis=2,keepdims=True)


#(B,C,W) -> (B,C,1), mean over W
def AdaptiveAvgPoolWidth(x):
    return x.mean(axis=2,keepdims=True)


#weight: (out_features,in_features), matching nn.Linear
def Linear(x,weight,bias):
    out = x@weight.T
    if bias is not None:
        out = out+bias
    return out


def Sigmoid(x):
    return 1.0/(1.0+np.exp(-x))


#one direction, one layer. x: (T,B,input_size). gate blocks in weight_ih/weight_hh
#(each 4*hidden rows) are ordered [input,forget,cell/g,output], pytorch's lstm convention
def LstmForward(x,weight_ih,weight_hh,bias_ih,bias_hh,hidden_size):
    T,B,_ = x.shape
    h = np.zeros((B,hidden_size),dtype=x.dtype)
    c = np.zeros((B,hidden_size),dtype=x.dtype)
    H = hidden_size
    outputs = np.empty((T,B,H),dtype=x.dtype)

    for t in range(T):
        gates = x[t]@weight_ih.T+bias_ih+h@weight_hh.T+bias_hh
        i = Sigmoid(gates[:,0*H:1*H])
        f = Sigmoid(gates[:,1*H:2*H])
        g = np.tanh(gates[:,2*H:3*H])
        o = Sigmoid(gates[:,3*H:4*H])
        c = f*c+i*g
        h = o*np.tanh(c)
        outputs[t] = h

    return outputs,(h,c)


#x: (T,B,input_size) -> (T,B,2*hidden_size)
def BidirectionalLstmForward(x,params_fwd,params_bwd,hidden_size):
    out_f,_ = LstmForward(x,**params_fwd,hidden_size=hidden_size)
    out_b,_ = LstmForward(x[::-1],**params_bwd,hidden_size=hidden_size)
    out_b = out_b[::-1]
    return np.concatenate([out_f,out_b],axis=2)


#layer_params: list of {"fwd": {...}, "bwd": {...}} weight dicts
def MultilayerBidirectionalLstm(x,layer_params,hidden_size):
    out = x
    for layer in layer_params:
        out = BidirectionalLstmForward(out,layer["fwd"],layer["bwd"],hidden_size)
    return out


def LogSoftmax(x,axis):
    m = np.max(x,axis=axis,keepdims=True)
    shifted = x-m
    lse = np.log(np.sum(np.exp(shifted),axis=axis,keepdims=True))
    return shifted-lse


def Softmax(x,axis):
    m = np.max(x,axis=axis,keepdims=True)
    e = np.exp(x-m)
    return e/np.sum(e,axis=axis,keepdims=True)


#resizes/pads one line image to the fixed classifier input size, same as SegmentLean's version
def ResizeLineImageFixed(pil_img,height=INPUT_HEIGHT,width=INPUT_WIDTH,pad_value=255):
    img = pil_img.convert("L")
    w,h = img.size
    if w<=0 or h<=0:
        return Image.new("L",(width,height),pad_value)
    s = min(height/h,width/w)
    new_w = max(1,round(w*s))
    new_h = max(1,round(h*s))
    resized = img.resize((new_w,new_h),Image.Resampling.BILINEAR)
    canvas = Image.new("L",(width,height),pad_value)
    canvas.paste(resized,(0,max(0,(height-new_h)//2)))
    return canvas


#pil image -> (1,1,H,W) numpy array, ink inverted to high values and normalised to ~[-1,1]
def TensorFromResized(pil_img):
    arr = np.array(pil_img,dtype=np.float32)/255.0
    arr = 1.0-arr
    arr = (arr-0.5)/0.5
    return arr[np.newaxis,np.newaxis,:,:]


#greedy ctc decode: argmax per timestep, collapse repeats, drop the blank
def DecodeCtc(log_probs):
    best_path = np.argmax(log_probs,axis=2)
    decoded = []
    for batch_idx in range(best_path.shape[1]):
        prev,chars = None,[]
        for token in best_path[:,batch_idx]:
            token = int(token)
            if token!=0 and token!=prev:
                chars.append(IDX_TO_CHAR.get(token,""))
            prev = token
        decoded.append("".join(chars))
    return decoded


#conv_block(in,out,n_layers) as (conv,bn,relu)*n_layers, conv at i*3, bn at i*3+1 --
#shared by both models below since they use the identical backbone architecture
def ConvBlockForward(x,sd,prefix,n_layers):
    for i in range(n_layers):
        c = i*3
        x = Conv2d(x,sd[f"{prefix}.{c}.weight"],sd[f"{prefix}.{c}.bias"],padding=1)
        x = BatchNorm2d(x,sd[f"{prefix}.{c+1}.weight"],sd[f"{prefix}.{c+1}.bias"],
                         sd[f"{prefix}.{c+1}.running_mean"],sd[f"{prefix}.{c+1}.running_var"])
        x = Relu(x)
    return x


#=== text recognizer: conv backbone + bidirectional lstm + ctc head ===
class PaperCRNNNumpy:
    def __init__(self,checkpoint_path=None,hidden_size=256):
        self.checkpoint_path = Path(checkpoint_path or DEFAULT_TEXT_WEIGHTS)
        #.npz (new, from TrainTextScratch.py) vs .pt (existing pickle path) --
        #see LoadStateDictAny's own comment
        if self.checkpoint_path.suffix==".npz":
            self.sd,_ = LoadStateDictAny(self.checkpoint_path)
        else:
            self.sd = LoadStateDict(self.checkpoint_path)
        self.hidden_size = hidden_size

        self.lstm_layer_params = []
        for l in range(2):
            def get(name,layer=l):
                return self.sd[f"sequence.{name}_l{layer}"]

            fwd = dict(weight_ih=get("weight_ih"),weight_hh=get("weight_hh"),
                       bias_ih=get("bias_ih"),bias_hh=get("bias_hh"))
            bwd = dict(weight_ih=self.sd[f"sequence.weight_ih_l{l}_reverse"],
                       weight_hh=self.sd[f"sequence.weight_hh_l{l}_reverse"],
                       bias_ih=self.sd[f"sequence.bias_ih_l{l}_reverse"],
                       bias_hh=self.sd[f"sequence.bias_hh_l{l}_reverse"])
            self.lstm_layer_params.append({"fwd":fwd,"bwd":bwd})

    #x: (B,1,H,W) numpy array -> (T,B,num_classes) log-probs
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
        seq = MultilayerBidirectionalLstm(x,self.lstm_layer_params,self.hidden_size)
        logits = Linear(seq,sd["text_head.weight"],sd["text_head.bias"])
        return LogSoftmax(logits,axis=2)


#image -> decoded string, via the text recognizer
def ReadText(pil_image,model):
    t = TensorFromResized(ResizeLineImageFixed(pil_image))
    log_probs = model.forward(t)
    return DecodeCtc(log_probs)[0]


#pure string edit distance
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


#1 minus character error rate, floored at 0
def CharAcc(pred,truth):
    if not truth:
        return 1.0 if not pred else 0.0
    return max(0.0,1.0-Levenshtein(pred,truth)/len(truth))


#fraction of the intended words recovered in order, via longest common subsequence
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


#=== writer-id classifier: same conv backbone + pooling head, no lstm ===
class AuthorClassifierCNNNumpy:
    def __init__(self,checkpoint_path=None):
        self.checkpoint_path = Path(checkpoint_path or DEFAULT_AUTHOR_WEIGHTS)
        #.npz (new, from TrainAuthorScratch.py) vs .pt (existing pickle path) --
        #see LoadStateDictAny's own comment
        if self.checkpoint_path.suffix==".npz":
            self.sd,self.author_mapping = LoadStateDictAny(self.checkpoint_path)
        else:
            raw = LoadCheckpoint(self.checkpoint_path)
            self.sd = raw["model_state_dict"] if isinstance(raw,dict) and "model_state_dict" in raw else raw
            self.author_mapping = raw.get("author_mapping") if isinstance(raw,dict) else None

    #x: (B,1,H,W) numpy array -> (B,num_authors) raw logits
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
        #author_head.2 is dropout, inactive at inference, skipped
        logits = Linear(h,sd["author_head.3.weight"],sd["author_head.3.bias"])
        return logits


#image -> (predicted author index, softmax probability vector). callers resolve
#the index to an author-id string via {v:k for k,v in author_mapping.items()}
def ClassifyImage(pil_image,model,author_mapping=None,apply_stroke_normalize=True):
    if apply_stroke_normalize:
        normImg = PIO.StrokeNormalize(np.array(pil_image.convert("L")))
        if normImg is not None:
            pil_image = normImg
    t = TensorFromResized(ResizeLineImageFixed(pil_image))
    logits = model.forward(t)
    probs = Softmax(logits,axis=1)[0]
    predIdx = int(np.argmax(probs))
    return predIdx,probs
