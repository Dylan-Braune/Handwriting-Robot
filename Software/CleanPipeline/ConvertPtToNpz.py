#one-time conversion: reads existing torch-trained .pt checkpoints and saves
#them as .npz, in the exact same format TrainScratch.py's own checkpoints
#use (so an old converted weight and a newly scratch-trained one are fully
#interchangeable, loaded by the exact same code path). this is the ONLY
#place in the project that still parses the .pt pickle/zip format -- once
#you've run this and repointed Classify.py at the .npz results, the .pt
#files on disk are never read by anything again (they stay as the original
#source, just unused by the live pipeline). this is NOT imported by
#Classify.py/Server.py/anything else -- run it by hand, once, per checkpoint.
import zipfile
from pathlib import Path

import numpy as np

CNN_DATA_DIR = Path(__file__).resolve().parent.parent/"CNN"
WEIGHTS_DIR = CNN_DATA_DIR/"weights"

STORAGE_DTYPES = {
    "FloatStorage": np.dtype("<f4"),"DoubleStorage": np.dtype("<f8"),"HalfStorage": np.dtype("<f2"),
    "LongStorage": np.dtype("<i8"),"IntStorage": np.dtype("<i4"),"ShortStorage": np.dtype("<i2"),
    "CharStorage": np.dtype("<i1"),"ByteStorage": np.dtype("<u1"),"BoolStorage": np.dtype("?"),
}


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
import pickle
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
        tag,storage_type,key,location,numel = pid
        assert tag=="storage"
        raw = self._zf.read(f"{self._data_prefix}data/{key}")
        return np.frombuffer(raw,dtype=storage_type.dtype,count=numel)


#unpickles a .pt checkpoint's full object graph, tensors as numpy arrays
def LoadCheckpoint(pt_path):
    import io
    with zipfile.ZipFile(str(pt_path),"r") as zf:
        pkl_name = next(n for n in zf.namelist() if n.endswith("data.pkl"))
        prefix = pkl_name[:-len("data.pkl")]
        with zf.open(pkl_name) as f:
            return TorchUnpickler(io.BytesIO(f.read()),zf,prefix).load()


#converts one .pt file to .npz, same layout TrainScratch.SaveCheckpoint uses:
#state dict keys as-is, author_mapping (if present) flattened to __author_names__
def ConvertFile(ptPath,npzPath):
    obj = LoadCheckpoint(ptPath)
    authorMapping = obj.get("author_mapping") if isinstance(obj,dict) else None
    sd = obj["model_state_dict"] if isinstance(obj,dict) and "model_state_dict" in obj else obj
    payload = dict(sd)
    if authorMapping:
        names = sorted(authorMapping,key=lambda a: authorMapping[a])
        payload["__author_names__"] = np.array(names)
    np.savez(str(npzPath),**payload)
    print(f"  {ptPath.name} -> {npzPath.name}  ({len(sd)} tensors"
          f"{', with author_mapping' if authorMapping else ''})")


#every checkpoint the live pipeline actually references and that exists on disk
CHECKPOINTS = [
    "paper_cnn_bilstm_ctc_joint_best.pt",
    "paper_cnn_bilstm_ctc_hf_best.pt",
    "paper_cnn_bilstm_ctc_personal_best.pt",
    "author_shape_10new_weights.pt",
    "author_classifier_10new_weights.pt",
]


def Main():
    print(f"[ConvertPtToNpz] converting {len(CHECKPOINTS)} checkpoint(s) in {WEIGHTS_DIR}")
    for name in CHECKPOINTS:
        ptPath = WEIGHTS_DIR/name
        if not ptPath.exists():
            print(f"  skip (not found): {name}")
            continue
        npzPath = ptPath.with_suffix(".npz")
        ConvertFile(ptPath,npzPath)
    print("\nDone. Point Classify.py/Evaluate.py/Server.py/BuildStyleProfile.py at the "
          ".npz names and the .pt unpickling code can be deleted.")


if __name__=="__main__":
    Main()
