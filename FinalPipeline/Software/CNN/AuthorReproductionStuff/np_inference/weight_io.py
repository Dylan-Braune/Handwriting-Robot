"""Load PyTorch .pt checkpoints (zip archive: pickled object graph + raw
tensor storage bytes) into plain numpy arrays, without importing torch.

A .pt file is a zip with `<archive>/data.pkl` (the pickled object graph,
tensors replaced by persistent-id placeholders) plus one `<archive>/data/<key>`
member per tensor storage (raw bytes, dtype/shape come from the pickle)."""

import io
import pickle
import struct
import zipfile
from pathlib import Path

import numpy as np

_STORAGE_DTYPES = {
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


class _StorageStub:
    """Placeholder returned by find_class for a torch.*Storage class. Real
    bytes arrive later via persistent_load; this just carries the dtype."""

    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, *args, **kwargs):
        # Some pickles call the storage "class" directly (rare); keep it a
        # harmless no-op returning self so unpickling doesn't crash.
        return self


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad=False,
                        backward_hooks=None, metadata=None):
    """Reconstruct the final numpy array from a storage array + view info.
    PyTorch strides are in ELEMENTS; as_strided needs strides in BYTES."""
    itemsize = storage.dtype.itemsize
    byte_strides = tuple(s * itemsize for s in stride)
    if len(size) == 0:
        return storage[storage_offset:storage_offset + 1].reshape(()).copy()
    view = np.lib.stride_tricks.as_strided(
        storage[storage_offset:], shape=tuple(size), strides=byte_strides,
    )
    return view.copy()


class _TorchUnpickler(pickle.Unpickler):
    def __init__(self, file, zf, data_prefix):
        super().__init__(file)
        self._zf = zf
        self._data_prefix = data_prefix

    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if module == "torch" and name in _STORAGE_DTYPES:
            return _StorageStub(_STORAGE_DTYPES[name])
        if module == "collections" and name == "OrderedDict":
            import collections
            return collections.OrderedDict
        if module == "torch._utils" and name == "_rebuild_parameter":
            return lambda data, requires_grad, backward_hooks: data
        # Anything else (numpy scalars, builtins, etc.) -- let pickle resolve
        # it normally; only torch-specific classes need interception.
        return super().find_class(module, name)

    def persistent_load(self, pid):
        # Format: ('storage', storage_type_stub, key, location, numel)
        tag, storage_type, key, location, numel = pid
        assert tag == "storage"
        dtype = storage_type.dtype
        raw = self._zf.read(f"{self._data_prefix}data/{key}")
        arr = np.frombuffer(raw, dtype=dtype, count=numel)
        return arr


def load_pt(pt_path):
    """Unpickle a .pt checkpoint's full object graph, tensors as numpy arrays."""
    pt_path = str(pt_path)
    with zipfile.ZipFile(pt_path, "r") as zf:
        names = zf.namelist()
        pkl_name = next(n for n in names if n.endswith("data.pkl"))
        # archive folder name varies (usually the file's stem) -- derive the
        # "<archive>/" prefix from data.pkl's own path instead of assuming it.
        prefix = pkl_name[: -len("data.pkl")]
        with zf.open(pkl_name) as f:
            buf = io.BytesIO(f.read())
            unpickler = _TorchUnpickler(buf, zf, prefix)
            return unpickler.load()


def load_state_dict_numpy(pt_path):
    """Thin wrapper: unwrap 'model_state_dict' if the checkpoint is a dict
    wrapping the state dict (as the author classifier checkpoints are)."""
    obj = load_pt(pt_path)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        obj = obj["model_state_dict"]
    return dict(obj)


def export_to_npz(pt_path, out_npz_path):
    """Convenience: dump a checkpoint's tensors to .npz (+ .json sidecar for
    any non-tensor metadata) so downstream code can skip re-parsing pickle."""
    import json

    obj = load_pt(pt_path)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        tensors = obj["model_state_dict"]
        meta = {k: v for k, v in obj.items() if k != "model_state_dict"}
    else:
        tensors = obj
        meta = {}

    out_npz_path = Path(out_npz_path)
    np.savez(out_npz_path, **{k: v for k, v in tensors.items()})

    def _jsonable(v):
        if isinstance(v, dict):
            return {str(k): _jsonable(x) for k, x in v.items()}
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.integer, np.floating)):
            return v.item()
        return v

    if meta:
        with open(out_npz_path.with_suffix(".json"), "w") as f:
            json.dump(_jsonable(meta), f, indent=2)
