"""
onedm_infer.py -- CPU, single-process inference wrapper around One-DM
(Dai et al., ECCV 2024; github.com/dailenson/One-DM, MIT).

The upstream test.py is torchrun/DDP + CUDA-only and drives a fixed dataset.
This wrapper skips all of that: build the UNet, load the checkpoint on CPU,
load an SD-1.5 VAE, and expose `generate_word(word, style_img)` -> PIL.

Everything it needs lives under  NOGIT/_vendor/  :
  One-DM/                     the cloned repo (code)
  ckpt/onedm_models/One-DM-ckpt.pt
  ckpt/sd15/vae/             AutoencoderKL (config.json + safetensors)
  ckpt/unifont.pickle        GNU-Unifont content symbols
  ckpt/style/img|laplace/<author>/*.png   real IAM64 style crops
See ../REVISED_METHOD.md section 7.
"""

import functools
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from _env import NOGIT_DIR

VENDOR = NOGIT_DIR / "_vendor"
ONEDM_REPO = VENDOR / "One-DM"
CKPT_DIR = VENDOR / "ckpt"
UNET_CKPT = CKPT_DIR / "onedm_models" / "One-DM-ckpt.pt"
VAE_DIR = CKPT_DIR / "sd15"
UNIFONT = CKPT_DIR / "unifont.pickle"
STYLE_DIR = CKPT_DIR / "style"

# One-DM's fixed alphabet + geometry (data_loader/loader.py)
_LETTERS = "_Only thewigsofrcvdampbkuq.A-210xT5'MDL,RYHJ\"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%"
_LAPLACE_K = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
_STYLE_LEN = 352
_H = 64


def _patch_resnet_weights():
    """fusion.py builds torchvision resnet18 with ImageNet weights (a
    download) only to overwrite them with the checkpoint -- disable that."""
    import torchvision.models as tvm
    if getattr(tvm.resnet18, "_onedm_patched", False):
        return
    orig = tvm.resnet18
    def resnet18(*a, **kw):
        kw["weights"] = None
        kw.pop("pretrained", None)
        return orig(*a, **kw)
    resnet18._onedm_patched = True
    tvm.resnet18 = resnet18


class OneDM:
    _shared = None

    def __init__(self, device="cpu", steps=40, eta=0.0):
        if not UNET_CKPT.exists():
            raise FileNotFoundError(
                f"{UNET_CKPT} missing -- see ../REVISED_METHOD.md section 7.")
        self.device = torch.device(device)
        self.steps = steps
        self.eta = eta

        if str(ONEDM_REPO) not in sys.path:
            sys.path.insert(0, str(ONEDM_REPO))
        _patch_resnet_weights()

        from models.unet import UNetModel
        from models.diffusion import Diffusion
        from diffusers import AutoencoderKL

        self.unet = UNetModel(
            in_channels=4, model_channels=512, out_channels=4,
            num_res_blocks=1, attention_resolutions=(1, 1),
            channel_mult=(1, 1), num_heads=4, context_dim=512,
        ).to(self.device)
        sd = torch.load(UNET_CKPT, map_location="cpu")
        sd = sd.get("model", sd) if isinstance(sd, dict) else sd
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        missing, unexpected = self.unet.load_state_dict(sd, strict=False)
        if missing:
            print(f"[One-DM] {len(missing)} missing keys (e.g. {missing[:2]})")
        self.unet.eval()

        self.vae = AutoencoderKL.from_pretrained(str(VAE_DIR), subfolder="vae").to(self.device)
        self.vae.requires_grad_(False)
        self.diffusion = Diffusion(device=self.device)

        with open(UNIFONT, "rb") as f:
            syms = pickle.load(f)
        syms = {s["idx"][0]: s["mat"].astype(np.float32) for s in syms}
        self._sym = {c: torch.from_numpy(syms[ord(c)]).float() for c in _LETTERS}
        self._l2i = {c: i for i, c in enumerate(_LETTERS)}

    # -- content ------------------------------------------------------------
    def _content(self, word):
        safe = "".join(c if c in self._l2i else " " for c in word) or " "
        mats = torch.stack([self._sym[c] for c in safe])       # [L,16,16]
        return (1.0 - mats).unsqueeze(0), len(safe)             # [1,L,16,16]

    # -- style ------------------------------------------------------------
    def _style_tensors(self, style_img):
        """style_img: PIL L, ink dark on light. -> style,[1,1,64,W] and its
        laplacian, matched to One-DM's /255 float convention."""
        g = style_img.convert("L")
        if g.height != _H:
            g = g.resize((max(1, round(g.width * _H / g.height)), _H),
                         Image.Resampling.BILINEAR)
        arr = np.asarray(g, np.float32) / 255.0
        if arr.shape[1] > _STYLE_LEN:
            arr = arr[:, :_STYLE_LEN]
        s = torch.from_numpy(arr)[None, None]                   # [1,1,64,W]
        lap = torch.nn.functional.conv2d(s, _LAPLACE_K, padding=1)
        lap = lap.clamp(0, 1)                                   # match stored maps
        return s.to(self.device), lap.to(self.device)

    @torch.no_grad()
    def generate_word(self, word, style_img, seed=0):
        g = torch.Generator().manual_seed(int(seed))
        content, L = self._content(word)
        content = content.to(self.device)
        style, laplace = self._style_tensors(style_img)
        x = torch.randn((1, 4, _H // 8, (L * 32) // 8), generator=g).to(self.device)
        img = self.diffusion.ddim_sample(
            self.unet, self.vae, 1, x, style, laplace, content,
            self.steps, self.eta)                               # [1,3,64,L*32]
        arr = (img[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("L")


def _author_crops(author_id):
    d = STYLE_DIR / "img" / str(author_id)
    if not d.is_dir():
        return []
    ims = [Image.open(c).convert("L") for c in sorted(d.glob("*.png"))]
    ims.sort(key=lambda im: -im.width)
    return ims


def stitched_style(author_id, target_w=_STYLE_LEN, gap=8, seed=0):
    """One-DM's style encoder can consume up to style_len (352) px of style
    context; a single IAM word crop is ~120 px. Stitch a few crops of this
    author side by side to fill it -- more context = better style transfer."""
    ims = _author_crops(author_id)
    if not ims:
        return None
    rng = random.Random(seed)
    # prefer several NARROW crops (more distinct letterforms in the context
    # window) over one wide one
    pool = sorted(ims, key=lambda im: im.width)[:14]
    rng.shuffle(pool)
    picked, w = [], 0
    for im in pool:
        if im.height != _H:
            im = im.resize((max(1, round(im.width * _H / im.height)), _H))
        if w + im.width > target_w and picked:
            break
        picked.append(im)
        w += im.width + gap
    canvas = Image.new("L", (min(target_w, max(1, w - gap)), _H), 255)
    x = 0
    for im in picked:
        if x >= canvas.width:
            break
        canvas.paste(im, (x, 0))
        x += im.width + gap
    return canvas


def load_style_crops(author_id, n=3, min_w=200, seed=0):
    """A stitched wide style image (index 0) plus a couple of single crops
    as alternates for the re-ranker."""
    out = []
    s = stitched_style(author_id, seed=seed)
    if s is not None:
        out.append(s)
    out += [im for im in _author_crops(author_id) if im.width >= min_w][: max(0, n - 1)]
    return out or _author_crops(author_id)[:n]


def load_style_crop(author_id, seed=0, min_w=128):
    c = load_style_crops(author_id, n=1)
    return c[0] if c else None
