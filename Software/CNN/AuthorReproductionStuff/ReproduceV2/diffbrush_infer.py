"""
diffbrush_infer.py -- CPU, single-process inference around DiffBrush
(Dai et al., ICCV 2025; github.com/dailenson/DiffBrush, MIT).

DiffBrush generates a WHOLE TEXT LINE in one diffusion pass (unlike One-DM's
per-word), with a content-decoupled style encoder that transfers writer
style more strongly. So there is no per-word composition step -- one call
per line.

Vendored under  NOGIT/_vendor/  :
  DiffBrush/                       the repo (code + files/unifont.pickle)
  ckpt/DiffBrush-ckpt.pt           the UNet weights
  ckpt/sd15/vae/                   AutoencoderKL (shared with One-DM)
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from _env import NOGIT_DIR

VENDOR = NOGIT_DIR / "_vendor"
DB_REPO = VENDOR / "DiffBrush"
CKPT = VENDOR / "ckpt" / "DiffBrush-ckpt.pt"
VAE_DIR = VENDOR / "ckpt" / "sd15"
UNIFONT = DB_REPO / "files" / "unifont.pickle"

_LETTERS = " _!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_FIXED_LEN = 1024
_H = 64
_WRITER_NUMS = 496


def _patch_resnet_weights():
    import torchvision.models as tvm
    if getattr(tvm.resnet18, "_db_patched", False):
        return
    orig = tvm.resnet18
    def resnet18(*a, **kw):
        kw["weights"] = None
        kw.pop("pretrained", None)
        return orig(*a, **kw)
    resnet18._db_patched = True
    tvm.resnet18 = resnet18


class DiffBrush:
    def __init__(self, device="cpu", steps=30, eta=0.0):
        if not CKPT.exists():
            raise FileNotFoundError(f"{CKPT} missing -- see ../REVISED_METHOD.md")
        self.device = torch.device(device)
        self.steps = steps
        self.eta = eta

        if str(DB_REPO) not in sys.path:
            sys.path.insert(0, str(DB_REPO))
        _patch_resnet_weights()

        from models.unet import UNetModel
        from models.diffusion import Diffusion
        from diffusers import AutoencoderKL

        # models/loss.py Proxy_Anchor hardcodes .cuda() on its proxy params
        # (train-only metric-learning state). Neutralise it for CPU
        # construction, then restore.
        _cuda = torch.Tensor.cuda
        torch.Tensor.cuda = lambda self, *a, **k: self
        try:
            self.unet = UNetModel(
                in_channels=4, model_channels=512, out_channels=4,
                num_res_blocks=1, attention_resolutions=(1, 1),
                channel_mult=(1, 1), num_heads=4, context_dim=512,
                nb_classes=_WRITER_NUMS,
            ).to(self.device)
        finally:
            torch.Tensor.cuda = _cuda
        sd = torch.load(CKPT, map_location="cpu")
        sd = sd.get("model", sd) if isinstance(sd, dict) else sd
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        missing, unexpected = self.unet.load_state_dict(sd, strict=False)
        miss_real = [m for m in missing if "proxy" not in m]  # proxies are train-only
        if miss_real:
            print(f"[DiffBrush] {len(miss_real)} missing keys e.g. {miss_real[:3]}")
        self.unet.eval()

        self.vae = AutoencoderKL.from_pretrained(str(VAE_DIR), subfolder="vae").to(self.device)
        self.vae.requires_grad_(False)
        self.diffusion = Diffusion(device=self.device)

        with open(UNIFONT, "rb") as f:
            syms = pickle.load(f)
        syms = {s["idx"][0]: s["mat"].astype(np.float32) for s in syms}
        self._sym = {c: torch.from_numpy(syms[ord(c)]).float() for c in _LETTERS}
        self._l2i = {c: i for i, c in enumerate(_LETTERS)}

    def _content(self, text):
        safe = "".join(c if c in self._l2i else " " for c in text) or " "
        mats = torch.stack([self._sym[c] for c in safe])       # [L,16,16]
        return (1.0 - mats).unsqueeze(0), len(safe)             # [1,L,16,16]

    def _style(self, style_img):
        """One wide (>512 px) grayscale strip of the author's writing,
        height 64, values /255 (white bg ~1, ink ~0). -> [1,1,64,W]"""
        g = style_img.convert("L")
        if g.height != _H:
            g = g.resize((max(1, round(g.width * _H / g.height)), _H),
                         Image.Resampling.BILINEAR)
        if g.width > _FIXED_LEN:
            g = g.crop((0, 0, _FIXED_LEN, _H))
        arr = np.asarray(g, np.float32) / 255.0
        return torch.from_numpy(arr)[None, None].to(self.device)  # [1,1,64,W]

    @torch.no_grad()
    def generate_line(self, text, style_img, seed=0):
        g = torch.Generator().manual_seed(int(seed))
        content, _ = self._content(text)
        content = content.to(self.device)
        style = self._style(style_img)
        x = torch.randn((1, 4, _H // 8, _FIXED_LEN // 8), generator=g).to(self.device)
        img = self.diffusion.ddim_sample(
            self.unet, self.vae, 1, x, style, content, self.steps, self.eta)
        arr = (img[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        im = Image.fromarray(arr).convert("L")
        # trim the fixed-length right-side padding to the ink
        a = np.asarray(im)
        cols = np.where((a < 235).any(axis=0))[0]
        if len(cols):
            im = im.crop((0, 0, min(im.width, cols.max() + 12), im.height))
        return im
