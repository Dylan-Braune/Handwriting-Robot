"""
Styled handwritten-text-generation backends.

The revised method (see ../REVISED_METHOD.md, stage D) generates each WORD
with a frozen pretrained model, then the orchestrator gates it on legibility
and re-ranks on style. The generator is behind this interface so it can be
swapped without touching the rest of the pipeline.

Backends
--------
PrintArchetypeBackend   works now, no downloads. Renders each word in a
                        printed font, sheared to the author's slant and
                        scaled to the author's x-height. This is the VATr
                        "visual archetype" -- also the deterministic
                        fallback the orchestrator drops to when the neural
                        backend's candidates all fail the legibility gate.

OneDMBackend            One-DM (Dai et al., ECCV 2024). Best cursive
                        handling. Needs the pretrained checkpoint. Stub.

VATrBackend             VATr++ (Pippi et al., CVPR 2023). Fast (GAN, one
                        forward pass). Needs the pretrained checkpoint. Stub.
"""

from abc import ABC, abstractmethod

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import _env  # noqa: F401  (path setup)


# ---------------------------------------------------------------------------
# interface
# ---------------------------------------------------------------------------
class StyledHTGBackend(ABC):
    name = "base"

    def generate(self, word, style, k=6, seed=0):
        """word-level: k grayscale word images in the author's style.
        A line-level backend overrides generate_line instead and may leave
        this unimplemented."""
        raise NotImplementedError(f"{self.name} is line-level; use generate_line")

    # a line-level backend implements this instead of generate()
    # def generate_line(self, text, style, k=6, seed=0) -> list[PIL.Image]: ...

    def describe_style(self, author_id, ref_line_images, profile):
        """Build the per-author style object this backend needs.
        ref_line_images: list[PIL.Image] real held-out lines.
        profile: the StyleProfiles10/<id>.json dict.
        Default: just hand the profile through."""
        return {"author_id": author_id, "profile": profile,
                "refs": ref_line_images}


# ---------------------------------------------------------------------------
# print archetype  (works now)
# ---------------------------------------------------------------------------
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _load_font(px):
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, px)
        except OSError:
            continue
    return ImageFont.load_default()


def _shear_x(img, slant_deg):
    """Positive slant_deg leans letters to the right (like italic)."""
    if abs(slant_deg) < 0.5:
        return img
    k = np.tan(np.radians(slant_deg))
    w, h = img.size
    extra = int(abs(k) * h) + 2
    canvas = Image.new("L", (w + extra, h), 255)
    canvas.paste(img, (extra // 2 if k < 0 else 0, 0))
    # PIL affine: x' = a*x + b*y + c ; shear uses b = -k so top shifts right
    return canvas.transform(
        canvas.size, Image.Transform.AFFINE, (1, -k, k * h if k > 0 else 0, 0, 1, 0),
        resample=Image.Resampling.BILINEAR, fillcolor=255)


class PrintArchetypeBackend(StyledHTGBackend):
    name = "print_archetype"

    # cap the printed slant we imitate -- a straight-up archetype is a safer
    # legibility fallback than a hard-sheared one; the real style lean comes
    # from a neural backend.
    MAX_SHEAR_DEG = 12.0

    def __init__(self, target_cap_px=44):
        self.cap = target_cap_px           # target ascender/cap height in px

    def generate(self, word, style, k=1, seed=0):
        prof = (style or {}).get("profile", {}) if isinstance(style, dict) else {}
        slant = float(np.clip(prof.get("slantDeg", 0.0),
                              -self.MAX_SHEAR_DEG, self.MAX_SHEAR_DEG))
        render_px = 160
        font = _load_font(render_px)
        pad = render_px
        canvas = Image.new("L", (pad, pad), 255)
        bbox = ImageDraw.Draw(canvas).textbbox((0, 0), word, font=font)
        img = Image.new("L", (bbox[2] - bbox[0] + 2 * pad,
                              bbox[3] - bbox[1] + 2 * pad), 255)
        ImageDraw.Draw(img).text((pad - bbox[0], pad - bbox[1]), word,
                                 fill=0, font=font)
        crop = img.getbbox()
        if crop is None:
            return [Image.new("L", (self.cap, self.cap), 255)]
        img = img.crop(crop)
        scale = self.cap / max(1, img.height)
        img = img.resize((max(1, round(img.width * scale)),
                          max(1, round(img.height * scale))),
                         Image.Resampling.LANCZOS)
        img = _shear_x(img, slant)
        crop = img.getbbox()
        return [img.crop(crop) if crop else img]


# ---------------------------------------------------------------------------
# legacy synthesiser  (works now -- wraps the existing dffeab0 pipeline)
# ---------------------------------------------------------------------------
class LegacySynthBackend(StyledHTGBackend):
    """The current glyph-library synthesiser (SynthesizeHandwriting.py),
    wrapped so reproduce_v2's legibility gate + style re-rank + line
    composition run on top of it. This is the Tier-0 backend: no downloads,
    CTC-readable for the print hands, and the honest baseline to compare a
    neural backend against. It carries the same cursive limitation."""
    name = "legacy_synth"

    def __init__(self, mm_per_xh=4.0, px_per_mm=18.0):
        import SynthesizeHandwriting as SY
        self._SY = SY
        self.mm_per_xh = mm_per_xh
        self.px_per_mm = px_per_mm

    def _render(self, text, prof, lam, seed):
        traj = self._SY.SynthesizeText(
            text, prof, mmPerXh=self.mm_per_xh, seed=seed,
            lineWidthMm=10_000.0, legibility=lam)
        return self._SY.RenderTrajectory(traj, pxPerMm=self.px_per_mm, profile=prof)

    def generate(self, word, style, k=6, seed=0):
        prof = (style or {}).get("profile") if isinstance(style, dict) else None
        if prof is None:
            return []
        lam_sweep = [prof.get("legibilityLambda", 0.65), 0.4, 0.75, 0.55]
        return [self._render(word, prof, lam_sweep[i % len(lam_sweep)], seed + i)
                for i in range(k)]

    def generate_line(self, text, style, k=6, seed=0):
        """Synthesise the whole line at once -- the legacy pipeline's native
        mode, and far more CTC-legible than stitching word crops."""
        prof = (style or {}).get("profile") if isinstance(style, dict) else None
        if prof is None:
            return []
        lam_sweep = [prof.get("legibilityLambda", 0.65), 0.45, 0.6, 0.75, 0.9]
        return [self._render(text, prof, lam_sweep[i % len(lam_sweep)], seed + i)
                for i in range(k)]


# ---------------------------------------------------------------------------
# One-DM  (Dai et al., ECCV 2024 -- github.com/dailenson/One-DM, MIT)
# frozen pretrained model, CPU inference via onedm_infer.py
# ---------------------------------------------------------------------------
class OneDMBackend(StyledHTGBackend):
    """Word-level neural generator. Needs NOGIT/_vendor/ populated -- see
    ../REVISED_METHOD.md section 7. ~10 s per word at 30 DDIM steps on CPU."""
    name = "one_dm"

    def __init__(self, steps=25, eta=0.0):
        from onedm_infer import OneDM, load_style_crops
        self._model = OneDM(device="cpu", steps=steps, eta=eta)
        self._load_style_crops = load_style_crops

    def describe_style(self, author_id, ref_line_images, profile):
        # One "anchor" style crop drives every word so the line stays in one
        # hand; a couple of alternates give the re-ranker something to vary.
        crops = self._load_style_crops(author_id, n=3, min_w=200)
        if not crops and ref_line_images:
            crops = [ref_line_images[0]]
        return {"author_id": author_id, "profile": profile, "crops": crops}

    def generate(self, word, style, k=6, seed=0):
        crops = (style or {}).get("crops") or []
        if not crops:
            return []
        out = []
        for i in range(k):
            # anchor crop for most candidates; vary the diffusion noise
            crop = crops[0] if i < max(1, k - 1) else crops[i % len(crops)]
            out.append(self._model.generate_word(word, crop, seed=seed + i))
        return out


# ---------------------------------------------------------------------------
# DiffBrush  (Dai et al., ICCV 2025 -- github.com/dailenson/DiffBrush, MIT)
# whole-line generation, stronger style encoder than One-DM
# ---------------------------------------------------------------------------
class DiffBrushBackend(StyledHTGBackend):
    """Line-level neural generator: one diffusion pass per line, no word
    composition. Style ref = one wide strip of the author's real writing.
    ~15-40 s per line on CPU depending on steps."""
    name = "diffbrush"

    def __init__(self, steps=25, eta=0.0):
        from diffbrush_infer import DiffBrush
        self._model = DiffBrush(device="cpu", steps=steps, eta=eta)

    def describe_style(self, author_id, ref_line_images, profile):
        # widest available real line for this author, as one strip
        refs = sorted(ref_line_images or [], key=lambda im: -im.width)
        return {"author_id": author_id, "profile": profile,
                "strip": refs[0] if refs else None, "_refs": refs}

    def generate_line(self, text, style, k=6, seed=0):
        strip = (style or {}).get("strip")
        if strip is None:
            return []
        return [self._model.generate_line(text, strip, seed=seed + i)
                for i in range(k)]


# ---------------------------------------------------------------------------
# VATr++  (needs pretrained checkpoint -- github.com/aimagelab/VATr, MIT)
# ---------------------------------------------------------------------------
class VATrBackend(StyledHTGBackend):
    """
    Setup: clone github.com/aimagelab/VATr, download vatr.pth + IAM-32.pickle.
    Style input: ~15 word crops of the author. GAN -> one forward pass, so
    generate k candidates by using k different random subsets of the crops.
    """
    name = "vatr"
    VATR_ROOT = None

    def __init__(self):
        if not self.VATR_ROOT:
            raise NotImplementedError(
                "VATrBackend: set VATR_ROOT and download vatr.pth + "
                "IAM-32.pickle. See ../REVISED_METHOD.md section 7 step 6.")

    def generate(self, word, style, k=6, seed=0):
        raise NotImplementedError


BACKENDS = {b.name: b for b in (PrintArchetypeBackend, LegacySynthBackend,
                                OneDMBackend, DiffBrushBackend, VATrBackend)}
