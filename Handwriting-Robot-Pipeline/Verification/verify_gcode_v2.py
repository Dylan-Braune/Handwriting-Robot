"""G-code text + author accuracy for every available author, scored with BOTH
writer-ID heads.

Why this exists
---------------
verify_end_to_end_fast.py answered "does synthesis work" with the shape head as
the only judge. Now that author_shape_10new_weights.pt has been retrained
end-to-end (previously its backbone was frozen at OCR features, which is why
owen collapsed into yeukita), the judge itself has changed -- so the previous
writer-ID figures on synthesised text are no longer comparable and must be
re-measured.

Crucially this reports the INK head as well. The ink classifier was never part
of the fix, so it is an independent instrument: if synthesis genuinely improved,
an independent judge should see it too. If only the changed judge moves, the
gain is in the judge rather than in the drawing.

Same scope as the previous run (3 novel sentences, best-of-8, 1 seed) so the
before/after numbers line up. Every line is scored twice -- on the rendered
trajectory and on the same strokes after the G-code round trip, which is what
the gantry actually draws.

Run from Software/CNN:
    python NOGIT/full_pipeline_run/verify_gcode_v2.py
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# path setup for the clean package layout
import sys as _sys
from pathlib import Path as _p
_root = _p(__file__).resolve().parent.parent
_cnn = _root / 'Software' / 'CNN'
_ar = _cnn / 'AuthorReproductionStuff'
_gantry = _root / 'Software' / 'GantryControl'
for _q in (_cnn, _ar, _ar / 'np_inference', _gantry):
    if str(_q) not in _sys.path:
        _sys.path.insert(0, str(_q))


THIS_DIR = Path(__file__).resolve().parent
CNN_DIR = _cnn

import SynthesizeHandwriting as SY  # noqa: E402
from Evaluate import (  # noqa: E402
    GcodeRoundTrip, LoadTextModel, ReadText, CharAcc, WordAcc,
)
from TrainText import resize_line_image_fixed, tensor_from_resized  # noqa: E402
from TrainAuthor import AuthorClassifierCNN, StrokeNormalize  # noqa: E402
from np_inference.text_model import PaperCRNNNumpy  # noqa: E402
from np_inference.author_model import AuthorClassifierCNNNumpy  # noqa: E402


WEIGHTS = _cnn / "weights"
SHAPE_W = WEIGHTS / "author_shape_10new_weights.pt"
INK_W = WEIGHTS / "author_classifier_10new_weights.pt"
TEXT_W = WEIGHTS / "paper_cnn_bilstm_ctc_joint_best.pt"
OUT_IMG = THIS_DIR / "e2e_v2"

SENTENCES = [
    "The gantry writes this sentence for the first time today",
    "My robot copies handwriting from ten different authors",
    "Please bring the blue notebook and a sharp pencil",
]
N_TRIES = 8
MM_PER_XH = 4.0
PX_PER_MM = 18.0
ALL13 = ["150", "151", "152", "153", "384", "551", "552", "588",
         "abhinav", "dylan", "owen", "thiya", "yeukita"]
T0 = time.time()


def say(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


class TorchHead:
    """One writer-ID head, with its own required preprocessing."""

    def __init__(self, path, normalize):
        self.path = path
        self.normalize = normalize
        sd = torch.load(path, map_location="cpu", weights_only=False)
        self.mapping = sd["author_mapping"]
        self.idx2author = {v: k for k, v in self.mapping.items()}
        m = AuthorClassifierCNN(num_authors=len(self.mapping))
        m.load_state_dict(sd["model_state_dict"])
        m.eval()
        self.model = m

    def __call__(self, pil):
        img = pil
        if self.normalize:
            n = StrokeNormalize(np.array(pil.convert("L")))
            if n is not None:
                img = n
        t = tensor_from_resized(resize_line_image_fixed(img)).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(t)
        p = torch.softmax(logits, dim=1)[0].numpy()
        i = int(p.argmax())
        return self.idx2author[i], float(p[i]), p


def main():
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    say(f"shape judge (FIXED): {SHAPE_W.name}")
    say(f"ink judge (independent): {INK_W.name}")
    shape = TorchHead(SHAPE_W, normalize=True)
    ink = TorchHead(INK_W, normalize=False)
    say("writer-ID heads loaded")

    textModel = LoadTextModel(torch.device("cpu"))
    say("text judge loaded")
    synthReader = PaperCRNNNumpy()
    synthAuthor = AuthorClassifierCNNNumpy()
    say("numpy synthesis scorers loaded (production path)")

    profiles = SY.LoadAllProfiles()
    cfg = SY.GantryConfig()
    say(f"profiles: {len(profiles)}")

    rows = []
    for ai, a in enumerate(sorted(profiles), 1):
        prof = profiles[a]
        for si, text in enumerate(SENTENCES):
            t0 = time.time()
            traj = SY.SynthesizeJointBestOf(
                a, text, prof, nTries=N_TRIES, mmPerXh=MM_PER_XH,
                lineWidthMm=10_000.0, jitter=0.5, seed=si,
                reader=synthReader, authorModel=synthAuthor,
                authorMapping=synthAuthor.author_mapping, pxPerMm=PX_PER_MM)
            img = SY.RenderTrajectory(traj, pxPerMm=PX_PER_MM, profile=prof)
            gimg = GcodeRoundTrip(traj, cfg, str(OUT_IMG / "_tmp_v2.gcode"), prof)

            rec = dict(author=a, sentence=text, seed=si)
            for tag, im in (("render", img), ("gcode", gimg)):
                if im is None:
                    continue
                sp, sc, _ = shape(im)
                ip, ic, _ = ink(im)
                got = ReadText(textModel, im, device=torch.device("cpu"))
                rec[f"shape_{tag}"] = sp
                rec[f"shape_conf_{tag}"] = sc
                rec[f"ink_{tag}"] = ip
                rec[f"ink_conf_{tag}"] = ic
                rec[f"char_{tag}"] = CharAcc(got, text)
                rec[f"word_{tag}"] = WordAcc(got, text)
                rec[f"read_{tag}"] = got
            if si == 0:
                img.save(OUT_IMG / f"{a}_synth.png")
                if gimg is not None:
                    gimg.save(OUT_IMG / f"{a}_gcode.png")
            rows.append(rec)
            say(f"  {a} s{si+1}/{len(SENTENCES)}: "
                f"gcode shape {'OK' if rec.get('shape_gcode')==a else 'MISS'} "
                f"ink {'OK' if rec.get('ink_gcode')==a else 'MISS'} "
                f"char {rec.get('char_gcode',0)*100:.0f}%  ({time.time()-t0:.0f}s)")
        say(f"== {ai}/{len(profiles)} done")

    # ---------------- summary ------------------------------------------------
    def mean(vals):
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else float("nan")

    print("\n" + "=" * 104, flush=True)
    print("G-CODE WRITER-ID AND TEXT ACCURACY  (shape = new fixed judge, ink = independent)",
          flush=True)
    print("=" * 104, flush=True)
    hdr = (f"{'author':<9}{'shape ID':>10}{'ink ID':>9}{'char':>8}{'word':>8}"
           f"{'shape ID(r)':>12}{'ink ID(r)':>10}{'char(r)':>9}")
    print(hdr, flush=True)
    per = {}
    for a in ALL13:
        sub = [r for r in rows if r["author"] == a]
        if not sub:
            continue
        g = lambda k: [r.get(k) for r in sub]
        sh = mean([r.get("shape_gcode") == a for r in sub])
        ik = mean([r.get("ink_gcode") == a for r in sub])
        ch = mean(g("char_gcode"))
        wo = mean(g("word_gcode"))
        shr = mean([r.get("shape_render") == a for r in sub])
        ikr = mean([r.get("ink_render") == a for r in sub])
        chr_ = mean(g("char_render"))
        per[a] = dict(shape_gcode=sh, ink_gcode=ik, char_gcode=ch, word_gcode=wo,
                      shape_render=shr, ink_render=ikr, char_render=chr_, n=len(sub))
        print(f"{a:<9}{sh*100:9.0f}%{ik*100:8.0f}%{ch*100:7.0f}%{wo*100:7.0f}%"
              f"{shr*100:11.0f}%{ikr*100:9.0f}%{chr_*100:8.0f}%", flush=True)
    print("-" * 104, flush=True)
    agg = dict(
        shape_gcode=mean([v["shape_gcode"] for v in per.values()]),
        ink_gcode=mean([v["ink_gcode"] for v in per.values()]),
        char_gcode=mean([v["char_gcode"] for v in per.values()]),
        word_gcode=mean([v["word_gcode"] for v in per.values()]),
        shape_render=mean([v["shape_render"] for v in per.values()]),
        ink_render=mean([v["ink_render"] for v in per.values()]),
        char_render=mean([v["char_render"] for v in per.values()]),
    )
    print(f"{'MEAN':<9}{agg['shape_gcode']*100:9.0f}%{agg['ink_gcode']*100:8.0f}%"
          f"{agg['char_gcode']*100:7.0f}%{agg['word_gcode']*100:7.0f}%"
          f"{agg['shape_render']*100:11.0f}%{agg['ink_render']*100:9.0f}%"
          f"{agg['char_render']*100:8.0f}%", flush=True)

    prev = THIS_DIR / "stage5_fast_summary.json"
    if prev.exists():
        p = json.loads(prev.read_text(encoding="utf-8"))
        print("\n" + "=" * 104)
        print("BEFORE vs AFTER THE JUDGE FIX (shape head on synthesised text)")
        print("=" * 104)
        pm = p["mean"]
        print(f"  {'metric':<28}{'before':>10}{'after':>10}")
        print(f"  {'shape writer-ID (render)':<28}{pm['idR']*100:9.0f}%"
              f"{agg['shape_render']*100:9.0f}%")
        print(f"  {'shape writer-ID (G-code)':<28}{pm['idG']*100:9.0f}%"
              f"{agg['shape_gcode']*100:9.0f}%")
        print(f"  {'text char (render)':<28}{pm['charR']*100:9.0f}%"
              f"{agg['char_render']*100:9.0f}%")
        print(f"  {'text char (G-code)':<28}{pm['charG']*100:9.0f}%"
              f"{agg['char_gcode']*100:9.0f}%")
        print(f"  {'text word (render)':<28}{pm['wordR']*100:9.0f}%"
              f"{agg['word_gcode']*100:9.0f}%")
        pw = p["per_author"]
        print(f"\n  Per author, shape writer-ID on the rendered line:")
        print(f"    {'author':<10}{'before':>9}{'after':>9}{'delta':>9}")
        for a in ALL13:
            if a in pw and a in per:
                b, aft = pw[a]["idR"] * 100, per[a]["shape_render"] * 100
                print(f"    {a:<10}{b:8.0f}%{aft:8.0f}%{aft-b:+8.0f}")

    out = dict(sentences=SENTENCES, tries=N_TRIES, per_author=per, mean=agg,
               rows=rows, shape_weights=SHAPE_W.name, ink_weights=INK_W.name,
               text_weights=TEXT_W.name,
               elapsed_seconds=round(time.time() - T0, 1))
    (THIS_DIR / "gcode_v2_summary.json").write_text(json.dumps(out, indent=2),
                                                    encoding="utf-8")
    with open(THIS_DIR / "gcode_v2_per_line.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    say("wrote gcode_v2_summary.json + gcode_v2_per_line.csv")
    say("DONE")


if __name__ == "__main__":
    main()
