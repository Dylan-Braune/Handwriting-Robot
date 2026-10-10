"""Render one arbitrary sentence in an author's style, write real G-code, and
read the G-code render back with the recognizer.

    python write_sentence.py yeukita "Good morning. I hope that everyone is doing well today."
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths  # noqa: F401

from np_inference.text_model import PaperCRNNNumpy, ReadText, CharAcc, WordAcc
from np_inference.author_model import AuthorClassifierCNNNumpy, ClassifyImage
import SynthesizeHandwriting as SY

author = sys.argv[1] if len(sys.argv) > 1 else "yeukita"
text = sys.argv[2] if len(sys.argv) > 2 else \
    "Good morning. I hope that everyone is doing well today."
tries = int(sys.argv[3]) if len(sys.argv) > 3 else 30

out = HERE / "out" / "sentences"
out.mkdir(parents=True, exist_ok=True)

prof = json.loads((SY.PROFILE_DIR / f"{author}.json").read_text(encoding="utf-8"))
reader = PaperCRNNNumpy()
am = AuthorClassifierCNNNumpy()
mapping = am.author_mapping
idxToAuthor = {v: k for k, v in mapping.items()}

print(f"author : {author}")
print(f"profile: {len(prof.get('glyphs', {}))} letters, lam={prof.get('legibilityLambda')}, "
      f"slant={prof.get('slantDeg')}, conn={prof.get('connectedness')}")
print(f"want   : {text}")

traj = SY.SynthesizeJointBestOf(
    author, text, prof, nTries=tries, mmPerXh=4.0, lineWidthMm=10_000.0,
    jitter=0.5, seed=0, reader=reader, authorModel=am, authorMapping=mapping,
    pxPerMm=18.0)

# 1. the vector synthesis (what the pen path is)
img = SY.RenderTrajectory(traj, pxPerMm=18.0, profile=prof)
img.save(out / f"{author}_goodmorning_synth.png")

# 2. the real machine artifact: G-code, parsed back, rendered
cfg = SY.GantryConfig()
gpath = out / f"{author}_goodmorning.gcode"
SY.WriteGcode(traj, cfg, str(gpath), title=author)
strokes = SY.ParseGcode(str(gpath))
gtraj = SY.Trajectory(strokes, dict(traj.meta))
gimg = SY.RenderTrajectory(gtraj, pxPerMm=18.0, profile=prof)
gimg.save(out / f"{author}_goodmorning_gcode.png")

got = ReadText(gimg, reader)
pred, probs = ClassifyImage(gimg, am)
print(f"read   : {got}")
print(f"char   : {CharAcc(got, text)*100:.1f}%   word {WordAcc(got, text)*100:.1f}%")
print(f"author : {'YES' if idxToAuthor[pred] == author else 'no'} "
      f"(p={probs[mapping[author]]:.2f}, said {idxToAuthor[pred]})")
print(f"gcode  : {gpath}  ({len(gpath.read_text(encoding='utf-8').splitlines())} lines, "
      f"{len(strokes)} strokes)")
print(f"images : {out / (author + '_goodmorning_synth.png')}")
print(f"         {out / (author + '_goodmorning_gcode.png')}")
print("glyph sources:", traj.meta.get("glyphSources"))
