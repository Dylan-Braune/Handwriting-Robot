"""G-code check for one author: synthesise, write real G-code, render it back,
score with the project's own recognizer and writer-ID classifier.

This is the deliverable path (the machine artifact), so it is what a glyph
change has to be judged on.

    python gcode_check.py yeukita                 # all 6 novel sentences
    python gcode_check.py yeukita --sentences 2
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths  # noqa: F401  (sys.path: our folder, then the originals)

from PIL import Image

from np_inference.text_model import PaperCRNNNumpy, ReadText, CharAcc, WordAcc
from np_inference.author_model import AuthorClassifierCNNNumpy, ClassifyImage
import SynthesizeHandwriting as SY

NOVEL = [
    "The gantry writes this sentence for the first time today",
    "My robot copies handwriting from ten different authors",
    "Please bring the blue notebook and a sharp pencil",
    "Every careful measurement makes the next result better",
    "We tested the machine on Monday and it worked quietly",
    "Nothing about this line appears in the training pages",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("author")
    ap.add_argument("--sentences", type=int, default=0, help="0 = all")
    ap.add_argument("--tries", type=int, default=30)
    ap.add_argument("--pxPerMm", type=float, default=18.0)
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    a = args.author
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    prof = json.loads((SY.PROFILE_DIR / f"{a}.json").read_text(encoding="utf-8"))
    reader = PaperCRNNNumpy()
    am = AuthorClassifierCNNNumpy()
    mapping = am.author_mapping
    idxToAuthor = {v: k for k, v in mapping.items()}
    cfg = SY.GantryConfig()
    tmp = out / "_tmp.gcode"
    sents = NOVEL[:args.sentences] if args.sentences else NOVEL

    print(f"author {a} | profile glyphs: {len(prof.get('glyphs', {}))} letters, "
          f"lam={prof.get('legibilityLambda')}")
    print(f"judges: {reader.checkpoint_path.name} / {am.checkpoint_path.name}\n")

    rows = []
    for si, text in enumerate(sents):
        traj = SY.SynthesizeJointBestOf(
            a, text, prof, nTries=args.tries, mmPerXh=4.0, lineWidthMm=10_000.0,
            jitter=0.5, seed=100 * si, reader=reader, authorModel=am,
            authorMapping=mapping, pxPerMm=args.pxPerMm)
        SY.WriteGcode(traj, cfg, str(tmp), title=a)
        strokes = SY.ParseGcode(str(tmp))
        gtraj = SY.Trajectory(strokes, dict(traj.meta))
        gimg = SY.RenderTrajectory(gtraj, pxPerMm=args.pxPerMm, profile=prof)
        pred, probs = ClassifyImage(gimg, am)
        got = ReadText(gimg, reader)
        cAcc = CharAcc(got, text)
        wAcc = WordAcc(got, text)
        idOk = idxToAuthor[pred] == a
        rows.append(dict(text=text, charAcc=cAcc, wordAcc=wAcc, idOk=idOk,
                         pred=idxToAuthor[pred], pAuthor=float(probs[mapping[a]]),
                         got=got))
        print(f"  [{si}] char {cAcc*100:5.1f}%  word {wAcc*100:5.1f}%  "
              f"author {'YES' if idOk else 'no ':>3} "
              f"(p={probs[mapping[a]]:.2f}, said {idxToAuthor[pred]})")
        print(f"      want: {text}\n      got : {got}")
        gimg.save(out / f"{a}_{args.tag}_{si}.png")
        (out / f"{a}_{args.tag}_{si}.gcode").write_text(
            tmp.read_text(encoding="utf-8"), encoding="utf-8")

    n = len(rows)
    print(f"\n--- {a} [{args.tag}] ---")
    print(f"  text char : {sum(r['charAcc'] for r in rows)/n*100:5.1f}%")
    print(f"  text word : {sum(r['wordAcc'] for r in rows)/n*100:5.1f}%")
    print(f"  writer-ID : {sum(r['idOk'] for r in rows)/n*100:5.1f}%  ({n} samples)")
    (out / f"result_{args.tag}.json").write_text(
        json.dumps(dict(author=a, tag=args.tag, rows=rows), indent=1),
        encoding="utf-8")
    print(f"  wrote {out}/result_{args.tag}.json")


if __name__ == "__main__":
    main()
