"""One image per photographed writer of the same sentence, at the shipped settings."""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
import numpy as np
from PIL import Image
import paths, recognise as REC, synthesise as SYN

TEXT   = "remove the inconsistencies on the belt pulley"
NTRIES = 14          # the tuned value, as the server uses
SEED   = 23

reader = REC.PaperCRNNNumpy(checkpointPath=REC.DEFAULT_TEXT_WEIGHTS)
shape  = REC.AuthorClassifierCNNNumpy(checkpointPath=REC.SHAPE_AUTHOR_WEIGHTS)
smap   = shape.authorMapping

rows = []
for i, w in enumerate(paths.PersonalWriters, 1):
    prof = json.load(open(paths.Profiles/("%s.json" % w), encoding="utf-8"))
    traj = SYN.SynthesizeJointBestOf(w, TEXT, prof, nTries=NTRIES, seed=SEED,
                                     reader=reader, authorModel=shape, authorMapping=smap)
    img = SYN.RenderTrajectory(traj, pxPerMm=18.0, profile=prof, uniformInk=True)
    arr = np.asarray(img, dtype=np.uint8)
    out = HERE/("%d_%s.png" % (i, w))
    Image.fromarray(arr).save(out)
    read = REC.ReadText(arr, reader)
    acc = REC.CharAcc(read, TEXT)
    _k, pr = REC.ClassifyImage(arr, shape)
    conf = float(pr[smap[w]])
    rows.append((w, acc, conf, read, arr.shape, out.name))
    print("%-9s char %5.1f%%  writer %5.1f%%  %s" % (w, 100*acc, 100*conf, out.name), flush=True)

with open(HERE/"RESULTS.txt", "w", encoding="utf-8") as f:
    f.write('sentence: "%s"  (%d characters)\n' % (TEXT, len(TEXT)))
    f.write("nTries %d, seed %d, C1 join, rendered at 18 px/mm\n\n" % (NTRIES, SEED))
    f.write("%-10s %8s %9s  %s\n" % ("writer", "char %", "writer %", "what the reader read back"))
    for w, acc, conf, read, shp, name in rows:
        f.write("%-10s %7.1f %8.1f  %s\n" % (w, 100*acc, 100*conf, read))
    f.write("\nimages are %s\n" % ", ".join(r[5] for r in rows))
print("wrote", HERE/"RESULTS.txt")
