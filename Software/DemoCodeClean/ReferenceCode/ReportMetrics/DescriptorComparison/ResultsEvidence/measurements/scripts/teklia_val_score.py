#the reader measured on Teklia's own held-out split.
#
#BuildTekliaSplit reads train/ for training and validation/ for validation,
#so validation/ is 976 line crops the reader never learned from.  that is the
#public-benchmark number, independent of this project's own pages and of its
#segmenter: the crops are Teklia's, already cut.
#
#the text-match lookup showed none of the six database writers' lines appear
#in validation/, so this set shares no line with any page measured elsewhere.
import sys
import warnings
from pathlib import Path

from PIL import Image
warnings.filterwarnings("ignore")

CP3 = Path(r"C:\Users\braun\OneDrive\Documents\GitHub\Handwriting-Robot\Software\CleanPipeline3")
sys.path.insert(0, str(CP3))
import recognise as REC

DL = Path(r"C:\Users\braun\Downloads")
VAL = Path(r"C:\Users\braun\OneDrive\Documents\GitHub\Handwriting-Robot"
           r"\Data\Datasets\TekliaIamLines\validation")

reader = REC.PaperCRNNNumpy(
    checkpointPath=str(DL/"paper_cnn_bilstm_ctc_scratch_best.npz"))
print("reader paper_cnn_bilstm_ctc_scratch_best.npz")

pairs = []
for png in sorted(VAL.glob("*.png")):
    lab = png.with_name(png.stem+"_labels.txt")
    if lab.exists():
        t = lab.read_text(encoding="utf-8").strip()
        if t:
            pairs.append((png, t))
print("%d held-out Teklia crops\n" % len(pairs))

d = n = wr = wn = exact = 0
for i, (png, t) in enumerate(pairs, 1):
    r = REC.ReadText(Image.open(png).convert("L"), reader)
    d += REC.Levenshtein(r, t)
    n += len(t)
    wr += sum(1 for a, b in zip(r.split(), t.split()) if a == b)
    wn += len(t.split())
    exact += (r.strip() == t)
    if i % 100 == 0:
        print("  %4d / %d   char %.2f %%" % (i, len(pairs), 100*max(0, 1-d/n)),
              flush=True)

print("\n" + "="*62)
print("Teklia held-out split   %d lines" % len(pairs))
print("  character accuracy    %.2f %%" % (100*max(0, 1-d/n)))
print("  word accuracy         %.2f %%" % (100*wr/wn))
print("  lines read exactly    %d / %d = %.2f %%"
      % (exact, len(pairs), 100*exact/len(pairs)))
