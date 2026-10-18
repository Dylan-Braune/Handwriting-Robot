#one-time export: downloads the pre-segmented, pre-labelled Teklia/IAM-line
#dataset from HuggingFace and dumps it to plain PNG + "<name>_labels.txt"
#sidecar files -- the same per-line-image + text-sidecar shape every other
#dataset in this project already uses. this is the ONLY place in the whole
#project that needs the `datasets` library (pip install datasets); nothing
#in the actual training/classification/reproduction pipeline imports it, or
#ever will -- once this has been run once and the files are on disk,
#`datasets` is never touched again. compare to downloading IAMpages671/10
#by hand: this just automates fetching+unpacking one specific dataset.
#run by hand, once: python ExportTekliaLines.py
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))
from Charset import CHAR_TO_IDX

OUT_DIR = Path(__file__).resolve().parents[2]/"Data"/"Datasets"/"TekliaIamLines"
MAX_SAFE_LABEL_CHARS = int((640//4)*0.75)   #matches TrainText.py's own ctc-time-dimension cap


#writes one split ("train"/"validation") to OUT_DIR/<split>/, one PNG +
#one "<idx>_labels.txt" per usable line, same filtering TrainText.py's old
#torch-based hf path used (drop chars outside CHARSET, drop empty/oversized)
def ExportSplit(hfSplit,splitName):
    outDir = OUT_DIR/splitName
    outDir.mkdir(parents=True,exist_ok=True)
    kept,droppedEmpty,droppedLong = 0,0,0
    total = len(hfSplit)
    for i,row in enumerate(hfSplit):
        text = "".join(c for c in row["text"].strip() if c in CHAR_TO_IDX)
        if len(text)==0:
            droppedEmpty+=1
            continue
        if len(text)>MAX_SAFE_LABEL_CHARS:
            droppedLong+=1
            continue
        stem = f"{i:06d}"
        row["image"].convert("L").save(outDir/f"{stem}.png")
        (outDir/f"{stem}_labels.txt").write_text(text,encoding="utf-8")
        kept+=1
        if kept%500==0:
            print(f"\r[{splitName}] exported {kept}/{total}",end="",flush=True)
    print(f"\n[{splitName}] done: {kept} kept, {droppedEmpty} empty, {droppedLong} over "
          f"{MAX_SAFE_LABEL_CHARS} chars (from {total} total)")


def Main():
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "the 'datasets' library is required for this one-time export only:\n"
            "    pip install datasets\n"
            "it is not needed anywhere else in this project -- safe to uninstall "
            "once this script finishes."
        )
    print("[ExportTekliaLines] downloading Teklia/IAM-line (one-time, cached by "
          "the datasets library on first run)...")
    ds = load_dataset("Teklia/IAM-line")
    ExportSplit(ds["train"],"train")
    ExportSplit(ds["validation"],"validation")
    print(f"\nDone. Exported to: {OUT_DIR}")
    print("TrainScratch.py's TekliaLineDataset reads this directly -- no further "
          "export runs or `datasets` imports needed.")


if __name__=="__main__":
    Main()
