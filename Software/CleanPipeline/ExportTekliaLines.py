#one-time export: downloads the pre-segmented, pre-labelled Teklia/IAM-line
#dataset from HuggingFace and dumps it to plain PNG + "<name>_labels.txt"
#sidecar files -- the same per-line-image + text-sidecar shape every other
#dataset in this project already uses. this is the ONLY place in the whole
#project that needs the `datasets` library (pip install datasets); nothing
#in the actual training/classification/reproduction pipeline imports it, or
#ever will -- once this has been run once and the files are on disk,
#`datasets` is never touched again. compare to downloading IAMpages671/10
#by hand: this just automates fetching+unpacking one specific dataset.
#run by hand, once: python ExportTekliaLines.py [--out-dir PATH]
#on Colab, pass --out-dir pointed at your mounted Drive directly (e.g.
#/content/drive/MyDrive/.../Data/Datasets/TekliaIamLines) so the export
#lands somewhere that survives the session ending -- the default path is
#relative to this script's own location, which on Colab is the ephemeral
#local clone, not Drive
import argparse
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))
from Charset import CHAR_TO_IDX

DEFAULT_OUT_DIR = Path(__file__).resolve().parents[2]/"Data"/"Datasets"/"TekliaIamLines"
MAX_SAFE_LABEL_CHARS = int((640//4)*0.75)   #matches TrainText.py's own ctc-time-dimension cap


#writes one split ("train"/"validation") to outDir/<split>/, one PNG +
#one "<idx>_labels.txt" per usable line, same filtering TrainText.py's old
#torch-based hf path used (drop chars outside CHARSET, drop empty/oversized)
def ExportSplit(hfSplit,splitName,outDirRoot):
    outDir = outDirRoot/splitName
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
    parser = argparse.ArgumentParser(description="One-time export of Teklia/IAM-line to plain PNG+label files.")
    parser.add_argument("--out-dir",type=str,default=None,
                        help="where to write train/validation folders -- point this at your "
                             "mounted Drive on Colab, or leave blank for the local default "
                             f"({DEFAULT_OUT_DIR})")
    args = parser.parse_args()
    outDirRoot = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_DIR

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "the 'datasets' library is required for this one-time export only:\n"
            "    pip install datasets\n"
            "it is not needed anywhere else in this project -- safe to uninstall "
            "once this script finishes."
        )
    print(f"[ExportTekliaLines] downloading Teklia/IAM-line (one-time, cached by "
          f"the datasets library on first run)... writing to {outDirRoot}")
    ds = load_dataset("Teklia/IAM-line")
    ExportSplit(ds["train"],"train",outDirRoot)
    ExportSplit(ds["validation"],"validation",outDirRoot)
    print(f"\nDone. Exported to: {outDirRoot}")
    print("TrainScratch.py's TekliaLineDataset reads this directly -- no further "
          "export runs or `datasets` imports needed.")


if __name__=="__main__":
    Main()
