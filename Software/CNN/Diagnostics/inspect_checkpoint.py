"""
Prints the metadata stored inside a training checkpoint (.pt) file, so you
can tell which of two conflicting copies (e.g. after a git stash/pull
conflict) actually matches your real training progress, instead of
guessing from filenames or timestamps.

Works on either file produced by TrainText.py:
  - *_checkpoint.pt  -- a full dict with epoch/best_val_loss/patience_counter/
                        restart_count/etc.
  - *_best.pt        -- just a raw model state_dict (no metadata to print,
                        this script will tell you that directly).

Usage:
    python inspect_checkpoint.py
        (prompts for one or more paths, comma-separated)
    python inspect_checkpoint.py --path paper_cnn_bilstm_ctc_checkpoint.pt
    python inspect_checkpoint.py --path file1.pt,file2.pt
        (prints both side by side so you can compare)
"""

import argparse
from pathlib import Path

import torch


def inspect_one(path):
    path = Path(path)
    print("\n" + "=" * 78)
    print(f"File: {path}")
    if not path.exists():
        print("  [Error] File not found.")
        return
    print(f"  Size: {path.stat().st_size / 1e6:.1f} MB")

    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  [Error] Could not load this file with torch.load: {e}")
        return

    if isinstance(obj, dict) and "epoch" in obj:
        print("  Type: full training checkpoint")
        print(f"  Epoch:            {obj.get('epoch')}")
        print(f"  Best val loss:    {obj.get('best_val_loss')}")
        print(f"  Patience counter: {obj.get('patience_counter')}")
        print(f"  Restart count:    {obj.get('restart_count', '(not present -- older checkpoint)')}")
        if "optimizer_state_dict" in obj:
            try:
                lr = obj["optimizer_state_dict"]["param_groups"][0]["lr"]
                print(f"  LR at save time:  {lr}")
            except Exception:
                pass
    elif isinstance(obj, dict):
        # A raw model state_dict (e.g. *_best.pt) has no training metadata --
        # just tensor weights keyed by layer name.
        print("  Type: raw model state_dict (e.g. a *_best.pt file) -- no epoch/loss "
              "metadata is stored in this kind of file, only the weights themselves.")
        print(f"  Number of parameter tensors: {len(obj)}")
    else:
        print(f"  Type: unrecognized object ({type(obj)})")


def main():
    parser = argparse.ArgumentParser(description="Inspect metadata inside one or more checkpoint .pt files.")
    parser.add_argument("--path", default=None,
                         help="Comma-separated path(s) to .pt file(s) to inspect and compare.")
    args = parser.parse_args()

    paths = args.path
    if paths is None:
        raw = input("Path(s) to .pt file(s) to inspect (comma-separated if more than one): ").strip()
        paths = raw

    path_list = [p.strip() for p in paths.split(",") if p.strip()]
    if not path_list:
        print("No paths given.")
        return

    for p in path_list:
        inspect_one(p)

    if len(path_list) > 1:
        print("\n" + "=" * 78)
        print("Compare the 'Epoch' and 'Best val loss' fields above against what you saw "
              "printed in your terminal/log during training -- the file whose numbers match "
              "your actual training run (e.g. epoch 81, best_val_loss ~1.06) is the correct one "
              "to keep. Whichever one this is, that's the one to keep at "
              "paper_cnn_bilstm_ctc_checkpoint.pt / paper_cnn_bilstm_ctc_best.pt going forward -- "
              "you can delete or rename the other one aside once you're sure.")


if __name__ == "__main__":
    main()
