"""
TrainTextHF_Sweep.py -- unattended architecture/hyperparameter sweep for the
text recogniser (PaperCRNN), trained on the same Teklia/IAM-line dataset
TrainTextHF.py uses.

WHAT THIS DOES DIFFERENTLY FROM TrainTextHF.py
    TrainTextHF.py trains ONE configuration you choose via CLI flags.
    This script trains a whole LIST of configurations back to back, with NO
    interactive step between them -- start it once and walk away. Each
    config gets its own fixed EPOCHS_PER_CONFIG epochs (default 50, no
    early stopping -- that's a deliberate choice so every config gets
    compared on equal terms, not "however long it happened to take to
    plateau"), its own checkpoint files, and one summary row appended to a
    master CSV log the moment it finishes.

WHAT'S BEING SWEPT (one-at-a-time from a known-good baseline, not a full
combinatorial grid -- 14 configs, not the ~700+ a full cross-product of
these 7 dimensions would be):
    learning rate    -- 3e-4, 1e-3 (baseline), 3e-3
    LSTM width        -- hidden=128, 256 (baseline), 384
    LSTM depth        -- 1, 2 (baseline), 3 layers
    conv width        -- 0.5x, 1x (baseline), 1.5x every conv stage's channels
    dropout           -- 0.0 (baseline), 0.15, 0.3
    weight decay      -- 0, 1e-5 (baseline), 1e-4
    optimizer         -- RMSprop (baseline), Adam
Batch size is held FIXED across every config on purpose -- it's a
memory/speed knob, not an architecture choice, and varying it alongside
everything else would make the comparisons harder to read, not easier.

RESUMABILITY (this WILL span multiple Colab sessions):
    Every completed config's summary row goes into the master CSV
    immediately. On restart, any config already in that CSV is skipped
    entirely; the config that was mid-training when the session died
    resumes from ITS OWN checkpoint (same per-epoch resume mechanism
    TrainTextHF.py uses). Re-running this exact same command after a
    disconnect is always the right move -- it never repeats finished work
    and never loses partial progress on the config in flight.

FAULT TOLERANCE: a config that diverges (NaN/Inf loss) or throws is caught,
logged to the CSV with status=FAILED and the exception message, and the
sweep moves on to the next config rather than stopping. A bad learning
rate should cost you one row in a spreadsheet, not the rest of the sweep.

USAGE
    python TrainTextHF_Sweep.py                       # run everything
    python TrainTextHF_Sweep.py --max-samples 300      # smoke test
    python TrainTextHF_Sweep.py --epochs-per-config 5  # override 50 (testing only)
"""
import argparse
import csv
import math
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from TrainText import CHARSET, PaperCRNN, evaluate
from TrainTextHF import HFLineDataset, collate_fn, load_hf

SCRIPT_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = SCRIPT_DIR / "NOGIT" / "weights_sweep"
LOG_CSV = WEIGHTS_DIR / "sweep_results.csv"
DEFAULT_HF_CACHE = SCRIPT_DIR / "NOGIT" / "hf_cache"

BASE = dict(lr=1e-3, lstm_hidden=256, lstm_layers=2, conv_width_mult=1.0,
           dropout=0.0, weight_decay=1e-5, optimizer="rmsprop")


def _cfg(name, **overrides):
    c = dict(BASE)
    c.update(overrides)
    c["name"] = name
    return c


CONFIGS = [
    _cfg("baseline"),
    # learning rate
    _cfg("lr_3e-4", lr=3e-4),
    _cfg("lr_3e-3", lr=3e-3),
    # LSTM width
    _cfg("hidden_128", lstm_hidden=128),
    _cfg("hidden_384", lstm_hidden=384),
    # LSTM depth
    _cfg("layers_1", lstm_layers=1),
    _cfg("layers_3", lstm_layers=3),
    # conv width (structural capacity of the CNN stages)
    _cfg("convwidth_0.5x", conv_width_mult=0.5),
    _cfg("convwidth_1.5x", conv_width_mult=1.5),
    # dropout
    _cfg("dropout_0.15", dropout=0.15),
    _cfg("dropout_0.3", dropout=0.3),
    # weight decay
    _cfg("wd_0", weight_decay=0.0),
    _cfg("wd_1e-4", weight_decay=1e-4),
    # optimizer
    _cfg("adam", optimizer="adam"),
]


def already_done(name):
    if not LOG_CSV.exists():
        return False
    with open(LOG_CSV, encoding="utf-8") as f:
        return any(row.get("name") == name and row.get("status") == "OK"
                   for row in csv.DictReader(f))


def append_log(row):
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not LOG_CSV.exists()
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            w.writeheader()
        w.writerow(row)


def build_model(cfg, device):
    return PaperCRNN(num_classes=len(CHARSET) + 1, lstm_hidden=cfg["lstm_hidden"],
                     lstm_layers=cfg["lstm_layers"], conv_width_mult=cfg["conv_width_mult"],
                     dropout=cfg["dropout"]).to(device)


def build_optimizer(cfg, model):
    if cfg["optimizer"] == "adam":
        return optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    return optim.RMSprop(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])


def run_config(cfg, train_loader, val_loader, device, epochs, batch_size, max_samples):
    name = cfg["name"]
    ckpt_path = WEIGHTS_DIR / f"{name}_checkpoint.pt"
    best_path = WEIGHTS_DIR / f"{name}_best.pt"

    model = build_model(cfg, device)
    optimizer = build_optimizer(cfg, model)
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    start_epoch, best_val_loss = 1, float("inf")
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        start_epoch = ck["epoch"] + 1
        best_val_loss = ck["best_val_loss"]
        print(f"  [Resume] {name}: epoch {ck['epoch']} -> {start_epoch}, "
              f"best_val_loss={best_val_loss:.4f}")

    t_start = time.time()
    final_char_acc = final_cer = float("nan")
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        tot, n_ok = 0.0, 0
        for images, targets, target_lengths, _texts in train_loader:
            images, targets = images.to(device), targets.to(device)
            target_lengths = target_lengths.to(device)
            log_probs = model(images)
            input_lengths = torch.full((images.size(0),), log_probs.size(0),
                                       dtype=torch.long, device=device)
            loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
            optimizer.zero_grad()
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                tot += float(loss.item()); n_ok += 1
        train_loss = tot / max(1, n_ok)
        if not math.isfinite(train_loss):
            raise RuntimeError(f"train loss diverged to {train_loss} at epoch {epoch}")

        val_loss, val_char_acc, val_cer = evaluate(model, val_loader, device, ctc_loss_fn)
        final_char_acc, final_cer = val_char_acc, val_cer
        saved = ""
        torch.save(dict(epoch=epoch, model_state_dict=model.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(),
                        best_val_loss=min(best_val_loss, val_loss)), ckpt_path)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            saved = " [BEST]"
        print(f"  [{name}] Epoch {epoch:03d}/{epochs} | Train {train_loss:.3f} "
              f"| Val {val_loss:.3f} | CharAcc {val_char_acc:.2%} | CER {val_cer:.2%}"
              f"{saved}", flush=True)

    dt_min = (time.time() - t_start) / 60.0
    return dict(name=name, status="OK", epochs=epochs, best_val_loss=round(best_val_loss, 4),
               final_char_acc=round(final_char_acc, 4), final_cer=round(final_cer, 4),
               minutes=round(dt_min, 1), lr=cfg["lr"], lstm_hidden=cfg["lstm_hidden"],
               lstm_layers=cfg["lstm_layers"], conv_width_mult=cfg["conv_width_mult"],
               dropout=cfg["dropout"], weight_decay=cfg["weight_decay"],
               optimizer=cfg["optimizer"], error="")


def main(epochs_per_config=50, batch_size=64, max_samples=None,
        cache_dir=str(DEFAULT_HF_CACHE)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    print(f"[Sweep] {len(CONFIGS)} configs x {epochs_per_config} epochs each\n")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # data loaded ONCE, reused for every config -- only the model/optimiser differ
    ds = load_hf(cache_dir)
    train_set = HFLineDataset(ds["train"], is_train=True, max_samples=max_samples)
    val_set = HFLineDataset(ds["validation"], is_train=False, max_samples=max_samples)
    is_cuda = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4 if is_cuda else 0,
                              pin_memory=is_cuda)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=4 if is_cuda else 0,
                            pin_memory=is_cuda)
    print(f"train lines: {len(train_set)}   val lines: {len(val_set)}\n")

    for cfg in CONFIGS:
        if already_done(cfg["name"]):
            print(f"[Sweep] {cfg['name']}: already completed, skipping")
            continue
        print(f"\n=== {cfg['name']} === {cfg}")
        try:
            row = run_config(cfg, train_loader, val_loader, device,
                             epochs_per_config, batch_size, max_samples)
        except Exception as e:
            print(f"[Sweep] {cfg['name']} FAILED: {e}")
            traceback.print_exc()
            row = dict(name=cfg["name"], status="FAILED", epochs=epochs_per_config,
                      best_val_loss="", final_char_acc="", final_cer="", minutes="",
                      lr=cfg["lr"], lstm_hidden=cfg["lstm_hidden"],
                      lstm_layers=cfg["lstm_layers"], conv_width_mult=cfg["conv_width_mult"],
                      dropout=cfg["dropout"], weight_decay=cfg["weight_decay"],
                      optimizer=cfg["optimizer"], error=str(e)[:200])
        append_log(row)
        print(f"[Sweep] {cfg['name']} -> {row['status']}  (logged to {LOG_CSV})")

    print(f"\n[Sweep] all configs attempted. Results: {LOG_CSV}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs-per-config", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--cache-dir", default=str(DEFAULT_HF_CACHE))
    args = ap.parse_args()
    main(epochs_per_config=args.epochs_per_config, batch_size=args.batch_size,
        max_samples=args.max_samples, cache_dir=args.cache_dir)
