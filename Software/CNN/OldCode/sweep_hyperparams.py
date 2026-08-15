"""
Runs several short training trials back-to-back, each with a different set of
hyperparameters, so you can compare how fast/steadily text-recognition accuracy
rises before committing hours to one configuration. Author-ID training is paused
for all trials (it already converges well) so every trial's signal is purely
about text recognition.

Trials run SEQUENTIALLY on purpose -- this is CPU-bound work, so running them
in parallel would just make every trial slower by fighting over the same cores.

Usage:
    python sweep_hyperparams.py

Every trial's output prints live to your terminal AND is written to its own
file under logs/sweep/ as it happens -- each line is flushed (and fsync'd) to
disk immediately, not buffered, so if your machine crashes mid-run you only
lose whatever hadn't printed yet, never the whole log. Edit the TRIALS list
below to change what gets tried.
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR / "train_handwriting_robot_v1_baseline.py"
SWEEP_LOG_DIR = SCRIPT_DIR / "logs" / "sweep"

EPOCHS_PER_TRIAL = 30

# Each dict becomes CLI flags. "name" is used as --run-name and the log filename.
TRIALS = [
    {"name": "lr0.001_bs16", "lr": "0.001", "batch_size": "16"},
    {"name": "lr0.002_bs16", "lr": "0.002", "batch_size": "16"},
    {"name": "lr0.0005_bs16", "lr": "0.0005", "batch_size": "16"},
    {"name": "lr0.001_bs32", "lr": "0.001", "batch_size": "32"},
    {"name": "lr0.002_bs32", "lr": "0.002", "batch_size": "32"},
]

results = {}  # name -> list of val char-acc floats, for the end-of-sweep summary


def write_durable(log_file, text):
    """Write + flush + fsync so the line survives a crash right after this call returns."""
    log_file.write(text)
    log_file.flush()
    os.fsync(log_file.fileno())


def run_trial(trial):
    name = trial["name"]
    log_path = SWEEP_LOG_DIR / f"{name}.log"
    cmd = [
        sys.executable, "-u", str(TRAIN_SCRIPT),
        "--epochs", str(EPOCHS_PER_TRIAL),
        "--batch-size", trial["batch_size"],
        "--lr", trial["lr"],
        "--eval-every", "1",
        "--checkpoint-every", "5",
        "--held-out-last-page",
        "--author-loss-weight", "0",
        "--run-name", name,
    ]

    header = (f"\n{'=' * 88}\nTrial: {name}   started {datetime.now().isoformat(timespec='seconds')}\n"
              f"Command: {' '.join(cmd[2:])}\nLog file: {log_path}\n{'=' * 88}\n")
    print(header, end="")

    char_accs = []
    # buffering=1 = line-buffered even though we also flush+fsync manually below;
    # belt-and-braces against any platform where line buffering alone isn't durable enough.
    with open(log_path, "a", encoding="utf-8", buffering=1) as log_file:
        write_durable(log_file, header)

        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=str(SCRIPT_DIR),
        )
        for line in process.stdout:
            print(line, end="")
            write_durable(log_file, line)
            if line.startswith("Epoch") and "val char-acc" in line:
                try:
                    part = line.split("val char-acc")[1].strip().split()[0]
                    char_accs.append(float(part))
                except (IndexError, ValueError):
                    pass
        process.wait()

        footer = f"\nTrial {name} finished: {'OK' if process.returncode == 0 else f'FAILED (exit {process.returncode})'}\n"
        print(footer, end="")
        write_durable(log_file, footer)

    results[name] = char_accs
    return process.returncode == 0


def summarize():
    print(f"\n{'=' * 88}\nSWEEP SUMMARY (val char-acc per trial)\n{'=' * 88}")
    for trial in TRIALS:
        char_accs = results.get(trial["name"], [])
        if char_accs:
            trend = " -> ".join(f"{v:.3f}" for v in char_accs[-5:])
            print(f"  {trial['name']:20s}: final={char_accs[-1]:.4f}  best={max(char_accs):.4f}  last few: {trend}")
        else:
            print(f"  {trial['name']:20s}: no epoch results parsed")
    print(f"\nFull per-line logs are in: {SWEEP_LOG_DIR}")


def main():
    SWEEP_LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Running {len(TRIALS)} trials x {EPOCHS_PER_TRIAL} epochs each. This will take a while -- "
          f"logs stream live to your terminal and are also saved (crash-safe) under {SWEEP_LOG_DIR}")

    for trial in TRIALS:
        run_trial(trial)

    summarize()


if __name__ == "__main__":
    main()
