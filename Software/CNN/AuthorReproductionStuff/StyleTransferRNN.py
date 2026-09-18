"""
StyleTransferRNN.py -- a SECOND, separate RNN approach to author-conditioned
handwriting synthesis. Kept alongside RNNHandwriting.py, not replacing it --
both remain independently runnable so either can be tested/compared.

WHY THIS IS DIFFERENT FROM RNNHandwriting.py
    RNNHandwriting.py generates a pen path from raw text, from scratch,
    autoregressively -- which means it has to learn everything at once:
    how a pen moves, how letters connect, AND how to track its own position
    through the target text as it writes (the window-attention mechanism).
    That last part is the expensive one to learn and is why that approach
    needs a large pretraining pass before it can spell reliably.

    This script sidesteps that entirely by using the ALREADY-CORRECT
    trajectory-library method (BuildStyleProfile.py / SynthesizeHandwriting.py)
    as scaffolding: synthesize the target text with the GENERIC anchor
    (legibility=1.0 -- always spells correctly, always structurally sound,
    but looks like nobody in particular), then train a network whose ONLY
    job is to warp that already-correct trajectory toward one author's real
    stroke shapes. No text is ever given to this network directly, and no
    window-attention is needed, because the anchor trajectory already
    encodes "where in the text am I" implicitly in its own point order.

    That turns an open-ended "generate handwriting from text" problem (needs
    tens of thousands of examples) into a much narrower "reshape this known
    curve to look like this person's ink" problem -- learnable from the
    ~70-100 real lines a single author actually has, no pretraining pass
    required, and the network itself can be a single plain bidirectional
    LSTM pass (non-autoregressive, no MDN, no sampling loop) instead of the
    heavier architecture RNNHandwriting.py needs -- so this should be both
    faster to train and faster to run.

    Style is JUST the moment-to-moment stroke shape here (the thing the
    generic anchor throws away). Slant, x-height, word/letter spacing and
    join rate are already handled reliably by the library method's own
    measured statistics and are kept as-is -- the network never has to
    relearn things that are already being computed correctly.

PIPELINE
    1. For every real line already in NOGIT/RNNStrokeCache10/<author>.pkl,
       synthesize the SAME text with that author's OWN profile at
       legibility=1.0 (SynthesizeHandwriting's generic anchor) -- this is
       the "content" input. The real extracted stroke sequence is the
       "style" target.
    2. Resample both to the same fixed number of points along arc length
       (so point i of the anchor and point i of the target refer to
       roughly the same fraction of the way through the line -- a
       simplification, not a claim of exact per-letter correspondence).
    3. Train a bidirectional LSTM (+ a small per-writer embedding) to map
       anchor deltas -> real deltas, by direct regression (Huber loss) --
       not a mixture density, since this is a much more determined mapping
       problem than generating from nothing.
    4. At generation time: synthesize the anchor for any new text with any
       author's profile, resample, run the network ONCE (no autoregressive
       loop), reconstruct absolute positions, render/export.

USAGE
    python StyleTransferRNN.py --build-cache
    python StyleTransferRNN.py --epochs 100 --batch-size 16
    python StyleTransferRNN.py --sample "hello world" --writer 153 --out sample.png
"""
import argparse
import math
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

import SynthesizeHandwriting as SY

SCRIPT_DIR = Path(__file__).resolve().parent
NOGIT_DIR = SCRIPT_DIR.parent / "NOGIT"
STROKE_CACHE_DIR = NOGIT_DIR / "RNNStrokeCache10"          # real strokes (shared with RNNHandwriting.py)
PAIR_CACHE_DIR = NOGIT_DIR / "StyleTransferCache10"        # (anchor, real) resampled pairs
WEIGHTS_DIR = NOGIT_DIR / "weights"
CKPT_PATH = WEIGHTS_DIR / "styletransfer_checkpoint.pt"
BEST_PATH = WEIGHTS_DIR / "styletransfer_best.pt"

N_POINTS = 400          # fixed resample length for both anchor and target
MM_PER_XH = 4.0         # matches the mmPerXh SynthesizeText is called with


# ---------------------------------------------------------------------------
# Trajectory <-> fixed-length (dx, dy) sequence, arc-length resampled
# ---------------------------------------------------------------------------
JUMP_WEIGHT = 0.03   # how much a pen-up transition counts toward arc length,
                      # relative to real ink -- see _resample_arclength


def _resample_arclength(points, n, eos=None):
    """points: (M,2) absolute xy. Returns (n,2), evenly spaced by arc length.
    If eos (M,) pen-lift flags are given, also returns an (n,) eos array:
    for every original point flagged as a stroke break, the NEAREST
    resampled index is flagged too -- so pen lifts survive resampling
    instead of being silently smoothed away into one continuous line.

    Pen-UP transitions (the straight jump from the end of one stroke to
    the start of the next -- e.g. the gap between letters or words) are
    downweighted to JUMP_WEIGHT of their real Euclidean length before the
    arc-length budget is computed. Without this, those jumps (often
    several x-heights for word gaps, vs a fraction of an x-height for
    actual letter ink) dominate total arc length and most of the fixed
    N_POINTS budget ends up sampling pen-up travel instead of ink --
    confirmed empirically: without this fix, trained output collapsed to
    a near-flat line, matching what you'd get if letter shape was
    drowned out by inter-word jumps in the training targets."""
    p = np.asarray(points, np.float64)
    if len(p) < 2:
        out = np.repeat(p[:1] if len(p) else np.zeros((1, 2)), n, axis=0)
        return (out, np.zeros(n, np.float32)) if eos is not None else out
    seg = np.hypot(*np.diff(p, axis=0).T)
    if eos is not None:
        is_jump = np.asarray(eos, np.float64)[:-1] > 0.5   # segment after an eos point
        seg = np.where(is_jump, seg * JUMP_WEIGHT, seg)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total < 1e-9:
        out = np.repeat(p[:1], n, axis=0)
        return (out, np.zeros(n, np.float32)) if eos is not None else out
    t = np.linspace(0.0, total, n)
    x = np.interp(t, cum, p[:, 0])
    y = np.interp(t, cum, p[:, 1])
    out = np.stack([x, y], axis=1)
    if eos is None:
        return out
    eos = np.asarray(eos, np.float64)
    out_eos = np.zeros(n, np.float32)
    # Mark EVERY resampled point that falls inside a pen-up jump segment as
    # a break (a downweighted jump can still span more than one resampled
    # step for a wide word gap, and any point left unmarked inside it gets
    # drawn as a spurious connecting line), AND ALSO mark the single
    # nearest resampled point to each original break distance -- a very
    # short/narrow jump can be downweighted to less than one resampling
    # step wide, so no point ever lands strictly inside it, but the two
    # samples straddling it still need a flag between them or they get
    # drawn connected across the gap.
    if len(is_jump):
        seg_idx = np.clip(np.searchsorted(cum, t, side="right") - 1, 0, len(is_jump) - 1)
        out_eos[is_jump[seg_idx]] = 1.0
    break_dists = cum[eos > 0.5]
    if len(break_dists):
        idx = np.clip(np.searchsorted(t, break_dists), 0, n - 1)
        out_eos[idx] = 1.0
    out_eos[-1] = 1.0
    return out, out_eos


def trajectory_to_xy(traj, scale):
    """Trajectory (mm, list of pen-down polylines) -> a flat (M,2) absolute
    xy array in x-height units plus an (M,) eos array marking the last
    point of each pen-down stroke (i.e. where the anchor itself lifts the
    pen) -- strokes are concatenated in order for the arc-length resample,
    but the lift points are preserved so rendering can still show real
    gaps between letters/words instead of one unbroken cursive line. The
    network's OWN job stays just the shape warp; where to lift the pen is
    always the anchor's already-correct decision, never learned."""
    pts, eos = [], []
    for s in traj.strokes:
        if len(s) < 1:
            continue
        for p in s:
            pts.append(p)
            eos.append(0.0)
        eos[-1] = 1.0
    if len(pts) < 2:
        return None, None
    return np.asarray(pts, np.float64) / scale, np.asarray(eos, np.float32)


def stroke_seq_to_xy(seq):
    """extract_line_strokes()-style (dx,dy,eos) deltas -> absolute (M,2) plus
    the same (M,) eos column, for symmetric real-line rendering."""
    abs_xy = np.cumsum(seq[:, :2], axis=0)
    return abs_xy, seq[:, 2]


def build_cache(force=False):
    PAIR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    profiles = SY.LoadAllProfiles()
    writers = sorted(p.stem for p in STROKE_CACHE_DIR.glob("*.pkl"))
    print(f"writers: {writers}")
    for w in writers:
        out_path = PAIR_CACHE_DIR / f"{w}.pkl"
        if out_path.exists() and not force:
            print(f"  {w}: cached, skipping")
            continue
        if w not in profiles:
            print(f"  {w}: no style profile, skipping")
            continue
        prof = profiles[w]
        rows = pickle.load(open(STROKE_CACHE_DIR / f"{w}.pkl", "rb"))
        out_rows = []
        for r in rows:
            text = r["text"]
            try:
                traj = SY.SynthesizeText(text, prof, mmPerXh=MM_PER_XH, seed=0,
                                         lineWidthMm=100_000.0, legibility=1.0)
            except Exception:
                continue
            anchor_xy, anchor_eos = trajectory_to_xy(traj, MM_PER_XH)
            if anchor_xy is None:
                continue
            target_xy, target_eos = stroke_seq_to_xy(r["stroke"])
            if len(target_xy) < 2:
                continue
            # anchor absolute position starts wherever SynthesizeText's
            # coordinate frame put it -- re-origin both to (0,0) so the
            # network only ever has to learn SHAPE, not absolute placement
            anchor_xy = anchor_xy - anchor_xy[0]
            target_xy = target_xy - target_xy[0]
            a_rs, a_eos_rs = _resample_arclength(anchor_xy, N_POINTS, eos=anchor_eos)
            t_rs, _ = _resample_arclength(target_xy, N_POINTS, eos=target_eos)
            out_rows.append(dict(text=text, anchor=a_rs.astype(np.float32),
                                 target=t_rs.astype(np.float32),
                                 anchor_eos=a_eos_rs.astype(np.float32),
                                 is_holdout=r["is_holdout"]))
        with open(out_path, "wb") as f:
            pickle.dump(out_rows, f)
        print(f"  {w}: {len(out_rows)}/{len(rows)} pairs built -> {out_path.name}")
    return writers


# ---------------------------------------------------------------------------
# Dataset -- everything is already a fixed length, so no padding/masking
# ---------------------------------------------------------------------------
class PairDataset(torch.utils.data.Dataset):
    def __init__(self, writers, is_train):
        self.samples = []
        self.writer_to_idx = {w: i for i, w in enumerate(writers)}
        for w in writers:
            rows = pickle.load(open(PAIR_CACHE_DIR / f"{w}.pkl", "rb"))
            for r in rows:
                if r["is_holdout"] == (not is_train):
                    self.samples.append((w, r["anchor"], r["target"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        w, anchor, target = self.samples[i]
        anchor_d = np.diff(anchor, axis=0, prepend=anchor[:1])
        target_d = np.diff(target, axis=0, prepend=target[:1])
        return self.writer_to_idx[w], anchor_d.astype(np.float32), target_d.astype(np.float32)


def collate(batch):
    writers = torch.tensor([b[0] for b in batch], dtype=torch.long)
    anchors = torch.from_numpy(np.stack([b[1] for b in batch]))
    targets = torch.from_numpy(np.stack([b[2] for b in batch]))
    return writers, anchors, targets


# ---------------------------------------------------------------------------
# Model: plain bidirectional LSTM, non-autoregressive -- the whole anchor
# sequence is already known up front, so there is no need to generate it
# one step at a time the way RNNHandwriting.py's text-conditioned model does.
# ---------------------------------------------------------------------------
class StyleTransferRNN(nn.Module):
    """Predicts a RESIDUAL correction on top of the anchor's own deltas,
    not a full replacement -- pred = anchor_deltas + out(h). The anchor is
    already a perfectly legible, correctly-spelled trajectory, so the
    network's job narrows to "how much lighter/heavier/tighter/looser is
    this author's stroke shape than the generic anchor" rather than having
    to reconstruct pen dynamics from nothing. Zero-initializing the output
    layer means the model starts as an exact identity (pure anchor) and
    only grows a style deviation as training finds one -- so an
    undertrained network degrades gracefully back toward the anchor's
    already-correct, always-legible shape instead of toward noise."""
    def __init__(self, n_writers, hidden=128, layers=2, writer_dim=32):
        super().__init__()
        self.writer_embed = nn.Embedding(n_writers, writer_dim)
        self.lstm = nn.LSTM(2 + writer_dim, hidden, num_layers=layers,
                            batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden, 2)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, anchor_deltas, writer_idx):
        B, T, _ = anchor_deltas.shape
        wemb = self.writer_embed(writer_idx).unsqueeze(1).expand(-1, T, -1)
        x = torch.cat([anchor_deltas, wemb], dim=-1)
        h, _ = self.lstm(x)
        return anchor_deltas + self.out(h)      # anchor + learned style residual


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(epochs=100, batch_size=16, lr=1e-3, hidden=128, layers=2, pos_weight=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    writers = sorted(p.stem for p in PAIR_CACHE_DIR.glob("*.pkl"))
    if not writers:
        print("No pair cache found -- run with --build-cache first.")
        return
    print(f"writers ({len(writers)}): {writers}")

    train_ds = PairDataset(writers, is_train=True)
    val_ds = PairDataset(writers, is_train=False)
    print(f"train lines: {len(train_ds)}   val lines: {len(val_ds)}")
    if len(train_ds) == 0:
        print("Nothing to train on -- stopping.")
        return
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
                                               shuffle=True, collate_fn=collate)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size,
                                             shuffle=False, collate_fn=collate)

    model = StyleTransferRNN(len(writers), hidden=hidden, layers=layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    start_epoch, best_val = 1, float("inf")
    if CKPT_PATH.exists():
        ck = torch.load(CKPT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"] + 1
        best_val = ck["best_val"]
        print(f"[Resume] epoch {ck['epoch']} -> {start_epoch}, best_val={best_val:.5f}")

    def run_batch(writers_b, anchors, targets, train_mode):
        writers_b, anchors, targets = writers_b.to(device), anchors.to(device), targets.to(device)
        pred = model(anchors, writers_b)
        delta_loss = F.smooth_l1_loss(pred, targets)
        # Per-step delta loss alone lets small, locally-plausible errors
        # accumulate through the cumsum reconstruction into large absolute
        # drift over 400 integration steps -- confirmed directly: a model
        # that achieved LOW delta loss on a training example still
        # rendered as an illegible scribble, because cumsum(pred) had
        # drifted far from cumsum(targets) despite each individual step
        # looking fine in isolation. This term penalizes that drift
        # directly, on the actual rendered path, not just its derivative.
        pos_loss = F.smooth_l1_loss(torch.cumsum(pred, dim=1), torch.cumsum(targets, dim=1))
        loss = delta_loss + pos_weight * pos_loss
        if train_mode:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        return loss.item()

    n_train_batches = len(train_loader)
    print(f"\n--- Training styletransfer | lr={lr} batch_size={batch_size} "
          f"max_epochs={epochs} | {n_train_batches} batches/epoch ---\n")

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for bi, batch in enumerate(train_loader):
            l = run_batch(*batch, train_mode=True)
            if math.isfinite(l):
                tot += l; n += 1
            if (bi + 1) % max(1, n_train_batches // 4) == 0:
                elapsed = time.time() - t0
                spb = elapsed / (bi + 1)
                print(f"[styletransfer] Epoch {epoch:03d} | Batch {bi+1:04d}/{n_train_batches} "
                      f"| Running loss {tot/max(1,n):.5f} | {spb:.3f}s/batch", flush=True)
        train_loss = tot / max(1, n)

        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                l = run_batch(*batch, train_mode=False)
                if math.isfinite(l):
                    tot += l; n += 1
        val_loss = tot / max(1, n)
        dt = time.time() - t0
        saved_msg = ""

        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), epoch=epoch,
                        best_val=min(best_val, val_loss), writers=writers,
                        hidden=hidden, layers=layers), CKPT_PATH)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(dict(model=model.state_dict(), writers=writers,
                            hidden=hidden, layers=layers), BEST_PATH)
            saved_msg = " [BEST SAVED]"
        print(f"[styletransfer] Epoch {epoch:03d}/{epochs} | Train {train_loss:.5f} "
              f"| Val {val_loss:.5f} | {dt:.1f}s/epoch{saved_msg}", flush=True)


# ---------------------------------------------------------------------------
# Generation -- ONE forward pass, no sampling loop
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample(text, writer_id, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(BEST_PATH, map_location=device, weights_only=False)
    writers = ck["writers"]
    if writer_id not in writers:
        raise ValueError(f"writer {writer_id!r} not in trained set {writers}")
    model = StyleTransferRNN(len(writers), hidden=ck["hidden"], layers=ck["layers"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    profiles = SY.LoadAllProfiles()
    prof = profiles[writer_id]
    traj = SY.SynthesizeText(text, prof, mmPerXh=MM_PER_XH, seed=0,
                             lineWidthMm=100_000.0, legibility=1.0)
    anchor_xy, anchor_eos = trajectory_to_xy(traj, MM_PER_XH)
    anchor_xy = anchor_xy - anchor_xy[0]
    a_rs, a_eos_rs = _resample_arclength(anchor_xy, N_POINTS, eos=anchor_eos)
    anchor_d = np.diff(a_rs, axis=0, prepend=a_rs[:1]).astype(np.float32)

    w_idx = torch.tensor([writers.index(writer_id)], device=device)
    pred_d = model(torch.from_numpy(anchor_d).unsqueeze(0).to(device), w_idx)
    pred_xy = np.cumsum(pred_d[0].cpu().numpy(), axis=0)
    return pred_xy, a_rs, a_eos_rs


def render_xy(xy, eos=None, px_per_xh=40, pad=20):
    """Render a (dx,dy)-derived polyline. When `eos` (per-point pen-lift
    flags) is given, breaks the drawn line at those points instead of
    drawing one unbroken stroke -- the network only ever predicts shape,
    the anchor's own lift points decide where letters/words separate."""
    xs, ys = xy[:, 0] * px_per_xh, -xy[:, 1] * px_per_xh
    W = int(xs.max() - xs.min()) + 2 * pad
    H = int(ys.max() - ys.min()) + 2 * pad
    im = Image.new("L", (max(10, W), max(10, H)), 255)
    d = ImageDraw.Draw(im)
    x0, y0 = xs.min() - pad, ys.min() - pad
    pts = list(zip((xs - x0).tolist(), (ys - y0).tolist()))
    if eos is None:
        d.line(pts, fill=0, width=2, joint="curve")
        return im
    chain = [pts[0]]
    for i in range(1, len(pts)):
        # check the break BEFORE adding point i to the chain -- otherwise
        # the closing draw call includes point i, drawing a spurious line
        # straight across the pen-lift gap to the next stroke (this is
        # exactly what was producing the connecting lines between words).
        if eos[i - 1] > 0.5:
            if len(chain) >= 2:
                d.line(chain, fill=0, width=2, joint="curve")
            chain = [pts[i]]
        else:
            chain.append(pts[i])
    if len(chain) >= 2:
        d.line(chain, fill=0, width=2, joint="curve")
    return im


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--force-cache", action="store_true")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--pos-weight", type=float, default=1.0,
                    help="weight on the integrated-path (cumsum) loss term, "
                         "which controls absolute drift -- see run_batch")
    ap.add_argument("--sample", default=None)
    ap.add_argument("--writer", default=None)
    ap.add_argument("--out", default="styletransfer_sample.png")
    ap.add_argument("--compare", action="store_true",
                    help="with --sample, also render the raw anchor beside it")
    args = ap.parse_args()

    if args.build_cache:
        build_cache(force=args.force_cache)
    elif args.sample:
        pred_xy, anchor_xy, anchor_eos = sample(args.sample, args.writer)
        im = render_xy(pred_xy, eos=anchor_eos)
        im.save(args.out)
        print(f"-> {args.out}  {im.size}")
        if args.compare:
            aim = render_xy(anchor_xy, eos=anchor_eos)
            aim.save(Path(args.out).with_stem(Path(args.out).stem + "_anchor"))
    else:
        train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
              hidden=args.hidden, layers=args.layers, pos_weight=args.pos_weight)
