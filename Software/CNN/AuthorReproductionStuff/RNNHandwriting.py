"""
RNNHandwriting.py -- a from-scratch, self-trained recurrent generator for
author-conditioned handwriting synthesis.

This is a DIFFERENT reproduction approach from BuildStyleProfile.py /
SynthesizeHandwriting.py (the trajectory-library method used elsewhere in
this project). Where that method stores real letter fragments and tiles
them, this one trains a single neural network -- built and trained by the
student, from scratch, on the student's own extracted stroke data -- to
GENERATE a continuous pen path directly, character by character, in a
chosen author's style. No pretrained model, weights, or reproduction
library is used anywhere in this file (contrast DiffBrush/GAN tools, which
are pretrained third-party generators and are NOT used here).

ARCHITECTURE (Graves 2013, "Generating Sequences With Recurrent Neural
Networks" -- the same architecture behind tools like Calligrapher.ai,
scaled down here for a ~600-line, 10-author dataset instead of the tens of
thousands of lines that architecture normally trains on):

    at each timestep t:
      input   x_t = (dx, dy, end-of-stroke) of the PREVIOUS real point,
              concatenated with a learned per-writer embedding
      LSTM 1  -> hidden state h1_t
      window  a Gaussian "soft attention" over the target text's one-hot
              characters, whose center (kappa) only ever moves forward --
              this is what lets the network read through the target text
              at its own pace while it writes, instead of needing the text
              pre-aligned to strokes the way the library method does
      LSTM 2  takes x_t, h1_t, and the window vector -> hidden state h2_t
      output  a mixture of M bivariate Gaussians over the next (dx, dy),
              plus a Bernoulli probability of lifting the pen

Trained by next-step log-likelihood (teacher forcing) on real stroke
sequences reconstructed from the SAME skeletonize-and-trace pipeline
BuildStyleProfile.py already uses -- but kept as one continuous per-LINE
sequence (ordered left-to-right) rather than cut into per-character pieces,
since this model needs to learn the actual pen dynamics between letters,
not just isolated letterforms.

HONEST LIMITATIONS, stated up front:
  * ~600 lines across 10 writers is very little data for this class of
    model. Expect it to need real training time and probably won't reach
    the polish of a model trained on thousands of writers -- that is the
    trade this approach makes for being fully self-trained on your own
    data instead of a pretrained library.
  * The stroke order is a RECONSTRUCTION (left-to-right by mean x), not a
    real recorded pen trace -- these are offline scans, not tablet data.
    A real writer occasionally breaks strict left-to-right order (dotting
    an i after finishing a word, say); that nuance is lost here.
  * Sampling from a freshly-trained mixture density network tends to look
    scratchy/illegible unless you turn DOWN the sampling variance at
    generation time (the `bias` parameter in sample()) -- this is a known,
    expected property of this architecture, not a bug.

USAGE
    python RNNHandwriting.py --build-cache        # extract + cache strokes (do this once)
    python RNNHandwriting.py --epochs 200 --batch-size 32
    python RNNHandwriting.py --sample "hello world" --writer 153 --bias 0.8
"""
import argparse
import math
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

import BuildStyleProfile as SP
from TrainText import IAMLineDatasetRaw, _decode_png, CHARSET

SCRIPT_DIR = Path(__file__).resolve().parent
NOGIT_DIR = SCRIPT_DIR.parent / "NOGIT"
STROKE_CACHE_DIR = NOGIT_DIR / "RNNStrokeCache10"
WEIGHTS_DIR = NOGIT_DIR / "weights"
CKPT_PATH = WEIGHTS_DIR / "rnn_handwriting_checkpoint.pt"
BEST_PATH = WEIGHTS_DIR / "rnn_handwriting_best.pt"

VOCAB = CHARSET + "\x00"          # \x00 = padding/unknown character slot
VOCAB_SIZE = len(VOCAB)
MAX_STROKE_LEN = 2000               # measured on real lines: median ~1300, up to
                                    # ~1500+ points for a full multi-word line at
                                    # this skeleton resolution -- an earlier guess
                                    # of 700 here was never checked against real
                                    # data and silently dropped ~95% of all lines
MAX_TEXT_LEN = 80


# ---------------------------------------------------------------------------
# Stroke extraction: same skeletonize-and-trace pipeline BuildStyleProfile.py
# uses, kept as ONE continuous per-line sequence instead of being cut into
# per-character pieces.
# ---------------------------------------------------------------------------
def extract_line_strokes(gray):
    """gray -> (N,3) float32 array of (dx, dy, end_of_stroke), x-height
    normalized, baseline at y=0, y up -- or None if the line is unusable.

    Order is imposed by sorting traced polylines by their own mean x
    (left-to-right); within a polyline, point order comes straight from
    the skeleton trace. This is the standard offline-to-pseudo-online
    approximation -- see the module docstring's limitations note."""
    ink = SP.BinarizeLine(gray)
    if not ink.any():
        return None
    band = SP.CoreBand(ink)
    if band is None:
        return None
    top, base = band
    xh = float(base - top)
    if xh < 6:
        return None
    skel = SP.Skeletonize(ink)
    if not skel.any():
        return None
    polys = [SP.SimplifyPolyline(p, eps=0.5) for p in SP.TracePolylines(skel)]
    polys = [p for p in polys if len(p) >= 2]
    if not polys:
        return None
    polys.sort(key=lambda p: float(np.mean([pt[0] for pt in p])))

    pts, eos = [], []
    for p in polys:
        for (x, y) in p:
            pts.append((x / xh, (base - y) / xh))
            eos.append(0.0)
        eos[-1] = 1.0                     # pen lifts after the last point of this stroke
    if len(pts) < 4:
        return None
    abs_xy = np.asarray(pts, dtype=np.float32)
    deltas = np.zeros_like(abs_xy)
    deltas[0] = abs_xy[0]                 # first point: offset from the origin
    deltas[1:] = abs_xy[1:] - abs_xy[:-1]
    seq = np.concatenate([deltas, np.asarray(eos, dtype=np.float32)[:, None]], axis=1)
    if len(seq) > MAX_STROKE_LEN:
        return None
    return seq


def text_to_ids(text):
    """Character -> index into VOCAB (0..VOCAB_SIZE-1). This model has no
    CTC blank symbol, so it uses its own plain index space over VOCAB
    rather than TrainText.CHAR_TO_IDX's 1-indexed (blank=0) scheme."""
    return [VOCAB.index(c) if c in VOCAB else VOCAB_SIZE - 1 for c in text[:MAX_TEXT_LEN]]


# ---------------------------------------------------------------------------
# Dataset: build once, cache to disk (same convention as GlyphCache10)
# ---------------------------------------------------------------------------
def build_cache(force=False):
    STROKE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    base = IAMLineDatasetRaw(root_dir=str(SP.DATA_DIR), cache_dir=str(SP.CACHE_DIR))
    by_author = {}
    for s in base.samples:
        by_author.setdefault(s["page_key"].split("/")[0], []).append(s)

    writers = sorted(by_author)
    print(f"writers: {writers}")
    for a in writers:
        out_path = STROKE_CACHE_DIR / f"{a}.pkl"
        if out_path.exists() and not force:
            print(f"  {a}: cached, skipping")
            continue
        rows = []
        for s in by_author[a]:
            gray = np.array(_decode_png(s["image_png"]).convert("L"))
            seq = extract_line_strokes(gray)
            if seq is None:
                continue
            rows.append(dict(text=s["text"], stroke=seq, is_holdout=s["is_holdout"]))
        with open(out_path, "wb") as f:
            pickle.dump(rows, f)
        print(f"  {a}: {len(rows)}/{len(by_author[a])} lines usable -> {out_path.name}")
    return writers


def build_teklia_cache(max_lines=2500, force=False):
    """Pretraining data: the same 6480-line Teklia/IAM-line HF dataset
    TrainTextHF.py uses (clean, pre-segmented, no local page/SegmentPage
    pipeline needed) -- but it carries only (image, text), NO writer ID.
    So it can only ever train the SHARED backbone (general pen dynamics),
    never per-author style -- everything here is filed under one placeholder
    writer, "GENERIC". Real per-author conditioning still comes only from
    build_cache()'s 10-author extraction. Capped at max_lines by default
    since skeletonizing+tracing every line is a real one-time CPU cost;
    raise it if you have the time to spare -- more pretraining data only
    helps the shared backbone."""
    from datasets import load_dataset
    out_dir = NOGIT_DIR / "RNNStrokeCache_teklia"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pretrain.pkl"
    if out_path.exists() and not force:
        print(f"{out_path} already exists, skipping (pass force=True to rebuild)")
        return out_path

    rows = []
    for split, is_train in [("train", True), ("validation", False)]:
        ds = load_dataset("Teklia/IAM-line", split=split)
        n = min(len(ds), max_lines if is_train else max(200, max_lines // 8))
        kept = 0
        for i in range(n):
            item = ds[i]
            gray = np.array(item["image"].convert("L"))
            seq = extract_line_strokes(gray)
            if seq is None:
                continue
            rows.append(dict(text=item["text"], stroke=seq, is_holdout=not is_train))
            kept += 1
            if (i + 1) % 250 == 0:
                print(f"  {split}: {i+1}/{n} scanned, {kept} usable so far", flush=True)
        print(f"{split}: {kept}/{n} lines usable")
    with open(out_path, "wb") as f:
        pickle.dump(rows, f)
    print(f"-> {out_path}  ({len(rows)} total lines)")
    return out_path


class StrokeDataset(torch.utils.data.Dataset):
    """writers=None + pretrain_path set -> the single-"GENERIC"-writer
    Teklia pretraining set. Otherwise the normal per-author 10-writer set."""
    def __init__(self, writers=None, is_train=True, pretrain_path=None):
        self.samples = []
        if pretrain_path is not None:
            self.writer_to_idx = {"GENERIC": 0}
            rows = pickle.load(open(pretrain_path, "rb"))
            for r in rows:
                if r["is_holdout"] == (not is_train):
                    self.samples.append(("GENERIC", r["text"], r["stroke"]))
            return
        self.writer_to_idx = {w: i for i, w in enumerate(writers)}
        for w in writers:
            rows = pickle.load(open(STROKE_CACHE_DIR / f"{w}.pkl", "rb"))
            for r in rows:
                if r["is_holdout"] == (not is_train):
                    self.samples.append((w, r["text"], r["stroke"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        w, text, stroke = self.samples[i]
        return (self.writer_to_idx[w], np.asarray(text_to_ids(text), dtype=np.int64),
                stroke.astype(np.float32))


def collate(batch):
    writers = torch.tensor([b[0] for b in batch], dtype=torch.long)
    text_lens = [len(b[1]) for b in batch]
    stroke_lens = [len(b[2]) for b in batch]
    Tt, Ts = max(text_lens), max(stroke_lens)
    texts = torch.full((len(batch), Tt), VOCAB_SIZE - 1, dtype=torch.long)
    strokes = torch.zeros(len(batch), Ts, 3, dtype=torch.float32)   # (dx, dy, eos)
    for i, (_w, t, s) in enumerate(batch):
        texts[i, :len(t)] = torch.from_numpy(t)
        strokes[i, :len(s)] = torch.from_numpy(s)
    return (writers, texts, torch.tensor(text_lens), strokes, torch.tensor(stroke_lens))


# ---------------------------------------------------------------------------
# Model: LSTM + Gaussian window attention + mixture-density output
# ---------------------------------------------------------------------------
class HandwritingRNN(nn.Module):
    def __init__(self, n_writers, hidden=256, n_window=6, n_mix=20, writer_dim=32):
        super().__init__()
        self.n_window = n_window
        self.n_mix = n_mix
        self.writer_embed = nn.Embedding(n_writers, writer_dim)
        in1 = 3 + writer_dim + VOCAB_SIZE
        self.lstm1 = nn.LSTM(in1, hidden, batch_first=True)
        self.window_layer = nn.Linear(hidden, 3 * n_window)
        in2 = 3 + writer_dim + hidden + VOCAB_SIZE
        self.lstm2 = nn.LSTM(in2, hidden, batch_first=True)
        # MDN output: n_mix * (pi, mux, muy, sx, sy, rho) + 1 (eos logit)
        self.out = nn.Linear(hidden, n_mix * 6 + 1)

    def forward(self, x, writer_idx, text_onehot, text_lens, kappa0=None, h1c1=None, h2c2=None):
        """x: (B,T,3) real/previous points. text_onehot: (B,U,VOCAB_SIZE).
        Returns mixture params, end-of-text-window flag, and final states
        (for step-by-step generation)."""
        B, T, _ = x.shape
        U = text_onehot.shape[1]
        wemb = self.writer_embed(writer_idx).unsqueeze(1).expand(-1, T, -1)
        if kappa0 is None:
            kappa = x.new_zeros(B, self.n_window)
        else:
            kappa = kappa0
        if h1c1 is None:
            h1 = x.new_zeros(1, B, self.lstm1.hidden_size)
            c1 = x.new_zeros(1, B, self.lstm1.hidden_size)
        else:
            h1, c1 = h1c1
        if h2c2 is None:
            h2 = x.new_zeros(1, B, self.lstm2.hidden_size)
            c2 = x.new_zeros(1, B, self.lstm2.hidden_size)
        else:
            h2, c2 = h2c2

        windows, phis = [], []
        h1_seq = []
        u_idx = torch.arange(U, device=x.device).float().view(1, 1, U)
        for t in range(T):
            inp1 = torch.cat([x[:, t], wemb[:, t], (windows[-1] if windows else
                              x.new_zeros(B, VOCAB_SIZE))], dim=-1).unsqueeze(1)
            out1, (h1, c1) = self.lstm1(inp1, (h1, c1))
            h1_seq.append(out1)
            wp = torch.exp(self.window_layer(out1.squeeze(1)))       # (B, 3K), all positive
            alpha, beta, kappa_hat = wp.chunk(3, dim=-1)
            kappa = kappa + kappa_hat                                 # monotone, never resets
            phi = (alpha.unsqueeze(-1) *
                   torch.exp(-beta.unsqueeze(-1) * (kappa.unsqueeze(-1) - u_idx) ** 2)
                   ).sum(dim=1)                                       # (B, U)
            phis.append(phi)
            w_t = torch.bmm(phi.unsqueeze(1), text_onehot).squeeze(1)  # (B, VOCAB_SIZE)
            windows.append(w_t)
        h1_seq = torch.cat(h1_seq, dim=1)
        window_seq = torch.stack(windows, dim=1)
        phi_seq = torch.stack(phis, dim=1)                            # (B,T,U)

        inp2 = torch.cat([x, wemb, h1_seq, window_seq], dim=-1)
        out2, (h2, c2) = self.lstm2(inp2, (h2, c2))
        raw = self.out(out2)
        M = self.n_mix
        pi_hat, mux, muy, sx_hat, sy_hat, rho_hat, eos_logit = torch.split(
            raw, [M, M, M, M, M, M, 1], dim=-1)
        pi = F.log_softmax(pi_hat, dim=-1)
        sx, sy = torch.exp(sx_hat).clamp(min=1e-4), torch.exp(sy_hat).clamp(min=1e-4)
        rho = torch.tanh(rho_hat) * 0.999
        eos_logit = eos_logit.squeeze(-1)

        # a text-length-aware "finished reading" flag, used at generation
        # time to know when to stop (Graves' termination rule: once the
        # window's attention has moved past the end of the text)
        done = phi_seq[:, :, -1] > phi_seq.max(dim=-1).values * 0.5 if U > 0 else None

        return (dict(logpi=pi, mux=mux, muy=muy, sx=sx, sy=sy, rho=rho, eos_logit=eos_logit),
                kappa, (h1, c1), (h2, c2), phi_seq)


def mdn_loss(params, target, mask):
    dx, dy, eos = target[..., 0], target[..., 1], target[..., 2]
    mux, muy, sx, sy, rho = params["mux"], params["muy"], params["sx"], params["sy"], params["rho"]
    dxn = (dx.unsqueeze(-1) - mux) / sx
    dyn = (dy.unsqueeze(-1) - muy) / sy
    z = dxn ** 2 + dyn ** 2 - 2 * rho * dxn * dyn
    one_minus_rho2 = (1 - rho ** 2).clamp(min=1e-6)
    log_gauss = (-math.log(2 * math.pi) - torch.log(sx) - torch.log(sy)
                 - 0.5 * torch.log(one_minus_rho2) - z / (2 * one_minus_rho2))
    log_mix = torch.logsumexp(params["logpi"] + log_gauss, dim=-1)
    log_eos = -F.binary_cross_entropy_with_logits(params["eos_logit"], eos, reduction="none")
    nll = -(log_mix + log_eos)
    return (nll * mask).sum() / mask.sum().clamp(min=1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(epochs=200, batch_size=32, lr=1e-3, hidden=256, n_mix=20, n_window=6,
         pretrain=False, finetune_from=None, name=None):
    import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    if pretrain:
        name = name or "rnn_pretrain"
        pretrain_path = NOGIT_DIR / "RNNStrokeCache_teklia" / "pretrain.pkl"
        if not pretrain_path.exists():
            print(f"No Teklia pretrain cache at {pretrain_path} -- run with "
                  f"--build-teklia-cache first.")
            return
        writers = ["GENERIC"]
        train_ds = StrokeDataset(pretrain_path=pretrain_path, is_train=True)
        val_ds = StrokeDataset(pretrain_path=pretrain_path, is_train=False)
    else:
        name = name or "rnn_handwriting"
        writers = sorted(p.stem for p in STROKE_CACHE_DIR.glob("*.pkl"))
        if not writers:
            print("No stroke cache found -- run with --build-cache first.")
            return
        train_ds = StrokeDataset(writers, is_train=True)
        val_ds = StrokeDataset(writers, is_train=False)
    print(f"writers ({len(writers)}): {writers}")
    print(f"train lines: {len(train_ds)}   val lines: {len(val_ds)}")
    if len(train_ds) == 0:
        print("Nothing to train on -- stopping.")
        return
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
                                               shuffle=True, collate_fn=collate)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size,
                                             shuffle=False, collate_fn=collate)

    ckpt_path = WEIGHTS_DIR / f"{name}_checkpoint.pt"
    best_path = WEIGHTS_DIR / f"{name}_best.pt"
    model = HandwritingRNN(len(writers), hidden=hidden, n_window=n_window, n_mix=n_mix).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    start_epoch, best_val = 1, float("inf")
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"] + 1
        best_val = ck["best_val"]
        print(f"[Resume] epoch {ck['epoch']} -> {start_epoch}, best_val={best_val:.4f}")
    elif finetune_from:
        # load everything EXCEPT the writer embedding (the pretrain run only
        # ever had one "GENERIC" writer slot; this run has real authors) --
        # the shared backbone (pen dynamics, window attention, MDN output)
        # transfers, the per-writer style embedding starts fresh
        src = torch.load(finetune_from, map_location=device, weights_only=False)
        src_sd = {k: v for k, v in src["model"].items() if not k.startswith("writer_embed")}
        missing, unexpected = model.load_state_dict(src_sd, strict=False)
        print(f"[Finetune] loaded backbone from {finetune_from} "
              f"(writer embedding left fresh; {len(missing)} keys reinitialized)")

    def run_batch(writers_b, texts, text_lens, strokes, stroke_lens, train_mode):
        writers_b, texts, strokes = writers_b.to(device), texts.to(device), strokes.to(device)
        onehot = F.one_hot(texts, VOCAB_SIZE).float()
        x = torch.cat([torch.zeros_like(strokes[:, :1]), strokes[:, :-1]], dim=1)
        target = strokes
        params, *_ = model(x, writers_b, onehot, text_lens)
        T = strokes.shape[1]
        mask = (torch.arange(T, device=device).unsqueeze(0) < stroke_lens.to(device).unsqueeze(1)).float()
        loss = mdn_loss(params, target, mask)
        if train_mode:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        return loss.item()

    n_train_batches = len(train_loader)
    print(f"\n--- Training {name} ({'pretrain, GENERIC writer' if pretrain else 'per-author'}) "
          f"| lr={lr} batch_size={batch_size} max_epochs={epochs} "
          f"| {n_train_batches} batches/epoch ---\n")

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
                eta = spb * (n_train_batches - bi - 1)
                print(f"[{name}] Epoch {epoch:03d} | Batch {bi+1:04d}/{n_train_batches} "
                      f"| Running NLL {tot/max(1,n):.3f} | {spb:.2f}s/batch | ETA {eta/60:.1f} min",
                      flush=True)
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
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                        epoch=epoch, best_val=min(best_val, val_loss), writers=writers,
                        hidden=hidden, n_mix=n_mix, n_window=n_window), ckpt_path)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(dict(model=model.state_dict(), writers=writers,
                            hidden=hidden, n_mix=n_mix, n_window=n_window), best_path)
            saved_msg = " [BEST SAVED]"
        print(f"[{name}] Epoch {epoch:03d}/{epochs} | Train NLL {train_loss:.3f} "
              f"| Val NLL {val_loss:.3f} | {dt:.1f}s/epoch{saved_msg}", flush=True)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample(text, writer_id, bias=0.5, max_steps=1200, device=None):
    """bias in [0, ~2]: HIGHER bias = lower sampling variance = more
    legible/typical output at the cost of natural-looking randomness --
    Graves' standard trick for making a freshly-sampled MDN output usable.
    bias=0 is unbiased (raw) sampling and usually looks scratchy."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(BEST_PATH, map_location=device, weights_only=False)
    writers = ck["writers"]
    model = HandwritingRNN(len(writers), hidden=ck["hidden"],
                           n_window=ck["n_window"], n_mix=ck["n_mix"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    if writer_id not in writers:
        raise ValueError(f"writer {writer_id!r} not in trained set {writers}")
    w_idx = torch.tensor([writers.index(writer_id)], device=device)
    ids = torch.tensor([text_to_ids(text)], dtype=torch.long, device=device)
    onehot = F.one_hot(ids, VOCAB_SIZE).float()
    U = onehot.shape[1]

    x = torch.zeros(1, 1, 3, device=device)
    kappa = h1c1 = h2c2 = None
    points = []
    for step in range(max_steps):
        params, kappa, h1c1, h2c2, phi = model(x, w_idx, onehot, None,
                                                kappa0=kappa, h1c1=h1c1, h2c2=h2c2)
        # Graves' bias trick: sharpen the mixture-choice distribution AND
        # shrink each component's spread, so sampling favours the most
        # typical stroke at each step instead of the full raw spread the
        # network learned. bias=0 is unbiased (usually scratchy); higher
        # values trade natural-looking variation for legibility.
        logpi = params["logpi"][:, 0] * (1.0 + bias)
        pi = F.softmax(logpi, dim=-1)[0]
        m = torch.multinomial(pi, 1).item()
        sx = params["sx"][0, 0, m].item() * math.exp(-bias)
        sy = params["sy"][0, 0, m].item() * math.exp(-bias)
        mux, muy = params["mux"][0, 0, m].item(), params["muy"][0, 0, m].item()
        rho = params["rho"][0, 0, m].item()
        z1, z2 = np.random.randn(), np.random.randn()
        dx = mux + sx * z1
        dy = muy + sy * (rho * z1 + math.sqrt(max(1e-6, 1 - rho ** 2)) * z2)
        eos = torch.sigmoid(params["eos_logit"][0, 0]).item() > 0.5
        points.append((dx, dy, float(eos)))
        x = torch.tensor([[[dx, dy, float(eos)]]], device=device)
        if phi[0, 0, -1].item() > phi[0, 0].max().item() * 0.5 and step > 10 * U:
            break
    return np.asarray(points, dtype=np.float32)


def render_strokes(seq, px_per_xh=40, pad=20):
    """deltas (dx,dy,eos) -> a PIL image, for looking at what came out."""
    abs_xy = np.cumsum(seq[:, :2], axis=0)
    xs, ys = abs_xy[:, 0] * px_per_xh, -abs_xy[:, 1] * px_per_xh
    W = int(xs.max() - xs.min()) + 2 * pad
    H = int(ys.max() - ys.min()) + 2 * pad
    im = Image.new("L", (max(10, W), max(10, H)), 255)
    d = ImageDraw.Draw(im)
    x0, y0 = xs.min() - pad, ys.min() - pad
    stroke = []
    for (x, y), eos in zip(zip(xs - x0, ys - y0), seq[:, 2]):
        stroke.append(x)
        stroke.append(y)
    pts = list(zip(xs - x0, ys - y0))
    chain = [pts[0]]
    for i in range(1, len(pts)):
        chain.append(pts[i])
        if seq[i - 1, 2] > 0.5:
            if len(chain) >= 2:
                d.line(chain, fill=0, width=2, joint="curve")
            chain = [pts[i]]
    if len(chain) >= 2:
        d.line(chain, fill=0, width=2, joint="curve")
    return im


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-cache", action="store_true",
                    help="extract+cache strokes for the 10 local authors")
    ap.add_argument("--build-teklia-cache", action="store_true",
                    help="extract+cache strokes from the Teklia/IAM-line HF "
                         "dataset for pretraining (no writer labels)")
    ap.add_argument("--force-cache", action="store_true")
    ap.add_argument("--max-teklia-lines", type=int, default=2500)
    ap.add_argument("--pretrain", action="store_true",
                    help="train the shared backbone on the Teklia cache "
                         "(single GENERIC writer, no per-author style)")
    ap.add_argument("--finetune-from", default=None,
                    help="path to a pretrain checkpoint's _best.pt to warm-"
                         "start the per-author run's shared backbone from")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--n-mix", type=int, default=20)
    ap.add_argument("--n-window", type=int, default=6)
    ap.add_argument("--sample", default=None, help="text to render")
    ap.add_argument("--writer", default=None)
    ap.add_argument("--bias", type=float, default=0.5)
    ap.add_argument("--out", default="rnn_sample.png")
    args = ap.parse_args()

    if args.build_cache:
        build_cache(force=args.force_cache)
    elif args.build_teklia_cache:
        build_teklia_cache(max_lines=args.max_teklia_lines, force=args.force_cache)
    elif args.sample:
        seq = sample(args.sample, args.writer, bias=args.bias)
        im = render_strokes(seq)
        im.save(args.out)
        print(f"-> {args.out}  {im.size}  ({len(seq)} points)")
    else:
        train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
              hidden=args.hidden, n_mix=args.n_mix, n_window=args.n_window,
              pretrain=args.pretrain, finetune_from=args.finetune_from)
