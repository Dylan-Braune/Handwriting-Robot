"""From-scratch (numpy-only) forward pass for PaperCRNN (TrainText.py) + a
ReadText() matching VerifyRewrite.ReadText's behavior. No torch."""

from pathlib import Path

import numpy as np
from PIL import Image

from . import layers as L
from . import weight_io as W

SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent  # .../Software/CNN

# Same charset/mapping as TrainText.py (index 0 reserved for CTC blank).
CHARSET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:'\"!?()-"
CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(CHARSET)}
IDX_TO_CHAR = {idx: char for char, idx in CHAR_TO_IDX.items()}

INPUT_HEIGHT = 64
INPUT_WIDTH = 640

# Mirrors VerifyRewrite.py's TEXT_WEIGHTS fallback chain exactly.
DEFAULT_TEXT_WEIGHTS = SCRIPT_DIR / "NOGIT" / "weights" / "paper_cnn_bilstm_ctc_best.pt"
for _name in ("paper_cnn_bilstm_ctc_joint_best.pt", "paper_cnn_bilstm_ctc_hf_best.pt"):
    _candidate = SCRIPT_DIR / "NOGIT" / "weights" / _name
    if _candidate.exists():
        DEFAULT_TEXT_WEIGHTS = _candidate
        break


def resize_line_image_fixed(pil_img, height=INPUT_HEIGHT, width=INPUT_WIDTH, pad_value=255):
    """Ported verbatim from TrainText.py (already pure PIL, no torch)."""
    img = pil_img.convert("L")
    w, h = img.size
    if w <= 0 or h <= 0:
        return Image.new("L", (width, height), pad_value)
    s = min(height / h, width / w)
    new_w = max(1, round(w * s))
    new_h = max(1, round(h * s))
    resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("L", (width, height), pad_value)
    canvas.paste(resized, (0, max(0, (height - new_h) // 2)))
    return canvas


def tensor_from_resized(pil_img):
    """Same as TrainText.py's version, returns a plain numpy array
    (B=1,C=1,H,W) instead of a torch tensor."""
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    arr = 1.0 - arr
    arr = (arr - 0.5) / 0.5
    return arr[np.newaxis, np.newaxis, :, :]  # (1,1,H,W)


def decode_ctc(log_probs):
    """Greedy CTC decode. log_probs: (T,B,num_classes) numpy array."""
    best_path = np.argmax(log_probs, axis=2)
    decoded = []
    for batch_idx in range(best_path.shape[1]):
        prev, chars = None, []
        for token in best_path[:, batch_idx]:
            token = int(token)
            if token != 0 and token != prev:
                chars.append(IDX_TO_CHAR.get(token, ""))
            prev = token
        decoded.append("".join(chars))
    return decoded


def _conv_block_forward(x, sd, prefix, n_layers):
    """conv_block(in,out,n_layers) as nn.Sequential(Conv,BN,ReLU)*n_layers,
    indices 0,1,2, 3,4,5, ... -- conv at i*3, bn at i*3+1."""
    for i in range(n_layers):
        c = i * 3
        x = L.conv2d(x, sd[f"{prefix}.{c}.weight"], sd[f"{prefix}.{c}.bias"], padding=1)
        x = L.batchnorm2d(x, sd[f"{prefix}.{c+1}.weight"], sd[f"{prefix}.{c+1}.bias"],
                           sd[f"{prefix}.{c+1}.running_mean"], sd[f"{prefix}.{c+1}.running_var"])
        x = L.relu(x)
    return x


class PaperCRNNNumpy:
    def __init__(self, checkpoint_path=None, hidden_size=256):
        self.checkpoint_path = Path(checkpoint_path or DEFAULT_TEXT_WEIGHTS)
        self.sd = W.load_state_dict_numpy(self.checkpoint_path)
        self.hidden_size = hidden_size

        self.lstm_layer_params = []
        for l in range(2):
            suffix = "" if l == 0 else "_reverse"

            def get(name, layer=l):
                return self.sd[f"sequence.{name}_l{layer}"]

            fwd = dict(weight_ih=get("weight_ih"), weight_hh=get("weight_hh"),
                       bias_ih=get("bias_ih"), bias_hh=get("bias_hh"))
            bwd = dict(weight_ih=self.sd[f"sequence.weight_ih_l{l}_reverse"],
                       weight_hh=self.sd[f"sequence.weight_hh_l{l}_reverse"],
                       bias_ih=self.sd[f"sequence.bias_ih_l{l}_reverse"],
                       bias_hh=self.sd[f"sequence.bias_hh_l{l}_reverse"])
            self.lstm_layer_params.append({"fwd": fwd, "bwd": bwd})

    def forward(self, x):
        """x: (B,1,H,W) numpy array -> (T,B,num_classes) log-probs."""
        sd = self.sd
        x = _conv_block_forward(x, sd, "stage1", 2)
        x = L.maxpool2d(x, 2, 2)
        x = _conv_block_forward(x, sd, "stage2", 4)
        x = L.maxpool2d(x, 2, 2)
        x = _conv_block_forward(x, sd, "stage3", 6)
        x = L.adaptive_max_pool_height(x)           # (B, 128, 1, W)
        x = x[:, :, 0, :]                            # (B, 128, W)
        x = np.transpose(x, (2, 0, 1))               # (W, B, 128)
        seq = L.multilayer_bidirectional_lstm(x, self.lstm_layer_params, self.hidden_size)
        logits = L.linear(seq, sd["text_head.weight"], sd["text_head.bias"])
        return L.log_softmax(logits, axis=2)


def ReadText(pil_image, model):
    """Matches VerifyRewrite.ReadText's behavior (image -> decoded string)."""
    t = tensor_from_resized(resize_line_image_fixed(pil_image))
    log_probs = model.forward(t)
    return decode_ctc(log_probs)[0]


def levenshtein(a, b):
    """Ported verbatim from TrainText.py -- pure string edit distance."""
    if len(a) < len(b):
        return levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    previous = list(range(len(b) + 1))
    for ca in a:
        current = [previous[0] + 1]
        for j, cb in enumerate(b):
            current.append(min(current[j] + 1, previous[j + 1] + 1, previous[j] + (ca != cb)))
        previous = current
    return previous[-1]


def CharAcc(pred, truth):
    """1 - CER, floored at 0. Ported verbatim from VerifyRewrite.py."""
    if not truth:
        return 1.0 if not pred else 0.0
    return max(0.0, 1.0 - levenshtein(pred, truth) / len(truth))


def WordAcc(pred, truth):
    """Fraction of the intended words recovered in order (LCS over words).
    Ported verbatim from VerifyRewrite.py."""
    tw, pw = truth.split(), pred.split()
    if not tw:
        return 1.0
    m, n = len(tw), len(pw)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            dp[i + 1][j + 1] = (dp[i][j] + 1 if tw[i].lower() == pw[j].lower()
                                else max(dp[i][j + 1], dp[i + 1][j]))
    return dp[m][n] / m
