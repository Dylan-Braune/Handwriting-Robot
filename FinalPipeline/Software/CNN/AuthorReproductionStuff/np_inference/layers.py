"""Numpy building blocks for the from-scratch inference forward passes.
No torch anywhere in this file."""

import numpy as np


def conv2d(x, weight, bias, stride=1, padding=1):
    """x: (B,C,H,W), weight: (OutC,InC,kh,kw). im2col + matmul."""
    B, C, H, W = x.shape
    OutC, InC, kh, kw = weight.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    Hp, Wp = xp.shape[2], xp.shape[3]
    outH = (Hp - kh) // stride + 1
    outW = (Wp - kw) // stride + 1

    # im2col via as_strided: (B, C, kh, kw, outH, outW)
    s = xp.strides
    shape = (B, C, kh, kw, outH, outW)
    strides = (s[0], s[1], s[2], s[3], s[2] * stride, s[3] * stride)
    cols = np.lib.stride_tricks.as_strided(xp, shape=shape, strides=strides)
    cols = cols.reshape(B, C * kh * kw, outH * outW)

    w = weight.reshape(OutC, C * kh * kw)
    # plain matmul dispatches to BLAS; einsum with this subscript pattern
    # doesn't and was ~10x slower for identical output (verified bit-for-bit
    # equal to float rounding noise, ~1e-13) -- B is always 1 here in
    # practice (one image per forward pass) so this loop never costs much.
    out = np.stack([w @ cols[b] for b in range(B)], axis=0).reshape(B, OutC, outH, outW)
    if bias is not None:
        out = out + bias.reshape(1, OutC, 1, 1)
    return out


def batchnorm2d(x, weight, bias, running_mean, running_var, eps=1e-5):
    scale = weight / np.sqrt(running_var + eps)
    shift = bias - running_mean * scale
    return x * scale.reshape(1, -1, 1, 1) + shift.reshape(1, -1, 1, 1)


def relu(x):
    return np.maximum(x, 0)


def maxpool2d(x, kernel=2, stride=2):
    B, C, H, W = x.shape
    outH, outW = H // stride, W // stride
    xc = x[:, :, : outH * stride, : outW * stride]
    s = xc.strides
    shape = (B, C, outH, outW, kernel, kernel)
    strides = (s[0], s[1], s[2] * stride, s[3] * stride, s[2], s[3])
    windows = np.lib.stride_tricks.as_strided(xc, shape=shape, strides=strides)
    return windows.max(axis=(4, 5))


def adaptive_max_pool_height(x):
    """(B,C,H,W) -> (B,C,1,W), max over H."""
    return x.max(axis=2, keepdims=True)


def adaptive_avg_pool_width_1d(x):
    """(B,C,W) -> (B,C,1), mean over W."""
    return x.mean(axis=2, keepdims=True)


def linear(x, weight, bias):
    """weight: (out_features, in_features), matching nn.Linear."""
    out = x @ weight.T
    if bias is not None:
        out = out + bias
    return out


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def lstm_forward(x, weight_ih, weight_hh, bias_ih, bias_hh, hidden_size):
    """One direction, one layer. x: (T,B,input_size).
    Gate blocks in weight_ih/weight_hh (each 4*hidden rows) are ordered
    [input, forget, cell/g, output] -- PyTorch's LSTM convention."""
    T, B, _ = x.shape
    h = np.zeros((B, hidden_size), dtype=x.dtype)
    c = np.zeros((B, hidden_size), dtype=x.dtype)
    H = hidden_size
    outputs = np.empty((T, B, H), dtype=x.dtype)

    for t in range(T):
        gates = x[t] @ weight_ih.T + bias_ih + h @ weight_hh.T + bias_hh
        i = sigmoid(gates[:, 0 * H:1 * H])
        f = sigmoid(gates[:, 1 * H:2 * H])
        g = np.tanh(gates[:, 2 * H:3 * H])
        o = sigmoid(gates[:, 3 * H:4 * H])
        c = f * c + i * g
        h = o * np.tanh(c)
        outputs[t] = h

    return outputs, (h, c)


def bidirectional_lstm_forward(x, params_fwd, params_bwd, hidden_size):
    """x: (T,B,input_size) -> (T,B,2*hidden_size)."""
    out_f, _ = lstm_forward(x, **params_fwd, hidden_size=hidden_size)
    out_b, _ = lstm_forward(x[::-1], **params_bwd, hidden_size=hidden_size)
    out_b = out_b[::-1]
    return np.concatenate([out_f, out_b], axis=2)


def multilayer_bidirectional_lstm(x, layer_params, hidden_size):
    """layer_params: list of {"fwd": {...}, "bwd": {...}} weight dicts."""
    out = x
    for layer in layer_params:
        out = bidirectional_lstm_forward(out, layer["fwd"], layer["bwd"], hidden_size)
    return out


def log_softmax(x, axis):
    m = np.max(x, axis=axis, keepdims=True)
    shifted = x - m
    lse = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    return shifted - lse


def softmax(x, axis):
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    x = rng.standard_normal((2, 3, 8, 8)).astype(np.float32)
    w = rng.standard_normal((5, 3, 3, 3)).astype(np.float32)
    b = rng.standard_normal(5).astype(np.float32)
    out = conv2d(x, w, b)
    assert out.shape == (2, 5, 8, 8), out.shape

    bn = batchnorm2d(out, np.ones(5, np.float32), np.zeros(5, np.float32),
                      np.zeros(5, np.float32), np.ones(5, np.float32))
    assert bn.shape == out.shape

    assert relu(out).shape == out.shape

    pooled = maxpool2d(out, 2, 2)
    assert pooled.shape == (2, 5, 4, 4), pooled.shape

    ahp = adaptive_max_pool_height(pooled)
    assert ahp.shape == (2, 5, 1, 4), ahp.shape

    x1d = rng.standard_normal((2, 5, 4)).astype(np.float32)
    awp = adaptive_avg_pool_width_1d(x1d)
    assert awp.shape == (2, 5, 1), awp.shape

    lw = rng.standard_normal((10, 5)).astype(np.float32)
    lb = rng.standard_normal(10).astype(np.float32)
    lin_out = linear(rng.standard_normal((2, 5)).astype(np.float32), lw, lb)
    assert lin_out.shape == (2, 10), lin_out.shape

    T, B, IN, H = 6, 2, 4, 3
    xt = rng.standard_normal((T, B, IN)).astype(np.float32)
    w_ih = rng.standard_normal((4 * H, IN)).astype(np.float32)
    w_hh = rng.standard_normal((4 * H, H)).astype(np.float32)
    b_ih = rng.standard_normal(4 * H).astype(np.float32)
    b_hh = rng.standard_normal(4 * H).astype(np.float32)
    seq, (hT, cT) = lstm_forward(xt, w_ih, w_hh, b_ih, b_hh, H)
    assert seq.shape == (T, B, H) and hT.shape == (B, H) and cT.shape == (B, H)

    params_fwd = dict(weight_ih=w_ih, weight_hh=w_hh, bias_ih=b_ih, bias_hh=b_hh)
    params_bwd = dict(weight_ih=w_ih.copy(), weight_hh=w_hh.copy(), bias_ih=b_ih.copy(), bias_hh=b_hh.copy())
    bidi = bidirectional_lstm_forward(xt, params_fwd, params_bwd, H)
    assert bidi.shape == (T, B, 2 * H), bidi.shape

    layer_params = [{"fwd": params_fwd, "bwd": params_bwd}]
    multi = multilayer_bidirectional_lstm(xt, layer_params, H)
    assert multi.shape == (T, B, 2 * H)

    ls = log_softmax(rng.standard_normal((3, 4)).astype(np.float32), axis=1)
    assert ls.shape == (3, 4)
    sm = softmax(rng.standard_normal((3, 4)).astype(np.float32), axis=1)
    assert sm.shape == (3, 4) and np.allclose(sm.sum(axis=1), 1.0, atol=1e-5)

    print("layers.py self-test: all shapes OK")
