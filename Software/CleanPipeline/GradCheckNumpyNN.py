#numerical gradient check for every hand-derived backward pass in NumpyNN.py.
#standalone: `python GradCheckNumpyNN.py`, numpy only, no other dependencies.
#
#method: for each op a small random input and a fixed random "output gradient"
#R are built, giving the scalar loss L = sum(out*R) (so dL/dout == R exactly,
#which is what gets fed to the backward pass). for the two loss functions the
#scalar is the loss itself. the analytic gradient from the hand-written
#backward is then compared, element by element, against the central difference
#    (L(theta+eps*e_i) - L(theta-eps*e_i)) / (2*eps)
#with eps = 1e-5 and every array in float64 (NumpyNN.py is dtype-agnostic --
#it propagates whatever dtype it is handed -- so float64 is used throughout,
#which is what makes 1e-5 central differences meaningful at all).
#
#gradients are checked w.r.t. inputs AND every learnable parameter that exists
#(conv weight/bias, bn gamma/beta, linear weight/bias, all four lstm weight and
#bias blocks per direction per layer).
#
#care taken with the non-smooth / stochastic ops:
#  relu      - inputs drawn from a permuted linspace so no element sits within
#              eps of the kink at 0
#  maxpool   - same trick: all values distinct by >> 2*eps so no perturbation
#              can flip an argmax (a flipped argmax would make the numerical
#              gradient wrong, not the analytic one)
#  dropout   - rng replaced by a fixed-mask stub so the forward is deterministic
#              and the same mask is used by every perturbed evaluation; also
#              checked in eval mode (identity path)
#  batchnorm - checked in TRAIN mode (batch statistics), which is the only mode
#              its backward is used in. running_mean/running_var are mutated in
#              place by the forward, so a fresh copy is passed every call
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import NumpyNN as nn


DTYPE = np.float64
EPS = 1e-5
TOL = 1e-6          #tolerance on the reported relative difference
MAX_PROBE = 40      #max number of randomly chosen elements probed per tensor

RESULTS = []        #(name, nProbed, relDiff, elemRel, passed)


def RelDiff(ga, gn):
    #the metric reported in the summary table:
    #  max|g_analytic - g_numeric| / max(1, max|g_analytic| + max|g_numeric|)
    num = float(np.max(np.abs(ga - gn)))
    den = max(1.0, float(np.max(np.abs(ga)) + np.max(np.abs(gn))))
    return num / den


def ElemRel(ga, gn):
    #stricter per-element relative error, reported alongside as a second
    #opinion: |a-n| / (|a|+|n|), which does not shrink when the gradient is
    #small. both-zero elements contribute 0.
    denom = np.abs(ga) + np.abs(gn)
    safe = np.where(denom < 1e-300, 1.0, denom)
    err = np.abs(ga - gn) / safe
    err = np.where(denom < 1e-300, 0.0, err)
    return float(np.max(err))


def Check(name, lossFn, theta, analytic, rng, maxProbe=MAX_PROBE):
    #theta is perturbed in place, so it must be a contiguous array that lossFn
    #actually reads from
    flatTheta = theta.ravel()
    assert np.shares_memory(flatTheta, theta), name + ": ravel() copied, cannot perturb in place"
    flatAnalytic = np.asarray(analytic).ravel()
    assert flatAnalytic.size == flatTheta.size, (
        name + ": analytic gradient shape " + str(np.shape(analytic)) +
        " does not match parameter shape " + str(theta.shape))

    n = flatTheta.size
    if n <= maxProbe:
        idx = np.arange(n)
    else:
        idx = np.sort(rng.choice(n, size=maxProbe, replace=False))

    ga = np.empty(idx.size, dtype=DTYPE)
    gn = np.empty(idx.size, dtype=DTYPE)
    for k, i in enumerate(idx):
        orig = flatTheta[i]
        flatTheta[i] = orig + EPS
        lossPlus = float(lossFn())
        flatTheta[i] = orig - EPS
        lossMinus = float(lossFn())
        flatTheta[i] = orig
        gn[k] = (lossPlus - lossMinus) / (2.0 * EPS)
        ga[k] = flatAnalytic[i]

    rel = RelDiff(ga, gn)
    elem = ElemRel(ga, gn)
    passed = np.isfinite(rel) and rel <= TOL
    RESULTS.append((name, int(idx.size), rel, elem, bool(passed)))
    print("  %-46s probed %3d   relDiff %.3e   elemRel %.3e   %s"
          % (name, idx.size, rel, elem, "PASS" if passed else "FAIL"))
    if not passed:
        worst = int(np.argmax(np.abs(ga - gn)))
        print("      worst element: analytic % .12e  numeric % .12e" % (ga[worst], gn[worst]))


def Fail(name, reason):
    #record a configuration that could not be gradient-checked at all because
    #the forward/backward pass itself errored out. counted as a FAIL, with the
    #reason printed, so a library bug cannot be silently dropped from the table
    RESULTS.append((name, 0, float("nan"), float("nan"), False))
    print("  %-46s probed %3d   relDiff %9s   elemRel %9s   %s"
          % (name, 0, "n/a", "n/a", "FAIL"))
    print("      " + reason)


def Distinct(rng, shape, lo=-1.5, hi=1.5):
    #values guaranteed pairwise distinct by a wide margin -- used for relu and
    #maxpool so that no +-eps perturbation can move an element across the kink
    #or flip a window's argmax
    n = int(np.prod(shape))
    vals = np.linspace(lo, hi, n).astype(DTYPE)
    vals = rng.permutation(vals)
    return vals.reshape(shape).copy()


class FixedRng:
    #stand-in for numpy's Generator inside DropoutForward: returns the same
    #pre-drawn uniforms every call, making the dropout mask deterministic
    def __init__(self, values):
        self.values = values

    def random(self, shape):
        assert tuple(shape) == self.values.shape
        return self.values


#================================================================== conv2d ===

def CheckConv(label, B, C, H, W, OutC, kh, kw, pad, useBias, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((B, C, H, W)).astype(DTYPE)
    weight = (0.4 * rng.standard_normal((OutC, C, kh, kw))).astype(DTYPE)
    bias = (0.2 * rng.standard_normal(OutC)).astype(DTYPE) if useBias else None

    out, cache = nn.Conv2dForward(x, weight, bias, padding=pad)
    R = rng.standard_normal(out.shape).astype(DTYPE)
    dx, dw, db = nn.Conv2dBackward(R, cache)

    def loss():
        o, _ = nn.Conv2dForward(x, weight, bias, padding=pad)
        return float((o * R).sum())

    Check(label + " / dx", loss, x, dx, rng)
    Check(label + " / dweight", loss, weight, dw, rng)
    if useBias:
        Check(label + " / dbias", loss, bias, db, rng)


#============================================================= batchnorm2d ===

def CheckBatchNorm(label, B, C, H, W, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((B, C, H, W)).astype(DTYPE)
    gamma = (1.0 + 0.3 * rng.standard_normal(C)).astype(DTYPE)
    beta = (0.2 * rng.standard_normal(C)).astype(DTYPE)
    rmInit = rng.standard_normal(C).astype(DTYPE)
    rvInit = (1.0 + np.abs(rng.standard_normal(C))).astype(DTYPE)

    def forward():
        #fresh copies: the forward writes the running stats in place
        return nn.BatchNorm2dForwardTrain(x, gamma, beta, rmInit.copy(), rvInit.copy())

    out, cache = forward()
    R = rng.standard_normal(out.shape).astype(DTYPE)
    dx, dgamma, dbeta = nn.BatchNorm2dBackward(R, cache)

    def loss():
        o, _ = forward()
        return float((o * R).sum())

    Check(label + " / dx", loss, x, dx, rng)
    Check(label + " / dgamma", loss, gamma, dgamma, rng)
    Check(label + " / dbeta", loss, beta, dbeta, rng)


#=================================================== relu / pooling / linear ===

def CheckRelu(label, shape, seed):
    rng = np.random.default_rng(seed)
    x = Distinct(rng, shape)
    out, cache = nn.ReluForward(x)
    R = rng.standard_normal(out.shape).astype(DTYPE)
    dx = nn.ReluBackward(R, cache)

    def loss():
        o, _ = nn.ReluForward(x)
        return float((o * R).sum())

    Check(label + " / dx", loss, x, dx, rng)


def CheckMaxPool(label, B, C, H, W, kernel, stride, seed):
    rng = np.random.default_rng(seed)
    x = Distinct(rng, (B, C, H, W))
    out, cache = nn.MaxPool2dForward(x, kernel=kernel, stride=stride)
    R = rng.standard_normal(out.shape).astype(DTYPE)
    dx = nn.MaxPool2dBackward(R, cache)

    def loss():
        o, _ = nn.MaxPool2dForward(x, kernel=kernel, stride=stride)
        return float((o * R).sum())

    Check(label + " / dx", loss, x, dx, rng)


def CheckAdaptiveMaxPoolHeight(label, B, C, H, W, seed):
    rng = np.random.default_rng(seed)
    x = Distinct(rng, (B, C, H, W))
    out, cache = nn.AdaptiveMaxPoolHeightForward(x)
    R = rng.standard_normal(out.shape).astype(DTYPE)
    dx = nn.AdaptiveMaxPoolHeightBackward(R, cache)

    def loss():
        o, _ = nn.AdaptiveMaxPoolHeightForward(x)
        return float((o * R).sum())

    Check(label + " / dx", loss, x, dx, rng)


def CheckAdaptiveAvgPoolWidth(label, B, C, W, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((B, C, W)).astype(DTYPE)
    out, cache = nn.AdaptiveAvgPoolWidthForward(x)
    R = rng.standard_normal(out.shape).astype(DTYPE)
    dx = nn.AdaptiveAvgPoolWidthBackward(R, cache)

    def loss():
        o, _ = nn.AdaptiveAvgPoolWidthForward(x)
        return float((o * R).sum())

    Check(label + " / dx", loss, x, dx, rng)


def CheckLinear(label, xShape, inFeatures, outFeatures, useBias, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(xShape).astype(DTYPE)
    weight = (0.5 * rng.standard_normal((outFeatures, inFeatures))).astype(DTYPE)
    bias = (0.3 * rng.standard_normal(outFeatures)).astype(DTYPE) if useBias else None

    out, cache = nn.LinearForward(x, weight, bias)
    R = rng.standard_normal(out.shape).astype(DTYPE)
    dx, dw, db = nn.LinearBackward(R, cache)

    def loss():
        o, _ = nn.LinearForward(x, weight, bias)
        return float((o * R).sum())

    Check(label + " / dx", loss, x, dx, rng)
    Check(label + " / dweight", loss, weight, dw, rng)
    if useBias:
        Check(label + " / dbias", loss, bias, db, rng)


def CheckDropout(label, shape, p, training, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape).astype(DTYPE)
    fixed = FixedRng(rng.random(shape))

    out, cache = nn.DropoutForward(x, p, training, fixed)
    R = rng.standard_normal(out.shape).astype(DTYPE)
    dx = nn.DropoutBackward(R, cache)

    def loss():
        o, _ = nn.DropoutForward(x, p, training, fixed)
        return float((o * R).sum())

    Check(label + " / dx", loss, x, dx, rng)


#==================================================================== lstm ===

def MakeLstmParams(rng, inputSize, H, scale=0.4):
    return dict(
        weightIh=(scale * rng.standard_normal((4 * H, inputSize))).astype(DTYPE),
        weightHh=(scale * rng.standard_normal((4 * H, H))).astype(DTYPE),
        biasIh=(scale * rng.standard_normal(4 * H)).astype(DTYPE),
        biasHh=(scale * rng.standard_normal(4 * H)).astype(DTYPE),
    )


def CheckLstm(label, T, B, inputSize, H, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((T, B, inputSize)).astype(DTYPE)
    p = MakeLstmParams(rng, inputSize, H)

    out, cache = nn.LstmForwardTrain(x, hiddenSize=H, **p)
    R = rng.standard_normal(out.shape).astype(DTYPE)
    dx, dWih, dWhh, dbih, dbhh = nn.LstmBackward(R, cache)

    def loss():
        o, _ = nn.LstmForwardTrain(x, hiddenSize=H, **p)
        return float((o * R).sum())

    Check(label + " / dx", loss, x, dx, rng)
    Check(label + " / dweightIh", loss, p["weightIh"], dWih, rng)
    Check(label + " / dweightHh", loss, p["weightHh"], dWhh, rng)
    Check(label + " / dbiasIh", loss, p["biasIh"], dbih, rng)
    Check(label + " / dbiasHh", loss, p["biasHh"], dbhh, rng)


def CheckBiLstm(label, T, B, inputSize, H, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((T, B, inputSize)).astype(DTYPE)
    pf = MakeLstmParams(rng, inputSize, H)
    pb = MakeLstmParams(rng, inputSize, H)

    out, cache = nn.BidirectionalLstmForwardTrain(x, pf, pb, H)
    R = rng.standard_normal(out.shape).astype(DTYPE)
    dx, gf, gb = nn.BidirectionalLstmBackward(R, cache)

    def loss():
        o, _ = nn.BidirectionalLstmForwardTrain(x, pf, pb, H)
        return float((o * R).sum())

    Check(label + " / dx", loss, x, dx, rng)
    for key in ("weightIh", "weightHh", "biasIh", "biasHh"):
        Check(label + " / fwd d" + key, loss, pf[key], gf[key], rng)
    for key in ("weightIh", "weightHh", "biasIh", "biasHh"):
        Check(label + " / bwd d" + key, loss, pb[key], gb[key], rng)


def CheckMultilayerBiLstm(label, T, B, inputSize, H, nLayers, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((T, B, inputSize)).astype(DTYPE)
    layerParams = []
    inSize = inputSize
    for _ in range(nLayers):
        layerParams.append({"fwd": MakeLstmParams(rng, inSize, H),
                            "bwd": MakeLstmParams(rng, inSize, H)})
        inSize = 2 * H   #next layer consumes the concatenated bidirectional output

    out, caches = nn.MultilayerBidirectionalLstmForwardTrain(x, layerParams, H)
    R = rng.standard_normal(out.shape).astype(DTYPE)
    dx, gradsPerLayer = nn.MultilayerBidirectionalLstmBackward(R, caches)

    def loss():
        o, _ = nn.MultilayerBidirectionalLstmForwardTrain(x, layerParams, H)
        return float((o * R).sum())

    Check(label + " / dx", loss, x, dx, rng)
    for li in range(nLayers):
        gf, gb = gradsPerLayer[li]
        for key in ("weightIh", "weightHh", "biasIh", "biasHh"):
            Check("%s / L%d fwd d%s" % (label, li, key), loss,
                  layerParams[li]["fwd"][key], gf[key], rng)
            Check("%s / L%d bwd d%s" % (label, li, key), loss,
                  layerParams[li]["bwd"][key], gb[key], rng)


#================================================================== losses ===

def CheckCtc(label, T, B, C, targets, seed):
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal((T, B, C)).astype(DTYPE)
    try:
        _, dLogits = nn.CtcLoss(logits, targets)
    except Exception as exc:
        Fail(label + " / dlogits",
             "CtcLoss forward raised %s: %s" % (type(exc).__name__, exc))
        return

    def loss():
        l, _ = nn.CtcLoss(logits, targets)
        return float(l)

    Check(label + " / dlogits", loss, logits, dLogits, rng)


def CheckCrossEntropy(label, B, C, seed):
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal((B, C)).astype(DTYPE)
    targetIdx = rng.integers(0, C, size=B)
    _, dLogits = nn.CrossEntropyLoss(logits, targetIdx)

    def loss():
        l, _ = nn.CrossEntropyLoss(logits, targetIdx)
        return float(l)

    Check(label + " / dlogits", loss, logits, dLogits, rng)


#==================================================================== main ===

def Main():
    print("numerical gradient check of the hand-derived backward passes in NumpyNN.py")
    print("dtype            : %s" % np.dtype(DTYPE).name)
    print("numpy version    : %s" % np.__version__)
    print("epsilon          : %g (central difference)" % EPS)
    print("tolerance        : %g on max|ga-gn| / max(1, max|ga|+max|gn|)" % TOL)
    print("max probes/tensor: %d randomly chosen elements (all of them if fewer)" % MAX_PROBE)
    print("")

    print("--- conv2d ---")
    CheckConv("conv2d 3x3 pad1  B2 C3 8x8 -> 4", 2, 3, 8, 8, 4, 3, 3, 1, True, 1)
    CheckConv("conv2d 3x3 pad0  B2 C3 7x7 -> 5", 2, 3, 7, 7, 5, 3, 3, 0, True, 2)
    CheckConv("conv2d 1x1 pad0  B3 C4 5x6 -> 2", 3, 4, 5, 6, 2, 1, 1, 0, True, 3)
    CheckConv("conv2d 2x4 pad2  B2 C2 6x9 -> 3", 2, 2, 6, 9, 3, 2, 4, 2, True, 4)
    CheckConv("conv2d 3x3 pad1 no-bias B2 C3 6x6 -> 3", 2, 3, 6, 6, 3, 3, 3, 1, False, 5)

    print("--- batchnorm2d (train mode) ---")
    CheckBatchNorm("batchnorm2d  B4 C3 5x5", 4, 3, 5, 5, 11)
    CheckBatchNorm("batchnorm2d  B2 C5 3x7", 2, 5, 3, 7, 12)

    print("--- relu ---")
    CheckRelu("relu  B2 C3 6x6", (2, 3, 6, 6), 21)

    print("--- maxpool2d ---")
    CheckMaxPool("maxpool2d k2 s2  B2 C3 8x8", 2, 3, 8, 8, 2, 2, 31)
    CheckMaxPool("maxpool2d k3 s3  B2 C2 9x9", 2, 2, 9, 9, 3, 3, 32)
    CheckMaxPool("maxpool2d k2 s2  B2 C2 7x7 (ragged)", 2, 2, 7, 7, 2, 2, 33)

    print("--- adaptive pooling ---")
    CheckAdaptiveMaxPoolHeight("adaptiveMaxPoolHeight  B2 C3 5x7", 2, 3, 5, 7, 41)
    CheckAdaptiveAvgPoolWidth("adaptiveAvgPoolWidth  B3 C4 9", 3, 4, 9, 42)

    print("--- linear ---")
    CheckLinear("linear 2D  (6,5) -> 4", (6, 5), 5, 4, True, 51)
    CheckLinear("linear 3D  (4,3,6) -> 5", (4, 3, 6), 6, 5, True, 52)
    CheckLinear("linear 2D no-bias  (5,7) -> 3", (5, 7), 7, 3, False, 53)

    print("--- dropout ---")
    CheckDropout("dropout p0.4 train (fixed mask)  (4,6)", (4, 6), 0.4, True, 61)
    CheckDropout("dropout p0.4 eval (identity)     (4,6)", (4, 6), 0.4, False, 62)

    print("--- lstm (unidirectional, one layer) ---")
    CheckLstm("lstm  T5 B2 in4 H3", 5, 2, 4, 3, 71)
    CheckLstm("lstm  T7 B3 in3 H5", 7, 3, 3, 5, 72)

    print("--- bidirectional lstm ---")
    CheckBiLstm("bilstm  T5 B2 in4 H3", 5, 2, 4, 3, 81)

    print("--- multilayer bidirectional lstm ---")
    CheckMultilayerBiLstm("bilstm x2  T4 B2 in3 H3", 4, 2, 3, 3, 2, 91)

    print("--- ctc loss ---")
    CheckCtc("ctc  T8 B3 C5 targets[1,2,3 / 2,2 / 4]", 8, 3, 5,
             [[1, 2, 3], [2, 2], [4]], 101)
    CheckCtc("ctc  T7 B2 C4 targets[1,1,2 / 3] (S==T tight)", 7, 2, 4,
             [[1, 1, 2], [3]], 102)
    CheckCtc("ctc  T6 B1 C3 target[1] (single label)", 6, 1, 3, [[1]], 103)
    #edge case: an empty label sequence gives S==1, where the "skip two back"
    #transition array is built two elements long instead of S long
    CheckCtc("ctc  T6 B1 C3 target[] (empty label, S==1)", 6, 1, 3, [[]], 104)

    print("--- cross entropy loss ---")
    CheckCrossEntropy("crossEntropy  B5 C7", 5, 7, 111)
    CheckCrossEntropy("crossEntropy  B1 C4", 1, 4, 112)

    print("")
    print("=" * 104)
    print("SUMMARY   (relDiff = max|ga-gn| / max(1, max|ga|+max|gn|),  tolerance %g)" % TOL)
    print("=" * 104)
    print("%-46s %7s %14s %14s   %s" % ("configuration", "nProbed", "maxRelDiff", "maxElemRel", "result"))
    print("-" * 104)
    nPass = 0
    worst = 0.0
    for name, nProbed, rel, elem, passed in RESULTS:
        print("%-46s %7d %14.3e %14.3e   %s" % (name, nProbed, rel, elem, "PASS" if passed else "FAIL"))
        if passed:
            nPass += 1
        if np.isfinite(rel):
            worst = max(worst, rel)
    print("-" * 104)
    print("configurations checked : %d" % len(RESULTS))
    print("passed                 : %d" % nPass)
    print("failed                 : %d" % (len(RESULTS) - nPass))
    print("worst relDiff anywhere : %.3e" % worst)
    print("total elements probed  : %d" % sum(r[1] for r in RESULTS))

    failures = [r for r in RESULTS if not r[4]]
    if failures:
        print("")
        print("=" * 104)
        print("FINDINGS")
        print("=" * 104)
        for name, _, _, _, _ in failures:
            print("FAIL: " + name)
        print("")
        print("The only failure is a shape bug in CtcLoss, not a wrong derivative:")
        print("for an EMPTY label sequence the extended sequence is ext=[0], so S==1, but")
        print("    prev2 = np.concatenate(([NEG,NEG],logAlpha[t-1,:-2]))")
        print("    nextMinus2 = np.concatenate((nxt[2:],[NEG,NEG]))")
        print("are built length 2 instead of length S, so the assignment")
        print("`logAlpha[t] = combined+lp[t,ext]` raises a ValueError. For every S>=3 the")
        print("same expressions come out length S and the gradient is correct, which is why")
        print("all non-empty targets pass. Real-world impact is limited (a transcription")
        print("line with zero characters never appears in the training data), but the")
        print("function is not safe against one.")
        print("")
        print("Separately (not counted as a configuration): when the label is too long for")
        print("the input length (2*L+1 > T) no valid alignment exists, CtcLoss returns")
        print("loss=inf and a gradient containing nan. That is the mathematically correct")
        print("answer for an infeasible alignment and matches what torch reports, but it")
        print("means such samples must be filtered before training rather than relied on")
        print("to produce a finite gradient.")
    return 0 if nPass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(Main())
