def bounds(m):
    y, x = np.where(m)
    return (int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1))

def clean(a):
    m = SEG.BinarizeLine(a, stripRules=True, closeGaps=False)
    band = SEG.CoreBand(m)
    if band:
        t, b = band
        xh = b - t
        nn, lab, stats, cents = cv2.connectedComponentsWithStats(m.astype('uint8'), 8)
        for i in range(1, nn):
            x, y, w, h, ar = stats[i]
            cy = cents[i, 1]
            if cy < t - 0.75 * xh or cy > b + 0.75 * xh or (h > 3 * xh and w < 0.12 * xh):
                m[lab == i] = False
    return m

def word_cuts(m, tokens):
    x0, y0, x1, y1 = bounds(m)
    n = len(tokens)
    if n == 1:
        return [x0, x1]
    p = m.any(0)
    v = np.diff(np.r_[True, p, True].astype(int))
    gs = np.where(v == -1)[0]
    ge = np.where(v == 1)[0]
    cand = {x0: 0.0, x1: 0.0}
    for s, e in zip(gs, ge):
        if s > 0 and e < len(p):
            cand[(int(s) + int(e)) // 2] = -2 * np.log1p((e - s) / max(3, (y1 - y0) / 2))
    prof = gaussian_filter1d(m.sum(0).astype(float), 1.0)
    for x in range(x0 + 2, x1 - 2):
        if prof[x] <= prof[x - 1] and prof[x] < prof[x + 1]:
            cand.setdefault(x, 5 * prof[x] / max(1, y1 - y0))
    xs = np.array(sorted((q for q in cand if x0 <= q <= x1)))
    cost = np.array([cand[q] for q in xs])
    N = len(xs)
    weight = lambda t: sum((0.45 if c in 'ilI1.,:;' else 0.65 if c in 'ftjr' else 1.35 if c in 'mwMW' else 1.0 for c in t))
    lens = np.array([weight(t) for t in tokens])
    unit = (x1 - x0) / (lens.sum() + 1.3 * (n - 1))
    want = (lens + 1.3) * unit
    want[[0, -1]] -= 0.65 * unit
    dp = np.full((n + 1, N), np.inf)
    pre = np.full((n + 1, N), -1, int)
    dp[0, 0] = 0
    for j in range(1, n + 1):
        for q in range(1, N):
            wid = xs[q] - xs[:q]
            loss = dp[j - 1, :q] + 0.5 * ((wid - want[j - 1]) / (0.35 * want[j - 1] + 0.6 * unit)) ** 2 + cost[q]
            loss[wid < 0.2 * unit] = np.inf
            k = int(loss.argmin())
            dp[j, q] = loss[k]
            pre[j, q] = k
    if not np.isfinite(dp[n, -1]):
        return None
    at = N - 1
    cuts = [int(xs[at])]
    for j in range(n, 0, -1):
        at = pre[j, at]
        cuts.append(int(xs[at]))
    return cuts[::-1]

def decode(raw):
    try:
        return np.load(io.BytesIO(zlib.decompress(raw)), allow_pickle=False)
    except Exception:
        return np.asarray(Image.open(io.BytesIO(raw)).convert('L'))
