"""
qtp1a_analyse.py -- QTP1(a): stylistic correlation by hand-crafted (non-neural) features.

Inputs : qtp_data/qtp1a_real_features.json   (every real line)
         qtp_data/qtp1a_synth_features.json  (10 writers x 6 novel sentences, production synthesis)
Outputs: qtp_data/qtp1a_results.json, Figures/qtp1_feature_scatter.png,
         Figures/qtp1_style_confusion.png

Definitions (fixed before looking at results):
  * per-writer mean of each feature over the writer's lines (nan-mean);
      real_held : real lines NOT used to build the style profile (IAM held-out page; personal
                  lines outside the profile-building split) -> clean but few lines (8-32/writer)
      real_all  : all real lines of the writer (also those the profile was built from)
  * per-feature Pearson r across the 10 writers, real mean vs synthesised mean
  * COMBINED stylistic correlation, two variants
      'within'  : each feature z-scored across the 10 writers SEPARATELY for real and for
                  synthesised means (removes any constant offset/scale between domains, e.g. the
                  deliberate uniform pen), then Pearson r of the pooled 10 x F vectors
      'absolute': both real and synthesised means z-scored with the REAL across-writer mean/std
                  (so systematic bias between real and synthesised features lowers r)
    each for ALL features and for the SHAPE-ONLY subset (without stroke width and ink density,
    the pen-dependent features that the single-pen gantry cannot reproduce)
  * non-neural writer identification: nearest real-writer centroid (Euclidean distance on features
    z-scored with the real across-writer mean/std; NaN features of a sample are skipped) for each
    of the 60 synthesised lines; headline = real_all centroids, ALL features
"""
import sys
import os
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qtp_common as C
import qtp_features as QF

FEATS = QF.FEATURES
W = C.ALL_AUTHORS


def writer_means(rows, writers=W):
    M = np.full((len(writers), len(FEATS)), np.nan)
    N = []
    for i, w in enumerate(writers):
        rr = [r for r in rows if r["writer"] == w]
        N.append(len(rr))
        for j, f in enumerate(FEATS):
            v = np.array([r.get(f, np.nan) if r.get(f) is not None else np.nan for r in rr], float)
            if np.isfinite(v).any():
                M[i, j] = np.nanmean(v)
    return M, N


def zs(M, mu=None, sd=None):
    mu = np.nanmean(M, axis=0) if mu is None else mu
    sd = np.nanstd(M, axis=0, ddof=0) if sd is None else sd
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (M - mu) / sd, mu, sd


def pooled_r(Mr, Ms, cols, mode):
    R, mur, sdr = zs(Mr[:, cols])
    if mode == "within":
        S_, _, _ = zs(Ms[:, cols])
    else:
        S_, _, _ = zs(Ms[:, cols], mur, sdr)
    a, b = R.ravel(), S_.ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    r = float(np.corrcoef(a[ok], b[ok])[0, 1])
    # bootstrap over writers (resample the 10 writers with replacement)
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(2000):
        idx = rng.integers(0, len(W), len(W))
        if mode == "within":
            Rb, _, _ = zs(Mr[idx][:, cols]); Sb, _, _ = zs(Ms[idx][:, cols])
        else:
            Rb, m1, s1 = zs(Mr[idx][:, cols]); Sb, _, _ = zs(Ms[idx][:, cols], m1, s1)
        x, y = Rb.ravel(), Sb.ravel()
        k = np.isfinite(x) & np.isfinite(y)
        if k.sum() > 3 and np.std(x[k]) > 0 and np.std(y[k]) > 0:
            bs.append(np.corrcoef(x[k], y[k])[0, 1])
    return r, [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def nn_identify(Mr, synth_rows, cols, writers=W):
    _, mu, sd = zs(Mr[:, cols])
    cent = (Mr[:, cols] - mu) / np.where(sd < 1e-12, 1.0, sd)
    conf = np.zeros((len(writers), len(writers)), int)
    correct = 0
    preds = []
    for r in synth_rows:
        x = np.array([r.get(FEATS[j], np.nan) if r.get(FEATS[j]) is not None else np.nan for j in cols], float)
        x = (x - mu) / np.where(sd < 1e-12, 1.0, sd)
        ok = np.isfinite(x)
        d = np.array([np.sqrt(np.nanmean(np.where(ok & np.isfinite(c), (x - c) ** 2, np.nan))) for c in cent])
        p = int(np.nanargmin(d))
        t = writers.index(r["writer"])
        conf[t, p] += 1
        correct += (p == t)
        preds.append((r["writer"], writers[p]))
    return correct / len(synth_rows), conf, preds


def main():
    real = C.load_json("qtp1a_real_features.json")["rows"]
    synth = C.load_json("qtp1a_synth_features.json")["A"]
    synth = [r for r in synth if r.get("slant_deg") is not None]
    print("real lines", len(real), "synth lines", len(synth))
    real_held = [r for r in real if r["heldout_profile"]]
    Mall, Nall = writer_means(real)
    Mheld, Nheld = writer_means(real_held)
    Ms, Ns = writer_means(synth)
    allc = list(range(len(FEATS)))
    shape = [j for j, f in enumerate(FEATS) if f in QF.SHAPE_FEATURES]

    res = dict(writers=W, features=FEATS, n_real_all=Nall, n_real_held=Nheld, n_synth=Ns,
               real_all_means=Mall, real_held_means=Mheld, synth_means=Ms)
    # per-feature r
    per = {}
    for refname, Mr in (("real_held", Mheld), ("real_all", Mall)):
        per[refname] = {}
        for j, f in enumerate(FEATS):
            ok = np.isfinite(Mr[:, j]) & np.isfinite(Ms[:, j])
            r, p = stats.pearsonr(Mr[ok, j], Ms[ok, j])
            rho, prho = stats.spearmanr(Mr[ok, j], Ms[ok, j])
            per[refname][f] = dict(pearson_r=float(r), p=float(p), spearman=float(rho),
                                   n_writers=int(ok.sum()),
                                   real_mean=float(np.nanmean(Mr[:, j])), synth_mean=float(np.nanmean(Ms[:, j])),
                                   real_sd_across_writers=float(np.nanstd(Mr[:, j])),
                                   synth_sd_across_writers=float(np.nanstd(Ms[:, j])))
    res["per_feature"] = per
    comb = {}
    for refname, Mr in (("real_held", Mheld), ("real_all", Mall)):
        comb[refname] = {}
        for sname, cols in (("all_features", allc), ("shape_only", shape)):
            comb[refname][sname] = {}
            for mode in ("within", "absolute"):
                r, ci = pooled_r(Mr, Ms, cols, mode)
                comb[refname][sname][mode] = dict(r=r, ci95_bootstrap_over_writers=ci)
    res["combined_correlation"] = comb

    ident = {}
    for refname, Mr in (("real_all", Mall), ("real_held", Mheld)):
        ident[refname] = {}
        for sname, cols in (("all_features", allc), ("shape_only", shape)):
            acc, conf, preds = nn_identify(Mr, synth, cols)
            ident[refname][sname] = dict(accuracy=acc, n=len(synth), confusion=conf.tolist(),
                                         per_writer_acc={w: float(conf[i, i] / max(1, conf[i].sum())) for i, w in enumerate(W)})
    # domain-adapted variant: remove per-feature mean offset between synthesised and real writer means
    Mr = Mall
    off = np.nanmean(Ms, axis=0) - np.nanmean(Mr, axis=0)
    synth_adj = []
    for r in synth:
        r2 = dict(r)
        for j, f in enumerate(FEATS):
            if r2.get(f) is not None and np.isfinite(r2[f]):
                r2[f] = r2[f] - off[j]
        synth_adj.append(r2)
    for sname, cols in (("all_features", allc), ("shape_only", shape)):
        acc, conf, _ = nn_identify(Mr, synth_adj, cols)
        ident.setdefault("real_all_offset_corrected", {})[sname] = dict(accuracy=acc, n=len(synth_adj),
                                                                        confusion=conf.tolist())
    res["nn_writer_id"] = ident

    # ---- noise ceiling: how well do REAL held-out lines agree with the SAME writers' other real lines? ----
    real_train = [r for r in real if not r["heldout_profile"]]
    Mtrain, Ntrain = writer_means(real_train)
    ceil = dict(n_real_train=Ntrain)
    for sname, cols in (("all_features", allc), ("shape_only", shape)):
        ceil[sname] = {m: dict(zip(("r", "ci95_bootstrap_over_writers"), pooled_r(Mtrain, Mheld, cols, m)))
                       for m in ("within", "absolute")}
    ceil["per_feature_r_heldout_vs_train"] = {}
    for j, f in enumerate(FEATS):
        ok = np.isfinite(Mtrain[:, j]) & np.isfinite(Mheld[:, j])
        ceil["per_feature_r_heldout_vs_train"][f] = float(stats.pearsonr(Mtrain[ok, j], Mheld[ok, j])[0])
    # single-line nearest-centroid accuracy of REAL held-out lines (centroids from the writer's non-held-out lines)
    for sname, cols in (("all_features", allc), ("shape_only", shape)):
        acc, conf, _ = nn_identify(Mtrain, real_held, cols)
        ceil.setdefault("real_heldout_single_line_nn_accuracy", {})[sname] = dict(accuracy=acc, n=len(real_held))
    res["noise_ceiling_real_vs_real"] = ceil
    print("noise ceiling (real held-out vs real train means), combined r:",
          {s: {m: round(v["r"], 3) for m, v in ceil[s].items()} for s in ("all_features", "shape_only")},
          "| single-line NN acc real held-out:", {k: round(v["accuracy"], 3) for k, v in ceil["real_heldout_single_line_nn_accuracy"].items()})
    res["chance"] = 1 / len(W)
    C.save_json(res, "qtp1a_results.json")

    # ---------------- print ----------------
    print("\nper-feature Pearson r (real_held vs synth | real_all vs synth)")
    for f in FEATS:
        print(f"  {f:16s} {per['real_held'][f]['pearson_r']:+.2f} (p={per['real_held'][f]['p']:.3f}) | "
              f"{per['real_all'][f]['pearson_r']:+.2f} (p={per['real_all'][f]['p']:.3f})")
    print("\ncombined r:", {k: {s: {m: round(v['r'], 3) for m, v in d2.items()} for s, d2 in d.items()} for k, d in comb.items()})
    print("NN writer-ID:", {k: {s: round(v['accuracy'], 3) for s, v in d.items()} for k, d in ident.items()})

    # ---------------- figures ----------------
    plt = C.mpl_style()
    ncol, nrow = 3, 3
    fig, axs = plt.subplots(nrow, ncol, figsize=(16 * C.CM, 15 * C.CM))
    for k, f in enumerate(FEATS):
        ax = axs[k // ncol, k % ncol]
        j = k
        ok = np.isfinite(Mall[:, j]) & np.isfinite(Ms[:, j])
        ax.scatter(Mall[ok, j], Ms[ok, j], s=22, color=C.PALETTE[0], zorder=3)
        for i, w in enumerate(W):
            if ok[i]:
                ax.annotate(w, (Mall[i, j], Ms[i, j]), fontsize=7, xytext=(2, 2), textcoords="offset points")
        lo = min(np.nanmin(Mall[:, j]), np.nanmin(Ms[:, j])); hi = max(np.nanmax(Mall[:, j]), np.nanmax(Ms[:, j]))
        pad = 0.05 * (hi - lo + 1e-9)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="grey", lw=0.8, ls="--")
        r = per["real_all"][f]["pearson_r"]
        tag = " *" if f in QF.PEN_FEATURES else ""
        ax.set_title(f"{f}{tag}\nr = {r:+.2f}", fontsize=9)
        ax.set_xlabel("real (writer mean)")
        ax.set_ylabel("synthesised (writer mean)")
    fig.suptitle("QTP1a: real vs synthesised hand-crafted features, 10 writers  (* = pen-dependent)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    C.savefig(fig, "qtp1_feature_scatter.png")
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(16 * C.CM, 7.5 * C.CM))
    for ax, sname, title in ((axs[0], "all_features", "all 9 features"), (axs[1], "shape_only", "shape-only (7 features)")):
        conf = np.array(ident["real_all"][sname]["confusion"], float)
        conf = conf / conf.sum(axis=1, keepdims=True)
        im = ax.imshow(conf, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(len(W))); ax.set_xticklabels(W, rotation=90)
        ax.set_yticks(range(len(W))); ax.set_yticklabels(W)
        ax.set_xlabel("nearest real-writer centroid"); ax.set_ylabel("writer synthesised")
        ax.set_title(f"{title}: acc {ident['real_all'][sname]['accuracy']*100:.0f}%", fontsize=9)
        ax.grid(False)
        for a in range(len(W)):
            for b in range(len(W)):
                if conf[a, b] > 0:
                    ax.text(b, a, f"{conf[a,b]*6:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if conf[a, b] > 0.5 else "black")
    fig.suptitle("QTP1a: non-neural nearest-centroid writer identification of synthesised lines (counts of 6)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    C.savefig(fig, "qtp1_style_confusion.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
