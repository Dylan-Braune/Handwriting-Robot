"""
qtp1b_overlay.py -- QTP1(b): pixel-domain overlay test.

Design (text-controlled):
  * 3 real held-out lines per writer (held out of the STYLE PROFILE; see qtp_lines.py), each
    with its transcription T.  Same-text pairs of two DIFFERENT real writers are rare (only the
    c03 passage of writers 150-153), so the control is built on the synthesis side instead:
  * T is synthesised (production SynthesizeJointBestOf, nTries=20, seed 7) in EVERY writer's
    profile W.  For a real line (writer A, text T):
        matched pair    : real(A,T) vs synth(A,T)
        mismatched pairs: real(A,T) vs synth(W,T), W != A          (same text, other style)
    so differences can only come from style (and from the unavoidable synthesis noise).
  * Secondary, literal version (real vs synthesised): where a DIFFERENT real writer B wrote
    exactly the same text T (normalised), real(B,T) vs synth(A,T), A != B, is compared with
    real(A,T) vs synth(A,T).  Pair counts are reported.

Normalisation (both images): grey -> ProfileIO.BinarizeLine ink -> CoreBand x-height -> grey
resized to x-height 28 px -> binarise again -> Zhang-Suen skeleton (pen-width independent) ->
pixel coordinates / x-height (unit = x-heights) -> centroid translated to the origin.
Two variants:
  'xh'       : same x-height, centroids aligned (the specified normalisation)
  'xh+width' : additionally the horizontal coordinates of the synthesised skeleton are scaled so
               that its ink bounding-box width equals the real line's (removes global spacing /
               text-length drift so that letters line up; shape and slant differences remain)
Distances (symmetric, in x-height units): chamfer = 0.5 (mean d(A->B) + mean d(B->A));
modified Hausdorff (Dubuisson & Jain 1994) = max(mean d(A->B), mean d(B->A)); both on skeleton
point sets via a k-d tree.
Effect measures: mean matched vs mismatched distance, rank-1 fraction (matched synth writer is the
nearest of the 10), mean rank, AUC = P(matched < mismatched) over all (matched, mismatched)
pairs sharing the real line, Cohen's d, bootstrap 95% CI over the real lines.
"""
import io
import os
import pickle
import sys

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import qtp_common as C
import qtp_features as QF
import qtp_lines as L
import ProfileIO as PIO

TARGET = QF.TARGET_XH


def skeleton_points(gray):
    """-> (N,2) float array of skeleton pixel coords (x,y) in x-height units, y DOWN; xh_px."""
    g = np.asarray(gray)
    if g.ndim == 3:
        g = g[..., 0]
    ink, band, xh = QF._band(g)
    if ink is None or xh < 4:
        return None
    if abs(TARGET / xh - 1.0) > 0.05:
        g = QF._resize_gray(g, TARGET / xh)
        ink, band, xh = QF._band(g)
        if ink is None:
            return None
    sk = PIO.Skeletonize(ink)
    ys, xs = np.nonzero(sk)
    if len(xs) < 20:
        return None
    pts = np.stack([xs, ys], axis=1).astype(float) / xh
    return pts


def align(a, b, mode):
    """a: real points, b: synthesised points. Returns centred copies (b optionally width-scaled)."""
    a = a - a.mean(axis=0)
    b = b.copy()
    if mode == "xh+width":
        wa = a[:, 0].max() - a[:, 0].min()
        wb = b[:, 0].max() - b[:, 0].min()
        b[:, 0] = (b[:, 0] - b[:, 0].mean()) * (wa / wb)
    else:
        b = b - b.mean(axis=0)
    return a, b


def dists(a, b):
    ta, tb = cKDTree(a), cKDTree(b)
    d_ab = tb.query(a)[0].mean()
    d_ba = ta.query(b)[0].mean()
    return 0.5 * (d_ab + d_ba), max(d_ab, d_ba)


def auc_matched(D, tid_writer, writers):
    """D[(t, w)] dict -> list of (matched, mismatched list) per text."""
    out = []
    for t, a in tid_writer.items():
        if a not in writers:
            continue
        m = D[t][writers.index(a)]
        mm = [D[t][j] for j, w in enumerate(writers) if w != a]
        out.append((m, mm))
    return out


def summarise(D, tw, writers, rng):
    """D: array (n_texts, 10) of distances; tw: list of real-writer index per text."""
    n = len(tw)
    matched = np.array([D[i, tw[i]] for i in range(n)])
    mism = np.array([np.mean([D[i, j] for j in range(len(writers)) if j != tw[i]]) for i in range(n)])
    ranks = np.array([1 + int(np.sum(D[i] < D[i, tw[i]])) for i in range(n)])
    wins = np.array([np.mean([(D[i, tw[i]] < D[i, j]) + 0.5 * (D[i, tw[i]] == D[i, j])
                              for j in range(len(writers)) if j != tw[i]]) for i in range(n)])
    allm = np.concatenate([[D[i, j] for j in range(len(writers)) if j != tw[i]] for i in range(n)])
    pooled_sd = np.sqrt(0.5 * (matched.var(ddof=1) + allm.var(ddof=1)))
    d = (allm.mean() - matched.mean()) / pooled_sd
    out = dict(n_lines=int(n), matched_mean=float(matched.mean()), mismatched_mean=float(allm.mean()),
               ratio_matched_over_mismatched=float(matched.mean() / allm.mean()),
               rank1_fraction=float(np.mean(ranks == 1)), mean_rank=float(ranks.mean()),
               top3_fraction=float(np.mean(ranks <= 3)),
               auc=float(wins.mean()), cohens_d=float(d), chance_rank1=1 / len(writers))
    bs_r1, bs_auc = [], []
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        bs_r1.append(np.mean(ranks[idx] == 1)); bs_auc.append(wins[idx].mean())
    out["rank1_ci95"] = [float(np.percentile(bs_r1, 2.5)), float(np.percentile(bs_r1, 97.5))]
    out["auc_ci95"] = [float(np.percentile(bs_auc, 2.5)), float(np.percentile(bs_auc, 97.5))]
    out["per_writer_rank1"] = {}
    for w in set(tw):
        sel = [i for i in range(n) if tw[i] == w]
        out["per_writer_rank1"][writers[w]] = dict(n=len(sel), rank1=float(np.mean(ranks[sel] == 1)),
                                                   mean_rank=float(ranks[sel].mean()))
    out["ranks"] = ranks.tolist()
    return out


def main():
    jobs = C.load_json("qtp1_synth_jobs.json")
    texts = jobs["overlay_texts"]
    lines = L.load_real_lines()
    real_by_key = {(l["writer"], l["page"], l["line_idx"]): l for l in lines}
    W = C.ALL_AUTHORS
    syn_dir = C.DATA_OUT / "synth"

    # which texts have a complete set of 10 syntheses
    pts_real, pts_syn, used = {}, {}, []
    for t in texts:
        tid = t["tid"]
        files = [syn_dir / f"B_t{tid:02d}_{w}.pkl" for w in W]
        if not all(f.exists() for f in files):
            continue
        real = real_by_key[(t["real_writer"], t["page"], t["line_idx"])]
        pr = skeleton_points(real["gray"])
        if pr is None:
            continue
        ps = []
        for f in files:
            with open(f, "rb") as fh:
                rec = pickle.load(fh)
            ps.append(skeleton_points(np.array(Image.open(io.BytesIO(rec["png"])).convert("L"))))
        if any(p is None for p in ps):
            continue
        pts_real[tid] = pr
        pts_syn[tid] = ps
        used.append(t)
    print("texts with complete syntheses:", len(used), "of", len(texts))

    out = dict(n_texts_used=len(used), n_texts_planned=len(texts), nTries=jobs["nTries"],
               texts_used=[dict(tid=t["tid"], real_writer=t["real_writer"], text=t["text"]) for t in used])
    rng = np.random.default_rng(0)
    tw = [W.index(t["real_writer"]) for t in used]
    mats = {}
    for mode in ("xh", "xh+width"):
        Dc = np.zeros((len(used), len(W))); Dh = np.zeros_like(Dc)
        for i, t in enumerate(used):
            for j in range(len(W)):
                a, b = align(pts_real[t["tid"]], pts_syn[t["tid"]][j], mode)
                Dc[i, j], Dh[i, j] = dists(a, b)
        mats[mode] = (Dc, Dh)
        out[mode] = dict(chamfer=summarise(Dc, tw, W, rng), modified_hausdorff=summarise(Dh, tw, W, rng),
                         chamfer_matrix_by_real_writer={
                             w: [float(np.mean([Dc[i, j] for i in range(len(used)) if tw[i] == W.index(w)])) if any(tw[i] == W.index(w) for i in range(len(used))) else None
                                 for j in range(len(W))] for w in W},
                         raw_chamfer=Dc.tolist(), raw_mhd=Dh.tolist())

    # ---- secondary: real(B,T) from a different real writer with identical text ----
    norm = L.norm_text
    sec = []
    for t in used:
        key = norm(t["text"])
        A = t["real_writer"]
        for l in lines:
            if l["writer"] != A and norm(l["text"]) == key:
                sec.append((t["tid"], A, l))
    pairs = []
    for tid, A, lB in sec:
        pb = skeleton_points(lB["gray"])
        if pb is None:
            continue
        ia = W.index(A)
        row = {}
        for mode in ("xh", "xh+width"):
            a_real, b_syn = align(pts_real[tid], pts_syn[tid][ia], mode)
            m_cham, m_mhd = dists(a_real, b_syn)               # real(A) vs synth(A)  matched
            a_B, b_syn2 = align(pb, pts_syn[tid][ia], mode)
            x_cham, x_mhd = dists(a_B, b_syn2)                 # real(B) vs synth(A)  mismatched writer
            row[mode] = dict(matched_chamfer=float(m_cham), cross_chamfer=float(x_cham),
                             matched_mhd=float(m_mhd), cross_mhd=float(x_mhd))
        pairs.append(dict(tid=tid, synth_writer=A, real_writer=lB["writer"], text=lB["text"].strip(), **row))
    out["secondary_real_B_vs_synth_A"] = dict(n_pairs=len(pairs), pairs=pairs)
    if pairs:
        for mode in ("xh", "xh+width"):
            mm = np.array([p[mode]["matched_chamfer"] for p in pairs]); xx = np.array([p[mode]["cross_chamfer"] for p in pairs])
            out["secondary_real_B_vs_synth_A"][mode] = dict(
                matched_mean=float(mm.mean()), cross_mean=float(xx.mean()),
                frac_matched_smaller=float(np.mean(mm < xx)))
    C.save_json(out, "qtp1b_results.json")

    for mode in ("xh", "xh+width"):
        for k in ("chamfer", "modified_hausdorff"):
            s = out[mode][k]
            print(f"{mode:9s} {k:18s} matched {s['matched_mean']:.3f} mismatched {s['mismatched_mean']:.3f} "
                  f"rank1 {s['rank1_fraction']:.2f} {s['rank1_ci95']} AUC {s['auc']:.3f} {s['auc_ci95']} d {s['cohens_d']:.2f}")
    print("secondary pairs:", len(pairs))

    # ---------------- figures ----------------
    plt = C.mpl_style()
    from matplotlib.lines import Line2D
    # distance matrix heatmap (chamfer, xh+width and xh)
    fig, axs = plt.subplots(1, 2, figsize=(16 * C.CM, 7.5 * C.CM))
    for ax, mode in zip(axs, ("xh", "xh+width")):
        Dc = mats[mode][0]
        M = np.full((len(W), len(W)), np.nan)
        for a_i, a in enumerate(W):
            sel = [i for i in range(len(used)) if tw[i] == a_i]
            if sel:
                M[a_i] = Dc[sel].mean(axis=0)
        im = ax.imshow(M, cmap="viridis_r")
        ax.set_xticks(range(len(W))); ax.set_xticklabels(W, rotation=90)
        ax.set_yticks(range(len(W))); ax.set_yticklabels(W)
        ax.set_xlabel("synthesised in profile of"); ax.set_ylabel("real line written by")
        ax.set_title(f"chamfer [x-heights], {mode}", fontsize=9); ax.grid(False)
        for a_i in range(len(W)):
            ax.add_patch(plt.Rectangle((a_i - 0.5, a_i - 0.5), 1, 1, fill=False, ec="red", lw=1))
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("QTP1b: mean distance, real line vs synthesis of the same text (red box = matched writer)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    C.savefig(fig, "qtp1_distance_matrix.png"); plt.close(fig)

    # matched vs mismatched
    fig, axs = plt.subplots(1, 2, figsize=(16 * C.CM, 7 * C.CM))
    for ax, mode in zip(axs, ("xh", "xh+width")):
        Dc = mats[mode][0]
        matched = [Dc[i, tw[i]] for i in range(len(used))]
        mism = [Dc[i, j] for i in range(len(used)) for j in range(len(W)) if j != tw[i]]
        bp = ax.boxplot([matched, mism], tick_labels=["matched\nwriter", "mismatched\nwriter"], widths=0.5,
                        patch_artist=True, showfliers=True)
        for patch, col in zip(bp["boxes"], (C.PALETTE[2], C.PALETTE[1])):
            patch.set_facecolor(col); patch.set_alpha(0.5)
        s = out[mode]["chamfer"]
        ax.set_title(f"{mode}: rank-1 {s['rank1_fraction']*100:.0f}% (chance 10%), AUC {s['auc']:.2f}", fontsize=9)
        ax.set_ylabel("chamfer distance [x-heights]")
    fig.suptitle(f"QTP1b: matched vs mismatched synthesised writer ({len(used)} real held-out lines)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    C.savefig(fig, "qtp1_matched_vs_mismatched.png"); plt.close(fig)

    # overlay examples: fixed rule -- rank-0 text of writers 153, 551, dylan, yeukita
    show = []
    for w in ("153", "551", "dylan", "yeukita"):
        cand = [t for t in used if t["real_writer"] == w]
        if cand:
            show.append(cand[0])
    if show:
        fig, axs = plt.subplots(len(show), 2, figsize=(16 * C.CM, 3.6 * C.CM * len(show)))
        axs = np.atleast_2d(axs)
        for r, t in enumerate(show):
            tid = t["tid"]; ia = W.index(t["real_writer"])
            # mismatched writer = the synthesised writer with the largest distance (worst) is cherry-picking;
            # use the fixed rule: next writer in list order
            jm = (ia + 1) % len(W)
            for c, (jj, ttl) in enumerate(((ia, f"matched: real vs synth {W[ia]}"),
                                           (jm, f"mismatched: real {W[ia]} vs synth {W[jm]}"))):
                a, b = align(pts_real[tid], pts_syn[tid][jj], "xh+width")
                ax = axs[r, c]
                ax.scatter(a[:, 0], -a[:, 1], s=1.5, color="#d62728", label="real")
                ax.scatter(b[:, 0], -b[:, 1], s=1.5, color="#1f77b4", label="synthesised", alpha=0.8)
                ax.set_aspect("equal"); ax.set_title(ttl + f"\nchamfer {mats['xh+width'][0][used.index(t), jj]:.2f} xh", fontsize=8)
                ax.set_xlabel("x [x-heights]") if r == len(show) - 1 else None
                ax.grid(False)
        handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728", label="real skeleton"),
                   Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", label="synthesised skeleton")]
        fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9)
        fig.suptitle("QTP1b: overlay examples (centroid-aligned, width-normalised; fixed selection rule, not best cases)", fontsize=9)
        fig.tight_layout(rect=(0, 0.04, 1, 0.97))
        C.savefig(fig, "qtp1_overlay_examples.png"); plt.close(fig)
        # also raw two-colour PNG overlays at full resolution for the report
        for t in show:
            tid = t["tid"]; ia = W.index(t["real_writer"])
            a, b = align(pts_real[tid], pts_syn[tid][ia], "xh+width")
            sc = 12
            allp = np.vstack([a, b])
            x0, y0 = allp.min(axis=0)
            wpx = int((allp[:, 0].max() - x0) * sc) + 8
            hpx = int((allp[:, 1].max() - y0) * sc) + 8
            canvas = np.full((hpx, wpx, 3), 255, np.uint8)
            for p in b:
                canvas[int((p[1] - y0) * sc) + 4, int((p[0] - x0) * sc) + 4] = (30, 90, 220)
            for p in a:
                canvas[int((p[1] - y0) * sc) + 4, int((p[0] - x0) * sc) + 4] = (220, 30, 30)
            import cv2
            canvas = cv2.dilate(255 - canvas, np.ones((2, 2), np.uint8))
            Image.fromarray(255 - canvas).save(C.DATA_OUT / f"qtp1b_overlay_{t['real_writer']}_t{tid:02d}.png")


if __name__ == "__main__":
    main()
