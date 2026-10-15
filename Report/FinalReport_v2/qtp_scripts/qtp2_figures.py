"""qtp2_figures.py -- figures for QTP2 from qtp_data/qtp2_results.json."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qtp_common as C


def main():
    d = C.load_json("qtp2_results.json")
    S = d["summary"]
    plt = C.mpl_style()

    # ---- fig 1: accuracies per evidence subset ----
    sets = []
    s = S["server"]
    sets.append(("personal photos\nALL lines\n(mostly train)", s["all_lines"]["text_raw"], s["all_lines"]["writer_line"], "#7f7f7f"))
    if s["val_split_lines_text"]["text_raw"]:
        sets.append(("personal photos\nVAL-split lines\n(held out)", s["val_split_lines_text"]["text_raw"], s["val_split_lines_author"]["writer_line"], "#1f77b4"))
    i = S["iam_heldout_pages"]
    sets.append(("IAM held-out page\ncached lines", i["cache"]["text_raw"], dict(acc=i["cache"]["line_writer_acc"], ci95=i["cache"]["line_ci95"], n_lines=i["cache"]["n_lines"]), "#2ca02c"))
    sets.append(("IAM held-out page\nProcessPage lines", i["server"]["text_raw"], dict(acc=i["server"]["line_writer_acc"], ci95=i["server"]["line_ci95"], n_lines=i["server"]["n_lines"]), "#ff7f0e"))
    fig, axs = plt.subplots(1, 2, figsize=(16 * C.CM, 8 * C.CM))
    xs = np.arange(len(sets))
    ax = axs[0]
    ax.bar(xs, [100 * t["char_acc_micro"] for _, t, _, _ in sets], color=[c for *_, c in sets])
    for k, (_, t, _, _) in enumerate(sets):
        ax.text(k, 100 * t["char_acc_micro"] + 1, f"{100*t['char_acc_micro']:.1f}\n(n={t['n_lines']})", ha="center", fontsize=8)
    ax.axhline(95, color="red", ls="--", lw=1); ax.text(len(sets) - 0.5, 95.6, "R2: 95 %", color="red", ha="right", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels([n for n, *_ in sets], fontsize=7)
    ax.set_ylim(0, 112); ax.set_ylabel("character accuracy [%] (micro, raw labels)")
    ax.set_title("text decoding", fontsize=9)
    ax = axs[1]
    vals = [100 * w["acc"] for _, _, w, _ in sets]
    lo = [100 * (w["acc"] - w["ci95"][0]) for _, _, w, _ in sets]
    hi = [100 * (w["ci95"][1] - w["acc"]) for _, _, w, _ in sets]
    ax.bar(xs, vals, color=[c for *_, c in sets], yerr=[lo, hi], capsize=3)
    for k, (_, _, w, _) in enumerate(sets):
        ax.text(k, 100 * w["acc"] + 3, f"{100*w['acc']:.1f}\n(n={w['n_lines']})", ha="center", fontsize=8)
    ax.axhline(95, color="red", ls="--", lw=1)
    ax.set_xticks(xs); ax.set_xticklabels([n for n, *_ in sets], fontsize=7)
    ax.set_ylim(0, 118); ax.set_ylabel("writer-ID accuracy per line [%] (95 % Wilson CI)")
    ax.set_title("writer identification (10-way)", fontsize=9)
    fig.tight_layout()
    C.savefig(fig, "qtp2_accuracy_subsets.png"); plt.close(fig)

    # ---- fig 2: per-page text accuracy and writer decision ----
    pages = [p for p in d["pages"] if p["path"] == "server"]
    fig, ax = plt.subplots(figsize=(16 * C.CM, 7 * C.CM))
    xs = np.arange(len(pages))
    cols = [C.PALETTE[0] if p["writer"] == "yeukita" else C.PALETTE[1] for p in pages]
    ax.bar(xs, [100 * p["server_style_char_acc"] for p in pages], color=cols)
    for k, p in enumerate(pages):
        ok = p["pred_writer"] == p["writer"]
        ax.text(k, 100 * p["server_style_char_acc"] + 1, "ID ok" if ok else "ID WRONG", ha="center", fontsize=7,
                color="black" if ok else "red", rotation=90)
    ax.axhline(95, color="red", ls="--", lw=1)
    ax.set_ylim(0, 115)
    ax.set_xticks(xs); ax.set_xticklabels([f"{p['writer'][0].upper()}{k+1}" for k, p in enumerate(pages)])
    ax.set_ylabel("page char accuracy, server /api/classify [%]")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C.PALETTE[0], label="yeukita"), Patch(color=C.PALETTE[1], label="dylan")], fontsize=8, loc="lower right")
    ax.set_title("QTP2: per-page results on the 13 photographs (server-style, all lines)", fontsize=9)
    fig.tight_layout()
    C.savefig(fig, "qtp2_per_page.png"); plt.close(fig)


if __name__ == "__main__":
    main()
