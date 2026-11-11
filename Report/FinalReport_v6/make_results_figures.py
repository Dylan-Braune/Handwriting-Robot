#u23539195 Dylan Braune.  the figures for section 4, all from one palette.
#
#   python make_results_figures.py
#
#Appendix 4's formatting rules drive every choice here:
#   * no frame around a figure, so every spine that is not an axis is removed
#   * no title inside the figure: the caption below it carries the description,
#     and Appendix 4 forbids a description at the top as well
#   * nothing smaller than 8 point, so the base size is 9 and nothing scales
#     down from it
#   * one font family and one size across every figure, which is why they all
#     come from this one file and one call to palette.Apply()
#   * figures roughly A4 column width and never below about 8 cm
#
#colours are black, silver, white and shades of blue only.  where a figure
#separates a met requirement from an unmet one it does so by saturation, dark
#blue against silver, and the caption says which is which, so the figure still
#reads in grey scale.
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIGS = HERE/"Figures"
FIGS.mkdir(exist_ok=True)
RESULTS = Path(r"C:\Users\braun\OneDrive\Documents\GitHub\Handwriting-Robot"
               r"\Software\CleanPipeline3\results")
sys.path.insert(0, str(RESULTS))
import palette as PAL
PAL.Apply()

SCR = Path(r"C:\Users\braun\AppData\Local\Temp\claude"
           r"\C--Users-braun-OneDrive-Documents-GitHub-Handwriting-Robot"
           r"\21096ef3-779e-4365-af8c-095a5dea04a3\scratchpad")
CM = 1/2.54


def NoFrame(ax, keep=("left", "bottom")):
    for s, sp in ax.spines.items():
        sp.set_visible(s in keep)
        sp.set_color(PAL.INK)


def Wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k/n
    d = 1+z*z/n
    c = (p+z*z/(2*n))/d
    h = z*((p*(1-p)/n+z*z/(4*n*n))**0.5)/d
    return 100*p, 100*max(0, c-h), 100*min(1, c+h)


def Confusion(rows, names, path, truthKey, predKey):
    """an 11x11 count matrix.  white at zero so an empty cell reads empty"""
    n = len(names)
    M = np.zeros((n, n), dtype=int)
    for r in rows:
        M[names.index(r[truthKey]), names.index(r[predKey])] += 1
    fig, ax = plt.subplots(figsize=(13*CM, 11.5*CM))
    vmax = M.max() if M.max() else 1
    ax.imshow(M, cmap=PAL.Cmap(), vmin=0, vmax=vmax)
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(names)
    ax.set_xlabel("writer named by the classifier")
    ax.set_ylabel("writer the page was synthesised in")
    for i in range(n):
        for j in range(n):
            if M[i, j]:
                #white text once the cell is dark enough to need it
                c = PAL.PAPER if M[i, j] > 0.55*vmax else PAL.INK
                ax.text(j, i, str(M[i, j]), ha="center", va="center",
                        color=c, fontsize=8)
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color=PAL.SILVER, linewidth=0.5)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    NoFrame(ax, keep=())
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return M


#======================================================================
rows = json.loads((SCR/"blocks"/"lines.json").read_text(encoding="utf-8"))
NAMES = sorted({r["writer"] for r in rows})
print("%d line samples, %d writers" % (len(rows), len(NAMES)))

#---1. confusion matrix, writer identification on the reproduced pages---
M = Confusion(rows, NAMES, FIGS/"res_confusion_reproduced.png", "writer", "named")
print("wrote res_confusion_reproduced.png   trace %d/%d" % (np.trace(M), M.sum()))

#---2. confusion matrix, the classifier's own held-out lines---
#every line's own prediction, recorded by the measurement itself rather than
#reconstructed from a per-writer summary, so the off-diagonal cells are the
#ones that actually occurred
hrows = json.loads((SCR/"holdout_namer"/"lines.json").read_text(encoding="utf-8"))
#the holdout measurement names its writers with the same labels as the rest of
#section 4, but its own axis order is taken from its own rows so that a label
#appearing only as a prediction still gets a column
HNAMES = sorted({r["writer"] for r in hrows} | {r["named"] for r in hrows})
Mh = Confusion(hrows, HNAMES, FIGS/"res_confusion_holdout.png", "writer", "named")
print("wrote res_confusion_holdout.png      trace %d/%d" % (np.trace(Mh), Mh.sum()))

#---3. the three blocks, writer identification and character accuracy---
#short tick labels: the full description belongs in the caption, and long
#ones collide at this width
BL = [("A", "A\nparagraph"), ("B", "B\npangram"), ("C", "C\nown text")]
fig, (axL, axR) = plt.subplots(1, 2, figsize=(17*CM, 8.5*CM))
xs = np.arange(len(BL))
acc, lo, hi, ns = [], [], [], []
for t, _lab in BL:
    sub = [r for r in rows if r["block"] == t]
    k = sum(1 for r in sub if r["named"] == r["writer"])
    p, a, b = Wilson(k, len(sub))
    acc.append(p); lo.append(p-a); hi.append(b-p); ns.append(len(sub))
axL.bar(xs, acc, color=PAL.DARK, width=0.6)
axL.errorbar(xs, acc, yerr=[lo, hi], fmt="none", ecolor=PAL.INK,
             capsize=4, linewidth=1)
axL.set_xticks(xs); axL.set_xticklabels([l for _t, l in BL])
axL.set_ylabel("writer identified correctly (% of lines)")
axL.set_ylim(0, 100)
for x, v, n, h in zip(xs, acc, ns, hi):
    axL.text(x, v+h+2.5, "%.1f%%" % v, ha="center", fontsize=8)
    axL.text(x, 3, "n=%d" % n, ha="center", fontsize=8, color=PAL.PAPER)
NoFrame(axL)

#micro character accuracy per block, summed not averaged
mic = []
for t, _lab in BL:
    sub = [r for r in rows if r["block"] == t]
    d = sum(r["dist"] for r in sub); nn = sum(r["refChars"] for r in sub)
    mic.append(100*max(0, 1-d/nn))
axR.bar(xs-0.17, mic, color=PAL.DARK, width=0.34, label="character (micro)")
mw = []
for t, _lab in BL:
    sub = [r for r in rows if r["block"] == t]
    wr = sum(r["wordsRight"] for r in sub); wn = sum(r["refWords"] for r in sub)
    mw.append(100*wr/wn if wn else 0)
axR.bar(xs+0.17, mw, color=PAL.SILVER, width=0.34, label="word (micro)")
axR.set_xticks(xs); axR.set_xticklabels([l for _t, l in BL])
axR.set_ylabel("accuracy (%)")
axR.set_ylim(0, 116)
axR.set_yticks([0, 20, 40, 60, 80, 100])
axR.legend(loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 1.03))
for x, v in zip(xs-0.17, mic):
    axR.text(x, v+2, "%.1f" % v, ha="center", fontsize=8)
for x, v in zip(xs+0.17, mw):
    axR.text(x, v+2, "%.1f" % v, ha="center", fontsize=8)
NoFrame(axR)
fig.tight_layout()
fig.savefig(FIGS/"res_blocks.png")
plt.close(fig)
print("wrote res_blocks.png")

#---4. the gantry geometry, measured against nominal---
fig, (a1, a2) = plt.subplots(1, 2, figsize=(17*CM, 8.5*CM))
sides = ["bottom", "top", "left", "right"]
meas = [100.3, 100.2, 100.1, 99.9]
a1.axhline(100.0, color=PAL.INK, linewidth=1.2, linestyle="--")
a1.bar(range(4), [m-100.0 for m in meas], bottom=100.0,
       color=[PAL.DARK, PAL.DARK, PAL.MID, PAL.MID], width=0.6)
a1.set_xticks(range(4))
a1.set_xticklabels(["bottom\n(X)", "top\n(X)", "left\n(Y)", "right\n(Y)"])
a1.set_ylabel("measured side length (mm)")
a1.set_ylim(99.5, 100.6)
for i, m in enumerate(meas):
    a1.text(i, m+0.02 if m > 100 else m-0.06, "%.1f" % m, ha="center", fontsize=8)
NoFrame(a1)

tri = ["base", "left side", "right side"]
tm, tn = [80.3, 56.2, 56.9], [80.0, 56.57, 56.57]
a2.bar(np.arange(3)-0.17, tn, width=0.34, color=PAL.SILVER, label="nominal")
a2.bar(np.arange(3)+0.17, tm, width=0.34, color=PAL.DARK, label="measured")
a2.set_xticks(range(3)); a2.set_xticklabels(tri)
a2.set_ylabel("length (mm)")
a2.legend(loc="upper right", fontsize=8)
for i, (v, w) in enumerate(zip(tn, tm)):
    a2.text(i-0.17, v+1, "%.2f" % v, ha="center", fontsize=8)
    a2.text(i+0.17, w+1, "%.1f" % w, ha="center", fontsize=8)
NoFrame(a2)
fig.tight_layout()
fig.savefig(FIGS/"res_gantry_geometry.png")
plt.close(fig)
print("wrote res_gantry_geometry.png")

#---5. the style signature: ink laid down for identical text---
pred = {r["writer"]: r for r in
        json.loads((SCR/"timing"/"predicted.json").read_text(encoding="utf-8"))}
ws = sorted(pred, key=lambda w: -pred[w]["downMm"])
fig, ax = plt.subplots(figsize=(15*CM, 8.5*CM))
ax.bar(range(len(ws)), [pred[w]["downMm"] for w in ws], color=PAL.DARK, width=0.62)
ax.set_xticks(range(len(ws)))
ax.set_xticklabels(ws, rotation=45, ha="right")
ax.set_ylabel("pen-down path length (mm)")
ax2 = ax.twinx()
ax2.plot(range(len(ws)), [pred[w]["pulses"] for w in ws], color=PAL.INK,
         marker="o", markersize=4, linewidth=1.2, label="pen lifts")
ax2.set_ylabel("pen lifts")
ax2.grid(False)
NoFrame(ax)
NoFrame(ax2, keep=("right", "bottom"))
ax2.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(FIGS/"res_style_signature.png")
plt.close(fig)
print("wrote res_style_signature.png")

#---6. accuracy per writer, sorted by character accuracy---
pa = json.loads((SCR/"blocks"/"per_author.json").read_text(encoding="utf-8"))
pa.sort(key=lambda r: -r["char"])
fig, ax = plt.subplots(figsize=(16*CM, 8.5*CM))
xs = np.arange(len(pa))
ax.bar(xs-0.21, [r["char"] for r in pa], width=0.42, color=PAL.DARK,
       label="character accuracy (micro)")
ax.bar(xs+0.21, [r["id"] for r in pa], width=0.42, color=PAL.SILVER,
       label="writer identified (% of lines)")
ax.errorbar(xs+0.21, [r["id"] for r in pa],
            yerr=[[r["id"]-r["lo"] for r in pa], [r["hi"]-r["id"] for r in pa]],
            fmt="none", ecolor=PAL.INK, capsize=3, linewidth=0.9)
ax.set_xticks(xs)
ax.set_xticklabels([r["label"] for r in pa])
ax.set_xlabel("writer")
ax.set_ylabel("accuracy (%)")
ax.set_ylim(0, 118)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.legend(loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 1.03))
for x, r in zip(xs, pa):
    ax.text(x, 3, "n=%d" % r["n"], ha="center", fontsize=8, color=PAL.INK)
NoFrame(ax)
fig.tight_layout()
fig.savefig(FIGS/"res_per_author.png")
plt.close(fig)
print("wrote res_per_author.png")

#---7. accuracy against connectedness---
fig, (c1, c2) = plt.subplots(1, 2, figsize=(17*CM, 8.5*CM), sharex=True)
lift = np.array([r["lifts"] for r in pa])
for ax, key, lab in ((c1, "char", "character accuracy (micro, %)"),
                     (c2, "id", "writer identified (% of lines)")):
    y = np.array([r[key] for r in pa])
    ax.scatter(lift, y, s=38, color=PAL.DARK, zorder=3)
    for r in pa:
        ax.annotate(r["label"], (r["lifts"], r[key]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    #least squares line, drawn only to show the trend the numbers state
    m, c = np.polyfit(lift, y, 1)
    xx = np.linspace(lift.min()-0.05, lift.max()+0.05, 10)
    ax.plot(xx, m*xx+c, color=PAL.SILVER, linewidth=1.4, zorder=2)
    rp = float(np.corrcoef(lift, y)[0, 1])
    rs = float(np.corrcoef(np.argsort(np.argsort(lift)).astype(float),
                           np.argsort(np.argsort(y)).astype(float))[0, 1])
    ax.set_xlabel("pen lifts per character")
    ax.set_ylabel(lab)
    ax.set_ylim(0, 112)
    ax.text(0.02, 0.03, "Pearson r = %+.3f\nSpearman rho = %+.3f\nn = 11 writers"
            % (rp, rs), transform=ax.transAxes, fontsize=8, va="bottom")
    NoFrame(ax)
fig.tight_layout()
fig.savefig(FIGS/"res_connectedness.png")
plt.close(fig)
print("wrote res_connectedness.png")

print("\nall figures in", FIGS)
