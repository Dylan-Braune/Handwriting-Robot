#u23539195 Dylan Braune.  one colour palette for every figure in the report.
#
#black, silver, white and shades of blue.  nothing else: no red, green, amber
#or orange, including for pass and fail.  a figure that has to separate a met
#requirement from an unmet one does it by SATURATION, dark blue against
#silver, and says which is which in the caption, so the figure still reads
#when it is printed in grey and when the reader cannot distinguish hues.
#
#   from palette import SEQ, CAT, INK, SILVER, PAPER, Cmap
#
#the same colours are defined for LaTeX in the report preamble; if one side
#changes the other has to change with it.

#---the sequential ramp, pale to dark.  white sits at zero so an empty cell
#of a confusion matrix reads as genuinely empty rather than as a low value
SEQ = ["#FFFFFF", "#E8F0F7", "#C3D9EC", "#8FB8D8",
       "#5590BF", "#2A6399", "#123A6B", "#0A1F3C"]

#---categorical series, in the order they should be used
CAT = ["#123A6B",    #dark blue    the first and most important series
       "#5590BF",    #mid blue     the second
       "#A9B2BA",    #silver       the de-emphasised or reference series
       "#0A0A0A",    #black        a target line, a boundary, an annotation
       "#8FB8D8"]    #pale blue    a fifth, rarely needed

INK = "#0A0A0A"       #text, axes, rules
SILVER = "#A9B2BA"    #gridlines, secondary outlines, the unmet case
PAPER = "#FFFFFF"     #backgrounds
MID = "#5590BF"
DARK = "#123A6B"

#a met requirement against an unmet one, by saturation and not by hue
MET, UNMET = DARK, SILVER


def Cmap(name="braune_blue"):
    """the sequential ramp as a matplotlib colormap"""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, SEQ)


def Lerp(t):
    """the ramp sampled at t in [0,1], returned as an (r,g,b) tuple.
    for PIL, which has no colormaps of its own"""
    t = max(0.0, min(1.0, float(t)))
    x = t*(len(SEQ)-1)
    i = int(x)
    if i >= len(SEQ)-1:
        i, f = len(SEQ)-2, 1.0
    else:
        f = x-i
    a = SEQ[i].lstrip("#")
    b = SEQ[i+1].lstrip("#")
    return tuple(int(int(a[j:j+2], 16)*(1-f) + int(b[j:j+2], 16)*f)
                 for j in (0, 2, 4))


def Apply():
    """house style for matplotlib: silver gridlines, black axes, no colour"""
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": PAPER, "axes.facecolor": PAPER,
        "axes.edgecolor": INK, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "grid.color": SILVER, "grid.linewidth": 0.6, "grid.alpha": 0.6,
        "axes.grid": True, "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=CAT),
        "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
        "savefig.facecolor": PAPER, "savefig.dpi": 200,
    })


#the same thing for the report preamble, printed so it can be pasted in
LATEX = r"""
% colour palette -- black, silver, white, shades of blue.  keep in step with
% Software/CleanPipeline3/results/palette.py
\definecolor{bInk}{HTML}{0A0A0A}
\definecolor{bSilver}{HTML}{A9B2BA}
\definecolor{bPale}{HTML}{C3D9EC}
\definecolor{bMid}{HTML}{5590BF}
\definecolor{bDark}{HTML}{123A6B}
\definecolor{bDeep}{HTML}{0A1F3C}
"""

if __name__ == "__main__":
    print("sequential ramp:", " ".join(SEQ))
    print("categorical    :", " ".join(CAT))
    print("\nLaTeX definitions:")
    print(LATEX)
