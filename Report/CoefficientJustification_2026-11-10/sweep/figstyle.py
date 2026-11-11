#u23539195 coefficient justification: one shared look for every sweep figure
#black, silver, white and shades of blue only, Times New Roman, 10 pt floor,
#authored at the 455.24 pt report column so LaTeX never rescales a panel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#the report text column, 455.24 pt, in inches
COLUMN_IN = 455.24/72.27

#pale to dark, white at zero so an empty cell reads as empty
RAMP = ["#FFFFFF","#E8F0F7","#C3D9EC","#8FB8D8","#5590BF","#2A6399",
        "#123A6B","#0A1F3C"]

#categorical, in order of use
SERIES = ["#123A6B","#5590BF","#A9B2BA","#0A0A0A","#8FB8D8"]

DARK   = "#123A6B"
MID    = "#5590BF"
PALE   = "#8FB8D8"
SILVER = "#A9B2BA"
BLACK  = "#0A0A0A"

#the chosen value's panel is outlined in dark blue, the others in silver
CHOSEN_EDGE = DARK
OTHER_EDGE  = SILVER


#---Apply: sets the rcParams every figure in this study is drawn with---
def Apply():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "font.size": 10.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 10.0,
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "legend.fontsize": 10.0,
        "figure.titlesize": 11.0,
        "axes.edgecolor": BLACK,
        "axes.labelcolor": BLACK,
        "text.color": BLACK,
        "xtick.color": BLACK,
        "ytick.color": BLACK,
        "grid.color": SILVER,
        "grid.linewidth": 0.5,
        "axes.grid": False,
        "savefig.dpi": 300,
        "figure.dpi": 110,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


#---Ramp: a matplotlib colormap built from the sequential ramp---
def Ramp():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("hwr_blue",RAMP)
