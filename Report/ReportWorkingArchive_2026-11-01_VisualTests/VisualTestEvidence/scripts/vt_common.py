#u23539195 Dylan Braune.  shared setup for the design visual tests.
#
#every figure in this folder is produced from live project code and real
#project data.  nothing here measures a specification: these are mechanism
#illustrations for the Design section, so a reader can see a stage work and
#see where it would break.  the qualification measurements stay in Results.
#
#palette: black, silver, white and shades of blue only, taken from
#Software/CleanPipeline3/results/palette.py so the figures match the report.
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent/"figures"
DATA = HERE.parent/"data"
ROOT = HERE.parent.parent.parent              #repository root
PIPE = ROOT/"Software"/"CleanPipeline3"
DOWNLOADS = Path("C:/Users/braun/Downloads")

sys.path.insert(0,str(PIPE))
sys.path.insert(0,str(PIPE/"results"))

import palette as P
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
P.Apply()

#serif text so the figures sit beside the report body without a font clash
matplotlib.rcParams.update({
    "font.family":"serif",
    "font.serif":["Times New Roman","DejaVu Serif"],
    "mathtext.fontset":"dejavuserif",
    "axes.linewidth":0.8,
    "xtick.direction":"out","ytick.direction":"out",
    "figure.dpi":110,
})

SEQ,CAT = P.SEQ,P.CAT
INK,SILVER,PAPER,MID,DARK = P.INK,P.SILVER,P.PAPER,P.MID,P.DARK
PALE = SEQ[2]
CMAP = P.Cmap()

#the ink ramp for a mask or a cost field: white paper through to deep blue
CMAP_R = P.Cmap("braune_blue_r")


#======================================================================
#WHERE THE REAL DATA IS
#======================================================================
import paths as PATHS

PAGES = PATHS.PersonalPages
PROFILES = PATHS.Profiles
WEIGHTS = PATHS.Weights
CROPS = ROOT/"Software/CleanPipeline2/testing/results/images/crops"

#---SEGCROPS: the canonical per-page line crops.  this is the set the Design
#draft's own worked traces were computed on (trace_real_calculations.py reads
#Crops/Owen/Owen01/00.png), so the figures use it and quote the same numbers
SEGCROPS = ROOT/"Software/CNN/SegmentationBest/Crops"


#---LineCrop: one numbered line crop of one page, as the draft traced it---
def LineCrop(writer,page,index=0):
    return SEGCROPS/writer.capitalize()/f"{writer.capitalize()}{page:02d}"/f"{index:02d}.png"
REPRO = PIPE/"ReportResults"/"FullReproducedPages"

#---READER: the deployed reader checkpoint, the one recognise.py resolves---
READER_NPZ = WEIGHTS/"paper_cnn_bilstm_ctc_joint_best.npz"

#---the eleven writer shape namers from the stroke width sweep.  these live
#in Downloads and are not in version control; flagged in the manifest
SHAPE_NPZ = {
    0.06:DOWNLOADS/"author_shape_11_pw060_weights.npz",
    0.09:DOWNLOADS/"author_shape_11_pw090_weights.npz",
    0.11:DOWNLOADS/"author_shape_11_pw110_weights.npz",
    0.17:DOWNLOADS/"author_shape_11_pw170_weights.npz",
}
SHAPE_SELECTED = SHAPE_NPZ[0.17]


#----------------------------------------------------------------------
#PersonalPage: one photographed page and its transcript, by name
#----------------------------------------------------------------------
def PersonalPage(writer,index=1):
    folder = PAGES/writer.capitalize()
    img = folder/"Pages"/f"{writer.capitalize()}{index:02d}.jpg"
    lab = folder/"Labels"/f"{writer.capitalize()}{index:02d}_labels.txt"
    return img,lab


#---PageLines: the transcript lines for a page, blanks dropped---
def PageLines(labelPath):
    txt = Path(labelPath).read_text(encoding="utf-8",errors="replace")
    return [l.strip() for l in txt.splitlines() if l.strip()]


#---Gray: load any image as a 2-D uint8 grey array---
#maxSide matches trace_real_calculations.py, which thumbnails a page to 1600
#before calling Deskew.  every worked number quoted in the Design draft is on
#that reduced page, so the figures use it too or they would disagree with the
#text they sit beside
def Gray(path,maxSide=None):
    from PIL import Image
    im = Image.open(path).convert("L")
    if maxSide:
        im.thumbnail((maxSide,maxSide))
    return np.asarray(im,dtype=np.uint8)


#---PAGE_MAXSIDE: the reduction the report's worked traces were computed on---
PAGE_MAXSIDE = 1600


#----------------------------------------------------------------------
#Save: write one figure as pdf and png under a test number
#the pdf is what the report includes; the png is for quick browsing
#----------------------------------------------------------------------
def Save(fig,stem,note=""):
    OUT.mkdir(parents=True,exist_ok=True)
    pdf = OUT/f"{stem}.pdf"
    fig.savefig(pdf,bbox_inches="tight",pad_inches=0.02)
    fig.savefig(OUT/f"{stem}.png",bbox_inches="tight",pad_inches=0.02,dpi=200)
    plt.close(fig)
    print(f"  wrote {stem}.pdf  {note}")
    return pdf


#----------------------------------------------------------------------
#ShowGray: an image panel with no ticks, optional title and frame
#----------------------------------------------------------------------
def ShowGray(ax,arr,title=None,vmin=0,vmax=255,cmap="gray",frame=True):
    ax.imshow(arr,cmap=cmap,vmin=vmin,vmax=vmax,interpolation="nearest",
              aspect="auto")
    ax.set_xticks([]);ax.set_yticks([]);ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(frame);s.set_color(SILVER);s.set_linewidth(0.6)
    if title:
        ax.set_title(title,fontsize=8,pad=3)
    return ax


#---ShowMask: a boolean mask drawn as ink on paper---
def ShowMask(ax,mask,title=None,**kw):
    return ShowGray(ax,255*(~np.asarray(mask,bool)),title,**kw)


#---Caption: the one line of standing text every figure carries---
def Caption(fig,text,y=-0.045):
    fig.text(0.5,y,text,ha="center",va="top",fontsize=7.2,color="#444444",
             wrap=True)


#---Stamp: the provenance line, so a figure can never lose its source---
def Stamp(fig,text):
    fig.text(0.995,-0.012,text,ha="right",va="top",fontsize=5.4,
             color=SILVER,family="monospace")
