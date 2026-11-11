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

#----------------------------------------------------------------------
#PAGE GEOMETRY AND TYPE SIZE
#
#Appendix 4 sets a floor of 8 pt on text inside a figure and asks for one
#font used consistently.  A figure only honours that floor if it is drawn
#at the width it is printed at, because LaTeX rescales the whole picture:
#\showthe\textwidth in the built report gives 455.24 pt = 6.30 in, so every
#figure here is authored 6.30 in wide and included at width=\linewidth.
#The scale factor is then exactly 1 and a point in the figure is a point on
#the page.  FS() carries the floor; nothing sets a bare fontsize.
#----------------------------------------------------------------------
WIDTH = 6.30                                  #inches, == \linewidth
FS_FLOOR = 8.0                                #Appendix 4 floor
FS_SCALE = 1.12

#every size in this folder is raised by FS_BUMP over the Appendix 4 floor, so
#the smallest text in any figure prints at FS_MIN rather than at the floor.
#the report body is 12 pt Times, and a 10 pt figure label beside it reads
#without the reader leaning in.  Appendix 4 sets a minimum, not a maximum.
FS_BUMP = 2.0
FS_MIN = FS_FLOOR+FS_BUMP                     #10.0 pt, the real floor used here


def FS(x):
    """a point size for figure text, never below the floor used here"""
    return max(FS_MIN,round(float(x)*FS_SCALE+FS_BUMP,1))


#matplotlib draws a subscript or a superscript at a fraction of its symbol's
#size, so a label set at the 8 pt floor prints its subscripts below it.  The
#fraction is raised here, and FSM() sets a floor for any label that carries
#a subscript or a superscript such that the smaller glyph still clears 8 pt.
SHRINK = 0.85
try:
    from matplotlib import _mathtext as _mt
    _mt.SHRINK_FACTOR = SHRINK
except Exception:                             #older matplotlib layout
    try:
        import matplotlib.mathtext as _mt
        _mt.SHRINK_FACTOR = SHRINK
    except Exception:
        SHRINK = 0.7

FSM_MIN = round(FS_MIN/SHRINK+0.05,1)


def FSM(x=0):
    """a point size for a label containing maths, subscripts included"""
    return max(FSM_MIN,round(float(x)*FS_SCALE+FS_BUMP,1))


#----------------------------------------------------------------------
#ONE FONT, AND IT IS THE REPORT'S OWN
#
#epr400.cls loads mathptmx, so the report body is Times.  The figures are
#therefore set in Times New Roman, and the maths is set in Times New Roman
#too rather than in a sans or a Computer Modern substitute: "custom" lets
#the roman, italic and bold maths faces be named, and matplotlib falls back
#to the STIX symbol faces, which share Times metrics, for the few glyphs
#Times itself has no outline for.  Nothing in this folder selects a family.
#----------------------------------------------------------------------
TIMES = "Times New Roman"

matplotlib.rcParams.update({
    "font.family":"serif",
    "font.serif":[TIMES,"STIXGeneral","DejaVu Serif"],
    "mathtext.fontset":"custom",
    "mathtext.rm":TIMES,
    "mathtext.it":f"{TIMES}:italic",
    "mathtext.bf":f"{TIMES}:bold",
    "mathtext.sf":TIMES,
    "mathtext.cal":f"{TIMES}:italic",
    "mathtext.tt":TIMES,
    "mathtext.default":"it",
    "font.size":FS_MIN,
    "axes.titlesize":FS(8.5),
    "axes.labelsize":FS_MIN,
    "xtick.labelsize":FS_MIN,
    "ytick.labelsize":FS_MIN,
    "legend.fontsize":FS_MIN,
    "axes.linewidth":0.8,
    "xtick.direction":"out","ytick.direction":"out",
    "figure.dpi":110,
    "pdf.fonttype":42,
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
#
#Appendix 4 puts the caption below the figure, in bold, as part of the
#document.  So the caption and the provenance line are no longer drawn
#onto the picture; Caption() and Stamp() record them and Save() writes
#them to figures/CAPTIONS.md, ready to paste into \caption{}.  That also
#removes the need to crop the footer off a figure before including it.
#----------------------------------------------------------------------
#======================================================================
#WRITER LABELS
#
#the five photographed writers are identified by letter everywhere a
#figure shows them, never by name.  Anon() is applied to every piece of
#text a figure draws and to the caption and provenance lines, so a name
#cannot reach a page through a title, a tick label or a legend
#======================================================================
AUTHOR = {"owen":"O","yeukita":"Y","dylan":"D","abhinav":"A","thiya":"T"}

import re as _re
_NAMEPAT = _re.compile(r"(?i)\b(owen|yeukita|dylan|abhinav|thiya)\s*(\d+)?")


#---Author: one writer id as the label a figure may show---
def Author(w):
    s = str(w)
    k = s.strip().lower()
    if k in AUTHOR:
        return "Author "+AUTHOR[k]
    return "Writer "+s if s.isdigit() else s


def _Sub(m):
    out = "Author "+AUTHOR[m.group(1).lower()]
    if m.group(2):
        out += " page %d" % int(m.group(2))
    return out


#---Anon: rewrite every writer name in a string as its letter---
def Anon(s):
    return _NAMEPAT.sub(_Sub,str(s))


_pending = {}
_captions = []


#---_AnonFigure: replace any writer name already drawn into the figure---
def _AnonFigure(fig):
    import matplotlib.text as _mt
    for t in fig.findobj(_mt.Text):
        try:
            old = t.get_text()
        except Exception:
            continue
        if old and _NAMEPAT.search(old):
            t.set_text(Anon(old))


def Save(fig,stem,note=""):
    OUT.mkdir(parents=True,exist_ok=True)
    _AnonFigure(fig)
    pdf = OUT/f"{stem}.pdf"
    fig.savefig(pdf,bbox_inches="tight",pad_inches=0.015)
    fig.savefig(OUT/f"{stem}.png",bbox_inches="tight",pad_inches=0.015,dpi=240)
    try:
        from panelexport import SavePanels
        nP = SavePanels(fig,stem)
        if nP:
            print(f"  wrote {nP} separate panel(s) for {stem}")
    except ImportError:
        pass
    plt.close(fig)
    _FixWidth(pdf,stem)
    _captions.append((stem,_pending.pop("caption",""),_pending.pop("stamp","")))
    _WriteCaptions()
    print(f"  wrote {stem}.pdf  {note}")
    return pdf


#---_FixWidth: force the saved page to exactly one \linewidth------------
#
#savefig(bbox_inches="tight") trims to the drawn content, so the page comes
#out a little narrower or wider than the figure was authored.  Included at
#width=\linewidth the page is then rescaled, and an 8 pt label stops being
#8 pt on paper.  The page is therefore padded out to exactly WIDTH, which
#only adds white margin and leaves the scale factor at 1.
def _FixWidth(pdf,stem,width=None):
    try:
        from pypdf import PdfReader,PdfWriter
    except ImportError:
        return
    want = 72.0*(width if width is not None else WIDTH)
    rd = PdfReader(str(pdf));pg = rd.pages[0]
    have = float(pg.mediabox.width)
    #within a few points the rescale is under one percent, which cannot
    #take an 8 pt label below the floor; past that it is a layout fault
    if abs(have-want)<=3.0:
        return
    if have>want:
        print(f"  ! {stem} is {have/72:.2f} in wide, over the {want/72:.2f} in "
              "column: it will be scaled down and its text will print small")
        return
    grow = (want-have)/2.0
    x0,y0 = float(pg.mediabox.left),float(pg.mediabox.bottom)
    x1,y1 = float(pg.mediabox.right),float(pg.mediabox.top)
    for box in (pg.mediabox,pg.cropbox):
        box.lower_left = (x0-grow,y0);box.upper_right = (x1+grow,y1)
    w = PdfWriter();w.add_page(pg)
    with open(pdf,"wb") as f:
        w.write(f)


def _WriteCaptions():
    out = ["# Captions and provenance",
           "",
           "The caption belongs below the figure in the report, set in bold, as",
           "Appendix 4 requires; it is kept here rather than drawn onto the",
           "picture.  The source line records which code and which data made the",
           "figure.","",""]
    for stem,cap,src in _captions:
        out += [f"## {stem}","",cap or "(none)","",f"Source: `{src}`" if src else "",""]
    (OUT/"CAPTIONS.md").write_text("\n".join(out),encoding="utf-8")


#----------------------------------------------------------------------
#ShowGray: an image panel with no ticks, optional title and frame
#
#aspect defaults to "equal" so a crop of handwriting is never stretched;
#a panel is sized to the crop instead of the crop being sized to the panel
#----------------------------------------------------------------------
def ShowGray(ax,arr,title=None,vmin=0,vmax=255,cmap="gray",frame=True,
             aspect="equal",crisp=True,bold=0):
    if crisp and cmap=="gray" and vmin==0 and vmax==255:
        arr = Crisp(arr,bold=bold)
    ax.imshow(arr,cmap=cmap,vmin=vmin,vmax=vmax,interpolation="nearest",
              aspect=aspect)
    ax.set_xticks([]);ax.set_yticks([]);ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(frame);s.set_color(SILVER);s.set_linewidth(0.6)
    if title:
        ax.set_title(title,fontsize=FS(8),pad=3)
    return ax


#---ShowMask: a boolean mask drawn as ink on paper---
def ShowMask(ax,mask,title=None,**kw):
    return ShowGray(ax,255*(~np.asarray(mask,bool)),title,**kw)


#---Caption: the one line of standing text every figure carries---
def Caption(fig,text,y=None):
    _pending["caption"] = " ".join(str(text).split())


#---Stamp: the provenance line, so a figure can never lose its source---
def Stamp(fig,text):
    _pending["stamp"] = Anon(text)


#======================================================================
#CROP HELPERS
#
#the complaint these answer: a line crop is roughly fifteen times wider
#than it is tall, so drawn across one column it is a few millimetres high
#and nothing inside it can be read.  InkRows throws away the blank margin
#a crop carries, Window takes a piece whose aspect suits a printed panel,
#and Bands stacks a long line so each piece is drawn at several times the
#scale a single row would allow
#======================================================================
#======================================================================
#CONTRAST
#
#the complaint these answer: a photographed line crop is a grey pencil
#stroke on a grey-white page, and printed small on a laser printer the
#stroke all but disappears.  Crisp stretches the crop so the darkest ink
#reaches black and the page reaches white, then pulls the mid-greys down
#so a thin stroke reads as a stroke rather than as a smudge.  Embolden
#thickens the stroke by a pixel or two where a panel is still small.
#
#this changes the appearance of the ink, never its geometry: no pixel
#moves, so a measurement or a traced centreline drawn over the panel
#still lands where the code put it.  Embolden is only ever applied to a
#display copy, and the figures say so in their captions where it is used.
#======================================================================
def Crisp(gray,lo=1.0,hi=97.0,gamma=1.55,bold=0):
    """one grey crop stretched to full black-on-white contrast"""
    a = np.asarray(gray)
    if a.dtype==bool:
        return np.where(a,0,255).astype(np.uint8)
    a = a.astype(np.float64)
    if bold:
        a = Embolden(a,bold)
    ink = a[a<200]
    dark = float(np.percentile(ink,lo)) if ink.size else float(a.min())
    page = float(np.percentile(a,hi))
    if page-dark<1.0:
        page = dark+1.0
    x = np.clip((a-dark)/(page-dark),0.0,1.0)
    return (255.0*x**gamma).astype(np.uint8)


def Embolden(gray,k=1):
    """the same crop with its dark ink thickened by k pixels each way"""
    a = np.asarray(gray,dtype=np.float64)
    out = a.copy()
    for dy in range(-k,k+1):
        for dx in range(-k,k+1):
            if dy==0 and dx==0:
                continue
            s = np.roll(np.roll(a,dy,axis=0),dx,axis=1)
            if dy>0:
                s[:dy,:] = a[:dy,:]
            elif dy<0:
                s[dy:,:] = a[dy:,:]
            if dx>0:
                s[:,:dx] = a[:,:dx]
            elif dx<0:
                s[:,dx:] = a[:,dx:]
            out = np.minimum(out,s)
    return out


def InkRows(gray,pad=0.10,thresh=200,minFrac=0.015):
    """the rows of a crop that carry ink, with a small margin kept

    a single speck of dust a long way below the writing would otherwise
    hold the window open and leave the line floating in white, so a row
    counts only once it carries minFrac of the busiest row's ink
    """
    a = np.asarray(gray)
    ink = (a<thresh) if a.dtype!=bool else a
    count = ink.sum(1)
    if not count.any():
        return 0,a.shape[0]
    rows = np.nonzero(count>=max(1.0,minFrac*count.max()))[0]
    if len(rows)==0:
        rows = np.nonzero(count)[0]
    h = rows[-1]-rows[0]+1
    m = max(2,int(pad*h))
    return max(0,int(rows[0])-m),min(a.shape[0],int(rows[-1])+1+m)


def TightRows(gray,pad=0.10,thresh=200):
    """the same crop with its blank top and bottom removed"""
    y0,y1 = InkRows(gray,pad,thresh)
    return np.asarray(gray)[y0:y1],y0


def Window(gray,aspect=5.0,start=None,thresh=200):
    """a left-anchored piece of a line whose width/height suits a panel"""
    a = np.asarray(gray)
    h = a.shape[0]
    if start is None:
        ink = (a<thresh) if a.dtype!=bool else a
        cols = np.nonzero(ink.any(0))[0]
        start = int(cols[0]) if len(cols) else 0
        start = max(0,start-max(2,int(0.05*h)))
    stop = min(a.shape[1],start+int(round(aspect*h)))
    return start,stop


def Bands(width,n):
    """n equal column ranges covering a line, for stacking it into bands"""
    edges = np.linspace(0,width,n+1).round().astype(int)
    return list(zip(edges[:-1],edges[1:]))


def PanelHeight(arr,width=None,nCols=1):
    """the height in inches an equal-aspect panel of this array needs"""
    w = (width if width is not None else WIDTH)/float(nCols)
    a = np.asarray(arr)
    return w*a.shape[0]/max(1,a.shape[1])
