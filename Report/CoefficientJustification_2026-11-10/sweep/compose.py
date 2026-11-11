#u23539195 coefficient justification: turns one sweep's five runs into one figure
#every panel is drawn here, so the whole study has one look: Times New Roman,
#10 pt floor, black/silver/white/blue only, authored at the report column width
import json
from pathlib import Path

import numpy as np

import figstyle as FS

FS.Apply()
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


#---Unpack: the packed polylines of one run back into a list---
def Unpack(pts,lens):
    out,i = [],0
    for n in lens:
        out.append(pts[i:i+int(n)])
        i += int(n)
    return out


#---Stretch: a grey crop contrast-stretched so faint ink reads on the page---
def Stretch(a):
    a = np.asarray(a,np.float64)
    if a.size == 0:
        return a
    lo,hi = np.percentile(a,1),np.percentile(a,99)
    if hi-lo < 1e-6:
        return np.clip(a,0,255)
    return np.clip((a-lo)*(255.0/(hi-lo)),0,255)


#---Window: cuts a crop to the leftmost part that fills a panel at a given aspect---
def Window(a,aspect=7.0):
    h,w = a.shape[:2]
    want = int(round(h*aspect))
    return a[:,:min(w,want)]


#======================================================================
#the panel kinds
#======================================================================

#---PanelImage: a grey array, drawn square-pixel and cropped to fill the panel---
def PanelImage(ax,npz,spec,metrics=None):
    key = spec.get("key","crop")
    if key not in npz:
        ax.text(0.5,0.5,"no output",ha="center",va="center",transform=ax.transAxes)
        ax.axis("off")
        return
    a = np.asarray(npz[key])
    if a.dtype == np.uint8 and spec.get("stretch",True):
        a = Stretch(a)
    if spec.get("aspect"):
        a = Window(a,float(spec["aspect"]))
    cmap = "gray" if not spec.get("binary") else None
    if spec.get("binary"):
        #a mask: ink dark blue on white, never a hue pair
        rgb = np.ones(a.shape+(3,),np.float64)
        on = np.asarray(a) > 0
        for c,v in enumerate((0x12/255,0x3A/255,0x6B/255)):
            rgb[...,c][on] = v
        ax.imshow(rgb,interpolation="nearest",aspect="equal")
    else:
        ax.imshow(a,cmap=cmap,vmin=0,vmax=255,interpolation="nearest",aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])


#---PanelSeams: the half-resolution page with every seam cut drawn across it---
def PanelSeams(ax,npz,spec,metrics=None):
    half = Stretch(np.asarray(npz["half"]))
    ax.imshow(half,cmap="gray",vmin=0,vmax=255,interpolation="nearest",aspect="equal")
    cuts = np.asarray(npz["cuts"])
    W = half.shape[1]
    x = np.arange(cuts.shape[1])
    #the cuts were measured on the half-size page the decisions were taken on
    sx = W/float(cuts.shape[1])
    for i in range(cuts.shape[0]):
        ax.plot(x*sx,cuts[i],color=FS.DARK,linewidth=0.45)
    ax.set_xlim(0,half.shape[1])
    ax.set_ylim(half.shape[0],0)
    ax.set_xticks([])
    ax.set_yticks([])


#---PanelPolys: a line crop with its traced centrelines drawn over it---
def PanelPolys(ax,npz,spec,metrics=None):
    base = npz.get("sk" if spec.get("over") == "sk" else "crop")
    a = np.asarray(base)
    if a.dtype == np.uint8 and a.max() <= 1:
        a = (1-a)*255.0
    else:
        a = Stretch(a)
    a = Window(a,float(spec.get("aspect",7.0)))
    ax.imshow(a,cmap="gray",vmin=0,vmax=255,interpolation="nearest",aspect="equal")
    polys = Unpack(np.asarray(npz["pts"]),np.asarray(npz["lens"]))
    #alternating blues so neighbouring traces can be told apart
    shades = [FS.DARK,FS.MID,FS.PALE]
    w = a.shape[1]
    for i,p in enumerate(polys):
        if len(p) < 2 or p[:,0].min() > w:
            continue
        ax.plot(p[:,0],p[:,1],color=shades[i%len(shades)],linewidth=0.7)
    ax.set_xlim(0,a.shape[1])
    ax.set_ylim(a.shape[0],0)
    ax.set_xticks([])
    ax.set_yticks([])


#---PanelGlyphs: the cut glyphs, each drawn in x-heights and set side by side---
def PanelGlyphs(ax,npz,spec,metrics=None):
    if "pts" not in npz or len(npz["lens"]) == 0:
        ax.text(0.5,0.5,"no glyphs",ha="center",va="center",transform=ax.transAxes)
        ax.axis("off")
        return
    polys = Unpack(np.asarray(npz["pts"]),np.asarray(npz["lens"]))
    per = np.asarray(npz["perGlyph"]) if "perGlyph" in npz else None
    groups,i = [],0
    if per is not None:
        for n in per:
            groups.append(polys[i:i+int(n)])
            i += int(n)
    else:
        groups = [[p] for p in polys]
    xoff = 0.0
    nShow = int(spec.get("maxGlyphs",14))
    for g in groups[:nShow]:
        pts = np.concatenate(g,0) if g else np.zeros((0,2))
        if len(pts) == 0:
            continue
        x0 = pts[:,0].min()
        wid = max(0.4,pts[:,0].max()-x0)
        for p in g:
            ax.plot(p[:,0]-x0+xoff,p[:,1],color=FS.DARK,linewidth=0.9,
                    solid_capstyle="round")
        xoff += wid+0.25
    ax.axhline(0.0,color=FS.SILVER,linewidth=0.5)
    ax.axhline(1.0,color=FS.SILVER,linewidth=0.5,linestyle=(0,(3,3)))
    ax.set_xlim(-0.2,max(1.0,xoff))
    ax.set_ylim(-0.9,2.1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


#---PanelPath: a machine path in millimetres, pen-down solid and pen-up dotted---
def PanelPath(ax,npz,spec,metrics=None):
    polys = Unpack(np.asarray(npz["pts"]),np.asarray(npz["lens"]))
    prev = None
    for p in polys:
        if len(p) == 0:
            continue
        ax.plot(p[:,0],p[:,1],color=FS.DARK,linewidth=0.8,solid_capstyle="round")
        if prev is not None and spec.get("showTravel",True):
            ax.plot([prev[0],p[0,0]],[prev[1],p[0,1]],color=FS.SILVER,
                    linewidth=0.5,linestyle=(0,(1,2)))
        prev = (p[-1,0],p[-1,1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


#---PanelGridImage: a square descriptor grid as a blue heat map---
def PanelGridImage(ax,npz,spec,metrics=None):
    a = np.asarray(npz[spec.get("key","grid")],np.float64)
    ax.imshow(a,cmap=FS.Ramp(),interpolation="nearest",aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])


#----------------------------------------------------------------------
#PanelBand: a line crop with the measured x-height band drawn over it, for a
#coefficient that changes a measurement rather than the picture
#----------------------------------------------------------------------
def PanelBand(ax,npz,spec,metrics=None):
    PanelImage(ax,npz,spec,metrics)
    band = (metrics or {}).get(spec.get("bandKey","coreBand"))
    if not band:
        return
    top,base = float(band[0]),float(band[1])
    x0,x1 = ax.get_xlim()
    ax.axhspan(top,base,xmin=0,xmax=1,facecolor=FS.PALE,alpha=0.30,zorder=2)
    for y,style in ((top,"-"),(base,"-")):
        ax.axhline(y,color=FS.DARK,linewidth=1.0,linestyle=style,zorder=3)
    ax.set_xlim(x0,x1)


PANELS = {
    "image": PanelImage,
    "band": PanelBand,
    "seams": PanelSeams,
    "polys": PanelPolys,
    "glyphs": PanelGlyphs,
    "path": PanelPath,
    "grid": PanelGridImage,
}


#----------------------------------------------------------------------
#MetricLine: the one number that says what the coefficient did, formatted for the
#strip under each panel
#----------------------------------------------------------------------
#short names, so a metric fits under a narrow panel at 10 pt
SHORT = {"nLines":"lines","nLinesTruth":"true lines","charAcc":"read acc",
         "inkPixels":"ink px","components":"pieces","coreBand":"band",
         "removedStraightPixels":"rule px","artifactPieces":"artefacts",
         "paperCoverage":"sheet","inkOutsidePaper":"off-sheet",
         "estimatedSeconds":"est s","dwellSeconds":"dwell s",
         "drawSeconds":"draw s","travelSeconds":"travel s",
         "penPulses":"pulses","joinedStrokes":"joins","joinedInkMm":"join mm",
         "closedRings":"rings","drawMm":"draw mm","travelMm":"travel mm",
         "nPolylines":"traces","totalLength":"trace len","meanPoints":"pts/trace",
         "skeletonPixels":"skel px","nGlyphs":"glyphs","nNone":"dropped",
         "alignConf":"conf","meanWidth":"mean w","strokeW":"stroke w",
         "nVariants":"variants","nChars":"chars","medianPriorD":"median d",
         "connectedness":"conn","predProb":"p(pred)","trueProb":"p(writer)",
         "inkFrac":"ink frac","widConf":"p(writer)","harmonic":"joint",
         "nStrokes":"strokes","penDownMm":"pen-down mm","seconds":"s",
         "rotation":"turned","axisScore0":"score 0","axisScore90":"score 90",
         "dropped":"dropped","scale":"scale","lines":"g-code lines",
         "steps":"steps","pred":"named","correct":"right","xh":"x-height",
         "slantDeg":"slant","pitch":"pitch",
         "charAccMean":"read acc","charAccSd":"sd","widConfMean":"p(writer)",
         "widConfSd":"sd p","harmonicMean":"joint","harmonicSd":"sd joint",
         "nSeeds":"seeds"}


def MetricLine(metrics,keys,sep="   "):
    bits = []
    for k in keys:
        v = metrics.get(k)
        if v is None:
            continue
        label = SHORT.get(k,k)
        if isinstance(v,float):
            s = ("%.4g" % v)
        elif isinstance(v,bool):
            s = "yes" if v else "no"
        elif isinstance(v,(list,dict)):
            continue
        else:
            s = str(v)
        bits.append("%s %s" % (label,s))
    return sep.join(bits)


#----------------------------------------------------------------------
#Figure: one coefficient's five runs as one figure, the chosen value's panel
#outlined in dark blue and the others in silver
#----------------------------------------------------------------------
def Figure(entry,runs,outPath):
    #runs: list of dicts {value, metrics, npzPath, chosen}
    spec = entry["panel"]
    kind = spec["kind"]
    layout = spec.get("layout","rows")
    n = len(runs)
    if layout == "rows":
        nr,nc = n,1
        panelH = float(spec.get("panelHeight",0.62))
        figsize = (FS.COLUMN_IN,nr*panelH+0.75)
    elif layout == "cols":
        nr,nc = 1,n
        figsize = (FS.COLUMN_IN,float(spec.get("panelHeight",2.4)))
    else:
        nr,nc = 2,3
        figsize = (FS.COLUMN_IN,float(spec.get("panelHeight",3.2)))
    fig,axes = plt.subplots(nr,nc,figsize=figsize,squeeze=False)
    flat = [axes[r][c] for r in range(nr) for c in range(nc)]
    for ax in flat[n:]:
        ax.axis("off")
    for ax,run in zip(flat,runs):
        npz = np.load(run["npzPath"]) if run.get("npzPath") and Path(run["npzPath"]).exists() else {}
        try:
            PANELS[kind](ax,npz,spec,run.get("metrics",{}))
        except Exception as e:                      # a failed run still gets a panel
            ax.axis("off")
            ax.text(0.5,0.5,"run failed: %s" % str(e)[:40],ha="center",va="center",
                    fontsize=10,transform=ax.transAxes)
        edge = FS.CHOSEN_EDGE if run["chosen"] else FS.OTHER_EDGE
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(edge)
            s.set_linewidth(1.4 if run["chosen"] else 0.6)
        #in a row of five narrow panels there is no room for a suffix or for
        #metrics side by side, so the value goes above and the metrics stack
        narrow = (nc > 2)
        mark = "" if narrow else ("  (submitted value)" if run["chosen"] else "")
        label = "%s = %s%s" % (entry.get("symbol",entry["key"]),
                               FmtValue(run["value"]),mark)
        #a wide row panel can carry three metrics side by side before the strip
        #grows wider than the panel and gets clipped
        shown = entry.get("metrics",[])
        if not narrow:
            shown = shown[:3]
        det = MetricLine(run["metrics"],shown,
                         sep=("\n" if narrow else "    "))
        ax.set_title(label,fontsize=10,color=FS.BLACK,
                     fontweight=("bold" if run["chosen"] else "normal"),pad=2.5,
                     loc=("center" if narrow else "left"))
        if det:
            ax.set_xlabel(det,fontsize=10,color=FS.BLACK,labelpad=1.5,
                          linespacing=1.25)
    fig.tight_layout(pad=0.35)
    Path(outPath).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(outPath)
    plt.close(fig)
    return outPath


#---FmtValue: a sweep value printed the way it appears in the source---
def FmtValue(v):
    if isinstance(v,float):
        s = ("%.6g" % v)
        return s
    return str(v)


#----------------------------------------------------------------------
#TrendFigure: the metric against the swept value, for a coefficient whose effect is
#a number rather than a picture
#----------------------------------------------------------------------
def TrendFigure(entry,runs,outPath,metricKeys=None):
    keys = metricKeys or entry.get("trendMetrics") or entry.get("metrics",[])[:2]
    keys = [k for k in keys
            if any(isinstance(r["metrics"].get(k),(int,float)) and
                   not isinstance(r["metrics"].get(k),bool) for r in runs)]
    if not keys:
        return None
    fig,axes = plt.subplots(1,len(keys),figsize=(FS.COLUMN_IN,2.1),squeeze=False)
    xs = [r["value"] for r in runs]
    numeric = all(isinstance(x,(int,float)) and not isinstance(x,bool) for x in xs)
    for j,k in enumerate(keys):
        ax = axes[0][j]
        ys = [r["metrics"].get(k) for r in runs]
        pos = xs if numeric else list(range(len(xs)))
        ax.plot(pos,ys,color=FS.MID,linewidth=1.0,marker="o",markersize=3.5,
                markerfacecolor=FS.MID,markeredgecolor=FS.MID,zorder=2)
        for r,x,y in zip(runs,pos,ys):
            if r["chosen"] and y is not None:
                ax.plot([x],[y],marker="o",markersize=6.5,markerfacecolor="none",
                        markeredgecolor=FS.DARK,markeredgewidth=1.4,zorder=3)
        if not numeric:
            ax.set_xticks(pos)
            ax.set_xticklabels([FmtValue(x) for x in xs])
        ax.set_xlabel(entry.get("symbol",entry["key"]),fontsize=10)
        ax.set_ylabel(k,fontsize=10)
        ax.grid(True,color=FS.SILVER,linewidth=0.4,alpha=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout(pad=0.4)
    Path(outPath).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(outPath)
    plt.close(fig)
    return outPath
