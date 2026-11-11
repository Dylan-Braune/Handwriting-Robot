#u23539195 D.R. Braune.  writes each panel of a finished figure as its own
#image file, so a panel can be used on its own instead of only as part of
#the composite.  off unless FIG_PANELS names a folder, so an ordinary run
#of a generator produces exactly what it produced before.
import os
import re
from pathlib import Path


#---_Slug: a panel title reduced to a file-name fragment---
def _Slug(s):
    s = re.sub(r"[^a-z0-9]+","-",str(s).lower()).strip("-")
    return s[:52] or "panel"


#----------------------------------------------------------------------
#SavePanels: one png per axes of a matplotlib figure
#each panel is cropped to its own tight bounding box, so it keeps its
#title, its tick labels and its colour bar.  where an axes holds a single
#image it is also written out as the bare array at full resolution, which
#is the version to re-crop or re-scale
#----------------------------------------------------------------------
def SavePanels(fig,stem,dpi=300):
    root = os.environ.get("FIG_PANELS")
    if not root:
        return 0
    out = Path(root)/stem
    out.mkdir(parents=True,exist_ok=True)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    n = 0
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        if ax.get_label()=="<colorbar>":
            continue   #the colour scale is not a panel of its own
        try:
            bb = ax.get_tightbbox(rend)
        except Exception:
            bb = None
        if bb is None or bb.width<=1 or bb.height<=1:
            continue
        n+=1
        title = ax.get_title() or ax.get_ylabel() or ax.get_xlabel() or ""
        name = f"{n:02d}_{_Slug(title)}"
        box = bb.expanded(1.03,1.03).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(out/f"{name}.png",bbox_inches=box,dpi=dpi,
                    facecolor=fig.get_facecolor())
        #an axes showing one image also goes out as the array itself
        imgs = [im for im in ax.get_images()]
        if len(imgs)==1:
            _SaveArray(imgs[0],out/f"{name}_image.png")
    return n


#---_SaveArray: the pixels an AxesImage holds, written at full size---
def _SaveArray(im,path):
    try:
        import numpy as np
        a = im.get_array()
        if a is None:
            return
        a = np.asarray(a)
        if a.ndim==2:
            lo,hi = im.get_clim()
            if hi is None or lo is None or hi<=lo:
                lo,hi = float(np.nanmin(a)),float(np.nanmax(a))
            g = np.clip((a.astype(np.float64)-lo)/max(hi-lo,1e-12),0,1)
            rgba = im.get_cmap()(g,bytes=True)
        else:
            rgba = np.asarray(a)
            if rgba.dtype!=np.uint8:
                rgba = (np.clip(rgba,0,1)*255).astype(np.uint8)
        if im.origin=="lower":
            rgba = rgba[::-1]
        _WritePng(path,rgba)
    except Exception as e:
        print(f"    (panel array not written for {path.name}: {e})")


#----------------------------------------------------------------------
#_WritePng: an rgb or rgba array to a png file
#written out here rather than handed to an imaging library, the same way
#the pipeline writes and reads its own image files
#----------------------------------------------------------------------
def _WritePng(path,a):
    import struct
    import zlib
    import numpy as np
    a = np.asarray(a,dtype=np.uint8)
    if a.ndim==2:
        a = np.dstack([a,a,a])
    h,w,ch = a.shape
    if ch==3:
        colour = 2
    elif ch==4:
        colour = 6
    else:
        a = np.dstack([a[:,:,0]]*3)
        h,w,ch = a.shape
        colour = 2
    raw = b"".join(b"\x00"+a[r].tobytes() for r in range(h))

    def Chunk(kind,body):
        return (struct.pack(">I",len(body))+kind+body
                +struct.pack(">I",zlib.crc32(kind+body)&0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB",w,h,8,colour,0,0,0)
    png = (b"\x89PNG\r\n\x1a\n"+Chunk(b"IHDR",ihdr)
           +Chunk(b"IDAT",zlib.compress(raw,6))+Chunk(b"IEND",b""))
    Path(path).write_bytes(png)
