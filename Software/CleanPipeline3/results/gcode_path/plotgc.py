import sys
from PIL import Image, ImageDraw
import colorsys

def Read(fn):
    """pen-down strokes and pen-up travels, in millimetres"""
    strokes=[]; travels=[]; cur=None; penUp=True; last=None
    for ln in open(fn,encoding="utf-8"):
        if ln.startswith("M5"):
            if cur and len(cur)>1: strokes.append(cur)
            cur=None; penUp=True
        elif ln.startswith("M3"):
            penUp=False; cur=[last] if last else []
        elif ln.startswith(("G0 X","G1 X")):
            pr=[p for p in ln.split() if p[0] in "XY"]
            if len(pr)<2: continue
            p=(float(pr[0][1:]),float(pr[1][1:]))
            if penUp:
                if last is not None and ln.startswith("G0"): travels.append((last,p))
            else:
                if cur is None: cur=[]
                cur.append(p)
            last=p
    if cur and len(cur)>1: strokes.append(cur)
    return strokes,travels

def Draw(fn,out,px=5.0,showTravel=True):
    strokes,travels=Read(fn)
    pts=[p for s in strokes for p in s]
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    x0,x1,y0,y1=min(xs),max(xs),min(ys),max(ys)
    W=int((x1-x0)*px)+40; H=int((y1-y0)*px)+40
    im=Image.new("RGB",(W,H),(255,255,255)); d=ImageDraw.Draw(im)
    def T(p): return (20+(p[0]-x0)*px, 20+(p[1]-y0)*px)
    if showTravel:
        for a,b in travels: d.line([T(a),T(b)],fill=(255,205,205),width=1)
    n=max(1,len(strokes))
    for i,s in enumerate(strokes):
        h=0.72*(i/n)                      # blue at the start, red at the end
        r,g,bl=[int(255*c) for c in colorsys.hsv_to_rgb(0.66-h*0.66,0.85,0.8)]
        d.line([T(p) for p in s],fill=(r,g,bl),width=2)
    im.save(out)
    return im.size,len(strokes),len(travels)

if __name__=="__main__":
    for fn,out in (("d01.gcode","plot_old.png"),("d02.gcode","plot_new.png")):
        print(fn,"->",out,Draw(fn,out))
