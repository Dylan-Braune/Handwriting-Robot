#colour g-code preview for the browser, display only.  rebuilt in
#testing/preview.py, kept here as the source to work from

def RenderGcodePreview(gcodePath,cfg,pxPerMm=6.0,strokePx=2,
                       showTravel=True,path=None):
    strokes = ParseGcode(gcodePath)
    W = int((cfg.boundsMaxXmm-cfg.boundsMinXmm)*pxPerMm)
    H = int((cfg.boundsMaxYmm-cfg.boundsMinYmm)*pxPerMm)
    img = Image.new('RGB',(max(8,W),max(8,H)),(255,255,255))
    d = ImageDraw.Draw(img)
    d.rectangle([0,0,W-1,H-1],outline=(210,210,210))

    def P(pt):
        #machine Y is up, image Y is down
        return ((pt[0]-cfg.boundsMinXmm)*pxPerMm,
                (cfg.boundsMaxYmm-pt[1])*pxPerMm)

    if showTravel:
        for a,b in zip(strokes,strokes[1:]):
            d.line([P(a[-1]),P(b[0])],fill=(235,170,170),width=1)
    for s in strokes:
        if len(s)>=2:
            d.line([P(p) for p in s],fill=(0,0,0),width=strokePx,
                   joint='curve')
    if path:
        img.save(path)
    return img,strokes

