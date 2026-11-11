"""Draw a corrected vector redraw of the user-provided hardware block sketch."""
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white, black
import math

HERE = Path(__file__).resolve().parent
OUT = HERE / "Figures" / "design-hardware-overview.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)
W, H = 520, 360
c = canvas.Canvas(str(OUT), pagesize=(W, H), pageCompression=1)
c.setTitle("Handwriting robot hardware system block diagram")
c.setAuthor("Redrawn from the student's supplied hardware sketch")

BLUE = HexColor("#e8f2fb")
GREY = HexColor("#f2f3f4")
ROSE = HexColor("#fff0ed")
GREEN = HexColor("#edf8ee")
CYAN = HexColor("#edf8fa")
INK = HexColor("#183c5a")
POWER = HexColor("#a6382d")
SIGNAL = HexColor("#236297")
MECH = HexColor("#457b55")

def box(x, y, w, h, lines, fill=white, fs=9.2, bold=False):
    c.setFillColor(fill); c.setStrokeColor(HexColor("#273a4b")); c.setLineWidth(1)
    c.roundRect(x, y, w, h, 6, fill=1, stroke=1)
    c.setFillColor(black); c.setFont("Helvetica-Bold" if bold else "Helvetica", fs)
    step = fs + 2
    yy = y + (h + (len(lines)-1)*step)/2 - fs*.34
    for line in lines:
        c.drawCentredString(x+w/2, yy, line); yy -= step

def arrow(points, color=SIGNAL, width=1.25, dashed=False, head=5):
    c.saveState(); c.setStrokeColor(color); c.setFillColor(color); c.setLineWidth(width)
    if dashed: c.setDash(4, 3)
    p = c.beginPath(); p.moveTo(*points[0])
    for xy in points[1:]: p.lineTo(*xy)
    c.drawPath(p, stroke=1, fill=0)
    x1,y1=points[-2]; x2,y2=points[-1]; a=math.atan2(y2-y1,x2-x1)
    q=c.beginPath(); q.moveTo(x2,y2)
    q.lineTo(x2-head*math.cos(a-.48), y2-head*math.sin(a-.48))
    q.lineTo(x2-head*math.cos(a+.48), y2-head*math.sin(a+.48)); q.close()
    c.drawPath(q, stroke=0, fill=1); c.restoreState()

def txt(x,y,s,color=INK,fs=8.5,align="center"):
    c.setFillColor(color); c.setFont("Helvetica-Bold",fs)
    fn = c.drawCentredString if align == "center" else c.drawString
    fn(x,y,s)

def group(x,y,w,h,title,color):
    c.setFillColor(color); c.setStrokeColor(HexColor("#98a9b7")); c.setLineWidth(.7)
    c.roundRect(x,y,w,h,5,fill=1,stroke=1)
    c.setFillColor(INK); c.setFont("Helvetica-Bold",9)
    c.drawString(x+7,y+h-13,title)

# A compact left-to-right system view with labelled blocks and short links.
box(12,278,118,48,["Camera housing / page", "8 MP IMX179", "USB webcam"],GREY,10.2)
box(12,208,118,48,["Laptop / browser", "Ethernet or Wi-Fi"],GREY,10.2)
box(12,137,118,46,["Supplied DC adapter", "output unverified"],ROSE,10.2)
box(153,310,140,37,["ODROID DC input", "voltage to verify"],ROSE,10.2)
box(153,225,140,70,["ODROID N2+", "USB capture, server", "40-pin GPIO header"],BLUE,10.2,True)
box(153,157,140,48,["X/Y limits + pen-up switch", "five active-high inputs"],GREEN,10.2)
box(153,60,140,68,["BC337 low-side switch", "GPIO35 through 1 kΩ", "motor power from 3.3 V rail"],GREEN,10.2)
box(323,310,184,37,["External stepper supply", "24 V; limit set to 2.11 A"],ROSE,10.2)
box(323,243,184,51,["A4988 X/Y drivers", "3.3 V STEP/DIR logic", "phase current set separately"],BLUE,10.2)
box(323,183,184,43,["42BYGH40-01 motors ×2", "1.8°; 1.68 A/phase rated"],GREY,10.2)
box(323,118,184,46,["GT2 belts, rails, carriage", "Y cross-shaft drives both sides"],CYAN,10.2)
box(323,60,184,43,["3.3 V geared pen motor", "cam and pen-up switch"],GREEN,10.2)

# USB data and operator network.
arrow([(130,302),(153,275)],SIGNAL); txt(140,310,"USB",SIGNAL,10.2)
arrow([(130,232),(153,260)],SIGNAL); arrow([(153,252),(130,224)],SIGNAL)
txt(141,195,"network",SIGNAL,10.2)
# ODROID wall adapter: unknown installed output, so it has no assumed value.
arrow([(130,160),(143,160),(143,329),(153,329)],POWER)
# Switch feedback travels to the GPIO inputs.
arrow([(223,205),(223,225)],SIGNAL); txt(232,210,"0 / 3.3 V",SIGNAL,10.2,"left")
# 24 V supply to VMOT is kept separate from GPIO control and phase outputs.
arrow([(415,310),(415,294)],POWER); txt(424,299,"VMOT",POWER,10.2,"left")
arrow([(293,267),(323,267)],SIGNAL)
arrow([(415,243),(415,226)],POWER); txt(424,230,"chopped coils",POWER,10.2,"left")
arrow([(415,183),(415,164)],MECH); txt(425,170,"belt motion",MECH,10.2,"left")
# Board control and dedicated 3.3 V supply feed the pen switching stage.
arrow([(293,236),(305,236),(305,95),(293,95)],SIGNAL)
arrow([(293,230),(312,230),(312,80),(323,80)],POWER)
arrow([(293,93),(323,93)],POWER)
arrow([(415,118),(415,103)],MECH,dashed=True); txt(424,107,"pen lift",MECH,10.2,"left")

# The driver input level and motor supply are intentionally not conflated.
c.setFillColor(INK); c.setFont("Helvetica",10.2)
c.drawString(12,33,"No flyback diode is fitted to the pen motor.")
c.setFillColor(SIGNAL); c.drawString(12,15,"Blue: data / logic")
c.setFillColor(POWER); c.drawCentredString(260,15,"Red: supply / switched power")
c.setFillColor(MECH); c.drawRightString(508,15,"Dashed green: mechanical link")
c.showPage(); c.save()
print(OUT)
