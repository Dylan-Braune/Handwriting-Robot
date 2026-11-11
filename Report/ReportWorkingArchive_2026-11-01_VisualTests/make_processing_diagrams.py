from pathlib import Path
from reportlab.pdfgen import canvas
import pypdfium2 as pdfium
HERE=Path(__file__).resolve().parent/'Figures'
BLUE=(.08,.25,.4)
def box(c,x,y,w,h,lines,fill=(.94,.97,.99)):
    c.setStrokeColorRGB(*BLUE);c.setFillColorRGB(*fill);c.roundRect(x,y,w,h,5,stroke=1,fill=1)
    c.setFillColorRGB(.05,.1,.15);c.setFont('Helvetica',11)
    for j,s in enumerate(lines):c.drawCentredString(x+w/2,y+h/2+(len(lines)-1)*7-j*14,s)
def arrow(c,a,b):
    c.setStrokeColorRGB(*BLUE);c.setLineWidth(1);c.line(*a,*b)
    import math
    t=math.atan2(b[1]-a[1],b[0]-a[0]);p=c.beginPath();p.moveTo(*b)
    for d in [2.65,-2.65]:p.lineTo(b[0]+6*math.cos(t+d),b[1]+6*math.sin(t+d));p.moveTo(*b)
    c.drawPath(p)
def text(c,x,y,s,size=11):c.setFillColorRGB(*BLUE);c.setFont('Helvetica-Bold',size);c.drawString(x,y,s)
def end(c,name):
    c.showPage();c.save();doc=pdfium.PdfDocument(str(HERE/(name+'.pdf')));doc[0].render(scale=1.8).to_pil().save(HERE/(name+'.png'));doc.close()
name='design-processing-overview';c=canvas.Canvas(str(HERE/(name+'.pdf')),pagesize=(720,465))
text(c,15,442,'Preparation / training (offline)')
for x,ls in [(15,['Labelled pages','and transcripts']),(205,['Line preparation','and training']),(395,['Reader / writer','model weights'])]:box(c,x,365,160,55,ls)
arrow(c,(175,392),(205,392));arrow(c,(365,392),(395,392))
box(c,205,282,160,55,['Glyph extraction','and style measurement']);box(c,395,282,160,55,['Persistent writer','profiles'])
arrow(c,(95,365),(205,310));arrow(c,(475,365),(365,310));arrow(c,(365,310),(395,310))
text(c,15,252,'Recognising a photographed page')
for x,ls in [(15,['Page / camera','photograph']),(205,['Conditioning and','line segmentation']),(395,['Reader and writer','inference']),(585,['Text and writer','+ per-line detail'])]:box(c,x,170,115 if x==585 else 160,60,ls)
arrow(c,(175,200),(205,200));arrow(c,(365,200),(395,200));arrow(c,(555,200),(585,200))
arrow(c,(555,392),(565,240));arrow(c,(565,240),(475,230))
text(c,15,138,'Generating handwriting from new text')
for x,ls in [(15,['New text +','selected profile']),(205,['Glyph layout and','candidate selection']),(395,['Trajectory to','G-code']),(585,['Gantry','execution'])]:box(c,x,45,115 if x==585 else 160,65,ls)
arrow(c,(175,77),(205,77));arrow(c,(365,77),(395,77));arrow(c,(555,77),(585,77))
arrow(c,(475,282),(565,282));arrow(c,(565,282),(565,124));arrow(c,(565,124),(95,124));arrow(c,(95,124),(95,110))
arrow(c,(475,170),(285,110))
text(c,15,15,'Models score candidates; ordinary recognition does not train models or build profiles.',10)
end(c,name)

name='design-network-architecture';c=canvas.Canvas(str(HERE/(name+'.pdf')),pagesize=(720,430))
text(c,15,408,'Common layer pattern; reader and writer have separately trained weights')
stages=[(15,['Grey input','1 x 64 x 640']),(190,['2 conv + BN + ReLU','32 x 64 x 640','2 x 2 max pool']),(365,['4 conv + BN + ReLU','64 x 32 x 320','2 x 2 max pool']),(540,['6 conv + BN + ReLU','128 x 16 x 160','no spatial pool'])]
for x,ls in stages:box(c,x,300,160,75,ls)
for x in [175,350,525]:arrow(c,(x,337),(x+15,337))
box(c,365,200,160,55,['Maximum over height','128 features x 160 frames']);arrow(c,(620,300),(525,230))
box(c,15,95,235,65,['Reader: 2 bidirectional LSTMs','256 units per direction per layer','160 frames x 512 features'])
box(c,15,15,235,50,['Linear -> 75 classes per frame','CTC training / greedy decoding'])
arrow(c,(365,225),(133,160));arrow(c,(133,95),(133,65))
box(c,435,95,265,65,['Writer: mean over width -> 128','Linear -> 96, ReLU, dropout (training)','Linear -> K writer classes'])
box(c,435,15,265,50,['Cross-entropy training','Softmax writer probabilities'])
arrow(c,(525,225),(567,160));arrow(c,(567,95),(567,65))
end(c,name)
