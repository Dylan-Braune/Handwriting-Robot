#u23539195 Dylan Braune.  T15, the software block diagram Section 3.2 asks for.
#
#three lanes: preparation off the device, reading a page, writing a new text.
#drawn from the module structure of CleanPipeline3, not from a sketch.
#
#  python vt_blockdiagram.py
from vt_common import *


def T15():
    fig,ax = plt.subplots(figsize=(7.4,5.9))
    ax.set_xlim(0,100);ax.set_ylim(5,100);ax.axis("off")

    DASH = (0,(1.4,1.4))
    BOUND = (0,(6,2,1,2))

    def Lane(y,h,title,note):
        ax.add_patch(plt.Rectangle((1,y),98,h,fc=SEQ[1],ec=SILVER,lw=0.7,zorder=0))
        ax.text(3,y+h-1.8,title,fontsize=8.8,va="top",color=INK,weight="bold")
        ax.text(3,y+h-5.4,note,fontsize=6.5,va="top",color="#445566")

    def Box(x,y,w,h,label,data=False,fs=6.7):
        ax.add_patch(plt.Rectangle((x,y),w,h,fc=PAPER,ec=INK,lw=0.9,
                                   ls=DASH if data else "-",zorder=2))
        ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=fs,zorder=3,
                linespacing=1.3)
        return (x,y,w,h)

    def Right(a,b,label="",ls="-",col=INK):
        x0,y0 = a[0]+a[2],a[1]+a[3]/2
        x1,y1 = b[0],b[1]+b[3]/2
        ax.annotate("",xy=(x1,y1),xytext=(x0,y0),
                    arrowprops=dict(arrowstyle="->",color=col,lw=0.85,
                                    linestyle=ls,shrinkA=0,shrinkB=0),zorder=1)
        if label:
            ax.text((x0+x1)/2,y0+1.0,label,ha="center",va="bottom",fontsize=5.6,
                    color=INK,zorder=4,bbox=dict(fc=PAPER,ec="none",pad=0.6))

    # ============================================ lane 1, preparation
    Lane(70,29,"1   Preparation, off the device",
         "run once on a desktop or a hosted GPU; not part of any request")
    a1 = Box(4,82,18,8,"labelled pages\nand line images",data=True)
    b1 = Box(27,82,18,8,"data preparation\nand augmentation")
    c1 = Box(50,82,19,8,"network training\n(train.py, trainlibrary.py)",fs=6.2)
    d1 = Box(74,82,21,8,"model weights (.npz)",data=True,fs=7.0)
    Right(a1,b1);Right(b1,c1);Right(c1,d1)

    a2 = Box(4,72,18,8,"labelled lines\n+ transcripts",data=True)
    b2 = Box(27,72,18,8,"glyph and style\nextraction (style.py)",fs=6.4)
    c2 = Box(50,72,45,8,"writer profiles   (StyleProfiles10/*.json)",
             data=True,fs=7.0)
    Right(a2,b2);Right(b2,c2)
    ax.annotate("",xy=(36,80),xytext=(36,82),
                arrowprops=dict(arrowstyle="->",color=SILVER,lw=0.85,ls="--"))
    ax.text(37.4,80.9,"the trained reader supplies the alignment",
            fontsize=5.7,color="#556677",va="center")

    # ============================================ lane 2, reading a page
    Lane(45,22,"2   Reading a page, on the ODROID",
         "one request: a photograph in, per-line text and one writer out")
    p1 = Box(4,47,18,8,"photograph\n(camera or upload)",data=True)
    p2 = Box(27,47,18,8,"conditioning and line\nsegmentation (segment.py)",fs=6.1)
    p3 = Box(50,47,19,8,"reader + writer models\n(recognise.py)",fs=6.2)
    p4 = Box(74,47,21,8,"per-line text,\nink-weighted page vote",data=True,fs=6.4)
    Right(p1,p2);Right(p2,p3);Right(p3,p4)
    ax.annotate("",xy=(60,55),xytext=(83,82),
                arrowprops=dict(arrowstyle="->",color=SILVER,lw=0.9,ls="--",
                                connectionstyle="arc3,rad=0.18"))
    ax.text(77,65,"weights loaded\nat start-up",fontsize=5.8,color="#556677",
            ha="center")

    # ============================================ lane 3, writing a text
    Lane(6,36,"3   Writing a new text, on the ODROID",
         "a separate request: digital text plus a chosen profile")
    w1 = Box(4,27,18,8,"digital text +\nselected profile",data=True)
    w2 = Box(27,27,20,8,"glyph layout and\nconnected synthesis\n(synthesise.py)",fs=6.2)
    w3 = Box(52,27,24,8,"best-of-N candidate search,\nscored by the same two models",
             fs=5.9)
    Right(w1,w2);Right(w2,w3)
    ax.annotate("",xy=(37,27),xytext=(64,27),
                arrowprops=dict(arrowstyle="->",color=SILVER,lw=0.9,ls="--",
                                connectionstyle="arc3,rad=0.45"))
    ax.text(50,23.4,"model scoring only, not physical validation",
            fontsize=5.8,color="#556677",ha="center",
            bbox=dict(fc=SEQ[1],ec="none",pad=0.8))

    w4 = Box(27,12,20,8,"pen trajectory\n(placed, ordered, decimated)",
             data=True,fs=6.0)
    w5 = Box(52,12,24,8,"G-code file",data=True,fs=7.4)
    w6 = Box(81,12,15,8,"gantry interpreter\n(reproduce.py)",fs=6.0)
    ax.annotate("",xy=(37,20),xytext=(60,27),
                arrowprops=dict(arrowstyle="->",color=INK,lw=0.85,
                                connectionstyle="arc3,rad=0.25"))
    Right(w4,w5)
    Right(w5,w6)

    ax.plot([78.5,78.5],[8.5,23.5],color=INK,lw=1.5,ls=BOUND,zorder=4)
    ax.text(78.5,24.2,"software  |  hardware",fontsize=6.5,color=INK,
            ha="center",va="bottom")

    ax.text(3,8.4,"Reading a page and writing a new text are separate requests: "
                  "classifying a page does not train a model, build a profile, "
                  "or start the gantry.",
            fontsize=6.5,color="#445566",va="bottom")

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig.legend(handles=[
        Patch(fc=PAPER,ec=INK,lw=0.9,label="processing stage"),
        Patch(fc=PAPER,ec=INK,lw=0.9,ls=DASH,label="stored data object"),
        Line2D([],[],color=INK,lw=0.9,label="data flow within a request"),
        Line2D([],[],color=SILVER,lw=0.9,ls="--",
               label="loaded weights / model scores"),
        Line2D([],[],color=INK,lw=1.5,ls=BOUND,
               label="software / hardware boundary")],
        fontsize=6.6,loc="upper center",bbox_to_anchor=(0.5,0.105),ncol=3)

    Caption(fig,"T15  The three things the software does, and when. Preparation runs off the device and produces the weights and "
                "the writer profiles; reading a page and writing a new text are separate on-device requests that share those "
                "stored objects. The G-code file is the boundary: the synthesiser ends there and the gantry interpreter begins. "
                "The candidate search feeds the reader's and the writer model's scores back into generation, which is model "
                "scoring and carries no information about the physical page.",y=-0.085)
    Stamp(fig,"CleanPipeline3 module structure")
    Save(fig,"T15_software_block_diagram")


if __name__=="__main__":
    print("T15  software block diagram")
    T15()
    print("done")
