#u23539195 Dylan Braune.  T61, the job states the server distinguishes.
#
#  python vt_jobstates.py
from vt_common import *


def T61():
    fig,ax = plt.subplots(figsize=(7.2,3.0))
    ax.set_xlim(0,108);ax.set_ylim(0,58);ax.axis("off")

    def Box(x,y,w,h,label,fill=PAPER,fs=7.2):
        ax.add_patch(plt.Rectangle((x,y),w,h,fc=fill,ec=INK,lw=1.0,zorder=2))
        ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=fs,
                zorder=3,linespacing=1.3)
        return (x,y,w,h)

    def Line(pts,label="",ls="-",lx=None,ly=None,ha="center"):
        xs = [p[0] for p in pts];ys = [p[1] for p in pts]
        ax.plot(xs[:-1],ys[:-1],color=INK,lw=0.95,ls=ls,zorder=1,
                solid_capstyle="butt")
        ax.annotate("",xy=pts[-1],xytext=pts[-2],
                    arrowprops=dict(arrowstyle="->",color=INK,lw=0.95,
                                    linestyle=ls,shrinkA=0,shrinkB=0),zorder=1)
        if label:
            ax.text(lx,ly,label,ha=ha,va="center",fontsize=FS(6.3),color=INK,
                    zorder=4,bbox=dict(fc=PAPER,ec="none",pad=0.9))

    gen = Box(2,30,21,11,"generated\ntrajectory, preview,\nG-code, metadata",fill=SEQ[1])
    chk = Box(30,30,20,11,"draw requested:\nconnected and\ncalibrated?")
    ref = Box(57,44,21,9,"refused\n(no motion attempted)",fill=SEQ[1])
    run = Box(57,30,21,11,"executing:\nthe pulse loop checks\nabort and end stops",fill=SEQ[2])
    don = Box(57,16,21,9,"completed",fill=SEQ[3])
    itr = Box(57,1,21,11,"interrupted or failed:\nlatched fault, retract,\npen raised, parked",fill=SILVER)

    Line([(23,35.5),(30,35.5)],"the operator inspects\nthe job first",
         lx=26.5,ly=43.5)
    Line([(50,37.5),(53.5,37.5),(53.5,48.5),(57,48.5)],"no",lx=55.2,ly=45.0)
    Line([(50,33.5),(57,33.5)],"yes",lx=53.5,ly=31.2)
    Line([(67.5,30),(67.5,25)],"the file ends",lx=69.0,ly=27.6,ha="left")

    #the abort path leaves executing on the right and goes round the completed
    #state, because an interrupted job never reaches it
    Line([(78,33.5),(84,33.5),(84,6.5),(78,6.5)],"abort requested,\n"
         "or an end stop\nconfirmed",ls="--",lx=85.5,ly=20,ha="left")

    ax.text(2,25,"The stop request is a software path,\n"
                 "not an independent disconnection of\nmotor power.",
            fontsize=FS(6.7),color="#445566",va="top")
    ax.text(2,12,"Calibration establishes the coordinate\n"
                 "reference in software; missed steps can\n"
                 "invalidate it without the controller\nknowing.",
            fontsize=FS(6.7),color="#445566",va="top")

    Caption(fig,"T61  The job states the server distinguishes and the paths between them. Generation and drawing are separate "
                "requests, so a trajectory can be inspected before the gantry moves, and a refusal is recorded distinctly from a "
                "failure. During motion the pulse loop checks an abort event and confirmed end stops on every iteration, and a "
                "trip latches a fault before the recovery routines run. No saved preview and no count of emitted pulses proves "
                "that the nib followed the intended path.")
    Stamp(fig,"web/server.py job states; reproduce.RequestAbort / EmergencyStop")
    Save(fig,"T61_job_states")


if __name__=="__main__":
    print("T61  job states")
    T61()
    print("done")
