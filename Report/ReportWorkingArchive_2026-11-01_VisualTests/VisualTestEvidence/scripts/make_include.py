#u23539195 Dylan Braune.  write INCLUDE.tex: a ready-made figure block for
#every visual test, grouped by the design subsection it belongs to.
#
#  python make_include.py
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

#stem prefix -> (caption, label, width, subsection)
F = {
 "T15":("software block diagram, three lanes with the G-code file marked as the "
        "software and hardware boundary","des-software-block","0.98","3.2"),
 "T52":("coordinated two-axis stepping on a real commanded move","des-bresenham","0.95","3.3.2"),
 "T53":("the directional pulse-delay penalty, predicted and evaluated on a real job","des-pulsepenalty","0.95","3.3.2"),
 "T21":("Otsu's criterion on the local-brightness ratio image of one real page","des-otsu","0.95","3.4.2"),
 "T22":("the complete deskew angle search on five photographed pages","des-skewsweep","0.95","3.4.3"),
 "T23":("the quarter-turn decision and its acceptance margin","des-axisturn","0.92","3.4.3"),
 "T24":("row profile, peak floor, separation rule and the line pitch","des-pitch","0.96","3.4.4"),
 "T25":("the minimum-cost seam between two text lines, and the cost field it follows","des-seam","0.96","3.4.4"),
 "T26":("the seam midpoint weight swept over six real gaps","des-seamweight","0.95","3.4.4"),
 "T27":("connected-component ownership correcting the seam-band assignment","des-ownership","0.92","3.4.5"),
 "T29":("the aspect-preserving fit onto the 64 by 640 network canvas","des-canvasfit","0.96","3.4.5"),
 "T30":("one convolution evaluated at a single position of a real line crop","des-convarith","0.96","3.5.2"),
 "T31":("the learned kernels of the first convolutional layer","des-kernels","0.95","3.5.2"),
 "T32":("feature maps through the implemented backbone, and the height collapse","des-featurestack","0.90","3.5.2"),
 "T33":("max pooling on a real feature map","des-maxpool","0.95","3.5.2"),
 "T35":("the correspondence between an output frame and a source pixel","des-frametopixel","0.96","3.5.3"),
 "T36":("the CTC output lattice, the forced-alignment path and best-path decoding","des-ctclattice","0.97","3.5.3"),
 "T37":("per-frame character posteriors along one line","des-posteriors","0.96","3.5.3"),
 "T40":("the stroke-width sweep, and the dilation kernel each fraction selects","des-widthsweep","0.96","3.5.4"),
 "T39":("the implemented training-time augmentation set","des-augment","0.96","3.5.5"),
 "T38":("the first-principles derivatives checked against central differences","des-gradcheck","0.95","3.5.6"),
 "T42":("refinement of a character boundary by the implemented cut cost","des-cutcost","0.96","3.6.2"),
 "T19":("the stored glyph pools and the quantities the library gates act on","des-glyphgates","0.95","3.6.3"),
 "T41":("skeleton tracing and polyline simplification on one real line","des-trace","0.96","3.6.3"),
 "T43":("the stored glyph coordinate frame against image pixels","des-glyphcoords","0.96","3.6.3"),
 "T45":("variant pools, medoids and denoised prototypes for three writers","des-variantpool","0.97","3.6.3"),
 "T46":("the occupancy descriptor used by the shape prior","des-shapegrid","0.80","3.6.3"),
 "T47":("slant measured by shearing a line to its most concentrated columns","des-slant","0.95","3.6.4"),
 "T49":("connectedness counted on one real line","des-connect","0.96","3.6.4"),
 "T48":("the same sentence at four jitter amplitudes","des-jitter","0.96","3.6.5"),
 "T50":("line-level style calibration, before and after","des-calibration","0.96","3.6.5"),
 "T20":("joining decided by each writer's own measured connection fraction","des-joining","0.95","3.6.5"),
 "T51_candidate_selection_owen":("candidate selection for one writer and one "
        "sentence","des-candidates","0.95","3.6.6"),
 "T51_candidate_selection_dylan":("candidate selection for a second writer, same "
        "search configuration","des-candidates-b","0.95","3.6.6"),
 "T54":("stroke-order optimisation and the pen-up travel it saves","des-strokeorder","0.95","3.6.7"),
 "T55":("path decimation at five minimum-segment settings","des-decimation","0.95","3.6.7"),
 "T56":("the converter's nominal time budget, decomposed","des-timebudget","0.95","3.6.7"),
 "T57":("a real job placed in the calibrated workspace","des-placement","0.68","3.6.7"),
 "T59":("combining per-line writer evidence into one page decision","des-pagevote","0.96","3.7.1"),
 "T61":("job states and the interruption paths","des-jobstates","0.95","3.7.2"),
 "T60":("trajectory, emitted motion file and a page the robot drew","des-roundtrip","0.96","3.7.3"),
}

BLOCK = (r"\begin{figure}[!htbp]\centering" "\n"
         r"\includegraphics[width=%s\linewidth]{Figures/%s.pdf}" "\n"
         r"\caption{%s.}" "\n"
         r"\label{fig:%s}" "\n"
         r"\end{figure}" "\n")

HEAD = (
 "% Design visual tests, 2026-11-01.\n"
 "% One ready-made block per figure, grouped by the design subsection it\n"
 "% belongs to.  Copy the ones you want and delete the rest.\n"
 "%\n"
 "% Two things to do first:\n"
 "%   1. copy DesignVisualTests_2026-11-01/figures/*.pdf into Figures/, or\n"
 "%      add that folder to \\graphicspath\n"
 r"%   2. give every figure you keep a \ref in the body text, or the" "\n"
 "%      verifier will report it as an unreferenced float\n"
 "%\n"
 "% The captions below are short labels.  The fuller explanatory caption for\n"
 "% each figure is printed inside the figure itself and set out in MANIFEST.md;\n"
 "% move whichever version suits the page budget.\n")


def Main():
    stems = sorted(p.stem for p in (ROOT/"figures").glob("*.pdf"))
    bysec,missing = {},[]
    for stem in stems:
        key = stem if stem in F else stem.split("_")[0]
        if key not in F:
            missing.append(stem);continue
        cap,lab,wid,sec = F[key]
        bysec.setdefault(sec,[]).append((stem,cap,lab,wid))

    out = [HEAD]
    for sec in sorted(bysec):
        out.append("\n%% ============================================== section "+sec+"\n")
        for stem,cap,lab,wid in bysec[sec]:
            out.append(BLOCK%(wid,stem,cap[0].upper()+cap[1:],lab))
    if missing:
        out.append("\n% no caption entry for: "+", ".join(missing)+"\n")

    (ROOT/"INCLUDE.tex").write_text("\n".join(out),encoding="utf-8")
    n = sum(len(v) for v in bysec.values())
    print(f"wrote INCLUDE.tex: {n} blocks across {len(bysec)} subsections"
          + (f"; {len(missing)} without captions" if missing else ""))


if __name__=="__main__":
    Main()
