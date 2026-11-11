#u23539195 coefficient justification: the seam and ownership coefficients, re-run on a
#densely written page
#on the standard probe (Owen, 20 lines, pitch 51) the seam cutter has clean paper
#to run through between every pair of lines, so none of its coefficients changed
#the outcome.  that is a property of the probe, not evidence that the
#coefficients do not matter.  this group repeats them on Abhinav's page, which is
#cursive and fits 26 lines at pitch 48 on the same paper, so ascenders and
#descenders of neighbouring lines actually meet
import probes as P


def C(**kw):
    return kw


DENSE = dict(page=P.PAGE_B,lines=[3],score=True,
             textWeights=P.TEXT_WEIGHTS,scoreLines=8)

SEAM_PANEL = dict(kind="seams",layout="cols",panelHeight=2.6)

SHARED_NOTE = (r"This is the same coefficient as in the entry above, run on a "
               r"second, denser page. The standard probe did not exercise it: "
               r"Owen's page is written at pitch $51$ with $20$ lines and the "
               r"seams have clean paper to follow. Abhinav's page is cursive and "
               r"carries $26$ lines at pitch $48$ on the same paper, so "
               r"neighbouring lines interlock and the seam has to work.")


ENTRIES = [

C(key="seam_ink_weight_dense",
  title="Weight on ink in the seam cost field, on a densely written page",
  symbol=r"$w_{\text{ink}}$",
  subsystem="Seam cutting on a dense page",
  module="segment", where="SeamCuts",
  default=4.0, values=[0.5,1.0,4.0,16.0,64.0],
  patches=lambda v: [("segment",
      "cuts.append(c0+LeastCostSeam(4.0*m[c0:c1+1]+0.012*((yy-mid)/halfGap)**2,step=7))",
      "cuts.append(c0+LeastCostSeam(%s*m[c0:c1+1]+0.012*((yy-mid)/halfGap)**2,step=7))" % v)],
  task="page_segment", params=DENSE, panel=SEAM_PANEL,
  metrics=["nLines","nLinesTruth","charAcc","pitch"],
  trendMetrics=["charAcc","nLines"],
  source=r"Least-cost seam carving \cite[10:3]{avidan2007}, applied to "
         r"handwritten line separation after \cite{arvanitopoulos2014}.",
  reason=r"See the main entry. The point of repeating it here is that the "
         r"balance between the ink term and the straightness and midline terms "
         r"only matters when the seam is forced to choose between cutting "
         r"through a descender and detouring a long way round it, and that "
         r"choice only arises on a page where the lines are close enough to "
         r"interlock.",
  understanding=r"The cost of one ink pixel is $w_{\text{ink}}$ against a "
                r"straightness penalty of $0.02$ per row of movement and a "
                r"midline penalty reaching $0.012$. At $w_{\text{ink}} = 0.5$ "
                r"one ink pixel costs the same as $25$ rows of movement, so the "
                r"seam will cut through a stroke rather than detour far; at "
                r"$64$ it costs $3\,200$ rows and the seam will go almost "
                r"anywhere to avoid ink.",
  alternatives=r"A straight horizontal cut, which is what a plain profile "
               r"segmenter does and which cuts every descender on a page like "
               r"this one.",
  worked=r"Abhinav's page has pitch $48$ px at half resolution against Owen's "
         r"$51$, with $26$ lines against $20$, so there is roughly a quarter "
         r"less clean paper between lines for the seam to use.",
  notes=SHARED_NOTE,
  ),

C(key="seam_tail_radius_dense",
  title="Closing radius applied before the seam is cut, on a densely written page",
  symbol=r"$r_{\text{tail}}$",
  subsystem="Seam cutting on a dense page",
  module="segment", where="SeamCuts",
  default=0.07, values=[0.0,0.03,0.07,0.16,0.30],
  patches=lambda v: [("segment","tailR = max(2,int(0.07*P))",
                      "tailR = max(2,int(%s*P))" % v)],
  task="page_segment", params=DENSE, panel=SEAM_PANEL,
  metrics=["nLines","nLinesTruth","charAcc","pitch"],
  trendMetrics=["charAcc","nLines"],
  source=r"Morphological closing before a path search \cite{haralick1987}.",
  reason=r"See the main entry. On a dense cursive page the closing has two "
         r"opposing jobs at once: it must seal the narrow channels a seam would "
         r"otherwise thread through a letter, and it must not weld a descender "
         r"of one line to an ascender of the next, because once they are welded "
         r"no seam can pass between those two lines at all and they merge.",
  understanding=r"At $P = 48$ px the radius is $\lfloor 0.07 \times 48 \rfloor "
                r"= 3$ px at the submitted value and $14$ px at $0.30$. The gap "
                r"between an ascender and the descender above it on this page is "
                r"a few tens of pixels, so a large radius genuinely can bridge "
                r"it.",
  alternatives=r"Sealing channels by a width test on the mask rather than a "
               r"closing, which would not fatten the stroke at all.",
  worked=r"$0.07 \times 48 = 3.36 \to 3$ px at the submitted value, against "
         r"$0.07 \times 51 = 3.57 \to 3$ px on the standard probe, so the "
         r"absolute radius is almost the same and only the spacing differs.",
  notes=SHARED_NOTE,
  ),

C(key="band_ink_share_dense",
  title="Share of a piece's ink that must lie in a band, on a densely written page",
  symbol=r"$\theta_{\text{band}}$",
  subsystem="Seam cutting on a dense page",
  module="segment", where="_BandOwnership",
  default=0.15, values=[0.02,0.08,0.15,0.30,0.50],
  patches=lambda v: [("segment",
      "substantial = cnt>=0.15*np.maximum(cnt.sum(1,keepdims=True),1)",
      "substantial = cnt>=%s*np.maximum(cnt.sum(1,keepdims=True),1)" % v)],
  task="page_segment", params=DENSE, panel=SEAM_PANEL,
  metrics=["nLines","nLinesTruth","charAcc","pitch"],
  trendMetrics=["charAcc","nLines"],
  source=r"Component-to-line assignment after a coarse cut "
         r"\cite{likforman2007}.",
  reason=r"See the main entry. Ownership only becomes contentious when a piece "
         r"of ink genuinely spans two bands, which on a well-spaced page is "
         r"rare and on a dense cursive page is common: a descender from one "
         r"line routinely reaches into the body of the next.",
  understanding=r"Because a piece substantial in more than one band is given to "
                r"the topmost, lowering $\theta_{\text{band}}$ pulls ink upward "
                r"and raising it falls back on majority ownership, which hands a "
                r"long descender to the line below the one it was written on.",
  alternatives=r"Majority ownership alone, or assignment by centroid.",
  worked=r"A piece with $300$ ink pixels counts as present in a band at $45$ "
         r"pixels at the submitted value, $6$ pixels at $0.02$ and $150$ at "
         r"$0.50$.",
  notes=SHARED_NOTE,
  ),

]
