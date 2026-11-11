#u23539195 coefficient justification: the seam, ownership and crop-building coefficients
import probes as P


def C(**kw):
    return kw


#the standard page-level run: segment the probe page, keep one crop, and score
#the first eight crops against the typed transcript
PAGE_RUN = dict(page=P.PAGE,lines=[P.LINE],score=True,
                textWeights=P.TEXT_WEIGHTS,scoreLines=8)

SEAM_PANEL = dict(kind="seams",layout="cols",panelHeight=2.6)
CROP_PANEL = dict(kind="image",key="crop%d" % P.LINE,aspect=7.0)


ENTRIES = [

C(key="rule_flat_len",
  title="Length above which a flat run on the page is printed structure",
  symbol=r"$L_{\text{flat}}$",
  subsystem="Printed-structure removal",
  module="segment", where="RemovePrintedStructure",
  default=1.25, values=[0.6,0.9,1.25,1.8,2.5],
  patches=lambda v: [("segment",
      "flat = AnyWithinRadius(LongRunMask(AnyWithinRadius(mask,1,0),int(1.25*P),1),1,0)&mask",
      "flat = AnyWithinRadius(LongRunMask(AnyWithinRadius(mask,1,0),int(%s*P),1),1,0)&mask" % v)],
  task="page_segment", params=PAGE_RUN, panel=SEAM_PANEL,
  metrics=["removedStraightPixels","nLines","charAcc","pitch"],
  trendMetrics=["removedStraightPixels","charAcc"],
  source=r"Long-run detection for printed rules and table borders is the "
         r"classical morphological approach \cite{haralick1987}, and removing "
         r"ruling before layout analysis is standard for camera-captured forms "
         r"\cite{shafait2008}.",
  reason=r"On a whole page the printed ruling is horizontal and runs much further "
         r"than any pen stroke. The limit is written in line pitches $P$, which "
         r"is measured on the page itself, so it adapts to how large the writing "
         r"is. One pitch is the distance between successive lines; $1.25P$ is "
         r"just over that, which no single horizontal pen stroke spans, while a "
         r"printed rule crosses the whole writing area and so spans many pitches. "
         r"The run is measured after a one-pixel vertical dilation, so a rule "
         r"broken by a pen stroke crossing it is still seen as one run.",
  understanding=r"Lowering $L_{\text{flat}}$ begins to delete the long horizontal "
                r"parts of the writing itself, which on a cursive page means the "
                r"baseline carrier stroke joining several letters. Raising it "
                r"leaves ruling in the mask, where it joins separate lines "
                r"together and defeats the seam cutter.",
  alternatives=r"A Hough line transform \cite{duda1972} to find the rules "
               r"explicitly, or removing them from the greyscale by estimating "
               r"and subtracting the periodic ruling, which also needs the pitch.",
  worked=r"On the probe page $P = 51$ px at half resolution, so "
         r"$1.25P = 63$ px: a flat run must be at least $63$ px long before it "
         r"is deleted. The submitted setting removes $1988$ pixels of straight "
         r"structure from this page.",
  ),

C(key="rule_upright_len",
  title="Length above which an upright run on the page is printed structure",
  symbol=r"$L_{\text{up}}$",
  subsystem="Printed-structure removal",
  module="segment", where="RemovePrintedStructure",
  default=1.4, values=[0.7,1.0,1.4,2.0,2.8],
  patches=lambda v: [("segment",
      "upright = AnyWithinRadius(LongRunMask(AnyWithinRadius(mask,3,1),int(1.4*P),0),3,1)&mask",
      "upright = AnyWithinRadius(LongRunMask(AnyWithinRadius(mask,3,1),int(%s*P),0),3,1)&mask" % v)],
  task="page_segment", params=PAGE_RUN, panel=SEAM_PANEL,
  metrics=["removedStraightPixels","nLines","charAcc","artifactPieces"],
  trendMetrics=["removedStraightPixels","charAcc"],
  source=r"As for the flat limit: long-run structure removal "
         r"\cite{haralick1987,shafait2008}.",
  reason=r"The vertical counterpart catches a margin rule, a spiral binding edge "
         r"or the fold shadow down a page. The limit is larger than the "
         r"horizontal one because tall pen strokes are more common than long "
         r"flat ones: an ascender plus descender through one line already reaches "
         r"most of a pitch, and two lines of writing whose ascenders and "
         r"descenders touch can reach beyond it. $1.4P$ is above that worst case, "
         r"so a stroke must cross nearly a line and a half before it is treated "
         r"as structure. The horizontal dilation radius of $3$ before the run "
         r"test tolerates a rule that wavers by a few pixels across the page.",
  understanding=r"This is the more dangerous of the two limits, because a "
                r"descender from one line meeting an ascender from the next makes "
                r"a genuinely long vertical run of real ink. The asymmetry "
                r"between $1.25$ and $1.4$ is exactly that risk difference.",
  alternatives=r"Deskewing first and then removing exactly vertical runs, which "
               r"this code does do, or a component-level aspect-ratio test, which "
               r"\texttt{DropArtifactPieces} applies afterwards as a second net.",
  worked=r"At $P = 51$ px, $1.4P = 71$ px, against an x-height of about $21$ px "
         r"at half resolution: a mark must be over three x-heights tall and under "
         r"about seven pixels wide to be removed.",
  ),

C(key="seam_ink_weight",
  title="Weight on ink in the seam cost field",
  symbol=r"$w_{\text{ink}}$",
  subsystem="Seam cutting",
  module="segment", where="SeamCuts",
  default=4.0, values=[1.0,2.0,4.0,8.0,16.0],
  patches=lambda v: [("segment",
      "cuts.append(c0+LeastCostSeam(4.0*m[c0:c1+1]+0.012*((yy-mid)/halfGap)**2,step=7))",
      "cuts.append(c0+LeastCostSeam(%s*m[c0:c1+1]+0.012*((yy-mid)/halfGap)**2,step=7))" % v)],
  task="page_segment", params=PAGE_RUN, panel=SEAM_PANEL,
  metrics=["nLines","charAcc","pitch"],
  trendMetrics=["charAcc","nLines"],
  source=r"A least-cost path cut through a cost field is seam carving "
         r"\cite[10:3]{avidan2007}; applying it between text lines so the cut "
         r"goes around descenders rather than through them is the standard "
         r"handwritten line-separation use \cite{arvanitopoulos2014}.",
  reason=r"The seam between two lines must prefer to pass through white paper and "
         r"only cross ink when it has no choice. The cost at a pixel is "
         r"$w_{\text{ink}}\,m + 0.012\,((y-\text{mid})/\text{halfGap})^2$, so "
         r"$w_{\text{ink}}$ is the price of cutting one ink pixel measured "
         r"against the price of wandering away from the midline. The two terms "
         r"are deliberately far apart in scale: $4.0$ against a quadratic that "
         r"reaches only $0.012$ at the full half-gap. One ink pixel therefore "
         r"costs the same as about $333$ pixel-rows of maximum drift, which is "
         r"the statement \enquote{go round the descender, however far round you "
         r"must go}.",
  understanding=r"Only the \emph{ratio} of the three coefficients matters, not "
                r"their absolute size, since the path that minimises the total is "
                r"unchanged by a common factor. Raising $w_{\text{ink}}$ makes "
                r"the seam ever more willing to detour; lowering it lets the seam "
                r"take a straighter path and slice through descenders, which "
                r"gives the line below a stray mark and the line above a "
                r"truncated letter.",
  alternatives=r"A straight horizontal cut at the midpoint, which is what most "
               r"simple profile segmenters do and which cuts every descender; or "
               r"the $A^*$ path of \cite{arvanitopoulos2014}, which is the same "
               r"idea with a different search.",
  worked=r"On the probe page the seam runs over $2448/2 = 1224$ columns with "
         r"$\text{step} = 7$, so each column may move up to seven rows and the "
         r"straightness penalty for the largest move is $0.02\times7 = 0.14$, "
         r"still far under the $4.0$ cost of one ink pixel.",
  prior=r"An eight-setting, 48-run sensitivity study of this field was recorded "
        r"earlier in \texttt{Evidence/coefficient\_sensitivity.json}; it varied "
        r"the midpoint and movement penalties separately on six preselected gaps "
        r"and found two raw-ink intersections retained at every setting. That "
        r"study did not establish a unique optimum, and neither does this sweep.",
  ),

C(key="seam_midline_pull",
  title="Weight pulling the seam back towards the midline",
  symbol=r"$w_{\text{mid}}$",
  subsystem="Seam cutting",
  module="segment", where="SeamCuts",
  default=0.012, values=[0.0,0.004,0.012,0.05,0.25],
  patches=lambda v: [("segment",
      "cuts.append(c0+LeastCostSeam(4.0*m[c0:c1+1]+0.012*((yy-mid)/halfGap)**2,step=7))",
      "cuts.append(c0+LeastCostSeam(4.0*m[c0:c1+1]+%s*((yy-mid)/halfGap)**2,step=7))" % v)],
  task="page_segment", params=PAGE_RUN, panel=SEAM_PANEL,
  metrics=["nLines","charAcc","pitch"],
  trendMetrics=["charAcc","nLines"],
  source=r"A regularising term added to a seam cost to keep the path near a "
         r"prior is the usual way to stop a minimum-cost path degenerating "
         r"\cite{avidan2007,arvanitopoulos2014}.",
  reason=r"Between two lines there is usually a wide band of clean paper, and "
         r"every path through it costs the same. Without a tie-breaker the seam "
         r"is then decided by floating-point accidents and can hug one line for "
         r"a whole page. The quadratic term makes the midline the cheapest of "
         r"those equal-cost paths while remaining negligible against real ink. It "
         r"is quadratic, not linear, so it is almost free near the middle and "
         r"grows only as the seam approaches a line.",
  understanding=r"At $0$ the seam is unconstrained in open paper. Raised far "
                r"enough it dominates the ink term and the seam becomes a "
                r"straight line through the midpoint, cutting descenders: at "
                r"$0.25$ the full-drift penalty is $0.25$ against $4.0$ for one "
                r"ink pixel, so the seam will cross ink rather than drift more "
                r"than about $16$ pixel-rows.",
  alternatives=r"Clamping the seam to a corridor around the midline, which is "
               r"harder in that it forbids the large excursion a long descender "
               r"needs, or no tie-breaker at all.",
  worked=r"At the half-gap, $((y-\text{mid})/\text{halfGap})^2 = 1$ and the "
         r"penalty is $0.012$, which is $0.3\%$ of the cost of one ink pixel.",
  ),

C(key="seam_step",
  title="Rows a seam may move between adjacent columns",
  symbol=r"$\Delta_{\max}$",
  subsystem="Seam cutting",
  module="segment", where="SeamCuts",
  default=7, values=[1,3,7,12,20],
  patches=lambda v: [("segment",
      "cuts.append(c0+LeastCostSeam(4.0*m[c0:c1+1]+0.012*((yy-mid)/halfGap)**2,step=7))",
      "cuts.append(c0+LeastCostSeam(4.0*m[c0:c1+1]+0.012*((yy-mid)/halfGap)**2,step=%s))" % v)],
  task="page_segment", params=PAGE_RUN, panel=SEAM_PANEL,
  metrics=["nLines","charAcc","seconds"],
  trendMetrics=["charAcc","seconds"],
  source=r"The connectivity of the seam, how far it may shift per column, is a "
         r"parameter of seam carving \cite[10:3]{avidan2007}; the classical "
         r"version allows one row per column.",
  reason=r"A seam that may only move one row per column, as in the original seam "
         r"carving, cannot climb around a long descender within the width of one "
         r"letter: it needs the gradient to be gentle. Handwritten descenders are "
         r"steep, so the step is widened to seven rows. The cost of that freedom "
         r"is paid twice: the search is $O(W\,h\,(2\Delta+1))$, so time grows "
         r"linearly in $\Delta$, and a large step lets the seam jump so abruptly "
         r"that it no longer looks like a cut between two lines.",
  understanding=r"$\Delta_{\max}$ interacts with the straightness penalty: each "
                r"row of movement costs $0.02$, so the largest move costs "
                r"$0.02\Delta$. At $\Delta = 7$ that is $0.14$, still cheap "
                r"against ink, so the limit rather than the penalty is what binds.",
  alternatives=r"Four-connectivity with an unlimited number of vertical moves per "
               r"column, which needs a different dynamic program, or cutting on a "
               r"downsampled page and upsampling the seam.",
  worked=r"With $\Delta = 7$ the seam can cross a $51$ px pitch in eight "
         r"columns, about a third of an x-height of horizontal travel, which is "
         r"enough to pass round the stem of a \texttt{g}.",
  ),

C(key="seam_tail_radius",
  title="Closing radius applied before the seam is cut",
  symbol=r"$r_{\text{tail}}$",
  subsystem="Seam cutting",
  module="segment", where="SeamCuts",
  default=0.07, values=[0.0,0.03,0.07,0.14,0.25],
  patches=lambda v: [("segment","tailR = max(2,int(0.07*P))",
                      "tailR = max(2,int(%s*P))" % v)],
  task="page_segment", params=PAGE_RUN, panel=SEAM_PANEL,
  metrics=["nLines","charAcc","pitch"],
  trendMetrics=["charAcc","nLines"],
  source=r"Morphological closing to suppress thin spurious passages before a "
         r"path search \cite{haralick1987}.",
  reason=r"A minimum-cost seam will find any gap, including the one-pixel channel "
         r"between the two sides of a narrow loop or beside a stem. Threading "
         r"such a channel splits a letter in half for no benefit. The mask is "
         r"therefore closed before the cost field is built, which seals channels "
         r"narrower than $r_{\text{tail}}$ while leaving the outer shape of the "
         r"stroke close to real. It is a closing and not a plain dilation exactly "
         r"so that the stroke does not grow.",
  understanding=r"$r_{\text{tail}}$ is in line pitches, so it scales with the "
                r"writing. At $0$ the seam is free to thread every gap; too "
                r"large and neighbouring letters, or the descender of one line "
                r"and the ascender of the next, are welded together, so the seam "
                r"can no longer pass between them at all and the two lines merge.",
  alternatives=r"Penalising the seam for sharp vertical excursions, which the "
               r"straightness term already does weakly, or post-checking each "
               r"seam for how much ink it crossed.",
  worked=r"At $P = 51$ px, $0.07P = 3.57 \to 3$, so a $3$-pixel closing: "
         r"channels up to about $6$ px wide are sealed before the search.",
  ),

C(key="band_ink_share",
  title="Share of a piece's ink that must lie in a band for it to count",
  symbol=r"$\theta_{\text{band}}$",
  subsystem="Line ownership",
  module="segment", where="_BandOwnership",
  default=0.15, values=[0.05,0.10,0.15,0.25,0.40],
  patches=lambda v: [("segment",
      "substantial = cnt>=0.15*np.maximum(cnt.sum(1,keepdims=True),1)",
      "substantial = cnt>=%s*np.maximum(cnt.sum(1,keepdims=True),1)" % v)],
  task="page_segment", params=PAGE_RUN, panel=CROP_PANEL,
  metrics=["nLines","charAcc","nLinesTruth"],
  trendMetrics=["charAcc","nLines"],
  source=r"Assigning a connected component to the text line holding most of it is "
         r"the usual resolution step after a coarse line cut "
         r"\cite{likforman2007}.",
  reason=r"After the seams are cut, each piece of ink is scored by how much of it "
         r"falls in each band. A piece is treated as genuinely present in a band "
         r"if at least $\theta_{\text{band}}$ of its ink is there. The share "
         r"must be low enough that a real descender, which puts only its tail "
         r"into the band below, is still seen in both, and high enough that a few "
         r"pixels of anti-aliased overlap do not count. A descender tail is "
         r"typically a fifth to a third of the letter; a spurious overlap is a "
         r"few per cent. $0.15$ lies between them.",
  understanding=r"Where a piece is substantial in more than one band it is given "
                r"to the \emph{topmost}, which is the rule that keeps a letter "
                r"with the line it was written on rather than the line its tail "
                r"reaches into. Lowering $\theta_{\text{band}}$ makes more pieces "
                r"look multi-band, and because of the topmost rule that pulls ink "
                r"upward; raising it makes a genuine two-line piece look "
                r"single-band and it is then assigned by majority instead.",
  alternatives=r"Majority ownership alone, which is the \texttt{cnt.argmax} "
               r"fallback already used when nothing is substantial, or assigning "
               r"by the centroid of the piece, which fails for a long descender.",
  worked=r"A piece with $300$ ink pixels is counted as present in a band once at "
         r"least $45$ of them fall inside it.",
  ),

C(key="crop_pad",
  title="Padding kept around a line crop",
  symbol=r"$p_{\text{crop}}$",
  subsystem="Crop building",
  module="segment", where="_BuildLineCrop",
  default=0.15, values=[0.0,0.07,0.15,0.30,0.50],
  patches=lambda v: [("segment","pad = int(0.15*P)","pad = int(%s*P)" % v)],
  task="page_segment", params=PAGE_RUN, panel=CROP_PANEL,
  metrics=["charAcc","nLines"],
  trendMetrics=["charAcc"],
  source=r"Leaving a margin around a text line before recognition is standard "
         r"practice for CRNN line recognisers, which are trained on padded line "
         r"images \cite{shi2017,puigcerver2017}.",
  reason=r"The crop is cut to the bounding box of the ink owned by the line and "
         r"then padded. Some padding is needed because the networks were trained "
         r"on line images with a margin, and because a stroke that touches the "
         r"edge of a crop is cut off by the resize. The padding is also what "
         r"defines the left margin every crop is shifted to, so that the reader "
         r"always sees writing start in the same place. Too much padding shrinks "
         r"the writing once the crop is fitted to the fixed $64\times640$ canvas, "
         r"because the fit preserves aspect ratio.",
  understanding=r"On the fixed canvas the scale is "
                r"$\min(64/h,\,640/w)$, so padding costs resolution directly: "
                r"padding a $156$ px tall crop by $0.5P = 25$ px on each side "
                r"makes it $206$ px tall and shrinks the writing by $24\%$ "
                r"before the network sees it.",
  alternatives=r"No padding and relying on the canvas fit alone, or padding in "
               r"x-heights rather than line pitches, which would be more faithful "
               r"to the writing but needs the band measured first.",
  worked=r"At $P = 51$ px (half resolution) the padding is $7$ px at half size, "
         r"$14$ px on the full-resolution crop the writing is finally cut from.",
  ),

C(key="darkness_gamma",
  title="Darkness exponent applied to a finished line crop",
  symbol=r"$\gamma$",
  subsystem="Crop building",
  module="segment", where="_BuildLineCrop",
  default=0.55, values=[0.30,0.42,0.55,0.75,1.00],
  patches=lambda v: [("segment",
      "darkness = np.clip((255.0-linear)/255.0,0,1)**0.55",
      "darkness = np.clip((255.0-linear)/255.0,0,1)**%s" % v)],
  task="page_segment", params=PAGE_RUN, panel=CROP_PANEL,
  metrics=["charAcc","inkPct"],
  trendMetrics=["charAcc"],
  source=r"A power law on intensity is gamma correction, the standard tone "
         r"transfer of image display \cite{itu601}; applying it to deepen faint "
         r"strokes before recognition is contrast enhancement in the sense of "
         r"\cite{haralick1987}.",
  reason=r"The crop is first stretched linearly between the typical ink level "
         r"(the $40$th percentile of pixels under the mask) and the paper level "
         r"(the $75$th percentile of the region). That alone leaves a ballpoint "
         r"stroke that skipped as a pale grey smear. Raising darkness to a power "
         r"below one lifts the mid-tones without touching black or white: "
         r"$0$ and $1$ are fixed points of $d^{\gamma}$, so the ink stays ink and "
         r"the paper stays paper while everything between is darkened. The value "
         r"has to deepen a faint stroke without turning paper texture into ink, "
         r"and $0.55$ roughly doubles a mid-grey: $0.5^{0.55} = 0.68$.",
  understanding=r"$\gamma = 1$ is no correction at all, which is the honest "
                r"bottom of the sweep. As $\gamma$ falls, the writing thickens "
                r"and faint strokes join up; past a point the paper grain in the "
                r"halo also darkens and the stroke edges smear. Because the "
                r"networks were trained on crops produced with $\gamma = 0.55$, "
                r"moving it is a test-time mismatch as well as a visual change.",
  alternatives=r"Histogram equalisation, which is data-dependent and so varies "
               r"between crops of the same page, or CLAHE, which adds a tile size "
               r"and a clip limit of its own.",
  worked=r"A pixel at $60\%$ darkness becomes $0.6^{0.55} = 0.76$, so a stroke "
         r"at grey level $102$ is redrawn at $61$, clearly darker but not black.",
  ),

C(key="crop_black_pct",
  title="Percentile of masked pixels taken as the ink level",
  symbol=r"$q_{\text{ink}}$",
  subsystem="Crop building",
  module="segment", where="_BuildLineCrop",
  default=40, values=[10,25,40,60,80],
  patches=lambda v: [("segment",
      "black = np.percentile(inkPix,40) if len(inkPix) else 0.0",
      "black = np.percentile(inkPix,%s) if len(inkPix) else 0.0" % v)],
  task="page_segment", params=PAGE_RUN, panel=CROP_PANEL,
  metrics=["charAcc","inkPct"],
  trendMetrics=["charAcc"],
  source=r"Percentile-based black and white points are the robust form of "
         r"contrast stretching, preferred to the extremes because a single "
         r"outlier sets those \cite{haralick1987}.",
  reason=r"The linear stretch needs a black point. Using the darkest pixel would "
         r"hand the whole stretch to one noise pixel or one blot, so a percentile "
         r"of the pixels under the stroke mask is used instead. It is taken below "
         r"the median on purpose: the mask includes the soft edges of the stroke, "
         r"which are lighter than the core, so the median of masked pixels is "
         r"already greyer than real ink. The $40$th percentile sits inside the "
         r"dark core while still being a robust statistic over many pixels.",
  understanding=r"Raising $q_{\text{ink}}$ takes a lighter pixel as black, so the "
                r"stretch is stronger and the stroke is driven to full black with "
                r"its edges, thickening the writing. Lowering it takes a darker "
                r"reference, so the stretch is weaker and the crop stays flat.",
  alternatives=r"The mean under the mask, which the stroke edges drag up, or "
               r"Otsu's threshold applied again to the masked pixels, which is "
               r"another criterion where a percentile suffices.",
  worked=r"On the probe crop the paper point is the $75$th percentile of the "
         r"whole region and the ink point the $40$th of the masked pixels; the "
         r"gain is $255/(\text{white}-\text{black})$.",
  ),

C(key="solid_abs_pct",
  title="Ink share above which a crop is a solid block, not writing",
  symbol=r"$\pi_{\max}$",
  subsystem="Crop rejection",
  module="segment", where="DropSolidCrops",
  default=15.0, values=[5.0,10.0,15.0,25.0,40.0],
  patches=lambda v: [("segment",
      "def DropSolidCrops(lines,maxAbsPct=15.0,maxMedianRatio=3.0):",
      "def DropSolidCrops(lines,maxAbsPct=%s,maxMedianRatio=3.0):" % v)],
  task="page_segment", params=PAGE_RUN, panel=SEAM_PANEL,
  metrics=["nLines","dropped","charAcc","nLinesTruth"],
  trendMetrics=["dropped","nLines"],
  source=r"Rejecting non-text regions on an ink-density statistic before "
         r"recognition is a standard page-analysis filter \cite{shafait2008}.",
  reason=r"A crop that is a quarter black is not a line of handwriting; it is a "
         r"shadow, a photograph of an object on the page, or a region where the "
         r"illumination normalisation failed. Ordinary handwritten crops on these "
         r"pages measure $3$--$9\%$ ink. The test is deliberately a conjunction "
         r"of two conditions, an absolute share \emph{and} a multiple of the "
         r"page's own median, so that a page written unusually heavily is not "
         r"thrown away wholesale. $15\%$ is above every genuine line measured on "
         r"the eleven writers.",
  understanding=r"Because both conditions must hold, lowering $\pi_{\max}$ alone "
                r"does not immediately delete good lines: the median test still "
                r"protects them unless they are also three times the page median. "
                r"The function also refuses to return an empty list, so a page "
                r"where every crop trips the test keeps all of them.",
  alternatives=r"Scoring each crop with the reader and dropping the ones it "
               r"cannot transcribe, which is far more expensive and circular if "
               r"the crops are what the reader is being judged on.",
  worked=r"On the probe page the crops measure a median of a few per cent ink and "
         r"none is dropped at the submitted setting; the sweep records the count "
         r"dropped as the limit is lowered.",
  ),

C(key="solid_median_ratio",
  title="Multiple of the page median ink a crop may not exceed",
  symbol=r"$\rho_{\max}$",
  subsystem="Crop rejection",
  module="segment", where="DropSolidCrops",
  default=3.0, values=[1.2,2.0,3.0,5.0,10.0],
  patches=lambda v: [("segment",
      "def DropSolidCrops(lines,maxAbsPct=15.0,maxMedianRatio=3.0):",
      "def DropSolidCrops(lines,maxAbsPct=15.0,maxMedianRatio=%s):" % v)],
  task="page_segment", params=PAGE_RUN, panel=SEAM_PANEL,
  metrics=["nLines","dropped","charAcc","nLinesTruth"],
  trendMetrics=["dropped","nLines"],
  source=r"A robust relative outlier test against the median of the same page "
         r"\cite{shafait2008}; the median is used rather than the mean because "
         r"the outlier being tested for would itself move the mean.",
  reason=r"The second half of the solid-crop test asks whether this crop is "
         r"anomalous \emph{for this page}. Lines of the same writing on the same "
         r"page have similar ink density, within roughly a factor of two between "
         r"a sparse last line and a full one. A factor of three is outside that "
         r"natural spread, so a crop exceeding it is unlike the rest of the page "
         r"rather than merely heavy.",
  understanding=r"The conjunction with $\pi_{\max}$ is what makes the pair safe: "
                r"$\rho_{\max}$ catches a page-specific anomaly and $\pi_{\max}$ "
                r"stops a page of very light writing from having its densest "
                r"normal line called an anomaly. At $\rho_{\max} = 1.2$ almost "
                r"any variation looks anomalous and the absolute limit is the "
                r"only thing preventing good crops being lost.",
  alternatives=r"A median absolute deviation test, which is the same idea with a "
               r"scale estimate instead of a fixed multiple, or an absolute limit "
               r"alone, which is what $\pi_{\max}$ would become.",
  worked=r"If the page median is $5\%$ ink, a crop must be over $15\%$ "
         r"\emph{and} over $15\%$ to be dropped, so the two limits coincide; at a "
         r"median of $3\%$ the median test binds first at $9\%$, but the absolute "
         r"test then spares the crop.",
  ),

C(key="min_ink_frac",
  title="Ink a band must hold before it becomes a line crop",
  symbol=r"$I_{\min}$",
  subsystem="Crop rejection",
  module="segment", where="SegmentLines",
  default=0.12, values=[0.02,0.06,0.12,0.25,0.45],
  patches=lambda v: [("segment",
      "minInk = max(60,0.12*np.median(perLine[perLine>0]))",
      "minInk = max(60,%s*np.median(perLine[perLine>0]))" % v)],
  task="page_segment", params=PAGE_RUN, panel=SEAM_PANEL,
  metrics=["nLines","nLinesTruth","charAcc"],
  trendMetrics=["nLines","charAcc"],
  source=r"A minimum-content test before emitting a text line is part of every "
         r"profile-based segmenter \cite{likforman2007}.",
  reason=r"Some bands between seams hold almost nothing: the strip above the "
         r"first line, the strip below the last, or a band created by a spurious "
         r"profile peak. Rather than a fixed pixel count, which depends on "
         r"photograph size, the limit is a fraction of the median ink per line on "
         r"the same page. A genuine short line, even one word, still carries well "
         r"over a tenth of a full line's ink, while an empty band carries almost "
         r"none, so $0.12$ sits in a wide gap. The absolute floor of $60$ pixels "
         r"covers the case where the median itself is tiny.",
  understanding=r"Raising $I_{\min}$ deletes genuine short lines, and as with the "
                r"peak floor there is no later stage that can restore one. "
                r"Lowering it emits near-empty crops, which the reader then "
                r"transcribes as noise and which pollute a style profile with "
                r"fragments.",
  worked=r"If the median band holds $9000$ owned pixels, a band must hold at "
         r"least $1080$ to become a crop.",
  ),

]
