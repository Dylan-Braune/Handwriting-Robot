#u23539195 coefficient justification: the image-processing and line-finding coefficients
#one entry per coefficient: where it lives, what it is, why that value, what else
#was possible, a worked number, and the five values the sweep draws
import probes as P

#shorthand so an entry reads as one block
def C(**kw):
    return kw


ENTRIES = [

C(key="otsu_hi",
  title="Upper edge of the histogram Otsu is built on, for a ratio image",
  symbol=r"$r_{\max}$",
  subsystem="Image toolkit",
  module="segment", where="Otsu",
  default=1.2, values=[1.0,1.1,1.2,1.4,1.6],
  patches=lambda v: [("segment","hi = 1.2 if hi is None else float(hi)",
                      "hi = %s if hi is None else float(hi)" % v)],
  task="line_ink",
  params=dict(page=P.PAGE,lineIdx=P.LINE),
  panel=dict(kind="image",key="ink",binary=True,aspect=7.0),
  metrics=["inkPixels","components","xh"],
  source=r"Otsu's between-class variance criterion \cite[p.~63]{otsu1979} is "
         r"defined over a discrete histogram; the number of bins and their range "
         r"are left to the implementer.",
  reason=r"Otsu is applied twice in this project: once to 8-bit greys, where the "
         r"histogram is exactly the 256 integer levels and nothing has to be "
         r"chosen, and once to the normalised ratio image $g/\mathrm{BoxMean}(g)$ "
         r"of \texttt{InkMask}, which is a float. Paper divided by its own local "
         r"mean sits just above $1.0$; ink sits well below it. The range therefore "
         r"has to reach a little past unity so the paper peak is inside the "
         r"histogram and not piled into the last bin, which would drag the "
         r"threshold up. $1.2$ places the paper peak at bin $213$ of $256$, "
         r"leaving $43$ bins above it for shading and glare.",
  understanding=r"$r_{\max}$ only sets the quantisation of the search, not the "
                r"criterion. Each bin is $r_{\max}/256$ wide, so the threshold is "
                r"returned to a resolution of $0.0047$ at $r_{\max}=1.2$. Raising "
                r"$r_{\max}$ coarsens every bin; lowering it clips the paper peak.",
  alternatives=r"A fixed global threshold (no criterion at all), Niblack "
               r"\cite{niblack1986} or Sauvola \cite{sauvola2000} local "
               r"thresholding, which need a window and a $k$ of their own, or "
               r"clipping at the observed maximum of the ratio, which makes the "
               r"bin width depend on the brightest speck of glare on the page.",
  worked=r"On the probe page the ratio image has $99.9$th percentile $1.08$, so "
         r"at $r_{\max}=1.2$ every pixel lands inside the histogram and the "
         r"returned threshold is a bin centre $(t+0.5)\,r_{\max}/256$.",
  ),

C(key="binarize_thin",
  title="Width below which a run of ink counts as thin, for printed-rule removal",
  symbol=r"$w_{\text{thin}}$",
  subsystem="Line binarisation",
  module="segment", where="BinarizeLine",
  default=0.12, values=[0.06,0.09,0.12,0.18,0.24],
  patches=lambda v: [("segment","thin = vRun<=max(3,int(0.12*h))",
                      "thin = vRun<=max(3,int(%s*h))" % v)],
  task="line_ink",
  params=dict(page=P.PAGE,lineIdx=P.LINE),
  panel=dict(kind="image",key="ink",binary=True,aspect=7.0),
  metrics=["inkPixels","components","slantDeg"],
  source=r"Removing long thin runs is the standard morphological test for printed "
         r"rules and table lines in document analysis \cite{haralick1987}; the "
         r"ruling on a form is the structure Shafait \textit{et al.} "
         r"\cite{shafait2008} evaluate layout methods against.",
  reason=r"A printed rule is the one mark on these pages that is simultaneously "
         r"very long horizontally and very short vertically. The test is the "
         r"conjunction of both, so the thinness limit only has to separate a "
         r"printed line from a pen stroke. Measured pen strokes on the "
         r"photographed pages run $4$--$9$ px thick at a crop height of "
         r"$156$ px, that is $0.026$--$0.058\,h$; the printed rule is $1$--$3$ px. "
         r"$0.12\,h$ sits above the rule and below the thinnest real stroke, "
         r"with the floor $\max(3,\cdot)$ protecting small crops.",
  understanding=r"$w_{\text{thin}}$ is a fraction of the crop height $h$, so it "
                r"scales with how big the writing was photographed. Too small and "
                r"a thick printed rule survives; too large and a horizontal pen "
                r"stroke, such as the bar of a \texttt{t} or the body of an "
                r"\texttt{e}, is deleted as structure.",
  alternatives=r"A Hough transform for straight lines \cite{duda1972}, which "
               r"finds rules at any angle but costs an accumulator per crop and "
               r"needs its own peak threshold, or projection-profile rule "
               r"detection, which is what \texttt{train.py} uses on whole pages "
               r"in \texttt{FindRuleLineClusters}.",
  worked=r"At $h=156$ px, $0.12\,h = 18.7 \to 18$ px, and the length test is "
         r"$\max(24,\lfloor 1.1h \rfloor) = 171$ px. A mark is deleted only if it "
         r"is at least $171$ px long and at most $18$ px tall.",
  ),

C(key="binarize_rulelen",
  title="Length above which a run of ink counts as a printed rule",
  symbol=r"$L_{\text{rule}}$",
  subsystem="Line binarisation",
  module="segment", where="BinarizeLine",
  default=1.1, values=[0.6,0.85,1.1,1.5,2.0],
  patches=lambda v: [("segment","rule = (hRun>=max(24,int(1.1*h)))&thin",
                      "rule = (hRun>=max(24,int(%s*h)))&thin" % v)],
  task="line_ink",
  params=dict(page=P.PAGE,lineIdx=P.LINE),
  panel=dict(kind="image",key="ink",binary=True,aspect=7.0),
  metrics=["inkPixels","components","slantDeg"],
  source=r"Same basis as the thinness test: long-run structure removal "
         r"\cite{haralick1987,shafait2008}.",
  reason=r"The rule on a ruled page runs the full width of the writing area, "
         r"which is many times the height of one line of writing. Expressing the "
         r"limit in crop heights makes it scale-free: a rule is longer than the "
         r"line is tall, a pen stroke is not. $1.1\,h$ is just over one line "
         r"height, which no single handwritten stroke on these pages reaches "
         r"horizontally while staying under $0.12\,h$ thick.",
  understanding=r"Lowering $L_{\text{rule}}$ makes the test fire on ordinary "
                r"writing, because a joined cursive word can be several line "
                r"heights long: only the thinness test then stands between the "
                r"writing and deletion. Raising it lets a short rule segment, or a "
                r"rule broken by the writing crossing it, survive into the mask.",
  alternatives=r"An absolute pixel length, which breaks as soon as the page is "
               r"photographed at a different distance, or connected-component "
               r"aspect ratio, which a long joined word also satisfies.",
  worked=r"At $h=156$ px the limit is $171$ px, which is $11\%$ of the "
         r"$2170$ px crop width, so a rule has to cross about a ninth of the "
         r"writing area before it is removed.",
  ),

C(key="binarize_close",
  title="Closing radius that bridges pinholes in a stroke",
  symbol=r"$k_{\text{close}}$",
  subsystem="Line binarisation",
  module="segment", where="BinarizeLine",
  default=0.045, values=[0.0,0.02,0.045,0.08,0.12],
  patches=lambda v: [("segment","k = int(np.clip(round(0.045*h),2,4))",
                      "k = int(np.clip(round(%s*h),2,4))" % v)],
  task="line_ink",
  params=dict(page=P.PAGE,lineIdx=P.LINE),
  panel=dict(kind="image",key="ink",binary=True,aspect=7.0),
  metrics=["inkPixels","components","xh"],
  source=r"Morphological closing, dilation followed by erosion by the same "
         r"element, is the classical repair for small holes and gaps and does not "
         r"grow the shape overall \cite{haralick1987}.",
  reason=r"A ballpoint on ordinary paper skips, so a single stroke binarises into "
         r"a chain of blobs. Those breaks matter twice over: they split one "
         r"component into several, and they put spurious endpoints into the "
         r"skeleton, which the tracer then follows as separate strokes. The "
         r"radius has to close a skip without fattening the stroke, so it is tied "
         r"to the crop height and then hard-clipped to $2$--$4$ px: the clip, not "
         r"the fraction, is what finally bounds the damage.",
  understanding=r"Because of the $\mathrm{clip}(\cdot,2,4)$ the coefficient only "
                r"has an effect between $h=44$ and $h=89$ px; outside that range "
                r"the clip decides. On the probe crop ($h=156$) every value from "
                r"$0.02$ upward gives $k=4$, which is why the sweep includes $0$: "
                r"that is the only setting that changes the picture, and it shows "
                r"what closing is buying.",
  alternatives=r"Not closing at all and letting the tracer rejoin the pieces at "
               r"junctions, which \texttt{\_JoinAtJunctions} does anyway; or a "
               r"larger closing, which merges adjacent letters and destroys the "
               r"character boundaries the glyph cutter depends on.",
  worked=r"At $h=156$ px, $0.045\,h = 7.0$, clipped to $4$, so a $4\times4$ "
         r"closing is applied: gaps up to about $4$ px of white inside a stroke "
         r"are filled.",
  ),

C(key="speck_area",
  title="Smallest component kept in a line mask",
  symbol=r"$A_{\min}$",
  subsystem="Line binarisation",
  module="segment", where="BinarizeLine",
  default=6, values=[1,3,6,12,24],
  patches=lambda v: [("segment","keep = areas>=6","keep = areas>=%s" % v)],
  task="line_ink",
  params=dict(page=P.PAGE,lineIdx=P.LINE),
  panel=dict(kind="image",key="ink",binary=True,aspect=7.0),
  metrics=["inkPixels","components"],
  source=r"Area opening, discarding components below a size, is the standard "
         r"speck filter after thresholding \cite{haralick1987}; components are "
         r"labelled by the two-pass union-find of Rosenfeld and Pfaltz "
         r"\cite{rosenfeld1966} with path compression after Tarjan "
         r"\cite{tarjan1975}.",
  reason=r"Thresholding a photograph leaves single-pixel noise from sensor grain "
         r"and paper texture. The smallest mark that must survive is the dot of "
         r"an \texttt{i} or a full stop, which on these crops is $20$--$60$ px. "
         r"$A_{\min}=6$ is an order of magnitude below the smallest real mark and "
         r"an order above single-pixel grain, so it is deliberately not a tuned "
         r"value: it sits in the gap.",
  understanding=r"Raising it eventually eats punctuation and \texttt{i} dots, "
                r"which the glyph cutter then has to recover separately in "
                r"\texttt{ExtractLineGlyphs}. Lowering it to $1$ keeps every "
                r"speck, which inflates the component count the ownership stage "
                r"has to reason about.",
  alternatives=r"A median or morphological opening before labelling, which also "
               r"rounds the stroke ends, or keeping everything and letting the "
               r"later size tests in \texttt{\_BandOwnership} reject it, which is "
               r"what happens for anything above $6$ px anyway.",
  worked=r"On the probe crop the mask holds $41$ components at $A_{\min}=6$; the "
         r"sweep records how that count moves with the limit.",
  ),

C(key="core_band_frac",
  title="Profile height defining the x-height band",
  symbol=r"$\tau_{\text{band}}$",
  subsystem="Line geometry",
  module="segment", where="CoreBand",
  default=0.45, values=[0.25,0.35,0.45,0.60,0.75],
  patches=lambda v: [("segment","rows = np.nonzero(prof>=0.45*prof.max())[0]",
                      "rows = np.nonzero(prof>=%s*prof.max())[0]" % v)],
  task="line_ink",
  params=dict(page=P.PAGE,lineIdx=P.LINE),
  panel=dict(kind="band",key="ink",binary=True,aspect=7.0),
  metrics=["xh","coreBand"],
  source=r"The horizontal projection profile of a text line peaks across the "
         r"x-height body and falls away over the ascender and descender zones; "
         r"this is the basis of profile-based line and zone finding "
         r"\cite{likforman2007}, and the same profile drives the IAM baseline "
         r"work \cite{marti2002}.",
  reason=r"Every shape in a style profile is stored in x-heights, so the x-height "
         r"is the one length the whole synthesis side depends on. It is measured "
         r"as the band of rows carrying at least $\tau_{\text{band}}$ of the peak "
         r"ink-per-row. The body of the writing contributes nearly every row of "
         r"ink; ascenders and descenders contribute only the few columns that "
         r"have them. The profile therefore has a plateau and a skirt, and "
         r"$\tau_{\text{band}}$ has to cut the skirt. At $0.45$ the cut is "
         r"roughly half the plateau height, which is above the ascender skirt on "
         r"all eleven writers and below the plateau edge on all of them.",
  understanding=r"This coefficient directly scales every stored glyph. If "
                r"$\tau_{\text{band}}$ is too low the band swallows the ascender "
                r"zone, the measured x-height grows, and every glyph is stored "
                r"too small, so the writing is later drawn too large for its "
                r"ruling. Too high and the band collapses onto the few darkest "
                r"rows. The guard \texttt{base-top>=3} rejects a collapsed band "
                r"outright rather than returning a bad number.",
  alternatives=r"Fitting four baselines explicitly, as Vinciarelli and Luettin do "
               r"\cite{vinciarelli2001}, which is more faithful but needs its own "
               r"model fit per line; or taking the median component height, which "
               r"\texttt{\_LetterHeight} does for line pitch but which mixes "
               r"ascenders into the estimate.",
  worked=r"On the probe crop the band is rows $36$--$78$, giving "
         r"$x_h = 42$ px at a crop height of $156$ px, so the x-height is "
         r"$27\%$ of the crop and the ascender and descender zones take the rest.",
  ),

C(key="inkmask_bgradius",
  title="Local background radius for the ink mask",
  symbol=r"$r_{\text{bg}}$",
  subsystem="Illumination normalisation",
  module="segment", where="InkMask",
  default=41, values=[15,25,41,70,120],
  patches=lambda v: [("segment","r = max(8,min(40,round(gray.shape[0]/41)))",
                      "r = max(8,min(40,round(gray.shape[0]/%s)))" % v)],
  task="page_segment",
  params=dict(page=P.PAGE,lines=[P.LINE],score=True,
              textWeights=P.TEXT_WEIGHTS,scoreLines=8),
  panel=dict(kind="seams",layout="cols",panelHeight=2.6),
  metrics=["nLines","charAcc","pitch","artifactPieces"],
  trendMetrics=["nLines","charAcc"],
  source=r"Dividing by a local mean computed from an integral image is Bradley "
         r"and Roth's adaptive threshold \cite{bradley2007}; the constant-time "
         r"rectangle sum it needs is Crow's summed-area table \cite{crow1984}.",
  reason=r"A phone photograph of a page is unevenly lit, so no single grey level "
         r"separates ink from paper across the whole sheet. Each pixel is "
         r"therefore compared with the mean of a window around it. The window has "
         r"to be much larger than a stroke, or it averages the ink into its own "
         r"background and the stroke centre is lost, and much smaller than the "
         r"illumination gradient, or it stops tracking the shading. Tying it to "
         r"$H/41$ makes it track how large the writing was photographed rather "
         r"than the sensor resolution, and the $\mathrm{clip}(8,40)$ keeps it "
         r"inside the range where both conditions hold.",
  understanding=r"At $H/41$ the window is about three x-heights across, so a "
                r"stroke occupies a few per cent of it. The clip means the "
                r"coefficient only bites for page heights between $328$ and "
                r"$1640$ px at half resolution; the probe page is $1632$ px tall "
                r"at half size, so it sits at the very top of that range and the "
                r"sweep moves the radius over most of its useful span.",
  alternatives=r"Sauvola's local mean and standard deviation \cite{sauvola2000} "
               r"or Niblack's \cite{niblack1986}, both of which add a sensitivity "
               r"constant; a morphological top-hat, which needs a structuring "
               r"element of the same scale; or global Otsu alone, which fails "
               r"outright on a shadowed corner.",
  worked=r"The probe page is $3264$ px tall, halved to $1632$, so "
         r"$1632/41 = 39.8 \to 40$, exactly at the clip. The window is "
         r"$81\times81$ px against an x-height of about $21$ px at half "
         r"resolution, that is $3.9$ x-heights wide.",
  ),

C(key="solid_block",
  title="Window fill above which a solid block is ruled out as writing",
  symbol=r"$\phi_{\text{solid}}$",
  subsystem="Illumination normalisation",
  module="segment", where="InkMask",
  default=0.999, values=[0.90,0.97,0.999,1.0,1.01],
  patches=lambda v: [("segment","solid = BoxMean(mask.astype(np.float64),4)>0.999",
                      "solid = BoxMean(mask.astype(np.float64),4)>%s" % v)],
  task="page_segment",
  params=dict(page=P.PAGE,lines=[P.LINE],score=True,
              textWeights=P.TEXT_WEIGHTS,scoreLines=8),
  panel=dict(kind="seams",layout="cols",panelHeight=2.6),
  metrics=["nLines","charAcc","inkOutsidePaper","artifactPieces"],
  source=r"Removing large filled regions before layout analysis is standard "
         r"pre-processing for photographed documents \cite{shafait2008}.",
  reason=r"Handwriting is thin. A $9\times9$ window that is completely full of "
         r"\emph{mask} is not a pen stroke at that resolution; it is a shadow, a "
         r"dark object in the photograph, or the edge of the sheet. The test is "
         r"deliberately set at \emph{completely} full rather than merely dark, "
         r"because the cost of deleting real writing is far higher than the cost "
         r"of leaving a shadow for the later artefact tests to catch. $0.999$ is "
         r"the floating-point way of writing \enquote{all $81$ pixels}, since "
         r"$80/81 = 0.9877$ already fails it.",
  understanding=r"The value is effectively a Boolean, not a tuned fraction: "
                r"anything in $(0.9877,1.0]$ behaves identically. The sweep makes "
                r"that explicit by including $0.90$ and $0.97$, which do admit "
                r"partially filled windows, and $1.01$, which can never fire and "
                r"so disables the test.",
  alternatives=r"An area-based test on labelled components, which is what "
               r"\texttt{DropArtifactPieces} applies later at page scale, or a "
               r"stroke-width transform, which would measure thickness directly "
               r"but costs a distance transform per page.",
  worked=r"$\mathrm{BoxMean}$ with $r=4$ averages $(2\cdot4+1)^2 = 81$ pixels, so "
         r"the possible values below $1$ are $k/81$; the largest is "
         r"$80/81 = 0.9877 < 0.999$.",
  ),

C(key="paper_dark",
  title="Fraction of the median brightness below which a block is off-sheet",
  symbol=r"$\tau_{\text{paper}}$",
  subsystem="Sheet detection",
  module="segment", where="PaperRegionMask",
  default=0.45, values=[0.25,0.35,0.45,0.60,0.75],
  patches=lambda v: [("segment","dark = c<0.45*np.median(c)",
                      "dark = c<%s*np.median(c)" % v)],
  task="page_segment",
  params=dict(page=P.PAGE,lines=[P.LINE],score=True,
              textWeights=P.TEXT_WEIGHTS,scoreLines=8),
  panel=dict(kind="seams",layout="cols",panelHeight=2.6),
  metrics=["paperCoverage","inkOutsidePaper","nLines","charAcc"],
  trendMetrics=["paperCoverage","nLines"],
  source=r"Finding the page as the largest bright connected region of a "
         r"block-averaged image follows the standard camera-document "
         r"pre-processing pipeline \cite{shafait2008}; the labelling is "
         r"\cite{rosenfeld1966}.",
  reason=r"The sheet is the brightest large object in the photograph and the "
         r"desk around it is darker. Comparing against the \emph{median} of the "
         r"block-averaged image rather than a fixed grey makes the test "
         r"independent of exposure: the median of a photograph that is mostly "
         r"paper is the paper level. A desk, a shadow or the edge of the frame is "
         r"well under half the paper level, while shading on the sheet itself "
         r"rarely falls below $0.6$ of it, so $0.45$ separates the two with "
         r"margin on both sides.",
  understanding=r"The test runs on a $4\times4$ block average, so it is a "
                r"judgement about regions, not pixels, and the writing itself "
                r"does not drag a block under the limit. Raising "
                r"$\tau_{\text{paper}}$ eats into the sheet and can clip a whole "
                r"margin of writing; lowering it lets the desk in, and the "
                r"largest bright blob then spans the sheet and its surroundings.",
  alternatives=r"Edge detection and a quadrilateral fit \cite{canny1986} to find "
               r"the sheet corners, which is more precise but fails when a corner "
               r"is out of frame; or assuming the whole frame is paper, which the "
               r"guard \texttt{dark.mean()<0.002} already falls back to when "
               r"almost nothing is dark.",
  worked=r"On the probe page the sheet covers $94.7\%$ of the frame at the "
         r"submitted setting and $4726$ mask pixels lie outside it and are "
         r"removed.",
  ),

C(key="axis_turn_margin",
  title="How much better the sideways reading must be before the page is turned",
  symbol=r"$m_{\text{axis}}$",
  subsystem="Page orientation",
  module="segment", where="UprightPage",
  default=1.35, values=[1.00,1.15,1.35,1.60,2.00],
  patches=lambda v: [("segment","AXIS_TURN_MARGIN = 1.35",
                      "AXIS_TURN_MARGIN = %s" % v)],
  task="page_segment",
  params=dict(page=P.PAGE,lines=[P.LINE],score=True,
              textWeights=P.TEXT_WEIGHTS,scoreLines=8),
  panel=dict(kind="seams",layout="cols",panelHeight=2.6),
  metrics=["rotation","axisScore0","axisScore90","nLines","charAcc"],
  source=r"Lines of text produce a strongly periodic projection profile along one "
         r"axis only, which is the signal every profile-based layout method uses "
         r"\cite{likforman2007,shafait2008}.",
  reason=r"The score is the fraction of rows that are nearly empty, which is high "
         r"across the gaps between lines and low along them. Both orientations "
         r"are scored and the page is turned only if the sideways score is "
         r"clearly better. A plain \enquote{greater than} would turn a page "
         r"whenever noise put the two scores a hair apart, and turning a page "
         r"wrongly costs the whole job, while failing to turn one the user "
         r"photographed correctly costs nothing. The margin is therefore "
         r"asymmetric on purpose, and $1.35$ is far outside the measured spread "
         r"between the two scores on a correctly held page.",
  understanding=r"$m_{\text{axis}}$ is a ratio, so it is unaffected by how much "
                r"ink is on the page. At $1.0$ the test becomes a bare "
                r"comparison; above the true ratio of a genuinely sideways page "
                r"it can never fire and a rotated page is never corrected.",
  worked=r"On the probe page the two scores are $s_0 = 0.470$ and "
         r"$s_{90} = 0.075$, a ratio of $0.16$ in favour of leaving it alone, so "
         r"the page is not turned at any setting in the sweep. The decision has "
         r"$8.4\times$ margin against the $1.35$ limit.",
  notes=r"The probe page is upright, so this sweep demonstrates the stability of "
        r"the decision rather than the turning itself; the recorded scores show "
        r"how far the page is from the limit.",
  ),

C(key="peak_floor",
  title="Profile height a candidate line peak must reach",
  symbol=r"$\tau_{\text{peak}}$",
  subsystem="Line finding",
  module="segment", where="LineCentres",
  default=0.06, values=[0.02,0.04,0.06,0.12,0.20],
  patches=lambda v: [
      ("segment",
       "peaks = [i for i in range(1,H-1) if sm[i]>=sm[i-1] and sm[i]>sm[i+1] and sm[i]>0.06*sm.max()]",
       "peaks = [i for i in range(1,H-1) if sm[i]>=sm[i-1] and sm[i]>sm[i+1] and sm[i]>%s*sm.max()]" % v),
      ("segment",
       "peaks = [i for i in range(1,len(sm)-1) if sm[i]>=sm[i-1] and sm[i]>sm[i+1] and sm[i]>0.06*sm.max()]",
       "peaks = [i for i in range(1,len(sm)-1) if sm[i]>=sm[i-1] and sm[i]>sm[i+1] and sm[i]>%s*sm.max()]" % v)],
  task="page_segment",
  params=dict(page=P.PAGE,lines=[P.LINE],score=True,
              textWeights=P.TEXT_WEIGHTS,scoreLines=8),
  panel=dict(kind="seams",layout="cols",panelHeight=2.6),
  metrics=["nLines","nLinesTruth","pitch","charAcc"],
  trendMetrics=["nLines","charAcc"],
  source=r"Peak picking on a smoothed horizontal projection profile is the "
         r"classical line-finding method \cite{likforman2007}; the profile and "
         r"its use on handwritten pages follow \cite{marti2002}.",
  reason=r"Every local maximum of the smoothed ink-per-row profile is a candidate "
         r"line centre, but a short line, a stray mark or the tail of a descender "
         r"also makes a bump. The floor is relative to the tallest peak on the "
         r"page, so it does not depend on how much was written. It has to sit "
         r"below the weakest real line, which on these pages is a one- or "
         r"two-word last line carrying roughly a tenth of the ink of a full line, "
         r"and above the texture left after smoothing. $0.06$ is below that "
         r"weakest-line level with margin.",
  understanding=r"This floor controls how many candidates reach the separation "
                r"test that follows; it cannot by itself create a line, because "
                r"the $0.6P$ separation rule then thins the candidates out. "
                r"Raising it drops genuine short lines outright, which is "
                r"unrecoverable: there is no later stage that puts a line back.",
  alternatives=r"Prominence-based peak picking, which \texttt{train.py} does "
               r"implement in \texttt{FindPeaks} with an explicit prominence and "
               r"distance, or fitting a fixed pitch and taking every period, "
               r"which fails on a page whose last line is short.",
  worked=r"The probe page has $20$ transcript lines. The sweep records how many "
         r"the segmenter returns at each floor, against that known count.",
  ),

C(key="centre_separation",
  title="Smallest spacing between two accepted line centres",
  symbol=r"$s_{\min}$",
  subsystem="Line finding",
  module="segment", where="LineCentres",
  default=0.6, values=[0.3,0.45,0.6,0.8,1.0],
  patches=lambda v: [("segment","if all(abs(c-o)>=0.6*P for o in centres):",
                      "if all(abs(c-o)>=%s*P for o in centres):" % v)],
  task="page_segment",
  params=dict(page=P.PAGE,lines=[P.LINE],score=True,
              textWeights=P.TEXT_WEIGHTS,scoreLines=8),
  panel=dict(kind="seams",layout="cols",panelHeight=2.6),
  metrics=["nLines","nLinesTruth","pitch","charAcc"],
  trendMetrics=["nLines","charAcc"],
  source=r"Enforcing a minimum distance between accepted peaks is the usual "
         r"non-maximum suppression step of profile line finding "
         r"\cite{likforman2007}.",
  reason=r"Candidates are taken strongest first and a new one is accepted only if "
         r"it is at least $s_{\min}$ pitches from every centre already kept. The "
         r"limit is expressed in measured line pitch $P$, not pixels, so it "
         r"follows the page. Two real lines are one pitch apart by definition, so "
         r"anything below $1.0$ admits them both; the profile of a single line, "
         r"however, is double-humped when the writing has many ascenders, and "
         r"those humps sit a few tenths of a pitch apart. $0.6$ is above that "
         r"internal structure and safely below the one-pitch spacing of real "
         r"lines.",
  understanding=r"Taken with the peak floor this is the whole line-count "
                r"decision. Too small and one line is reported as two, which "
                r"splits words across crops; too large and two closely written "
                r"lines are merged into one crop, which the reader then cannot "
                r"transcribe. Because candidates are sorted by strength, the "
                r"rule keeps the strongest of a cluster rather than the first.",
  alternatives=r"Merging on the depth of the valley between two peaks, which this "
               r"code also implements as \texttt{LINE\_MERGE\_VALLEY} and leaves "
               r"disabled, or a fixed pitch comb.",
  worked=r"On the probe page $P = 51$ px at half resolution, so $0.6P = 30.6$ px: "
         r"two centres closer than $31$ px are treated as one line.",
  ),

C(key="line_merge_valley",
  title="Valley depth below which two line centres are merged",
  symbol=r"$\tau_{\text{valley}}$",
  subsystem="Line finding",
  module="segment", where="LineCentres",
  default=1.0, values=[0.3,0.5,0.7,0.9,1.0],
  patches=lambda v: [("segment","LINE_MERGE_VALLEY = 1.0",
                      "LINE_MERGE_VALLEY = %s" % v)],
  task="page_segment",
  params=dict(page=P.PAGE,lines=[P.LINE],score=True,
              textWeights=P.TEXT_WEIGHTS,scoreLines=8),
  panel=dict(kind="seams",layout="cols",panelHeight=2.6),
  metrics=["nLines","nLinesTruth","charAcc"],
  trendMetrics=["nLines","charAcc"],
  source=r"Merging profile peaks separated only by a shallow valley is the "
         r"standard refinement where projection peaks over-segment "
         r"\cite{likforman2007}.",
  reason=r"This is a second, optional merge: for each adjacent pair of centres "
         r"the lowest profile value between them is compared with the smaller of "
         r"the two peaks, and the pair is merged if the dip is shallow. The "
         r"submitted value is $1.0$, and the loop is guarded by "
         r"\texttt{thr<1.0}, so \textbf{the merge is switched off in the "
         r"submitted system}. That is the honest description of this "
         r"coefficient: it is a disabled mechanism, kept because "
         r"\texttt{SegmentLines} exposes \texttt{mergeValley} as an argument so a "
         r"difficult page can be re-run with it on.",
  understanding=r"The value is a ratio of ink depths, so $1.0$ means "
                r"\enquote{merge even when the valley is as high as the peak}, "
                r"which is why it is used as the off switch rather than a large "
                r"number. Lowering it to $0.7$ merges only genuinely shallow "
                r"dips.",
  alternatives=r"Leaving over-segmentation to the ownership stage, which is what "
               r"happens now, or topological persistence on the profile, which is "
               r"the same idea with a cleaner definition.",
  worked=r"With $\tau_{\text{valley}} = 1.0$ the \texttt{while} test "
         r"$1.0 < 1.0$ is false and no merge is ever attempted, so the sweep "
         r"shows what the project chose not to use.",
  ),

C(key="pitch_peak_floor_xh",
  title="Smallest peak gap counted as line spacing, in letter heights",
  symbol=r"$g_{\min}$",
  subsystem="Line finding",
  module="segment", where="LinePitch",
  default=1.5, values=[0.8,1.2,1.5,2.0,2.5],
  patches=lambda v: [("segment","PEAK_MIN_PITCH_XH = 1.5",
                      "PEAK_MIN_PITCH_XH = %s" % v)],
  task="page_segment",
  params=dict(page=P.PAGE,lines=[P.LINE],score=True,
              textWeights=P.TEXT_WEIGHTS,scoreLines=8),
  panel=dict(kind="seams",layout="cols",panelHeight=2.6),
  metrics=["pitch","nLines","charAcc"],
  trendMetrics=["pitch","nLines"],
  source=r"Line pitch from the median gap between profile peaks, with "
         r"autocorrelation as a fallback, follows standard profile analysis "
         r"\cite{likforman2007}.",
  reason=r"The median peak gap is only the line pitch if the peaks are different "
         r"lines. Within one line the profile has structure at the scale of a "
         r"letter, so a gap of about one letter height is internal and a gap of "
         r"about two is between lines. The floor is written in median letter "
         r"heights, measured independently by \texttt{\_LetterHeight}, and "
         r"$1.5$ sits between those two cases. If the median gap falls below it "
         r"the estimate is rejected and the autocorrelation fallback is used "
         r"instead, so the coefficient selects between two estimators rather than "
         r"setting the pitch itself.",
  understanding=r"Too low and intra-line structure is accepted as the pitch, "
                r"which makes $P$ far too small and every $P$-scaled test in the "
                r"segmenter too tight. Too high and the peak estimator is always "
                r"rejected in favour of autocorrelation, which is more robust but "
                r"blind to a page with only two or three lines.",
  worked=r"On the probe page the median letter height is such that the peak "
         r"estimate is accepted and $P = 51$ px at half resolution, against a "
         r"ruling of $8.4$ mm on the real page.",
  ),

]
