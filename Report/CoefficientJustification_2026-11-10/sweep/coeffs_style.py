#u23539195 coefficient justification: the tracing and glyph-cutting coefficients
import probes as P


def C(**kw):
    return kw


TRACE_RUN = dict(page=P.PAGE,lineIdx=P.LINE)
GLYPH_RUN = dict(page=P.PAGE,lineIdx=P.LINE,textWeights=P.TEXT_WEIGHTS)

GLYPH_PANEL = dict(kind="glyphs",maxGlyphs=12,panelHeight=0.95)
TRACE_PANEL = dict(kind="polys",aspect=6.0,panelHeight=0.72)


ENTRIES = [

C(key="join_angle",
  title="Largest turn two traced chains may make and still be rejoined",
  symbol=r"$\alpha_{\text{join}}$",
  subsystem="Centreline tracing",
  module="style", where="_JoinAtJunctions",
  default=75.0, values=[25.0,50.0,75.0,105.0,140.0],
  patches=lambda v: [("style","def _JoinAtJunctions(chains,maxAngleDeg=75.0):",
                      "def _JoinAtJunctions(chains,maxAngleDeg=%s):" % v)],
  task="trace", params=TRACE_RUN, panel=TRACE_PANEL,
  metrics=["nPolylines","totalLength","meanPoints"],
  trendMetrics=["nPolylines","meanPoints"],
  source=r"Thinning breaks a stroke into separate chains wherever three or more "
         r"branches meet; Zhang and Suen's algorithm \cite{zhang1984} guarantees "
         r"connectivity but not that a pen stroke stays one chain, and the "
         r"junction artefacts are analysed by Lam \textit{et al.} "
         r"\cite{lam1992}. Reconnecting chains by continuity of direction is the "
         r"standard repair in stroke recovery from static images "
         r"\cite{plamondon2000}.",
  reason=r"The glyph store is meant to hold pen strokes, because the gantry "
         r"draws strokes: a letter stored as five fragments is drawn with four "
         r"extra pen lifts. Where a cursive letter crosses itself, thinning emits "
         r"several chains meeting at a point. Two of them are the same pen stroke "
         r"if the pen passed straight through, which means the chains leave the "
         r"junction in nearly opposite directions, so the angle between the "
         r"outgoing direction of one and the incoming direction of the other is "
         r"small. A true branch, such as the bar of a \texttt{t} meeting its "
         r"stem, leaves at a large angle. $75^\circ$ is comfortably above the "
         r"wobble of a hand-drawn continuation, measured over the last four "
         r"points of each chain, and comfortably below a right-angle branch.",
  understanding=r"The direction is estimated over $n = 4$ points, so it is a "
                r"local tangent and noisy on a tightly curved stroke; that noise "
                r"is why the limit is as loose as $75^\circ$ rather than the "
                r"$20$--$30^\circ$ a clean continuation would need. Raising it "
                r"past $90^\circ$ starts welding genuine branches together, which "
                r"merges a crossbar into a stem; lowering it leaves real strokes "
                r"fragmented.",
  alternatives=r"Choosing pairings by a global matching over all chain ends "
               r"rather than the greedy smallest-angle merge used here; or "
               r"recovering stroke order from the image with a learned model "
               r"\cite{plamondon2000}, which needs training data this project "
               r"does not have.",
  worked=r"The chains must also end within $1.5$ px of each other in both axes "
         r"before the angle is examined, so the test is local in position as well "
         r"as direction. On the probe line the submitted setting leaves the "
         r"recorded number of polylines; the sweep shows the count falling as "
         r"$\alpha_{\text{join}}$ rises.",
  ),

C(key="spur_len",
  title="Shortest dangling chain kept after thinning",
  symbol=r"$\ell_{\text{spur}}$",
  subsystem="Centreline tracing",
  module="style", where="_PruneSpurs",
  default=2.5, values=[0.0,1.0,2.5,5.0,9.0],
  patches=lambda v: [("style","def _PruneSpurs(chains,minLen=2.5):",
                      "def _PruneSpurs(chains,minLen=%s):" % v)],
  task="trace", params=TRACE_RUN, panel=TRACE_PANEL,
  metrics=["nPolylines","totalLength","meanPoints"],
  trendMetrics=["nPolylines","totalLength"],
  source=r"Short spurs at junctions are a known artefact of thinning rather than "
         r"a feature of the shape \cite{lam1992}; pruning them is the standard "
         r"post-process \cite{haralick1987}.",
  reason=r"Where a stroke is locally thick, for instance where the pen paused or "
         r"two strokes overlap, the thinned result grows a short barb a few "
         r"pixels long with one free end. Those barbs are not pen movement, and "
         r"if kept they become separate strokes in the glyph store and separate "
         r"pen-down moves on the machine. A barb is distinguished from a real "
         r"short stroke, such as an \texttt{i} dot, by being \emph{dangling}: one "
         r"of its ends is shared with another chain. The limit of $2.5$ px is "
         r"below the shortest real mark on these crops and above the typical "
         r"one- or two-pixel barb.",
  understanding=r"The test is a conjunction with the dangling condition and with "
                r"$\texttt{len(chains)>1}$, so an isolated short chain is never "
                r"deleted and the last chain is never deleted. Raising "
                r"$\ell_{\text{spur}}$ eventually removes genuine short "
                r"connecting strokes; at $0$ nothing is pruned.",
  alternatives=r"Thinning algorithms with built-in spur suppression, or "
               r"smoothing the mask before thinning, which also rounds real "
               r"corners.",
  worked=r"At an x-height of $42$ px, $2.5$ px is $6\%$ of an x-height: a barb "
         r"must be shorter than about one sixteenth of a letter to be pruned.",
  ),

C(key="simplify_eps",
  title="Douglas--Peucker tolerance for a stored stroke",
  symbol=r"$\varepsilon$",
  subsystem="Glyph extraction",
  module="style", where="SimplifyPolyline",
  default=0.6, values=[0.15,0.3,0.6,1.5,3.0],
  patches=lambda v: [("style","p = SimplifyPolyline(p,eps=0.6)",
                      "p = SimplifyPolyline(p,eps=%s)" % v)],
  task="glyph_cut", params=GLYPH_RUN, panel=GLYPH_PANEL,
  metrics=["nGlyphs","meanWidth","strokeW","alignConf"],
  trendMetrics=["nGlyphs","meanWidth"],
  source=r"The algorithm is Douglas and Peucker's line simplification "
         r"\cite{douglas1973}: recursively keep the point furthest from the "
         r"chord while that distance exceeds $\varepsilon$.",
  reason=r"A traced centreline has one point per pixel, so a letter is a few "
         r"hundred points. Stored unsimplified, a profile becomes very large and "
         r"every stored point survives into the G-code as a separate "
         r"\texttt{G1} move. The tolerance is in \emph{source pixels}, applied "
         r"before the stroke is divided by the x-height, so it must be read "
         r"against the pixel staircase of the skeleton: a one-pixel-wide "
         r"centreline on a diagonal deviates by up to $1/\sqrt{2} = 0.707$ px "
         r"from the straight line it represents. Setting "
         r"$\varepsilon = 0.6$ just under that keeps real curvature while "
         r"removing collinear runs, so the simplification is lossless with "
         r"respect to the staircase but not with respect to the shape.",
  understanding=r"Above about $0.71$ px the tolerance begins to cut genuine "
                r"curvature, and a curve becomes a chain of chords: at "
                r"$\varepsilon = 3$ an \texttt{o} is a polygon. Below it, "
                r"nothing is gained because the staircase itself sets the floor. "
                r"The function also has a closed-curve branch: when the two ends "
                r"are near each other it splits at the furthest point first, "
                r"because a closed loop has a chord of zero length and would "
                r"otherwise collapse entirely.",
  alternatives=r"Resampling to a fixed number of points, which \texttt{\_ResampleN} "
               r"does later for prototype averaging but which loses corners; or "
               r"curvature-based keypoint selection; or fitting splines, which "
               r"would be smoother but puts control points rather than pen "
               r"positions into the store.",
  worked=r"A diagonal staircase deviates by at most $0.707$ px from its chord, "
         r"so at $\varepsilon = 0.6$ every point of a straight diagonal run is "
         r"still discarded only when it is within tolerance; a curve of radius "
         r"$r$ keeps points about every $\sqrt{8r\varepsilon}$ px, which at "
         r"$r = 20$ px gives a point every $9.8$ px.",
  ),

C(key="glyph_max_w",
  title="Widest glyph accepted, in x-heights",
  symbol=r"$W_{\text{glyph}}$",
  subsystem="Glyph filtering",
  module="style", where="MAX_GLYPH_W",
  default=2.1, values=[1.1,1.6,2.1,3.0,4.5],
  patches=lambda v: [("style","MAX_GLYPH_W = 2.1","MAX_GLYPH_W = %s" % v)],
  task="profile",
  params=dict(author=P.AUTHOR,textWeights=P.TEXT_WEIGHTS,maxLines=18),
  panel=dict(kind="glyphs",maxGlyphs=9,panelHeight=0.95),
  metrics=["nChars","nVariants","medianPriorD","connectedness"],
  trendMetrics=["nVariants","medianPriorD"],
  source=r"Width normalised by x-height is the scale-free measure of letter size "
         r"used throughout handwriting analysis \cite{vinciarelli2001}; rejecting "
         r"over-wide segmentation hypotheses is the standard guard in "
         r"segmentation-based recognition \cite{likforman2007}.",
  reason=r"A glyph is cut at the boundaries the reader's forced alignment "
         r"suggests, and that alignment is sometimes wrong: a boundary placed too "
         r"far right makes one cell swallow its neighbour. The resulting shape is "
         r"then two letters stored under one character, which later draws as "
         r"nonsense. Real lowercase letters are at most about one x-height wide "
         r"and the widest, \texttt{m} and \texttt{w}, reach roughly $1.5$. A cell "
         r"holding two letters is at least $2$ wide. $2.1$ sits above every "
         r"single letter and at the bottom of the two-letter range, and the wide "
         r"characters get their own larger limit of $2.9$ so they are not "
         r"penalised for being wide.",
  understanding=r"This is one of a battery of filters in "
                r"\texttt{BuildAuthorProfile}, and it is the one that catches the "
                r"specific failure of a swallowed neighbour. Raising it lets "
                r"merged pairs into the store, which shows up later as a median "
                r"prior distance that climbs, because a two-letter shape is "
                r"nothing like the pooled prototype of its character.",
  alternatives=r"Rejecting on the alignment confidence alone, which is already "
               r"done at line level by \texttt{ALIGN\_CONF\_PCT}, or comparing "
               r"each glyph against the letter prior and rejecting outliers, "
               r"which \texttt{PriorFilter} does as a later and softer stage.",
  worked=r"At $x_h = 42$ px a glyph may be at most $2.1 \times 42 = 88$ px wide, "
         r"against a cell advance that is typically $0.7$--$1.0$ x-heights.",
  ),

C(key="shape_grid_n",
  title="Side of the square occupancy descriptor",
  symbol=r"$n_{\text{grid}}$",
  subsystem="Glyph filtering",
  module="style", where="_ShapeGrid",
  default=12, values=[6,8,12,16,24],
  patches=lambda v: [("style","def _ShapeGrid(g,n=12):",
                      "def _ShapeGrid(g,n=%s):" % v)],
  task="profile",
  params=dict(author=P.AUTHOR,textWeights=P.TEXT_WEIGHTS,maxLines=18),
  panel=dict(kind="glyphs",maxGlyphs=9,panelHeight=0.95),
  metrics=["nChars","nVariants","medianPriorD"],
  trendMetrics=["medianPriorD","nVariants"],
  source=r"A normalised grid of ink occupancy is the classical shape descriptor "
         r"for character matching \cite{barrow1977,dubuisson1994}; the "
         r"square-root damping and blur follow the same reasoning as the "
         r"gradient-histogram features used for writer identification "
         r"\cite{bulacu2007}.",
  reason=r"The descriptor exists to answer one question: how far is this glyph "
         r"from the way this letter usually looks? It must therefore be coarse "
         r"enough that two honest writings of the same letter match, and fine "
         r"enough that a fragment or a merged pair does not. Each cell of a "
         r"$12\times12$ grid is $1/12$ of the letter box, about $3.5$ px on a "
         r"$42$ px x-height, so a stroke placed one cell differently still "
         r"overlaps after the $3\times3$ blur that follows. Finer grids make two "
         r"writings of the same letter disagree simply because the pen wavered; "
         r"coarser ones stop distinguishing \texttt{c} from \texttt{e}.",
  understanding=r"The descriptor is normalised to unit length and compared by dot "
                r"product, so $n_{\text{grid}}$ changes the dimension of the "
                r"space, $144$ at the submitted value. Distances are not "
                r"comparable between grid sizes, which is why the sweep reports "
                r"the median prior distance it produces rather than claiming one "
                r"size scores better than another: a change in the median is "
                r"partly a change of units.",
  alternatives=r"Hausdorff distance between point sets \cite{dubuisson1994}, "
               r"which needs no grid but is dominated by outliers; chamfer "
               r"matching \cite{barrow1977}; or resampling both strokes and "
               r"comparing point by point, which \texttt{\_DenoisedPrototype} does "
               r"for averaging but which needs the strokes to correspond.",
  worked=r"The blur weights are $0.25$ at the centre and $0.09375$ at each of the "
         r"eight neighbours, summing to $0.25+8 \times 0.09375 = 1.0$, so the "
         r"blur preserves total ink before normalisation.",
  ),

C(key="cut_cost_top",
  title="Weight on how high the ink reaches in the cut-cost profile",
  symbol=r"$w_{\text{top}}$",
  subsystem="Glyph extraction",
  module="style", where="ExtractLineGlyphs",
  default=1.15, values=[0.0,0.5,1.15,2.5,5.0],
  patches=lambda v: [("style","cutCost = thin+1.15*topAbove",
                      "cutCost = thin+%s*topAbove" % v)],
  task="glyph_cut", params=GLYPH_RUN, panel=GLYPH_PANEL,
  metrics=["nGlyphs","meanWidth","alignConf"],
  trendMetrics=["nGlyphs","meanWidth"],
  source=r"Cutting a word into characters at columns of least ink is the "
         r"classical vertical-projection segmentation of cursive script "
         r"\cite{likforman2007,vinciarelli2001}.",
  reason=r"The reader's alignment gives an approximate column for each character "
         r"boundary; the boundary is then snapped to the cheapest nearby column. "
         r"The cost has two parts: how much ink the column carries, and how high "
         r"that ink reaches above the baseline. The second term exists because a "
         r"column may be thin and still be a bad place to cut: the thin part of "
         r"an \texttt{o} near its top, or an ascender, is thin but is in the "
         r"middle of a letter. A good cut is through the \emph{low} ligature "
         r"between letters. $w_{\text{top}} = 1.15$ makes a column whose ink "
         r"reaches one x-height cost slightly more than a column carrying the "
         r"typical amount of ink, so height and thinness are weighed about "
         r"equally.",
  understanding=r"\texttt{thin} is normalised by the $60$th percentile of "
                r"non-zero column sums, so it is about $1.0$ for a typical inked "
                r"column, and \texttt{topAbove} is clipped to $[0,3]$ x-heights. "
                r"At $w_{\text{top}} = 0$ the cut is decided by ink alone and "
                r"will happily slice through the top of a letter; at $5.0$ it is "
                r"decided almost entirely by height and ignores how much ink it "
                r"is cutting through.",
  alternatives=r"Cutting at the CTC alignment column directly with no snap, which "
               r"leaves boundaries mid-stroke; or a full segmentation-free "
               r"approach where glyphs are never cut out, which would mean giving "
               r"up the per-letter store the synthesis depends on.",
  worked=r"A column with the typical ink and no ink above the baseline costs "
         r"about $1.0$; a column with half the typical ink reaching one x-height "
         r"costs $0.5+1.15 = 1.65$, so the thinner but taller column is the worse "
         r"cut.",
  ),

C(key="snap_radius",
  title="How far a character boundary may be moved to find a cheap column",
  symbol=r"$r_{\text{snap}}$",
  subsystem="Glyph extraction",
  module="style", where="ExtractLineGlyphs",
  default=0.5, values=[0.1,0.25,0.5,0.9,1.5],
  patches=lambda v: [("style","snapR = max(3,int(round(xh*0.5)))",
                      "snapR = max(3,int(round(xh*%s)))" % v)],
  task="glyph_cut", params=GLYPH_RUN, panel=GLYPH_PANEL,
  metrics=["nGlyphs","meanWidth","alignConf"],
  trendMetrics=["nGlyphs","meanWidth"],
  source=r"Refining an approximate segmentation point by local search over a "
         r"cost profile is standard in segmentation-based handwriting "
         r"recognition \cite{likforman2007}.",
  reason=r"The CTC alignment is accurate to a few frames, and one frame is four "
         r"canvas columns, which after the canvas-to-pixel mapping is a "
         r"noticeable fraction of a letter. The snap radius must therefore be "
         r"large enough to reach the true ligature from the aligned guess, and "
         r"small enough that the boundary cannot run into the neighbouring "
         r"letter. Half an x-height is about half a letter width, which bounds "
         r"the move to within the letter it belongs to.",
  understanding=r"The search minimises $\text{cutCost} + 0.6\,|x-x_0|/r$, so "
                r"there is a mild pull back towards the aligned guess: a far-away "
                r"column must be clearly cheaper to win. Boundaries are also "
                r"forced to be strictly increasing with a minimum cell of "
                r"$0.08\,x_h$, so a large radius cannot collapse two boundaries "
                r"onto one column.",
  alternatives=r"No snap at all, or a snap to the nearest column of zero ink, "
               r"which does not exist inside a cursive word.",
  worked=r"At $x_h = 42$ px, $r_{\text{snap}} = 21$ px, and one CTC frame maps to "
         r"$2170/160 = 13.6$ px of the original crop, so the snap can correct an "
         r"alignment error of about $1.5$ frames.",
  ),

C(key="min_cell",
  title="Smallest cell a character boundary pair may enclose",
  symbol=r"$c_{\min}$",
  subsystem="Glyph extraction",
  module="style", where="ExtractLineGlyphs",
  default=0.08, values=[0.01,0.04,0.08,0.20,0.40],
  patches=lambda v: [("style","minCell = max(1.0,0.08*xh)",
                      "minCell = max(1.0,%s*xh)" % v)],
  task="glyph_cut", params=GLYPH_RUN, panel=GLYPH_PANEL,
  metrics=["nGlyphs","meanWidth","nNone"],
  trendMetrics=["nGlyphs","meanWidth"],
  source=r"Monotonic, non-degenerate boundaries are a requirement of any "
         r"segmentation that must produce one region per label "
         r"\cite{likforman2007}; the CTC alignment that supplies the guesses is "
         r"itself monotonic \cite[p.~370]{graves2006}.",
  reason=r"Boundaries are built into one shared, strictly increasing array "
         r"precisely so that two neighbouring boundaries cannot land on the same "
         r"column and leave a character with an empty cell. $c_{\min}$ is the "
         r"width that guarantees it, and it must also leave room for every "
         r"remaining character, which is why the upper limit for boundary $k$ is "
         r"$\text{bnd}[n]-(n-k)c_{\min}$. At $0.08$ x-heights a cell is at least "
         r"a few pixels wide, enough to hold the thinnest real mark, such as the "
         r"stem of an \texttt{l} or a full stop.",
  understanding=r"Raising $c_{\min}$ forces cells apart and so forces boundaries "
                r"away from where the cost profile wants them, eventually "
                r"overriding the snap entirely and producing an evenly spaced "
                r"comb. Lowering it allows very narrow cells, whose glyphs are "
                r"then rejected downstream for being too thin.",
  worked=r"At $x_h = 42$ px, $c_{\min} = 3.4$ px. For a line of $48$ characters "
         r"the boundaries must span at least $48 \times 3.4 = 161$ px, against a "
         r"crop width of $2170$ px, so the constraint is slack on a normal line.",
  ),

C(key="align_conf_pct",
  title="Reading-confidence percentile below which a whole line is not trusted",
  symbol=r"$q_{\text{conf}}$",
  subsystem="Profile building",
  module="style", where="ALIGN_CONF_PCT",
  default=25, values=[0,10,25,50,75],
  patches=lambda v: [("style","ALIGN_CONF_PCT = 25","ALIGN_CONF_PCT = %s" % v)],
  task="profile",
  params=dict(author=P.AUTHOR,textWeights=P.TEXT_WEIGHTS,maxLines=18),
  panel=dict(kind="glyphs",maxGlyphs=9,panelHeight=0.95),
  metrics=["nLines","nChars","nVariants","medianPriorD"],
  trendMetrics=["nVariants","medianPriorD"],
  source=r"The confidence is the mean per-frame log probability of the forced "
         r"alignment path \cite[pp.~370--372]{graves2006}; discarding "
         r"low-confidence training material is standard practice in "
         r"semi-supervised and self-labelled recognition pipelines "
         r"\cite{wigington2017}.",
  reason=r"Glyph boundaries are only as good as the alignment that produced them, "
         r"and the alignment is poor where the crop is poor: a line cut badly, "
         r"a line the writer struck through, a line with an unreadable word. "
         r"Those lines would contribute wrongly cut glyphs to the store. A "
         r"\emph{relative} limit is used rather than an absolute log probability "
         r"because confidence depends on the writer and on line length; "
         r"discarding the worst quarter of a writer's own lines is a statement "
         r"about that writer's pages, not about an absolute scale. It is applied "
         r"only when there are at least eight lines, since a quartile of fewer is "
         r"meaningless.",
  understanding=r"At $q_{\text{conf}} = 0$ nothing is discarded; at $75$ only the "
                r"best quarter of lines contributes, which gives cleaner glyphs "
                r"but far fewer of them, and some characters then fall below the "
                r"minimum variant count and disappear from the store entirely. "
                r"The trade is coverage against cleanliness, and the sweep "
                r"reports both: variants kept, and their median prior distance.",
  alternatives=r"An absolute confidence floor, which would need recalibrating per "
               r"checkpoint; or weighting glyphs by confidence instead of "
               r"discarding lines, which the later filters partly achieve.",
  worked=r"With $18$ lines, discarding the lowest quartile removes the four or "
         r"five worst-read lines before any glyph is stored.",
  ),

C(key="max_variants",
  title="Shapes of one letter kept per writer",
  symbol=r"$V_{\max}$",
  subsystem="Profile building",
  module="style", where="MAX_VARIANTS_PER_CHAR",
  default=12, values=[2,5,12,25,60],
  patches=lambda v: [("style","MAX_VARIANTS_PER_CHAR = 12",
                      "MAX_VARIANTS_PER_CHAR = %s" % v)],
  task="profile",
  params=dict(author=P.AUTHOR,textWeights=P.TEXT_WEIGHTS,maxLines=18),
  panel=dict(kind="glyphs",maxGlyphs=9,panelHeight=0.95),
  metrics=["nChars","nVariants","medianPriorD"],
  trendMetrics=["nVariants","medianPriorD"],
  source=r"Storing several exemplars per character and choosing among them at "
         r"synthesis time is the allographic model of handwriting: a writer has a "
         r"repertoire of forms per letter rather than one \cite{schomaker1990}, "
         r"and natural variation between repetitions is intrinsic to the motor "
         r"process \cite{plamondon2000}.",
  reason=r"One shape per letter gives output in which every \texttt{e} is "
         r"identical, which no reader accepts as handwriting. Keeping every shape "
         r"instead means keeping the badly cut ones too, and makes the profile "
         r"large. $12$ is set above the number needed for the repetition not to "
         r"be visible within a line: an ordinary line holds a few instances of "
         r"the commonest letters, so a dozen forms means a letter rarely repeats "
         r"its exact shape on one line. The variants are sorted by typicality "
         r"first, so the cut keeps the best twelve, and the synthesis then draws "
         r"from only the top $5$ of them with a decreasing weight.",
  understanding=r"Because \texttt{\_ELIGIBLE\_VARIANTS} is $5$, raising "
                r"$V_{\max}$ beyond $5$ changes the stored profile size without "
                r"changing what is usually drawn; it matters mainly for the "
                r"medoid and prototype calculations, which pool all kept "
                r"variants. Lowering it below $5$ directly reduces the variety "
                r"the synthesis can draw on.",
  alternatives=r"Clustering the variants and keeping one exemplar per cluster, "
               r"which is closer to the allographic account \cite{schomaker1990} "
               r"but needs a cluster count; or keeping all and selecting at draw "
               r"time, which is what the eligibility limit approximates.",
  worked=r"Owen's profile holds $43$ characters and $307$ variants at the "
         r"submitted setting, an average of $7.1$ per character, so the cap binds "
         r"only for the commonest letters.",
  ),

]
