#u23539195 coefficient justification: the handwriting-synthesis coefficients
#every panel here is a line of new text written in Owen's hand, so a change in a
#coefficient is a change in the handwriting itself
import probes as P


def C(**kw):
    return kw


#the profile is the deployed one, so these sweeps isolate the synthesis
#coefficient from any change in how the profile was built
PROFILE = r"C:/Users/braun/AppData/Local/Temp/claude/C--Users-braun-OneDrive-Documents-GitHub-Handwriting-Robot/685927e6-23ae-495d-95f3-1824d845cbe0/scratchpad/out/prof_owen.json"

#twelve seeds, so a one-character difference between settings can be told from
#the spread of the random process that produces it
SEEDS = [0,1,2,3,4,5,6,7,8,9,10,11]

SYN_RUN = dict(profile=PROFILE,text=P.SENTENCE,author=P.AUTHOR,
               textWeights=P.TEXT_WEIGHTS,authorWeights=P.AUTHOR_WEIGHTS,
               seed=0,jitter=0.15,mmPerXh=4.0,lineWidthMm=400.0,score=True,
               seeds=SEEDS)

SYN_PANEL = dict(kind="image",key="img",aspect=9.0,panelHeight=0.70)


ENTRIES = [

C(key="join_conn_floor",
  title="Joinedness below which a hand is drawn as print and makes no joins",
  symbol=r"$c_{\text{print}}$",
  subsystem="Synthesis: joining",
  module="synthesise", where="JOIN_PRINT_CONN_FLOOR",
  default=0.45, values=[0.0,0.25,0.45,0.65,0.95],
  patches=lambda v: [("synthesise","JOIN_PRINT_CONN_FLOOR = 0.45",
                      "JOIN_PRINT_CONN_FLOOR = %s" % v)],
  task="synth", params=SYN_RUN, panel=SYN_PANEL,
  metrics=["charAccMean","charAccSd","widConfMean","harmonicMean","nStrokes"],
  trendMetrics=["charAccMean","harmonicMean"],
  source=r"Whether a hand joins its letters is one of the standard descriptors "
         r"of writing style and is used directly as a writer-identification "
         r"feature \cite{bulacu2007}; the connected-versus-print distinction is "
         r"basic to handwriting analysis \cite{plamondon2000}.",
  reason=r"The profile measures \texttt{connectedness} as the share of adjacent "
         r"in-word letter pairs whose centrelines actually crossed the boundary "
         r"between them, so it is an observed property of the writer, between $0$ "
         r"and $1$. It is then used as a probability: each pair is joined with "
         r"that chance. A hand that joins occasionally, say one pair in five, is "
         r"not a joining hand, and drawing random ligatures into a print hand "
         r"looks worse than drawing none, because the join appears in arbitrary "
         r"places. The floor forces the probability to exactly zero below "
         r"$c_{\text{print}}$ rather than letting it taper. $0.45$ is below the "
         r"middle of the range and above the measured connectedness of the clear "
         r"print hands in this set.",
  understanding=r"This is a threshold on a measured quantity, so its effect is a "
                r"step, not a gradient, and which side a given writer falls on "
                r"is what matters. Owen measures $0.319$, so at the submitted "
                r"setting Owen is drawn as a print hand with no joins at all; at "
                r"$c_{\text{print}} = 0.25$ Owen crosses the floor and joins "
                r"about a third of pairs. The sweep therefore shows exactly the "
                r"decision this coefficient makes for this writer.",
  alternatives=r"No floor, letting the raw probability act however small, which "
               r"sprinkles isolated ligatures through print writing; or a smooth "
               r"ramp between print and cursive, which replaces one threshold "
               r"with two.",
  worked=r"Owen's connectedness is $0.319$. At the submitted floor of $0.45$, "
         r"$0.319 < 0.45$ so \texttt{connEff} is set to $0$ and no pair is "
         r"joined; at a floor of $0.25$, \texttt{connEff} is $0.319$ and roughly "
         r"$32\%$ of adjacent letter pairs are joined.",
  ),

C(key="max_join_gap",
  title="Furthest apart two letters may be and still be joined",
  symbol=r"$g_{\text{join}}$",
  subsystem="Synthesis: joining",
  module="synthesise", where="MAX_JOIN_GAP_XH",
  default=0.80, values=[0.2,0.5,0.8,1.3,2.5],
  patches=lambda v: [("synthesise","MAX_JOIN_GAP_XH = 0.80",
                      "MAX_JOIN_GAP_XH = %s" % v)],
  task="synth",
  params=dict(SYN_RUN,profile=PROFILE),
  panel=SYN_PANEL,
  metrics=["charAccMean","charAccSd","widConfMean","harmonicMean","nStrokes"],
  trendMetrics=["charAccMean","harmonicMean"],
  source=r"A ligature is a pen movement between two letters; its length is "
         r"bounded by the letter spacing of the hand \cite{plamondon2000}.",
  reason=r"A join is only drawn if the next letter starts within "
         r"$g_{\text{join}}$ x-heights of where the pen left the last one. "
         r"Beyond that the connecting stroke is longer than a letter and reads as "
         r"a line through the word rather than a ligature. Letter gaps in these "
         r"profiles are held between $0.03$ and $0.30$ x-heights, so "
         r"$0.80$ allows a join across an ordinary gap with margin, and refuses "
         r"one where a collision push or an unusually wide glyph has opened a "
         r"real space.",
  understanding=r"The gap is measured from the previous exit point to the start "
                r"of the body stroke of the next letter, after the collision "
                r"push, so it reflects where the ink actually is rather than the "
                r"nominal advance. The test sits in a conjunction: both letters "
                r"must be alphabetic, from compatible sources, and the random "
                r"draw against \texttt{connEff} must succeed.",
  notes=r"Owen is below the print floor, so no joins are drawn at the submitted "
        r"settings and this sweep is expected to show little or no change for "
        r"this writer. That is itself the finding, and it is reported rather "
        r"than hidden; the trend shows the stroke count unmoved.",
  alternatives=r"Scaling the limit by the measured word gap of the writer, which "
               r"would make it writer-specific at the cost of another measured "
               r"quantity in the chain.",
  worked=r"At $4$ mm per x-height a join is refused once the gap exceeds "
         r"$0.8 \times 4 = 3.2$ mm.",
  ),

C(key="collision_gap",
  title="Clearance kept between one letter's ink and the next",
  symbol=r"$d_{\text{clear}}$",
  subsystem="Synthesis: spacing",
  module="synthesise", where="COLLISION_MIN_GAP_XH",
  default=0.05, values=[-0.10,0.0,0.05,0.20,0.50],
  patches=lambda v: [("synthesise","COLLISION_MIN_GAP_XH = 0.05",
                      "COLLISION_MIN_GAP_XH = %s" % v)],
  task="synth", params=SYN_RUN, panel=SYN_PANEL,
  metrics=["charAccMean","charAccSd","widConfMean","harmonicMean"],
  trendMetrics=["charAccMean","harmonicMean"],
  source=r"Overlap resolution between adjacent glyphs is the layout problem "
         r"solved by kerning in typography; in handwriting the equivalent "
         r"constraint is that the pen does not retrace the previous letter "
         r"\cite{plamondon2000}.",
  reason=r"Each letter is placed by advancing the pen by the previous letter's "
         r"advance, which is a measured number and takes no account of the "
         r"particular shapes drawn. Two wide variants in a row can therefore "
         r"overlap. The rule compares the right-most ink of the previous body "
         r"stroke with the left-most ink of the new letter and pushes the new "
         r"letter right if they are closer than $d_{\text{clear}}$. The value is "
         r"small on purpose: the aim is to stop ink touching, not to impose "
         r"uniform spacing, because uniform spacing is precisely what makes "
         r"synthetic writing look printed. $0.05$ x-heights is about two pixels "
         r"at the rendering scale, just enough to keep strokes apart.",
  understanding=r"A negative value lets letters overlap, which the sweep includes "
                r"to show what the rule is preventing. A large value forces every "
                r"letter clear of the last and so overrides the writer's own "
                r"measured advances, producing evenly spaced, mechanical output "
                r"and a longer line.",
  alternatives=r"Testing true ink overlap by rasterising each pair, which is "
               r"exact but far more expensive; or adjusting the advance rather "
               r"than pushing, which changes the writer's measured rhythm.",
  worked=r"At $4$ mm per x-height the clearance is $0.2$ mm, against a pen width "
         r"of $0.085 \times 4 = 0.34$ mm, so adjacent strokes are kept about "
         r"half a pen width apart.",
  ),

C(key="drift_persistence",
  title="Persistence of the slow baseline drift",
  symbol=r"$\phi_{\text{drift}}$",
  subsystem="Synthesis: variation",
  module="synthesise", where="SynthesizeText",
  default=0.85, values=[0.0,0.5,0.85,0.97,1.0],
  patches=lambda v: [("synthesise",
      "driftV = 0.85*driftV+rng.gauss(0.0,driftAmp*0.35)",
      "driftV = %s*driftV+rng.gauss(0.0,driftAmp*0.35)" % v)],
  task="synth", params=SYN_RUN, panel=SYN_PANEL,
  metrics=["charAccMean","charAccSd","widConfMean","harmonicMean"],
  trendMetrics=["charAccMean","harmonicMean"],
  source=r"Handwriting is not produced on a perfectly straight baseline; slow "
         r"drift is part of the motor variability of the writing process "
         r"\cite{plamondon2000}, and baseline undulation is used as a "
         r"writer-identification feature \cite{bulacu2007}.",
  reason=r"Independent random offsets per letter give jitter, not drift: the "
         r"baseline then looks noisy rather than wandering, because each letter "
         r"is unrelated to its neighbour. Drift is modelled instead as a "
         r"first-order autoregression, where a velocity $v$ is updated as "
         r"$v \leftarrow \phi v + \mathcal{N}(0,\sigma)$ and accumulated into the "
         r"offset. $\phi$ is the memory: it sets how many letters a drift "
         r"persists for. At $0.85$ the correlation decays by half in "
         r"$\ln 2/\ln(1/0.85) = 4.3$ letters, so a drift lasts about a word, "
         r"which is the scale at which real baselines wander.",
  understanding=r"At $\phi = 0$ the velocity is white noise and the baseline "
                r"jitters per letter. At $\phi = 1$ the velocity is a random walk "
                r"that never returns, so the line runs away downwards or upwards "
                r"until the clip at $\pm 0.25$ x-heights catches it; that clip is "
                r"what makes $\phi = 1$ survivable at all and is swept "
                r"separately.",
  alternatives=r"A low-pass-filtered noise field, which is the same thing in the "
               r"frequency domain, or a sinusoidal baseline, which is too regular "
               r"to pass as handwriting.",
  worked=r"With $\phi = 0.85$ the stationary standard deviation of the velocity "
         r"is $\sigma/\sqrt{1-\phi^2} = 1.90\sigma$, where "
         r"$\sigma = 0.35 \times 0.05 \times \text{jitter}$ x-heights.",
  ),

C(key="drift_clip",
  title="Largest baseline drift allowed",
  symbol=r"$d_{\max}$",
  subsystem="Synthesis: variation",
  module="synthesise", where="SynthesizeText",
  default=0.25, values=[0.02,0.10,0.25,0.60,1.50],
  patches=lambda v: [("synthesise",
      "drift = float(np.clip(drift+driftV,-0.25,0.25))",
      "drift = float(np.clip(drift+driftV,-%s,%s))" % (v,v))],
  task="synth", params=SYN_RUN, panel=SYN_PANEL,
  metrics=["charAccMean","charAccSd","widConfMean","harmonicMean"],
  trendMetrics=["charAccMean","harmonicMean"],
  source=r"Bounding an accumulated random walk is necessary whenever the walk "
         r"has no restoring force; the writing-line constraint itself is "
         r"physical, since the writer sees and corrects the baseline "
         r"\cite{plamondon2000}.",
  reason=r"The drift velocity is autoregressive but the drift itself is a sum of "
         r"those velocities, which is a random walk: over a long line it would "
         r"wander arbitrarily far. Real writers correct against the ruling, so "
         r"the model clips the accumulated drift. The limit is in x-heights and "
         r"$0.25$ is a quarter of the body height, which is visible as a wavering "
         r"baseline but never enough to let one line collide with the next, since "
         r"lines are set $\text{asc}-\text{desc}+0.85$ x-heights apart.",
  understanding=r"The clip is also what keeps $\phi$ near $1$ usable. It acts as "
                r"a hard wall rather than a restoring force, so with a persistent "
                r"velocity the drift can sit against the wall for many letters, "
                r"which looks like a sloping line rather than a wandering one.",
  alternatives=r"An Ornstein--Uhlenbeck process on the drift itself, which pulls "
               r"back towards zero smoothly instead of clipping and would be the "
               r"better model; or re-centring at each word.",
  worked=r"At $4$ mm per x-height the drift is bounded at $\pm 1.0$ mm, against "
         r"a line pitch of $(1.51+0.68+0.85) \times 4 = 12.2$ mm for Owen.",
  ),

C(key="uniform_pen_width",
  title="Pen width the synthesised line is rendered at",
  symbol=r"$w_{\text{draw}}$",
  subsystem="Synthesis: rendering",
  module="synthesise", where="UNIFORM_PEN_WIDTH_XH",
  default=0.085, values=[0.03,0.055,0.085,0.14,0.22],
  patches=lambda v: [("synthesise","UNIFORM_PEN_WIDTH_XH = 0.085",
                      "UNIFORM_PEN_WIDTH_XH = %s" % v)],
  task="synth", params=SYN_RUN, panel=SYN_PANEL,
  metrics=["charAccMean","charAccSd","widConfMean","harmonicMean"],
  trendMetrics=["charAccMean","widConfMean"],
  source=r"Stroke width is a writer-discriminative property \cite{brink2012, "
         r"bulacu2007}, which is why a single machine pen must be modelled "
         r"explicitly rather than inherited from the source writer.",
  reason=r"The rendered image exists to be judged by the same two networks that "
         r"judge real pages, so it must look like what the machine will actually "
         r"draw. The machine has one pen, so every writer is rendered at one "
         r"width, and that width is the physical one: a $0.7$ mm ballpoint at "
         r"$4$ mm per x-height is $0.7/4 = 0.175$ x-heights of \emph{diameter}, "
         r"and the renderer takes $w_{\text{draw}}$ as a width in x-heights "
         r"before the soft edge is added, which at $0.085$ plus a $30\%$ soft "
         r"fringe lands close to that measured line. It is deliberately not the "
         r"writer's own measured \texttt{strokeWidthXh}, because the machine "
         r"cannot vary its pen.",
  understanding=r"Rendering too thin makes the judges see a different kind of "
                r"image from their training data, in which real ballpoint strokes "
                r"are several pixels wide; too thick fills in the counters of "
                r"\texttt{a}, \texttt{e} and \texttt{o}, and the reader's "
                r"accuracy falls because the letters stop being distinguishable. "
                r"The sweep measures both judges, so the two failure modes are "
                r"visible separately.",
  alternatives=r"Rendering each writer at their own measured stroke width, which "
               r"flatters the writer classifier but misrepresents the machine; or "
               r"not rendering at all and scoring the trajectory geometrically, "
               r"which cannot use the trained judges.",
  worked=r"At $4$ mm per x-height and $18$ px/mm the pen is "
         r"$0.085 \times 4 \times 18 = 6.1 \to 6$ px, blurred with "
         r"$\sigma = 0.30 \times 6 = 1.8$ px.",
  ),

C(key="uniform_soft_frac",
  title="Softness of the rendered stroke edge",
  symbol=r"$s_{\text{soft}}$",
  subsystem="Synthesis: rendering",
  module="synthesise", where="UNIFORM_SOFT_FRAC",
  default=0.30, values=[0.0,0.15,0.30,0.60,1.00],
  patches=lambda v: [("synthesise","UNIFORM_SOFT_FRAC = 0.30",
                      "UNIFORM_SOFT_FRAC = %s" % v)],
  task="synth", params=SYN_RUN, panel=SYN_PANEL,
  metrics=["charAccMean","charAccSd","widConfMean","harmonicMean"],
  trendMetrics=["charAccMean","widConfMean"],
  source=r"A Gaussian point-spread is the standard first-order model of optical "
         r"and sensor blur in an imaging chain \cite{itu601}; anti-aliasing a "
         r"drawn edge is the equivalent in synthesis.",
  reason=r"A polyline drawn with a hard round pen has a perfectly sharp edge, "
         r"which no photograph of real ink has. The networks were trained on "
         r"photographed crops whose stroke edges are soft, so a hard-edged render "
         r"is out of distribution in a way that has nothing to do with the "
         r"handwriting. The blur radius is tied to the pen width rather than "
         r"fixed, so the softness scales with the writing. $0.30$ of the pen "
         r"width is a little under half a stroke, enough to remove the staircase "
         r"without visibly thickening the line.",
  understanding=r"The blur is applied to the rendered image, so it also dims the "
                r"stroke core slightly for a thin pen: at $s_{\text{soft}} = 1.0$ "
                r"the Gaussian is as wide as the stroke and the ink no longer "
                r"reaches full black, which changes the ink level the writer "
                r"classifier sees.",
  alternatives=r"Supersampling the render and downsampling, which gives true "
               r"area coverage instead of a blur, at several times the cost; or "
               r"no softening at all.",
  worked=r"At a $6$ px pen the blur is $\sigma = 1.8$ px, cut off at three "
         r"sigma, so the kernel is $11$ px wide.",
  ),

C(key="eligible_variants",
  title="How many of a letter's stored shapes may be drawn from",
  symbol=r"$V_{\text{elig}}$",
  subsystem="Synthesis: shape choice",
  module="synthesise", where="_ELIGIBLE_VARIANTS",
  default=5, values=[1,2,5,10,20],
  patches=lambda v: [("synthesise","_ELIGIBLE_VARIANTS = 5",
                      "_ELIGIBLE_VARIANTS = %s" % v)],
  task="synth", params=SYN_RUN, panel=SYN_PANEL,
  metrics=["charAccMean","charAccSd","widConfMean","harmonicMean"],
  trendMetrics=["charAccMean","harmonicMean"],
  source=r"A writer draws on a repertoire of allographs per letter rather than "
         r"one fixed form \cite{schomaker1990}; repetition of an identical form "
         r"is a signature of machine rather than human production "
         r"\cite{plamondon2000}.",
  reason=r"The stored variants are ranked by how typical they are, and only the "
         r"top $V_{\text{elig}}$ are eligible, with weight $1/(1+1.4i)$ falling "
         r"down the ranking. The limit exists because the tail of the ranking is "
         r"where the badly cut glyphs are: a store of twelve variants may hold "
         r"three or four that are fragments. Drawing from five keeps visible "
         r"variety while staying in the part of the ranking that has been "
         r"filtered twice, by the letter prior and by this writer's own "
         r"consensus.",
  understanding=r"At $V_{\text{elig}} = 1$ the most typical shape is drawn every "
                r"time, so every instance of a letter is identical and the output "
                r"looks like a font built from the writer's hand. Raising it "
                r"reaches further into the ranking; because the weights fall as "
                r"$1/(1+1.4i)$, variant $10$ is drawn with weight $0.067$ against "
                r"$1.0$ for the first, so the effect saturates.",
  alternatives=r"Uniform sampling over all variants, which draws fragments as "
               r"often as good shapes; or sampling weighted by the prior distance "
               r"directly rather than by rank.",
  worked=r"The weights for the first five ranks are "
         r"$1,\,0.417,\,0.263,\,0.192,\,0.152$, summing to $2.02$, so the most "
         r"typical shape is drawn about half the time.",
  ),

C(key="core_slant_cap",
  title="Largest lean a fallback letterform is drawn at",
  symbol=r"$\theta_{\text{cap}}$",
  subsystem="Synthesis: shape choice",
  module="synthesise", where="CORE_SLANT_CAP_DEG",
  default=22.0, values=[0.0,10.0,22.0,35.0,50.0],
  patches=lambda v: [("synthesise","CORE_SLANT_CAP_DEG = 22.0",
                      "CORE_SLANT_CAP_DEG = %s" % v)],
  task="synth", params=SYN_RUN, panel=SYN_PANEL,
  metrics=["charAccMean","charAccSd","widConfMean","harmonicMean"],
  trendMetrics=["charAccMean","widConfMean"],
  source=r"Slant is a principal axis of handwriting style and is estimated and "
         r"normalised in essentially every handwriting recogniser "
         r"\cite{vinciarelli2001}; it is also a writer-identification feature "
         r"\cite{bulacu2007}.",
  reason=r"When a stored shape is too rough to use, a designed fallback "
         r"letterform is drawn instead. It is given the writer's lean so it does "
         r"not stand out, but only up to a cap, because the fallback is a "
         r"general-purpose shape that was not drawn at an extreme slant and "
         r"shears badly: at large angles the round letters become ellipses on a "
         r"diagonal and the result is less legible than an upright letter would "
         r"have been. $22^\circ$ is beyond the measured slant of every writer in "
         r"this set, so for them the cap is inactive, and it exists to bound the "
         r"behaviour on a writer whose slant estimate is extreme or wrong.",
  understanding=r"The shear applied is $\tan\theta$, which grows faster than "
                r"linearly: $\tan 22^\circ = 0.404$ against "
                r"$\tan 50^\circ = 1.19$, three times as much. That non-linearity "
                r"is why a cap in degrees is applied rather than a cap on the "
                r"shear. A separate, looser cap of $30^\circ$ applies to the "
                r"joined fallback alphabet, which was drawn with a lean already.",
  worked=r"Owen's measured slant is $0.0^\circ$, so the cap is inactive for this "
         r"writer and the sweep is expected to show little change; that is "
         r"reported rather than hidden.",
  alternatives=r"Shearing the fallback and then re-fitting its width, which "
               r"compensates for the horizontal stretch shear introduces; or "
               r"holding a set of fallback alphabets at several slants.",
  ),

C(key="auto_lam_good",
  title="Prior distance at which a writer's glyph store is still fully trusted",
  symbol=r"$\lambda_{\text{good}}$",
  subsystem="Synthesis: legibility",
  module="synthesise", where="_AUTO_LAM_GOOD",
  default=0.24, values=[0.10,0.18,0.24,0.30,0.45],
  patches=lambda v: [("synthesise","_AUTO_LAM_GOOD = 0.24",
                      "_AUTO_LAM_GOOD = %s" % v)],
  task="synth", params=SYN_RUN, panel=SYN_PANEL,
  metrics=["charAccMean","charAccSd","widConfMean","harmonicMean"],
  trendMetrics=["charAccMean","widConfMean"],
  source=r"Trading fidelity against legibility by interpolating towards a clean "
         r"prototype is the same trade a handwriting generator makes between "
         r"style and content \cite{graves2013,kang2020}.",
  reason=r"The system must write text that is both legible and in the writer's "
         r"hand, and for a rough store those two goals conflict. The legibility "
         r"setting $\lambda$ is derived automatically from the median prior "
         r"distance of the writer's own stored shapes, ramping linearly from $0$ "
         r"at $\lambda_{\text{good}}$ to $1$ at $\lambda_{\text{bad}}$. "
         r"$\lambda_{\text{good}} = 0.24$ is the distance below which the stored "
         r"shapes are judged to be genuine letters that need no help, and it was "
         r"set from the distribution of measured median distances across the "
         r"eleven writers, not from a single page.",
  understanding=r"$\lambda$ feeds \texttt{\_SwapProb}, which decides per letter "
                r"whether to draw the writer's own shape or the clean fallback. "
                r"Lowering $\lambda_{\text{good}}$ makes the system distrust good "
                r"stores and substitute fallbacks into writing that did not need "
                r"them, which raises reading accuracy and lowers writer "
                r"confidence: the sweep shows that trade directly, and the "
                r"\texttt{glyphSources} counts show how many letters came from "
                r"the writer rather than the fallback.",
  alternatives=r"A fixed $\lambda$ per writer set by hand, which is what the "
               r"\texttt{legibility} argument still allows; or choosing per "
               r"letter from the judges, which \texttt{SynthesizeJointBestOf} "
               r"does as a repair pass.",
  worked=r"Owen's median prior distance over the stored variants is recorded in "
         r"the profile; $\lambda = \mathrm{clip}((\text{med}-0.24)/(0.32-0.24), "
         r"0,1)$, so a median of $0.28$ gives $\lambda = 0.5$.",
  ),

]
