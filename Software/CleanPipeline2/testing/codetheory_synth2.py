"""synthesise.py, part 2: layout, rasterising, and the search.

Laying the chosen glyphs out into lines, turning the pen path into an
image the networks can judge, and the best-of-N search that picks the best
attempt.
"""

FUNCS = {

 "_Jitter": dict(
   inputs=[("strokes", "one glyph's strokes"),
           ("rng", "the seeded random generator"),
           ("amp", "the position shift amplitude, in x-heights"),
           ("rotAmp", "the rotation amplitude, in degrees"),
           ("scaleAmp", "the scale variation amplitude, as a fraction")],
   output="the strokes shifted, rotated and scaled by small random amounts.",
   theory=[
     ("h", "Why randomness is added deliberately"),
     ("p", "Without it, every instance of a letter on the page is "
           "identical to the pixel, because the same stored variant is "
           "drawn the same way. Identical repetition is the visual "
           "signature of print, and a page of it reads as a font however "
           "handwritten the individual letterforms are. Real writing never "
           "repeats a letter exactly, so a small random variation per glyph "
           "is not a flaw introduced into the output; it is a property of "
           "handwriting that must be put back."),
     ("h", "The three transformations"),
     ("eq", r"\begin{pmatrix} x' \\ y' \end{pmatrix} \;=\; "
            r"\begin{pmatrix} \cos\theta & -\sin\theta \\ "
            r"\sin\theta & \cos\theta \end{pmatrix} "
            r"\begin{pmatrix} s_x x \\ s_y y \end{pmatrix} \;+\; "
            r"\begin{pmatrix} \delta_x \\ \delta_y \end{pmatrix}",
      [("x,\\,y", "the glyph's original coordinates, in x-heights"),
       ("x',\\,y'", "the jittered coordinates"),
       (r"\theta", "a small random rotation, drawn from a normal "
                   "distribution"),
       ("s_x,\\,s_y", "independent random scale factors, each near 1"),
       (r"\delta_x,\,\delta_y", "a random translation"),
       (r"\begin{pmatrix}\cos\theta & -\sin\theta \\ \sin\theta & "
        r"\cos\theta\end{pmatrix}",
        "the standard 2-D rotation matrix, which turns a point "
        "anticlockwise about the origin by theta")],
      None),
     ("p", "The order is scale, then rotate, then translate, and it is not "
           "interchangeable: rotating before scaling would make the "
           "independent x and y scalings act along the rotated axes, "
           "shearing the glyph rather than stretching it. Scaling x and y "
           "independently is itself deliberate, since a writer's letters "
           "vary slightly in proportion as well as in size."),
     ("h", "Background: why a normal distribution"),
     ("p", "Each amount is drawn from a **normal distribution**, the "
           "bell-shaped curve, rather than uniformly from a range. The "
           "difference is that a normal distribution makes small deviations "
           "much commoner than large ones, with no hard limit, whereas a "
           "uniform draw makes a large deviation exactly as likely as a "
           "tiny one. Real variation behaves the first way: most letters "
           "are close to the writer's usual form and a few are noticeably "
           "off. A uniform draw produces output that wobbles "
           "unconvincingly, because every letter is about equally "
           "displaced."),
     ("h", "Why the vertical shift is smaller"),
     ("p", "The vertical amplitude is 0.6 of the horizontal. Vertical "
           "position is read more sharply than horizontal: the eye uses the "
           "baseline as a reference, so a letter sitting high stands out in "
           "a way that one sitting slightly right does not. The recogniser "
           "is similarly sensitive, since its features were learned "
           "relative to the baseline. Reducing the vertical component keeps "
           "the variation from reading as a fault."),
     ("note", "The amplitudes reaching this function are already scaled by "
              "the caller's jitter setting and, for an anchor glyph, by "
              "CORE_JITTER_FRAC. So the same function serves for a fully "
              "varied extracted glyph and a nearly rigid anchor, with the "
              "policy decision kept in the caller rather than here."),
   ]),

 "_Resample": dict(
   inputs=[("poly", "a path"), ("step", "the desired spacing between "
                                        "points")],
   output="the path with points evenly spaced at about that interval.",
   theory=[
     ("h", "What it is for"),
     ("p", "Respaces a path so its points sit at roughly equal distances "
           "along it, by the arc-length method described under _ResampleN in "
           "style.py: cumulative distance along the path is computed and new "
           "points are read off at even distances by interpolation."),
     ("h", "Why layout needs it"),
     ("p", "Paths reaching this point come from several sources with very "
           "different point densities. A stored glyph was simplified, so a "
           "straight run is two points. A ligature is generated from three "
           "control points. A built-in letterform was laid out by hand at "
           "whatever spacing was convenient. Concatenating them gives a path "
           "whose density varies several-fold along its length, and two "
           "later stages are sensitive to that."),
     ("items", [
       "**The rasteriser** draws by stepping along the path, so a sparse "
       "region is drawn from fewer samples; with a long enough gap the "
       "round pen's footprints no longer overlap and the stroke appears as "
       "a row of dots.",
       "**The G-code writer** emits a move per point, so the machine's "
       "motion is smoother where points are dense. Even spacing gives even "
       "pen speed, which is visible in the drawn line.",
     ]),
     ("h", "Why 0.35 millimetres"),
     ("p", "The spacing used throughout layout. It is fine enough that "
           "successive pen positions overlap at any plotted size, and "
           "coarse enough that a line of text does not become tens of "
           "thousands of points, which would slow both the rasteriser and "
           "the machine's motion planning for no visible gain."),
     ("note", "This duplicates _ResampleN in style.py, differing in that it "
              "takes a spacing rather than a count and returns tuples rather "
              "than an array. Both forms are needed: averaging requires a "
              "fixed count so that points correspond between strokes, while "
              "drawing requires a fixed spacing so that density is uniform. "
              "Neither can be expressed as the other."),
   ]),

 "USE_STYLE_CALIBRATION": dict(
   output="true, enabling the measured correction of the output's style back "
          "toward the writer's real pages.",
   theory=[
     ("h", "What it enables"),
     ("p", "A feedback step: render a sample of synthetic text, measure its "
           "slant and reach the same way they were measured on the writer's "
           "real lines, compare, and correct. Turning it off uses the "
           "profile's measurements directly."),
     ("h", "Why any correction is needed"),
     ("p", "The profile records the writer's slant and reach as measured "
           "from their pages, and synthesis applies exactly those numbers. "
           "One might expect the output to measure the same, and it does "
           "not. The discrepancy accumulates through the pipeline: glyphs "
           "were de-slanted during extraction and are re-sheared during "
           "layout, but not every stroke shears the same way; the jitter "
           "rotates glyphs slightly; the anchor glyphs are capped at a "
           "shallower slant than the writer's own; and the baseline drift "
           "shifts letters vertically. Each is small and the measured "
           "result is systematically off."),
     ("p", "Rather than trying to account for each effect analytically, the "
           "calibration measures the total discrepancy and corrects for it. "
           "That is the sound approach when the individual contributions "
           "are numerous, small, and interact, which is the case here."),
   ]),

 "_STYLE_CAL": dict(
   output="a cache of the computed correction, keyed by writer and render "
          "scale.",
   theory=[
     ("h", "Why it is cached"),
     ("p", "Computing the correction requires rendering a long sample "
           "sentence three times and measuring each, which is far more work "
           "than drawing the text actually requested. Since the correction "
           "depends only on the profile and the render scale, it is the "
           "same every time for a given writer, so computing it once per "
           "program run is sufficient."),
     ("h", "Why the key includes the scale"),
     ("p", "The key is the writer together with the x-height in millimetres "
           "and the pixels per millimetre. Both affect the measurement: at "
           "a small rendered size the stroke width is a larger fraction of "
           "the x-height, which changes what the binarisation and the band "
           "detection find, so a correction computed at one scale does not "
           "apply at another. Keying on the scale as well as the writer "
           "prevents a cached value being reused where it does not hold."),
     ("note", "This is a module-level cache that is never cleared, so a "
              "profile edited within a single run would keep its old "
              "correction. That is acceptable because profiles are loaded "
              "from disk and not modified in place, but it is the kind of "
              "assumption worth knowing when a cache is keyed by identity "
              "rather than content."),
   ]),

 "StyleCalibration": dict(
   inputs=[("profile", "the writer's style profile"),
           ("mmPerXh", "the x-height in millimetres, default 4.0"),
           ("pxPerMm", "the render resolution, default 18.0")],
   output="a dictionary of three corrections: a shear adjustment and two "
          "zone scale factors.",
   theory=[
     ("h", "The method: measure the output, compare, correct, repeat"),
     ("p", "A fixed sample sentence is rendered with the current "
           "correction applied, the rendered image is measured exactly as "
           "RenderRefStats measured the writer's real lines, and the "
           "correction is updated from the difference. That is repeated "
           "three times."),
     ("h", "Why the comparison is valid"),
     ("p", "The crucial property is that both sides are measured by the "
           "same procedure. The reference values stored in the profile as "
           "slantMeasRef, ascMeasRef and descMeasRef were produced by "
           "running the same binarisation, band detection and slant "
           "estimation on the writer's real crops, for exactly this "
           "purpose. Comparing a quantity measured one way with the same "
           "quantity measured another way would confound the measurement "
           "methods with the thing being measured; comparing like with like "
           "isolates the real discrepancy."),
     ("h", "The three corrections"),
     ("eq", r"\begin{aligned} "
            r"\Delta_{\mathrm{shear}} &\mathrel{+}= "
            r"\tan(\mathrm{slant}_{\mathrm{ref}}) - "
            r"\tan(\mathrm{slant}_{\mathrm{got}}) \\[3pt] "
            r"k_{\mathrm{asc}} &\mathrel{\ast}= "
            r"\frac{\mathrm{asc}_{\mathrm{ref}} - 1}"
            r"{\mathrm{asc}_{\mathrm{got}} - 1} \\[3pt] "
            r"k_{\mathrm{desc}} &\mathrel{\ast}= "
            r"\frac{\mathrm{desc}_{\mathrm{ref}}}"
            r"{\mathrm{desc}_{\mathrm{got}}} \end{aligned}",
      [(r"\Delta_{\mathrm{shear}}", "the additive shear correction, applied "
                                    "on top of the profile's slant"),
       ("k_{\\mathrm{asc}}", "the multiplicative correction to the ascender "
                             "zone"),
       ("k_{\\mathrm{desc}}", "the same for the descender zone"),
       (r"\mathrm{ref}", "the value measured on the writer's real lines"),
       (r"\mathrm{got}", "the value measured on this render"),
       (r"\mathrel{+}=,\; \mathrel{\ast}=", "the correction is accumulated "
                                            "across rounds rather than "
                                            "replaced"),
       ("-1", "subtracted from the ascender figures because the ascender "
              "zone is measured from the top of the x-height band, not from "
              "the baseline")],
      None),
     ("p", "The shear correction is additive and is computed on the tangent "
           "rather than on the angle. That is because shear is applied as a "
           "tangent: a point is displaced horizontally in proportion to its "
           "height, with the tangent as the constant. Differences of "
           "tangents therefore compose by addition, while differences of "
           "angles do not, so correcting in degrees would be wrong by an "
           "amount growing with the angle."),
     ("p", "The zone corrections are multiplicative because they scale a "
           "zone's extent, and the subtraction of 1 in the ascender case "
           "matters: the ascender is measured as a height above the "
           "baseline, of which the first 1.0 is the x-height band itself. "
           "The part that can be corrected is the reach beyond the band, so "
           "the ratio must be taken of the excess rather than of the total. "
           "Taking the ratio of the totals would under-correct, since the "
           "fixed band would dilute the difference."),
     ("h", "Why it iterates, and only three times"),
     ("p", "One round is not exact, because the corrections interact: "
           "stretching the ascender zone changes which pixels the band "
           "detection finds, which changes the measured slant. So each "
           "round measures the state the previous round produced and "
           "corrects the remaining error. Three rounds are used because the "
           "residual after the first is already small and the process is "
           "convergent, so further rounds would change little at "
           "considerable cost, each requiring a full render and "
           "measurement."),
     ("h", "Why the corrections are clipped"),
     ("p", "Each zone factor is held between 0.5 and 1.6. The ratios are "
           "quotients, and a quotient whose denominator is near zero is "
           "unbounded: a render in which the measured ascender came out "
           "barely above the band gives a denominator near zero and a "
           "correction of hundreds, which would stretch the next render's "
           "ascenders off the page. Clipping bounds the damage from a bad "
           "measurement, and the bounds are wide enough not to interfere "
           "with a genuine correction, which is a few tens of percent."),
     ("h", "The cases where no correction is made"),
     ("p", "If any reference value is missing or not a number, or if the "
           "render cannot be measured, the neutral correction is returned "
           "and cached. Those are the cases where there is nothing to "
           "compare against, and applying a correction derived from a "
           "failed measurement would be worse than applying none."),
     ("note", "The sample sentence is a pangram-like text chosen to contain "
              "ascenders, descenders and a wide spread of letters, rendered "
              "with jitter switched off and on a line wide enough that no "
              "wrapping occurs. All three are necessary: without jitter the "
              "measurement is repeatable, and without wrapping the measured "
              "band belongs to one line rather than several stacked."),
   ]),

 "_ApplyCal": dict(
   inputs=[("strokes", "one glyph's strokes"),
           ("cal", "the correction dictionary")],
   output="the strokes with the ascender and descender zones rescaled.",
   theory=[
     ("h", "What it does"),
     ("p", "Applies the two zone corrections, stretching or compressing "
           "only the parts of a glyph outside the x-height band. The shear "
           "correction is not applied here; it was folded into the slant "
           "before layout began, since shear depends on the final height and "
           "must be applied as the glyph is placed."),
     ("eq", r"y' \;=\; \begin{cases} "
            r"1 + (y-1)\,k_{\mathrm{asc}} & y > 1 \\[3pt] "
            r"y \cdot k_{\mathrm{desc}} & y < 0 \\[3pt] "
            r"y & 0 \le y \le 1 \end{cases}",
      [("y", "the glyph's height at this point, in x-heights"),
       ("y'", "the corrected height"),
       ("k_{\\mathrm{asc}}", "the ascender correction factor"),
       ("k_{\\mathrm{desc}}", "the descender correction factor")],
      None),
     ("p", "This is the same piecewise form as _BaselineGlyph uses for "
           "adapting a built-in letterform, and for the same reason: the "
           "band from 0 to 1 is the reference and must not move, so only "
           "the reach beyond it is scaled. Writing the ascender case as 1 "
           "plus a scaled excess, rather than scaling y directly, is what "
           "keeps the band's top edge fixed at exactly 1."),
     ("h", "The early exit"),
     ("p", "When both factors are within a thousandth of 1, the strokes are "
           "returned untouched. That is the common case, since many writers "
           "need no zone correction, and the check avoids rebuilding every "
           "point of every glyph to multiply by one."),
     ("note", "This is applied only to extracted glyphs, never to an anchor. "
              "The anchor's proportions were either designed to be legible "
              "or are the writer's own best letterform, and re-stretching "
              "them would reintroduce the distortion the anchor exists to "
              "avoid. The caller enforces that by testing the source label "
              "before calling."),
   ]),

 "_Ligature": dict(
   inputs=[("p0", "where the pen left the previous letter, in millimetres"),
           ("p1", "where it enters the next"),
           ("xh", "the x-height in millimetres"),
           ("rng", "the seeded random generator")],
   output="a path joining the two points, as a resampled curve.",
   theory=[
     ("h", "What a ligature is here"),
     ("p", "The connecting stroke drawn between two joined letters. Because "
           "the stored glyphs had their ligature tails trimmed off during "
           "extraction, as described under _TrimLigatureTails, synthesis "
           "must supply its own, drawn between whatever pair of letters it "
           "actually placed. That is the advantage of trimming: the join "
           "suits the pair at hand rather than the pair the glyph was cut "
           "beside."),
     ("h", "The shape: a dip whose depth depends on the distance"),
     ("eq", r"\mathrm{sag} \;=\; \min\big(0.11\,\mathrm{xh},\; "
            r"0.16\,|\Delta x|\big) \;+\; \varepsilon, \qquad "
            r"\varepsilon \sim \mathcal{N}(0,\, 0.015\,\mathrm{xh})",
      [(r"\mathrm{sag}", "how far below the lower endpoint the curve's "
                         "midpoint is placed"),
       (r"\Delta x", "the horizontal distance between the two endpoints"),
       (r"\mathrm{xh}", "the x-height in millimetres"),
       ("0.11", "the ceiling on the dip, as a fraction of the x-height"),
       ("0.16", "the dip as a fraction of the distance travelled"),
       (r"\varepsilon", "a small random variation"),
       (r"\mathcal{N}(0,\sigma)", "a normal distribution with mean 0 and "
                                  "standard deviation sigma")],
      None),
     ("p", "The code comment names the defect this avoids: a fixed dip "
           "turns a short join into a loop, and tangled loops are what makes "
           "synthetic joined writing unreadable. Where two letters are close "
           "together, a dip of a fixed depth has to descend and return "
           "within a very short horizontal distance, which draws a narrow "
           "vertical loop below the baseline rather than a connecting "
           "stroke. Making the dip proportional to the distance travelled "
           "means a short join is nearly straight and only a long one "
           "sags."),
     ("p", "The ceiling then caps the other end: over a long gap, "
           "proportionality alone would produce a dip deeper than a "
           "descender, which would collide with the line below. Taking the "
           "smaller of the two expressions gives a dip that grows with "
           "distance up to a limit, which is how a real pen behaves."),
     ("h", "Why a single midpoint and interpolation, not a formal curve"),
     ("p", "The curve is constructed by placing one point below the "
           "midpoint of the two endpoints and resampling the three-point "
           "path, which the even resampling turns into a smooth arc. A "
           "formal spline would give a more precisely controlled curve, and "
           "it is not needed: a ligature spans a fraction of an x-height "
           "and is drawn with a pen whose width is a tenth of that, so the "
           "difference between a resampled three-point path and an exact "
           "curve is well below what the pen can express."),
     ("h", "The degenerate case"),
     ("p", "When the horizontal distance is under 0.06 x-heights, a "
           "straight segment between the two points is returned. At that "
           "distance the two letters are effectively touching and any dip "
           "would be the vertical loop the whole design avoids. Returning "
           "the two points still produces a connection, which is what "
           "joining means, without drawing anything misleading."),
   ]),

 "_SplitWords": dict(
   inputs=[("text", "the text to be written")],
   output="a list of words with newline markers between the lines.",
   theory=[
     ("h", "What it is for"),
     ("p", "Breaks the text into the units layout works with. Words are "
           "the unit because wrapping happens at word boundaries: a word "
           "that will not fit on the current line moves to the next one "
           "whole, since handwriting is not hyphenated."),
     ("h", "Why newlines survive as markers"),
     ("p", "The text is split on newlines first and a marker is inserted "
           "between the resulting chunks, so an explicit line break in the "
           "input is preserved rather than being treated as a space. The "
           "distinction matters because an explicit break is the writer's "
           "instruction and must be obeyed wherever the line happens to be, "
           "while a space is merely a place where a break is permitted. "
           "Keeping the two distinct lets layout handle them as what they "
           "are."),
     ("note", "Empty strings between consecutive spaces are discarded, so "
              "repeated spaces collapse to one gap, and a trailing marker "
              "is removed so text ending in a newline does not produce an "
              "empty final line. Both are small conveniences that keep the "
              "layout loop from needing to test for them."),
   ]),

 "_WordWidth": dict(
   inputs=[("word", "one word"), ("profile", "the writer's style profile")],
   output="approximately how wide the word will be, in x-heights.",
   theory=[
     ("h", "What it is for"),
     ("p", "An estimate used only to decide whether a word fits on the "
           "current line. It sums the writer's median advance for each "
           "character, falling back to 0.9 x-heights for a character with "
           "no recorded advance."),
     ("h", "Why an estimate is acceptable here"),
     ("p", "The true width is not available in advance, because it depends "
           "on which variant of each letter is chosen, how the jitter "
           "scales them, and whether the collision guard pushes anything "
           "right. All of that happens while the word is being drawn, which "
           "is after the decision to put it on this line has been made. "
           "Determining the exact width would mean laying the word out, "
           "measuring it, and discarding the result if it did not fit, "
           "which is several times the work for a decision whose only "
           "consequence is where a line breaks."),
     ("p", "The estimate is slightly low on average, since it ignores the "
           "collision guard, which only ever widens. So a word occasionally "
           "overruns the requested line width by a small margin. That is "
           "the right direction for the error to fall: a line a few percent "
           "too long is unremarkable, where a conservative estimate that "
           "broke lines early would leave visible ragged gaps."),
   ]),

 "SynthesizeText": dict(
   inputs=[("text", "the text to write"),
           ("profile", "the writer's style profile"),
           ("mmPerXh", "the x-height in millimetres, default 4.0"),
           ("seed", "the random seed, so a run is reproducible"),
           ("lineWidthMm", "where to wrap, default 180"),
           ("jitter", "how much random variation, default 0.15"),
           ("legibility", "the lam setting, derived if not given"),
           ("perCharLam", "per-position overrides used by the repair loop"),
           ("_cal", "a precomputed calibration, to avoid recomputing it")],
   output="a Trajectory: the pen path for that text in that hand.",
   theory=[
     ("h", "What it is for"),
     ("p", "The main layout function. It walks the text character by "
           "character and builds the pen path, handling glyph selection, "
           "calibration, variation, placement, collision, joining and line "
           "breaking. Everything above exists to be called from here."),
     ("h", "The per-character sequence"),
     ("items", [
       "**Choose a shape** with _GlyphSource, passing the height the pen "
       "left the previous letter at so that a joinable variant can be "
       "preferred.",
       "**Calibrate** it, if it is an extracted glyph, so its zones match "
       "the writer's measured reach.",
       "**Jitter** it, at full strength for an extracted glyph and a "
       "fraction of that for an anchor.",
       "**Place** it at the pen position, applying the shear for slant and "
       "the current baseline drift.",
       "**Check for collision** against the previous glyph's body and push "
       "right if needed.",
       "**Identify the body stroke** and decide whether to join to the "
       "previous letter.",
       "**Advance** the pen position.",
     ]),
     ("h", "Why the slant cap depends on legibility"),
     ("eq", r"\mathrm{cap}_{\mathrm{author}} \;=\; 45 \;-\; "
            r"19 \cdot \mathrm{clip}(\mathrm{lam},\,0,\,1)",
      [(r"\mathrm{cap}_{\mathrm{author}}", "the greatest lean in degrees "
                                           "applied to the writer's own "
                                           "glyphs"),
       (r"\mathrm{lam}", "the legibility setting"),
       ("45", "the cap at lam 0, which is permissive"),
       ("19", "how much the cap tightens as lam rises, giving 26 degrees at "
              "lam 1")],
      None),
     ("p", "Steep slant and poor legibility tend to occur together, because "
           "a steeply slanted hand is usually a joined one and a joined one "
           "extracts badly. So a high lam is evidence that the writing is "
           "hard to read, and shearing a glyph steeply makes it harder "
           "still, for the reasons given under CORE_SLANT_CAP_DEG. Linking "
           "the cap to lam reduces the lean exactly for the writers whose "
           "output most needs the help, while leaving a clean upright-ish "
           "hand free to keep its full measured slant. The anchor glyphs "
           "are capped harder again, at 22 or 30 degrees."),
     ("h", "The baseline drift"),
     ("eq", r"v_{t} \;=\; 0.85\,v_{t-1} \;+\; \eta_t, \qquad "
            r"\mathrm{drift}_t \;=\; \mathrm{clip}\big("
            r"\mathrm{drift}_{t-1} + v_t,\; -0.25,\; 0.25\big)",
      [("v_t", "the drift velocity after this character"),
       ("0.85", "how much of the previous velocity is retained, which is "
                "what makes the wander slow"),
       (r"\eta_t", "a small random impulse drawn from a normal "
                   "distribution"),
       (r"\mathrm{drift}_t", "the vertical offset applied to this "
                             "character, in x-heights"),
       (r"\mathrm{clip}(\cdot,-0.25,0.25)", "the offset held within a "
                                            "quarter of an x-height either "
                                            "way")],
      None),
     ("p", "A real hand does not write along a perfectly straight line; the "
           "baseline wanders slowly, especially on unruled paper. Adding "
           "an independent random offset per character would not reproduce "
           "that, because it would produce letters jumping up and down "
           "independently, which looks like a fault rather than a wander. "
           "What distinguishes a wander is that it is **correlated**: if "
           "this letter is high, the next is probably also high."),
     ("p", "The construction achieves that by making the random impulse act "
           "on a velocity rather than on the position, and by retaining 85 "
           "percent of the velocity each step. The offset is then the "
           "accumulated velocity, which changes direction only gradually. "
           "This is a discrete **autoregressive** process, and it is the "
           "standard way to generate a slowly varying random signal. The "
           "clip prevents the accumulation drifting arbitrarily far over a "
           "long line, which an unbounded random walk would."),
     ("h", "Why the body stroke is identified"),
     ("p", "A glyph may consist of several pen-down strokes: the body, plus "
           "a dot, a crossbar or an accent. Only one of them carries the "
           "connection to the next letter. The body is identified as the "
           "stroke with the greatest horizontal extent, and the join is "
           "made from and to it."),
     ("p", "Getting this wrong has a specific and visible consequence. If "
           "the exit point were taken from the last stroke drawn rather "
           "than the body, then for an i the exit would be the dot, and the "
           "ligature to the next letter would be drawn from the dot, "
           "sloping down from above the x-height. The width test identifies "
           "the body reliably because a dot or an accent is by nature "
           "narrow while the body spans the letter."),
     ("h", "The conditions for joining"),
     ("items", [
       "**Not the first letter of the word,** and a previous exit point "
       "exists.",
       "**Both characters are alphabetic.** Writing does not join across a "
       "digit or a comma.",
       "**Both glyphs come from compatible sources:** either both are the "
       "writer's own shapes, or both are cursive anchor forms. A printed "
       "anchor never joins, because it has no entry or exit designed for "
       "one.",
       "**A random draw** against the effective connectedness succeeds.",
     ]),
     ("p", "The source-compatibility condition is the one that needs "
           "explaining. A join between an extracted glyph and a printed "
           "anchor would be drawn to a point on the anchor that was never "
           "meant to receive one, typically partway up its side, producing "
           "a stroke across the middle of the letter. Requiring both ends "
           "of a join to be of a joinable kind means an anchor substituted "
           "into cursive writing creates a clean break in the word rather "
           "than a malformed connection, which is the less damaging of the "
           "two outcomes."),
     ("h", "How strokes accumulate"),
     ("p", "A running chain holds the current pen-down path. When a glyph "
           "joins, its ligature and body are appended to the chain, so the "
           "pen stays down across the join and several letters become one "
           "continuous stroke, which is what cursive writing is. When it "
           "does not join, the chain is flushed as a completed stroke and a "
           "new one begins. Non-body strokes such as dots are always "
           "flushed separately, since the pen must lift to draw them."),
     ("note", "The metadata returned records the glyph source counts, the "
              "writer, the slant, the connectedness and the line count. The "
              "source counts in particular are what make the "
              "fidelity-legibility trade-off measurable after the fact, and "
              "they are the reason _GlyphSource returns a label at all."),
   ]),

 "_Draw": dict(
   inputs=[("traj", "the trajectory to draw"),
           ("pxPerMm", "the resolution in pixels per millimetre"),
           ("padMm", "the margin in millimetres"),
           ("strokePx", "the pen width in pixels"),
           ("paper", "the background value, default 255"),
           ("ink", "the stroke value, default 0"),
           ("softPx", "the blur radius in pixels, 0 for none")],
   output="a grey image of the trajectory, ink on paper.",
   theory=[
     ("h", "Why the pen path is turned into an image at all"),
     ("p", "The plotter draws from the trajectory and never needs an image. "
           "The image exists because the two judging networks score "
           "pictures, not paths. Every evaluation in this project, and the "
           "best-of-N search itself, works by rendering a candidate and "
           "asking the networks what they see. So the rasteriser is not a "
           "display convenience; it is part of the optimisation loop, and "
           "how faithfully it imitates a photograph determines whether the "
           "scores mean anything."),
     ("h", "The coordinate flip"),
     ("eq", r"x_{\mathrm{px}} \;=\; (x - x_0 + \mathrm{pad}) \cdot "
            r"\mathrm{ppm}, \qquad "
            r"y_{\mathrm{px}} \;=\; (y_1 - y + \mathrm{pad}) \cdot "
            r"\mathrm{ppm}",
      [(r"x_{\mathrm{px}},\, y_{\mathrm{px}}", "the pixel position"),
       ("x,\\,y", "the trajectory position in millimetres, y upward"),
       ("x_0", "the leftmost extent of the trajectory"),
       ("y_1", "the topmost extent"),
       (r"\mathrm{pad}", "the margin in millimetres"),
       (r"\mathrm{ppm}", "pixels per millimetre")],
      None),
     ("p", "The x expression merely shifts the origin and scales. The y "
           "expression also reverses direction, by subtracting from the top "
           "extent rather than adding to the bottom one, because the "
           "trajectory has y increasing upward and an image has row 0 at "
           "the top. This is the single place that conversion happens on "
           "the rendering path, which is the discipline described under "
           "Trajectory: each consumer converts at its own boundary."),
     ("h", "Why the round pen needs no end caps"),
     ("p", "Each stroke is drawn by DrawPolyline, which sets a pixel "
           "according to its distance from the path rather than by stepping "
           "along and stamping a shape. A consequence worth drawing out is "
           "that round ends and round joins come out automatically: the set "
           "of points within a fixed distance of a line segment is a "
           "rectangle with a semicircle at each end, and the union of two "
           "such regions is already correctly rounded where they meet. A "
           "stamping rasteriser would need explicit end caps and join "
           "handling, and would leave notches at sharp corners."),
     ("h", "Why the blur comes last"),
     ("p", "All the strokes are drawn and then the whole image is blurred "
           "once. Blurring each stroke as it was drawn would be both slower "
           "and wrong: where two strokes overlap, each having been "
           "individually softened, the overlap would be darker than a "
           "single stroke's core, producing a visible dark patch at every "
           "crossing. Blurring the finished image treats a crossing as the "
           "single region of ink that it is."),
     ("note", "The canvas is sized from the trajectory's own bounds plus "
              "the margin, so it is exactly as large as the writing "
              "requires. A fixed canvas size would either clip long text or "
              "surround short text with blank paper, and the second matters "
              "as much as the first, because the proportion of the crop "
              "that is ink is one of the statistics the networks respond "
              "to."),
   ]),

 "RenderTrajectory": dict(
   inputs=[("traj", "the trajectory to render"),
           ("pxPerMm", "the resolution, default 8.0"),
           ("padMm", "the margin, derived from the writing size if not "
                     "given"),
           ("strokePx", "the pen width in pixels, derived if not given"),
           ("profile", "the writer's profile, for the non-uniform path"),
           ("size", "a final size to resample to, or nothing"),
           ("matchInk", "whether to search for the pen width matching the "
                        "writer's ink density"),
           ("uniformInk", "whether to use one pen width for every writer")],
   output="a grey image of the trajectory, prepared to resemble the crops "
          "the networks were trained on.",
   theory=[
     ("h", "What it is for"),
     ("p", "The policy layer above _Draw. It decides the pen width, the "
           "margin, the ink and paper levels and the softness, so that the "
           "rendered image resembles a photographed crop closely enough for "
           "the networks' scores to be meaningful."),
     ("h", "The two modes"),
     ("items", [
       "**Uniform,** the default: one pen width for every writer, black on "
       "white, matching what the plotter physically draws. This is the mode "
       "the best-of-N search uses, because the search must optimise the "
       "thing that will actually be produced.",
       "**Per-writer:** the pen width, ink level, paper level and softness "
       "taken from the writer's profile, optionally with a search for the "
       "width that matches their measured ink density. This is for "
       "comparison against their real photographed pages, where the real "
       "page does carry their own pen.",
     ]),
     ("h", "Why the margin is proportional to the writing"),
     ("p", "The margin defaults to 0.18 of the x-height rather than a fixed "
           "number of millimetres. The code comment gives the reason: a "
           "fixed pad would inflate the crop height and skew the line's "
           "aspect ratio. A line of writing is long and thin, so a fixed "
           "margin adds a large fraction to its height and a small one to "
           "its width, changing the proportions. Aspect ratio is one of the "
           "statistics recorded in the profile as a reference, and the "
           "networks were trained on crops of a characteristic shape, so "
           "distorting it is a domain shift of exactly the kind the "
           "rendering is designed to avoid."),
     ("h", "Why the ink core is painted darker than the profile's ink level"),
     ("p", "The profile's inkLevel is the median of the already-dark pixels "
           "of a real crop, which includes the soft edges of every stroke "
           "as well as their cores. It is therefore an average over the "
           "whole stroke, not the colour of its centre. Painting the core "
           "at that value and then blurring would darken nothing and "
           "lighten the edges further, giving a stroke lighter overall than "
           "the real one. So the core is painted dark and the blur produces "
           "the intermediate values, which is the order the physical "
           "process works in."),
     ("note", "Where a final size is requested, the image is resampled with "
              "the project's own resizer rather than being drawn at that "
              "size directly. Drawing at the final size would be better in "
              "principle, since resampling softens the image slightly, but "
              "the pen width is quantised to whole pixels, so at small "
              "sizes drawing directly would force a one-pixel pen and lose "
              "control of the stroke weight entirely."),
   ]),

 "_MatchInkDensity": dict(
   inputs=[("traj", "the trajectory"), ("pxPerMm", "the resolution"),
           ("padMm", "the margin"), ("startPx", "the pen width to start "
                                                "from"),
           ("target", "the ink fraction to match"),
           ("mm", "the x-height in millimetres"),
           ("paper", "the paper level"), ("ink", "the ink level"),
           ("soft", "the blur radius")],
   output="a pair of the rendered image and the pen width that produced it.",
   theory=[
     ("h", "What it is for"),
     ("p", "Finds the pen width whose rendered image has the same fraction "
           "of dark pixels as the writer's real pages. Ink fraction is one "
           "of the statistics a network responds to most directly, since it "
           "is visible at every scale, so matching it removes one axis of "
           "domain shift between synthetic and photographed crops."),
     ("h", "Why a search rather than a formula"),
     ("p", "The relationship between pen width and ink fraction cannot be "
           "written down. Doubling the width does not double the ink, "
           "because strokes overlap: where two strokes cross, the ink is "
           "counted once, and the amount of overlap depends on the "
           "particular text and letterforms. The blur and the threshold "
           "used to count dark pixels add further non-linearity. So the "
           "quantity is evaluated rather than predicted."),
     ("h", "The search: hill climbing in one dimension"),
     ("p", "The current width is probed, the direction that should help is "
           "determined from whether there is too much ink or too little, "
           "and steps of one pixel are taken in that direction for as long "
           "as the error decreases. It stops at the first step that does "
           "not improve."),
     ("eq", r"\mathrm{err}(w) \;=\; \big|\, f(w) - f^{*} \,\big|, "
            r"\qquad f(w) \;=\; \text{fraction of pixels below 128 at pen "
            r"width } w",
      [(r"\mathrm{err}(w)", "the error at pen width w, which the search "
                            "reduces"),
       ("f(w)", "the ink fraction produced by that width"),
       ("f^{*}", "the target ink fraction from the writer's real pages"),
       (r"|\cdot|", "the absolute value, since overshooting and "
                    "undershooting are equally wrong")],
      None),
     ("p", "This is **hill climbing**: move in whichever direction improves "
           "the objective and stop when no step does. It is only guaranteed "
           "to find a local best, but here that is enough, because ink "
           "fraction increases with pen width throughout, so the error has "
           "a single minimum and any local best is the global one."),
     ("h", "Why the pen width is a whole number of pixels"),
     ("p", "The width is quantised to whole pixels because that is what the "
           "rasteriser can express, which is what makes a one-pixel step "
           "the natural increment and makes the search finite. The "
           "consequence is that the target can rarely be hit exactly; the "
           "search returns the closest achievable, and at small rendered "
           "sizes the available widths are coarse enough that the residual "
           "error is noticeable."),
     ("h", "Why results are cached"),
     ("p", "Each probe requires a full render and a resize, which is the "
           "expensive part. The search revisits a width when it steps past "
           "the minimum and compares back, so caching by width removes "
           "those repeats. The upper bound on the width, a little over half "
           "an x-height, exists so that a target the geometry cannot reach "
           "makes the search terminate rather than thickening the pen until "
           "the text is a solid block."),
   ]),

 "_LineLogProbs": dict(
   inputs=[("reader", "a loaded reader"),
           ("img", "a rendered line image")],
   output="the reader's log probabilities, one row per column and one entry "
          "per character.",
   theory=[
     ("h", "What it is for"),
     ("p", "Runs a rendered image through the reader and returns the raw "
           "per-column scores rather than decoded text. Those scores are "
           "what the alignment needs, and the alignment is what identifies "
           "which characters are weak."),
     ("h", "Why the raw scores and not the text"),
     ("p", "Decoded text says what the reader concluded; the scores say how "
           "confidently, and where. The repair loop needs the second, "
           "because its question is not whether the line reads correctly "
           "overall but which particular characters are responsible for it "
           "not doing so. That information exists only in the per-column "
           "scores, and decoding discards it."),
     ("note", "The batch axis is indexed away immediately, since rendering "
              "and scoring are done one line at a time. The reader retains "
              "the axis because the training code uses it."),
   ]),

 "_WeakChars": dict(
   inputs=[("reader", "a loaded reader"),
           ("img", "a rendered line image"),
           ("visible", "the text that was supposed to be written")],
   output="the positions of the characters the reader scored worst, worst "
          "first.",
   theory=[
     ("h", "What it is for"),
     ("p", "Identifies which characters of the rendered line the reader is "
           "least sure about. The repair loop uses the answer to redraw "
           "only those characters, instead of redrawing the whole line and "
           "hoping for a better outcome."),
     ("h", "Why forced alignment rather than decoding"),
     ("p", "The text is known, so there is no recognition to be done. What "
           "is wanted is per-character evidence, and forced alignment "
           "provides it: it finds where each known character sits in the "
           "rendered line, after which the reader's own score for that "
           "character over exactly those columns is a direct measure of how "
           "legible that character came out."),
     ("eq", r"s_i \;=\; \frac{1}{t_1 - t_0} \sum_{t=t_0}^{t_1-1} "
            r"\log p_t\big(c_i\big)",
      [("s_i", "the score for the i-th character: higher is more legible"),
       ("c_i", "that character"),
       ("t_0,\\, t_1", "the first and last columns the alignment assigned "
                       "to it"),
       (r"\log p_t(c_i)", "the reader's log probability for that character "
                          "at column t"),
       (r"\frac{1}{t_1-t_0}", "the mean over its columns, so a wide letter "
                              "is not favoured over a narrow one merely for "
                              "occupying more columns")],
      None),
     ("p", "Averaging rather than summing is the point of that factor. A "
           "summed score would rank an m below an i automatically, since it "
           "spans more columns and so accumulates more negative terms, and "
           "the repair loop would spend its attention on wide letters "
           "regardless of how legible they were."),
     ("h", "The unaligned case"),
     ("p", "A character the alignment could not place at all is given a "
           "very large negative score, so it sorts to the front. That is "
           "the correct ranking: a character with no columns of its own was "
           "so poorly drawn that the aligner had to pass through it in a "
           "single step shared with its neighbours, which is a more "
           "complete failure than a character that was placed but scored "
           "badly."),
     ("note", "Whitespace is skipped, since a space is not a drawn shape "
              "and cannot be repaired by substituting a letterform. "
              "Positions are returned as indices into the original text, "
              "not into the filtered character list, because that is what "
              "the per-character override table is keyed by."),
   ]),

 "REPAIR_ROUNDS": dict(
   output="3, how many targeted redraw passes follow the search.",
   theory=[
     ("h", "What a repair round is"),
     ("p", "After the best-of-N search has chosen its best attempt, the "
           "weakest characters in it are identified and pinned to the "
           "anchor letterform, and the line is drawn again. If the result "
           "scores better it is kept. That is one round, and three are "
           "performed."),
     ("h", "Why repair is more effective than more attempts"),
     ("p", "This is the substantive point. Drawing the whole line again "
           "with a different seed changes every character at once, so a "
           "line failing because of one bad letter is as likely to acquire "
           "a new bad letter elsewhere as to fix the original. Repair "
           "changes only the characters identified as weak and leaves the "
           "rest exactly as they were, so it is a directed improvement "
           "rather than another sample. It is the difference between "
           "searching randomly and using the gradient."),
     ("h", "Why three"),
     ("p", "Each round costs a full render and two network evaluations, and "
           "the returns fall away quickly: the first round addresses the "
           "worst characters, which are where most of the loss is, and by "
           "the third the remaining weak characters are usually ones the "
           "anchor cannot help, because the difficulty lies in the "
           "neighbouring letters or the join rather than in the letterform. "
           "Three rounds reach that point without spending on further "
           "ones."),
   ]),

 "REPAIR_WORST_K": dict(
   output="3, how many characters are pinned to the anchor per round.",
   theory=[
     ("h", "Why not just the worst one"),
     ("p", "Pinning one character per round would need many more rounds to "
           "fix a line with several bad letters, and each round costs a "
           "render and two network evaluations. Pinning three advances "
           "faster at the same cost per round."),
     ("h", "Why not all of them"),
     ("p", "Because the effects interact and the acceptance test is "
           "all-or-nothing. A round's result is kept only if the overall "
           "score improves, so a round that fixes four characters and "
           "breaks a fifth may be rejected entirely, losing the four "
           "improvements with it. Changing a few at a time keeps each "
           "round's effect small enough to be judged usefully. Pinning "
           "everything at once would simply be the lam-1 case, which "
           "abandons the writer's hand altogether."),
     ("note", "Characters already pinned are not reconsidered in later "
               "rounds, whether or not the round that pinned them was "
               "accepted. That prevents the loop cycling on the same "
               "characters and guarantees it terminates, at the cost of "
               "occasionally passing over a character that a different "
               "combination would have fixed."),
   ]),

 "DEFAULT_LEGIBILITY": dict(
   output="0.65, the legibility setting used when none is supplied.",
   theory=[
     ("h", "Where it applies"),
     ("p", "Only in SynthesizeLegible, the earlier search that optimises "
           "text accuracy alone. The joint search derives its setting from "
           "AutoLegibility instead, which is the better approach because it "
           "responds to the writer's own library rather than applying one "
           "figure to everyone."),
     ("h", "Why a middling value"),
     ("p", "0.65 leans toward legibility without abandoning the writer's "
           "shapes, which is the reasonable default for a search whose only "
           "objective is text accuracy: with nothing measuring fidelity, a "
           "setting that preserved more of the hand would not be rewarded "
           "for it. The value is a compromise standing in for a measurement "
           "that the function does not make, which is precisely what "
           "AutoLegibility supplies and why the joint search prefers it."),
   ]),

 "SynthesizeLegible": dict(
   inputs=[("text", "the text to write"),
           ("profile", "the writer's style profile"),
           ("nTries", "how many attempts, default 6"),
           ("mmPerXh", "the x-height in millimetres"),
           ("lineWidthMm", "where to wrap"),
           ("jitter", "the variation amount, default 0.5"),
           ("seed", "the base random seed"),
           ("reader", "a loaded reader, constructed if not given"),
           ("pxPerMm", "the render resolution"),
           ("legibility", "the lam setting"),
           ("repair", "whether to run the repair rounds, default true")],
   output="the best trajectory found, with its scores recorded in the "
          "metadata.",
   theory=[
     ("h", "What it does"),
     ("p", "Draws the text several times with different seeds, renders and "
           "reads each, keeps the attempt the reader reads most accurately, "
           "and then runs repair rounds on it. It optimises text accuracy "
           "alone."),
     ("h", "Why a search is needed at all"),
     ("p", "Synthesis is random: which variant is chosen for each "
           "character, whether each is swapped for the anchor, the jitter "
           "and the drift all depend on the seed. So the same text in the "
           "same hand has many possible renderings of differing quality, "
           "and nothing in the layout process knows which it has produced. "
           "Drawing several and measuring them is a direct way to find a "
           "good one, and it requires no model of what makes a rendering "
           "good: the reader supplies that."),
     ("h", "The escalation ladder in the repair loop"),
     ("items", [
       "**A fresh weak character** is pinned to the anchor, which is "
       "preferably the writer's own most typical letterform.",
       "**A character still weak after being pinned** escalates to the "
       "printed alphabet, which is the last resort and abandons the "
       "writer's hand for that character entirely. Only the single worst "
       "such character escalates per round.",
     ]),
     ("p", "The ladder exists because the anchor sometimes does not help. "
           "If the writer's own best form of a letter is itself the "
           "problem, pinning to it changes nothing, and the only remaining "
           "option is a designed letterform. Escalating one character per "
           "round, and only one that has already failed at the lower rung, "
           "keeps the loss of fidelity to the minimum that produces a "
           "readable line."),
     ("note", "The function falls back to plain layout if a reader cannot "
              "be constructed, returning text drawn without any search. "
              "That keeps synthesis usable without the network weights "
              "present, which matters for testing the layout alone."),
   ]),

 "SynthesizeJointBestOf": dict(
   inputs=[("author", "the writer's name, for looking up their output slot"),
           ("text", "the text to write"),
           ("profile", "their style profile"),
           ("nTries", "how many attempts, default 6"),
           ("mmPerXh", "the x-height in millimetres"),
           ("lineWidthMm", "where to wrap"),
           ("jitter", "the variation amount, default 0.5"),
           ("seed", "the base random seed"),
           ("reader", "a loaded reader"),
           ("authorModel", "a loaded namer"),
           ("authorMapping", "the mapping from writer name to output slot"),
           ("pxPerMm", "the render resolution"),
           ("legibility", "the lam setting, derived if not given"),
           ("repair", "whether to run repair rounds"),
           ("repairRounds", "how many, default 3"),
           ("repairWorstK", "how many characters per round, default 2")],
   output="the best trajectory found, judged by text accuracy and writer "
          "identification together.",
   theory=[
     ("h", "What it is for"),
     ("p", "The search this project's reported results were produced with. "
           "It differs from the one above in judging each attempt by both "
           "networks at once: the reader, for whether the output says the "
           "right words, and the namer, for whether it still looks like "
           "this writer."),
     ("h", "Why both judges are needed"),
     ("p", "Optimising text accuracy alone has a predictable failure. The "
           "most readable rendering of any text is the one drawn entirely "
           "from clean anchor letterforms, so a search rewarded only for "
           "readability will drive the output toward generic handwriting. "
           "It would score well and would have abandoned the purpose of the "
           "project. Adding the namer puts a cost on that, so the search "
           "cannot buy readability with fidelity."),
     ("h", "The harmonic mean, and why not an average"),
     ("eq", r"S \;=\; \frac{2\,a\,w}{a + w}",
      [("S", "the combined score the search maximises"),
       ("a", "the text accuracy: the fraction of characters the reader got "
             "right"),
       ("w", "the namer's confidence that this is the right writer"),
       ("2aw/(a+w)", "the harmonic mean of the two, which is zero if either "
                     "is zero")],
      None),
     ("p", "The **harmonic mean** is used rather than the arithmetic mean, "
           "and the difference is the whole point of the combinator. An "
           "arithmetic mean lets one measure compensate for the other: an "
           "attempt scoring 0.95 on text and 0.15 on writer averages 0.55, "
           "the same as a balanced 0.55 and 0.55, so the search would be "
           "indifferent between a perfectly readable page in nobody's hand "
           "and a moderate page in the right one. The harmonic mean of the "
           "first pair is 0.26 and of the second 0.55, so the balanced "
           "candidate wins clearly."),
     ("p", "The mechanism is that the harmonic mean is dominated by the "
           "smaller value: as either term approaches zero the product in "
           "the numerator does too, while the denominator does not. So it "
           "rewards being adequate on both axes rather than excellent on "
           "one, which is exactly the requirement here. It is the same "
           "quantity as the F-measure used to combine precision and recall, "
           "and for the same reason."),
     ("h", "Why the repair loop is more cautious here"),
     ("p", "Two characters are pinned per round rather than three, and an "
           "accepted round's pins are retained while a rejected round's are "
           "discarded. The caution follows from what is being optimised. "
           "Pinning a character to the anchor reliably helps text accuracy "
           "and reliably costs writer identification, so under the joint "
           "score a repair is no longer close to free, and taking smaller "
           "steps keeps each one's net effect measurable."),
     ("h", "Why the search stops early at 0.99"),
     ("p", "Both loops break once the combined score reaches 0.99, since "
           "further attempts cannot improve meaningfully and each costs a "
           "render and two network evaluations. With a search of six to "
           "fourteen attempts per line and a page of many lines, that "
           "saving is substantial on the writers whose first attempt is "
           "already good."),
     ("note", "The writer mapping defaults to an attribute read from the "
              "namer under the name `author_mapping`, while the namer "
              "stores it as `authorMapping`. The default therefore raises "
              "an attribute error if reached, and every working caller "
              "passes the mapping explicitly. It is recorded here because a "
              "reader would otherwise assume the argument is optional."),
   ]),
}
