"""Write-up for synthesise.py: writing new text in somebody's hand."""

MODULES = {
 "synthesise.py": dict(intro=[
   "**Functional unit 6, requirement R5.** A style profile and a sentence "
   "in, a pen path in millimetres out. This is the inverse of style.py: that "
   "file takes handwriting apart, this one puts it back together with "
   "different words.",

   ("The method is exemplar-based. For each character the synthesiser picks "
    "one of the shapes the writer actually drew, rather than generating a "
    "new shape from a learned model of their hand. That choice has a clear "
    "consequence which is visible throughout the measured results: it works "
    "well when the stored shapes are genuinely letters, and it fails when "
    "they are fragments, because there is nothing in the method that can "
    "repair a bad exemplar. A generative model could interpolate; an "
    "exemplar method can only choose."),

   ("Laying letters out involves four decisions per character, and each has "
    "its own reasoning below: which stored shape to use, whether to replace "
    "it with a cleaner fallback letterform, how far to advance before the "
    "next letter, and whether to join the two. On top of that sits a small "
    "amount of deliberate randomness, because a hand that draws every 'e' "
    "identically does not read as handwriting."),

   ("The central tension in the whole file is between **legibility** and "
    "**identity**. Using the writer's own shapes preserves who wrote it and "
    "risks being unreadable; substituting clean letterforms guarantees "
    "readability and erodes the identity. One parameter, called the "
    "legibility lambda, controls where on that trade-off a given writer "
    "sits, and it is derived automatically from how good that writer's "
    "library is. Understanding that parameter is most of understanding this "
    "file."),

   ("The output is a Trajectory: lists of points the pen draws with the pen "
    "down, in millimetres, with y upwards. Nothing here knows about the "
    "machine. reproduce.py turns the trajectory into g-code."),
 ]),
}

FUNCS = {

 "SCRIPT_DIR": dict(output="This file's own folder.",
   body=["Used to locate data relative to the code rather than relative to "
         "whatever directory the program happened to be started from."]),

 "CNN_DATA_DIR": dict(output="The data folder beside the code.",
   body=["Where profiles and weights are found."]),

 "_FB": dict(
   output="A dictionary from character to a list of strokes, each a list of "
          "`(x, y)` points.",
   body=[
     "A plain printed alphabet, drawn by hand as point lists. Coordinates "
     "are in the same convention as stored glyphs: x to the right, y upwards "
     "from the baseline, 1.0 being one x-height, so the same definition "
     "works at any size.",
     "This exists for the cases where the writer's own shape cannot be used: "
     "the library has no example of that character, or every example is too "
     "damaged to draw. Without a fallback the synthesiser would have to omit "
     "characters, which would be worse than drawing a plain one.",
     "These shapes are mine, drawn to be unambiguous rather than "
     "attractive. They are not derived from any writer and are not "
     "presented as anyone's handwriting.",
   ]),

 "_CURSIVE": dict(
   output="A joined alphabet, one stroke per letter.",
   body=[
     "The same idea for a joined hand. Every letter in this alphabet begins "
     "and ends at the same height above the baseline, set by `_CX_IN` and "
     "`_CX_OUT`, so any two of them can be joined without a step.",
     "A separate alphabet is needed because substituting a printed letter "
     "into joined writing is conspicuous twice over: the shape is wrong for "
     "the hand, and the join breaks on both sides of it. With a joined "
     "fallback, a substituted letter can still connect to its neighbours.",
   ]),

 "_FB_ADV": dict(
   output="How far the pen advances after each fallback letter.",
   body=["Per-character advances for the fallback alphabets, so that a "
         "substituted 'i' does not reserve as much width as a substituted "
         "'m'."]),

 "_SIMILAR": dict(
   output="A map of characters that can stand in for one another.",
   body=[
     "Some characters are near-identical in handwriting: capital I and "
     "lowercase l, capital O and zero, o and zero, colon and semicolon, "
     "apostrophe and quote. If the library has one but not the other, "
     "borrowing is better than falling back to a generic shape, because the "
     "borrowed shape is still the writer's own.",
   ]),

 "_BaselineGlyph": dict(
   inputs="`ch`, the character. `profile`, the writer. `printOnly`, force "
          "the printed alphabet.",
   output="`(strokes, advance)` for a fallback letter scaled to this writer, "
          "or None.",
   body=[
     "Fetches a fallback letter and stretches it to the writer's own "
     "proportions. The generic alphabets are defined at standard "
     "proportions; a writer whose ascenders reach 2.7 x-heights would "
     "otherwise get a substituted 'b' that is visibly too short beside their "
     "own letters.",
     "The scaling is applied to the parts of the letter that extend beyond "
     "the x-height band, so the body stays the right size and only the reach "
     "is adjusted.",
   ]),

 "UNIFORM_INK": dict(
   output="Whether to render every writer at one pen width.",
   body=[
     "True by default, and that default is a statement about the machine "
     "rather than about handwriting. The gantry holds one pen; it cannot "
     "vary stroke width or pressure. Rendering synthesised output with "
     "uniform ink therefore shows what will actually be drawn.",
     "The alternative, searching for the pen width that matches the "
     "writer's measured ink density, exists for the case where the output is "
     "being compared against a photograph of real writing.",
   ]),

 "UNIFORM_PEN_WIDTH_XH": dict(
   output="The pen width in x-heights.",
   body=["Expressed relative to the x-height so the rendered appearance is "
         "scale independent."]),

 "UNIFORM_SOFT_FRAC": dict(
   output="How soft the drawn ink edge is, as a fraction of pen width.",
   body=[
     "A real pen edge is not a hard step: ink spreads slightly into the "
     "paper. Since the networks scoring the output were trained on "
     "photographs of real ink, rendering with a perfectly hard edge presents "
     "them with something outside their training distribution."
   ]),

 "JOIN_PROB_CAP": dict(
   output="The maximum per-pair probability of drawing a join.",
   body=["A ceiling on joining, left at 1.0, so that a fully joined hand is "
         "reproduced as fully joined."]),

 "CORE_SLANT_CAP_DEG": dict(
   output="How far a fallback letter may lean.",
   body=[
     "A substituted letter keeps the writer's lean, but capped. The generic "
     "shapes were drawn upright and sheared far enough they stop being "
     "legible, which defeats the purpose of substituting them."
   ]),

 "CORE_JITTER_FRAC": dict(
   output="How much of the usual random variation a fallback letter gets.",
   body=[
     "Less than a real glyph. A fallback is there to be readable, and "
     "wobbling it as hard as an extracted shape gives up part of what it was "
     "substituted for. This is also why a substituted letter is visible in "
     "the output: it sits straighter and cleaner than its neighbours."
   ]),

 "COLLISION_MIN_GAP_XH": dict(
   output="The clearance kept between consecutive letters.",
   body=[
     "Each stored glyph carries its own advance, but a variant wider than "
     "typical can still overlap its neighbour. This is the minimum gap the "
     "collision guard enforces by pushing a letter right."
   ]),

 "_ELIGIBLE_VARIANTS": dict(
   output="How many of a character's stored shapes may be picked from, 5.",
   body=[
     "A deliberate restriction, and the reasoning is about identity rather "
     "than quality. A person forms a given letter fairly consistently, so "
     "sampling evenly across a dozen stored variants averages their "
     "signature away: the output becomes 'a plausible hand' rather than "
     "'this hand'. Restricting the choice to the most typical few keeps the "
     "characteristic shape while still varying.",
     "The consequence is that a large part of each stored library can never "
     "be drawn. Measured across thirteen writers, 47 per cent of all stored "
     "variants are beyond this cut-off — 1762 of 3720. They are not wasted, "
     "since they establish the ranking and the medoid, but they will never "
     "appear on a page.",
   ]),

 "Trajectory": dict(
   inputs="Constructed with `strokes`, a list of pen-down paths, and "
          "`meta`, a dictionary of information about how it was made.",
   output="An object with `Bounds()` and `PenTravelMm()`.",
   body=[
     "The pipeline's representation of a drawing: a list of strokes, each a "
     "list of `(x, y)` points in millimetres, drawn with the pen down, with "
     "the pen lifted between consecutive strokes.",
     "y is upwards here, which is the plotter convention, while images use y "
     "downwards. The conversion happens in exactly two places — when "
     "rendering to an image, and when writing g-code — and both are marked "
     "in the source.",
     "`PenTravelMm` reports pen-down and pen-up distance separately. That "
     "matters for two reasons: pen-up travel is wasted machine time and is "
     "what stroke ordering minimises, and the ratio of the two is a "
     "judge-free indicator of how connected the output is, which was used to "
     "confirm that a change to the joining rule had the mechanical effect "
     "intended.",
   ]),

 "_GLYPH_CLEAN": dict(
   output="The prior distance at or below which a shape is kept, 0.12.",
   body=["A shape this close to the reference is a well-formed letter and is "
         "never substituted on quality grounds."]),

 "_GLYPH_GARBAGE": dict(
   output="The distance at or above which a shape is always replaced, 0.55.",
   body=["Beyond this the shape is not recognisably the letter, so it is "
         "replaced regardless of the legibility setting. This is why even a "
         "writer with lambda at zero shows a small substitution rate."]),

 "_COMMIT_LAM": dict(
   output="The legibility level above which the library is abandoned "
          "entirely, 0.85.",
   body=["Above this every character is substituted. The reasoning is that a "
         "library this untrustworthy is better replaced wholesale than used "
         "intermittently, which would produce output in two visibly "
         "different styles."]),

 "_SwapProb": dict(
   inputs="`lam`, the legibility setting. `priorD`, this shape's distance "
          "from the reference.",
   output="The probability in [0, 1] that this shape is replaced by the "
          "fallback letterform.",
   body=[
     "The rule that resolves the legibility-identity trade-off, per "
     "character instance:",
     ("formula", ["if priorD >= 0.55       : 1.0      always replace",
                  "if lam    >= 0.85       : 1.0      library abandoned",
                  "if lam    <= 0.001      : 0.0      trust the library",
                  "if priorD <= 0.12       : 0.06*lam a clean shape, rarely",
                  "otherwise:",
                  "   bad = (priorD - 0.12) / (0.55 - 0.12)",
                  "   p   = lam * (0.55 + 0.9*bad)"]),
     "The interpolation in the last case is what makes this a graded "
     "decision rather than a switch: the probability rises both with how "
     "untrustworthy the library is overall and with how poor this particular "
     "shape is. A rough shape in a trusted library is still usually kept; "
     "the same shape in a suspect library is usually replaced.",
     "Because it is a probability rather than a threshold, the decision is "
     "taken independently for every instance of every character. One "
     "consequence worth understanding is that a single word can therefore "
     "contain letters from both sources, which is visible in the output: the "
     "substituted letter sits straighter and cleaner than its neighbours, "
     "and the join breaks on both sides of it.",
   ]),

 "_AUTO_LAM_GOOD": dict(
   output="Median prior distance at or below which a library is fully "
          "trusted, 0.24.",
   body=["Calibrated against the measured distribution of median prior "
         "distances across the thirteen writers."]),

 "_AUTO_LAM_BAD": dict(
   output="Median prior distance at or above which a library is treated as "
          "fragments, 0.32.",
   body=["Above this the stored variants are judged not to be letters."]),

 "AutoLegibility": dict(
   inputs="`profile`, a writer's profile.",
   output="A lambda in [0, 1].",
   body=[
     "Derives how far to trust a writer's library from the library itself, "
     "so that no human judgement about a writer is needed:",
     ("formula", ["med = median( priorD over every stored variant )",
                  "lam = clamp( (med - 0.24) / (0.32 - 0.24) , 0 , 1 )"]),
     ("sub", "Why the median prior distance is the right signal"),
     "The failure mode this is detecting is a library of fragments rather "
     "than letters, and fragments are exactly what sit far from the pooled "
     "reference shape for their character. A hand that is strongly joined "
     "offers no honest place to cut between letters, so its library fills "
     "with pieces, so its median distance is high. The measure therefore "
     "tracks the real cause rather than a proxy.",
     ("sub", "What it produces in practice"),
     "Across the thirteen writers this gives lambda 0.00 for eleven of them "
     "and 0.84 and 0.97 for the two most joined hands. The two extremes are "
     "visible in the output: the eleven draw almost entirely with their own "
     "shapes, while for the hand at 0.97 every single character is "
     "substituted and none of the writer's own letters reach the page.",
     ("sub", "The limitation"),
     "One number decides the fate of all forty-odd characters. A writer "
     "whose median is poor loses every letter, including the ones that were "
     "cut perfectly well. Replacing this with a per-character decision is "
     "the single change most likely to improve both legibility and identity "
     "at once, since the trade-off would then be paid only on the letters "
     "that actually cost legibility.",
   ]),

 "_CoreGlyph": dict(
   inputs="`ch`, the character. `profile`, the writer.",
   output="`(strokes, advance, lead, isCursive)` for the fallback "
          "letterform, or None.",
   body=[
     "Produces the shape that a substitution uses. It prefers the writer's "
     "own most typical version of that letter, and only drops to the generic "
     "alphabet when even that is too far from the reference.",
     "That preference matters: a substitution using the writer's own medoid "
     "is still their handwriting, just their cleanest version of it, so "
     "identity survives a substitution far better than it would with a "
     "generic shape. This is why the one writer whose library is entirely "
     "replaced still scores high writer confidence — the replacements are "
     "mostly her own medoids.",
     "A hand whose connectedness reaches 0.5 gets the joined version of the "
     "generic alphabet when it has to use one, so that substituted letters "
     "can still connect.",
   ]),

 "_GlyphSource": dict(
   inputs="`profile`, `ch`, `rng`, the previous letter's exit height, `lam`, "
          "and flags to force a fallback or force print.",
   output="`(strokes, advance, lead, source)` where source is one of 'own', "
          "'core', 'case', 'similar' or a fallback marker.",
   body=[
     "Chooses what to draw for one character instance. Four steps.",
     ("sub", "1. Rank"),
     "The character's stored variants are sorted by prior distance, so the "
     "most typical comes first.",
     ("sub", "2. Pick, favouring the typical"),
     "Only the top `_ELIGIBLE_VARIANTS` are eligible, and they are weighted:",
     ("formula", ["w_i = 1 / (1 + 1.4*i)      i = 0 for the best",
                  "p_i = w_i / sum of w",
                  "gives roughly  49%, 21%, 13%, 9%, 7%"]),
     "A uniform choice over the eligible set would draw the fifth-best "
     "variant as often as the best; this biases strongly towards the typical "
     "while still varying. When joining, the weight is further multiplied by "
     "a gaussian in the difference between this variant's entry height and "
     "where the pen currently is, so that a variant which starts near the "
     "pen is preferred and the join has less distance to cover.",
     ("sub", "3. Maybe substitute"),
     "`_SwapProb` is evaluated for the chosen variant and the fallback is "
     "used if it fires.",
     ("sub", "4. Or borrow"),
     "If the character is absent from the library entirely, a case-swapped "
     "or visually similar character is borrowed; a borrowed lowercase shape "
     "used for a capital is scaled up to capital height. Failing that, the "
     "generic alphabet.",
     "The returned source label is what makes the usage statistics in the "
     "results possible — the proportion of letters drawn from the writer's "
     "own shapes versus substituted is reported per writer and is the most "
     "direct evidence of where the legibility trade-off landed.",
   ]),

 "_Jitter": dict(
   inputs="A glyph's strokes, a random generator, and amplitudes for "
          "position, rotation and scale.",
   output="The strokes, slightly perturbed.",
   body=[
     "Applies a small random shift, rotation and size change to one letter.",
     "This is not decoration. Repetition is the strongest signal that "
     "something was produced mechanically: a page where every 'e' is pixel "
     "identical reads as printed even when the letterform itself is "
     "perfectly natural. Real handwriting never repeats exactly, because the "
     "hand is a noisy actuator.",
     "The amplitudes are small enough that no individual letter looks wrong "
     "and the letter remains the same shape; what changes is the statistics "
     "of the page.",
   ]),

 "_Resample": dict(
   inputs="A path and a step size.",
   output="The path respaced at even intervals.",
   body=["Used to give drawn curves and joins an even point spacing, so the "
         "rasteriser's round pen produces a stroke of even weight rather "
         "than one that is darker where points bunch."]),

 "USE_STYLE_CALIBRATION": dict(
   output="Whether to correct measured style drift.",
   body=["Enables the correction described under `StyleCalibration`."]),

 "_STYLE_CAL": dict(
   output="The cached correction per writer and size.",
   body=["Calibration requires synthesising sample text and measuring it, "
         "which is slow, so the result is cached per writer."]),

 "StyleCalibration": dict(
   inputs="`profile`, `mmPerXh`, `pxPerMm`.",
   output="A dictionary of corrections: a shear delta and scale factors for "
          "ascender and descender reach.",
   body=[
     ("sub", "The problem"),
     "A profile records that a writer's slant is, say, 30 degrees. The "
     "synthesiser applies 30 degrees. But when the resulting image is "
     "measured by the same code that measured the original, it may come back "
     "as 26 — because slant is measured from the ink, and the rendered ink "
     "is not identical to real ink: the pen width is uniform, the letters "
     "are differently distributed, and the measurement itself has a "
     "quantisation step.",
     ("sub", "The correction"),
     "Rather than trying to model that discrepancy, it is measured "
     "empirically. A fixed sample sentence is synthesised, measured with the "
     "same functions used on the writer's real pages, and the difference "
     "becomes a correction applied to subsequent output. The loop runs a few "
     "times so the correction converges.",
     "This is a closed-loop correction on the pipeline's own measurements, "
     "which is legitimate because both sides use the same measurement code: "
     "the target is the writer's real measured values, not a network's "
     "opinion.",
   ]),

 "_ApplyCal": dict(
   inputs="A glyph's strokes and the calibration.",
   output="The corrected strokes.",
   body=["Applies the shear and reach corrections. Not applied to fallback "
         "letterforms, which are deliberately left closer to upright."]),

 "_Ligature": dict(
   inputs="`p0`, where the pen left the previous letter. `p1`, where it "
          "enters the next. `xh`, the x-height. `rng`.",
   output="A list of points forming the joining stroke.",
   body=[
     "Draws the connection between two joined letters. The curve has a "
     "control point below the two ends, so the pen dips as it carries "
     "across, which is what a real joining stroke does.",
     ("formula", ["sag = min( 0.11*xh , 0.16*|dx| ) + small noise"]),
     ("sub", "Why the sag is proportional to the distance"),
     "This is the one genuinely important detail. A fixed dip applied to a "
     "short join produces a loop, because the pen has to descend and climb "
     "within a very small horizontal distance. Loops where there should be "
     "short connections are the single most recognisable failure of "
     "synthetic cursive. Making the sag proportional to the distance "
     "travelled means a short join is nearly flat and only a long one dips.",
     ("sub", "A measured improvement"),
     "The original form placed the control point below *both* endpoints. "
     "When a join has to climb — from a letter that ends low to one that "
     "enters high — that produces a dive followed by a hook. Blending the "
     "control height between the two ends and drawing a quadratic through "
     "it, so the pen leaves and arrives along its direction of travel, was "
     "measured as the only change tested in this project that improved both "
     "metrics at once: character accuracy from 79.0 to 80.6 per cent and "
     "writer confidence from 72.5 to 76.9 per cent across nine writers, with "
     "the gains concentrated on the joined hands, as predicted.",
   ]),

 "_SplitWords": dict(
   inputs="`text`.",
   output="A list of words and explicit line-break markers.",
   body=["Splits on spaces and newlines, keeping the breaks as items so that "
         "the layout loop can handle an explicit newline and an automatic "
         "wrap through the same path."]),

 "_WordWidth": dict(
   inputs="`word`, `profile`.",
   output="An estimated width in x-heights.",
   body=[
     "Sums the writer's own per-character advances to estimate how wide a "
     "word will be, which is needed before drawing it in order to decide "
     "whether it fits on the current line.",
     "It is an estimate rather than exact because the actual width depends "
     "on which variants get chosen and on the jitter. Wrapping therefore "
     "uses a slightly conservative margin."
   ]),

 "SynthesizeText": dict(
   inputs="`text`, `profile`, `mmPerXh` for physical size, `seed`, "
          "`lineWidthMm`, `jitter`, `legibility` (defaults to "
          "`AutoLegibility`), and `perCharLam` for per-character overrides.",
   output="A `Trajectory`.",
   body=[
     "The main layout routine. It walks the text character by character and, "
     "for each one, performs the following in order.",
     ("formula", ["1  choose the shape            _GlyphSource",
                  "2  correct measured drift      _ApplyCal",
                  "3  add human variation         _Jitter",
                  "4  place at the pen position   lead, slant, drift",
                  "5  push right if it collides   collision guard",
                  "6  join to the previous letter _Ligature, by probability",
                  "7  advance the pen             the variant's own advance"]),
     ("sub", "Why the variant's own advance"),
     "Each stored glyph carries the advance measured from the cell it was "
     "cut from, and that is used rather than a per-character median. A wide "
     "variant advanced by a narrow median would collide with the next "
     "letter; the advance and the shape have to come from the same "
     "measurement.",
     ("sub", "The baseline wander"),
     "The baseline is not held perfectly level. A slow random walk, bounded "
     "to a quarter of an x-height, drifts it as the line proceeds, because a "
     "real hand wanders. This is applied as a smoothed process rather than "
     "independent noise per letter, so neighbouring letters move together "
     "and the line looks like it was written by an unsteady hand rather "
     "than shaken.",
     ("sub", "Joining"),
     "A join is drawn only when both letters are alphabetic, both came from "
     "compatible sources, and a per-pair random draw falls under the "
     "writer's measured connectedness. Using a probability per pair rather "
     "than a threshold is what reproduces a hand that joins some pairs and "
     "not others, which is what most real hands do.",
     ("sub", "Which stroke carries the join"),
     "The widest stroke of the placed letter is taken as the body, and the "
     "join attaches to that, so that a dot or a crossbar never becomes the "
     "connection point. This is a heuristic and it is wrong for a letter "
     "whose crossbar is wider than its body.",
   ]),

 "_Draw": dict(
   inputs="A trajectory, pixels per millimetre, padding, stroke width, paper "
          "and ink levels, and an edge softness.",
   output="A rendered grey image.",
   body=[
     "Rasterises the pen path with the round-pen rasteriser from segment.py, "
     "then softens the edge slightly.",
     "This is where y flips: the trajectory has y upwards in millimetres, "
     "the image has y downwards in pixels, so the conversion is `(y1 - y)` "
     "measuring down from the top of the bounds.",
     "The rendering is not a side matter. Both judging networks score this "
     "image, so how the ink is laid down is part of the design: too hard an "
     "edge or too thin a stroke and the networks see something unlike their "
     "training data, and the score says more about the renderer than about "
     "the handwriting.",
   ]),

 "RenderTrajectory": dict(
   inputs="A trajectory, plus `pxPerMm`, optional padding and stroke width, "
          "the profile, and `uniformInk`.",
   output="A grey crop in the style of the training data, black ink on white "
          "paper.",
   body=[
     "Produces an image comparable with the crops the networks were trained "
     "on. With `uniformInk` true, which is the default, one pen width is "
     "used for every writer, which is what the single-pen gantry actually "
     "draws. With it false, the pen width is searched for so that the "
     "rendered ink density matches the writer's measured density, which is "
     "the right setting when comparing against a photograph.",
   ]),

 "_MatchInkDensity": dict(
   inputs="A trajectory, the profile, and rendering parameters.",
   output="The pen width that best reproduces the writer's measured ink "
          "fraction.",
   body=[
     "A one-dimensional search over pen width. Ink fraction increases "
     "monotonically with pen width, so a simple search converges; there is "
     "no need for anything more elaborate.",
     "Why bother: the ink-trained writer namer is sensitive to ink density, "
     "so comparing synthesised output against real crops without matching "
     "density measures the pen rather than the handwriting."
   ]),

 "_LineLogProbs": dict(
   inputs="A loaded reader and a rendered image.",
   output="The reader's per-column log probabilities.",
   body=["Runs the rendered line back through the recogniser, which is the "
         "first half of checking whether the synthesiser produced readable "
         "text."]),

 "_WeakChars": dict(
   inputs="A reader, a rendered image, and the text that should be visible.",
   output="A list of positions whose characters the reader scored worst.",
   body=[
     "Finds which letters the reader is least confident about, by aligning "
     "the known text against the scores with `CtcForcedAlign` and reading "
     "off the per-character scores.",
     "This is what makes targeted repair possible. Knowing that a line reads "
     "badly is not actionable; knowing that the third and eleventh "
     "characters are the problem means those two can be redrawn and "
     "everything else left alone, which preserves the writer's own shapes "
     "everywhere the output was already readable.",
   ]),

 "REPAIR_ROUNDS": dict(output="How many repair passes to run.",
   body=["Each pass redraws the worst characters and re-scores."]),

 "REPAIR_WORST_K": dict(output="How many characters to repair per pass.",
   body=["Repairing a few at a time rather than all at once, so that the "
         "minimum number of the writer's own letters are given up."]),

 "DEFAULT_LEGIBILITY": dict(output="The legibility used when a caller does "
          "not specify and no profile value exists.",
   body=["A starting point only; `AutoLegibility` normally supplies the "
         "value."]),

 "SynthesizeLegible": dict(
   inputs="The writer, text, profile, and a reader.",
   output="A trajectory that has been drawn, checked and selectively "
          "repaired.",
   body=[
     "Draw, read back, find the weak characters, redraw just those with "
     "their fallback letterform, repeat for a few rounds.",
     "The design point is that repair is **local**. Raising the global "
     "legibility would replace every character, including the many that were "
     "already readable; replacing only the worst few spends identity exactly "
     "where legibility requires it. This is a step towards the per-character "
     "decision discussed under `AutoLegibility`, applied after the fact "
     "rather than during selection.",
   ]),

 "SynthesizeJointBestOf": dict(
   inputs="The writer, text, profile, `nTries`, a seed, a reader, a writer "
          "namer and its mapping.",
   output="The best trajectory of several attempts, with its scores recorded "
          "in `meta`.",
   body=[
     "Draws the line several times and keeps the best, where 'best' is "
     "judged by both networks at once.",
     ("sub", "Why both judges together"),
     "Optimising for the reader alone drives the output towards clean "
     "fallback letterforms, because those are the most readable thing "
     "available — the text accuracy rises and the handwriting stops being "
     "anyone's. Optimising for the namer alone drives it towards the "
     "writer's most distinctive and least legible shapes. Neither alone is "
     "the objective.",
     ("sub", "The combined score"),
     "The two scores are combined as a harmonic mean:",
     ("formula", ["S = 2*a*w / (a + w)",
                  "   a = character accuracy of the read-back",
                  "   w = the namer's confidence in the intended writer"]),
     "The harmonic mean is chosen because it is dominated by the smaller "
     "term: a draw that reads perfectly but looks like nobody scores badly, "
     "and so does one that is unmistakably the writer but unreadable. An "
     "arithmetic mean would let a very high score on one side buy a very low "
     "score on the other, which is exactly the trade this is meant to "
     "prevent.",
     ("sub", "Why repeated attempts work at all"),
     "Because glyph selection, jitter, baseline drift and joining are all "
     "random, each attempt is a different sample from the space of pages "
     "this writer might have written. Taking the best of several is a "
     "simple search over that space. Measured across writers it is worth "
     "several points on both metrics compared with a single draw, and "
     "raising the number of attempts raises both figures together, at a "
     "linear cost in time.",
     ("sub", "The caveat that must accompany any number from this function"),
     "The two networks that score the attempts are the same two that are "
     "used to report the quality of the result. A figure produced this way "
     "measures whether the pipeline can satisfy its own judges, not whether "
     "a person would read the output or recognise the hand. It is reported "
     "as such throughout, and the independent measurements in the project — "
     "segmentation line counts against typed transcripts, and the g-code "
     "round trip against an independent parser — are the ones that carry no "
     "such circularity.",
   ]),
}
