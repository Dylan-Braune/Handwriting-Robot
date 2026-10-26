"""style.py, part 2: cutting letters, filtering them, and the profile.

The cutter that divides a line into characters, the five gates that decide
whether a cut is really a letter, the prototype construction, and the
measurements that make up a writer's profile.
"""

FUNCS = {

 "_TrimLigatureTails": dict(
   inputs=[("strokes", "the glyph's strokes, normalised and ordered left to "
                       "right"),
           ("connL", "whether ink genuinely crossed the left boundary"),
           ("connR", "whether ink genuinely crossed the right boundary"),
           ("lowY", "how high a carrier stroke may run, in x-heights, "
                    "default 0.42"),
           ("minKeep", "the least fraction of the glyph's width that must "
                       "survive, default 0.18"),
           ("climbY", "the higher ceiling used by the second, steeper test, "
                      "default 1.6"),
           ("climbSlope", "the steepest slope that second test tolerates, "
                          "default 2.2")],
   output="three things: the trimmed strokes, the point where the pen "
          "entered the glyph, and the point where it left.",
   theory=[
     ("h", "The problem it solves"),
     ("p", "In joined-up writing the letters are connected by **ligatures**, "
           "the carrying strokes that run from the end of one letter to the "
           "start of the next. A cut placed between two letters therefore "
           "leaves part of a ligature attached to each: the stored glyph "
           "begins with a stroke running in from the previous letter and "
           "ends with one running out toward the next."),
     ("p", "Storing those tails is wrong for a specific reason. The glyph "
           "will later be placed next to some other letter, chosen at "
           "synthesis time, and the tail it carries was shaped to reach the "
           "letter that followed it originally. Keeping it produces a join "
           "that points at nothing, which is visible in the output as a "
           "stray lead-in stroke, or a doubled connector where the next "
           "glyph brings its own. What should be stored is the letter "
           "proper, together with a note of where the pen entered and left "
           "it, so that synthesis can draw a join appropriate to whatever "
           "pair of letters it actually places."),
     ("h", "How a tail is recognised"),
     ("p", "A ligature has a consistent geometry, because of what it is for: "
           "it carries the pen from the end of one letter across to the "
           "start of the next, along the baseline. So it runs low, it runs "
           "outward, and it runs fairly flat. Three conditions follow, and "
           "the leading points of a stroke are consumed as long as all "
           "three hold."),
     ("items", [
       "**Low.** The point sits below a ceiling, by default 0.42 "
       "x-heights, which is under half the height of a letter body. A "
       "ligature travels near the baseline; the letter itself rises above "
       "it.",
       "**Outward.** The step moves in the positive x direction. A tail "
       "entering from the left runs rightward into the letter; a stroke "
       "that doubles back is part of the letter.",
       "**Flat.** The vertical step is small relative to the horizontal "
       "one, so the stroke is travelling across rather than climbing. The "
       "slope limit is 0.7 for the low test.",
     ]),
     ("h", "Why there is a second, steeper test"),
     ("p", "Not every ligature stays low. In some hands the join into a "
           "letter such as an l or a b climbs steeply from the baseline "
           "almost to the ascender height before the letter proper begins. "
           "The low test cannot see such a tail, because it exits at the "
           "first point above the ceiling, which is only a short way in. So "
           "a second test is tried when the first finds nothing, with a "
           "much higher ceiling of 1.6 x-heights and a correspondingly "
           "looser slope limit of 2.2."),
     ("p", "The two are tried in order rather than combined, and the order "
           "is the safeguard. The low test is specific and fires on the "
           "common case; the climbing test is permissive and could consume "
           "a real stroke, so it is only consulted when the specific test "
           "found nothing at all. This is the same specific-before-general "
           "ordering used among the ownership rules in segment.py."),
     ("h", "The safeguard on how much may be removed"),
     ("p", "Whatever the tests say, a trim is only applied if what remains "
           "spans more than 18 percent of the glyph's original width. "
           "Without that condition a permissive test could consume almost "
           "the whole glyph, leaving a stub, and a stub is exactly the "
           "fragment that the later filters are built to catch. It is "
           "better to leave a tail attached, which costs a slightly wrong "
           "join, than to destroy a letter, which costs the letter."),
     ("h", "Why the right side is handled by mirroring"),
     ("p", "The exit connector leaves to the right, so it is the reverse of "
           "the entry case in both senses: it runs outward in the opposite "
           "direction and it is at the end of the stroke rather than the "
           "start. Rather than writing a second version of the test with "
           "the comparisons reversed, the stroke is reversed and its x "
           "coordinates negated, which makes an outgoing rightward tail "
           "look exactly like an incoming leftward one, and the same test "
           "is reused. Two tests that must agree are a place for them to "
           "disagree; one test applied to a transformed input cannot."),
     ("h", "Why the connection flags are required"),
     ("p", "Trimming only happens on a side where ink genuinely crossed the "
           "boundary, which the caller establishes by observing whether a "
           "traced centre line actually passed through the cut. That is the "
           "honest test for whether the pen was lifted there. Without it, "
           "the trimmer would attack the legitimate opening stroke of a "
           "letter at the start of a word, where there is no preceding "
           "letter and therefore no ligature, and the opening stroke of "
           "many letters is low, outward and flat."),
     ("note", "The entry and exit points returned are the ends of the "
              "trimmed strokes, not of the original ones, so they mark "
              "where the letter proper begins and ends. Those are the "
              "points synthesis attaches its own connector to, which is "
              "what makes a join between two glyphs meet the ink rather "
              "than the air beside it."),
   ]),

 "ExtractLineGlyphs": dict(
   inputs=[("gray", "one line crop as a grey image"),
           ("text", "the known text of that line"),
           ("model", "a loaded reader, used for alignment")],
   output="a pair: a list holding one glyph dictionary per character of the "
          "text, with nothing where a character could not be cut, and a "
          "dictionary of per-line statistics.",
   theory=[
     ("h", "What it is for"),
     ("p", "This is the central function of the file and the hardest step "
           "in the project. It takes one line of handwriting and its known "
           "text, and returns one stored letterform per character. "
           "Everything else in this file either prepares its input or "
           "judges its output."),
     ("h", "The problem: where does one letter end and the next begin"),
     ("p", "In print this question is trivial, because letters are "
           "separated by white space. In handwriting it often has no clean "
           "answer at all. In cursive the pen does not lift between "
           "letters, so there is no gap anywhere, and the boundary is a "
           "place on a continuous stroke that the writer never marked. Any "
           "cut will leave some of one letter with the other. The aim is "
           "therefore not a perfect cut, which does not exist, but the "
           "least damaging one."),
     ("h", "Step one: ask the reader where the characters are"),
     ("p", "No separate letter-finding method is used. The reader already "
           "knows, implicitly, where each character sits, because it "
           "produces character scores per column. Forced alignment extracts "
           "that, as described under CtcForcedAlign, returning for each "
           "character the span of network columns it occupies. Those spans "
           "are converted back to pixel positions in the original crop by "
           "FrameXToPixel, undoing the scaling that fitted the crop to the "
           "network's canvas."),
     ("p", "Using the recogniser for this is the design decision that "
           "makes the rest tractable. A geometric letter-finder would have "
           "to solve segmentation and recognition together, which is the "
           "problem CTC was adopted to avoid. Here the recognition is "
           "already done and the positions come free with it."),
     ("h", "Step two: the cut cost, and why the reader's boundary is moved"),
     ("p", "The aligner's boundary is approximately right but not "
           "precisely placed, because the network's columns are coarse, "
           "eight original pixels each, and because the network was never "
           "trained to put boundaries anywhere in particular. Cutting "
           "exactly there often cuts through the thick part of a letter. "
           "So the boundary is moved to the best nearby column, where best "
           "is defined by a cost that encodes what an honest cut looks "
           "like."),
     ("eq", r"\mathrm{cutCost}[x] \;=\; \mathrm{thin}[x] \;+\; "
            r"1.15 \cdot \mathrm{topAbove}[x]",
      [(r"\mathrm{cutCost}[x]", "the price of cutting at column x"),
       (r"\mathrm{thin}[x]", "how much ink the column holds, relative to "
                             "the 60th percentile of the inked columns, so "
                             "1 is a typical column and less is thinner"),
       (r"\mathrm{topAbove}[x]", "how high the highest ink in that column "
                                 "reaches above the baseline, in x-heights, "
                                 "capped at 3"),
       ("1.15", "how much the height term counts relative to the thickness "
                "term")],
      None),
     ("p", "The two terms express the two properties of a good cut. "
           "**Thin** matters because a column crossing one stroke damages "
           "less than one crossing three. **Low** matters because the place "
           "where two cursive letters genuinely meet is the ligature near "
           "the baseline, so a cut that stays low passes through the join "
           "rather than through a letter body. A column with no ink at all "
           "is given a cost of zero outright, which guarantees that a real "
           "gap, where one exists, always wins."),
     ("p", "The column profile is smoothed before use, over a window of "
           "about 0.12 x-heights, so that a single-pixel gap inside a "
           "stroke does not read as a cheap cut. Normalising the thickness "
           "against the 60th percentile rather than the maximum makes the "
           "scale robust: the maximum is set by the single thickest column "
           "on the line, often a crossing, while a percentile describes the "
           "typical column."),
     ("h", "Step three: snapping, with a pull toward the original"),
     ("p", "Within a search radius of half an x-height either side, the "
           "column of least cost is chosen, with a small penalty added for "
           "distance from the aligner's own boundary."),
     ("eq", r"x^{*} \;=\; \arg\min_{|x - x_0| \le R} \Big( "
            r"\mathrm{cutCost}[x] \;+\; 0.6 \cdot "
            r"\frac{|x - x_0|}{R} \Big)",
      [("x^{*}", "the chosen cut column"),
       ("x_0", "the aligner's boundary, the midpoint between two character "
               "centres"),
       ("R", "the search radius, half an x-height"),
       ("0.6", "the strength of the pull back toward the original "
               "boundary"),
       (r"\arg\min", "the column at which the total is smallest")],
      None),
     ("p", "The pull is a tie-breaker of the same kind as the midline "
           "penalty in SeamCuts: where several columns are about equally "
           "good, it prefers the one the recogniser nominated, because that "
           "carries information the geometry does not. The radius is "
           "limited to half an x-height for the same reason that the "
           "penalty exists at all: the aligner's boundary is "
           "approximately right, and a search allowed to wander a whole "
           "letter away could find a cheap column belonging to a different "
           "letter entirely."),
     ("h", "Step four: making the boundaries consistent"),
     ("p", "Snapping each boundary independently can produce an "
           "inconsistency: two neighbouring boundaries can land on the same "
           "column, giving a character a cell of zero width, or they can "
           "cross, giving a negative one. So the boundaries are built as a "
           "single increasing sequence, each constrained to leave at least "
           "a minimum cell width, 0.08 x-heights, between itself and its "
           "predecessor, and to leave enough room for all the cells still "
           "to come before the right-hand end."),
     ("p", "This is a small instance of a general point about pipelines of "
           "independent local decisions: each decision can be locally "
           "optimal and the set of them still be jointly invalid. The "
           "remedy here is to impose the joint constraint explicitly as the "
           "sequence is built, rather than to detect and repair violations "
           "afterwards."),
     ("h", "Step five: thin once, then cut the paths"),
     ("p", "The whole line is thinned once, as a whole, and the resulting "
           "centre lines are traced and only then divided at the "
           "boundaries. The alternative, slicing the image into cells and "
           "thinning each, is wrong, and the reason is worth stating. "
           "Thinning is a boundary-driven process: it erodes a shape from "
           "its edges inward. A cut edge is a boundary as far as the "
           "thinning is concerned, so thinning a cell produces a spurious "
           "spine running along the cut, which becomes a vertical stroke in "
           "the stored glyph that the writer never drew."),
     ("p", "Dividing the traced paths instead means a path that crosses a "
           "boundary is split into two pieces at that point, each assigned "
           "to its own character. The crossings are recorded as they "
           "happen, and they serve a second purpose: a centre line crossing "
           "a boundary means the pen never lifted there, which is the "
           "honest definition of two letters being joined, and it is what "
           "the connectedness measurement is built from."),
     ("h", "Step six: recovering the dots"),
     ("p", "A dot thins to a single pixel, which is not a path, so the "
           "tracer produces nothing for it. Any connected component of ink "
           "for which the tracer produced no path at all is therefore "
           "recovered, provided it is larger than the dust threshold, and "
           "added as a short horizontal dash across its own width. The "
           "representation is crude but correct in the respects that "
           "matter: it is in the right place, it has the right extent, and "
           "it will be drawn as a mark rather than omitted."),
     ("h", "Step seven: normalising into the stored frame"),
     ("p", "Each surviving path is simplified and then converted into the "
           "stored convention, in one expression per point."),
     ("eq", r"y_n \;=\; \frac{y_{\mathrm{base}} - y}{\mathrm{xh}}, "
            r"\qquad x_n \;=\; \frac{x - x_{\mathrm{left}}}{\mathrm{xh}} "
            r"\;-\; \tan(\mathrm{slant}) \cdot y_n",
      [("x_n,\\, y_n", "the stored coordinates, in x-heights"),
       ("x,\\, y", "the pixel coordinates in the crop"),
       ("y_{\\mathrm{base}}", "the baseline row, so subtracting it puts the "
                              "baseline at zero and makes y increase "
                              "upward"),
       ("x_{\\mathrm{left}}", "the left boundary of this character's cell, "
                              "so each glyph starts at its own origin"),
       (r"\mathrm{xh}", "the measured x-height in pixels, which is what "
                        "makes the result scale-free"),
       (r"\tan(\mathrm{slant})", "the shear that removes the writer's "
                                 "slant, described next")],
      None),
     ("h", "Why the slant is removed and then put back"),
     ("p", "The last term is a **shear**: each point is shifted "
           "horizontally in proportion to its height, which tilts the glyph "
           "upright. The reason for doing it is comparison. Two writings of "
           "the same letter at different slants are different shapes as "
           "numbers, so any averaging or distance between them is dominated "
           "by the slant rather than by the letterform. De-slanting "
           "separates the two, so that shape can be compared as shape and "
           "slant is carried once, as a single number in the profile."),
     ("p", "But the glyph will be drawn re-slanted, and its width and left "
           "edge must describe how it will look then, not how it looks "
           "upright. So after trimming, the extent is measured in the "
           "re-slanted frame by applying the shear back, and the glyph is "
           "shifted so its leftmost re-slanted point is at zero. Measuring "
           "the width upright would misstate it, because shearing a tall "
           "shape changes how much horizontal room it needs."),
     ("h", "What is stored per glyph"),
     ("items", [
       "**strokes:** the normalised paths, ordered left to right and each "
       "oriented left to right, so that two variants can be matched stroke "
       "by stroke.",
       "**width:** how much horizontal room the ink occupies, re-slanted.",
       "**advance:** how far the pen should move on to the next character, "
       "taken from the cell width. This is what preserves the writer's own "
       "letter spacing.",
       "**lead:** the gap from the left edge of the cell to where the ink "
       "starts. Without it, synthesis would butt every glyph against its "
       "neighbour and lose the writer's internal spacing.",
       "**top and bot:** the highest and lowest points, which is what the "
       "height-class filter tests.",
       "**entry and exit:** where the pen came in and went out, for drawing "
       "joins.",
       "**connL and connR:** whether ink actually crossed each boundary, "
       "the evidence for connectedness.",
     ]),
     ("note", "The returned list has one slot per character of the text, "
              "with nothing in the slots that could not be cut, rather than "
              "being a shorter list of the successes. Preserving the "
              "positions is what allows the caller to ask whether two "
              "*adjacent* characters were joined, which the connectedness "
              "measure needs and which a compacted list could not answer."),
   ]),

 "_ASCEND": dict(
   output="the set of characters expected to reach above the x-height band.",
   theory=[
     ("h", "What it encodes"),
     ("p", "An **ascender** is the part of a letter rising above the "
           "x-height band, as in b, d, f, h, k, l and t. Every capital and "
           "every digit also reaches up, because capitals are built to the "
           "ascender height and digits in most hands match them. Collecting "
           "them into one set lets a single test ask whether a glyph "
           "reaches where its character ought to."),
     ("note", "Lowercase i and j are deliberately excluded, although their "
              "dots sit high. The dot is a separate mark and is often lost "
              "or recovered crudely, so requiring height of an i would "
              "reject every correctly cut i whose dot the tracer dropped. "
              "The letter t is included even though in many hands it is "
              "shorter than a full ascender, because it still clearly "
              "exceeds the x-height band, which is all the test asks."),
   ]),

 "_DESCEND": dict(
   output="the set of characters expected to drop below the baseline.",
   theory=[
     ("h", "What it encodes"),
     ("p", "A **descender** is the part of a letter below the baseline, as "
           "in g, j, p, q and y. These five are the only lowercase letters "
           "in English with one. A glyph of one of these that never goes "
           "below the baseline is missing its tail, which means the cut "
           "took only the body, and that is a fragment."),
   ]),

 "_XONLY": dict(
   output="the set of characters expected to stay inside the x-height band.",
   theory=[
     ("h", "What it encodes"),
     ("p", "The lowercase letters with neither ascender nor descender: a, "
           "c, e, m, n, o, r, s, u, v, w, x and z. These are the letters "
           "for which the test runs the other way. Rather than requiring "
           "reach, it forbids it: a glyph of one of these that extends well "
           "above or below the band has collected ink from a neighbour, "
           "most often the ascender of the letter after it or the descender "
           "of the one before."),
     ("note", "The three sets are not exhaustive and are not meant to be. "
              "Punctuation, the space and a few letters belong to none of "
              "them, and for those the test simply passes. A filter that "
              "cannot judge a case should abstain rather than guess, "
              "because every one of these gates can only reject, and a "
              "wrong rejection is a permanently lost letterform."),
   ]),

 "_ClassOk": dict(
   inputs=[("ch", "the character this glyph is supposed to be"),
           ("top", "its highest point, in x-heights above the baseline"),
           ("bot", "its lowest point, negative below the baseline")],
   output="true if the glyph reaches where that character should reach.",
   theory=[
     ("h", "Why this is the cheapest and most reliable gate"),
     ("p", "The height class of a letter is one of the few facts about a "
           "letterform that is true for every writer. Individual hands vary "
           "enormously in slant, width, roundness and how joined they are, "
           "but an l reaches above the x-height band in all of them, "
           "because otherwise it would not be an l. A test on reach is "
           "therefore almost free of false rejections while catching a "
           "large class of bad cuts, which is why it runs first."),
     ("h", "The three tests"),
     ("eq", r"\begin{aligned}"
            r"\text{x-only:} \quad & \mathrm{top} \le 1.42 \;\text{ and }\;"
            r" \mathrm{bot} \ge -0.42 \\[2pt]"
            r"\text{ascender:} \quad & \mathrm{top} \ge 1.25 \\[2pt]"
            r"\text{descender:} \quad & \mathrm{bot} \le -0.20"
            r"\end{aligned}",
      [(r"\mathrm{top}", "the glyph's highest point, in x-heights, where "
                         "1.0 is the top of the x-height band"),
       (r"\mathrm{bot}", "its lowest point, negative below the baseline"),
       ("1.42", "how far above the band an x-only letter may stray before "
                "it is judged to have collected a neighbour's ascender"),
       ("1.25", "how far above the band an ascender letter must reach"),
       ("-0.20", "how far below the baseline a descender letter must drop"),
       ("-0.42", "how far below an x-only letter may stray")],
      None),
     ("h", "Why the thresholds are not at 1.0"),
     ("p", "The band is nominally 0 to 1, so it might seem that an x-only "
           "letter should be required to stay below 1.0 and an ascender to "
           "exceed it. The margins exist because the band itself is "
           "measured, not known. CoreBand estimates the x-height from a row "
           "profile, and that estimate carries error, so a letter sitting "
           "exactly at the top of the true band can measure as 1.1. The "
           "limits are placed outside the range of that measurement error."),
     ("p", "The asymmetry between 1.42 and 1.25 is deliberate and follows "
           "from the direction each test guards. An x-only letter is being "
           "checked for contamination, which when it happens is gross: a "
           "swallowed ascender reaches 1.7 or more, so the limit can be "
           "placed generously at 1.42 and still catch it. An ascender "
           "letter is being checked for truncation, and must be given the "
           "benefit of the doubt only as far as is safe, so 1.25 sits just "
           "above where a complete x-only letter could reach. Between 1.25 "
           "and 1.42 both tests would pass, which is intentional: that band "
           "is where the measurement is genuinely ambiguous, and neither "
           "test should fire there."),
   ]),

 "_PenLength": dict(
   inputs=[("g", "a glyph dictionary")],
   output="the total length of all its strokes, in x-heights.",
   theory=[
     ("h", "What it measures"),
     ("p", "How far the pen travelled to draw this glyph, obtained by "
           "summing the lengths of every stroke, each of which is the sum "
           "of the straight-line distances between its consecutive points. "
           "It is the quantity the fragment test compares against a "
           "per-character floor, for the reasons set out under "
           "FRAGMENT_PEN_FRAC."),
     ("h", "Why total rather than per stroke"),
     ("p", "A letter may legitimately be drawn in one stroke or several, "
           "depending on the hand and on whether the pen lifted. The same "
           "letter from the same writer can appear with two strokes on one "
           "line and one on the next, where a join happened to survive the "
           "thinning. The total is invariant to that, so it measures the "
           "letter rather than the accident of how it came out."),
   ]),

 "_LongestStroke": dict(
   inputs=[("g", "a glyph dictionary")],
   output="the length of its single longest stroke, in x-heights.",
   theory=[
     ("h", "What it measures, and why it is a separate test"),
     ("p", "The longest single continuous stroke. Where the total pen "
           "length asks how much writing is present, this asks whether any "
           "of it is a real stroke. The two are independent: a scatter of "
           "eight short scraps passes the total test and fails this one, "
           "while a single clean stroke that is slightly short passes this "
           "and fails the total. Keeping them separate means each can be "
           "set to catch its own defect, rather than one compromise "
           "threshold catching neither well."),
     ("note", "Both measures ignore strokes of fewer than two points, which "
              "have no length to contribute. A single isolated point can "
              "survive into a glyph from the dot recovery, and including it "
              "as a zero-length stroke would be harmless here but would "
              "divide by zero in the resampling that forms prototypes."),
   ]),

 "_ShapeGrid": dict(
   inputs=[("g", "a glyph dictionary"),
           ("n", "the grid resolution, default 12")],
   output="a vector of 144 numbers describing the glyph's shape, scaled to "
          "unit length; or nothing if the glyph has no points.",
   theory=[
     ("h", "What it is for"),
     ("p", "Several filters need to ask how similar two glyphs are, and one "
           "needs to ask how similar a glyph is to the general idea of a "
           "letter. Comparing two lists of strokes directly is not "
           "workable: they may have different numbers of strokes, different "
           "numbers of points, and the points need not correspond. What is "
           "needed is a **descriptor**, a fixed-length vector computed from "
           "a shape such that similar shapes give similar vectors. Then "
           "comparing shapes becomes comparing vectors, which is "
           "arithmetic."),
     ("h", "Step one: turn strokes into points"),
     ("p", "Each segment of each stroke is filled in with evenly spaced "
           "points along it, at a density of about 24 per unit length. This "
           "is necessary because the stored strokes have been simplified, "
           "so a long straight run is two points, and counting points "
           "directly would weight a simplified stroke far less than a "
           "wiggly one of the same length. Filling in restores the "
           "proportionality between a stroke's length and the number of "
           "points representing it."),
     ("h", "Step two: scale the shape into the grid"),
     ("p", "The points are rescaled so the shape exactly fills the grid in "
           "both directions, and each point is assigned to a cell. This "
           "discards absolute size and aspect ratio deliberately, so that "
           "the same letter written large and small gives the same "
           "descriptor. Size is already tested separately by the width and "
           "height gates, so there is nothing lost by removing it here, and "
           "a great deal gained: the descriptor then responds to shape "
           "alone."),
     ("h", "Step three: count, then damp"),
     ("eq", r"G[i,j] \;=\; \sqrt{\,\big|\{\,\text{points falling in cell } "
            r"(i,j)\,\}\big|\,}",
      [("G[i,j]", "the value stored for grid cell i, j"),
       (r"|\{\cdot\}|", "the number of points that landed in that cell")],
      None),
     ("p", "The square root is a **damping** transformation. Without it, a "
           "cell where the pen dwelt, at a sharp corner or where two "
           "strokes overlap, accumulates many times the count of an "
           "ordinary cell and dominates the whole descriptor, so two "
           "glyphs would be judged similar or not chiefly by where their "
           "pen dwelt. Taking the square root compresses large counts "
           "toward small ones while keeping the ordering, so a cell with "
           "ink still counts for more than an empty one but not "
           "overwhelmingly more."),
     ("h", "Step four: blur"),
     ("p", "The grid is blurred with a small 3 by 3 kernel, weight 0.25 in "
           "the centre and 0.09375 at each of the eight neighbours. The "
           "problem it solves is that a grid is a hard partition: two "
           "writings of the same letter that differ by a hair can put "
           "corresponding ink in adjacent cells, which as vectors makes "
           "them share nothing at all in those positions. Blurring spreads "
           "each cell's value into its neighbours, so near-misses "
           "contribute to the similarity instead of being lost. This is the "
           "same reasoning that puts tolerance into any grid-based "
           "descriptor, and without it the measure would be far too "
           "brittle to be useful."),
     ("h", "Step five: normalise to unit length"),
     ("eq", r"\mathbf{v} \;=\; \frac{\mathbf{g}}{\|\mathbf{g}\|}, "
            r"\qquad \|\mathbf{g}\| \;=\; \sqrt{\textstyle\sum_{k} g_k^2}",
      [(r"\mathbf{v}", "the returned descriptor, of length 1"),
       (r"\mathbf{g}", "the blurred grid flattened into a 144-element "
                       "vector"),
       (r"\|\mathbf{g}\|", "its Euclidean length, the square root of the "
                           "sum of the squares of its entries")],
      None),
     ("p", "Scaling to unit length removes the total amount of ink from the "
           "descriptor, leaving only how it is distributed. The decisive "
           "benefit is what it does to the comparison: for two unit "
           "vectors, the dot product is exactly the cosine of the angle "
           "between them, which is 1 for identical shapes and falls toward "
           "0 as they diverge. So similarity becomes a single "
           "multiplication and sum, with a value on a fixed and "
           "interpretable scale. This is **cosine similarity**, and it is "
           "the standard way to compare descriptors precisely because "
           "normalisation makes magnitude irrelevant."),
     ("note", "The resolution of 12 by 12 is a compromise. Coarser would "
              "stop distinguishing letters that differ in a small region, "
              "such as a and o, which agree everywhere but the stem. Finer "
              "would make the descriptor sensitive to exactly the "
              "variation between two writings of one letter that it is "
              "meant to tolerate, and the blur would have to widen to "
              "compensate, which gives back the resolution that was bought."),
   ]),

 "_LETTER_PRIOR": dict(
   output="a cache holding the reference shape for each letter, empty until "
          "filled.",
   theory=[
     ("h", "What it is for"),
     ("p", "A module-level dictionary that holds the cross-writer reference "
           "shapes once they have been computed, so that building them, "
           "which requires descriptors for every variant of every writer, "
           "happens once rather than once per writer processed."),
     ("note", "A mutable value shared across a whole module is a pattern "
              "worth being cautious with, because anything that writes to "
              "it affects every later caller invisibly. It is acceptable "
              "here because what it holds is derived from fixed data and so "
              "is the same whenever it is computed, which means a stale "
              "entry cannot be wrong, only old."),
   ]),

 "BuildLetterPrior": dict(
   inputs=[("libsByAuthor", "a mapping from writer to that writer's glyph "
                            "library"),
           ("minSamples", "the least number of examples a letter needs "
                          "before a reference is formed, default 12")],
   output="a mapping from character to a reference descriptor, including "
          "only the letters with enough examples.",
   theory=[
     ("h", "What it is for, and the one thing it must not do"),
     ("p", "Several of this project's gates reject a glyph for not looking "
           "enough like the letter it claims to be, which requires knowing "
           "what that letter looks like. This function supplies that: for "
           "each character it pools every variant from every writer and "
           "forms a single reference descriptor, the **prior**, meaning "
           "what is expected before any particular writer is considered."),
     ("p", "The restriction on its use matters as much as its "
           "construction. It is used only to reject a writer's own bad "
           "cuts. It is never used to supply a shape, so no part of another "
           "person's handwriting ever enters a writer's profile. The "
           "distinction is the difference between using the pool as a "
           "quality check and using it as a source, and only the first is "
           "legitimate for a system whose purpose is to reproduce an "
           "individual's hand."),
     ("h", "Why the median across writers"),
     ("eq", r"\mathbf{p}_c \;=\; \frac{\mathbf{m}_c}"
            r"{\|\mathbf{m}_c\|}, \qquad "
            r"\mathbf{m}_c[k] \;=\; \mathrm{median}\big\{\, "
            r"\mathbf{v}^{(1)}[k],\, \dots,\, \mathbf{v}^{(N)}[k] \,\big\}",
      [(r"\mathbf{p}_c", "the reference descriptor for character c"),
       (r"\mathbf{m}_c", "the elementwise median across every pooled "
                         "example"),
       (r"\mathbf{v}^{(i)}", "the descriptor of the i-th pooled example of "
                             "that character"),
       ("k", "which of the 144 descriptor positions"),
       ("N", "how many examples were pooled"),
       (r"\|\cdot\|", "the length, so the reference is itself a unit vector "
                      "and can be compared by dot product")],
      None),
     ("p", "The median is taken independently in each of the 144 positions, "
           "and it is chosen over the mean for a reason that is central to "
           "whether this works at all. The pool is built from unfiltered "
           "glyph libraries, so it contains fragments and over-wide cuts: "
           "the very defects the prior is meant to detect. A mean would "
           "absorb them, shifting the reference toward the typical defect, "
           "and the gate built on it would then fail to reject what it was "
           "built to reject. The median ignores them entirely so long as "
           "they are a minority, which they are."),
     ("p", "Pooling across writers is what makes the reference describe the "
           "letter rather than any one hand. Individual idiosyncrasies "
           "differ between writers and so disagree in the pool, where the "
           "median removes them; what every writer's a has in common is the "
           "thing that makes it an a, and that is what survives."),
     ("h", "Why twelve examples are required"),
     ("p", "A median over very few samples is not robust: with four "
           "examples, two of them fragments, the median sits between a "
           "fragment and a letter. Twelve is enough that a minority of "
           "defects cannot reach the middle. Characters with fewer are "
           "simply omitted, and the gates then abstain on them rather than "
           "testing against an unreliable reference, which is the correct "
           "behaviour for a filter that can only reject."),
   ]),

 "_PriorScore": dict(
   inputs=[("g", "a glyph dictionary"),
           ("ch", "the character it claims to be"),
           ("prior", "the mapping of reference descriptors")],
   output="a distance from 0, identical to the reference, upward; 0 when no "
          "comparison can be made.",
   theory=[
     ("h", "What it measures"),
     ("p", "How far this glyph's shape sits from the cross-writer idea of "
           "its letter. It is the quantity stored on each glyph as "
           "`priorD`, and synthesis later uses it to judge how far a "
           "writer's own stored shape can be trusted."),
     ("eq", r"D \;=\; 1 \;-\; \mathbf{v} \cdot \mathbf{p}_c",
      [("D", "the returned distance: 0 when the glyph matches the reference "
             "exactly, larger as it diverges"),
       (r"\mathbf{v}", "the glyph's own descriptor, a unit vector"),
       (r"\mathbf{p}_c", "the reference descriptor for that character, also "
                         "a unit vector"),
       (r"\cdot", "the dot product, which for unit vectors is the cosine of "
                  "the angle between them")],
      None),
     ("p", "Subtracting the cosine similarity from 1 converts a similarity "
           "into a distance, so that small means good. This is the "
           "**cosine distance**. Since both descriptors have non-negative "
           "entries, their dot product cannot be negative, so the result "
           "lies between 0 and 1."),
     ("h", "Why a missing reference scores zero rather than failing"),
     ("p", "When no reference exists for a character, or the glyph yields "
           "no descriptor, the score returned is 0, which is the value "
           "meaning a perfect match. That is deliberate: it makes the gate "
           "abstain. Every filter in this file can only reject, so the safe "
           "value in the absence of evidence is the one that rejects "
           "nothing. Returning a large value instead would quietly delete "
           "every glyph of every rare character, since those are exactly "
           "the characters without enough examples to have a reference."),
   ]),

 "PriorFilter": dict(
   inputs=[("variants", "this writer's glyphs for one character"),
           ("ch", "that character"),
           ("prior", "the mapping of reference descriptors"),
           ("keepFrac", "the fraction to keep, default 0.7"),
           ("minKeep", "the least number to keep, default 3")],
   output="the kept variants, sorted best first.",
   theory=[
     ("h", "What it does"),
     ("p", "Sorts this writer's variants of one character by their distance "
           "from the cross-writer reference and keeps the best 70 percent. "
           "The worst 30 percent are the ones least like the letter is "
           "supposed to look, which is where the mis-cuts that survived the "
           "geometric gates are concentrated."),
     ("h", "Why a proportion rather than a threshold"),
     ("p", "A fixed distance threshold would be wrong per writer, for "
           "exactly the reason given for the alignment confidence "
           "percentile. A writer whose hand is unusual sits further from "
           "the cross-writer reference in every variant, so a threshold "
           "tuned on a conventional hand would reject that writer's letters "
           "wholesale. Keeping a proportion asks which of this writer's own "
           "variants are the least typical, which is a question about them "
           "rather than about the population, and every writer therefore "
           "keeps their best seven tenths."),
     ("h", "The floor, and the abstention"),
     ("p", "At least three variants are kept whatever the proportion says, "
           "and the filter does nothing at all when fewer than three exist "
           "or when no reference is available for the character. Three is "
           "the smallest number from which the consensus filter downstream "
           "can work and from which a prototype can be formed, so dropping "
           "below it costs more than the bad variant does. For a rare "
           "character, having three imperfect examples is better than "
           "having one perfect one, because synthesis needs something to "
           "choose among and a prototype needs several shapes to average."),
     ("note", "The variants come back sorted best first, which is not "
              "incidental. Later stages take the first few, and the ranking "
              "established here is part of what determines which of a "
              "writer's shapes actually get drawn."),
   ]),

 "_ConsensusFilter": dict(
   inputs=[("variants", "this writer's glyphs for one character"),
           ("keepFrac", "the fraction to keep, default 0.75")],
   output="the kept variants, ordered by how typical they are.",
   theory=[
     ("h", "What it adds that the prior filter cannot"),
     ("p", "The prior filter compares a writer against everyone else, which "
           "catches a shape that is not really the letter. This one "
           "compares a writer against themselves, which catches a shape "
           "that is not really *their* letter. The two find different "
           "faults. A glyph carrying half of its neighbour may still "
           "resemble the general idea of the letter closely enough to pass "
           "the prior, while differing clearly from the writer's own "
           "twenty other examples, because those twenty agree with each "
           "other and it does not."),
     ("h", "Background: the medoid"),
     ("p", "The most typical member must be identified before the outliers "
           "can be found. A mean cannot serve, because the mean of several "
           "descriptors is not itself any real glyph and, being an average, "
           "is pulled by the outliers. Instead the **medoid** is used: the "
           "member of the set whose total distance to all the others is "
           "least. It is an actual glyph, and it is the multidimensional "
           "analogue of the median, with the same robustness."),
     ("eq", r"m \;=\; \arg\min_{i} \sum_{j} \big\| \mathbf{v}^{(i)} - "
            r"\mathbf{v}^{(j)} \big\|_{1}, \qquad "
            r"\big\|\mathbf{a} - \mathbf{b}\big\|_{1} \;=\; "
            r"\sum_{k} \big| a_k - b_k \big|",
      [("m", "the index of the medoid variant"),
       (r"\mathbf{v}^{(i)}", "the descriptor of variant i"),
       (r"\|\cdot\|_{1}", "the **L1 distance**, or Manhattan distance: the "
                          "sum of the absolute differences, as opposed to "
                          "the square root of the sum of squares"),
       ("k", "which descriptor position"),
       (r"\arg\min_i", "the variant for which that total is smallest")],
      None),
     ("p", "The L1 distance is used rather than the ordinary straight-line "
           "distance because squaring amplifies large differences, so a "
           "single badly disagreeing cell would dominate the comparison. "
           "Summing absolute differences treats every cell's disagreement "
           "in proportion to its size, which is the behaviour wanted when "
           "the question is overall typicality rather than worst-case "
           "mismatch. It is the same robustness argument that favours the "
           "median over the mean."),
     ("h", "The filter itself"),
     ("p", "With the medoid identified, the variants are ordered by "
           "distance from it and the furthest quarter is dropped. Ordering "
           "by distance from the most typical member means the kept "
           "variants are the ones that agree with the writer's own "
           "consensus, and the ordering also survives into the output, so "
           "later stages taking the first few get the most typical ones."),
     ("h", "Why both filters are run, in this order"),
     ("p", "The prior filter runs first and the consensus filter second. "
           "The order matters because the consensus filter trusts the set "
           "it is given: it finds the most typical member of whatever it "
           "receives. If most of the variants were mis-cuts, the medoid "
           "would be a mis-cut and the filter would discard the correct "
           "letters as outliers. Running the prior filter first removes the "
           "glyphs that are not the letter at all, using an external "
           "reference that cannot be corrupted by this writer's cuts, so "
           "the consensus filter operates on a set in which the majority "
           "really is correct, which is the condition it needs."),
     ("note", "The distance matrix is computed between every pair, which is "
              "quadratic in the number of variants. With at most a few "
              "dozen variants per character that is trivial, and the whole "
              "matrix is wanted anyway, since the medoid is defined by row "
              "sums of exactly that matrix."),
   ]),

 "_ResampleN": dict(
   inputs=[("poly", "a polyline"), ("n", "how many points to produce")],
   output="an array of n points evenly spaced along the path.",
   theory=[
     ("h", "What it is for"),
     ("p", "Averaging several writings of a letter requires adding "
           "corresponding points together, which requires that they "
           "correspond. Traced strokes have no correspondence at all: one "
           "may have 40 points and another 55, placed wherever the pixel "
           "grid and the simplification happened to put them. Resampling "
           "puts every stroke on the same footing, n points spaced equally "
           "along its length, so that the k-th point of one stroke and the "
           "k-th of another both sit at the same fraction of the way "
           "along."),
     ("h", "How it works: arc-length parameterisation"),
     ("p", "The cumulative distance along the path is computed, giving for "
           "each original point how far along the path it sits. That turns "
           "the path into a function of distance, and the new points are "
           "then read off at n equally spaced distances by linear "
           "interpolation between the original points."),
     ("eq", r"s_i \;=\; \sum_{j<i} \big\| \mathbf{p}_{j+1} - "
            r"\mathbf{p}_{j} \big\|, \qquad "
            r"t_k \;=\; \frac{k}{n-1}\, s_{\mathrm{last}}, \qquad "
            r"k = 0, \dots, n-1",
      [("s_i", "the cumulative distance from the start of the path to its "
               "i-th point"),
       (r"\mathbf{p}_j", "the j-th original point"),
       (r"\|\cdot\|", "the straight-line distance between two consecutive "
                      "points"),
       ("t_k", "the distance at which the k-th new point is placed"),
       ("s_{\\mathrm{last}}", "the total path length"),
       ("n", "how many points are wanted")],
      None),
     ("p", "This is called **arc-length parameterisation**, and the "
           "alternative makes clear why it is necessary. Spacing the new "
           "points evenly by index rather than by distance would place them "
           "densely wherever the original happened to be dense, so two "
           "strokes resampled that way would still not correspond. "
           "Distance is the only parameterisation that both strokes share."),
     ("note", "A path of zero length, which a dot recovery can produce, "
              "returns n copies of its single point. That is the correct "
              "degenerate answer and it keeps the array shape uniform, so "
              "the averaging that follows needs no special case."),
   ]),

 "_AlignStroke": dict(
   inputs=[("s", "one stroke"), ("N", "the number of points to resample to")],
   output="the stroke resampled to N points with a consistent starting point "
          "and direction, or nothing if it is too short.",
   theory=[
     ("h", "The problem beyond resampling"),
     ("p", "Resampling makes two strokes have the same number of points, "
           "but corresponding points still need to be at corresponding "
           "places, and two things can spoil that. A stroke may be traced "
           "in either direction, so one writing of a letter may run left to "
           "right and another right to left, in which case point 0 of one "
           "corresponds to point N−1 of the other. And a closed loop has no "
           "inherent starting point at all, so two tracings of the same o "
           "may begin anywhere on the circle."),
     ("h", "The two fixes"),
     ("items", [
       "**Open strokes are oriented left to right.** If the stroke ends "
       "left of where it began, it is reversed. Left to right is chosen "
       "because it is the direction of writing, so it is the orientation "
       "most strokes already have.",
       "**Closed strokes are rotated to begin at their leftmost point.** A "
       "stroke is recognised as closed when its two ends come within 0.3 "
       "x-heights of each other. The points are then rotated so the "
       "leftmost becomes first, and the original start is appended to close "
       "the loop. The leftmost point is used because it is well defined for "
       "any loop and is found identically in every tracing of it.",
     ]),
     ("p", "Both fixes work by choosing a canonical form: a rule that "
           "picks, from among the several equivalent ways of representing "
           "the same stroke, one particular way that depends only on the "
           "shape. Any such rule works provided every stroke obeys the same "
           "one, which is the real requirement."),
     ("note", "Without these fixes the averaging would not merely be "
              "inaccurate; it would be nonsense. Averaging a stroke with a "
              "reversed copy of itself yields a shape collapsing toward its "
              "own midpoint, since the first point of one is averaged with "
              "the last of the other. The prototype would be a smear "
              "through the middle of the letter."),
   ]),

 "_DenoisedPrototype": dict(
   inputs=[("variants", "this writer's kept glyphs for one character"),
           ("minN", "the least variants needed, default 3"),
           ("N", "the number of points per stroke, default 44")],
   output="one clean letterform as a list of strokes, or nothing if no "
          "reliable prototype can be formed.",
   theory=[
     ("h", "What it is for"),
     ("p", "Every individual variant carries noise: the staircase left by "
           "the pixel grid, a boundary placed a little wrongly, a tremor in "
           "the hand. Those errors differ between variants while the letter "
           "does not, so averaging several writings cancels the noise and "
           "leaves the shape the writer meant. The result is the "
           "**prototype**, the cleanest available statement of how this "
           "writer forms this letter."),
     ("h", "Step one: group by stroke count"),
     ("p", "Only variants with the same number of strokes can be averaged, "
           "because the averaging matches strokes one to one. The variants "
           "are grouped by stroke count and the largest group is used. "
           "Taking the largest rather than any particular count lets the "
           "data decide how the writer usually forms the letter: if most of "
           "their a's are one stroke, the prototype is a one-stroke a."),
     ("p", "A group with fewer than three members, or with more than five "
           "strokes, is refused outright. Three is the least from which an "
           "average means anything. The five-stroke limit guards against "
           "averaging glyphs whose stroke counts are high because the "
           "tracing fragmented them, where the stroke-to-stroke matching "
           "would be matching pieces that do not correspond."),
     ("h", "Step two: match strokes between variants"),
     ("p", "Within the group, each variant's strokes are sorted by their "
           "leftmost point and matched by position in that order, then "
           "aligned and resampled to a common 44 points. Sorting by "
           "position is a weak way to establish correspondence, but a "
           "sound one for this case: the strokes of one letter written by "
           "one person occur in a consistent left-to-right arrangement, so "
           "the leftmost stroke of one a is the same stroke as the leftmost "
           "of another."),
     ("h", "Step three: iterative alignment to the median"),
     ("p", "This is the step that distinguishes the method from a plain "
           "average. The variants are not yet aligned horizontally, because "
           "each was cut at a slightly different boundary, so one may sit a "
           "tenth of an x-height right of another. Averaging them in that "
           "state blurs the prototype horizontally. So each variant is "
           "shifted along x to whichever offset, among nine candidates "
           "spanning plus and minus 0.22 x-heights, brings it closest to "
           "the current median, and the median is then recomputed from the "
           "shifted variants. That is repeated twice."),
     ("eq", r"\delta^{*}_c \;=\; \arg\min_{\delta} \; \frac{1}{N} "
            r"\sum_{k=1}^{N} \Big\| \big(\mathbf{c}_k + "
            r"\delta\,\hat{\mathbf{x}}\big) - \mathbf{m}_k \Big\|",
      [(r"\delta^{*}_c", "the best horizontal shift for variant c"),
       (r"\mathbf{c}_k", "the k-th point of that variant's stroke"),
       (r"\mathbf{m}_k", "the k-th point of the current median stroke"),
       (r"\hat{\mathbf{x}}", "the unit vector along the horizontal axis, so "
                             "the shift moves x only"),
       (r"\delta", "a candidate shift, one of nine values from −0.22 to "
                   "+0.22 x-heights"),
       ("N", "the number of points per stroke, 44")],
      None),
     ("p", "The alternation of aligning to an estimate and then re-forming "
           "the estimate is the general shape of an **iterative refinement**, "
           "the same pattern as the classical algorithms for registering "
           "point sets. It is not guaranteed to find the best possible "
           "joint alignment, but each step cannot make the agreement worse, "
           "and two rounds are enough here because the misalignment being "
           "corrected is small and the first round removes most of it."),
     ("h", "Step four: reject the worst fifth, then take the median"),
     ("p", "After alignment, each variant's mean deviation from the median "
           "is computed, and the worst 20 percent are dropped before the "
           "final median is taken. This is a second line of defence: a "
           "variant that survived every earlier gate but is still wrong "
           "will disagree with the aligned consensus, and here it is "
           "identified by that disagreement alone, with no reference to "
           "what kind of defect it has."),
     ("p", "The median, not the mean, at every stage. Its robustness is "
           "exactly what is wanted: a prototype must not be bent toward a "
           "single bad variant, and with the median one outlier among "
           "several has no influence at all on the result rather than a "
           "proportional one."),
     ("note", "Returning nothing is common and is not a failure. A "
              "character with few examples, or whose examples disagree "
              "about stroke count, has no reliable prototype, and inventing "
              "one from two variants would produce a shape less trustworthy "
              "than either. The caller falls back on the writer's actual "
              "variants, which is the right behaviour: a real if imperfect "
              "letterform beats a confidently wrong average."),
   ]),

 "SmoothStroke": dict(
   inputs=[("pts", "a stroke"),
           ("step", "the spacing to resample to, in x-heights, default "
                    "0.055"),
           ("passes", "how many smoothing passes, default 2")],
   output="the stroke resampled evenly and smoothed, as rounded "
          "coordinates.",
   theory=[
     ("h", "What it is for"),
     ("p", "A traced stroke still carries the pixel staircase: it was "
           "assembled from whole-pixel positions, so it zigzags at the "
           "scale of one pixel even where the writer's stroke was smooth. "
           "Drawn by a plotter that staircase is visible, and it also "
           "misrepresents the writing, since the writer drew a curve. This "
           "function removes it."),
     ("h", "Step one: resample evenly"),
     ("p", "The stroke is first resampled to points about 0.055 x-heights "
           "apart. Even spacing is required before smoothing, because the "
           "smoothing averages each point with its neighbours by index, and "
           "if the points are unevenly spaced then that averaging reaches "
           "different distances in different places, smoothing the sparse "
           "parts far more than the dense ones."),
     ("h", "Step two: the smoothing pass"),
     ("eq", r"q'_k \;=\; 0.25\,q_{k-1} \;+\; 0.5\,q_k \;+\; "
            r"0.25\,q_{k+1}",
      [("q'_k", "the new position of the k-th point"),
       ("q_k", "its current position"),
       ("q_{k-1},\\, q_{k+1}", "its neighbours along the stroke"),
       ("0.25,\\,0.5,\\,0.25", "the weights, which sum to 1 so the stroke "
                               "is not scaled or shifted")],
      None),
     ("p", "This is a convolution along the stroke with a three-point "
           "kernel, the same operation as the profile smoothing in "
           "LinePitch but applied to a path rather than a signal. Weights "
           "summing to 1 are what guarantee the stroke keeps its position "
           "and size; weights that summed to less would shrink it toward "
           "the origin a little with every pass."),
     ("p", "Two passes are applied. Applying the same small kernel "
           "repeatedly is equivalent to one wider kernel, and it is the "
           "cheaper way to get a gentle wide smoothing, since each pass "
           "costs three multiplications per point. Two passes of this "
           "kernel remove the one-pixel zigzag while leaving the curvature "
           "of a letter, which acts over many points, essentially "
           "untouched."),
     ("h", "Why the endpoints are held fixed"),
     ("p", "Only the interior points are smoothed; the first and last are "
           "kept exactly. The endpoints carry meaning that the interior "
           "does not: they are where the pen entered and left, they are "
           "what a join is drawn to, and they define the glyph's extent. "
           "Smoothing them would pull them inward, shortening every stroke "
           "slightly with each pass and, worse, moving the attachment "
           "points that synthesis relies on."),
     ("note", "Coordinates are rounded to three decimal places on output. "
              "At a scale where 1 is the x-height, three decimals is a "
              "thousandth of a letter's height, far finer than any plotter "
              "can draw or any descriptor can distinguish. The rounding "
              "roughly halves the size of the stored profile, which runs to "
              "thousands of strokes, at no cost in fidelity."),
   ]),

 "RenderRefStats": dict(
   inputs=[("gray", "one real line crop from this writer")],
   output="a dictionary of measurements describing how that line looks, "
          "both to the networks and geometrically.",
   theory=[
     ("h", "What it is for, and the idea behind it"),
     ("p", "Reproducing a hand is not only a matter of letter shapes. The "
           "output must also *look* like a photograph of handwriting to "
           "the networks that judge it, and that involves properties no "
           "glyph library captures: how much of the image is ink, how dark "
           "the ink is, how bright the paper is, the proportions of the "
           "line. This function measures those on a real line, so that "
           "synthesis has a target to match rather than having to guess."),
     ("h", "Background: domain shift"),
     ("p", "**Domain shift** is the name for the situation where a model is "
           "used on data drawn from a different distribution than it was "
           "trained on. A network trained on photographs of handwriting has "
           "learned the statistics of photographs: paper that is not pure "
           "white, ink that is not pure black, and soft grey edges where "
           "the two meet. Rendering a synthetic line as pure black on pure "
           "white produces an image that is correct as geometry but "
           "statistically unlike anything in training, and the network's "
           "response to it is unreliable, which would corrupt the judgement "
           "of the output."),
     ("p", "This is why the paper and ink levels are measured as "
           "percentiles of the real crop rather than assumed: the median of "
           "the bright pixels is what this writer's paper actually is, and "
           "the median of the dark pixels is what their ink actually is. "
           "The fraction of intermediate greys is measured too, since that "
           "describes the softness of the stroke edges."),
     ("h", "Why the word gap is measured from the ink, not the recogniser"),
     ("p", "The width of the space character as the aligner reports it is "
           "not a reliable measure of how widely this writer spaces their "
           "words. CTC concentrates its output and tends to assign a space "
           "a narrow territory, well inside the actual gap. So the gap is "
           "measured directly instead: the columns with no ink are found, "
           "the runs between inked columns are collected, those under half "
           "an x-height are discarded as being gaps within a word, and the "
           "median of the rest is taken."),
     ("p", "The half-x-height threshold is what separates the two kinds of "
           "gap. Within a word the pen may lift, leaving a small gap, but a "
           "word boundary is a deliberate space and is substantially wider. "
           "The median of the wide gaps therefore describes word spacing, "
           "and the median again rather than the mean, so that one unusually "
           "wide gap at a paragraph end does not inflate it."),
     ("h", "Why slant, ascender and descender are measured here too"),
     ("p", "These are also measured in the profile from the glyphs "
           "themselves, so measuring them again on the rendered line looks "
           "redundant. It is not, and the comment in the code gives the "
           "reason: these are measured *the same way* they will be measured "
           "on the synthesised output. A quantity measured by two different "
           "methods is not comparable with itself, so to check whether the "
           "synthesis reproduced the writer's slant, both sides must be "
           "measured by the same procedure. These values exist to be the "
           "reference half of that comparison."),
     ("note", "Fields that cannot be measured are filled with a not-a-number "
              "value rather than a plausible default. That is deliberate: "
              "the caller takes a median across lines with those values "
              "excluded, so a line that could not be measured contributes "
              "nothing, whereas a default would contribute a fabricated "
              "measurement that is indistinguishable from a real one."),
   ]),

 "_MedianWordGap": dict(
   inputs=[("refs", "the per-line reference statistics"),
           ("spaceAdvs", "the widths the aligner gave to space characters")],
   output="the typical gap this writer leaves between words, in x-heights.",
   theory=[
     ("h", "The preference order"),
     ("p", "Three sources are tried in order. The measured ink gaps from "
           "RenderRefStats are preferred, because they measure the real "
           "thing. If none is available, the aligner's space widths are "
           "used, which are biased narrow for the reason given above but "
           "are at least derived from this writer. If there are no spaces "
           "at all, 1.2 x-heights is returned as a generic value."),
     ("h", "Why the fallback order is this way round"),
     ("p", "It descends from the most direct evidence to the least. The "
           "same cascade pattern appears in LinePitch and in the checkpoint "
           "selection: try the best-informed method, fall back only when it "
           "cannot apply. The consequence worth noting is that the fallback "
           "is known to be biased, so a profile built from a writer with no "
           "measurable gaps will have word spacing that is too narrow, and "
           "that is a better failure than having none at all."),
     ("note", "Not-a-number values are dropped by testing whether each "
              "value equals itself, which is false only for not-a-number. "
              "That is the standard idiom and it works without importing "
              "anything, which is why it is used in a file that avoids "
              "dependencies."),
   ]),

 "ExtractAuthorRaw": dict(
   inputs=[("authorId", "which writer these lines belong to"),
           ("lineItems", "pairs of a line crop and its known text"),
           ("model", "a loaded reader for alignment")],
   output="a list of per-line results, each a pair of the glyphs cut from "
          "that line and its statistics.",
   theory=[
     ("h", "What it is for"),
     ("p", "Runs the cutter over every line of one writer and collects the "
           "results. It is the expensive half of profile building, and the "
           "half that gets cached."),
     ("h", "Why the output is deliberately unfiltered"),
     ("p", "Nothing is rejected here. Every glyph the cutter produced is "
           "kept, including the fragments and the over-wide cuts. This is "
           "the design decision that makes the filters tunable: the "
           "filtering happens in BuildAuthorProfile, which runs on this "
           "output, so a threshold can be changed and the effect measured "
           "in seconds against a cache that is still valid. Had filtering "
           "happened here, every change to a filter would require "
           "re-reading and re-cutting every page."),
     ("p", "It is also what made the project's rejection statistics "
           "possible. Reporting what fraction of cuts were discarded, and "
           "why, requires having kept the discarded ones; a pipeline that "
           "filtered as it extracted could report only what survived."),
     ("note", "A line whose extraction returned nothing is skipped rather "
              "than recorded as empty. Those are the lines where alignment "
              "failed outright or no x-height band could be found, which "
              "carry no glyphs and no usable statistics, so an entry for "
              "them would only have to be filtered out again by every "
              "consumer."),
   ]),

 "BuildAuthorProfile": dict(
   inputs=[("authorId", "which writer this profile is for"),
           ("parsed", "the raw per-line extraction from ExtractAuthorRaw"),
           ("refs", "the per-line reference statistics, taken from the "
                    "extraction if not supplied"),
           ("prior", "the cross-writer reference descriptors"),
           ("verbose", "whether to print a one-line summary, default true")],
   output="the finished style profile as a dictionary, or nothing if the "
          "writer has too little usable data.",
   theory=[
     ("h", "What it is for"),
     ("p", "The second half of profile building, and the cheap one. It "
           "takes everything the cutter produced, discards what is not a "
           "letter, keeps the most typical examples of what remains, builds "
           "fallback letterforms, and measures the hand. Its output is the "
           "profile that synthesise.py consumes."),
     ("h", "Stage one: discard the badly aligned lines"),
     ("p", "Lines in the worst quarter by alignment confidence are dropped "
           "before anything else, for the reasons set out under "
           "ALIGN_CONF_PCT. This is first because it is the only filter "
           "that acts on whole lines: a line whose alignment failed has "
           "every boundary wrong, so its glyphs are unreliable as a set, "
           "and judging them individually afterwards would be more work for "
           "a worse result."),
     ("h", "Stage two: the gates, cheapest first"),
     ("p", "Every surviving glyph passes through a series of tests, each of "
           "which can only reject. The order is by cost, so that the "
           "expensive shape comparisons run on the fewest glyphs."),
     ("items", [
       "**Extent.** Width within bounds, which depends on the character; "
       "height over 0.25 x-heights; no more than six strokes. Three "
       "comparisons on numbers already stored.",
       "**Height class.** Does the ink reach where this letter should "
       "reach, from _ClassOk.",
       "**Cell overflow.** The ink must not be much wider than the cell it "
       "was cut from: lead plus width at most 1.45 times the advance plus "
       "0.25. Ink spilling well outside its own cell means the cut took "
       "part of a neighbour, which is a different and more sensitive test "
       "than the absolute width limit.",
       "**Pen path.** Total path above the per-character floor, and longest "
       "single stroke above 0.55 x-heights.",
       "**Size consistency.** Width between 0.6 and 1.5 times the 40th "
       "percentile of the surviving widths, and height between 0.55 and "
       "1.7 times the median height. Applied only if at least three "
       "variants survive it.",
       "**Cross-writer prior,** then **internal consensus.** The two "
       "descriptor-based filters, in that order, for the reason given "
       "under _ConsensusFilter.",
     ]),
     ("h", "Why the 40th percentile and not the median"),
     ("p", "The reference width for the consistency test is taken at the "
           "40th percentile rather than the middle. The reason is specific: "
           "a doubled letter, where the cut took two copies, is wide, and "
           "if such cuts are common for a character they push the median up "
           "and the test then accepts them while rejecting the correct "
           "narrow ones. Anchoring a little below the middle biases the "
           "reference toward the narrower, more likely correct, population."),
     ("h", "Stage three: the two fallback letterforms"),
     ("p", "Two single letterforms are built per character, and they are "
           "different things built for different purposes."),
     ("items", [
       "**idealMedoid:** the one real variant most typical of this writer's "
       "own set, found as the medoid of the descriptors. It is genuine "
       "writing, with all the irregularity that implies.",
       "**idealGlyphs:** the denoised prototype, the per-point median of "
       "several aligned variants. It is cleaner than any real variant but "
       "is an average, so it is not something the writer actually wrote.",
     ]),
     ("p", "Keeping both lets the choice be made according to what is "
           "needed: the medoid when authenticity matters, the prototype "
           "when legibility does. That trade-off between resembling the "
           "writer and being readable is the central tension of the whole "
           "project, and having both forms available is what lets "
           "synthesis navigate it."),
     ("h", "When the prototype is promoted to a variant"),
     ("eq", r"\mathrm{promote} \;\Longleftrightarrow\; "
            r"D_{\mathrm{proto}} < D_{\mathrm{raw}} - 0.02 "
            r"\;\;\text{ and }\;\; D_{\mathrm{proto}} < 0.32",
      [(r"D_{\mathrm{proto}}", "the prior distance of the denoised "
                               "prototype"),
       (r"D_{\mathrm{raw}}", "the median prior distance of this "
                             "character's kept variants"),
       ("0.02", "the margin by which the prototype must beat them, so a "
                "tie does not count"),
       ("0.32", "an absolute ceiling, so a prototype that is merely better "
                "than a bad set is still not accepted"),
       (r"\Longleftrightarrow", "if and only if: the promotion happens "
                                "exactly when both conditions hold")],
      None),
     ("p", "The prototype is placed at the front of the variant list, so "
           "that it is drawn in preference, only when it is measurably "
           "closer to the cross-writer idea of the letter than the writer's "
           "own variants are. Both conditions matter. The margin prevents "
           "promotion on a difference within noise. The absolute ceiling "
           "prevents the comparison being won by default when every variant "
           "is poor, which is exactly the situation where an averaged shape "
           "is least trustworthy, since averaging several bad variants "
           "gives a clean rendering of a bad shape."),
     ("h", "Stage four: measuring the hand"),
     ("items", [
       "**slantDeg,** the median per-line slant: how far the writing leans.",
       "**xHeightPx,** the median x-height in pixels, which records the "
       "scale the profile was built at.",
       "**strokeWidthXh,** ink area divided by skeleton length, which is "
       "the average stroke thickness, expressed in x-heights.",
       "**ascender and descender,** how far above and below the band this "
       "writer's letters actually reach, measured only from glyphs that "
       "clearly do reach, so that a mis-cut flat l cannot pull the "
       "ascender figure down.",
       "**wordSpaceXh,** the gap left between words.",
       "**connectedness,** the fraction of adjacent in-word letter pairs "
       "where ink genuinely crossed the boundary.",
       "**letterAdvance,** the median advance per character, which "
       "preserves this writer's own letter spacing per letter rather than "
       "one global figure.",
     ]),
     ("h", "Why connectedness is measured this way"),
     ("eq", r"\mathrm{connectedness} \;=\; "
            r"\frac{\big|\{\text{adjacent in-word pairs with ink crossing "
            r"the cut}\}\big|}"
            r"{\big|\{\text{adjacent in-word pairs}\}\big|}",
      [(r"\mathrm{connectedness}", "a number from 0 for fully separated "
                                   "printing to 1 for fully joined cursive"),
       (r"\text{adjacent in-word pairs}", "pairs of neighbouring characters "
                                          "where both were cut successfully "
                                          "and neither is a space")],
      None),
     ("p", "The numerator counts the pairs where a traced centre line "
           "actually passed through the boundary, which means the pen was "
           "never lifted there. That is the honest definition, as opposed "
           "to inferring joining from the gap between letters, which "
           "confuses tight spacing with genuine connection. Spaces are "
           "excluded because letters either side of a space are not "
           "expected to join and including them would make the measure "
           "depend on how many words the sample happened to contain."),
     ("p", "This single number turns out to be the most useful descriptor "
           "of a hand in the whole profile. Synthesis uses it to decide how "
           "to join letters, and it is the figure this project's results "
           "use to separate print-like hands from cursive ones, since "
           "reproduction accuracy falls as it rises."),
     ("h", "The two ways the function declines to produce a profile"),
     ("p", "Nothing is returned when there are no parsed lines at all, and "
           "again when no line survived the confidence cut with a "
           "measurable x-height. Returning nothing rather than a profile "
           "with empty fields is the right behaviour, because every "
           "consumer of a profile assumes its measurements are real, and a "
           "profile full of defaults would be used confidently and produce "
           "output in nobody's hand."),
     ("note", "The verbose summary printed at the end reports the line "
              "count, character count, variant count, slant, connectedness "
              "and reach. Those six numbers are enough to recognise a "
              "failed build at a glance: a writer with far fewer variants "
              "than usual has had most of their cuts rejected, and a "
              "connectedness of 0 or 1 exactly is almost always a sign that "
              "the crossing detection has gone wrong rather than a real "
              "hand."),
   ]),
}
