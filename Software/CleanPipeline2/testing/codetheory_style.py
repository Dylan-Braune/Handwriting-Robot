"""style.py, part 1: the settings, and ink to centre lines.

This chapter covers how a writer's handwriting is turned into a stored
profile.  Part 1 is the tuning constants and the geometry that converts
thinned ink into ordered pen paths.  Part 2, in codetheory_style2.py,
covers cutting a line into letters, the filters, and the profile itself.
"""

INTRO = [
 ("p", "This file answers one question: given some pages a person wrote, "
       "what is their handwriting? The answer it produces is a **style "
       "profile**, which holds a library of that writer's own letterforms "
       "together with measurements of how they lay letters out. "
       "synthesise.py then uses that profile to write new text in the same "
       "hand."),
 ("h", "The four steps"),
 ("items", [
   "**Trace.** The ink is thinned to a one-pixel centre line and then "
   "followed, turning a picture of a stroke into an ordered list of points: "
   "a path the pen could have taken.",
   "**Cut.** Each line is divided into individual characters, using the "
   "reader from recognise.py to say where each character sits, so no "
   "separate letter-finding method is needed.",
   "**Filter.** Most cuts are imperfect, so a series of gates rejects "
   "anything that is not a plausible letter. A gate only ever rejects; "
   "nothing is invented or repaired.",
   "**Measure.** Slant, x-height, how far ascenders reach, the gap left "
   "between words, and how joined-up the hand is.",
 ]),
 ("h", "The one convention that makes the rest possible"),
 ("p", "Every stored coordinate is measured in **x-heights**, not pixels. "
       "The x-height is the height of a letter body such as an o or an x, "
       "with no ascender or descender. A glyph is stored with its origin at "
       "the left edge of its cell and the baseline at y = 0, with y "
       "increasing upward, and both axes divided by the x-height."),
 ("p", "Three consequences follow, and they are why the convention is "
       "worth the trouble. The same stored glyph can be drawn at any size, "
       "because scaling is a single multiplication. Glyphs from pages "
       "photographed at different distances become directly comparable, so "
       "they can be averaged and compared. And the measurements of the hand "
       "describe the writing rather than the photograph, so a profile built "
       "from a close-up page and one built from a distant page agree."),
 ("note", "Stored y increases upward, which is the opposite of image "
          "coordinates, where row 0 is the top. The conversion happens once, "
          "inside ExtractLineGlyphs, and everything downstream works in the "
          "upward convention because that is how handwriting is naturally "
          "described: an ascender has a positive height and a descender a "
          "negative one."),
]


FUNCS = {

 "DATA_DIR": dict(
   output="the folder holding the public handwriting database pages.",
   theory=[
     ("h", "What it is for"),
     ("p", "Names where the dataset pages live. The path is built from the "
           "central paths module rather than written out, so the project "
           "has one definition of where data sits. The dataset itself is "
           "the IAM database of handwritten English text \\cite{marti2002}, "
           "a standard corpus of scanned pages with transcripts, which is "
           "what makes it usable here: the transcripts are what the forced "
           "alignment needs in order to place characters."),
   ]),

 "CACHE_DIR": dict(
   output="the folder holding cached per-line extractions.",
   theory=[
     ("h", "Why anything is cached"),
     ("p", "Cutting a page into letters is the slow part of building a "
           "profile, because every line has to be read by the network, "
           "thinned, traced and divided. The filtering and measuring that "
           "follow are fast. Caching the result of the slow part means that "
           "adjusting a filter threshold, which is the thing most often "
           "adjusted, costs seconds instead of re-running the whole "
           "extraction."),
     ("p", "This is the ordinary reason for a cache, but the division of "
           "labour here is worth noticing as a design choice: the "
           "expensive stage was deliberately made to produce an unfiltered "
           "result, so that the cheap stage can be re-run against it "
           "without invalidating it. A cache that stored filtered glyphs "
           "would have to be thrown away whenever a filter changed, which "
           "is exactly when its value is wanted."),
   ]),

 "NOGIT_DIR": dict(
   output="the folder for large generated files that are kept out of "
          "version control.",
   theory=[
     ("h", "Why it exists"),
     ("p", "Extraction caches and glyph libraries run to hundreds of "
           "megabytes. Version control stores a full copy of every revision "
           "of every file it tracks, so a large file that changes often "
           "makes the repository grow without bound, and nothing is gained, "
           "because the files can be regenerated from the code and the "
           "data. Keeping them under one named directory means a single "
           "ignore rule covers all of them, rather than a rule per file "
           "pattern that someone must remember to extend."),
   ]),

 "TEXT_WEIGHTS": dict(
   output="the path of the reader checkpoint used to place character "
          "boundaries.",
   theory=[
     ("h", "What it is for"),
     ("p", "Glyph extraction needs a reader, because the character "
           "positions come from forced alignment of that reader's output. "
           "This constant names which checkpoint to use for that purpose."),
     ("h", "Why the choice of checkpoint matters here"),
     ("p", "The quality of the alignment sets the quality of every glyph "
           "cut from it. A reader that is unsure where a character sits "
           "produces boundaries in the wrong place, and a boundary in the "
           "wrong place cuts a letter in half or sweeps part of its "
           "neighbour into it. Those are precisely the two defects the "
           "filters later spend most of their effort rejecting. So a "
           "stronger reader does not merely improve recognition; it reduces "
           "the amount of work the filters have to do and raises the "
           "proportion of cuts that survive them."),
     ("note", "The filename this constant points at is not present in the "
              "weights directory of this project as it stands. Code that "
              "depends on it therefore has to pass an explicit checkpoint "
              "path, which the callers do. It is recorded here because a "
              "constant naming a file that does not exist will mislead "
              "anyone reading the module to believe it is the checkpoint "
              "actually in use."),
   ]),

 "PERSONAL_WEIGHTS": dict(
   output="the path of the reader checkpoint adapted to the photographed "
          "writers.",
   theory=[
     ("h", "What makes it different"),
     ("p", "A reader trained on the public database has seen thousands of "
           "writers but none of the five photographed for this project, and "
           "it has seen scans rather than phone photographs. A checkpoint "
           "further trained on this project's own pages is better matched "
           "to them on both counts. For alignment, where the text is "
           "already known and only its position is in question, that "
           "closer match is directly useful."),
     ("h", "The trade-off being made"),
     ("p", "Adapting a network to a small set of writers risks making it "
           "worse on everyone else, a phenomenon called **catastrophic "
           "forgetting**: training on new data overwrites what was learned "
           "from the old. That is acceptable here precisely because this "
           "checkpoint has one narrow job. It is used to align pages from "
           "the writers it was adapted to, and the database writers are "
           "aligned with a checkpoint that has seen them."),
   ]),

 "RAW_DIR": dict(
   output="the folder holding the uncut, unfiltered extraction per writer.",
   theory=[
     ("h", "What it is for"),
     ("p", "Where the output of ExtractAuthorRaw is kept, one entry per "
           "writer. The word raw is the important part: what is stored is "
           "every glyph the cutter produced, including the fragments and "
           "the over-wide cuts, before any gate has rejected anything. That "
           "is what allows a filter to be changed and the effect measured "
           "without re-extracting, and it is also what allowed the "
           "rejection rates quoted in this project's results to be computed "
           "at all, since a filtered cache could not report what it had "
           "discarded."),
   ]),

 "MAX_VARIANTS_PER_CHAR": dict(
   output="12, the number of different shapes of one letter kept per "
          "writer.",
   theory=[
     ("h", "Why more than one shape is kept"),
     ("p", "A person does not write a letter identically twice. Storing one "
           "shape per letter would make the output visibly mechanical, "
           "because every a on the page would be the same a, which no "
           "handwriting ever is. Keeping several genuine variants lets "
           "synthesis choose among them and gives the output the "
           "irregularity real writing has."),
     ("h", "Why not keep all of them"),
     ("p", "A writer's pages yield far more than twelve examples of a "
           "common letter, and the extras are not free. They enlarge the "
           "profile, they slow the search that synthesis runs over "
           "candidate glyphs, and, most importantly, the ones beyond the "
           "first dozen are the least typical: the variants are sorted by "
           "how close they sit to the writer's own usual size and shape "
           "before this limit is applied, so the ones discarded are the "
           "outliers. Keeping a bad variant is worse than keeping no extra "
           "variant, because synthesis may select it."),
     ("note", "This cap interacts with a separate limit applied at "
              "synthesis time, which considers only a few of the stored "
              "variants per character when choosing. The consequence is "
              "that a substantial share of stored variants can never "
              "actually be drawn. That is a property of the two limits "
              "together rather than of either alone, and it is worth "
              "knowing before the cap here is raised in the hope of more "
              "variety."),
   ]),

 "ALIGN_CONF_PCT": dict(
   output="25, the percentile of alignment confidence below which a line is "
          "not used.",
   theory=[
     ("h", "What it is for"),
     ("p", "Forced alignment returns a confidence, the mean log probability "
           "per column along the chosen path, as described in the entry for "
           "CtcForcedAlign. A low value means the reader could not "
           "comfortably fit the known text to what it saw, so the character "
           "positions it produced are not to be trusted. Lines in the "
           "bottom quarter by this measure are dropped before any glyph "
           "from them is used."),
     ("h", "Why a percentile rather than a fixed threshold"),
     ("p", "This is the important design point. A fixed confidence cut "
           "would mean something different for every writer, because "
           "confidence depends on how closely a hand resembles what the "
           "reader was trained on. A clearly written upright hand produces "
           "high confidences throughout, and a cut placed to suit it would "
           "discard almost every line of a heavily cursive writer, leaving "
           "them with no profile at all. Conversely a cut placed low enough "
           "to admit the cursive writer would admit everything from the "
           "clear one, including its genuinely bad lines."),
     ("p", "A percentile is **relative**: it asks which lines are the worst "
           "quarter *for this writer*, whatever the absolute numbers. Every "
           "writer therefore contributes their best three quarters, which "
           "is the intended behaviour. The cost is that a writer whose "
           "every line aligned badly still contributes their best three "
           "quarters, since the measure cannot express that all of them are "
           "poor in absolute terms."),
     ("h", "Why a quarter"),
     ("p", "The figure trades the number of glyphs against their quality. "
           "Discarding nothing admits the lines where alignment failed "
           "outright; discarding half would roughly halve the glyph count, "
           "and since a letter needs several examples before a prototype "
           "can be formed from it, that is a real loss for the rarer "
           "letters. A quarter removes the clear failures while leaving "
           "enough examples of an uncommon letter such as a q or a z to be "
           "usable."),
     ("note", "The percentile is only applied when at least eight lines are "
              "available. Below that the ranking is meaningless, since a "
              "quarter of five lines is one line chosen on very little "
              "evidence, so the cut is disabled and every line is kept."),
   ]),

 "FRAGMENT_PEN_FRAC": dict(
   output="0.5, the fraction of a letter's usual pen path that a cut must "
          "have to count as a whole letter.",
   theory=[
     ("h", "The defect it targets"),
     ("p", "The commonest bad cut is a **fragment**: a boundary placed "
           "inside a letter, so that what is stored is half an a or the "
           "first stroke of an n. A fragment is dangerous in a way that "
           "other bad cuts are not, because it looks like a legitimate "
           "small glyph. Its bounding box is plausible, its stroke count is "
           "plausible, and it will be drawn without complaint, producing a "
           "letter that is simply missing half of itself."),
     ("h", "Why pen path length is the right measure"),
     ("p", "The bounding box cannot detect a fragment. Half of an o spans "
           "nearly the full width and the full height of the whole letter "
           "while containing half the ink, so any test on extent accepts "
           "it. **Pen path length**, the total distance the pen travels, "
           "measures how much writing the shape actually represents. Half "
           "an o has half the path of a whole one, and no amount of clever "
           "positioning changes that."),
     ("h", "Why the threshold is relative to the letter itself"),
     ("p", "The comparison is against the median pen length of that "
           "writer's own examples of that same character, not a universal "
           "figure. This matters because letters differ enormously in how "
           "much pen path they contain: an i is a short stroke and a dot, "
           "while an m is three arches, perhaps four times the path. A "
           "single absolute threshold would either accept half an m or "
           "reject every whole i. Taking each character's own median gives "
           "each letter its own floor, derived from how that writer "
           "actually writes it."),
     ("eq", r"\mathrm{penFloor}(c) \;=\; \max\Big(\, "
            r"\mathrm{MIN\_PEN\_ABS},\;\; 0.5 \cdot "
            r"\mathrm{median}\big\{\,L(g) : g \in \mathrm{instances}(c)\,"
            r"\big\} \,\Big)",
      [(r"\mathrm{penFloor}(c)", "the least pen path a glyph of character c "
                                 "may have"),
       ("L(g)", "the total pen path length of glyph g, in x-heights"),
       (r"\mathrm{instances}(c)", "every cut of character c this writer "
                                  "produced, before filtering"),
       ("0.5", "this constant: the fraction of the typical path required"),
       (r"\mathrm{MIN\_PEN\_ABS}", "an absolute floor, described in its own "
                                   "entry")],
      None),
     ("h", "Why one half"),
     ("p", "The threshold has to sit between the shortest legitimate "
           "variant of a letter and the longest fragment. Legitimate "
           "variation in how much pen path one writer puts into one letter "
           "is modest, perhaps 20 or 30 percent, because they are writing "
           "the same shape each time. A fragment, by definition, is a part "
           "of a letter, so it has well under all of it. One half sits in "
           "the gap: generous enough that an unusually compact but complete "
           "letter passes, strict enough that a genuine half rarely does."),
     ("note", "The median, not the mean, for the reason it is used "
              "throughout this project: the pool being summarised still "
              "contains the fragments at the point the floor is computed, "
              "and a mean would be dragged down by them, lowering the very "
              "threshold meant to exclude them. The median is unmoved as "
              "long as fragments are a minority."),
   ]),

 "MIN_PEN_ABS": dict(
   output="0.30, the least pen path any glyph may have, in x-heights.",
   theory=[
     ("h", "Why a relative threshold needs an absolute floor beneath it"),
     ("p", "The relative test above compares a glyph with the median of its "
           "own character's instances, which fails in one specific "
           "situation: when most of the instances are themselves bad. If "
           "two thirds of the cuts of some rare letter are fragments, the "
           "median is a fragment's length, half of that is a quarter of a "
           "fragment, and almost anything passes. The relative test has no "
           "way to detect this, because it has no external reference for "
           "what the letter should be."),
     ("p", "An absolute floor supplies that reference. Whatever the median "
           "says, a glyph with less than 0.3 x-heights of pen path is "
           "rejected. Three tenths of an x-height is shorter than the "
           "shortest meaningful mark in handwriting: the stem of an i "
           "alone is a full x-height, and a comma or an apostrophe is a "
           "short stroke but still around a third. So the floor cannot "
           "reject a real letter while it does catch the degenerate case."),
     ("note", "The pairing of a relative test with an absolute floor "
              "appears repeatedly in this file, and the reason is always "
              "the same. A relative test adapts to the writer, which is "
              "what is wanted, but it assumes the population it measures "
              "is mostly sound. The floor is what holds when that "
              "assumption fails."),
   ]),

 "MIN_LONGEST_STROKE": dict(
   output="0.55, the least length the longest single stroke of a glyph may "
          "have, in x-heights.",
   theory=[
     ("h", "The defect it targets"),
     ("p", "Total pen path can be reached by accumulation: a glyph made of "
           "eight short disconnected scraps has the same total path as one "
           "made of a single proper stroke. The total test therefore "
           "accepts a scatter of debris, which is what a cut through a "
           "crowded cursive region produces. This test adds the "
           "requirement that at least one of the strokes be substantial on "
           "its own."),
     ("h", "Why the longest stroke is the right thing to measure"),
     ("p", "Every letter contains one dominant stroke. An l is a single "
           "long stroke, the body of an a is one continuous curve, an n has "
           "two arches of which either is substantial. The dominant stroke "
           "is the letter's skeleton in the ordinary sense of the word, and "
           "if it is absent then whatever was cut is not a letter however "
           "much ink it contains. Asking for the longest stroke rather than "
           "the average also means a legitimate letter with a small extra "
           "mark, such as the dot of an i or the bar of a t, is not "
           "penalised for having one short stroke among its strokes."),
     ("h", "Why 0.55"),
     ("p", "A little over half an x-height. The shortest dominant stroke in "
           "ordinary writing belongs to a letter like a c or an r, which "
           "spans roughly the x-height as a curve and so has a path at "
           "least comparable to it. Setting the bar at just over half gives "
           "that case clear room while excluding the scraps, which are "
           "typically a fraction of an x-height each."),
   ]),

 "DOT_MIN_AREA": dict(
   output="0.004, the least area a component may have to be treated as a "
          "dot rather than dust, in squared x-heights.",
   theory=[
     ("h", "The problem it solves"),
     ("p", "A dot, such as the one over an i, is nearly round and only a "
           "few pixels across. Thinning reduces it to a single pixel, and "
           "a single pixel is not a path, so the polyline tracer produces "
           "nothing for it and the dot is silently lost. An i stored "
           "without its dot is a wrong letterform that will be drawn as a "
           "bare stem."),
     ("p", "The recovery is to find connected components of ink that the "
           "tracer produced no path for, and add each as a short horizontal "
           "dash across its own width. This threshold separates the dots "
           "worth recovering from the specks that are not."),
     ("h", "Why the units are squared x-heights"),
     ("eq", r"\mathrm{minDotPx} \;=\; \max\!\big(2,\; "
            r"\mathrm{round}(0.004 \cdot \mathrm{xh}^{2})\big)",
      [(r"\mathrm{minDotPx}", "the least number of pixels a component must "
                              "contain to be recovered"),
       (r"\mathrm{xh}", "the measured x-height of this line, in pixels"),
       ("0.004", "this constant, an area expressed as a fraction of the "
                 "square of the x-height"),
       (r"\max(2,\cdot)", "a floor of two pixels, so a single isolated "
                          "pixel is never recovered")],
      None),
     ("p", "Area scales with the square of a length, so an area threshold "
           "that is to mean the same thing at every resolution must be "
           "proportional to the square of the x-height. A dot of 0.004 "
           "squared x-heights is about one sixteenth of an x-height across, "
           "which is the right order for the dot of an i. Expressing it in "
           "pixels instead would make it correct at one resolution only: at "
           "twice the resolution, a dot has four times the pixels."),
   ]),

 "MAX_GLYPH_W": dict(
   output="2.1, the greatest width a glyph may have, in x-heights.",
   theory=[
     ("h", "The defect it targets"),
     ("p", "The mirror image of the fragment. When a boundary is placed too "
           "far to one side, the cut takes its own letter plus part or all "
           "of a neighbour. The stored glyph is then not one letter but "
           "one and a bit, and drawing it produces a doubled letter in the "
           "output. This is the defect that makes reproduced text contain "
           "letters with pieces of other letters attached."),
     ("h", "Why width detects it"),
     ("p", "A letter has a bounded width relative to its own height. In "
           "ordinary handwriting a lowercase letter is roughly as wide as "
           "the x-height, perhaps a little more for a cursive hand that "
           "stretches its letters. Two letters together are about twice "
           "that. A limit of 2.1 x-heights therefore sits above any single "
           "letter and below most pairs, which is exactly where a cut-off "
           "between the two belongs."),
     ("note", "It is a blunt instrument, and deliberately so. It cannot "
              "catch a cut that took only a small part of a neighbour, "
              "because that stays within the width limit. That case is left "
              "to the shape-based filters further on, which compare the "
              "glyph with what the letter is supposed to look like. The "
              "ordering is intentional: the cheap test runs first and "
              "removes the gross failures, so the expensive test has fewer "
              "glyphs to consider."),
   ]),

 "MAX_GLYPH_W_WIDE": dict(
   output="2.9, the same limit for the letters that genuinely are wide.",
   theory=[
     ("h", "Why a second limit is needed"),
     ("p", "A limit chosen to suit most letters is wrong for m and w. An m "
           "really is about twice as wide as an a, so the general limit "
           "would reject every legitimate m while still permitting an a "
           "that had swallowed its neighbour. The two cases cannot be "
           "separated by width alone, so the limit is made to depend on "
           "which letter is being judged."),
     ("h", "Why not simply raise the general limit"),
     ("p", "Because that would weaken the test everywhere to accommodate "
           "two letters. The point of a per-character limit is that each "
           "character is judged against what that character should be, "
           "which is more information than a single threshold can use. The "
           "same reasoning produced the per-character pen floor described "
           "above, and it is the general pattern of the filters in this "
           "file: compare a glyph with its own letter's norm wherever the "
           "norm is available."),
   ]),

 "_WIDE_CHARS": dict(
   output="the set of characters that receive the wider width allowance.",
   theory=[
     ("h", "Which letters, and why only these"),
     ("p", "Lowercase m and w and their capitals. These are the letters "
           "built from repeated arches or strokes, so they occupy two "
           "letter widths by construction rather than by accident. Every "
           "other letter in the charset fits the general limit, including "
           "the ones that feel wide: a capital W is in the set, while a "
           "capital M is too, and letters such as O or D are tall rather "
           "than wide and so stay within the ordinary bound."),
     ("note", "A set, not a list, because the only operation performed on "
              "it is a membership test, and a set answers that in constant "
              "time by hashing rather than by scanning. With four members "
              "the difference is immaterial; the habit is worth keeping "
              "because the same test inside a loop over every glyph of "
              "every line is not immaterial."),
   ]),

 "_OFFS": dict(
   output="the eight coordinate offsets from a pixel to its neighbours.",
   theory=[
     ("h", "What 8-connectivity means"),
     ("p", "Two notions of neighbour exist for a pixel grid. "
           "**4-connectivity** counts only the four pixels sharing an edge, "
           "up, down, left and right. **8-connectivity** also counts the "
           "four sharing only a corner, so a pixel has eight neighbours. "
           "This list is the eight offsets of the second kind."),
     ("h", "Why 8 and not 4 for following a stroke"),
     ("p", "A diagonal line drawn on a pixel grid is a staircase of pixels "
           "that touch only at their corners. Under 4-connectivity such a "
           "line is not connected at all: every pixel is isolated from the "
           "next. A tracer using 4-connectivity would therefore break every "
           "diagonal stroke into a sequence of single pixels, which for "
           "handwriting, where most strokes are slanted, would break "
           "practically everything. 8-connectivity is the only workable "
           "choice for following a thinned stroke."),
     ("note", "The choice is not free of consequences. Under 8-connectivity "
              "two strokes that merely pass corner to corner count as "
              "joined, so the tracer can walk from one stroke into another "
              "that it only touches. That is part of why the tracer is "
              "followed by the spur pruning and junction rejoining "
              "described below, rather than being trusted on its own."),
   ]),

 "TracePolylines": dict(
   inputs=[("skel", "a boolean array holding the thinned ink, one pixel "
                    "wide")],
   output="a list of polylines, each an ordered list of (x, y) points "
          "following one stroke.",
   theory=[
     ("h", "What it is for"),
     ("p", "Thinning turns a stroke into a one-pixel-wide curve, but the "
           "result is still a *set* of pixels: a picture, with no notion of "
           "order. A pen path is an ordered sequence. This function "
           "supplies the order, by walking along connected pixels, so that "
           "a stroke becomes something a plotter could draw and something "
           "that can be compared, resampled and averaged."),
     ("h", "Background: degree, and the three kinds of pixel"),
     ("p", "The **degree** of a pixel is how many of its eight neighbours "
           "are also ink. Degree classifies every pixel of a skeleton into "
           "one of three kinds, and the whole algorithm is organised around "
           "the classification."),
     ("items", [
       "**Degree 1, an endpoint.** The loose end of a stroke. A stroke has "
       "two of these, one at each end.",
       "**Degree 2, an interior point.** The middle of a stroke, with one "
       "neighbour behind and one ahead. Walking is possible here because "
       "the continuation is unambiguous.",
       "**Degree 3 or more, a junction.** Where strokes meet, as in the "
       "crossing of a t or the point where a cursive join meets a letter. "
       "Walking must stop, because there is more than one way to continue "
       "and nothing locally distinguishes them.",
     ]),
     ("h", "The walk"),
     ("p", "Starting from a pixel that is not of degree 2, that is an "
           "endpoint or a junction, the walk steps to a neighbour and then "
           "continues as long as the current pixel has degree exactly 2, "
           "each time moving to the neighbour that is not the one just "
           "left. It halts on reaching anything that is not of degree 2, "
           "which is either the far end of the stroke or a junction. So "
           "each walk traces one unambiguous run of stroke between two "
           "decision points."),
     ("p", "Starting from the non-degree-2 pixels rather than anywhere is "
           "what makes the decomposition well defined. Beginning in the "
           "middle of a stroke would produce two half-strokes that would "
           "then have to be matched up again."),
     ("h", "Why edges are marked as used, rather than pixels"),
     ("p", "The walk records that it has traversed the connection between "
           "two pixels, in both directions, rather than recording that it "
           "has visited a pixel. The distinction matters at a junction: "
           "the junction pixel belongs to every stroke meeting there, so "
           "marking it as visited would prevent all but the first stroke "
           "from using it, and the remaining strokes would be traced as "
           "starting one pixel away from where they really start. Marking "
           "connections instead lets each stroke pass through the junction "
           "while still guaranteeing no connection is traced twice."),
     ("h", "The pure-loop case"),
     ("p", "A closed loop, such as a well-formed o, consists entirely of "
           "degree-2 pixels. It therefore has no endpoint and no junction, "
           "so the seeding rule finds nothing to start from and the loop "
           "would be missed. A second pass handles it: any pixel not yet "
           "covered by a traced polyline is used as an arbitrary starting "
           "point, which for a loop gives the whole loop traced from a "
           "place chosen arbitrarily. That is acceptable, because a loop "
           "has no natural starting point to find."),
     ("h", "Suppressing redundant diagonal steps"),
     ("p", "When a diagonal neighbour is available and so is a path to it "
           "through an orthogonal neighbour, the diagonal step is not "
           "offered. Without this, a stroke two pixels thick in places, "
           "which thinning can leave, allows the walk to cut a corner "
           "diagonally and then find itself able to step back, producing a "
           "short spurious loop. Preferring the orthogonal route keeps the "
           "traced path on the pixels the skeleton actually laid out."),
     ("h", "The two tidying passes"),
     ("p", "The raw traces are then cleaned by _PruneSpurs, which removes "
           "the short whiskers thinning leaves at thick junctions, and "
           "_JoinAtJunctions, which rejoins traces that the junction split "
           "but which were one pen stroke. Both have their own entries. The "
           "important point here is that the tracer is not expected to be "
           "correct by itself: it produces a decomposition that is correct "
           "as graph theory, and the two passes then adjust it toward what "
           "a pen actually did, which is a different question."),
     ("note", "Points come out as (x, y) floating-point pairs, with x the "
              "column and y the row, swapped from the (row, column) order "
              "the pixels were found in. The swap is to the convention the "
              "rest of the geometry uses, where the first coordinate is "
              "horizontal, and it happens once, here, rather than being "
              "remembered at every later use."),
   ]),

 "_Dir": dict(
   inputs=[("pts", "a polyline"),
           ("fromEnd", "true for the direction at the last point, false for "
                       "the first"),
           ("n", "how many points back to look, default 4")],
   output="a unit vector giving the direction the path is heading at that "
          "end.",
   theory=[
     ("h", "What it is for"),
     ("p", "Deciding whether two traced paths were one pen stroke requires "
           "knowing which way each is heading where they meet. This "
           "measures that direction at either end of a path."),
     ("h", "Why it looks several points back"),
     ("p", "The direction could be taken from the last two points, but "
           "those are adjacent pixels, so the step between them is one of "
           "only eight possible directions, each 45 degrees apart. That "
           "resolution is far too coarse to decide whether two strokes "
           "continue smoothly. Measuring from a point four steps back gives "
           "a chord over several pixels, whose angle is continuous and "
           "which averages out the staircase the pixel grid imposes."),
     ("eq", r"\mathbf{d} \;=\; \frac{\mathbf{p}_{\mathrm{end}} - "
            r"\mathbf{p}_{\mathrm{end}-n}}{\big\|\,"
            r"\mathbf{p}_{\mathrm{end}} - \mathbf{p}_{\mathrm{end}-n}"
            r"\,\big\|}",
      [(r"\mathbf{d}", "the returned direction, a vector of length 1"),
       (r"\mathbf{p}_{\mathrm{end}}", "the point at the end being measured"),
       (r"\mathbf{p}_{\mathrm{end}-n}", "the point n steps in from that "
                                        "end"),
       ("n", "how far back to look, 4 by default"),
       (r"\|\cdot\|", "the length of the vector, by Pythagoras")],
      None),
     ("h", "Why it is normalised to unit length"),
     ("p", "Dividing by the length leaves only the direction, discarding "
           "how far apart the two points were. That is what makes two "
           "directions comparable by a dot product, as the next function "
           "does: for unit vectors the dot product is the cosine of the "
           "angle between them, with no dependence on the lengths. Without "
           "normalisation the same comparison would confound the angle with "
           "how long the paths happened to be."),
     ("note", "A degenerate path whose endpoints coincide has no direction, "
              "and a default pointing along the positive x axis is "
              "returned. The value is arbitrary; what matters is that it is "
              "a unit vector, so the caller's angle arithmetic stays well "
              "defined instead of producing a division by zero."),
   ]),

 "_PruneSpurs": dict(
   inputs=[("chains", "the traced polylines"),
           ("minLen", "the shortest a dangling path may be before it is "
                      "removed, default 2.5 pixels")],
   output="the polylines with the short dangling ones removed.",
   theory=[
     ("h", "The artefact it removes"),
     ("p", "Thinning a shape where strokes cross leaves a characteristic "
           "defect. Where the ink is thick, as at the junction of a t or "
           "where a cursive join meets a letter body, the thinning does not "
           "produce a clean crossing but a small cluster from which short "
           "stubs project, going nowhere. These are **spurs**, and they are "
           "an artefact of the thinning rather than anything the pen did."),
     ("p", "They matter because they are traced as polylines like any "
           "other, so a letter that should have two strokes acquires five, "
           "three of which are two-pixel whiskers. That inflates the stroke "
           "count, which several filters test, it disturbs the stroke "
           "ordering, and it confuses the averaging that forms a prototype, "
           "which matches strokes between variants by position."),
     ("h", "How a spur is recognised"),
     ("p", "Two conditions together. It is short, under 2.5 pixels of path "
           "length. And it is **dangling**, meaning at least one of its two "
           "endpoints is not shared with any other path, so it leads "
           "nowhere. Requiring both is what makes the test safe: a short "
           "path joined at both ends is a genuine short connection between "
           "two strokes and must be kept, while a long dangling path is a "
           "real stroke with a free end, such as the stem of a t."),
     ("p", "The sharing is determined by counting how many paths have an "
           "endpoint at each coordinate. A count of one means that endpoint "
           "belongs to this path alone."),
     ("h", "Why the threshold is in pixels and not x-heights"),
     ("p", "Almost every threshold in this file is in x-heights, for the "
           "scale-independence reasons given in the introduction. This one "
           "is deliberately not, because the artefact it targets is not a "
           "property of the writing at all: it is a property of the "
           "thinning algorithm, whose spurs are a few pixels long whatever "
           "the resolution, because thinning works pixel by pixel. A "
           "threshold in x-heights would grow with resolution and start "
           "removing real strokes from high-resolution crops."),
     ("note", "If the pruning would remove everything, the original paths "
              "are returned instead. A glyph consisting entirely of short "
              "dangling paths is most likely debris, but returning nothing "
              "would hide it from the filters that are designed to judge "
              "it; returning it unchanged lets them do so."),
   ]),

 "_JoinAtJunctions": dict(
   inputs=[("chains", "the traced polylines"),
           ("maxAngleDeg", "how much the direction may change at a join and "
                           "still count as one stroke, default 75 degrees")],
   output="the polylines with those that continue through a junction merged "
          "into one.",
   theory=[
     ("h", "The problem it solves"),
     ("p", "The tracer stops at every junction, because locally there is no "
           "way to know which way to continue. But a pen usually does "
           "continue. Where a cursive join meets the body of a letter, or "
           "where one stroke crosses another, the tracer produces several "
           "short paths where the pen drew one long one. Leaving them "
           "separate misrepresents the writing: the glyph appears to have "
           "more strokes than it does, and a plotter drawing it would lift "
           "the pen where the writer did not."),
     ("h", "The criterion: continuation of direction"),
     ("p", "What distinguishes a continuation from a genuine corner is "
           "direction. If one path arrives at a junction heading one way "
           "and another leaves heading the same way, they were one stroke. "
           "If the second leaves at a sharp angle, the junction is a real "
           "corner or a crossing of two different strokes."),
     ("eq", r"\cos\theta \;=\; -\,\mathbf{d}_i \cdot \mathbf{d}_j, "
            r"\qquad \theta \;=\; \arccos\big(\mathrm{clip}"
            r"(\cos\theta,\,-1,\,1)\big)",
      [(r"\theta", "the turn angle: 0 for a path continuing perfectly "
                   "straight through the junction"),
       (r"\mathbf{d}_i,\, \mathbf{d}_j", "the unit directions at the two "
                                         "ends being considered, each from "
                                         "_Dir"),
       (r"\cdot", "the dot product, which for unit vectors equals the "
                  "cosine of the angle between them"),
       (r"-", "the negation, explained below"),
       (r"\arccos", "the inverse cosine, converting a cosine back to an "
                    "angle"),
       (r"\mathrm{clip}(\cdot,-1,1)", "the cosine forced into its valid "
                                      "range before the inverse cosine is "
                                      "taken")],
      None),
     ("h", "Why the dot product is negated"),
     ("p", "This is the step most easily got wrong. Both directions are "
           "measured *outward* from their own path's end. Two paths that "
           "continue through a junction therefore point in opposite "
           "directions: one arrives pointing into the junction, the other "
           "leaves pointing away, and as outward measurements those are "
           "roughly antiparallel. So perfect continuation gives a dot "
           "product of −1, not +1, and negating it turns continuation into "
           "a cosine of +1, which is what the angle formula expects."),
     ("h", "Why the cosine is clipped"),
     ("p", "The inverse cosine is defined only on the range −1 to 1. Two "
           "unit vectors cannot mathematically produce a dot product "
           "outside it, but floating-point rounding can: a vector whose "
           "length is stored as 1.0000000001 yields a dot product a "
           "fraction beyond 1, and the inverse cosine of that is not a "
           "number. The result would then propagate silently through every "
           "comparison, since comparisons involving it are all false. "
           "Clipping first costs nothing and removes the possibility."),
     ("h", "Why 75 degrees is generous"),
     ("p", "A threshold of 75 degrees permits a substantial change of "
           "direction, far more than a straight continuation. That is "
           "deliberate, because handwriting genuinely turns: the pen rounds "
           "a corner at the bottom of a u or the top of an n without "
           "lifting, so insisting on near-straightness would refuse to "
           "rejoin strokes that really were continuous. The errors are "
           "asymmetric, which is what justifies the generosity. Failing to "
           "join two halves of one stroke leaves a glyph that a filter will "
           "then reject as a fragment, losing a usable letter. Joining two "
           "strokes that were separate merely merges two strokes of the "
           "same letter, which affects the stroke count and little else."),
     ("h", "Why it repeats until nothing changes"),
     ("p", "One merge can create a new opportunity. Three paths meeting in "
           "a line are joined by first merging two, after which the merged "
           "path has a new free end that may continue into the third. So "
           "the search repeats until a pass finds nothing to merge. Each "
           "pass also takes the single best candidate, the pair with the "
           "smallest turn angle, rather than merging everything that "
           "qualifies. That matters at a junction where three or more paths "
           "meet and more than one pairing is permissible: taking the best "
           "first means the most convincing continuation is committed to "
           "before the others are considered, which is the greedy strategy "
           "used for the line centres in segment.py, and it is correct for "
           "the same reason."),
     ("note", "The search compares every pair of ends on every pass, which "
              "is quadratic in the number of paths and runs inside a loop. "
              "That would matter for hundreds of paths; here it runs on the "
              "handful belonging to one letter, where the cost is "
              "irrelevant and the clarity of an exhaustive search is worth "
              "having."),
   ]),

 "SimplifyPolyline": dict(
   inputs=[("poly", "a polyline"),
           ("eps", "how far a point may sit from the simplified path before "
                   "it must be kept, default 0.6 pixels")],
   output="the polyline with the points that add nothing removed.",
   theory=[
     ("h", "What it is for"),
     ("p", "A traced path has one point per pixel, so a straight stroke "
           "fifty pixels long is stored as fifty points when two would "
           "describe it exactly. Removing the redundant points shrinks the "
           "stored profile, speeds up everything that iterates over points, "
           "and removes the pixel staircase, leaving a path described by "
           "where it actually changes direction."),
     ("h", "The Douglas-Peucker method"),
     ("p", "The algorithm \\cite{douglaspeucker1973} is recursive and "
           "rests on one idea: a path can be replaced by the straight line "
           "between its endpoints if no point on it strays far from that "
           "line. So measure the furthest stray; if it is within tolerance, "
           "the whole path becomes two points, and if not, the path is "
           "split at the furthest point and each half is simplified the "
           "same way."),
     ("eq", r"d_k \;=\; \frac{\big|\,(x_b-x_a)(y_k-y_a) - "
            r"(y_b-y_a)(x_k-x_a)\,\big|}"
            r"{\sqrt{(x_b-x_a)^2 + (y_b-y_a)^2}}",
      [("d_k", "the perpendicular distance from point k to the straight "
               "line joining the two endpoints"),
       ("(x_a,y_a)", "the first point of the path"),
       ("(x_b,y_b)", "the last point of the path"),
       ("(x_k,y_k)", "the point whose distance is being measured"),
       (r"|\cdot|", "the absolute value, since a distance has no sign"),
       (r"\sqrt{\cdot}", "the length of the line segment, which turns the "
                         "numerator into a true distance")],
      "douglaspeucker1973"),
     ("p", "The numerator is the **cross product** of the vector along the "
           "line and the vector to the point. Its magnitude equals the area "
           "of the parallelogram the two vectors span, and that area also "
           "equals the base times the height. The base is the length of the "
           "line segment, so dividing by it leaves the height, which is the "
           "perpendicular distance. This is why the formula computes a "
           "distance without any trigonometry."),
     ("h", "Why the tolerance is 0.6 pixels"),
     ("p", "Just over half a pixel. The reasoning is that the staircase to "
           "be removed deviates from a true straight line by up to half a "
           "pixel, that being the most a rounded position can differ from "
           "the real one. A tolerance just above that removes the staircase "
           "and nothing else: any deviation larger than half a pixel is a "
           "real change of direction that the writer put there, and it is "
           "kept."),
     ("h", "The closed-path special case"),
     ("p", "A loop, such as the body of an o, defeats the basic method. Its "
           "endpoints coincide or nearly so, which makes the reference "
           "line a point rather than a line, and every distance measured "
           "from it is meaningless. The function detects this by checking "
           "whether the far end of the path returns close to the start, and "
           "if so splits the path first at the point furthest from the "
           "start, simplifying the two halves separately. Each half is then "
           "an ordinary open path with distinct endpoints, and the method "
           "applies."),
     ("note", "The two halves are rejoined by dropping the last point of "
              "the first before concatenating, since the split point "
              "belongs to both and would otherwise be stored twice. A "
              "duplicated point is a zero-length segment, which is harmless "
              "to draw but divides by zero in any calculation that "
              "normalises by segment length."),
   ]),
}


#the rest of the chapter, in order
for _part in ("codetheory_style2",):
    try:
        _m = __import__(_part)
    except ModuleNotFoundError:
        continue
    FUNCS.update(_m.FUNCS)
