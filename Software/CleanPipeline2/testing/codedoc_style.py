"""Write-up for style.py: turning somebody's pages into a reusable hand."""

MODULES = {
 "style.py": dict(intro=[
   "**Functional unit 5, requirement R5.** The question this file answers "
   "is: given photographs of pages somebody wrote, what exactly is their "
   "handwriting, stored in a form that can be used to write something they "
   "never wrote?",

   ("The answer has two halves. The first is a library of letterforms: for "
    "each character, several examples of how this person actually draws it, "
    "stored as pen paths rather than as pictures, so they can be redrawn at "
    "any size. The second is a set of global measurements — how far the "
    "writing leans, how tall the letter bodies are, how far above and below "
    "the line the ascenders and descenders reach, how wide the word gaps "
    "are, and how often consecutive letters are joined. The library "
    "provides the shapes; the measurements provide the layout."),

   ("The central difficulty is cutting. To collect examples of a letter you "
    "must first find that letter in a line of writing, and in joined-up "
    "handwriting there is frequently no gap where one letter ends and the "
    "next begins. The approach taken is to let the recogniser say where it "
    "thinks each character sits — it was trained on exactly this problem — "
    "and then snap the boundary to the most defensible nearby column, which "
    "means thin ink low down, since that is what a ligature crossing looks "
    "like. The method is sound but its limits are visible in the results, "
    "and the reasons are set out under `ExtractLineGlyphs`."),

   ("The second difficulty is that cutting will sometimes be wrong, and a "
    "wrong cut produces a shape that is not a letter: half a letter, a "
    "letter with a piece of its neighbour attached, or a fragment of a "
    "ligature. If these entered the library the synthesiser would draw them. "
    "So roughly half this file is filtering, arranged as five gates applied "
    "cheapest first: simple geometry, then outlier rejection against the "
    "writer's own variants, then comparison against what that letter looks "
    "like across all writers, then internal consensus, then a typicality "
    "ranking. A gate only ever rejects a shape; none of them ever "
    "substitutes another writer's letter, which would make the profile a "
    "fiction."),

   ("A note on what is stored. Glyph coordinates are in x-height units with "
    "the origin at the left edge of the character's cell on the baseline, y "
    "increasing upwards, and the writer's slant removed. Storing them "
    "upright and unit-scaled is what allows the synthesiser to re-apply the "
    "slant, choose a size and place letters next to each other without "
    "having to know anything about the page they came from."),
 ]),
}

FUNCS = {

 "TracePolylines": dict(
   inputs="`skel`, a thinned one-pixel-wide ink mask.",
   output="A list of polylines, each an ordered list of `(x, y)` points in "
          "pixel coordinates.",
   body=[
     ("sub", "From a picture of a stroke to the stroke itself"),
     "Thinning gives a one-pixel-wide image of the centreline, but it is "
     "still an image: a set of lit pixels with no order. A pen stroke is an "
     "ordered path. This converts one to the other by walking along "
     "connected pixels from each loose end, recording the order it visits "
     "them in.",
     "Walking from the ends rather than from an arbitrary pixel is what "
     "makes the resulting path a sensible traversal rather than one that "
     "starts in the middle and has to be stitched together.",
     ("sub", "Two tidying steps, and why both are needed"),
     "Thinning produces two characteristic artefacts, and the raw walk "
     "would faithfully preserve both. Where strokes cross or a stroke is "
     "locally thick, the thinning leaves a short stub going nowhere, which "
     "`_PruneSpurs` removes. And at a junction, one real pen stroke is "
     "split into several paths, which `_JoinAtJunctions` reassembles when "
     "the directions agree.",
     "Without the first, the glyph library fills with letters that have "
     "small whiskers on them. Without the second, a letter drawn in one "
     "continuous movement is stored as three disconnected pieces, and the "
     "synthesiser would lift the pen twice while drawing it.",
   ]),

 "_Dir": dict(
   inputs="`pts`, a path. `fromEnd`, which end to measure. `n`, how many "
          "points to average over.",
   output="A unit vector pointing away from the path at that end.",
   body=[
     "Estimates which way a path is heading at one of its ends, used to "
     "decide whether two paths meeting at a junction are really one stroke.",
     "It averages over the last few points rather than using the final step, "
     "because a single step on a thinned centreline is one of only eight "
     "possible directions and is therefore a very noisy estimate of the "
     "true heading. Averaging four points smooths that quantisation out "
     "while still being local enough to describe the end rather than the "
     "whole path.",
   ]),

 "_PruneSpurs": dict(
   inputs="`chains`, the traced paths. `minLen`, the length below which a "
          "stub is removed.",
   output="The paths with short dead ends removed.",
   body=[
     "Removes the whiskers that thinning leaves behind. Where a stroke is "
     "locally wider than elsewhere, or where two strokes cross, the thinning "
     "cannot resolve the shape into a single line and leaves a short branch "
     "attached at one end only.",
     "The test is length plus attachment: a path shorter than the minimum "
     "and connected at just one end is a spur, because a real stroke that "
     "short would have to be a dot, and a dot is not attached to anything. "
     "The threshold is in x-height units so it means the same thing on any "
     "size of writing.",
   ]),

 "_JoinAtJunctions": dict(
   inputs="`chains`, the traced paths. `maxAngleDeg`, how much direction "
          "change is tolerated across a join.",
   output="The paths with collinear pairs merged.",
   body=[
     "Reverses the fragmentation that thinning causes at junctions. When "
     "one pen stroke passes through a point where another stroke touches it, "
     "the thinned image has a junction there and the walk produces three or "
     "four separate paths that all end at the same place.",
     "Two paths are merged when their ends meet and their directions at "
     "those ends agree to within the tolerance, measured with `_Dir`. The "
     "reasoning is that a pen moving in a straight line through a crossing "
     "does not change direction at the crossing, whereas two genuinely "
     "different strokes meet at an angle. The default tolerance of 75 "
     "degrees is generous because handwriting curves as it goes.",
   ]),

 "SimplifyPolyline": dict(
   inputs="`poly`, a path. `eps`, the maximum allowed deviation in pixels.",
   output="A path with fewer points that stays within `eps` of the "
          "original.",
   cite="DouglasPeucker1973",
   body=[
     "A traced path has one point per pixel, most of which say nothing: a "
     "straight run of forty pixels is fully described by its two ends. This "
     "removes a point when doing so moves the path by less than the "
     "tolerance, following the standard line simplification approach "
     "[DouglasPeucker1973].",
     "Why it matters here is not storage but smoothness. The points on a "
     "thinned centreline lie on the pixel grid, so a gently sloping stroke "
     "is stored as a staircase. Simplifying with a tolerance of a little "
     "over half a pixel removes the staircase while leaving every genuine "
     "corner, because a corner is exactly the place where removing a point "
     "would move the path a long way.",
   ]),

 "_TrimLigatureTails": dict(
   inputs="`strokes`, one glyph's strokes. `connL`, `connR`, whether tracing "
          "confirmed a join on each side. Plus thresholds for how low and "
          "how steep a carrier stroke may be.",
   output="`(strokes, entry, exit)` — the trimmed strokes and the points "
          "where the pen entered and left.",
   body=[
     ("sub", "What a ligature tail is"),
     "In joined writing the pen does not lift between letters, so the ink "
     "belonging to the join is shared. When the line is cut at a boundary, "
     "each side keeps part of that join: the letter on the left ends with a "
     "tail reaching towards its neighbour, and the letter on the right "
     "starts with one reaching back.",
     "If those were stored as part of the letter, the synthesiser would draw "
     "the letter *and* draw its own join, so every joined pair would have a "
     "doubled connection. So the tail is removed and, instead, the point "
     "where the pen entered and left is recorded, which is what the "
     "synthesiser needs in order to draw a join of the correct length.",
     ("sub", "How a tail is recognised"),
     "A join is a stroke that runs outward from the letter, stays low, and "
     "does not climb steeply — it is a carrier between two letter bodies, "
     "not part of a body. The test walks inward from each end while those "
     "conditions hold, and trims what it walked over. A second, more "
     "permissive test allows a steeper climb, which catches the joins that "
     "rise towards a tall letter.",
     ("sub", "The two guards"),
     "Trimming only happens on a side where tracing actually confirmed a "
     "join, so a letter the writer lifted the pen before is left alone. And "
     "a minimum fraction of the letter's width must survive, so a short "
     "letter whose whole body looks like a low outward run is not consumed "
     "entirely.",
     "This function is where a known limitation lives. The trimming is a "
     "heuristic, and when it trims too much the stored letter starts part "
     "way up, so the synthesiser's join has to climb from the baseline to "
     "an unnatural height. That is one of the causes of awkward-looking "
     "connections in synthesised cursive.",
   ]),

 "ExtractLineGlyphs": dict(
   inputs="`gray`, one line crop. `text`, its transcript. `model`, a loaded "
          "reader.",
   output="`(glyphs, stats)` — one glyph dictionary per character (or None "
          "where extraction failed) and per-line measurements. `(None, "
          "None)` if the line cannot be aligned.",
   cite="Graves2006, Viterbi1967",
   body=[
     "The heart of the file: one line of writing and its text in, one glyph "
     "per character out. Six stages.",
     ("sub", "1. Ask the recogniser where the characters are"),
     "`CtcForcedAlign` gives the column span of every character. This is the "
     "key idea of the whole approach: rather than trying to find letter "
     "boundaries by looking for gaps, which fails on joined writing, it uses "
     "a network trained on reading to say where it believes each character "
     "sits.",
     ("sub", "2. Snap each boundary to a defensible column"),
     "The recogniser's boundaries are approximate, so each is moved to the "
     "cheapest column nearby, under a cost that prefers thin ink low down:",
     ("formula", ["cutCost = thinness + 1.15 * heightOfHighestInk",
                  "cutCost = 0 wherever there is no ink at all"]),
     "The reasoning is that an honest place to cut joined writing is where "
     "the ligature crosses — one stroke thick and near the baseline. A true "
     "gap costs nothing and therefore always wins. A small penalty keeps the "
     "chosen column near the recogniser's own estimate so the snap cannot "
     "wander to an unrelated gap.",
     ("sub", "3. Build one shared, increasing set of boundaries"),
     "The boundaries are built as a single increasing array rather than "
     "snapped independently. Independent snapping can place two neighbouring "
     "boundaries on the same column, which would give one character an empty "
     "cell, and can also place them out of order. A minimum cell width is "
     "enforced and room is reserved for the cells still to come.",
     ("sub", "4. Thin the whole line once, then cut the centrelines"),
     "The line is skeletonised as a whole and the traced paths are then cut "
     "at the boundaries. Thinning each column slice separately would be "
     "wrong: thinning works on shape, and a slice has an artificial straight "
     "edge at the cut, so a spurious spine appears along it.",
     ("sub", "5. Normalise each letter"),
     "Each character's strokes are converted to the stored convention: "
     "origin at the left edge of the cell on the baseline, y upwards, "
     "divided by the x-height, and the writer's slant removed by shearing. "
     "Strokes are ordered left to right and each is oriented left to right.",
     ("sub", "6. Trim the joins and record the measurements"),
     "`_TrimLigatureTails` removes shared ligature ink, and the glyph is "
     "stored with its width, its own advance, its lead-in gap, its top and "
     "bottom reach, its entry and exit points, and whether tracing found it "
     "joined on each side.",
     ("sub", "Known limitations of this method"),
     "Two are worth recording because they are visible in the results. "
     "First, the cut is a straight vertical line, and letters overlap "
     "horizontally even in well separated handwriting: a 't' crossbar "
     "reaches over its neighbour, and 'y' and 'g' descenders swing back "
     "underneath. Any such stroke is cut at the column and the overhanging "
     "part is filed under the wrong character. Second, the recorded "
     "orientation is left-to-right rather than the true drawing direction, "
     "so the stored entry and exit describe the leftmost and rightmost ends "
     "of the letter rather than where the pen genuinely started and "
     "stopped.",
     "An experiment on assigning each whole traced stroke to the character "
     "owning most of its length, instead of cutting it, improved synthesis "
     "measurably for the most joined hand — character accuracy from 75 to 86 "
     "per cent and writer confidence from 56 to 84 per cent — while making "
     "things worse for several others, so it is recorded here as a measured "
     "result rather than adopted.",
   ]),

 "_ClassOk": dict(
   inputs="`ch`, the character. `top`, `bot`, its ink extent in x-heights "
          "above and below the baseline.",
   output="True if the extent is consistent with that letter.",
   body=[
     "The cheapest and one of the most effective rejection tests. Every "
     "letter belongs to a height class: 'a' and 'o' stay inside the "
     "x-height band, 'b' and 'l' reach up, 'g' and 'p' hang down. If the "
     "stored shape does not reach where that letter must reach, the cut was "
     "wrong.",
     ("formula", ["x-height only   top <= 1.42  and  bot >= -0.42",
                  "ascenders       top >= 1.25",
                  "descenders      bot <= -0.20"]),
     "An 'l' that never rises above the band is not an 'l', whatever else "
     "about it looks reasonable. The tolerances are loose because real "
     "handwriting is variable; the test is there to catch gross failure, not "
     "to enforce typography.",
   ]),

 "_ResampleN": dict(
   inputs="`poly`, a path. `n`, the number of points wanted.",
   output="A path of exactly n points, evenly spaced along the original's "
          "arc length.",
   body=[
     "Respaces a path so that comparisons between two writings of the same "
     "letter are meaningful. Two traced versions of an 'a' will have "
     "different numbers of points in different places, so the fifth point "
     "of one does not correspond to the fifth point of the other. After "
     "resampling both to the same count by arc length, point i of each is at "
     "the same relative distance along the stroke, and they can be compared "
     "or averaged directly.",
   ]),

 "_AlignStroke": dict(
   inputs="`s`, a stroke. `N`, the target point count.",
   output="The stroke resampled and oriented consistently.",
   body=[
     "Prepares one stroke for averaging with others: resample to a common "
     "count, and make sure it runs the same way round as the others, since "
     "averaging a stroke with a reversed copy of itself produces a straight "
     "line through the middle of the letter.",
   ]),

 "_DenoisedPrototype": dict(
   inputs="`variants`, several writings of one character. `minN`, the "
          "minimum number needed. `N`, the resampling resolution.",
   output="One averaged shape, or None if there are too few variants.",
   body=[
     "Builds one clean letterform by averaging several of the writer's own "
     "writings of that letter.",
     "The reasoning is the usual one for averaging repeated measurements. "
     "Each writing of a letter is the shape the person intends plus some "
     "wobble; the intention is common to all of them and the wobble is not, "
     "so averaging suppresses the wobble in proportion to the square root of "
     "the number of samples while leaving the intended shape.",
     "The median is taken at each resampled point rather than the mean, "
     "which matters because the inputs are not clean: one bad cut among ten "
     "variants would drag a mean noticeably, while the median ignores it "
     "entirely unless it is in the majority.",
     "This is used as a fallback letterform, and only when it measurably "
     "beats the writer's real variants on the typicality measure, so a "
     "writer whose letters are consistent keeps their real shapes.",
   ]),

 "SmoothStroke": dict(
   inputs="`pts`, a stroke. `step`, the resampling spacing. `passes`, how "
          "many smoothing passes.",
   output="A smoother stroke.",
   body=[
     "Removes the pixel staircase from a path by resampling at even spacing "
     "and applying a short moving average. The staircase is an artefact of "
     "the pixel grid, not of the handwriting, and it is visible in the "
     "drawn output as faint angularity on curves.",
     "The number of passes is small deliberately. Repeated smoothing is a "
     "diffusion process: it will eventually turn any letter into a straight "
     "line, and it rounds off genuine corners before it finishes removing "
     "the staircase. Two passes is enough to clear the quantisation.",
   ]),

 "_PenLength": dict(
   inputs="`g`, a glyph.",
   output="The total pen path length of all its strokes, in x-heights.",
   body=[
     "Used as a size test. A fragment of a letter has far less pen path than "
     "the letter itself, so comparing a candidate's pen length against the "
     "median for that character is a reliable way to catch a cut that kept "
     "only part of the shape.",
     "Pen length is used rather than bounding box because a fragment can "
     "have a deceptively large bounding box — half an 'o' spans nearly the "
     "full width and height of the whole letter while containing half the "
     "ink.",
   ]),

 "_LongestStroke": dict(
   inputs="`g`, a glyph.",
   output="The length of its single longest stroke.",
   body=[
     "A complementary test to total pen length. A glyph can accumulate "
     "enough total length from several short pieces while containing no "
     "single stroke long enough to be a letter body, which is what a "
     "collection of fragments looks like. Requiring one substantial stroke "
     "rejects that case.",
   ]),

 "_ShapeGrid": dict(
   inputs="`g`, a glyph. `n`, the grid resolution, 12 by default.",
   output="A unit-length vector of n*n numbers describing the shape, or None "
          "if the glyph has no ink.",
   body=[
     ("sub", "The purpose"),
     "To compare letter shapes numerically, a shape has to become a vector. "
     "This lays the glyph on a 12 by 12 grid, counts how much pen path falls "
     "in each cell, blurs the result slightly, and normalises it to unit "
     "length. Two shapes can then be compared by the angle between their "
     "vectors.",
     ("sub", "Each step, and why"),
     "The strokes are walked at fine intervals and accumulated into cells, "
     "so the descriptor measures pen path rather than vertex positions, "
     "which makes it independent of how the path happened to be sampled. "
     "The square root of each cell count is taken, which damps the effect of "
     "a place where the pen dwelt or where strokes overlap. A 3 by 3 blur "
     "then spreads each cell into its neighbours, so that two writings of "
     "the same letter whose strokes fall a cell apart still produce similar "
     "vectors — without it the descriptor would be far too brittle. "
     "Finally the vector is scaled to unit length so that comparison by dot "
     "product is a comparison of shape and not of size.",
     ("sub", "A limitation that was measured"),
     "The x axis is divided by the glyph's own width and the y axis by its "
     "own height, independently. That makes the descriptor invariant to "
     "aspect ratio — which also means it **cannot see** that a glyph is "
     "twice as wide as that letter should be, which is the single clearest "
     "symptom of a cut that swallowed part of a neighbour. The shape is "
     "squeezed back into the same box before being scored.",
     "There is a second, subtler weakness. Both vectors are non-negative "
     "and blurred, so their dot product is high for almost any pair, and the "
     "useful range is compressed: across the twelve variants of one letter "
     "for one writer the distances span only 0.088 to 0.238, meaning cosine "
     "similarities from 0.91 down to 0.76. With that little spread the "
     "ordering within the top few variants is close to arbitrary.",
     "An experiment using one scale for both axes, so that width-to-height "
     "ratio survives into the descriptor, changed every prior distance and "
     "raised measured writer confidence by about 7 points while costing "
     "about 3 points of character accuracy. It also shifted the scale of "
     "the distances themselves, which moves the automatic legibility "
     "threshold described in synthesise.py, so adopting it would require "
     "recalibrating that.",
   ]),

 "BuildLetterPrior": dict(
   inputs="`libsByAuthor`, every writer's raw variants. `minSamples`, the "
          "minimum number of examples before a character gets a reference, "
          "12 by default.",
   output="A dictionary from character to a unit reference vector.",
   body=[
     "Builds a reference shape for each letter by pooling every writer's "
     "examples and taking the median value in each grid cell.",
     ("sub", "Why pool across writers"),
     "The question a filter needs to answer is 'is this shape a letter at "
     "all', which is different from 'is this shape typical of this writer'. "
     "The second is answered by `_ConsensusFilter` using the writer's own "
     "variants; the first needs an outside opinion, and pooling across "
     "thirteen hands provides one.",
     ("sub", "Why this cannot contaminate a profile"),
     "The pooled shape is used **only to reject**. It is never written into "
     "a profile and never drawn. If a writer's 'a' is unusual but "
     "well-formed it survives; the prior only removes shapes that are not "
     "recognisably that letter in anyone's hand. This boundary is the reason "
     "a synthesised page can honestly be called that writer's hand.",
     ("sub", "Why the median"),
     "The pool contains bad cuts, since it is assembled before filtering. "
     "The median in each cell is insensitive to them unless they are the "
     "majority, where a mean would be pulled by every fragment.",
     ("sub", "A circularity worth noting"),
     "Because the pool is unfiltered, the reference has absorbed whatever "
     "contamination the cutter produces — and the cutter fails in the same "
     "way for every writer, so it is systematic rather than random noise "
     "that the median could cancel. Building the prior, filtering with it, "
     "and rebuilding from the survivors was tested and raised writer "
     "confidence by about 3 points while costing about 2.5 points of "
     "character accuracy.",
     "The minimum sample count means rare characters get no reference and "
     "are passed through unfiltered, which is deliberate: a reference built "
     "from four examples would be noise.",
   ]),

 "_PriorScore": dict(
   inputs="`g`, a glyph. `ch`, its character. `prior`, the reference "
          "dictionary.",
   output="A distance in [0, 2], where 0 means identical in shape to the "
          "reference.",
   body=[
     "The cosine distance between the glyph's descriptor and the pooled "
     "reference for its character:",
     ("formula", ["priorD = 1 - dot( shapeGrid(g) , prior[ch] )"]),
     "Both vectors are unit length, so the dot product is the cosine of the "
     "angle between them and the distance is zero for identical shapes.",
     "This single number is stored on every glyph as `priorD` and is used "
     "throughout the rest of the pipeline: to rank a character's variants, "
     "to decide whether a shape is trustworthy enough to draw, and, through "
     "its median over the whole library, to decide how far the writer's "
     "library can be trusted at all. If the prior has no entry for the "
     "character the function returns 0.0, which silently marks the glyph as "
     "a perfect match — a loophole worth knowing about, since it makes "
     "unfiltered rare characters rank first.",
   ]),

 "PriorFilter": dict(
   inputs="`variants`, one character's glyphs. `ch`, the character. "
          "`prior`. `keepFrac`, the fraction to keep. `minKeep`, the floor.",
   output="The surviving variants.",
   body=[
     "Ranks a character's variants by prior distance and keeps the best "
     "fraction. This is a relative test rather than an absolute threshold, "
     "which is the right choice because what counts as a good score varies "
     "by letter: a simple 'l' matches its reference closely and an "
     "elaborate 'k' never will, so one fixed cut-off would keep every 'l' "
     "and reject every 'k'.",
     "The floor on how many to keep prevents a character with few examples "
     "being emptied, since an empty entry forces the synthesiser onto a "
     "fallback letterform for that character everywhere.",
   ]),

 "_ConsensusFilter": dict(
   inputs="`variants`, one character's glyphs. `keepFrac`.",
   output="The variants closest to their own medoid.",
   body=[
     "Asks a different question from the prior filter: not 'is this a "
     "letter' but 'does this agree with how *this writer* forms this "
     "letter'. The most typical variant is found as the medoid — the one "
     "with the smallest total distance to all the others — and the variants "
     "furthest from it are dropped.",
     "The medoid is used rather than the mean because it is an actual "
     "variant, so it cannot be a blend of a good shape and a bad cut. For "
     "the same reason it is the shape used as the writer's fallback "
     "letterform elsewhere.",
     "This catches a specific failure the prior filter cannot: a glyph "
     "contaminated by a neighbouring letter can still resemble the pooled "
     "reference for its character, because the descriptor ignores aspect "
     "ratio, but it will not resemble the writer's own other writings of "
     "that letter.",
   ]),

 "RenderRefStats": dict(
   inputs="`gray`, one real line crop from the writer.",
   output="A dictionary of measurements: ink fraction, aspect, paper and ink "
          "levels, mid-band fraction, measured slant, ascender and descender "
          "reach, word gap, and height in x-heights.",
   body=[
     "Measures how the writer's real line looks *through the same "
     "preprocessing the networks apply*, which is what synthesis has to "
     "reproduce.",
     "The reason this exists separately from the glyph library is that "
     "matching letter shapes is not sufficient. A synthetic line can be "
     "built from genuinely correct letterforms and still be rejected by the "
     "writer namer because it is too lightly inked, too widely spaced, or "
     "sits at a different height in its crop. These are properties of the "
     "line as a whole, so they are measured on the whole line and compared "
     "against the synthesiser's own output in `StyleCalibration`.",
     "Word gaps are measured from the actual gaps between ink runs rather "
     "than from a typographic rule, since how far apart somebody puts their "
     "words is part of their hand.",
   ]),

 "ExtractAuthorRaw": dict(
   inputs="`authorId`, `lineItems` as `(gray, text)` pairs, and a loaded "
          "reader.",
   output="A list of `(glyphs, stats)`, one entry per line that aligned "
          "successfully.",
   body=[
     "Runs the cutting over every line of one writer and returns the "
     "**unfiltered** result, including the per-line reference statistics.",
     "Keeping the raw output separate from the filtering is a deliberate "
     "split, and it is what made the experiments in this project possible. "
     "Alignment and cutting are the slow part, taking minutes per writer; "
     "filtering is seconds. Caching the raw result means a change to a "
     "filter threshold can be evaluated across all writers in seconds "
     "instead of being a full rebuild, and it means two filtering "
     "configurations can be compared on provably identical input.",
   ]),

 "_MedianWordGap": dict(
   inputs="`refs`, the per-line reference statistics. `spaceAdvs`, the "
          "widths of space characters from alignment.",
   output="A typical word gap in x-heights.",
   body=[
     "Prefers the gaps measured from real ink over the widths that "
     "alignment assigned to space characters, falling back to the latter "
     "only when the former are unavailable.",
     "The preference is because a space has no ink, so the recogniser's "
     "notion of where it sits is the least constrained part of its "
     "alignment and the width it assigns is correspondingly unreliable. The "
     "actual distance between the end of one word's ink and the start of "
     "the next is directly observable.",
     "The median rather than the mean, again, because a line that ends "
     "mid-sentence or a missed word boundary produces an enormous outlier.",
   ]),

 "BuildAuthorProfile": dict(
   inputs="`authorId`, `parsed` from `ExtractAuthorRaw`, optional `refs` and "
          "`prior`, and a `verbose` flag.",
   output="The complete profile as a dictionary, ready to be written as "
          "JSON, or None if no usable lines survive.",
   body=[
     "Takes the raw cut glyphs and produces the stored profile. This is "
     "where all the filtering happens, in a deliberate order.",
     ("sub", "Reject whole lines first"),
     "Lines whose alignment confidence falls below a percentile of this "
     "writer's own distribution are discarded entirely. A line the "
     "recogniser could not read confidently has unreliable character "
     "boundaries throughout, so every glyph from it is suspect; rejecting "
     "the line is cheaper and safer than trying to salvage glyphs from it.",
     ("sub", "The five gates, cheapest first"),
     "Within each character: simple geometry first (width within limits, "
     "some vertical extent, not too many strokes, the height class test, "
     "ink not much wider than the cell it was cut from, enough pen path, "
     "one long enough stroke). Then outlier rejection on width and height "
     "against the character's own 40th-percentile width and median height. "
     "Then `PriorFilter` against the cross-writer reference. Then "
     "`_ConsensusFilter` against the writer's own medoid. Finally a ranking "
     "by typicality, keeping at most `MAX_VARIANTS_PER_CHAR`.",
     "The ordering is pure economy: the geometric tests are a few "
     "comparisons per glyph and remove most of the rubbish, so the "
     "expensive descriptor comparisons run on far fewer candidates.",
     ("sub", "The fallback letterforms"),
     "Two per character are stored. The medoid is the single most typical "
     "real variant, intact. The denoised prototype is the per-point median "
     "of several variants, which is smoother but can blur a letter the "
     "writer forms inconsistently — so it is stored only when it measurably "
     "beats the real variants on prior distance.",
     ("sub", "The global measurements"),
     "Slant, x-height, stroke width, ascender and descender reach, word "
     "gap, per-character advance, and connectedness. Connectedness is the "
     "fraction of adjacent in-word letter pairs where tracing found the ink "
     "continuous, which makes it a measured property of the hand rather "
     "than a judgement — and it turns out to be the single most predictive "
     "number in the whole profile, since nearly every quality difference "
     "between writers in the results tracks it.",
     ("sub", "A note on the stored order"),
     "The variants are finally sorted by closeness to the reference width "
     "and height, not by prior distance, and then truncated. So the order "
     "in the JSON file is not the order the synthesiser uses, and a variant "
     "with an excellent prior distance can be dropped at this point for "
     "being the wrong size.",
   ]),
}
