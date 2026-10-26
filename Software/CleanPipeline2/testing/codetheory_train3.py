"""train.py, part 3: data, augmentation, initialisation and the loops.

Getting training pages off disk and into batches, inventing extra examples
from the ones that exist, choosing the starting weights, and the two
training loops themselves.
"""

FUNCS = {

 "EncodeCrop": dict(
   inputs=[("arr", "a line crop as an array")],
   output="the crop packed into bytes for storage.",
   theory=[
     ("h", "Why crops are cached at all"),
     ("p", "Extracting line crops from a page is slow: the page must be "
           "read, binarised, analysed for line positions and cut. Training "
           "visits every line once per epoch for dozens of epochs, so "
           "repeating that work would dominate the run. The crops are "
           "therefore extracted once and stored."),
     ("h", "What encoding means here"),
     ("p", "The array is flattened into a byte sequence with its shape "
           "recorded alongside, so it can be written into a single file and "
           "read back by offset. Storing each crop as a separate image file "
           "would mean tens of thousands of small files, which is slow to "
           "open and wasteful of disk, since a filesystem allocates a whole "
           "block to a file however small."),
   ]),

 "DecodeCrop": dict(
   inputs=[("raw", "the stored bytes")],
   output="the crop as an array again.",
   theory=[
     ("h", "What it does"),
     ("p", "Reverses the encoding: reads the shape and reinterprets the "
           "bytes as an array of that shape. Reinterpreting rather than "
           "converting means no arithmetic is performed and no copy is made "
           "beyond the read itself, which matters because this runs for "
           "every crop of every batch of every epoch."),
   ]),

 "ReadLabelLines": dict(
   inputs=[("imgPath", "the page image's path"),
           ("labelPath", "the transcript's path, derived from the image's "
                         "if not given")],
   output="the transcript as a list of lines.",
   theory=[
     ("h", "What it is for"),
     ("p", "Reads the text that was written on a page. Supervised training "
           "needs the correct answer for every input, and for a page of "
           "handwriting that is the transcript, typed when the page was "
           "collected."),
     ("h", "Why the transcript is per line rather than per page"),
     ("p", "The reader works on one line at a time, so each crop needs its "
           "own label. Keeping the transcript as one line of text per line "
           "of writing means the correspondence is positional: the n-th "
           "transcript line labels the n-th crop. That is what makes it "
           "possible to check the extraction, since a page whose crop count "
           "does not match its transcript line count has been segmented "
           "wrongly and can be rejected rather than silently mislabelled."),
     ("note", "Deriving the transcript's path from the image's by "
              "convention, rather than being told it, is what lets a "
              "directory of pages be processed by listing the images alone. "
              "The cost is that the convention is implicit: a transcript "
              "named differently is simply not found, and the page is "
              "skipped."),
   ]),

 "ClusterRows": dict(
   inputs=[("rows", "row indices"),
           ("maxGap", "the largest gap that still counts as the same "
                      "cluster, default 5")],
   output="the indices grouped into runs of nearby rows.",
   theory=[
     ("h", "What it is for"),
     ("p", "Given a set of interesting rows, such as those containing a "
           "printed rule, groups the ones that are close together. A single "
           "printed line a few pixels thick registers as several adjacent "
           "rows, and what is wanted is one entry per rule rather than one "
           "per row."),
     ("h", "The rule"),
     ("p", "The indices are taken in order and a new cluster is started "
           "whenever the gap to the previous one exceeds the limit. This is "
           "single-link clustering in one dimension, and it is the natural "
           "formulation when the data is ordered: two rows belong together "
           "if they are close, and closeness chains, so a run of rows each "
           "near the last forms one cluster however long it is."),
     ("note", "A gap of five pixels suits the printed rules on ruled paper, "
              "which are a pixel or two thick with occasional breaks where "
              "writing crosses them. The chaining property is what bridges "
              "those breaks, and it is also the method's weakness: a dense "
              "enough sequence of separate features merges into one "
              "cluster."),
   ]),

 "FindRuleLineClusters": dict(
   inputs=[("binaryImg", "the binarised page"),
           ("rawGrayscaleArray", "the original grey page, if available")],
   output="the row clusters that look like printed rules.",
   theory=[
     ("h", "What it is for"),
     ("p", "Finds the printed ruling on lined paper, so that it can be "
           "removed before the handwriting is analysed and so that its "
           "positions can hint at where the lines of writing sit."),
     ("h", "How a printed rule is recognised"),
     ("p", "By the same property segment.py uses: a printed rule crosses "
           "the whole page in one straight unbroken run, which handwriting "
           "never does. Rows containing an unusually long horizontal run of "
           "ink are therefore candidates, and clustering them groups each "
           "rule into one entry."),
     ("h", "Why the rules are useful as well as unwanted"),
     ("p", "They are removed because they corrupt the analysis, joining "
           "every letter on a line into one blob. But their positions are "
           "also the best possible evidence about where lines of writing "
           "are, because the writer wrote on them. So they are returned "
           "rather than merely deleted, and the line splitter takes them as "
           "hints."),
     ("note", "This is a separate implementation from the equivalent in "
              "segment.py, because this file processes database pages and "
              "that one processes photographs, which differ enough that one "
              "routine serving both would be a compromise for each. The "
              "duplication is real but the inputs genuinely differ: a "
              "database scan has even lighting and no perspective."),
   ]),

 "RemoveStraightFormLines": dict(
   inputs=[("hwRegion", "the region of the page containing handwriting")],
   output="the region with the printed ruling deleted.",
   theory=[
     ("h", "What it does"),
     ("p", "Deletes the long straight runs identified as printing, leaving "
           "the handwriting. The reasoning is set out at length under "
           "RemovePrintedStructure in segment.py: length, not darkness or "
           "thickness, distinguishes printing from writing, and the "
           "threshold sits in a wide gap between the two."),
     ("h", "Why the removal must be of whole runs"),
     ("p", "Removing the rule at the places where writing crosses it would "
           "cut the writing too, since at a crossing the same pixels belong "
           "to both. The remedy used here and in segment.py is to restrict "
           "the deletion to pixels that form part of a long run, which a "
           "letter's stroke does not, so a crossing loses the rule and "
           "keeps the letter."),
   ]),

 "DetectPageBounds": dict(
   inputs=[("binaryImg", "the binarised page"),
           ("rawGrayscaleArray", "the grey page, if available")],
   output="the rectangle of the page that holds handwriting.",
   theory=[
     ("h", "What it is for"),
     ("p", "A database page is a scanned form, with a printed header, a "
           "typed prompt and sometimes a footer, of which only the "
           "handwritten region is wanted. This finds that region so the "
           "rest can be ignored."),
     ("h", "Why the printed parts must be excluded rather than tolerated"),
     ("p", "Printed text is text, so the reader would read it, and it has "
           "no corresponding transcript line, so every crop after it would "
           "be labelled with the wrong line. A single extra crop at the top "
           "of a page misaligns the whole page's labels, which is far worse "
           "than losing the page: a mislabelled example teaches the network "
           "something false, where a missing one teaches nothing."),
     ("note", "This is why the crop count is checked against the transcript "
              "line count and the page rejected if they disagree. The "
              "check cannot tell which crop is wrong, only that something "
              "is, and rejecting the page is the right response: the "
              "database has thousands of pages and losing a few is "
              "cheaper than training on bad labels."),
   ]),

 "MergeUndersizedFragments": dict(
   inputs=[("lines", "the candidate line regions"),
           ("minRelativeHeight", "the least height relative to the typical "
                                 "line, default 0.4")],
   output="the regions with the too-short ones merged into their "
          "neighbours.",
   theory=[
     ("h", "The problem it solves"),
     ("p", "Line splitting can divide one line of writing into two, "
           "typically when a row of low ink inside the line is mistaken for "
           "the gap between lines. The symptom is a region much shorter "
           "than the others, since one real line has been halved."),
     ("h", "Why the test is relative"),
     ("p", "The height is compared with the typical line height on the same "
           "page rather than with a fixed figure, because the right height "
           "depends on the writing and the scan resolution. A region under "
           "40 percent of the typical height cannot be a line, since no "
           "writer varies their line height that much within a page, so it "
           "must be a fragment of one."),
     ("h", "Why merging rather than discarding"),
     ("p", "A fragment contains real writing, which the transcript expects "
           "to find. Discarding it would lose that text and, worse, would "
           "change the crop count and so misalign every label after it. "
           "Merging it back into its neighbour restores both the writing and "
           "the count."),
   ]),

 "MergeOrSplitToExpectedLineCount": dict(
   inputs=[("lines", "the candidate line regions"),
           ("rowInkCount", "the ink per row profile"),
           ("expectedLineCount", "how many lines the transcript has")],
   output="the regions adjusted to match that count.",
   theory=[
     ("h", "The idea: use the answer to fix the method"),
     ("p", "The transcript says how many lines the page has. That is "
           "information the splitter did not use, and it is reliable, "
           "because it was typed by a person reading the page. So if the "
           "splitter produced the wrong number, the discrepancy is known "
           "exactly, and the regions can be adjusted until the count "
           "agrees: merging the most doubtful boundary when there are too "
           "many, splitting the most promising region when there are too "
           "few."),
     ("h", "Why this is legitimate and where its limit is"),
     ("p", "Using the label to guide the extraction would be improper if "
           "the label were being predicted, and it is not: the task here is "
           "to prepare training data, and the transcript is given for "
           "exactly that purpose. It is the same reasoning that justifies "
           "forced alignment in style.py, where the known text is used to "
           "locate characters rather than to recognise them."),
     ("p", "The limit is that it cannot verify correctness, only count. A "
           "page split in the wrong places but into the right number of "
           "pieces passes, and its labels are then confidently wrong. The "
           "count is a necessary condition for a correct extraction, not a "
           "sufficient one."),
   ]),

 "EstimateLinePitch": dict(
   inputs=[("smoothedRowInk", "the smoothed ink per row profile"),
           ("minPitch", "the smallest pitch considered, default 40"),
           ("maxPitch", "the largest, default 400")],
   output="the estimated spacing between lines of writing, in pixels.",
   theory=[
     ("h", "What it is for"),
     ("p", "The same quantity LinePitch finds in segment.py, estimated from "
           "a smoothed row profile, and used as the scale for the line "
           "splitting below. The reasoning for using a projection profile, "
           "and for smoothing it first, is given in that entry."),
     ("h", "Why the search is bounded"),
     ("p", "The bounds keep the estimate within the range a line of writing "
           "on a scanned page can plausibly occupy. Without a lower bound "
           "the method can lock onto the spacing between the strokes within "
           "a word, which is a genuine periodicity in the profile and a much "
           "shorter one. Without an upper bound it can lock onto the "
           "spacing between paragraphs. Both are real patterns; the bounds "
           "say which scale is being asked about."),
   ]),

 "SplitLinesByPeriodicity": dict(
   inputs=[("binaryRegion", "the handwriting region"),
           ("ruleLineHints", "the printed rule positions, if found"),
           ("smoothWindow", "the profile smoothing width, default 9"),
           ("minLineHeight", "the shortest acceptable line, default 30")],
   output="the row ranges of the individual lines of writing.",
   theory=[
     ("h", "The approach, and why periodicity is the right tool"),
     ("p", "Lines of writing on a page are close to evenly spaced, because "
           "the writer followed the ruling or kept a steady hand. That "
           "regularity is strong evidence and it is what the splitter uses: "
           "rather than looking for each gap independently, it establishes "
           "the pitch and then expects boundaries at roughly that spacing."),
     ("p", "The advantage over independent gap-finding is robustness to a "
           "single bad gap. A line whose descenders reach into the next "
           "line leaves almost no gap at all, and a method looking only for "
           "gaps misses that boundary entirely, merging two lines. A method "
           "that knows the expected spacing looks for a boundary where one "
           "ought to be and places it at the best available row, even where "
           "that row is not a clear gap."),
     ("h", "Why the printed rules are used as hints"),
     ("p", "Where the ruling was found, it says where the writer intended "
           "each line to sit, which is better evidence than the ink "
           "profile: it is unaffected by descenders, by a line of unusual "
           "height, or by a blank line. The hints are used to guide the "
           "boundaries rather than to set them, because writers do not "
           "follow ruling exactly."),
     ("note", "This whole routine is a separate, earlier solution to the "
              "problem segment.py solves with seam carving. The seam "
              "approach is better, since it cuts around descenders instead "
              "of choosing a row to cut through. This one remains because "
              "it is what the cached training data was extracted with, and "
              "re-extracting it would change the data the reported "
              "checkpoints were trained on."),
   ]),

 "ExtractLinePatches": dict(
   inputs=[("page", "the page image"),
           ("targetHeight", "the height to scale crops to, default 32"),
           ("maxWidth", "the greatest width kept, default 1024"),
           ("expectedLineCount", "how many lines the transcript has"),
           ("labelLines", "the transcript lines")],
   output="the line crops, with their labels where the count matched.",
   theory=[
     ("h", "What it is for"),
     ("p", "The whole page-to-crops pipeline for database pages, in one "
           "call: find the page bounds, remove the printed ruling, split "
           "into lines, merge the fragments, reconcile against the expected "
           "count, and cut and scale each crop."),
     ("h", "Why the crops are scaled here"),
     ("p", "Each crop is scaled to a fixed height, preserving its aspect "
           "ratio. Doing it at extraction rather than at training time "
           "means the cache holds crops of a known height, so loading a "
           "batch needs no resizing, and the cache is smaller, since a "
           "scanned line is several times taller than the network needs."),
     ("h", "Why width is capped rather than scaled"),
     ("p", "An unusually wide crop is truncated rather than squeezed. "
           "Squeezing would change the aspect ratio and so the letter "
           "proportions, for the reasons given under ResizeLineImageFixed. "
           "An over-wide crop is also usually a fault rather than a long "
           "line, two lines having been merged, so truncating loses little "
           "that was right."),
     ("note", "Where the crop count does not match the transcript, the "
              "crops are returned without labels and the page cannot be "
              "used for training the reader, though it remains usable for "
              "the writer namer, which needs only to know who wrote the "
              "page. That is why the two networks are trained from "
              "different views of the same cache."),
   ]),

 "EncodeText": dict(
   inputs=[("text", "a line of text")],
   output="the sequence of class indices for its characters.",
   theory=[
     ("h", "What it does"),
     ("p", "Converts a label into the numbers CTC works with, by looking "
           "each character up in the charset mapping from recognise.py and "
           "dropping anything not in it."),
     ("h", "Why unknown characters are dropped rather than substituted"),
     ("p", "A character outside the charset has no output slot, so the "
           "network cannot be asked to produce it. Substituting a "
           "placeholder would teach it to emit that placeholder whenever "
           "something unfamiliar appears, which is worse than omitting it: "
           "the network would learn a response to a situation it can never "
           "handle correctly."),
     ("p", "Dropping the character does introduce a small inconsistency, "
           "since the image contains a mark the label does not mention. CTC "
           "tolerates that, because it allows the network to emit blank "
           "there, and the alternative would be discarding every line "
           "containing any unusual character, which over a large database "
           "is a substantial loss."),
   ]),

 "FindPeaks": dict(
   inputs=[("sm", "a smoothed profile"),
           ("floorFrac", "the least height as a fraction of the maximum"),
           ("minGap", "the least separation between peaks"),
           ("prominence", "the least prominence required"),
           ("distance", "an alternative name for the separation")],
   output="the indices of the accepted peaks, in order.",
   theory=[
     ("h", "What it is for"),
     ("p", "A general peak finder, used by the line splitting. It applies "
           "three filters in turn, each addressing a different way a "
           "spurious peak arises."),
     ("h", "Finding candidates"),
     ("p", "A candidate is at least as large as its left neighbour and "
           "strictly larger than its right. The asymmetry is deliberate, as "
           "the code comment notes: if both comparisons were inclusive, a "
           "flat-topped peak several samples wide would report every one of "
           "those samples. With one strict comparison it reports exactly "
           "one."),
     ("h", "The height floor, and a trap in it"),
     ("p", "A candidate must exceed a fraction of the profile's maximum. "
           "The comment records a real pitfall: the floor must not default "
           "to zero, because the profile can be negative. A profile with "
           "its mean removed has negative valleys, and a peak in a "
           "negative region is still a peak. Defaulting the floor to "
           "negative infinity rather than zero is what makes the filter "
           "optional rather than silently active."),
     ("h", "Prominence, which is the interesting filter"),
     ("p", "**Prominence** measures how much a peak stands out from its "
           "surroundings, as opposed to how tall it is. It is defined as "
           "the peak's height above the higher of the two lowest points "
           "reached before the profile rises above the peak again on either "
           "side."),
     ("eq", r"\mathrm{prom}(i) \;=\; \mathrm{sm}[i] \;-\; \max\big("
            r"\min_{j \in \mathrm{left}} \mathrm{sm}[j],\; "
            r"\min_{j \in \mathrm{right}} \mathrm{sm}[j]\big)",
      [(r"\mathrm{prom}(i)", "the prominence of the peak at index i"),
       (r"\mathrm{sm}[i]", "the profile value at the peak"),
       (r"\mathrm{left},\, \mathrm{right}", "the stretches on each side, "
                                            "walked outward until the "
                                            "profile exceeds the peak"),
       (r"\min", "the lowest value reached in each stretch, called the key "
                 "saddle"),
       (r"\max", "the higher of the two, which is what the peak must rise "
                 "above to be independent")],
      None),
     ("p", "The code comment names what this rejects: a ripple riding on a "
           "big peak's flank. Such a ripple can be tall in absolute terms, "
           "taller than a genuine small peak elsewhere, so a height filter "
           "keeps it. But it does not descend far on either side before the "
           "profile rises again, so its prominence is small and this filter "
           "removes it. Height asks how large a peak is; prominence asks "
           "whether it is a peak at all rather than a bump on something "
           "larger. The term is borrowed from topography, where it "
           "distinguishes an independent summit from a shoulder of a bigger "
           "mountain."),
     ("h", "The separation filter"),
     ("p", "Finally, peaks too close together are thinned by the same "
           "greedy strongest-first suppression used for the line centres in "
           "segment.py: take the tallest, exclude everything within the "
           "minimum separation, repeat. Processing tallest first is what "
           "makes it correct rather than arbitrary, for the reason given "
           "there."),
     ("note", "The three filters are each optional and disabled by default, "
              "so the function is a plain local-maximum finder unless "
              "asked for more. That keeps one implementation usable for "
              "both the careful line-splitting case and the simple ones."),
   ]),

 "MyAffine": dict(
   inputs=[("img", "the image"), ("angleDeg", "the rotation in degrees"),
           ("shearDeg", "the shear in degrees"),
           ("fillValue", "what to fill uncovered pixels with, default 255")],
   output="the image rotated and sheared about its centre.",
   theory=[
     ("h", "Background: what an affine transformation is"),
     ("p", "An **affine** transformation is any combination of rotation, "
           "scaling, shearing and translation: it maps straight lines to "
           "straight lines and keeps parallel lines parallel. Every such "
           "transformation can be written as a matrix multiplication once "
           "the coordinates are expressed in **homogeneous** form, which "
           "means appending a 1 to each point so that a translation becomes "
           "part of the matrix rather than a separate addition."),
     ("h", "Why the transformations are composed as matrices"),
     ("eq", r"M \;=\; T_{+c} \; R \; S \; T_{-c}",
      [("M", "the combined transformation"),
       ("T_{-c}", "a translation moving the image's centre to the origin"),
       ("S", "the shear"),
       ("R", "the rotation"),
       ("T_{+c}", "a translation moving the centre back")],
      None),
     ("p", "Rotation and shear are both defined about the origin, which for "
           "an image is a corner. Applying them directly would swing the "
           "image about that corner and out of frame. Translating the "
           "centre to the origin first, transforming, and translating back "
           "makes both act about the centre instead. Matrix composition is "
           "what allows the four steps to be combined into one, so the "
           "image is resampled once rather than four times, which matters "
           "because each resampling blurs it slightly."),
     ("p", "The order reads right to left, since the rightmost matrix acts "
           "on the point first. That is the usual convention for "
           "column-vector coordinates and it is the opposite of the order "
           "the operations are listed in prose."),
     ("h", "Why the matrix is inverted"),
     ("p", "The combined matrix is inverted before being used. The reason "
           "is the direction resampling works in: for each *output* pixel, "
           "the warp asks which input position it came from, which is the "
           "inverse map. Working forward instead, pushing each input pixel "
           "to its output position, leaves gaps where no input pixel "
           "happens to land and collisions where two do. Working backwards "
           "fills every output pixel exactly once, which is why it is "
           "universal in image warping. The same argument appears under "
           "Warp in segment.py."),
   ]),

 "AugmentShear": dict(
   inputs=[("img", "a line crop")],
   output="the crop with a random shear applied.",
   theory=[
     ("h", "Background: what augmentation is for"),
     ("p", "**Data augmentation** creates extra training examples by "
           "altering existing ones in ways that do not change their label. "
           "A sheared line of handwriting still says the same words, so it "
           "is a valid extra example. The purpose is to teach the network "
           "which variations are irrelevant: having seen the same line at "
           "many slants, it learns to read the letters rather than to "
           "recognise one particular slant."),
     ("h", "Why shear in particular"),
     ("p", "Slant is the largest single source of variation between hands, "
           "and between one writer's pages. Augmenting it is directly "
           "useful because the network must read every slant, and it is "
           "cheap because shearing is exactly the transformation that "
           "changes slant and nothing else."),
     ("h", "The range, and why it is expressed as a tangent"),
     ("p", "The shear factor is drawn uniformly from −0.6 to 0.6 and then "
           "converted to an angle. Working in the factor rather than the "
           "angle is deliberate: the factor is what the shear matrix "
           "contains, so a uniform draw on it gives a uniform "
           "transformation strength, where a uniform draw on the angle "
           "would concentrate the effect near the extremes. A factor of 0.6 "
           "is about 31 degrees, which spans the range of real handwriting, "
           "from upright to the steepest cursive in this project's "
           "writers."),
   ]),

 "AugmentRotate": dict(
   inputs=[("img", "a line crop")],
   output="the crop rotated by a small random angle.",
   theory=[
     ("h", "What it simulates"),
     ("p", "A line that is not quite level, which happens when a page is "
           "photographed slightly askew or when a writer's line drifts. The "
           "range is only 2.5 degrees either way, far smaller than the "
           "shear range."),
     ("h", "Why the range is small"),
     ("p", "Because the real variation is small. The segmenter deskews "
           "every page before cutting crops, so by the time a crop reaches "
           "the network its residual tilt is a fraction of a degree. "
           "Augmenting much further than the real variation teaches the "
           "network to handle a situation that never occurs, at the cost of "
           "capacity it could spend on ones that do. A rotation also "
           "changes the measured slant and the apparent x-height, so a "
           "large one would conflict with the shear augmentation rather "
           "than complementing it."),
   ]),

 "FindPerspectiveCoeffs": dict(
   inputs=[("startpoints", "the four source corners"),
           ("endpoints", "where they should land")],
   output="the eight coefficients of the perspective transformation.",
   theory=[
     ("h", "Background: what a perspective transformation is"),
     ("p", "An affine transformation keeps parallel lines parallel, which a "
           "photograph taken at an angle does not: the far edge of a page "
           "appears shorter than the near one, so its sides converge. A "
           "**perspective** or **projective** transformation is the more "
           "general map that allows that convergence, and it is what "
           "actually happens when a flat page is photographed from an "
           "angle."),
     ("eq", r"x' \;=\; \frac{a x + b y + c}{g x + h y + 1}, \qquad "
            r"y' \;=\; \frac{d x + e y + f}{g x + h y + 1}",
      [("x,\\,y", "a point in the source image"),
       ("x',\\,y'", "where it lands"),
       ("a \\dots f", "the six coefficients an affine map would also have"),
       ("g,\\,h", "the two extra coefficients that produce the perspective "
                  "effect"),
       ("1", "the fixed ninth coefficient; the map is unchanged by scaling "
             "all nine, so one can be fixed")],
      None),
     ("p", "The division by a function of position is what distinguishes it "
           "from an affine map and what makes parallel lines converge: "
           "points further away in the implied depth direction are divided "
           "by a larger number and so are compressed toward one another."),
     ("h", "How the eight coefficients are found"),
     ("p", "Four corner correspondences are given, each fixing two "
           "equations, giving eight equations for the eight unknowns. The "
           "equations above are not linear in the coefficients because of "
           "the division, but multiplying each through by its denominator "
           "makes them linear, and that rearrangement is what the rows of "
           "the matrix in the code are: each row is one such multiplied-out "
           "equation, with the cross terms carrying the negative signs the "
           "rearrangement produces."),
     ("p", "The resulting system is then solved by standard linear algebra. "
           "The reason four corners are exactly right is that eight "
           "unknowns need eight equations: three corners would leave the "
           "system underdetermined and five would overdetermine it, "
           "requiring a least-squares fit rather than a solution."),
   ]),

 "AugmentPerspective": dict(
   inputs=[("img", "a line crop")],
   output="the crop with a random perspective distortion applied.",
   theory=[
     ("h", "What it simulates, and why it matters for this project"),
     ("p", "A page photographed from an angle rather than square on. This "
           "augmentation matters more here than it would for a database-"
           "only system, because this project's own pages are phone "
           "photographs: a page on a desk photographed by hand always "
           "carries some perspective. A network trained only on flat scans "
           "would be seeing a distortion it had never met."),
     ("h", "Why it is applied to the line rather than the page"),
     ("p", "Strictly the perspective applies to the whole page, and a line "
           "crop is a narrow strip of it, so the distortion within one "
           "strip is a small part of the page's. Applying a crop-sized "
           "perspective is therefore an approximation, and a slightly "
           "stronger one than reality. That is acceptable for augmentation, "
           "whose purpose is to cover the variation rather than to model it "
           "exactly, and erring toward stronger distortion is the safer "
           "direction."),
   ]),

 "PerspectiveJitterPoints": dict(
   inputs=[("width", "the crop width"), ("height", "its height"),
           ("DistortionScale", "how far the corners may move, as a "
                               "fraction")],
   output="a pair of corner lists: the originals and the jittered "
          "destinations.",
   theory=[
     ("h", "What it does"),
     ("p", "Moves each of the four corners by a random amount within a "
           "fraction of the crop's size, producing the correspondences the "
           "coefficient solver needs."),
     ("h", "Why the corners are moved independently"),
     ("p", "Independent corner displacements generate a general perspective "
           "distortion rather than a parameterised one. The alternative, "
           "choosing a viewing angle and deriving the corners from it, "
           "would be more physically faithful and would require a camera "
           "model, a focal length and an assumed distance, none of which "
           "are known. Moving the corners directly covers the same space of "
           "distortions with no model at all, which is what augmentation "
           "needs."),
     ("note", "The displacement is a fraction of the crop's dimensions "
              "rather than a number of pixels, so the strength of the "
              "distortion is the same for a crop of any size. This is the "
              "same scale-relative reasoning that governs the thresholds "
              "throughout the project."),
   ]),

 "AugmentElastic": dict(
   inputs=[("img", "a line crop")],
   output="the crop with a smooth random warp applied.",
   theory=[
     ("h", "What it simulates"),
     ("p", "**Elastic distortion** \\cite{simard2003} displaces every pixel "
           "by a small smoothly varying random amount, so the image is "
           "stretched here and compressed there. For handwriting it "
           "imitates the natural variation of a hand: the same person "
           "writing the same word twice produces shapes that differ in just "
           "this way, locally stretched and squeezed rather than globally "
           "transformed."),
     ("h", "How the smoothness is obtained, and why it is essential"),
     ("p", "Two random displacement fields are generated, one per axis, and "
           "each is blurred with a Gaussian before being applied. The "
           "blurring is what makes the method work. An unblurred random "
           "displacement moves each pixel independently, which is simply "
           "noise: it destroys the strokes rather than deforming them. "
           "After blurring, nearby pixels move together, so a stroke stays "
           "a connected stroke and is merely bent."),
     ("p", "The blur width therefore controls the character of the "
           "distortion. A wide blur gives a nearly uniform displacement, "
           "which is just a translation; a narrow one approaches noise. The "
           "useful range is a width comparable to the features being "
           "deformed, which for handwriting is around a letter's size. The "
           "amplitude scales the displacement after blurring, since blurring "
           "reduces the magnitude as well as smoothing it."),
     ("note", "The implementation calls the Elastic function in segment.py, "
              "which is the same operation the glyph pipeline uses. Sharing "
              "it means the augmentation applied during training and any "
              "distortion applied elsewhere are identical, which matters if "
              "the two are ever compared."),
   ]),

 "AugmentLineImage": dict(
   inputs=[("pilImg", "a line crop")],
   output="the crop with a randomly chosen augmentation applied, or "
          "unchanged.",
   theory=[
     ("h", "What it does"),
     ("p", "Picks one augmentation at random from the available list and "
           "applies it, or sometimes leaves the crop alone."),
     ("h", "Why one at a time rather than all of them"),
     ("p", "Composing several distortions compounds their severity, and the "
           "result quickly stops resembling handwriting: a crop sheared, "
           "rotated, warped in perspective and elastically distorted is "
           "unreadable, and training on unreadable examples teaches "
           "nothing. Applying one at a time keeps every augmented example "
           "plausible while still covering each kind of variation across "
           "the training set as a whole."),
     ("h", "Why unaugmented crops are kept too"),
     ("p", "Some proportion of crops pass through untouched, so the network "
           "sees the real distribution as well as the augmented one. If "
           "every example were distorted, the network would be trained "
           "exclusively on distorted data and would be slightly mismatched "
           "to the undistorted crops it meets at inference, which is the "
           "domain-shift problem applied to augmentation itself."),
   ]),

 "AUGMENTATIONS": dict(
   output="the list of augmentation functions to choose from.",
   theory=[
     ("h", "Why the augmentations are a list"),
     ("p", "Collecting them in a list means the chooser needs no knowledge "
           "of what they are: it picks an entry and calls it. Adding or "
           "removing one is an edit to the list alone, and more "
           "importantly, running an experiment on which augmentations help "
           "means editing one line, which is how such questions are "
           "answered in practice."),
     ("note", "All four take a crop and return a crop, which is what makes "
              "them interchangeable. That uniform signature is the real "
              "content of the design: a function needing extra parameters "
              "could not be in the list, so each augmentation draws its own "
              "random amounts internally rather than being given them."),
   ]),

 "LineImageCache": dict(
   output="an object holding extracted crops on disk and serving them by "
          "index.",
   theory=[
     ("h", "What it is for"),
     ("p", "Stores every extracted crop in a single file with an index of "
           "where each one starts, so a crop can be read by number without "
           "reading the rest. It is what makes the extraction cost payable "
           "once rather than once per epoch."),
     ("h", "Why one file with an index"),
     ("p", "Tens of thousands of separate image files would be slow, "
           "because each open is a separate filesystem operation, and "
           "wasteful, because a filesystem allocates a whole block per "
           "file. One file read at an offset is a single operation, and the "
           "index is small enough to hold in memory."),
     ("h", "Why the whole cache is not simply loaded into memory"),
     ("p", "The extracted crops for the full database run to several "
           "gigabytes, more than is comfortable to hold while also holding "
           "the activations a training step needs. Reading each crop as the "
           "batch needs it keeps the memory use to the batch, and the "
           "operating system's own file cache keeps the recently used parts "
           "in memory anyway, which recovers most of the speed without the "
           "program managing it."),
   ]),

 "IAMLineDatasetRaw": dict(
   output="an object presenting the database's pages as a sequence of "
          "labelled line crops.",
   theory=[
     ("h", "What it is for"),
     ("p", "Wraps the cache and the labels so that training sees a simple "
           "sequence: ask for item n and get a crop and its text. Where the "
           "pages came from, how they were cut and how they are stored are "
           "all hidden behind that."),
     ("h", "Why this separation matters"),
     ("p", "The training loop should not know whether its data came from "
           "the public database, from this project's own photographs, or "
           "from a third source. Presenting every source through the same "
           "interface is what allows TrainTextModel to accept a source "
           "argument and otherwise be unchanged, and what allows the two "
           "sources to be combined by AddPersonalSamples without the loop "
           "being aware of it."),
     ("note", "The dataset is the public IAM database of handwritten "
              "English \\cite{marti2002}, whose value for this project is "
              "its transcripts: without a known text per line there is "
              "nothing to train against and no way to align characters."),
   ]),

 "TrainEvalView": dict(
   output="a view of a dataset that applies augmentation or does not.",
   theory=[
     ("h", "Why the same data needs two views"),
     ("p", "Augmentation must be applied when training and must not be "
           "applied when validating. The reason is what validation is for: "
           "it estimates how the network will perform on real data, so it "
           "must use real data. Measuring on augmented crops would measure "
           "performance on distorted handwriting, which nobody is going to "
           "present."),
     ("p", "A **view** wraps the same underlying dataset and changes how "
           "items are served, rather than copying the data. So the training "
           "and validation views share one cache and one set of labels, and "
           "there is no possibility of them drifting apart, which two copies "
           "would allow."),
     ("note", "This also makes the training and validation split a property "
              "of which indices each view holds rather than of the data "
              "itself. A line can therefore appear in only one of them, "
              "which is the whole requirement of a held-out set, and it is "
              "enforced by construction rather than by care."),
   ]),

 "TextCollateFn": dict(
   inputs=[("batch", "a list of crop and label pairs")],
   output="the batch assembled into arrays.",
   theory=[
     ("h", "What collation is"),
     ("p", "A dataset serves one item at a time; a network wants a batch as "
           "one array. **Collation** is the step between: taking a list of "
           "items and stacking them, which for images means padding them to "
           "a common size."),
     ("h", "Why the labels stay as a list"),
     ("p", "The crops are stacked into one array, but the labels cannot be: "
           "they are sequences of different lengths, and CTC requires each "
           "one's true length, since padding a label would add characters "
           "the image does not contain. So the labels are kept as a list of "
           "varying-length sequences, which is exactly what CtcLoss expects "
           "and why that function loops over the batch."),
   ]),

 "AuthorLabeledView": dict(
   output="a view of the same crops labelled by who wrote them.",
   theory=[
     ("h", "What it is for"),
     ("p", "Presents the same cached crops with a different label: the "
           "writer's identity instead of the text. The namer is trained "
           "from this view and the reader from the text view, over one "
           "shared cache."),
     ("h", "Why one cache can serve both"),
     ("p", "Because the input is the same object and only the question "
           "differs. Extracting the crops twice would double the slowest "
           "part of the pipeline and, worse, would allow the two "
           "extractions to differ, so that a finding about one network "
           "might not transfer to the other."),
     ("p", "There is also a practical gain: a page whose crop count did not "
           "match its transcript is unusable for the reader but perfectly "
           "usable here, since the writer's identity is a property of the "
           "page and does not depend on the lines being correctly divided. "
           "So this view can draw on pages the text view must reject."),
   ]),

 "AuthorCollateFn": dict(
   inputs=[("batch", "a list of crop and writer-index pairs")],
   output="the batch assembled into arrays.",
   theory=[
     ("h", "How it differs from the text version"),
     ("p", "Simpler, because the labels are single integers rather than "
           "sequences, so they stack into one array like the crops. There "
           "is no variable length to preserve and no padding question for "
           "the labels."),
     ("note", "That simplicity is a direct consequence of the task: "
              "classification has one answer per example, where "
              "transcription has a sequence whose length varies. It is the "
              "same distinction that makes the namer need no CTC and no "
              "recurrence."),
   ]),

 "AddPersonalSamples": dict(
   inputs=[("base", "the database-derived dataset")],
   output="the dataset with this project's own photographed pages added.",
   theory=[
     ("h", "What it is for"),
     ("p", "Mixes the five photographed writers' pages into the database "
           "training set. The result is the joint checkpoint, as opposed "
           "to the scratch one trained on the database alone."),
     ("h", "Why mixing rather than training separately"),
     ("p", "Two things are wanted at once. The database supplies volume: "
           "thousands of writers, which is what teaches the network the "
           "general shape of handwriting. This project's pages supply the "
           "specific domain: phone photographs of these particular hands, "
           "which is what the system will actually be given."),
     ("p", "Training on the database alone leaves a domain gap, since the "
           "network has never seen a photograph. Training on the personal "
           "pages alone gives far too few examples, a few hundred lines "
           "against tens of thousands, and the network would overfit them "
           "immediately. Mixing gets the volume and the domain together, "
           "which is why the joint checkpoint measures so much better on "
           "this project's crops than the scratch one: 96 to 99 percent "
           "character accuracy against 22 to 26."),
     ("note", "That difference is the single largest measured effect in the "
              "project, and it is a statement about which data a network "
              "was trained on rather than about any architectural choice. "
              "It is also why the checkpoint actually loaded at inference "
              "matters so much, and why the candidate ordering in "
              "recognise.py deserves attention."),
   ]),

 "TekliaLineDataset": dict(
   output="a dataset presenting an alternative public corpus of line "
          "images.",
   theory=[
     ("h", "What it is for"),
     ("p", "A second public source of labelled line images, already "
           "segmented into lines, selectable by the source argument of the "
           "training function."),
     ("h", "Why a second source is useful"),
     ("p", "It arrives already cut into lines, so it exercises the training "
           "path without the page-segmentation path, which separates two "
           "possible causes of poor results: a network that trains well on "
           "pre-cut lines and badly on self-cut ones has a segmentation "
           "problem rather than a training one. It is also simply more "
           "data, from a different collection with different scanning "
           "conventions, which broadens what the network has seen."),
   ]),

 "MAX_SAFE_LABEL_CHARS": dict(
   output="the longest label that can safely be trained on, derived from "
          "the input width.",
   theory=[
     ("h", "The constraint it expresses"),
     ("p", "CTC needs at least one column per character, and in practice "
           "more, since a character usually occupies several columns and "
           "blanks need somewhere to go. The network produces one column "
           "per four input pixels, after two halvings, so the number of "
           "columns is fixed by the input width and the longest trainable "
           "label follows from it."),
     ("eq", r"\mathrm{maxChars} \;=\; \left\lfloor "
            r"0.75 \cdot \frac{\mathrm{INPUT\_WIDTH}}{4} \right\rfloor",
      [(r"\mathrm{maxChars}", "the longest label accepted"),
       (r"\mathrm{INPUT\_WIDTH}", "the network's input width, 640"),
       ("4", "the downsampling factor from the two poolings"),
       ("0.75", "a safety margin, leaving a quarter of the columns spare "
                "for blanks and for characters occupying more than one")],
      None),
     ("h", "What happens without the limit"),
     ("p", "A label longer than the available columns has no valid "
           "alignment at all, so its total probability is zero and the loss "
           "is infinite. Infinity propagates through the gradient and "
           "destroys every weight in the network on that single step. So "
           "this is not a quality filter but a correctness one: a single "
           "over-long label in a training set ruins the run, and the symptom "
           "is a loss that becomes not-a-number and never recovers."),
   ]),

 "HIDDEN_SIZE": dict(
   output="256, the LSTM width per direction.",
   theory=[
     ("h", "What it sets"),
     ("p", "How many numbers the recurrent layer carries from one column to "
           "the next, per direction. With two directions concatenated, each "
           "position is described by 512 numbers after the first layer."),
     ("h", "The trade-off"),
     ("p", "A wider state can remember more about the line, which the task "
           "needs: the reader must carry enough about preceding letters to "
           "resolve an ambiguous one. But the recurrent weight matrices are "
           "four times the hidden size by the hidden size, so the parameter "
           "count grows with the square of this number, as does the work "
           "per step, and the cached activations grow with it linearly. At "
           "256 the two layers hold the large majority of the network's "
           "parameters."),
     ("note", "256 is the value the architecture this network follows uses "
              "\\cite{shi2017}. Taking a published figure rather than "
              "searching for one is reasonable for a parameter whose effect "
              "is gradual and whose evaluation would cost a full training "
              "run each time."),
   ]),

 "WEIGHTS_DIR": dict(
   output="the directory checkpoints are written to and read from.",
   theory=[
     ("h", "What it is for"),
     ("p", "One named location for trained weights, taken from the central "
           "paths module, so that the training code and the inference code "
           "agree on where checkpoints live without either hard-coding a "
           "path."),
   ]),

 "DEFAULT_TEXT_CACHE_DIR": dict(
   output="where the reader's extracted line crops are cached.",
   theory=[
     ("h", "Why the two networks have separate caches"),
     ("p", "The reader and the namer are trained from different page sets: "
           "the reader from as much labelled text as possible, the namer "
           "from the specific enrolled writers. The extractions differ in "
           "which pages they cover and in whether a page lacking a usable "
           "transcript is kept, so they are stored separately rather than "
           "one being a subset of the other."),
   ]),

 "DEFAULT_AUTHOR_CACHE_DIR": dict(
   output="where the namer's extracted crops are cached.",
   theory=[
     ("h", "What it holds"),
     ("p", "Crops from the ten writers the namer is trained to distinguish, "
           "labelled by writer rather than by text. The directory name "
           "records the writer count, which is useful because the namer's "
           "output layer size is fixed by it: a cache built for a different "
           "number of writers is incompatible with a checkpoint trained on "
           "this one, and a differently named directory makes that visible."),
   ]),

 "DEFAULT_AUTHOR_DATA_DIR": dict(
   output="where the source pages for the namer live.",
   theory=[
     ("h", "What it points at"),
     ("p", "The page images and transcripts for the enrolled writers, built "
           "from the central paths module. The distinction from the cache "
           "directory is worth keeping clear: this holds the original "
           "scanned or photographed pages, which are the input, and the "
           "cache holds what was extracted from them, which can be "
           "regenerated and is excluded from version control for that "
           "reason."),
   ]),

 "DEFAULT_TEXT_DATA_DIR": dict(
   output="where the source pages for the reader live, the same directory "
          "as the namer's.",
   theory=[
     ("h", "Why they are the same"),
     ("p", "Both networks are trained from the same collection of pages, "
           "differing in what is taken from them rather than in where they "
           "come from. Defining one as the other rather than repeating the "
           "path means the two cannot drift apart, and it records that the "
           "sharing is intentional rather than coincidental."),
   ]),

 "InitConv": dict(
   inputs=[("rng", "the seeded random generator"),
           ("outCh", "the number of filters"),
           ("inCh", "the input channel count"),
           ("kh", "the filter height"), ("kw", "its width")],
   output="a pair: the randomly initialised filters and a zero bias.",
   theory=[
     ("h", "Why initialisation matters at all"),
     ("p", "The starting weights are random, but not arbitrarily so. If "
           "they are too small, the signal shrinks as it passes through "
           "each layer and arrives at the output as nearly nothing, so "
           "there is no gradient to learn from. If too large, it grows and "
           "saturates. With twelve convolutional layers either effect "
           "compounds twelve times, so the scale of the initial weights "
           "determines whether the network trains at all."),
     ("h", "He initialisation"),
     ("eq", r"w \;\sim\; \mathcal{N}\!\left(0,\; "
            r"\sqrt{\frac{2}{\mathrm{fanIn}}}\,\right), \qquad "
            r"\mathrm{fanIn} \;=\; \mathrm{inCh} \times k_h \times k_w",
      [("w", "each weight, drawn independently"),
       (r"\mathcal{N}(0,s)", "a normal distribution with mean 0 and "
                             "standard deviation s"),
       (r"\mathrm{fanIn}", "how many input values each output is computed "
                           "from"),
       ("2", "the factor that accounts for ReLU, explained below")],
      "he2015"),
     ("p", "The reasoning runs as follows. An output is a sum of fanIn "
           "products of a weight and an input. For independent terms, "
           "variances add, so the output's variance is fanIn times the "
           "product of the weight and input variances. Keeping the output's "
           "variance equal to the input's therefore requires the weight "
           "variance to be one over fanIn. That is the Glorot argument "
           "\\cite{glorot2010}."),
     ("p", "The factor of 2 is the correction for ReLU \\cite{he2015}. ReLU "
           "sets every negative value to zero, discarding about half the "
           "signal's variance, so without compensation the variance would "
           "halve at every layer and after twelve layers be reduced by a "
           "factor of four thousand. Doubling the weight variance exactly "
           "offsets it. The difference between the two initialisations is "
           "only that factor, and for a network this deep it is the "
           "difference between training and not."),
     ("h", "Why the bias starts at zero"),
     ("p", "There is nothing to break symmetry in a bias, because the "
           "weights are already random, so each unit already computes "
           "something different. Zero is the neutral starting point, and it "
           "is also harmless here because the batch normalisation that "
           "follows removes any constant offset anyway, as noted under "
           "ConvBlockForward."),
   ]),

 "InitBatchNorm": dict(
   inputs=[("ch", "the channel count")],
   output="four arrays: the scale at one, the shift at zero, and the "
          "running statistics at zero and one.",
   theory=[
     ("h", "Why the scale starts at one and the shift at zero"),
     ("p", "Those values make the learned part of the layer the identity: "
           "the output is the standardised input unchanged. So the layer "
           "begins by doing exactly the normalisation it exists for and "
           "nothing else, and any departure from that is something training "
           "chose. Starting the scale at a random value would add a "
           "per-channel distortion the network would have to learn to undo "
           "before making progress."),
     ("h", "Why the running statistics start at zero and one"),
     ("p", "A mean of zero and a variance of one describe already-"
           "standardised data, so a network evaluated before any training "
           "behaves as though normalisation had been applied correctly. "
           "They are also the values the exponential average converges away "
           "from fastest: being the neutral assumption, any real data moves "
           "them in a meaningful direction immediately."),
     ("note", "These two pairs are treated differently everywhere else in "
              "the file. The scale and shift are parameters with gradients "
              "and are updated by the optimiser; the running statistics have "
              "no gradients and are updated by observation. TrainableParams "
              "separates them by name, which is why the naming convention "
              "for them is load-bearing."),
   ]),

 "InitLinear": dict(
   inputs=[("rng", "the random generator"), ("outF", "the output size"),
           ("inF", "the input size")],
   output="a pair: the randomly initialised weight matrix and bias.",
   theory=[
     ("h", "The distribution used"),
     ("eq", r"w,\, b \;\sim\; \mathrm{Uniform}\!\left(-\frac{1}"
            r"{\sqrt{\mathrm{inF}}},\; +\frac{1}{\sqrt{\mathrm{inF}}}"
            r"\right)",
      [("w,\\,b", "each weight and each bias, drawn independently"),
       (r"\mathrm{Uniform}(a,b)", "a uniform distribution between the two "
                                  "bounds"),
       (r"\mathrm{inF}", "the number of inputs to each output")],
      None),
     ("p", "The same one-over-fan-in scaling as the convolution, expressed "
           "as a uniform range rather than a normal distribution and without "
           "the ReLU factor of two. The reasoning for the scale is "
           "identical; the differences are matters of convention, and they "
           "are followed here because the reference framework's linear layer "
           "uses exactly this, which keeps checkpoints interchangeable."),
     ("h", "Why the bias is random here but zero for a convolution"),
     ("p", "Because no normalisation follows a linear layer. A convolution's "
           "bias is removed by the batch normalisation after it, so its "
           "initial value is immaterial; a linear layer's bias survives into "
           "the output and a small random value is as good a starting point "
           "as zero."),
   ]),

 "InitLstmDirection": dict(
   inputs=[("rng", "the random generator"),
           ("inputSize", "the input width"),
           ("hiddenSize", "the hidden width")],
   output="four arrays: both weight matrices and both biases for one "
          "direction.",
   theory=[
     ("h", "The distribution"),
     ("p", "Uniform between plus and minus one over the square root of the "
           "hidden size, for every weight and both biases. The code comment "
           "records that this matches the reference framework's own "
           "initialisation, which is the reason for the specific form."),
     ("h", "Why the hidden size rather than the input size sets the scale"),
     ("p", "The input weights are scaled by the hidden size too, although "
           "their fan-in is the input size. That is the framework's "
           "convention rather than what the variance argument would give. It "
           "is defensible because the two are comparable in this network, "
           "the input to the second layer being twice the hidden size, and "
           "because the gates' sigmoids are forgiving: they saturate "
           "gracefully rather than destroying the signal, so the "
           "initialisation scale is less critical than for a deep "
           "convolutional stack."),
     ("note", "All four gates are initialised identically here. A common "
              "refinement is to initialise the forget gate's bias to a "
              "positive value, so the gate starts near 1 and the cell "
              "retains information from the first step, which helps a "
              "network learn long dependencies sooner. It is not done here, "
              "which matches the reference framework's default."),
   ]),

 "InitConvBlock": dict(
   inputs=[("rng", "the random generator"),
           ("params", "the parameter dictionary, filled in place"),
           ("prefix", "the block's name prefix"),
           ("inCh", "the input channels"), ("outCh", "the output channels"),
           ("nLayers", "how many groups")],
   output="nothing; the parameters are added to the dictionary.",
   theory=[
     ("h", "What it does"),
     ("p", "Creates one block's worth of parameters under the names the "
           "forward and backward passes expect, with the channel count "
           "switching from the block's input to its output after the first "
           "group."),
     ("h", "Why the naming convention is load-bearing"),
     ("p", "These names are a contract between four separate pieces of "
           "code: this initialiser, the forward pass, the backward pass and "
           "the checkpoint format. They must agree exactly, and a "
           "disagreement fails with a missing key rather than silently "
           "training the wrong array, which is the reason for building them "
           "from the same arithmetic in every one of those places rather "
           "than listing them."),
     ("note", "The first group takes the block's input channel count and "
              "every later group takes its output count, which is why the "
              "channel variable is reassigned inside the loop. Getting that "
              "wrong produces a shape mismatch at the first forward pass "
              "rather than a subtle fault."),
   ]),

 "InitBackboneParams": dict(
   inputs=[("rng", "the random generator"),
           ("params", "the parameter dictionary, filled in place")],
   output="nothing; the backbone's parameters are added.",
   theory=[
     ("h", "The channel progression"),
     ("p", "One channel in, then 32, then 64, then 128, with two, four and "
           "six groups in the three stages. The channel count doubles at "
           "each stage while the spatial size halves, which is the standard "
           "arrangement in a convolutional backbone."),
     ("h", "Why that arrangement"),
     ("p", "Two reasons, one about capacity and one about cost. Deeper "
           "layers represent larger and more varied structures, strokes "
           "giving way to junctions and then to letter parts, and there are "
           "more distinguishable large structures than small ones, so more "
           "channels are needed to hold them. And since each pooling "
           "quarters the spatial size, doubling the channels leaves the "
           "work per layer roughly halved rather than growing, so the "
           "deeper stages are affordable despite being wider and having "
           "more groups."),
     ("h", "Why the backbone is shared between the two networks"),
     ("p", "Both the reader and the namer use exactly this backbone, "
           "initialised by this one function. The features useful for "
           "recognising a letter and for recognising a hand are largely the "
           "same features, as argued under AuthorClassifierCNNNumpy, so "
           "there is no reason for the two to differ, and sharing the "
           "initialiser means a change to the architecture applies to both "
           "rather than to one."),
   ]),

 "TrainableParams": dict(
   inputs=[("params", "the full parameter dictionary")],
   output="the subset the optimiser may update.",
   theory=[
     ("h", "What it excludes and why"),
     ("p", "The running mean and variance of every batch normalisation. "
           "They are stored alongside the parameters, because they must be "
           "saved in the checkpoint, but they are not parameters: no "
           "gradient is computed for them, and they are updated by "
           "observing the data rather than by descending a loss."),
     ("h", "What would happen if they were included"),
     ("p", "The optimiser would look up a gradient for them, find none, and "
           "fail with a missing key, which is the benign outcome. Had a "
           "zero gradient been supplied instead, the optimiser would apply "
           "weight decay to them and shrink them steadily toward zero, so "
           "the stored variances would fall and inference would divide by a "
           "smaller and smaller number, amplifying every activation. That "
           "is a silent corruption which would appear only at inference, as "
           "a network whose validation accuracy during training was fine "
           "and whose saved checkpoint performs badly."),
     ("note", "The separation is by name suffix, which works but ties "
              "correctness to a naming convention. A parameter accidentally "
              "named with one of those suffixes would be silently frozen, "
              "and one of these accidentally named otherwise would be "
              "silently decayed."),
   ]),

 "SaveCheckpoint": dict(
   inputs=[("path", "where to write"), ("params", "the parameters"),
           ("extra", "additional values such as the epoch number")],
   output="nothing; the checkpoint file is written.",
   theory=[
     ("h", "What goes into a checkpoint"),
     ("p", "Every parameter, including the running statistics, plus "
           "metadata: which epoch it came from and the best validation loss "
           "seen so far. The metadata is what makes resuming possible."),
     ("h", "Why the metadata must be in the file"),
     ("p", "Resuming a run requires knowing which epoch to continue from "
           "and what the best loss was, so that the scheduler and the "
           "best-checkpoint logic behave as though the run had never "
           "stopped. Keeping those in the checkpoint rather than in a "
           "separate file means they cannot be separated from the weights "
           "they describe, which is the failure a parallel log file "
           "invites: resuming from the wrong epoch number silently "
           "misschedules the learning rate."),
     ("note", "The metadata keys begin with a double underscore, which is "
              "the same convention LoadStateDictAny uses to tell metadata "
              "from weights. The two functions are in different files and "
              "must agree on it, which is worth noting as another implicit "
              "contract."),
   ]),

 "LoadCheckpointMeta": dict(
   inputs=[("path", "a checkpoint file")],
   output="the epoch it was saved at and the best loss recorded, where "
          "present.",
   theory=[
     ("h", "What it is for"),
     ("p", "Reads only the metadata, without the weights, so that a resume "
           "can establish where it is continuing from. Reading the "
           "metadata alone is cheap, where loading several hundred "
           "megabytes of weights to read two numbers is not."),
     ("note", "Both values are optional, since a checkpoint written before "
              "the metadata convention existed has neither. Returning "
              "nothing rather than a default lets the caller distinguish "
              "\"no record\" from \"epoch zero\", and it starts the run at "
              "epoch 1 rather than claiming to resume."),
   ]),

 "LoadParamsInto": dict(
   inputs=[("params", "the parameter dictionary, filled in place"),
           ("checkpointPath", "the checkpoint to read")],
   output="how many arrays were loaded.",
   theory=[
     ("h", "Why it loads into an existing dictionary"),
     ("p", "The parameters are created first by the initialiser, with the "
           "right names and shapes, and the checkpoint's values are then "
           "copied into them. The alternative, building the dictionary from "
           "the checkpoint, would take the network's structure from the "
           "file, so a checkpoint from a different architecture would load "
           "without complaint and fail later with a shape mismatch deep in "
           "a forward pass."),
     ("h", "Why partial loading is allowed, and why the count is returned"),
     ("p", "Names present in both are copied and the rest are left as "
           "initialised. That is what makes a **warm start** possible: "
           "loading the backbone from one checkpoint while leaving the head "
           "freshly initialised, which is how the namer is started from the "
           "reader's learned features."),
     ("p", "But partial loading is also how a wrong checkpoint fails "
           "quietly: a file matching nothing loads nothing and training "
           "proceeds from random weights, which looks like training that is "
           "simply going badly. Returning the count, which the callers "
           "print as a fraction of the total, is what makes that visible. A "
           "warm start reporting a small fraction is a warning that the two "
           "checkpoints do not correspond."),
   ]),

 "AppendLog": dict(
   inputs=[("path", "the log file"), ("line", "the line to add")],
   output="nothing; the line is appended.",
   theory=[
     ("h", "Why a log file as well as printed output"),
     ("p", "Training runs for days, so the console scrollback is lost or "
           "truncated long before the run ends. A log file persists, and "
           "because each epoch writes one line of keyed values, the whole "
           "history of a run can be read or plotted afterwards."),
     ("h", "Why appending rather than rewriting"),
     ("p", "Appending means a resumed run continues the same log rather "
           "than replacing it, so the record spans the interruptions. "
           "Appending is also safe against a crash: each line is complete "
           "when written, so an interrupted run leaves a readable log "
           "ending at the last finished epoch, where a rewritten file could "
           "be truncated mid-write."),
   ]),

 "BackboneForwardTrain": dict(
   inputs=[("x", "the input crops"), ("params", "the parameters")],
   output="a pair: the collapsed feature sequence, and the cache.",
   theory=[
     ("h", "What it is for"),
     ("p", "The shared convolutional backbone with caching: three blocks, "
           "two poolings, and the collapse of the height. Both networks "
           "call it, so the part they have in common is written once."),
     ("h", "Why sharing the implementation matters more than sharing the "
           "weights"),
     ("p", "The two networks do not share weights; each trains its own copy. "
           "What they share is the code, and that is what guarantees the "
           "architectures are identical, which in turn is what makes a warm "
           "start from one to the other possible: the parameter names and "
           "shapes agree because they were produced by the same function."),
   ]),

 "BackboneBackwardTrain": dict(
   inputs=[("dx", "the gradient with respect to the backbone's output"),
           ("cache", "what the forward pass saved")],
   output="the gradient with respect to the input and every backbone weight "
          "gradient.",
   theory=[
     ("h", "What it does"),
     ("p", "Walks the backbone in reverse: uncollapse the height, then the "
           "third block, then the second pooling, the second block, the "
           "first pooling and the first block, collecting weight gradients. "
           "It is the mirror of the forward sequence and contains no "
           "arithmetic beyond the sequencing."),
     ("note", "The input gradient is computed and returned although nothing "
              "uses it, since the crops are data rather than parameters. It "
              "is produced because each layer's backward pass produces it as "
              "a matter of course, and discarding it earlier would mean a "
              "special case in the first convolution. It is also what the "
              "numerical gradient check compares against when verifying the "
              "backbone."),
   ]),

 "EvalBackboneForward": dict(
   inputs=[("params", "the parameters"), ("x", "the input crops")],
   output="the backbone's output, with no cache.",
   theory=[
     ("h", "Why a third version of the backbone exists"),
     ("p", "Validation needs a forward pass with the inference behaviour, "
           "using the stored batch statistics rather than the batch's own, "
           "and with no caching, since no backward pass follows. Running "
           "the training version and discarding the caches would use the "
           "wrong statistics and would allocate the caches anyway."),
     ("h", "Why using the stored statistics for validation is the point"),
     ("p", "Validation estimates how the saved checkpoint will perform, and "
           "the saved checkpoint will be used with the stored statistics. "
           "Validating with batch statistics would measure something "
           "slightly different and, worse, something that depends on how "
           "the validation set happened to be divided into batches, so the "
           "same weights would score differently with a different batch "
           "size."),
   ]),

 "IterateBatches": dict(
   inputs=[("dataset", "the dataset to iterate"),
           ("batchSize", "how many items per batch"),
           ("rng", "the random generator"),
           ("shuffle", "whether to shuffle the order")],
   output="batches of items, one at a time.",
   theory=[
     ("h", "Why the order is shuffled for training"),
     ("p", "The examples arrive grouped by page and therefore by writer. "
           "Without shuffling, consecutive batches would each contain one "
           "writer's hand, so the network would be pulled toward that hand "
           "and then away from it, which both slows convergence and makes "
           "the loss oscillate with the data order rather than with "
           "learning. Shuffling makes each batch an approximately "
           "representative sample, which is the assumption the gradient "
           "estimate rests on."),
     ("h", "Why validation is not shuffled"),
     ("p", "Validation computes a mean over the whole set, and a mean does "
           "not depend on the order. Keeping the order fixed makes the "
           "measurement exactly reproducible between runs, which matters "
           "when comparing two checkpoints: any difference is then "
           "attributable to the weights rather than to the batching."),
     ("note", "Batches are produced one at a time rather than built as a "
              "list, so only the current batch is in memory. For a dataset "
              "of tens of thousands of crops that is the difference between "
              "a few hundred megabytes and several gigabytes."),
   ]),

 "InitTextModelParams": dict(
   inputs=[("seed", "the random seed"),
           ("numClasses", "the output size, the charset plus the blank")],
   output="the reader's complete parameter dictionary.",
   theory=[
     ("h", "What it builds"),
     ("p", "The shared backbone, two bidirectional LSTM layers, and a "
           "linear text head producing one score per class."),
     ("h", "Why the seed is explicit"),
     ("p", "Every random draw comes from a generator created from this "
           "seed, so the same seed gives the same initial weights. That is "
           "what makes a training run reproducible, which matters for "
           "comparing two configurations: without it, a difference in the "
           "result could be a difference in the starting point rather than "
           "in the change being tested."),
     ("h", "Why the class count is one more than the charset"),
     ("p", "The extra class is the CTC blank, which is not a character but "
           "needs an output slot of its own, as described under DecodeCtc. "
           "Deriving the count from the charset rather than stating it means "
           "the two cannot disagree, and a disagreement would be severe: "
           "the network would be trained with every character's index "
           "shifted by one relative to what inference assumes."),
   ]),

 "BuildLstmLayerParams": dict(
   inputs=[("sd", "the parameter dictionary")],
   output="the LSTM parameters gathered into the nested structure the "
          "forward pass expects.",
   theory=[
     ("h", "What it is for"),
     ("p", "The parameters are stored flat, keyed by name, because that is "
           "what a checkpoint is and what the optimiser iterates over. The "
           "recurrent forward pass wants them nested: a list of layers, "
           "each with a forward and a backward direction, each with its four "
           "arrays. This converts one view to the other."),
     ("h", "Why the flat storage is the primary form"),
     ("p", "Because three separate things need it flat: the optimiser, "
           "which iterates over every parameter without knowing the "
           "structure; the checkpoint, which is a flat collection of named "
           "arrays; and the gradient dictionary, which must be keyed the "
           "same way. The nested form is needed by one function, so it is "
           "the derived one."),
     ("note", "The nested structure holds references to the same arrays, "
              "not copies, so an update applied by the optimiser to the flat "
              "dictionary is visible through the nested view. That is "
              "essential and would break silently if any step copied."),
   ]),

 "TextModelForward": dict(
   inputs=[("x", "the input crops"), ("params", "the parameters")],
   output="a pair: the raw per-column class scores, and the cache.",
   theory=[
     ("h", "What it does"),
     ("p", "The reader's whole forward pass with caching: the backbone, the "
           "transpose that turns the image into a sequence, the two "
           "recurrent layers, and the text head."),
     ("h", "Why it returns raw scores rather than log probabilities"),
     ("p", "The inference version ends with a log softmax; this one stops at "
           "the raw scores, because CtcLoss performs the softmax itself. "
           "The reason is the cancellation described in that entry: taking "
           "the softmax and the loss together gives a gradient that is a "
           "simple difference, whereas computing the softmax here would "
           "require differentiating back through it as a separate step, "
           "which is both more work and numerically worse."),
     ("note", "This is why the gradient the loss returns is named for the "
              "raw logits: it is the gradient with respect to the output of "
              "this function, so the backward pass can start here with no "
              "intervening layer."),
   ]),

 "TextModelBackward": dict(
   inputs=[("dRawLogits", "the gradient from the loss"),
           ("cache", "what the forward pass saved")],
   output="the gradient with respect to the input and every weight "
          "gradient.",
   theory=[
     ("h", "What it does"),
     ("p", "Reverses the forward sequence: the text head, the two recurrent "
           "layers, the transpose, and the backbone, collecting weight "
           "gradients keyed by parameter name."),
     ("h", "The transpose, reversed"),
     ("p", "The forward pass moved the width axis to the front to make a "
           "sequence; the backward pass moves it back, so the gradient "
           "leaving for the backbone is an image gradient again. A "
           "transpose is its own inverse given the right axis order, so "
           "nothing is computed, but the axes must be put back in exactly "
           "the order the backbone expects, and an error there produces a "
           "gradient that is correct in magnitude and scrambled in "
           "arrangement."),
   ]),

 "EvalTextForward": dict(
   inputs=[("params", "the parameters"), ("x", "the input crops")],
   output="the log probabilities per column, with no cache.",
   theory=[
     ("h", "What it is for"),
     ("p", "The reader's forward pass in inference mode, for validation: "
           "stored batch statistics, no caching, and the log softmax "
           "applied, since validation decodes the output rather than "
           "passing it to the loss."),
   ]),

 "TextNumpyBatch": dict(
   inputs=[("batch", "a collated batch")],
   output="the images, the encoded labels and the original texts.",
   theory=[
     ("h", "What it does"),
     ("p", "Prepares a batch for the network: the images converted to the "
           "input convention and the labels encoded into class indices. The "
           "original texts are carried through as well so that validation "
           "can compare the decoded output against them."),
     ("note", "The image conversion must match TensorFromResized in "
              "recognise.py exactly, including the inversion and the "
              "standardisation. A difference between the two would produce "
              "a network that trains successfully and reads nothing at "
              "inference, which is the hardest class of fault to find "
              "because both halves appear to work."),
   ]),

 "ValidateTextModel": dict(
   inputs=[("valSet", "the held-out set"), ("params", "the parameters"),
           ("batchSize", "the batch size"), ("rng", "the random generator")],
   output="a pair: the mean validation loss and the character accuracy.",
   theory=[
     ("h", "Why a held-out set is necessary"),
     ("p", "Training loss falls whether or not the network is learning "
           "anything general, because a network with enough capacity can "
           "memorise its training examples. The only way to tell "
           "memorisation from learning is to measure on data the network "
           "has never been trained on. That is the **held-out** or "
           "validation set, and the gap between training and validation "
           "loss is the measure of overfitting."),
     ("h", "Why two numbers are reported"),
     ("p", "The loss and the character accuracy answer different questions, "
           "and both are needed. The loss is what training minimises, so it "
           "is the quantity the scheduler and the best-checkpoint logic must "
           "use, and it is sensitive to small changes in confidence. "
           "Character accuracy is what anyone actually cares about, but it "
           "is coarse: a network can improve its loss substantially with no "
           "change in accuracy, because it has become more confident about "
           "answers it already had right."),
     ("note", "Augmentation is off here, through the evaluation view, and "
              "batch normalisation uses its stored statistics, through the "
              "evaluation forward pass. Both are necessary for the "
              "measurement to describe the checkpoint that will be saved "
              "rather than some variant of it."),
   ]),

 "BuildPagesSplit": dict(
   inputs=[("maxPages", "how many pages to use"),
           ("dataDir", "where the pages are")],
   output="a pair of training and validation datasets.",
   theory=[
     ("h", "Why the split is by page and not by line"),
     ("p", "This is the important decision. Lines from one page are written "
           "by one person, on one sheet, with one pen, in one session. "
           "Splitting by line would put some of a page's lines in training "
           "and others in validation, and the network could then score well "
           "on validation by recognising that particular page's "
           "characteristics rather than by reading. The validation figure "
           "would be optimistic, and the amount by which it was optimistic "
           "would be unknowable."),
     ("p", "Splitting by page makes the validation lines come from sheets "
           "the network has never seen, which is the situation it will "
           "actually face. This is **leakage** avoidance, and it is the "
           "commonest way a held-out measurement is quietly invalidated: the "
           "split has to be made at the level at which examples are "
           "genuinely correlated, which here is the page."),
     ("note", "The page limit exists to allow short runs during "
              "development. It takes the first n pages rather than a random "
              "sample, which means a short run always uses the same pages, "
              "so two such runs are comparable with each other even though "
              "neither represents the whole database."),
   ]),

 "BuildTekliaSplit": dict(
   inputs=[("dataDir", "where the corpus is")],
   output="a pair of training and validation datasets from the alternative "
          "corpus.",
   theory=[
     ("h", "What it is for"),
     ("p", "The same split for the second corpus, which arrives already cut "
           "into lines. Having both behind the same interface is what lets "
           "the training function take a source argument and otherwise be "
           "identical, so a change to the training procedure applies to both "
           "without being made twice."),
   ]),

 "TrainTextModel": dict(
   inputs=[("numEpochs", "how many passes over the data"),
           ("batchSize", "examples per step"),
           ("lr", "the initial learning rate"),
           ("seed", "the random seed"),
           ("maxPages", "how many pages to use"),
           ("dataDir", "where the data is"),
           ("source", "which corpus, default pages"),
           ("resume", "whether to continue a previous run"),
           ("warmStart", "a checkpoint to initialise from")],
   output="the best validation loss reached.",
   theory=[
     ("h", "The training loop"),
     ("items", [
       "**Build the split** and initialise the parameters.",
       "**Create the optimiser and scheduler:** RMSprop with a small weight "
       "decay, and the plateau scheduler watching validation loss.",
       "**Optionally warm start or resume.**",
       "**For each epoch:** iterate shuffled batches, and for each run the "
       "forward pass, compute the CTC loss and its gradient, run the "
       "backward pass, select the trainable gradients and take an optimiser "
       "step.",
       "**After each epoch:** validate, step the scheduler, save the "
       "checkpoint and append to the log.",
     ]),
     ("h", "Why two checkpoints are kept"),
     ("p", "A best checkpoint, written only when validation loss improves, "
           "and a rolling checkpoint written every epoch. They serve "
           "different purposes and neither substitutes for the other. The "
           "best one is the model to use, since later epochs may overfit and "
           "be worse. The rolling one is for resuming, since resuming must "
           "continue from where the run actually stopped, not from the best "
           "epoch, or the epochs between would be repeated."),
     ("h", "Why the best is chosen on loss rather than accuracy"),
     ("p", "Loss is the finer measure. Character accuracy on a validation "
           "set of a few thousand lines moves in visible steps, so several "
           "epochs can tie, and choosing between them would be arbitrary. "
           "Loss distinguishes them, because it responds to confidence as "
           "well as to correctness."),
     ("h", "What warm starting is for"),
     ("p", "Initialising from an existing checkpoint rather than from "
           "random weights. It is how the joint checkpoint is produced: "
           "start from the database-trained weights and continue on the "
           "mixed data, so the network keeps what it learned from the large "
           "corpus while adapting to the photographs. Training that from "
           "scratch would need the whole corpus again and would reach a "
           "similar place far more slowly."),
     ("note", "The progress line is printed with a carriage return and no "
              "newline, so it overwrites itself rather than filling the "
              "console with thousands of lines. The per-epoch summary is "
              "printed normally and also written to the log, so the "
              "transient detail stays transient and the durable record is "
              "kept."),
   ]),

 "InitAuthorModelParams": dict(
   inputs=[("seed", "the random seed"),
           ("numAuthors", "how many writers to distinguish")],
   output="the namer's complete parameter dictionary.",
   theory=[
     ("h", "What it builds"),
     ("p", "The same shared backbone, then a two-layer head: a linear "
           "layer, a ReLU, a dropout, and a second linear layer producing "
           "one score per writer. No recurrent layers, for the reasons "
           "given under AuthorClassifierCNNNumpy."),
     ("h", "Why the writer count is a parameter"),
     ("p", "It sets the final layer's output size, so a checkpoint is tied "
           "to the number of writers it was trained on. Adding a writer "
           "means a new final layer and retraining it, and a checkpoint from "
           "a different count cannot be loaded. That is also why the writer "
           "names are stored inside the checkpoint, as described under "
           "LoadStateDictAny: the count alone is not enough, since the same "
           "ten writers in a different order would load without complaint "
           "and attribute every page to the wrong person."),
   ]),

 "AuthorModelForward": dict(
   inputs=[("x", "the input crops"), ("params", "the parameters"),
           ("training", "whether dropout should be active"),
           ("rng", "the random generator for dropout")],
   output="a pair: the raw per-writer scores, and the cache.",
   theory=[
     ("h", "What it does"),
     ("p", "The backbone, the average pooling over position, and the "
           "two-layer head with dropout between the layers."),
     ("h", "Why the training flag is an argument"),
     ("p", "Because dropout must be active during training and inactive "
           "during validation, and the same function serves both. Passing "
           "it explicitly rather than holding it as state makes it "
           "impossible to validate with dropout accidentally left on, which "
           "would add noise to the measurement, or to train with it off, "
           "which would silently remove the regularisation."),
     ("note", "The random generator is passed in rather than created here, "
              "so that the dropout masks come from the run's own seeded "
              "generator and a training run is reproducible. A generator "
              "created inside the function would reseed itself on every "
              "call and the masks would be unrepeatable."),
   ]),

 "AuthorModelBackward": dict(
   inputs=[("dLogits", "the gradient from the loss"),
           ("cache", "what the forward pass saved")],
   output="the gradient with respect to the input and every weight "
          "gradient.",
   theory=[
     ("h", "What it does"),
     ("p", "Reverses the namer's forward sequence: the second linear layer, "
           "the dropout, the ReLU, the first linear layer, the average "
           "pooling, and the backbone. The average pooling's backward pass "
           "spreads the gradient evenly back over every position, which is "
           "why the namer's backbone receives gradient everywhere rather "
           "than only at the winning rows as the reader's does."),
   ]),

 "EvalAuthorForward": dict(
   inputs=[("params", "the parameters"), ("x", "the input crops")],
   output="the raw per-writer scores, with no cache and no dropout.",
   theory=[
     ("h", "What it is for"),
     ("p", "The namer's inference-mode forward pass for validation: stored "
           "batch statistics, dropout off, no caching. The same three "
           "differences as the reader's evaluation pass, and for the same "
           "reasons."),
   ]),

 "AuthorNumpyBatch": dict(
   inputs=[("batch", "a collated batch")],
   output="the images and the writer indices.",
   theory=[
     ("h", "What it does"),
     ("p", "Prepares a batch for the namer. Simpler than the text version, "
           "because the labels are single integers needing no encoding, "
           "though the image conversion is identical and must stay so."),
   ]),

 "ValidateAuthorModel": dict(
   inputs=[("valSet", "the held-out set"), ("params", "the parameters"),
           ("batchSize", "the batch size"), ("rng", "the random generator")],
   output="a pair: the mean validation loss and the accuracy.",
   theory=[
     ("h", "What it measures"),
     ("p", "The cross-entropy loss and the fraction of crops attributed to "
           "the right writer, over the held-out set with dropout off and "
           "stored statistics in use."),
     ("h", "Why accuracy means more here than for the reader"),
     ("p", "For a classifier, accuracy is the quantity of interest directly: "
           "the crop is attributed correctly or it is not. For the reader, "
           "character accuracy is an aggregate over a sequence and a line "
           "can be mostly right, so the measure is coarser relative to what "
           "matters. The namer's accuracy is therefore the headline figure "
           "and the loss is the finer measure used for choosing the "
           "checkpoint."),
     ("note", "The split for the namer must also be by page, for the "
              "leakage reason given under BuildPagesSplit, and it matters "
              "more here. The namer's task is to recognise a writer, and "
              "lines from the same sheet share far more than handwriting: "
              "the same paper, pen, lighting and scan. A line-level split "
              "would let it score nearly perfectly by recognising the "
              "sheet."),
   ]),

 "TrainAuthorModel": dict(
   inputs=[("numEpochs", "how many passes"), ("batchSize", "examples per "
                                              "step"),
           ("lr", "the initial learning rate"), ("seed", "the random seed"),
           ("resume", "whether to continue a previous run"),
           ("warmStart", "a checkpoint to initialise from")],
   output="the best validation loss reached.",
   theory=[
     ("h", "What differs from the reader's loop"),
     ("p", "The same structure: batches, forward, loss, backward, step, "
           "validate, save, log. The differences are the loss, which is "
           "cross-entropy rather than CTC, the dropout, which must be "
           "enabled for training and disabled for validation, and the data, "
           "which is labelled by writer."),
     ("h", "Why warm starting matters more here"),
     ("p", "The namer has far less data: ten writers' pages against a "
           "database of thousands. Training its backbone from scratch on "
           "that would overfit badly, because a deep convolutional stack "
           "has ample capacity to memorise a few thousand crops. Starting "
           "from the reader's backbone gives it features already learned "
           "from a large corpus, so the small dataset is spent on learning "
           "to tell writers apart rather than on learning what a stroke is."),
     ("p", "This is **transfer learning**, and the reason it works here is "
           "the argument made earlier: the features that distinguish "
           "letters and the features that distinguish hands are largely the "
           "same features. The reader learned them from data the namer "
           "could never have had."),
     ("note", "The two networks are trained separately rather than together "
              "with a shared backbone and two heads, which would be the "
              "obvious alternative. Separate training keeps each network's "
              "accuracy independent of the other's, which matters for this "
              "project because both are used as judges of the synthesised "
              "output: a shared backbone would make the two scores "
              "correlated, and the harmonic mean that combines them assumes "
              "they are measuring different things."),
   ]),
}
