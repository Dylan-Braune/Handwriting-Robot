r"""Theory for segment.py: the image toolkit and the page segmenter.

This chapter is long, so it is written across several files.  Each part adds
its entries to FUNCS at the end of this module.
"""

INTRO = [
 ("p", "This chapter covers two things that live in one file: the image "
       "toolkit used by the whole pipeline, and the page segmenter that is "
       "its heaviest user. They are together because the toolkit has no "
       "purpose on its own, and because other modules then import the few "
       "functions they need from here rather than from a separate module "
       "that would exist only to hold them."),

 ("h", "The problem this file solves"),
 ("p", "A photograph of a handwritten page arrives as a single large grid of "
       "brightness values. Everything downstream, both networks and the "
       "glyph extractor, expects one tidy image per line of writing: "
       "straightened, with the printed ruling removed, cropped to the "
       "writing, and containing that line and no part of its neighbours. "
       "Getting from one to the other is this file's job."),

 ("p", "That is harder than it sounds for four reasons, and the structure of "
       "the file follows from them."),
 ("items", [
   "The photograph is not evenly lit. One side is brighter than the other, "
   "there may be a shadow, and the paper is not pure white. So there is no "
   "single brightness that separates ink from paper everywhere on the page.",
   "The page is never perfectly square to the camera. Even a degree of tilt "
   "means a horizontal cut through the image crosses several lines of "
   "writing.",
   "The paper is usually ruled, and the printed rules are darker and more "
   "regular than the handwriting. If they are not removed they dominate "
   "every measurement.",
   "Lines of handwriting overlap vertically. A descender from one line hangs "
   "into the next, and a straight cut between two lines slices it in half, "
   "leaving a fragment attached to the wrong line.",
 ]),

 ("h", "How the toolkit is arranged"),
 ("p", "The first half of the chapter is built strictly from the bottom "
       "upwards, because each layer is used by the one above it: sums over "
       "rectangles, then shape operations built from those sums, then "
       "thresholding, then grouping touching pixels into objects, then "
       "thinning a stroke to its centreline, then resizing, then geometric "
       "transforms, then measurement."),

 ("h", "Why so much is built on rectangle sums"),
 ("p", "Almost every operation in the first half needs a local average or a "
       "local count: the average brightness near a pixel, whether any ink "
       "lies within a few pixels, how much ink is in a window. Computed "
       "directly, a window of radius r costs (2r+1) squared additions per "
       "pixel, so a radius of 40 on a page of several million pixels is "
       "thousands of additions each and the program becomes unusable. A "
       "summed-area table reduces every such query to four lookups, whatever "
       "the radius \\cite{crow1984}. That single idea is what makes the rest "
       "of the file practical, which is why it comes first."),
]


FUNCS = {

# ============================================================ window sums
 "Integral": dict(
   inputs=[("img", "a two-dimensional array of numbers, normally a greyscale "
                   "image")],
   output="An array one row and one column larger than the input, in which "
          "each entry holds the total of everything above and to the left of "
          "it.",
   theory=[
     ("h", "What it is for"),
     ("p", "This is the foundation of the toolkit. On its own it computes "
           "nothing useful; what it provides is the ability to ask for the "
           "total of any rectangle in the image at a cost that does not "
           "depend on the rectangle's size. Every local average, every "
           "shape-growing operation and every local ink count in this file "
           "uses it."),

     ("h", "Background: what a cumulative sum is"),
     ("p", "If a list of numbers is replaced by a list in which each entry is "
           "the total so far, then the sum of any stretch of the original "
           "list can be found by subtracting two entries of the new list. For "
           "the list 2, 3, 1, 4 the running totals are 2, 5, 6, 10, and the "
           "sum of the middle two numbers is 6 minus 2, which is 4. No "
           "addition is needed at query time, however long the stretch."),
     ("p", "A summed-area table, also called an integral image, is the same "
           "idea in two dimensions \\cite{crow1984}. Each cell holds the "
           "total of the whole rectangle from the top-left corner of the "
           "image down to that cell:"),

     ("eq", r"\mathrm{ii}[y,x] \;=\; \sum_{i<y}\;\sum_{j<x} \mathrm{img}[i,j]",
      [(r"\mathrm{ii}[y,x]", "the value stored at row y, column x of the "
                             "table"),
       (r"\mathrm{img}[i,j]", "the original pixel at row i, column j"),
       ("y, x", "the row and column being computed, counted from zero")],
      "crow1984"),

     ("h", "How a rectangle total is then obtained"),
     ("p", "To get the total inside a rectangle, take the table value at its "
           "bottom-right corner, which includes the rectangle but also "
           "everything above and to the left of it. Subtract the value at the "
           "top-right corner, which removes the band above. Subtract the "
           "value at the bottom-left corner, which removes the band to the "
           "left. Those two subtractions each removed the top-left block, so "
           "it has been taken away twice and must be added back once:"),

     ("eq", r"S \;=\; \mathrm{ii}[y_2,x_2] \;-\; \mathrm{ii}[y_1,x_2]"
            r"\;-\; \mathrm{ii}[y_2,x_1] \;+\; \mathrm{ii}[y_1,x_1]",
      [("S", "the total of all pixels inside the rectangle"),
       ("y_1, x_1", "the top-left corner of the rectangle"),
       ("y_2, x_2", "one row and one column past the bottom-right "
                    "corner, so the rectangle stops just before them")],
      "crow1984"),

     ("h", "Why the extra row and column of zeros"),
     ("p", "The formula above needs to read the table at the row above the "
           "rectangle and the column to its left. For a rectangle touching "
           "the top or left edge of the image those positions would be "
           "outside the array. Storing the table one larger in each "
           "direction, with the first row and column filled with zeros, means "
           "those reads land on genuine zeros instead and no special case is "
           "needed anywhere in the file."),

     ("h", "Why the result is stored as floating point"),
     ("p", "The entry in the bottom-right corner is the sum of every pixel in "
           "the image. For a 3000 by 2200 photograph with values up to 255 "
           "that is about 1.7 billion, which exceeds the range of a 32-bit "
           "signed integer. An overflow would not raise an error; it would "
           "wrap around to a negative number and quietly corrupt every "
           "rectangle total computed from that part of the table. Using "
           "64-bit floating point avoids this with room to spare."),

     ("h", "Cost"),
     ("p", "Two passes over the image to build, then four lookups per query "
           "regardless of window size. For the background estimate used by "
           "ink detection, with a radius of 40 pixels, the direct method "
           "would need 6561 additions per pixel; this needs four.")
   ]),

 "BoxSum": dict(
   inputs=[("img", "a two-dimensional array"),
           ("ry", "the window radius in rows, so the window spans 2*ry+1 rows"),
           ("rx", "the window radius in columns")],
   output="An array the same shape as the input, where each entry is the "
          "total of the input inside the window centred on that position.",
   theory=[
     ("h", "What it is for"),
     ("p", "Applies the rectangle query of the summed-area table at every "
           "position at once. This is the function that actually gets used; "
           "`Integral` exists to make it possible."),
     ("h", "How it works"),
     ("p", "The four corner positions of every pixel's window are built as "
           "whole arrays of indices rather than being computed one pixel at a "
           "time, and the table is then read at all of them simultaneously. "
           "The result is the same four-term expression as before, evaluated "
           "for the entire image in a handful of array operations."),
     ("p", "Doing it this way rather than with a loop matters because the "
           "loop would run in the Python interpreter, which is roughly two "
           "orders of magnitude slower than the same arithmetic performed "
           "inside numpy's compiled code. For a multi-megapixel photograph "
           "that is the difference between a fraction of a second and "
           "several minutes."),
     ("h", "What happens at the edges"),
     ("p", "A window centred near the border of the image hangs off the edge. "
           "The corner indices are clipped to stay inside the array, so the "
           "window is effectively cut short and the total covers fewer pixels "
           "than it would in the middle of the image. That is not corrected "
           "here, because the correction depends on what the caller wants: "
           "`BoxMean` divides by the number of pixels actually covered, while "
           "`Erode` compares against that same count. Both get what they need "
           "from `BoxCount`.")
   ]),

 "BoxCount": dict(
   inputs=[("shape or array", "used for its dimensions"),
           ("ry", "the window radius in rows"),
           ("rx", "the window radius in columns")],
   output="An integer array giving, for each position, how many pixels its "
          "window actually covered once clipped to the image.",
   theory=[
     ("h", "What it is for"),
     ("p", "Window sums near the border of an image cover fewer pixels than "
           "those in the middle. This counts how many were covered, which is "
           "what makes the border behaviour of the operations above it "
           "correct rather than approximately correct."),
     ("h", "How it works"),
     ("p", "The count is the clipped height of the window multiplied by its "
           "clipped width. Because those two depend only on the row and only "
           "on the column respectively, each is computed as a short "
           "one-dimensional array and the two are then multiplied together, "
           "which is far less work than computing a two-dimensional count "
           "directly. A quantity that factorises this way is called "
           "separable, and the same idea appears again in `Resize` and "
           "`Blur`."),
     ("h", "Why it matters: two different uses"),
     ("items", [
       "`BoxMean` divides the window sum by this count instead of by the "
       "nominal window area. Without that, the average near the border would "
       "be computed as though the missing pixels were zero, so the estimated "
       "background would fall towards black near every edge and ink "
       "detection would misbehave in a band around the whole page.",
       "`Erode` asks whether every pixel in the window is set, which it tests "
       "by comparing the window sum against this count. Comparing against "
       "the nominal area instead would make the test impossible to satisfy "
       "anywhere near the border, and a band around the page would be eroded "
       "away.",
     ])
   ]),

 "BoxMean": dict(
   inputs=[("img", "a two-dimensional array"),
           ("r", "a single radius used for both rows and columns")],
   output="A floating point array the same shape as the input, giving the "
          "average value in the window centred on each position.",
   theory=[
     ("h", "What it is for"),
     ("p", "This is how the pipeline estimates the local background "
           "brightness of a photograph. Comparing a pixel against the average "
           "of its own surroundings, rather than against one number for the "
           "whole page, is what allows ink to be found correctly on a page "
           "that is brighter on one side, or has a shadow across it, or was "
           "photographed with the flash off to one side."),
     ("h", "How it works"),
     ("eq", r"\mathrm{mean}[y,x] \;=\; \frac{\mathrm{BoxSum}[y,x]}"
            r"{\mathrm{BoxCount}[y,x]}",
      [(r"\mathrm{mean}[y,x]", "the local average at this position"),
       (r"\mathrm{BoxSum}[y,x]", "the total inside the window, from the "
                                 "summed-area table"),
       (r"\mathrm{BoxCount}[y,x]", "how many pixels that window covered")],
      None),
     ("h", "Why a flat window rather than a smooth one"),
     ("p", "A gaussian-weighted average, which weights nearby pixels more "
           "than distant ones, would be slightly smoother. A flat window is "
           "used because it can be computed from the summed-area table in "
           "constant time per pixel whatever the radius, where a gaussian "
           "cannot. For a background estimate the difference is not visible: "
           "what is wanted is the general level over a region, and the fine "
           "detail of how that region is weighted does not affect which "
           "pixels end up classified as ink.")
   ]),

# ============================================================== morphology
 "Dilate": dict(
   inputs=[("m", "a mask: an array of true and false values"),
           ("r", "the radius to grow by, in pixels")],
   output="A mask in which a position is true if any true pixel of the input "
          "lay within its window.",
   theory=[
     ("h", "Background: what morphology is"),
     ("p", "Morphology is the family of operations that change the shapes in "
           "a mask rather than the values in an image. The two basic "
           "operations are growing a shape outwards and shrinking it inwards, "
           "and almost everything else is built by combining them. The window "
           "they look through is called the structuring element; here it is "
           "always a square."),
     ("h", "What it is for"),
     ("p", "Growing shapes is used in this pipeline to make separate pieces "
           "of ink touch, so that they can be treated as one object. The "
           "clearest example is a letter such as i or j, which is written as "
           "two separate marks: growing both slightly makes them touch, after "
           "which the grouping stage treats the letter and its dot as a "
           "single piece and cannot assign them to different lines of "
           "writing."),
     ("h", "How it works"),
     ("p", "A position should be turned on if any true pixel lies within its "
           "window. Asking whether any pixel in a region is true is the same "
           "as asking whether the number of true pixels is greater than zero, "
           "and that number is exactly what a window sum provides:"),
     ("eq", r"\mathrm{dilate}(m)[p] \;=\; \big(\,\mathrm{BoxSum}(m,r,r)[p] "
            r"\;>\; 0 \,\big)",
      [("m", "the input mask, with true counted as 1 and false as 0"),
       ("p", "the position being computed"),
       ("r", "the radius of the square structuring element")],
      None),
     ("p", "Expressing it as a window sum means dilation inherits the "
           "constant cost per pixel of the summed-area table, so growing by "
           "twenty pixels costs the same as growing by two.")
   ]),

 "Erode": dict(
   inputs=[("m", "a mask"), ("r", "the radius to shrink by")],
   output="A mask in which a position is true only if every pixel its window "
          "covered was true in the input.",
   theory=[
     ("h", "What it is for"),
     ("p", "Shrinking shapes. On its own it is used to remove small specks "
           "and to thin a shape; its main use here is as the second half of "
           "a closing, described next."),
     ("h", "How it works, and the one subtlety"),
     ("p", "Where dilation asks whether \\emph{any} pixel in the window is "
           "set, erosion asks whether \\emph{every} one is. That is true "
           "exactly when the number of set pixels equals the number of pixels "
           "the window covered:"),
     ("eq", r"\mathrm{erode}(m)[p] \;=\; \big(\,\mathrm{BoxSum}(m,r,r)[p]"
            r"\;=\;\mathrm{BoxCount}(r,r)[p]\,\big)",
      [("m", "the input mask"),
       ("p", "the position being computed"),
       (r"\mathrm{BoxCount}(r,r)[p]", "how many pixels the window at p "
                                      "actually covered after clipping")],
      None),
     ("p", "The comparison must be against the covered count and not against "
           "the full window area of (2r+1) squared. Near the border the "
           "window is clipped, so its sum can never reach the full area, and "
           "a test against the area would report false for every position "
           "within r pixels of the edge. The visible result would be a band "
           "eroded away around the whole page, and since erosion is half of "
           "a closing, that band would then be removed from every mask the "
           "pipeline closes.")
   ]),

 "Close": dict(
   inputs=[("m", "a mask"), ("r", "the radius")],
   output="A mask with gaps narrower than the structuring element filled in, "
          "and the original thickness of the shapes preserved.",
   theory=[
     ("h", "What it is for"),
     ("p", "Repairing ink that should be continuous but is not. Thresholding "
           "a photograph leaves small holes inside strokes where the paper "
           "texture shows through, and a stroke drawn with a drying pen can "
           "have genuine breaks in it. Both are repaired by a closing."),
     ("h", "How it works, and why the two steps are needed"),
     ("p", "A closing is a dilation followed by an erosion by the same "
           "radius. The dilation joins anything separated by less than the "
           "window, which fills the gaps, but it also makes every stroke "
           "fatter. The erosion then removes a layer of the same thickness, "
           "which restores the original width. What does not come back is the "
           "gap: once two sides have been joined, eroding the joined shape "
           "thins it but does not split it again."),
     ("eq", r"\mathrm{close}(m,r) \;=\; \mathrm{erode}\big("
            r"\mathrm{dilate}(m,r),\, r\big)",
      [("m", "the input mask"),
       ("r", "the radius, which sets the largest gap that will be bridged")],
      None),
     ("p", "So a closing bridges a gap narrower than the kernel while leaving "
           "a thin stroke thin, which a plain dilation would not. That "
           "property is why it, and not dilation, is used before thinning: "
           "thinning is sensitive to both holes and thickness, so the "
           "operation applied beforehand must fix the first without changing "
           "the second.")
   ]),

 "AnyWithinRadius": dict(
   inputs=[("mask", "a mask"),
           ("r", "the radius"),
           ("axis", "which axis to work along: 0 for down the rows, 1 for "
                    "across the columns")],
   output="A mask that is true wherever any element within r positions along "
          "that single axis was true.",
   theory=[
     ("h", "What it is for"),
     ("p", "A dilation along one direction only. It is used where the "
           "tolerance being applied has a direction to it, and growing in "
           "both directions would be wrong."),
     ("p", "There are two such places. When a printed rule has been detected, "
           "it is widened slightly along its own length, so that a rule which "
           "is a fraction of a degree off horizontal is still removed over "
           "its whole length rather than in broken pieces; widening it "
           "vertically as well would eat into the writing above and below it. "
           "And when the paper region has been found, it is shrunk inwards by "
           "a margin in each axis separately, to keep the slightly soft edge "
           "of the sheet from being read as ink.")
   ]),

# ============================================================ runs of ink
 "LongRunMask": dict(
   inputs=[("m", "a mask"),
           ("n", "the minimum length, in pixels, that a run must reach"),
           ("axis", "the direction along which the run is measured")],
   output="A mask that is true at every pixel belonging to an unbroken run of "
          "at least n set pixels along that axis.",
   theory=[
     ("h", "What it is for"),
     ("p", "Telling printed lines apart from handwriting. This is the test "
           "that removes the ruling from a lined page and the dark edge of a "
           "binder, both of which would otherwise dominate every measurement "
           "the segmenter makes."),
     ("h", "Why run length is the right test"),
     ("p", "A printed rule is drawn by a machine: it is straight, continuous, "
           "and spans the page. A pen stroke changes direction constantly, so "
           "the number of pixels it continues in any one direction without a "
           "break is small. At the sizes this pipeline works with, the "
           "threshold is set to between 1.25 and 1.4 times the spacing "
           "between lines of writing, which is roughly 60 to 80 pixels; "
           "almost no handwritten stroke runs that far in a straight line, "
           "and a rule runs much further."),
     ("h", "How it works"),
     ("p", "Two passes of cumulative sums. The first finds where a long run "
           "begins. A stretch of n pixels starting at position i is entirely "
           "set exactly when the number of set pixels inside it equals n, and "
           "that count comes from one subtraction of the cumulative sum:"),
     ("eq", r"\mathrm{starts}[i] \;=\; \big(\, c[i+n] - c[i] \;=\; n \,\big)",
      [(r"\mathrm{starts}[i]", "true when a run of length n begins at "
                               "position i"),
       ("c", "the cumulative sum of the mask along the chosen axis, with a "
             "leading zero"),
       ("n", "the minimum run length"),
       ("i", "the candidate starting position")],
      None),
     ("p", "The second pass turns starting positions into coverage. A run of "
           "length n that covers position j must begin somewhere between n-1 "
           "places before j and j itself, and it cannot begin so late that it "
           "would run off the end. A second cumulative sum over the starts "
           "then answers \"is there any start in that range\" with another "
           "single subtraction:"),
     ("eq", r"\mathrm{covered}[j] \;=\; \big(\, cs[h_j+1] - cs[l_j] > 0 "
            r"\,\big), \qquad l_j = \max(0,\,j-n+1),\; h_j = \min(j,\,L-n)",
      [(r"\mathrm{covered}[j]", "true when position j lies inside some long "
                                "run"),
       ("cs", "the cumulative sum of the starts array"),
       ("l_j, h_j", "the first and last positions at which a run covering j "
                    "could begin, clamped to the valid range"),
       ("L", "the length of the axis being measured")],
      None),
     ("h", "Why it is done with cumulative sums"),
     ("p", "The cost is proportional to the number of pixels and does not "
           "depend on n. Scanning each window directly would cost n times as "
           "much, and n here is around 80."),
     ("h", "How one function handles both directions"),
     ("p", "The axis of interest is moved to the front of the array, the "
           "fixed code runs down it, and the axis is moved back afterwards. "
           "Moving an axis in numpy only changes how the array is "
           "interpreted, not the data in memory, so this costs nothing and "
           "avoids writing the same logic twice."),
     ("note", "A simpler equivalent exists. `RowRunLength`, described next, "
              "already gives every pixel the length of its own run, which "
              "makes this function the same as asking whether that length is "
              "at least n, with a transpose for the other axis. That form was "
              "checked against this one on 4800 randomly generated masks of "
              "every shape, with n from 1 to 40 and both axes, and never "
              "disagreed; on a page-sized array it also ran about twice as "
              "fast, because it makes one labelling pass and one count "
              "instead of building two cumulative-sum arrays and several "
              "arrays of clamped indices.")
   ]),

 "RowRunLength": dict(
   inputs=[("mask", "a two-dimensional mask; runs are measured across the "
                    "rows")],
   output="An integer array in which every set pixel holds the length of the "
          "unbroken horizontal run it belongs to, and every clear pixel holds "
          "zero.",
   theory=[
     ("h", "What it is for"),
     ("p", "Measuring how far ink continues horizontally through each pixel. "
           "It is used to find printed rules, and in `BinarizeLine` to delete "
           "them from a line crop."),
     ("h", "How it works, in three vectorised steps"),
     ("p", "First, the places where a run begins are found as the positions "
           "where the row steps from clear to set, which is a comparison "
           "between the row and a copy of itself shifted by one."),
     ("p", "Second, a cumulative sum over those starting positions numbers "
           "the runs consecutively: the first run becomes 1, the second 2, "
           "and so on. The numbering is then cleared wherever there is no "
           "ink, so that only ink carries a run number."),
     ("p", "Third, counting how many pixels carry each number gives the "
           "length of each run, and indexing those counts by the run number "
           "image hands every pixel the length of its own run."),
     ("h", "Why the numbering does not leak between rows"),
     ("p", "The cumulative sum is taken over the whole array flattened into "
           "one long sequence, which would be wrong if a run could continue "
           "from the end of one row into the start of the next. It cannot: "
           "the comparison that finds starts is padded at both ends of every "
           "row, so each row begins and ends with a boundary and the last run "
           "of a row is always closed before the next row's first run "
           "begins."),
     ("h", "Background: what counting occurrences means here"),
     ("p", "The counting step uses a standard operation that, given a list of "
           "non-negative whole numbers, returns how many times each value "
           "occurs. Applied to an image of run numbers it gives the size of "
           "every run in one pass, which is why no loop over runs is needed.")
   ]),

# =========================================================== thresholding
 "Otsu": dict(
   inputs=[("v", "an array of pixel values, either whole numbers or "
                 "fractions"),
           ("hi", "when the input is fractional, the largest value worth "
                  "considering; defaults to 1.2, which suits the ratio image "
                  "produced by `InkMask` because nothing above that carries "
                  "ink")],
   output="A threshold. For whole-number input it is a brightness level from "
          "0 to 255; for fractional input it is a value in the input's own "
          "units. Pixels below the threshold are ink.",
   theory=[
     ("h", "What it is for"),
     ("p", "Deciding, for a given image, which brightness separates ink from "
           "paper. The point of computing it rather than fixing it is that no "
           "single number works for two photographs: one taken in daylight "
           "and one under a lamp have completely different brightness "
           "ranges, and a fixed threshold would classify the whole of one of "
           "them as paper."),

     ("h", "Background: what a histogram is"),
     ("p", "A histogram counts how many pixels fall at each brightness level, "
           "ignoring where they are. For a page of handwriting it usually has "
           "two humps: a large one at the bright end for the paper and a "
           "small one at the dark end for the ink, with a valley between "
           "them. The threshold should sit in that valley."),

     ("h", "The idea: measure the quality of every candidate threshold"),
     ("p", "Otsu's method \\cite{otsu1979} turns the choice into something "
           "measurable. Any candidate threshold divides the pixels into two "
           "groups. A good threshold is one where the two groups are as "
           "different from each other as possible, which is measured by the "
           "variance \\emph{between} the groups:"),

     ("eq", r"\sigma_B^2(t) \;=\; \omega_B(t)\,\omega_F(t)\,"
            r"\big[\mu_B(t) - \mu_F(t)\big]^2",
      [(r"\sigma_B^2(t)", "the between-class variance, the quantity being "
                          "maximised"),
       ("t", "the candidate threshold"),
       (r"\omega_B(t)", "the fraction of all pixels at or below t, that is "
                        "the share in the darker group"),
       (r"\omega_F(t)", "the fraction above t, so the two fractions add to 1"),
       (r"\mu_B(t)", "the mean brightness of the darker group"),
       (r"\mu_F(t)", "the mean brightness of the brighter group")],
      "otsu1979"),

     ("p", "Reading the formula: the squared bracket is how far apart the two "
           "group means are, so it rewards a threshold that genuinely "
           "separates dark from bright. The product of the two fractions in "
           "front is largest when the groups are of comparable size and falls "
           "to zero when either is empty, which prevents the degenerate "
           "answer of putting the threshold at one extreme and calling almost "
           "everything one group."),

     ("h", "Why maximising this is the right objective"),
     ("p", "The total variance of the image is a fixed property of the image "
           "and does not depend on the threshold. It can be split exactly "
           "into the variance within the two groups plus the variance between "
           "them. Since the total is fixed, making the between-group part as "
           "large as possible is the same as making the within-group part as "
           "small as possible, that is, choosing the threshold that makes "
           "each group internally as uniform as it can be. The between-group "
           "form is used because it needs only running totals, where the "
           "within-group form would need a second pass over the data for "
           "every candidate."),

     ("h", "How all 256 candidates are evaluated at once"),
     ("p", "Writing p for the histogram scaled so that its entries sum to "
           "one, the two quantities needed are themselves cumulative sums, "
           "and the criterion rearranges into a form built entirely from "
           "them:"),

     ("eq", r"\sigma_B^2(t)\;=\;\frac{\big[\mu_T\,\omega(t)-\mu(t)\big]^2}"
            r"{\omega(t)\,\big(1-\omega(t)\big)}",
      [(r"\omega(t)", "the cumulative sum of p up to t, the share of pixels "
                      "at or below t"),
       (r"\mu(t)", "the cumulative sum of p weighted by brightness, the "
                   "contribution of those pixels to the mean"),
       (r"\mu_T", "the mean brightness of the whole image, which is the final "
                  "value of the weighted cumulative sum")],
      "otsu1979"),

     ("p", "Both cumulative sums are computed once, the expression is "
           "evaluated for every t simultaneously as array arithmetic, and the "
           "largest entry is the answer. No loop over thresholds is needed."),

     ("h", "The guard in the denominator"),
     ("p", "At the two extreme thresholds one group is empty, so the product "
           "in the denominator is exactly zero. Dividing by it would give "
           "infinity, and since the method picks the largest value those "
           "positions would always win. The denominator is therefore floored "
           "at a very small positive number, which leaves every genuine "
           "candidate unchanged and makes the degenerate ones lose."),

     ("h", "Why whole numbers and fractions are handled separately"),
     ("p", "A whole-number image has exactly 256 possible values, so its "
           "histogram can be counted exactly and the answer returned as a "
           "brightness level. A fractional image, such as the ratio of each "
           "pixel to its local background, has no natural set of levels, so "
           "its values are clipped at `hi` and sorted into 256 bins, and the "
           "answer is returned as the value at the centre of the chosen bin "
           "rather than as a bin number.")
   ]),

 "BinarizeLine": dict(
   inputs=[("gray", "a greyscale image of one line of writing")],
   output="A mask of the ink in that crop, with small holes closed, specks "
          "removed and printed rules deleted.",
   theory=[
     ("h", "What it is for"),
     ("p", "The standard way of turning a grey line crop into ink. The glyph "
           "extractor uses it before tracing strokes, and the synthesiser "
           "uses it to measure its own rendered output with the same code "
           "that measured the writer's real pages, which is what makes the "
           "two comparable."),
     ("h", "The four steps and why each is needed"),
     ("p", "\\textbf{Threshold.} `Otsu` chooses the level, so nothing about "
           "the brightness of this particular photograph is assumed."),
     ("p", "\\textbf{Close small holes.} Thresholding a photograph leaves "
           "single-pixel holes inside strokes where the paper texture shows "
           "through. These matter more than their size suggests, because the "
           "next stage after binarising is usually thinning, and thinning "
           "preserves the topology of the shape it is given: a hole in a "
           "stroke forces the resulting centreline to split and run round "
           "both sides of it, producing a small loop that was never written."),
     ("p", "\\textbf{Remove specks.} Pixels are grouped into connected "
           "components and any group too small to be a pen mark is deleted as "
           "dust or camera noise. The decision is made for a whole group at "
           "once, so a letter is never partly removed."),
     ("p", "\\textbf{Delete printed rules.} Horizontal runs longer than a "
           "limit are removed, on the reasoning given under `LongRunMask`: a "
           "pen stroke does not run level for that distance, and a printed "
           "rule does nothing else.")
   ]),

 "CoreBand": dict(
   inputs=[("ink", "a mask of one line of writing")],
   output="A pair of row indices marking the top and the baseline of the "
          "x-height band, or nothing at all if there is no ink or the band is "
          "under three rows tall.",
   theory=[
     ("h", "Background: what the x-height is"),
     ("p", "In any piece of writing, most lower case letters have a body of "
           "roughly the same height, such as a, c, e, m and o. The height of "
           "that body, measured up from the line the writing sits on, is "
           "called the x-height, after the letter x. Some letters reach above "
           "it, such as b, d and l, and these are called ascenders; some "
           "reach below the line, such as g, p and y, and these are called "
           "descenders."),
     ("h", "What it is for, and why it is so important"),
     ("p", "The x-height is the unit of measurement for the entire pipeline. "
           "Every stored letter shape is divided by it, so that a letter "
           "extracted from a photograph at one resolution can be drawn at any "
           "physical size. Every proportion in a style profile is expressed "
           "as a multiple of it: how far ascenders reach, how far descenders "
           "drop, how wide the gaps between words are, how thick the pen is. "
           "Without a reliable x-height none of those numbers would be "
           "comparable between two writers or between two photographs of the "
           "same writer."),
     ("h", "How it works"),
     ("p", "Counting the ink in each row of the crop gives a one-dimensional "
           "summary called a projection profile. Rows that cross the bodies "
           "of the letters contain ink from nearly every letter on the line; "
           "rows that contain only ascenders or only descenders contain ink "
           "from the few letters that have them. The band is therefore the "
           "set of rows whose ink count is a large fraction of the maximum:"),
     ("eq", r"\mathcal{B} \;=\; \Big\{\, y \;:\; \mathrm{prof}[y] \;\ge\; "
            r"0.45\,\max_{y'} \mathrm{prof}[y'] \,\Big\}",
      [(r"\mathcal{B}", "the set of rows making up the band; its first and "
                        "last members are returned"),
       (r"\mathrm{prof}[y]", "the number of ink pixels in row y"),
       ("0.45", "the fraction of the peak that a row must reach to count as "
                "part of the band")],
      None),
     ("h", "Where the 0.45 comes from"),
     ("p", "It is the one tuned constant, and the reasoning behind it is "
           "that the quantity being thresholded has a wide gap in it. A row "
           "through the letter bodies collects ink from most letters on the "
           "line, while a row above the band collects ink only from the few "
           "ascenders, which in ordinary English text is a small minority of "
           "letters. Any threshold within that gap gives the same answer, so "
           "the exact figure is not critical; what would fail is a value low "
           "enough to admit the ascender rows, which would inflate the "
           "measured x-height, or high enough to admit only the single "
           "densest row, which would collapse the band to nothing."),
     ("h", "The rejection case"),
     ("p", "A band under three rows tall is reported as a failure rather "
           "than returned. That is what happens when a crop contains a stray "
           "mark rather than a line of writing, and returning a nonsensical "
           "x-height would propagate into every measurement taken from that "
           "crop.")
   ]),
}


#the rest of the chapter, in order
for _part in ("codetheory_segment2", "codetheory_segment3",
              "codetheory_segment4"):
    try:
        _m = __import__(_part)
    except ModuleNotFoundError:
        continue
    FUNCS.update(_m.FUNCS)
