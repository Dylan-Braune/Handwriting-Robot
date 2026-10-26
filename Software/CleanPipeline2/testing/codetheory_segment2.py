r"""Theory for segment.py, part 2: components, thinning and resizing."""

FUNCS = {

 "LabelComponents": dict(
   inputs=[("mask", "a two-dimensional mask of ink")],
   output="A pair: an array in which every ink pixel carries the number of "
          "the object it belongs to, counting from 1, with 0 for background; "
          "and the number of objects found.",
   theory=[
     ("h", "Background: what a connected component is"),
     ("p", "Two ink pixels that touch belong to the same physical mark. "
           "Following that relation outwards from any pixel gives a group of "
           "pixels all connected to one another, called a connected "
           "component. On a page of handwriting the components are the "
           "individual pen marks: a letter, the dot of an i, a speck of dust, "
           "or several joined letters where the writer did not lift the pen."),
     ("p", "What counts as touching has to be decided. Counting only the four "
           "pixels above, below, left and right is called four-connectivity; "
           "also counting the four diagonals is eight-connectivity. This "
           "implementation uses eight, because a thin pen stroke drawn at an "
           "angle advances diagonally, and under four-connectivity such a "
           "stroke would break into a chain of separate dots."),
     ("h", "What it is for"),
     ("p", "This is what lets the pipeline reason about objects rather than "
           "pixels, and that distinction runs through the whole segmenter. "
           "Whether a mark is dust is a question about a whole object; which "
           "line of writing a letter belongs to is a question about a whole "
           "object. Tests made pixel by pixel would remove half a letter, or "
           "split one letter across two lines."),
     ("h", "How it works: runs instead of pixels"),
     ("p", "Each row is first reduced to horizontal runs of ink, found from "
           "the places where the row steps from clear to set and back again. "
           "Every pixel in a run necessarily belongs to the same component, "
           "so the algorithm reasons about runs, of which there are far "
           "fewer. This is the run-based two-scan approach \\cite{he2008}."),
     ("p", "Each run is compared against the runs of the row above. Because "
           "the runs in both rows are already in left-to-right order, a "
           "single index walks forward through the previous row and never "
           "goes back, so the total work is proportional to the number of "
           "runs rather than to their number squared. A run above counts as "
           "touching when its span overlaps, and the comparison is written so "
           "that a touch at a single corner counts, which is what produces "
           "eight-connectivity."),
     ("h", "The problem that makes this non-trivial"),
     ("p", "Labels must be handed out while scanning downwards, before it is "
           "known how a shape continues. A letter U is the clearest example: "
           "its two arms are separate for many rows and receive different "
           "labels, and only at the bottom does a run join them, revealing "
           "that the two labels were always one object."),
     ("h", "Union-find, and why it is fast"),
     ("p", "Rather than going back and relabelling, which would mean "
           "rescanning everything above, the fact that two labels are the "
           "same is recorded. Each label stores a pointer to another label, "
           "and following the pointers reaches a representative for the "
           "group; joining two groups is then one assignment, pointing the "
           "larger representative at the smaller. This structure is called "
           "union-find \\cite{tarjan1975}."),
     ("p", "Chains of pointers could grow long, so each lookup shortens the "
           "chain it walks by pointing every label it passes at its "
           "grandparent:"),
     ("code", ["while parent[a] != a:",
               "    parent[a] = parent[parent[a]]   # halve the chain",
               "    a = parent[a]"]),
     ("p", "With that shortening the cost of a lookup is so close to constant "
           "that for any practical image it can be treated as constant "
           "\\cite{tarjan1975}."),
     ("h", "The final renumbering, and why callers depend on it"),
     ("p", "Because labels were handed out before the merges were known, the "
           "surviving labels have gaps: label 7 may have been absorbed into "
           "label 3, so no pixel carries 7 any more. A closing pass resolves "
           "every label to its representative and renumbers the survivors "
           "consecutively from 1. Callers rely on this, because it means the "
           "components are numbered 1 to n with nothing missing, so their "
           "sizes can be counted in a single operation and used directly as "
           "array indices."),
   ]),

 "PieceBounds": dict(
   inputs=[("pl", "the component number of every ink pixel"),
           ("ys", "the row coordinate of every ink pixel"),
           ("xs", "the column coordinate of every ink pixel"),
           ("nc", "how many components there are"),
           ("H", "the image height"),
           ("W", "the image width")],
   output="Five arrays indexed by component number: each object's topmost "
          "row, bottommost row, leftmost column, rightmost column and pixel "
          "count.",
   theory=[
     ("h", "What it is for"),
     ("p", "Every ownership rule in the segmenter asks geometric questions "
           "about objects: how tall is this piece, how wide is it, where does "
           "it sit, how much ink does it contain. This answers all of them "
           "for every object at once."),
     ("h", "Why it is written as a scatter rather than a loop"),
     ("p", "The obvious implementation loops over components and, for each "
           "one, selects the pixels belonging to it and takes their minimum "
           "and maximum. That reads the whole image once per component, so on "
           "a page with eight hundred pen marks it reads the image eight "
           "hundred times."),
     ("p", "Instead the coordinates are scattered into per-component "
           "accumulators in a single pass, using operations that take a list "
           "of destinations and a list of values and apply a minimum or a "
           "maximum at each destination. The pixel count comes from the same "
           "occurrence-counting operation used by `RowRunLength`. The result "
           "is one pass over the ink rather than one pass per object."),
   ]),

 "Skeletonize": dict(
   inputs=[("mask", "a mask of ink")],
   output="A mask of the one-pixel-wide centreline of that ink.",
   theory=[
     ("h", "What it is for"),
     ("p", "A photograph records a pen stroke as a thick band of dark "
           "pixels. What the project needs is the path the pen travelled down "
           "the middle of that band, because a path can be stored as a short "
           "list of points, compared against another writing of the same "
           "letter, and above all redrawn by a machine holding a pen. "
           "Thinning recovers it."),
     ("h", "Background: what thinning must preserve"),
     ("p", "Deleting pixels from the edge of a shape repeatedly will "
           "eventually make it thin, but done naively it destroys the shape: "
           "an o can be broken open into a c, and a crossing can fall apart "
           "into four disconnected arms. A thinning algorithm must therefore "
           "delete only pixels whose removal cannot change how the shape is "
           "connected. The algorithm here is the standard parallel method of "
           "Zhang and Suen \\cite{zhangsuen1984}."),
     ("h", "The two quantities the test is built from"),
     ("p", "Consider a pixel and its eight neighbours, numbered clockwise "
           "from the one directly above. Two numbers describe the local "
           "situation:"),
     ("eq", r"B(P_1) \;=\; \sum_{k=2}^{9} P_k, \qquad "
            r"A(P_1) \;=\; \#\big\{\, k \;:\; P_k = 0 \;\wedge\; "
            r"P_{k+1} = 1 \,\big\}",
      [("P_1", "the pixel being considered"),
       ("P_2 \\ldots P_9", "its eight neighbours, clockwise from the north, "
                           "each 1 where there is ink and 0 where there is "
                           "not"),
       ("B(P_1)", "how many of the eight neighbours are ink"),
       ("A(P_1)", "how many times the neighbours change from empty to ink "
                  "while walking once round the ring, the ring being treated "
                  "as closed so that the ninth neighbour is the second again")],
      "zhangsuen1984"),
     ("p", "The second quantity deserves an explanation, because it is the "
           "heart of the method. Walking once round the eight neighbours and "
           "counting the steps from empty into ink tells you how many "
           "separate groups of ink surround the pixel. If the answer is one, "
           "all the neighbouring ink forms a single unbroken arc, so the "
           "pixel lies on the edge of the stroke and deleting it leaves that "
           "arc joined. If the answer is two or more, the pixel sits between "
           "separate groups and is the only thing holding them together, so "
           "deleting it would break the stroke apart or open a hole."),
     ("h", "The deletion test"),
     ("eq", r"\begin{aligned}"
            r"&\text{(a)}\quad && 2 \le B(P_1) \le 6 \\"
            r"&\text{(b)}\quad && A(P_1) = 1 \\"
            r"&\text{(c)}\quad && P_2 \cdot P_4 \cdot P_6 = 0 \\"
            r"&\text{(d)}\quad && P_4 \cdot P_6 \cdot P_8 = 0"
            r"\end{aligned}",
      [("(a)", "at least two neighbours, so the tip of a stroke is kept and "
               "the line is not eaten back from its end; at most six, so a "
               "pixel deep inside a thick region is not removed before the "
               "layers outside it"),
       ("(b)", "the neighbouring ink forms a single arc, so removal cannot "
               "disconnect anything"),
       ("(c), (d)", "the corner conditions, which restrict this "
                    "sub-iteration to thinning from particular sides; in the "
                    "second sub-iteration they become P2*P4*P8 = 0 and "
                    "P2*P6*P8 = 0")],
      "zhangsuen1984"),
     ("h", "Why there must be two alternating sub-iterations"),
     ("p", "If deletion were allowed from every side at once, a line two "
           "pixels wide would vanish completely: both halves satisfy the test "
           "simultaneously and both are deleted in the same pass, leaving "
           "nothing behind. Alternating the corner conditions means each pass "
           "thins from opposite sides, so one side is always held while the "
           "other is removed. The result stays connected and the centreline "
           "does not drift to one side of the original stroke."),
     ("h", "Termination and cost"),
     ("p", "The pair of sub-iterations repeats until a full pass deletes "
           "nothing, at which point no pixel can be removed without changing "
           "the connectivity, and the shape is one pixel wide everywhere. The "
           "number of passes is proportional to the thickest stroke, since "
           "one layer comes off each side per pass, and each pass is a fixed "
           "number of whole-array operations."),
     ("note", "This is also why `BinarizeLine` closes small holes before "
              "thinning. Thinning preserves connectivity faithfully, "
              "including the connectivity of a hole that was never written: a "
              "pinhole inside a stroke forces the centreline to divide and "
              "pass round both sides of it, leaving a small loop in the "
              "recorded pen path that the writer never drew."),
   ]),

 "_Weights": dict(
   inputs=[("inSize", "the length of the axis before resizing"),
           ("outSize", "the length wanted afterwards"),
           ("kernel", "which filter to use: \"linear\" for a triangle, "
                      "\"box\" for a flat window")],
   output="A matrix with one row per output sample and one column per input "
          "sample; each row holds the weight of every input sample in that "
          "output sample, and each row sums to one.",
   theory=[
     ("h", "Background: what resampling actually is"),
     ("p", "Changing the size of an image means asking for brightness values "
           "at positions where there were none. In an output image half the "
           "width, output pixel 1 sits between input pixels 2 and 3, so its "
           "value has to be built from the inputs around it. Choosing how to "
           "build it is choosing a filter, and a filter is described by a "
           "weight for each nearby input sample."),
     ("h", "What it is for"),
     ("p", "Every size change in the pipeline goes through one resampler, so "
           "there is exactly one piece of code to get right: reducing a page "
           "for the skew search, fitting a crop to the networks' fixed input "
           "size, and rendering glyphs. This function builds the weights and "
           "`Resize` applies them."),
     ("h", "Where an output sample sits"),
     ("eq", r"s \;=\; \frac{\mathrm{inSize}}{\mathrm{outSize}}, \qquad "
            r"c_i \;=\; \big(i + \tfrac{1}{2}\big)\, s",
      [("s", "the scale factor; greater than one means the image is being "
             "shrunk"),
       ("c_i", "the centre of output sample i, expressed in input "
               "coordinates"),
       ("i", "the index of the output sample, counting from zero")],
      "wolberg1990"),
     ("p", "The halves matter. A pixel is not a point at an integer "
           "coordinate; it is a small square whose centre lies at an integer "
           "plus a half. Treating pixels as points displaces the whole image "
           "by half a pixel, which shows as a slight softening and as a "
           "systematic shift that accumulates if an image is resized more "
           "than once."),
     ("h", "Why the filter widens when shrinking"),
     ("p", "This is the part that is easy to get wrong and expensive to get "
           "wrong. The width of the filter is multiplied by the scale factor "
           "whenever that factor exceeds one:"),
     ("eq", r"f \;=\; \max(1,\, s), \qquad \mathrm{support} \;=\; k_0\, f",
      [("f", "the widening factor: one when magnifying, equal to the scale "
             "factor when minifying"),
       ("s", "the scale factor defined above"),
       ("k_0", "the base support of the chosen kernel: one for the triangle, "
               "one half for the box"),
       (r"\mathrm{support}", "how far the filter reaches from its centre, "
                             "measured in input samples")],
      "wolberg1990"),
     ("p", "If the filter did not widen, shrinking by a factor of four would "
           "consult one input sample in every four and ignore the rest. Thin "
           "strokes falling between the consulted samples would disappear "
           "entirely, and fine texture would turn into coarse false patterns, "
           "an effect called aliasing. Widening the filter in step with the "
           "reduction means every input sample contributes to some output "
           "sample \\cite{wolberg1990}. The page reduction used by the skew "
           "search shrinks by exactly such a factor, so this is load bearing "
           "rather than theoretical."),
     ("h", "The two kernels"),
     ("eq", r"w_{\mathrm{linear}}(t) = \begin{cases} 1 - |t| & |t| < 1 \\ 0 & "
            r"\text{otherwise} \end{cases} \qquad\quad "
            r"w_{\mathrm{box}}(t) = \begin{cases} 1 & |t| \le \tfrac{1}{2} \\ "
            r"0 & \text{otherwise} \end{cases}",
      [("t", "the distance from the centre of the output sample's window to "
             "an input sample, measured in filter widths"),
       ("w(t)", "the weight that input sample receives")],
      "wolberg1990"),
     ("p", "The triangle weights nearby samples more than distant ones and "
           "falls smoothly to zero, which gives a smooth result. The box "
           "weights everything inside the window equally, which is exactly a "
           "local average and is the right choice when reducing by a whole "
           "number, because then each output pixel covers a whole number of "
           "input pixels."),
     ("h", "Edges, and why weights are accumulated rather than assigned"),
     ("p", "A window near the border reaches past the end of the axis. Those "
           "out-of-range positions are folded back onto the first or last "
           "sample, so several positions in one window can refer to the same "
           "input sample. Their weights must therefore be added together "
           "rather than written one over another: assigning would keep only "
           "the last and discard the rest, leaving the edge of the image too "
           "dark."),
     ("p", "Each row is finally divided by its own total so that it sums to "
           "one. That is what guarantees a uniform area keeps its brightness "
           "after resizing, and in particular that the white padding around a "
           "fitted crop does not come back grey."),
   ]),

 "Resize": dict(
   inputs=[("img", "the image to resize"),
           ("newW", "the width wanted"),
           ("newH", "the height wanted"),
           ("kernel", "which filter to use")],
   output="The resampled image at the requested size, in the same number "
          "format as the input.",
   theory=[
     ("h", "What it is for"),
     ("p", "The single resampling routine used by the whole pipeline."),
     ("h", "Background: what separable means, and why it is worth it"),
     ("p", "Resizing in two dimensions could be done with a two-dimensional "
           "filter, where each output pixel is a weighted sum over a square "
           "of input pixels. For a filter k samples across that is k squared "
           "multiplications per output pixel."),
     ("p", "A filter is called separable when its two-dimensional weight is "
           "the product of two one-dimensional weights, one per axis. Both "
           "kernels used here are separable, so the same result is obtained "
           "by resizing along the rows and then along the columns, at a cost "
           "of 2k multiplications per pixel instead of k squared "
           "\\cite{wolberg1990}. At the filter widths used when reducing a "
           "page that is roughly an order of magnitude less work."),
     ("h", "How it works"),
     ("p", "`_Weights` is called once per axis and the two matrices are "
           "applied in turn as matrix products, so the arithmetic happens "
           "inside numpy's compiled routines rather than in interpreted "
           "loops."),
   ]),

 "Halve": dict(
   inputs=[("img", "the image to reduce")],
   output="The image at half the width and half the height.",
   theory=[
     ("h", "What it is for"),
     ("p", "A fast reduction for stages that do not need full resolution: "
           "finding the sheet of paper inside the photograph, and the coarse "
           "stage of the skew search."),
     ("h", "Why it is a separate function"),
     ("p", "At the exact ratio of two to one a box filter reduces to "
           "averaging each two-by-two block, which can be done by reshaping "
           "the array and taking a mean, with no weights built at all. It is "
           "both the cheapest correct reduction at that ratio and exactly "
           "what the general resampler would compute, so nothing is lost by "
           "using it."),
     ("p", "Why reduce at all: every operation in the toolkit costs time "
           "proportional to the number of pixels, so halving each axis "
           "quarters the work. For a decision that only needs to locate "
           "something to within a few pixels, the reduction costs nothing in "
           "accuracy."),
   ]),

 "FitFixedCanvas": dict(
   inputs=[("gray", "a line crop"),
           ("height", "the canvas height the networks expect, 64"),
           ("width", "the canvas width the networks expect, 640"),
           ("padValue", "the value filling the unused area; 255 means white "
                        "paper")],
   output="An image of exactly the requested size containing the crop, "
          "scaled to fit, centred vertically and flush to the left, on a "
          "paper-coloured background.",
   theory=[
     ("h", "What it is for"),
     ("p", "Both trained networks accept one exact input size and nothing "
           "else, because the number of weights in their final layers depends "
           "on it. Every crop therefore has to be brought to that size before "
           "it can be read, and how that is done affects the answer."),
     ("h", "The central rule: one scale factor, never two"),
     ("eq", r"\sigma \;=\; \min\!\left(\frac{H_{\mathrm{canvas}}}"
            r"{H_{\mathrm{crop}}},\; "
            r"\frac{W_{\mathrm{canvas}}}{W_{\mathrm{crop}}}\right)",
      [(r"\sigma", "the single scale factor applied to both axes"),
       (r"H_{\mathrm{canvas}},\, W_{\mathrm{canvas}}", "the fixed canvas "
                                                       "size the networks "
                                                       "were trained on"),
       (r"H_{\mathrm{crop}},\, W_{\mathrm{crop}}", "the size of the crop "
                                                   "being fitted")],
      None),
     ("p", "Taking the smaller of the two ratios means the crop is scaled "
           "until whichever axis runs out of room first exactly fills the "
           "canvas, leaving the other axis short; the leftover space is "
           "filled with the paper colour."),
     ("p", "The alternative, scaling each axis independently so the crop "
           "fills the canvas exactly, would change the ratio of width to "
           "height. That ratio is precisely what both networks read: the "
           "reader distinguishes letters partly by their proportions, and the "
           "writer namer treats proportion as a characteristic of a hand. A "
           "stretched crop is, to these networks, a different person's "
           "handwriting."),
     ("h", "Why centred vertically but flush to the left"),
     ("p", "It matches how the training crops were produced. Writing starts "
           "at the left margin, so the left edge carries meaning and is "
           "preserved. The vertical position of a line of writing inside its "
           "crop carries none, so centring removes a difference the network "
           "would otherwise have to learn to ignore."),
     ("h", "A small implementation guard"),
     ("p", "The scaled size is rounded to whole pixels, and rounding can "
           "produce a result one pixel larger than the canvas. The final copy "
           "therefore limits both the source and the destination, so an "
           "overshoot is trimmed instead of raising an error."),
   ]),

 "_Sample": dict(
   inputs=[("img", "the source image"),
           ("sy", "for each output pixel, the row to read from, which may "
                  "fall between rows"),
           ("sx", "the corresponding column, which may fall between columns")],
   output="The sampled values, one per requested position.",
   theory=[
     ("h", "What it is for"),
     ("p", "Rotating or bending an image asks for brightness at positions "
           "falling between pixels. This supplies them, and is the shared "
           "sampler behind every geometric transform in the file."),
     ("h", "Background: bilinear interpolation"),
     ("p", "The value at a position between four pixels is their weighted "
           "average, each weight given by how close the position is to that "
           "pixel. It is computed one axis at a time: blend the upper pair "
           "horizontally, blend the lower pair horizontally, then blend those "
           "two results vertically."),
     ("eq", r"\begin{aligned}"
            r"v_{\mathrm{top}} &= p_{00}\,(1-f_x) + p_{01}\, f_x \\[2pt]"
            r"v_{\mathrm{bot}} &= p_{10}\,(1-f_x) + p_{11}\, f_x \\[2pt]"
            r"v &= v_{\mathrm{top}}\,(1-f_y) + v_{\mathrm{bot}}\, f_y"
            r"\end{aligned}",
      [("p_{00},\\, p_{01}", "the two source pixels above the position, left "
                             "and right"),
       ("p_{10},\\, p_{11}", "the two below it"),
       ("f_x", "how far the position lies between the left and right "
               "columns: 0 at the left pixel, 1 at the right"),
       ("f_y", "the same fraction between the upper and lower rows"),
       ("v", "the interpolated value returned")],
      "wolberg1990"),
     ("h", "Why interpolate at all"),
     ("p", "The cheaper alternative is to take whichever single pixel the "
           "position is nearest. On a rotated image the result is a visible "
           "staircase along every edge, because neighbouring output pixels "
           "jump between different source pixels. Blending removes it, and "
           "that matters here beyond appearance: the ink masks and skeletons "
           "computed downstream are sensitive to jagged edges, which produce "
           "spurious spurs when thinned."),
   ]),

 "_Like": dict(
   inputs=[("out", "a floating point result"),
           ("ref", "the array whose number format should be matched")],
   output="The result converted back to the reference's format, rounded and "
          "clipped to its range.",
   theory=[
     ("h", "What it is for"),
     ("p", "Interpolation must be done in floating point, but the pipeline "
           "passes images around as whole numbers from 0 to 255, because that "
           "is what a photograph is and what the networks were trained on. "
           "This performs the conversion back."),
     ("h", "The two things it gets right"),
     ("p", "It rounds rather than truncating. Truncation would bias every "
           "transformed image slightly darker, since 254.7 would become 254 "
           "rather than 255, and the bias would accumulate through successive "
           "transforms."),
     ("p", "It clips to the valid range. A blend can overshoot slightly past "
           "255 or below 0, and converting such a value to an 8-bit whole "
           "number without clipping makes it wrap around to the opposite "
           "extreme, turning a bright pixel black. A handful of such pixels "
           "is enough to disturb a threshold or a component count."),
   ]),
}
