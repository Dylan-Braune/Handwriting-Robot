r"""Theory for segment.py, part 3: geometric transforms, drawing, measuring."""

FUNCS = {

 "Rotate": dict(
   inputs=[("img", "the image to rotate"),
           ("deg", "the angle in degrees"),
           ("fill", "the value used where no source pixel exists")],
   output="The rotated image, on a canvas the same size as the input.",
   theory=[
     ("h", "What it is for"),
     ("p", "Straightening a photographed page. The page is never square to "
           "the camera, and a tilt of even one degree means a horizontal cut "
           "through the image crosses more than one line of writing, which "
           "makes every later stage harder."),
     ("h", "Background: forward and inverse mapping"),
     ("p", "There are two ways to transform an image. The obvious one, called "
           "forward mapping, takes each input pixel, works out where it "
           "should land, and writes it there. The problem is that the "
           "destinations are fractional and unevenly spread, so some output "
           "pixels receive two input pixels and others receive none, leaving "
           "a scatter of holes."),
     ("p", "The method used here is inverse mapping: for each \\emph{output} "
           "pixel, work out where it came from in the input and read that "
           "position. Every output pixel is then visited exactly once, so "
           "holes are impossible \\cite{wolberg1990}. The source position is "
           "fractional, which is what `_Sample` is for."),
     ("h", "The mapping"),
     ("eq", r"\begin{aligned}"
            r"s_x &= \cos\theta\,(x - c_x) + \sin\theta\,(y - c_y) + c_x \\"
            r"s_y &= -\sin\theta\,(x - c_x) + \cos\theta\,(y - c_y) + c_y"
            r"\end{aligned}",
      [("x,\\, y", "the output pixel being filled"),
       ("s_x,\\, s_y", "the position in the input it is read from"),
       ("c_x,\\, c_y", "the centre of the image, which the rotation is about"),
       (r"\theta", "the rotation angle in radians")],
      "wolberg1990"),
     ("p", "Note that the signs are those of a rotation by minus theta: "
           "because the mapping runs backwards from output to input, it must "
           "be the inverse of the rotation being applied."),
     ("h", "Filling rather than clamping"),
     ("p", "A source position outside the image is filled with the paper "
           "value rather than clamped to the nearest edge pixel. Clamping "
           "would smear whatever lies on the boundary across the whole "
           "corner, producing streaks that the ink detector would then have "
           "to cope with."),
     ("h", "The edge convention, and why it was determined experimentally"),
     ("p", "Deciding whether a source position lies inside the image is less "
           "obvious than it looks. If a pixel is a point at an integer "
           "coordinate, the image spans 0 to size minus 1. If a pixel is a "
           "unit square centred on its index, the image spans minus a half to "
           "size minus a half. This implementation uses the second, area, "
           "convention."),
     ("note", "The convention was not guessed. Replacing a library rotation "
              "with this one was verified by comparing the two on a test "
              "page: with the area convention the two agreed to a maximum "
              "difference of one grey level out of 255, while the other "
              "plausible convention differed by up to 245 along the edges. "
              "A difference of that size is not a rounding detail, it is a "
              "row of pixels in the wrong place."),
   ]),

 "Warp": dict(
   inputs=[("img", "the source image"),
           ("coeffs", "six numbers for an affine transform, or eight for a "
                      "perspective one"),
           ("fill", "the background value")],
   output="The transformed image.",
   theory=[
     ("h", "What it is for"),
     ("p", "One function covering every straight-line-preserving transform "
           "the pipeline needs: scaling, shearing, rotation, and the "
           "keystone effect of a page photographed from an angle. It is used "
           "mainly to generate training variations of a crop."),
     ("h", "Background: affine and perspective"),
     ("p", "An affine transform is any combination of scaling, rotation, "
           "shear and translation. It keeps parallel lines parallel, so a "
           "rectangle becomes a parallelogram. A perspective transform, also "
           "called a homography, additionally allows parallel lines to "
           "converge, so a rectangle becomes any four-sided shape. That is "
           "what happens when a flat page is photographed obliquely: the far "
           "edge appears shorter than the near one."),
     ("h", "The mapping"),
     ("eq", r"s_x = \frac{a\,x + b\,y + c}{g\,x + h\,y + 1}, \qquad "
            r"s_y = \frac{d\,x + e\,y + f}{g\,x + h\,y + 1}",
      [("x,\\, y", "the output pixel being filled"),
       ("s_x,\\, s_y", "the source position it is read from"),
       ("a,\\, b,\\, d,\\, e", "the linear part: scaling, rotation and shear"),
       ("c,\\, f", "the translation"),
       ("g,\\, h", "the perspective part; both zero gives a purely affine "
                   "transform, since the divisor is then 1 everywhere")],
      "wolberg1990"),
     ("p", "The divisor is what distinguishes the two cases. When g and h are "
           "zero it equals one at every pixel and the mapping is linear; "
           "otherwise it varies across the image, so equal steps in the "
           "output correspond to unequal steps in the input, which is exactly "
           "the foreshortening that perspective produces."),
     ("h", "The half-pixel convention"),
     ("p", "The coefficients act on the \\emph{centre} of the output pixel, "
           "so a half is added to the output coordinate before the mapping is "
           "applied and subtracted from the result afterwards. This follows "
           "from the same pixels-are-squares reasoning as in `_Weights`."),
     ("note", "Like the rotation convention, this was settled by "
              "measurement rather than argument: with the half-pixel offsets "
              "in place the output matched the library implementation it "
              "replaced on zero pixels of difference across a test page, and "
              "without them the entire image was displaced by half a pixel."),
   ]),

 "Blur": dict(
   inputs=[("img", "the image to blur"),
           ("sigma", "the standard deviation of the gaussian, in pixels, "
                     "which sets how far the blurring reaches")],
   output="The blurred image, in the input's number format.",
   theory=[
     ("h", "What it is for"),
     ("p", "Two uses. It smooths the random displacement fields used for "
           "training augmentation, which is what turns random noise into a "
           "coherent bending of the strokes. And it softens the edge of "
           "synthesised ink, so that a drawn stroke has the gradual edge of "
           "a real pen rather than a hard step, which matters because the "
           "networks judging that output were trained on photographs of real "
           "ink."),
     ("h", "Background: what a gaussian blur is"),
     ("p", "Each output pixel is a weighted average of its neighbourhood, "
           "with weights following the bell-shaped gaussian curve: strongest "
           "at the centre and falling away smoothly with distance. Compared "
           "with a flat average it produces no visible edge at the limit of "
           "the window, because the weights reach zero gradually."),
     ("h", "Why it is done as two one-dimensional passes"),
     ("p", "A two-dimensional gaussian factorises exactly into the product of "
           "two one-dimensional gaussians:"),
     ("eq", r"G(x,y) \;=\; \exp\!\left(-\frac{x^2 + y^2}{2\sigma^2}\right) "
            r"\;=\; \exp\!\left(-\frac{x^2}{2\sigma^2}\right)\cdot"
            r"\exp\!\left(-\frac{y^2}{2\sigma^2}\right) \;=\; G(x)\,G(y)",
      [("G(x,y)", "the two-dimensional weight at an offset of x columns and "
                  "y rows from the centre"),
       (r"\sigma", "the standard deviation, which sets the width of the "
                   "bell"),
       ("G(x),\\, G(y)", "the one-dimensional weights along each axis")],
      None),
     ("p", "So blurring down the rows and then across the columns gives "
           "exactly the same answer as a single two-dimensional pass, at a "
           "cost of 2k multiplications per pixel instead of k squared. This "
           "is the same separability argument as for `Resize`."),
     ("h", "Truncation and normalisation"),
     ("p", "The gaussian never reaches zero, so the kernel is cut off at "
           "about three standard deviations, where under one per cent of the "
           "total weight remains. The remaining weights are then divided by "
           "their sum so they total one, which keeps the overall brightness "
           "of the image unchanged; without that, blurring would darken the "
           "image by the fraction of weight that was discarded."),
   ]),

 "Elastic": dict(
   inputs=[("img", "the image to distort"),
           ("rng", "a seeded random number generator, so a distortion can be "
                   "reproduced"),
           ("alpha", "how far pixels are moved, in pixels"),
           ("sigma", "how smoothly the displacement varies, in pixels")],
   output="A locally bent version of the image.",
   theory=[
     ("h", "What it is for"),
     ("p", "Training augmentation. A network trained on a fixed set of "
           "examples can memorise them; showing it plausible variations of "
           "each example forces it to learn the general pattern instead. For "
           "handwriting the most useful variation is the one that looks like "
           "the same person writing the same words again, which is not a "
           "rotation or a scaling but a local bending of the strokes "
           "\\cite{simard2003}."),
     ("h", "How it works"),
     ("eq", r"\begin{aligned}"
            r"d_x &= \alpha \cdot \mathrm{Blur}\big(U(-1,1),\, \sigma\big) \\"
            r"d_y &= \alpha \cdot \mathrm{Blur}\big(U(-1,1),\, \sigma\big) \\"
            r"\mathrm{out}[y,x] &= \mathrm{sample}\big(\mathrm{img},\; "
            r"y + d_y[y,x],\; x + d_x[y,x]\big)"
            r"\end{aligned}",
      [("d_x,\\, d_y", "the displacement fields: how far the pixel at each "
                       "position moves horizontally and vertically"),
       ("U(-1,1)", "an array of independent random numbers drawn uniformly "
                   "between minus one and one, one per pixel"),
       (r"\alpha", "the displacement magnitude in pixels"),
       (r"\sigma", "the blur width, which controls how smoothly the "
                   "displacement varies from pixel to pixel"),
       (r"\mathrm{sample}", "bilinear interpolation, since the displaced "
                            "positions are fractional")],
      "simard2003"),
     ("h", "Why the smoothing is the essential step"),
     ("p", "Without the blur, each pixel would move independently of its "
           "neighbours and the image would be scrambled into noise: strokes "
           "would break into disconnected dots. Blurring the displacement "
           "field means nearby pixels move by nearly the same amount, so a "
           "stroke moves as a stroke, bending and stretching while staying "
           "continuous. The width of the blur sets the scale of the bending: "
           "small gives wobbly strokes, large gives a gentle warp of the "
           "whole crop."),
     ("p", "The two parameters interact. A large displacement with a small "
           "blur tears the writing apart; a large displacement with a large "
           "blur is nearly a rigid shift. Useful augmentation lives between "
           "those extremes."),
   ]),

 "DrawPolyline": dict(
   inputs=[("img", "the array to draw into, which is modified in place"),
           ("pts", "the path as a list of (x, y) points"),
           ("value", "the ink level to write, 0 being black"),
           ("width", "the pen width in pixels")],
   output="The same array, returned for convenience when chaining calls.",
   theory=[
     ("h", "What it is for"),
     ("p", "Drawing a pen stroke into an image. It is used to render "
           "synthesised handwriting so that the networks can judge it, to "
           "redraw ink at a uniform width inside stroke normalisation, and in "
           "the measurement harness to compare two sets of geometry. Because "
           "all three use the same function, they all see ink laid down the "
           "same way."),
     ("h", "The idea: fill by distance rather than by stepping along the "
           "line"),
     ("p", "The familiar way to draw a line is to step along it filling "
           "pixels. That is fast but makes a round pen awkward, because the "
           "ends and the corners between segments have to be handled "
           "separately or they come out square and notched."),
     ("p", "Instead, every pixel near the segment is tested by how far it "
           "lies from the nearest point \\emph{on} the segment, and filled if "
           "that distance is within half the pen width. The nearest point is "
           "found by projecting onto the segment and clamping to its ends:"),
     ("eq", r"t \;=\; \mathrm{clamp}\!\left(\frac{(P-A)\cdot(B-A)}"
            r"{\lVert B-A \rVert^2},\, 0,\, 1\right), \qquad "
            r"Q \;=\; A + t\,(B-A)",
      [("A,\\, B", "the two ends of the segment being drawn"),
       ("P", "the pixel being tested"),
       ("t", "how far along the segment the nearest point lies, 0 at A and 1 "
             "at B"),
       ("Q", "the nearest point on the segment to P"),
       (r"(P-A)\cdot(B-A)", "the dot product, which measures how far along "
                            "the segment's direction P lies")],
      None),
     ("p", "A pixel is then filled when the distance from P to Q is at most "
           "half the pen width."),
     ("h", "Why the clamp gives round ends and joins for free"),
     ("p", "Without the clamp, t could fall outside the range 0 to 1 and the "
           "distance would be measured to the infinite line through A and B, "
           "so ink would extend beyond the ends of the segment. Clamping "
           "means that for a pixel beyond an end, the nearest point is the "
           "end itself, so the filled region there is a disc of radius half "
           "the pen width. That is a round cap. At a corner between two "
           "segments both discs overlap, which is a round join. No separate "
           "handling is needed for either."),
     ("h", "Why it is affordable"),
     ("p", "Testing every pixel in the image against every segment would be "
           "hopeless. Work is restricted to the bounding box of each segment, "
           "expanded by the pen radius, since no pixel outside that box can "
           "possibly be within half a pen width of it. The cost is therefore "
           "proportional to the amount of ink drawn rather than to the size "
           "of the canvas."),
     ("p", "A segment of zero length, which happens when a path contains a "
           "repeated point or consists of a single point, falls back to plain "
           "distance from that point, which draws a dot."),
   ]),

 "EstimateSlantDeg": dict(
   inputs=[("ink", "a mask of one line of writing")],
   output="The slant in degrees, positive for the usual forward lean. Returns "
          "zero when there are fewer than fifty ink pixels.",
   theory=[
     ("h", "What it is for"),
     ("p", "Two things. The glyph extractor removes the slant before storing "
           "a letter, so that stored letters are upright and can be compared "
           "with one another; and the synthesiser re-applies the writer's own "
           "slant when drawing, because how far somebody leans their writing "
           "is one of the most visible features of their hand."),
     ("h", "Background: what shearing is"),
     ("p", "Shearing slides each row of an image sideways by an amount "
           "proportional to its height, which tilts upright strokes without "
           "changing the height of anything. It is the operation that turns "
           "upright writing into slanted writing, so undoing it is how "
           "slanted writing is made upright."),
     ("h", "The method: try angles and score the result"),
     ("p", "For each candidate angle the ink is sheared by that amount and "
           "the ink in each column is counted, giving a column profile. The "
           "angle whose profile is the \\emph{sharpest} is the answer "
           "\\cite{vinciarelli2001}:"),
     ("eq", r"x' \;=\; x - \tan(\theta)\,(y - \bar{y}), \qquad "
            r"\mathrm{score}(\theta) \;=\; \sum_j \mathrm{prof}_\theta[j]^2",
      [("x,\\, y", "the coordinates of an ink pixel"),
       ("x'", "its column after shearing by the trial angle"),
       (r"\bar{y}", "the mean row of all the ink, which is the height the "
                    "shear pivots about"),
       (r"\theta", "the trial angle"),
       (r"\mathrm{prof}_\theta[j]", "how many ink pixels land in column j "
                                    "after that shear"),
       (r"\mathrm{score}(\theta)", "the sharpness being maximised")],
      "vinciarelli2001"),
     ("h", "Why squaring the profile measures sharpness"),
     ("p", "This is the key step and it is worth being precise about. The "
           "total amount of ink does not change when the image is sheared, so "
           "the sum of the profile is the same for every angle; only its "
           "\\emph{distribution} changes. For a fixed total, the sum of "
           "squares is largest when the quantity is concentrated in a few "
           "entries and smallest when it is spread evenly. Ten pixels in one "
           "column score a hundred; one pixel in each of ten columns scores "
           "ten."),
     ("p", "So maximising the sum of squares finds the angle at which the "
           "ink stacks into the fewest, tallest columns, which is the angle "
           "at which the vertical strokes of the writing are truly vertical. "
           "The same reasoning underlies `SkewScore`, applied to rows instead "
           "of columns."),
     ("h", "Why coordinates are sheared rather than the image"),
     ("p", "The score needs only the column each ink pixel lands in, not an "
           "image. Shearing the coordinates and counting them is therefore "
           "enough, and it avoids interpolating a whole image once per "
           "candidate angle. With sixty candidate angles that is the "
           "difference between a measurement that is free and one that "
           "dominates the runtime."),
     ("h", "Resolution and the sign"),
     ("p", "The search runs from minus 45 to plus 45 degrees in steps of 1.5 "
           "degrees, which bounds how precisely the answer can be reported. "
           "The result is negated before being returned so that a positive "
           "number means the ordinary forward lean, which makes the "
           "deslanting step read naturally as subtracting the tangent of the "
           "slant times the height."),
     ("p", "Fewer than fifty ink pixels is treated as too little to measure "
           "and zero is returned, since on that much ink the profile is "
           "dominated by where the individual pixels happen to fall."),
   ]),

 "PolylineLength": dict(
   inputs=[("p", "a path as a list or array of (x, y) points")],
   output="The total distance walked along the path.",
   theory=[
     ("h", "What it is for"),
     ("p", "A size measure for pen strokes. Its main use is rejecting "
           "fragments: a piece cut out of a line of writing that contains far "
           "less pen path than a real example of that letter is a fragment "
           "rather than a letter, whatever its bounding box suggests."),
     ("h", "How it works"),
     ("eq", r"L \;=\; \sum_{i} \sqrt{(x_{i+1}-x_i)^2 + (y_{i+1}-y_i)^2}",
      [("L", "the total path length, also called the arc length"),
       ("x_i,\\, y_i", "the coordinates of the i-th point on the path")],
      None),
     ("p", "Each term is the straight-line distance between two consecutive "
           "points, by Pythagoras, and the total is their sum. It is computed "
           "as whole-array arithmetic: the differences between consecutive "
           "points, then their lengths, then one sum."),
     ("h", "Why path length and not bounding box"),
     ("p", "A fragment can have a deceptively large bounding box. Half of a "
           "letter o spans nearly the full width and height of the whole "
           "letter while containing only half the ink, so a box-based test "
           "would accept it. Path length measures how much pen movement the "
           "shape actually represents, which is what distinguishes half a "
           "letter from a whole one."),
   ]),
}
