"""The written half of the code reference.  codedoc.py supplies the skeleton.

FUNCS[name] = dict(
    inputs = "what goes in",
    output = "what the return value means",
    cite   = "reference keys, comma separated",
    body   = [ "paragraph", ("formula", ["line", ...]), ("sub", "heading") ],
)
"""

PREAMBLE = (
    "D.R. Braune, u23539195.  A function by function reference for the "
    "handwriting reproduction pipeline.  Every signature, default value, "
    "return expression and call relationship in this document is read "
    "directly out of the source with Python's own parser, so it cannot "
    "disagree with the code.  The explanations are written by hand.  Where an "
    "algorithm is a known published method it is cited at the point of use, "
    "and the reference list at the end gives the source; where a choice is "
    "mine it is described as such."
)

HOWTO = [
 "Each entry begins with a table. **Signature** is the real signature "
 "including default values. **Defined** gives the file and line range. "
 "**Inputs** describes what the arguments mean and what shape or type they "
 "are expected to be. **Returns** lists every distinct return expression as "
 "it appears in the code, so a function with several exit paths shows all of "
 "them. **Calls** and **Called by** are the project's own call graph, which "
 "is how you can follow a value from the camera to the g-code file.",

 "After the table comes the explanation: what the function is for, how it "
 "works, the mathematics where there is any, and the failure cases or edge "
 "conditions that the code guards against. Formulas are written in plain "
 "text in a monospace block, using `*` for multiplication and `^` for "
 "powers, so that they can be read without a symbol font.",

 "A note on conventions used throughout the pipeline. Images are numpy "
 "arrays indexed `[row, column]`, that is `[y, x]`, with y increasing "
 "**downwards**, which is the normal image convention. Pen trajectories are "
 "lists of `(x, y)` points in millimetres with y increasing **upwards**, "
 "which is the normal plotter convention. The flip between the two happens "
 "in exactly two places and both are noted where they occur. Glyph outlines "
 "are stored in x-height units, where 1.0 is the height of a lower case "
 "letter body, so a stored letter can be redrawn at any physical size.",

 "Nothing in the pipeline uses an image processing or machine learning "
 "library. numpy is used for array arithmetic only. Every threshold, "
 "filter, morphological operation, network layer, loss and optimiser in "
 "this document is implemented in the files described here.",
]

REFERENCES = {
 "Otsu1979":
 "N. Otsu, \"A Threshold Selection Method from Gray-Level Histograms\", "
 "IEEE Transactions on Systems, Man, and Cybernetics, vol. 9, no. 1, "
 "pp. 62-66, 1979.",

 "Crow1984":
 "F. C. Crow, \"Summed-Area Tables for Texture Mapping\", ACM SIGGRAPH "
 "Computer Graphics, vol. 18, no. 3, pp. 207-212, 1984.",

 "ZhangSuen1984":
 "T. Y. Zhang and C. Y. Suen, \"A Fast Parallel Algorithm for Thinning "
 "Digital Patterns\", Communications of the ACM, vol. 27, no. 3, "
 "pp. 236-239, 1984.",

 "He2008":
 "L. He, Y. Chao and K. Suzuki, \"A Run-Based Two-Scan Labeling Algorithm\", "
 "IEEE Transactions on Image Processing, vol. 17, no. 5, pp. 749-756, 2008.",

 "Tarjan1975":
 "R. E. Tarjan, \"Efficiency of a Good But Not Linear Set Union Algorithm\", "
 "Journal of the ACM, vol. 22, no. 2, pp. 215-225, 1975.",

 "Postl1986":
 "W. Postl, \"Detection of Linear Oblique Structures and Skew Scan in "
 "Digitized Documents\", Proc. 8th International Conference on Pattern "
 "Recognition, pp. 687-689, 1986.",

 "Baird1987":
 "H. S. Baird, \"The Skew Angle of Printed Documents\", Proc. SPSE 40th "
 "Annual Conference, pp. 14-21, 1987.",

 "Vinciarelli2001":
 "A. Vinciarelli and J. Luettin, \"A New Normalization Technique for "
 "Cursive Handwritten Words\", Pattern Recognition Letters, vol. 22, "
 "no. 9, pp. 1043-1050, 2001.",

 "Arvanitopoulos2014":
 "N. Arvanitopoulos and S. Susstrunk, \"Seam Carving for Text Line "
 "Extraction on Color and Grayscale Historical Manuscripts\", Proc. 14th "
 "International Conference on Frontiers in Handwriting Recognition, "
 "pp. 726-731, 2014.",

 "Avidan2007":
 "S. Avidan and A. Shamir, \"Seam Carving for Content-Aware Image "
 "Resizing\", ACM Transactions on Graphics, vol. 26, no. 3, 2007.",

 "Bellman1954":
 "R. Bellman, \"The Theory of Dynamic Programming\", Bulletin of the "
 "American Mathematical Society, vol. 60, no. 6, pp. 503-515, 1954.",

 "Wolberg1990":
 "G. Wolberg, Digital Image Warping, IEEE Computer Society Press, 1990. "
 "(separable resampling, filter support when minifying, inverse mapping)",

 "Simard2003":
 "P. Y. Simard, D. Steinkraus and J. C. Platt, \"Best Practices for "
 "Convolutional Neural Networks Applied to Visual Document Analysis\", "
 "Proc. 7th International Conference on Document Analysis and Recognition, "
 "pp. 958-963, 2003.  (elastic distortion)",

 "DouglasPeucker1973":
 "D. H. Douglas and T. K. Peucker, \"Algorithms for the Reduction of the "
 "Number of Points Required to Represent a Digitized Line or its "
 "Caricature\", Cartographica, vol. 10, no. 2, pp. 112-122, 1973.",

 "Graves2006":
 "A. Graves, S. Fernandez, F. Gomez and J. Schmidhuber, \"Connectionist "
 "Temporal Classification: Labelling Unsegmented Sequence Data with "
 "Recurrent Neural Networks\", Proc. 23rd International Conference on "
 "Machine Learning, pp. 369-376, 2006.",

 "Hochreiter1997":
 "S. Hochreiter and J. Schmidhuber, \"Long Short-Term Memory\", Neural "
 "Computation, vol. 9, no. 8, pp. 1735-1780, 1997.",

 "Shi2017":
 "B. Shi, X. Bai and C. Yao, \"An End-to-End Trainable Neural Network for "
 "Image-Based Sequence Recognition and Its Application to Scene Text "
 "Recognition\", IEEE Transactions on Pattern Analysis and Machine "
 "Intelligence, vol. 39, no. 11, pp. 2298-2304, 2017.  (CRNN)",

 "Chellapilla2006":
 "K. Chellapilla, S. Puri and P. Simard, \"High Performance Convolutional "
 "Neural Networks for Document Processing\", Proc. 10th International "
 "Workshop on Frontiers in Handwriting Recognition, 2006.  (unfolding a "
 "convolution into a matrix product)",

 "Ioffe2015":
 "S. Ioffe and C. Szegedy, \"Batch Normalization: Accelerating Deep Network "
 "Training by Reducing Internal Covariate Shift\", Proc. 32nd International "
 "Conference on Machine Learning, pp. 448-456, 2015.",

 "Kingma2015":
 "D. P. Kingma and J. Ba, \"Adam: A Method for Stochastic Optimization\", "
 "Proc. 3rd International Conference on Learning Representations, 2015.",

 "Loshchilov2019":
 "I. Loshchilov and F. Hutter, \"Decoupled Weight Decay Regularization\", "
 "Proc. 7th International Conference on Learning Representations, 2019.  "
 "(AdamW)",

 "Tieleman2012":
 "T. Tieleman and G. Hinton, \"Lecture 6.5 - RMSProp\", COURSERA: Neural "
 "Networks for Machine Learning, 2012.",

 "Levenshtein1966":
 "V. I. Levenshtein, \"Binary Codes Capable of Correcting Deletions, "
 "Insertions, and Reversals\", Soviet Physics Doklady, vol. 10, no. 8, "
 "pp. 707-710, 1966.",

 "Viterbi1967":
 "A. J. Viterbi, \"Error Bounds for Convolutional Codes and an "
 "Asymptotically Optimum Decoding Algorithm\", IEEE Transactions on "
 "Information Theory, vol. 13, no. 2, pp. 260-269, 1967.",

 "Bresenham1965":
 "J. E. Bresenham, \"Algorithm for Computer Control of a Digital Plotter\", "
 "IBM Systems Journal, vol. 4, no. 1, pp. 25-30, 1965.",

 "Srivastava2014":
 "N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever and R. "
 "Salakhutdinov, \"Dropout: A Simple Way to Prevent Neural Networks from "
 "Overfitting\", Journal of Machine Learning Research, vol. 15, "
 "pp. 1929-1958, 2014.",

 "Glorot2010":
 "X. Glorot and Y. Bengio, \"Understanding the Difficulty of Training Deep "
 "Feedforward Neural Networks\", Proc. 13th International Conference on "
 "Artificial Intelligence and Statistics, pp. 249-256, 2010.",

 "He2015":
 "K. He, X. Zhang, S. Ren and J. Sun, \"Delving Deep into Rectifiers: "
 "Surpassing Human-Level Performance on ImageNet Classification\", Proc. "
 "IEEE International Conference on Computer Vision, pp. 1026-1034, 2015.",

 "Pascanu2013":
 "R. Pascanu, T. Mikolov and Y. Bengio, \"On the Difficulty of Training "
 "Recurrent Neural Networks\", Proc. 30th International Conference on "
 "Machine Learning, pp. 1310-1318, 2013.  (gradient clipping)",

 "Marti2002":
 "U.-V. Marti and H. Bunke, \"The IAM-Database: An English Sentence "
 "Database for Offline Handwriting Recognition\", International Journal on "
 "Document Analysis and Recognition, vol. 5, pp. 39-46, 2002.",
}

MODULES = {
 "paths.py": dict(intro=[
   "Configuration only: where the data sits on disk, and who the thirteen "
   "writers are. It contains no algorithm. It exists as its own file because "
   "every other module needs it, and putting it inside one of them would "
   "force the others to import that module merely to find out where the "
   "weights live.",
   "Every path is derived from the location of this file, so the project can "
   "be moved without editing anything. The data deliberately sits beside the "
   "code rather than inside it, so that none of it is duplicated into the "
   "repository.",
 ]),

 "segment.py": dict(intro=[
   "Two things live here: the image toolkit that the whole pipeline shares, "
   "and the page segmenter that uses it. They are in one file because the "
   "toolkit has no meaning on its own and the segmenter is its heaviest "
   "user; the other modules import the handful of toolkit functions they "
   "need from here.",
   "**Functional units 2.1 and 2.2, requirement R2.** The job is: one "
   "photograph of a handwritten page in, one clean deskewed image per line "
   "of writing out. Nothing in this file opens a file or uses an image "
   "library; it takes arrays and returns arrays.",
   "The toolkit is arranged from the bottom up, because each layer is built "
   "from the one before it: rectangle sums, then morphology built from those "
   "sums, then thresholding, then connected components, then thinning, then "
   "resampling, then geometric transforms, then measurement. The segmenter "
   "then reads as a sequence: condition the page, remove what was printed on "
   "it, find where the lines are, cut between them, decide which line owns "
   "each piece of ink, and build a crop per line.",
   "A note on why so much is built on rectangle sums. A local average over "
   "a window of radius r normally costs (2r+1)^2 additions per pixel. With a "
   "summed-area table it costs four lookups regardless of r [Crow1984]. "
   "Since the background estimate for ink detection uses a radius of up to "
   "40 pixels on a multi-megapixel photograph, this is the difference "
   "between a usable program and an unusable one.",
 ]),
}

FUNCS = {

# ------------------------------------------------------------------ paths
 "PersonalWriterPages": dict(
   inputs="`writer`, the lower case name of one photographed writer, for "
          "example \"owen\".",
   output="A generator of `(imagePath, labelPath)` pairs, one per page, in "
          "sorted filename order. The label file is the typed transcript of "
          "that page, one line of text per line of writing.",
   body=[
     "Finds the photographed pages belonging to one writer. The folder "
     "layout is fixed by convention: `PersonalDataset/<Writer>/Pages/` holds "
     "the JPEGs and `PersonalDataset/<Writer>/Labels/` holds a "
     "`<stem>_labels.txt` beside each one.",
     "Sorted order matters and is not cosmetic. The transcript file is "
     "matched to the image by filename, and several measurement scripts pair "
     "the n-th detected line crop with the n-th transcript line, so a "
     "non-deterministic directory order would silently mispair labels.",
     "It is a generator rather than a list so that a caller processing a "
     "large set of pages does not hold every path in memory, and so that a "
     "caller can stop early.",
   ]),

# ---------------------------------------------------- toolkit: window sums
 "Integral": dict(
   inputs="`img`, a 2-D numeric array.",
   output="A `(H+1, W+1)` float64 array `ii`, where `ii[y, x]` is the sum of "
          "everything strictly above and strictly left of pixel `(y, x)`. "
          "The extra zero row and column mean no boundary test is needed "
          "when indexing.",
   cite="Crow1984",
   body=[
     "The summed-area table, which every local average and every "
     "morphological operation in this file is built on. Each cell holds the "
     "running total of the rectangle from the origin to that cell:",
     ("formula", ["ii[y, x] = sum over i < y, j < x of img[i, j]"]),
     "It is computed as a cumulative sum down the rows followed by a "
     "cumulative sum across the columns, which is two linear passes.",
     "The reason it is worth building is what it costs to query afterwards. "
     "The sum inside the rectangle with corners (y1, x1) and (y2, x2) is",
     ("formula", ["S = ii[y2, x2] - ii[y1, x2] - ii[y2, x1] + ii[y1, x1]"]),
     "The corner term is added back because the two edge subtractions each "
     "removed it once. That is four array lookups for a window of any size, "
     "so the cost of a local average does not grow with the radius. For the "
     "40-pixel background radius used by `InkMask` on a page photograph, the "
     "naive form would be 6561 additions per pixel.",
     "The output is float64 rather than the input dtype on purpose: the "
     "running total over a few million 8-bit pixels overflows a 32-bit "
     "integer, and silently wrapping would corrupt every window sum on the "
     "page.",
   ]),

 "BoxSum": dict(
   inputs="`img`, a 2-D array. `ry`, `rx`, the window radius in rows and "
          "columns, so the window is `(2*ry+1)` by `(2*rx+1)`.",
   output="An array the same shape as `img`, where each element is the sum "
          "of the input inside the window centred on that pixel.",
   cite="Crow1984",
   body=[
     "Applies the four-lookup rectangle query of `Integral` at every pixel "
     "at once. The window bounds are built as vectors and clipped to the "
     "image, then used to index the table, so the whole operation is a "
     "handful of array expressions with no Python loop over pixels.",
     "Windows near the border hang off the image. They are clipped rather "
     "than reflected or wrapped, which means the sum near an edge covers "
     "fewer pixels than in the middle. That is corrected by dividing by "
     "`BoxCount` rather than by the nominal window area, which is what "
     "`BoxMean` does.",
   ]),

 "BoxCount": dict(
   inputs="Same arguments as `BoxSum`: an array for its shape, and the "
          "window radii.",
   output="An integer array giving, for each pixel, how many pixels its "
          "window actually covered after clipping to the image.",
   body=[
     "The companion of `BoxSum`. Computed as the product of the clipped "
     "height and the clipped width of the window at each position, which is "
     "separable and so costs two 1-D computations rather than a 2-D one.",
     "It has two uses. Dividing `BoxSum` by it gives an unbiased local mean "
     "at the borders. And `Erode` compares the window sum against it to ask "
     "whether *every* covered pixel was set, which is the only correct way "
     "to erode near an edge: comparing against the nominal area would erode "
     "the border of every shape that touches the image edge.",
   ]),

 "BoxMean": dict(
   inputs="`img`, a 2-D array. `r`, a single radius used for both axes.",
   output="A float array the same shape as `img`, the local mean over the "
          "`(2r+1)` square window centred on each pixel.",
   body=[
     "The local average, `BoxSum / BoxCount`. This is the background "
     "estimate used by `InkMask`: comparing a pixel against the average of "
     "its own neighbourhood rather than against one number for the whole "
     "page is what makes ink detection survive uneven lighting, a shadow "
     "across the paper, or a photograph that is brighter on one side.",
     "It is also used as a cheap smoother in several places where a "
     "gaussian would be overkill, for example when deciding whether a region "
     "is solidly dark.",
   ]),

# ----------------------------------------------------- toolkit: morphology
 "Dilate": dict(
   inputs="`m`, a boolean mask. `r`, the radius to grow by.",
   output="A boolean mask in which a pixel is true if any true pixel of the "
          "input lies within its window.",
   body=[
     "Morphological dilation with a square structuring element, expressed "
     "as a window sum rather than as a neighbourhood scan:",
     ("formula", ["dilate(m)[p] = ( BoxSum(m, r, r)[p] > 0 )"]),
     "Asking whether the window sum is positive is exactly asking whether "
     "any neighbour was set, so dilation inherits the property that its cost "
     "does not depend on the radius.",
     "Used to close the gaps between the separate strokes of one letter "
     "before the connected component pass, so that an 'i' and its dot can be "
     "reasoned about as one piece.",
   ]),

 "Erode": dict(
   inputs="`m`, a boolean mask. `r`, the radius to shrink by.",
   output="A boolean mask in which a pixel is true only if every covered "
          "pixel of its window was true in the input.",
   body=[
     "Morphological erosion, the dual of dilation:",
     ("formula", ["erode(m)[p] = ( BoxSum(m, r, r)[p] == BoxCount(r, r)[p] )"]),
     "The comparison is against `BoxCount` and not against `(2r+1)^2`, which "
     "matters at the image border where the window is clipped. If the "
     "nominal area were used, no pixel within r of the edge could ever "
     "satisfy the test and a band around the whole page would be eroded "
     "away.",
   ]),

 "Close": dict(
   inputs="`m`, a boolean mask. `r`, the radius.",
   output="A boolean mask with gaps narrower than the structuring element "
          "filled in, and the original stroke thickness preserved.",
   body=[
     "A morphological closing: dilate then erode by the same radius. The "
     "dilation joins anything separated by less than the window; the erosion "
     "then gives back the thickness the dilation added. The net effect is "
     "that a gap narrower than the kernel is bridged while a thin stroke "
     "stays thin, which is why closing is used to heal a broken pen stroke "
     "rather than a plain dilation.",
     "This is used inside `BinarizeLine` to repair the pinholes left in ink "
     "by thresholding a photograph, before anything downstream tries to "
     "trace the stroke.",
   ]),

 "AnyWithinRadius": dict(
   inputs="`mask`, a boolean array. `r`, the radius. `axis`, which axis to "
          "work along.",
   output="A boolean array, true where any element within r positions along "
          "that one axis was true.",
   body=[
     "A one-dimensional dilation. Where `Dilate` grows a shape in both "
     "directions at once, this grows it along a single axis, which is what "
     "is wanted when the thing being tolerated is directional.",
     "Both uses in this file are of that kind. `RemovePrintedStructure` "
     "widens a detected ruled line slightly along its own direction so that "
     "a line which is not perfectly horizontal is still removed along its "
     "whole length, and `PaperRegionMask` shrinks back from the paper edge "
     "by a margin in each axis separately.",
   ]),

# ----------------------------------------------------------- toolkit: runs
 "LongRunMask": dict(
   inputs="`m`, a boolean mask. `n`, the minimum run length. `axis`, the "
          "direction the run is measured along.",
   output="A boolean mask, true at every pixel that belongs to an unbroken "
          "run of at least n set pixels along that axis.",
   body=[
     "Finds ink that continues in a straight line for a long way, which is "
     "how a printed rule and the binder edge are told apart from "
     "handwriting: a pen stroke of 60 to 80 unbroken pixels in one direction "
     "is rare, a ruled line is nothing else.",
     "The implementation uses two cumulative-sum passes. The first finds "
     "where a long run *starts*: the window of n pixels beginning at i is "
     "entirely set exactly when the count inside it equals its length,",
     ("formula", ["starts[i] = ( c[i+n] - c[i] == n )",
                  "   where c is the prefix sum of the mask along the axis"]),
     "The second pass turns starts into coverage. A run of length n covering "
     "pixel j must begin somewhere in the window `[j-n+1, j]`, clamped to "
     "the valid range `[0, L-n]`, so a second prefix sum over `starts` "
     "answers \"is there any start in that range\" with one subtraction:",
     ("formula", ["lo = max(0, j-n+1),   hi = min(j, L-n)",
                  "covered[j] = ( cs[hi+1] - cs[lo] > 0 )  and  hi >= lo"]),
     "The reason for the double prefix sum rather than a direct scan is "
     "cost: it is linear in the number of pixels regardless of n, where a "
     "window scan would be proportional to n as well.",
     "The axis argument is handled by moving the axis of interest to the "
     "front with `moveaxis`, running the fixed code, and moving it back, so "
     "rows and columns share one implementation.",
     "**Note for maintenance.** `RowRunLength`, defined two functions below, "
     "already assigns every pixel the length of its own run, which makes "
     "this function equivalent to `RowRunLength(m) >= n` with a transpose "
     "for the other axis. That form was verified identical across 4800 "
     "randomised masks and runs about twice as fast on a page sized array, "
     "because it uses one labelling pass and a bincount instead of two "
     "prefix-sum arrays and clamped index vectors.",
   ]),

 "RowRunLength": dict(
   inputs="`mask`, a 2-D boolean array. Runs are measured along rows.",
   output="An integer array the same shape, where each set pixel holds the "
          "length of the unbroken horizontal run it belongs to, and each "
          "clear pixel holds zero.",
   cite="He2008",
   body=[
     "Labels every horizontal run of ink and then hands each pixel the "
     "length of its own run, in three vectorised steps rather than by "
     "scanning.",
     "First, run starts are found as the places where the row steps from "
     "clear to set. Second, a cumulative sum over those starts, taken over "
     "the flattened array, numbers the runs consecutively; the numbering is "
     "then cleared at non-ink pixels so that only ink carries a run number. "
     "Third, `np.bincount` counts how many pixels carry each number, which "
     "is the length of each run, and indexing those counts by the run number "
     "image gives every pixel its own run length.",
     "The flattened cumulative sum is what makes this work without a loop, "
     "and it is safe because a run cannot span two rows: a row always ends "
     "with a clear boundary in the padded difference, so the numbering "
     "cannot leak from the end of one row into the start of the next.",
   ]),

# --------------------------------------------------- toolkit: thresholding
 "Otsu": dict(
   inputs="`v`, an array of pixel values, either integer or float. `hi`, the "
          "upper limit used when binning a float image; defaults to 1.2, "
          "which suits the ratio image produced by `InkMask` because nothing "
          "above that carries ink.",
   output="For an integer image, the threshold as a bin index in 0..255. For "
          "a float image, the value at the centre of the chosen bin. In both "
          "cases, pixels below the threshold are ink.",
   cite="Otsu1979",
   body=[
     "Chooses the brightness that best separates ink from paper, with no "
     "fixed constant, which is what lets the same code handle a dark "
     "photograph and a bright one.",
     "Otsu's criterion is to pick the threshold t that maximises the "
     "variance *between* the two groups it creates. Writing wB(t) for the "
     "fraction of pixels at or below t, wF(t) = 1 - wB(t) for the rest, and "
     "muB, muF for the two group means:",
     ("formula", ["sigma_between^2(t) = wB(t) * wF(t) * ( muB(t) - muF(t) )^2"]),
     "Maximising that is equivalent to minimising the variance within the "
     "groups, because the total variance is fixed, and the between-class "
     "form is far cheaper since it needs only running totals.",
     "The code evaluates it for all 256 candidate thresholds at once. With "
     "`p` the normalised histogram, `w = cumsum(p)` and `m = cumsum(p * "
     "index)`, the between-class variance reduces to",
     ("formula", ["sigma_between^2(t) = ( mTotal * w(t) - m(t) )^2",
                  "                    / ( w(t) * (1 - w(t)) )"]),
     "which is the expression in the code. The denominator is floored at "
     "1e-12 because at the two extreme thresholds one group is empty and the "
     "product `w*(1-w)` is exactly zero; without the guard those positions "
     "would evaluate to infinity and always win.",
     "Integer and float inputs are handled separately because an integer "
     "image can be histogrammed exactly with `bincount`, while a ratio image "
     "has to be clipped and binned, and its threshold must be returned as a "
     "value rather than as a bin index.",
   ]),

 "BinarizeLine": dict(
   inputs="`gray`, a grey line crop as a 2-D array.",
   output="A boolean ink mask the same shape, with pinholes closed, specks "
          "removed and long straight runs deleted.",
   cite="Otsu1979",
   body=[
     "The standard entry point for turning a grey line crop into ink, used "
     "by the glyph extractor and by synthesis when it needs to measure its "
     "own output. Four steps, in order.",
     ("sub", "1. Threshold"),
     "`Otsu` picks the cut, so no brightness constant is baked in.",
     ("sub", "2. Close pinholes"),
     "Thresholding a photograph leaves single-pixel holes inside strokes "
     "where the paper texture shows through. `Close` with a small radius "
     "fills them without thickening the stroke, which matters because the "
     "thinning stage downstream is sensitive to holes: a hole forces the "
     "skeleton to loop around it and produces a spurious ring.",
     ("sub", "3. Drop specks"),
     "`LabelComponents` groups the ink and components below a minimum area "
     "are deleted as dust or sensor noise. The test is on the whole "
     "component, so a letter is never partly removed.",
     ("sub", "4. Delete printed rules"),
     "`RowRunLength` finds horizontal runs and anything longer than the "
     "limit is treated as ruling rather than writing, since a pen stroke "
     "does not run perfectly level for that distance.",
   ]),

 "CoreBand": dict(
   inputs="`ink`, a boolean mask of one line of writing.",
   output="`(top, base)` row indices bounding the x-height band, or `None` "
          "when there is no ink or the band is under three rows tall.",
   body=[
     "Finds the band of rows holding the bodies of the lower case letters, "
     "which is the vertical scale everything else in the pipeline is "
     "measured against. A glyph stored in x-height units can be redrawn at "
     "any physical size, and the slant, ascender and descender measurements "
     "are all expressed as multiples of this band's height.",
     "The method is a horizontal projection profile: count ink per row, then "
     "keep the rows holding at least 45 per cent of the peak count.",
     ("formula", ["prof[y] = number of ink pixels in row y",
                  "band    = { y : prof[y] >= 0.45 * max(prof) }",
                  "return (first such row, last such row)"]),
     "The 45 per cent figure is the one tuned constant. The reasoning is "
     "that ascenders and descenders are comparatively rare, so rows "
     "containing only them hold much less ink than rows crossing every "
     "letter body; a threshold anywhere in the broad middle of that gap "
     "separates the two. Too low and the band swallows the ascenders, too "
     "high and it collapses onto the single densest row.",
     "A band under three rows tall is rejected as not a band at all, which "
     "is what happens on a crop containing a stray mark rather than writing.",
   ]),

# ------------------------------------------------------ toolkit: component
 "LabelComponents": dict(
   inputs="`mask`, a 2-D boolean array.",
   output="`(labels, count)`. `labels` is an int32 array where each ink "
          "pixel carries its component number starting at 1 and background "
          "is 0; `count` is how many components there are.",
   cite="He2008, Tarjan1975",
   body=[
     "Groups touching ink into pieces, which is how the pipeline reasons "
     "about a letter, a dot or a speck as a single object rather than as "
     "loose pixels. It is a run-based two-scan labelling algorithm with "
     "union-find, which is the standard approach [He2008].",
     ("sub", "Why runs rather than pixels"),
     "Each row is first reduced to horizontal runs of ink, found from the "
     "difference of the padded row: a run starts where the row steps 0 to 1 "
     "and ends where it steps 1 to 0. All pixels in a run necessarily belong "
     "to the same component, so the algorithm reasons about runs, of which "
     "there are far fewer than pixels.",
     ("sub", "Joining to the row above"),
     "Each run is compared against the runs of the previous row. Because the "
     "runs on both rows are in increasing order, a single moving index `k` "
     "walks forward and never restarts, so the comparison is linear overall "
     "rather than quadratic. A run above is considered touching when its "
     "span overlaps, tested with `prev[kk][0] <= b`; using `<=` against the "
     "exclusive end is what makes a diagonal touch count, giving "
     "8-connectivity rather than 4.",
     ("sub", "Union-find"),
     "A run may touch two runs above that have so far been numbered "
     "differently, which means those two numbers are really one component. "
     "Rather than relabelling, the smaller number is recorded as the parent "
     "of the larger in a `parent` list, and `find` follows the chain to the "
     "representative. `find` also performs path halving, `parent[a] = "
     "parent[parent[a]]`, which flattens the chain as it walks and keeps the "
     "amortised cost very close to constant [Tarjan1975].",
     ("sub", "Renumbering"),
     "Labels are handed out before the merges are known, so the final labels "
     "have gaps. A closing pass resolves every label to its root and "
     "`np.unique(..., return_inverse=True)` renumbers the survivors "
     "consecutively from 1, so the caller can rely on labels 1..count with "
     "nothing missing and can use `bincount` on them directly.",
   ]),

 "PieceBounds": dict(
   inputs="`pl`, the label of each ink pixel; `ys`, `xs`, those pixels' "
          "coordinates; `nc`, the number of components; `H`, `W`, the image "
          "shape.",
   output="`(top, bot, xmin, xmax, area)`, five arrays indexed by component "
          "number, giving each piece's bounding box and pixel count.",
   body=[
     "Measures every component in one pass. Instead of looping over "
     "components and masking the image for each, it scatters the per-pixel "
     "coordinates into per-label accumulators with `np.minimum.at` and "
     "`np.maximum.at`, and counts with `bincount`.",
     "This matters because the ownership rules in the segmenter ask "
     "geometric questions about every piece on the page, several times over, "
     "and a per-component loop with masking would be the dominant cost of "
     "the whole segmentation.",
   ]),

# -------------------------------------------------------- toolkit: thinning
 "Skeletonize": dict(
   inputs="`mask`, a boolean ink mask.",
   output="A boolean mask of the one-pixel-wide centreline of the ink.",
   cite="ZhangSuen1984",
   body=[
     "Reduces a thick pen stroke to the path the pen travelled, which is "
     "what the glyph extractor follows to recover stroke geometry. This is "
     "the Zhang-Suen parallel thinning algorithm [ZhangSuen1984], "
     "implemented with whole-array operations.",
     ("sub", "The neighbourhood"),
     "`ring` returns the eight neighbours of every pixel simultaneously as "
     "eight shifted copies of the padded image, ordered clockwise from "
     "north, named P2 to P9 after the convention in the original paper. Two "
     "quantities are then computed for every pixel at once:",
     ("formula", ["B(P1) = P2 + P3 + ... + P9        (set neighbours)",
                  "A(P1) = number of 0 -> 1 steps going round P2..P9, P2"]),
     ("sub", "The deletion test"),
     "A pixel is deleted in a sub-iteration when all of the following hold:",
     ("formula", ["(a)  2 <= B(P1) <= 6",
                  "(b)  A(P1) == 1",
                  "(c)  P2*P4*P6 == 0   (first sub-iteration)",
                  "(d)  P4*P6*P8 == 0   (first sub-iteration)",
                  "     P2*P4*P8 == 0   and  P2*P6*P8 == 0  (second)"]),
     "Condition (a) protects endpoints, which have one neighbour, and "
     "interior pixels, which have more than six. Condition (b) is the "
     "important one: A = 1 means the set neighbours form a single connected "
     "arc around the pixel, so removing it cannot disconnect the stroke or "
     "open a hole. Conditions (c) and (d) restrict which side is thinned in "
     "each sub-iteration.",
     ("sub", "Why two sub-iterations"),
     "Deleting from all sides at once can erase a two-pixel-wide line "
     "entirely, since both halves satisfy the test simultaneously. "
     "Alternating the corner conditions between the two sub-iterations "
     "thins from opposite sides on alternate passes, so the result stays "
     "connected and the centreline does not drift off to one side. The loop "
     "repeats until a full pass deletes nothing.",
     "The whole test is evaluated as boolean array arithmetic, so each "
     "sub-iteration is a fixed number of array operations rather than a scan "
     "over pixels. The cost is proportional to the stroke thickness, since "
     "that is how many layers have to come off.",
   ]),

# ------------------------------------------------------- toolkit: resizing
 "_Weights": dict(
   inputs="`inSize`, `outSize`, the length of one axis before and after. "
          "`kernel`, either \"linear\" for a triangle filter or \"box\" for a "
          "flat one.",
   output="An `(outSize, inSize)` matrix whose rows each sum to 1; row i "
          "gives the weight of every input sample in output sample i.",
   cite="Wolberg1990",
   body=[
     "Builds the resampling weights for one axis. Resizing is then a matrix "
     "product, and because the two axes are handled separately the cost is "
     "linear in each rather than quadratic.",
     ("sub", "Where an output sample sits"),
     "Output pixel i covers input coordinates centred at",
     ("formula", ["scale   = inSize / outSize",
                  "centre  = (i + 0.5) * scale"]),
     "The half-pixel offsets are because a pixel is a square of unit extent "
     "whose centre is at an integer plus a half, not a point at an integer. "
     "Getting this wrong shifts the whole image by half a pixel, which is "
     "visible as a slight blur and a systematic offset.",
     ("sub", "Widening the filter when shrinking"),
     "The filter support is multiplied by `filterScale = max(1, scale)`, so "
     "it widens when minifying and stays put when magnifying. This is the "
     "standard treatment [Wolberg1990] and it is what prevents aliasing: if "
     "the filter stayed one input pixel wide while the image shrank by a "
     "factor of four, three quarters of the input would not be consulted at "
     "all and thin strokes would simply disappear between samples. Since the "
     "page reduction used by the skew search shrinks by exactly such a "
     "factor, this is load bearing.",
     ("sub", "The kernels"),
     "With t the distance from the window centre measured in filter widths:",
     ("formula", ["linear:  w(t) = 1 - |t|   for |t| < 1,  else 0",
                  "box:     w(t) = 1         for |t| <= 0.5, else 0"]),
     ("sub", "Edges and normalisation"),
     "A window near the border runs off the image. The out-of-range indices "
     "are clipped back onto the first or last sample, and the weights are "
     "accumulated with `np.add.at` rather than assigned, because several "
     "clipped positions fold onto the same index and their weights must sum "
     "instead of overwriting one another. Each row is then divided by its "
     "own total so it sums to one, which is why a padded crop does not come "
     "back with a dark border.",
   ]),

 "Resize": dict(
   inputs="`img`, a 2-D array. `newW`, `newH`, the target size. `kernel`, "
          "the filter name passed through to `_Weights`.",
   output="The resampled image at the requested size, in the input dtype.",
   cite="Wolberg1990",
   body=[
     "Separable resampling: apply the row weights, then the column weights. "
     "For an H by W image going to H' by W' this costs O(H*W*k) rather than "
     "the O(H'*W'*k^2) of a true 2-D filter, where k is the filter width.",
     "Everything that changes image size in the pipeline goes through here, "
     "so there is exactly one resampling implementation to be correct about: "
     "the page reduction for the skew search, the fit onto the classifier "
     "canvas, and the glyph rendering.",
   ]),

 "Halve": dict(
   inputs="`img`, a 2-D array.",
   output="The image at half size in each axis, by exact 2 by 2 averaging.",
   body=[
     "The quick page reduction, used where a decision does not need full "
     "resolution: finding the sheet inside the photograph, and the coarse "
     "part of the skew search. Averaging two-by-two blocks is both an exact "
     "box filter at that ratio and the cheapest possible correct reduction.",
   ]),

 "FitFixedCanvas": dict(
   inputs="`gray`, a crop. `height`, `width`, the fixed canvas size. "
          "`padValue`, the fill for the unused area, 255 (white paper) by "
          "default.",
   output="A `(height, width)` uint8 image containing the crop scaled to fit, "
          "vertically centred, flush to the left, on a paper-coloured "
          "background.",
   body=[
     "Both networks were trained on one exact input size, 64 by 640, so "
     "every crop has to be brought to that size before it can be read.",
     "The scale factor is the smaller of the two ratios:",
     ("formula", ["s = min( height / h , width / w )"]),
     "One factor is used for both axes, never two. Stretching a crop to fill "
     "the canvas would change the aspect ratio of the letters, and letter "
     "proportion is exactly what both networks are reading, so a stretched "
     "crop is a different hand as far as they are concerned.",
     "The result is centred vertically and left-flush horizontally, which "
     "matches how the training crops were produced: text starts at the left "
     "margin and the vertical position of a line within its crop is not "
     "meaningful. The final assignment slices both source and destination, "
     "which silently absorbs the one-pixel overshoot that rounding the "
     "scaled size can produce.",
   ]),

# ------------------------------------------------- toolkit: geometric maps
 "_Sample": dict(
   inputs="`img`, the source. `sy`, `sx`, arrays of fractional source "
          "coordinates, one per output pixel.",
   output="An array of sampled values, the same shape as the coordinate "
          "arrays.",
   body=[
     "Bilinear interpolation, the sampler behind every geometric transform "
     "in the file. The value at a fractional position is the weighted "
     "average of the four surrounding pixels, weighted by how close each one "
     "is:",
     ("formula", ["fy = sy - floor(sy),   fx = sx - floor(sx)",
                  "top    = p00*(1-fx) + p01*fx",
                  "bottom = p10*(1-fx) + p11*fx",
                  "value  = top*(1-fy) + bottom*fy"]),
     "Without interpolation a rotated image shows staircase edges, because "
     "each output pixel would take whichever single input pixel it happened "
     "to land nearest.",
   ]),

 "_Like": dict(
   inputs="`out`, a float result. `ref`, the array whose dtype should be "
          "matched.",
   output="`out` rounded and cast back to `ref`'s dtype, clipped to its "
          "range where that applies.",
   body=[
     "Interpolation has to be done in floating point, but the pipeline "
     "passes uint8 images around. This performs the trip back: round rather "
     "than truncate, and clip so that a value which overshot 255 or went "
     "below 0 does not wrap around to the opposite extreme.",
   ]),

 "Rotate": dict(
   inputs="`img`, the source. `deg`, the angle in degrees. `fill`, the value "
          "for areas with no source.",
   output="The rotated image, same canvas size as the input.",
   cite="Wolberg1990",
   body=[
     "Rotates about the image centre using inverse mapping: for each OUTPUT "
     "pixel it computes where that came from in the input, rather than "
     "pushing input pixels forward. Forward mapping leaves holes wherever "
     "two output pixels do not receive a source pixel; inverse mapping "
     "cannot, because every output pixel is visited exactly once.",
     ("formula", ["sx = cos(th)*(x - cx) + sin(th)*(y - cy) + cx",
                  "sy = -sin(th)*(x - cx) + cos(th)*(y - cy) + cy"]),
     "Source positions outside the image are filled rather than clamped, "
     "because clamping would smear the edge pixel across the corner.",
     "**Edge convention.** A source coordinate counts as inside the image "
     "when `-0.5 < s < size - 0.5`, not when `0 <= s <= size - 1`. This "
     "follows from treating a pixel as a unit square centred on its index: "
     "the image occupies the interval from -0.5 to size-0.5. The convention "
     "was confirmed empirically against the library implementation it "
     "replaced, where the two agreed to a maximum difference of one grey "
     "level over a whole page; the other plausible convention differed by up "
     "to 245.",
   ]),

 "Warp": dict(
   inputs="`img`, the source. `coeffs`, six or eight numbers describing the "
          "mapping. `fill`, the background value.",
   output="The transformed image.",
   cite="Wolberg1990",
   body=[
     "One function covering affine and perspective transforms, again by "
     "inverse mapping. With six coefficients the mapping is affine; with "
     "eight it is a homography, where the divisor varies per pixel and is "
     "what produces the converging-lines appearance of a page photographed "
     "off square:",
     ("formula", ["affine:       sx = a*x + b*y + c",
                  "              sy = d*x + e*y + f",
                  "perspective:  w  = g*x + h*y + 1",
                  "              sx = (a*x + b*y + c) / w",
                  "              sy = (d*x + e*y + f) / w"]),
     "**Half-pixel convention.** The coefficients act on the *centre* of the "
     "output pixel, so the code adds 0.5 to the output coordinate before "
     "applying the mapping and subtracts 0.5 from the result. This was not "
     "guesswork: with the offsets in place the output matches the library "
     "implementation it replaced on zero pixels of difference across a test "
     "page, and without them the whole image is displaced by half a pixel.",
   ]),

 "Blur": dict(
   inputs="`img`, the source. `sigma`, the standard deviation in pixels.",
   output="The blurred image, in the input dtype.",
   body=[
     "A gaussian blur done as two one-dimensional passes, because a 2-D "
     "gaussian is the product of two 1-D gaussians:",
     ("formula", ["G(x, y) = G(x) * G(y)",
                  "G(t)    = exp( -t^2 / (2*sigma^2) )"]),
     "That separability turns a k by k convolution into two convolutions of "
     "length k, which for the radii used here is roughly an order of "
     "magnitude less work. The kernel is truncated at about three sigma, "
     "where the remaining weight is under one per cent, and normalised to "
     "sum to one so the image's overall brightness is unchanged.",
     "Two callers: the background estimate for ink detection, and softening "
     "the edge of synthesised ink so that a rendered stroke has the gradual "
     "edge of a real pen rather than a hard step.",
   ]),

 "Elastic": dict(
   inputs="`img`, the source. `rng`, a seeded random generator. `alpha`, how "
          "far pixels move. `sigma`, how smoothly the displacement varies.",
   output="A locally bent version of the image.",
   cite="Simard2003",
   body=[
     "Elastic distortion, the augmentation that makes a training sample look "
     "like the same hand writing the same text again rather than a copy of "
     "it [Simard2003]. Two random fields are generated, one per axis, "
     "smoothed with `Blur` and scaled by alpha, then the image is read at "
     "the displaced positions:",
     ("formula", ["dx = alpha * Blur( uniform(-1, 1), sigma )",
                  "dy = alpha * Blur( uniform(-1, 1), sigma )",
                  "out[y, x] = sample( img, y + dy[y, x], x + dx[y, x] )"]),
     "The smoothing is the essential part. An unsmoothed displacement field "
     "scrambles pixels independently and destroys the strokes; a field "
     "smoothed over several pixels moves neighbouring pixels together, so "
     "strokes bend and stretch while staying continuous, which is how "
     "natural variation in handwriting actually behaves.",
   ]),

 "DrawPolyline": dict(
   inputs="`img`, the array to draw into, modified in place. `pts`, a list "
          "of `(x, y)` points. `value`, the ink level. `width`, the pen "
          "width in pixels.",
   output="The same array, for convenience in chaining.",
   body=[
     "Draws a stroke with a round pen. For each segment, every pixel in the "
     "segment's bounding box is tested by distance to the nearest point ON "
     "the segment, and filled if that distance is within half the pen width:",
     ("formula", ["t = clamp( dot(P - A, B - A) / |B - A|^2 , 0, 1 )",
                  "nearest point = A + t*(B - A)",
                  "fill where  |P - nearest| <= width/2"]),
     "The clamp on t is what makes the distance measure to the *segment* "
     "rather than to the infinite line through it, and it is also what gives "
     "round end caps and round joins for free: near an endpoint the nearest "
     "point is the endpoint itself, so the filled region is a disc. No "
     "separate cap or join handling is needed.",
     "Work is restricted to each segment's bounding box, expanded by the pen "
     "radius, so a long path over a big canvas costs time proportional to "
     "the ink and not to the canvas area. A zero-length segment falls back "
     "to plain distance-to-point, which is how a single dot is drawn.",
     "This rasteriser is used both for rendering synthesised handwriting and "
     "for the geometry comparisons in the measurement harness, so the two "
     "see ink laid down the same way.",
   ]),

# ---------------------------------------------------- toolkit: measurement
 "EstimateSlantDeg": dict(
   inputs="`ink`, a boolean mask of one line of writing.",
   output="The slant in degrees, positive for a normal forward lean. Returns "
          "0.0 when there are fewer than 50 ink pixels.",
   cite="Vinciarelli2001",
   body=[
     "Measures how far the writing leans, which synthesis needs in order to "
     "reproduce a hand and which the glyph extractor needs in order to store "
     "letters upright.",
     "The method is to shear the ink by a series of trial angles and keep "
     "the angle at which the vertical strokes stack up most sharply "
     "[Vinciarelli2001]. Sharpness is measured as the sum of squares of the "
     "column profile:",
     ("formula", ["x'    = x - tan(angle) * (y - yc)",
                  "prof  = histogram of x' over all ink pixels",
                  "score = sum of prof^2,   maximise over angle"]),
     "Squaring is what makes the score prefer concentration. For a fixed "
     "total amount of ink, the sum of squares is largest when the ink is "
     "piled into few columns and smallest when it is spread evenly, so the "
     "best angle is the one at which upright strokes become truly vertical "
     "and align into tall narrow columns.",
     "The search runs from -45 to +45 degrees in 1.5 degree steps, which "
     "bounds the resolution of the result. It shears coordinates rather than "
     "the image, so each trial is a histogram over the ink pixels and not an "
     "image warp, which is what makes a 60-angle search cheap.",
     "The returned value is negated so that a positive number means a normal "
     "forward lean, which makes the deslanting step read as `x -= "
     "tan(slant) * y`.",
   ]),

 "PolylineLength": dict(
   inputs="`p`, a list or array of `(x, y)` points.",
   output="The total distance walked along the path, as a float.",
   body=[
     "Arc length of a polyline. `np.diff` gives the step vectors, `np.hypot` "
     "their lengths, and the sum is the total:",
     ("formula", ["L = sum over i of sqrt( (x[i+1]-x[i])^2 + (y[i+1]-y[i])^2 )"]),
     "Used as a size measure for glyphs, where a fragment is recognised by "
     "having far less pen path than a real letter of that character, and in "
     "the measurement harness for reporting stroke lengths.",
   ]),

# ------------------------------------------------- segmenter: conditioning
 "InkMask": dict(
   inputs="`gray`, the full page as a 2-D array.",
   output="`(mask, ratio)`. `mask` is the boolean ink mask; `ratio` is the "
          "intermediate image of each pixel divided by its local background, "
          "returned because later stages reuse it.",
   body=[
     "Decides what is ink on a photographed page, where the lighting is "
     "never even.",
     ("sub", "Local, not global"),
     "Each pixel is divided by the mean of its own neighbourhood, and the "
     "threshold is taken on that ratio:",
     ("formula", ["ratio = gray / max( BoxMean(gray, r), 1 )",
                  "mask  = ratio < Otsu(ratio)"]),
     "Division rather than subtraction is deliberate. Shading in a "
     "photograph is multiplicative, since it comes from how much light "
     "reached the paper, so dividing cancels it while subtracting leaves a "
     "residue that varies across the page. The radius scales with image "
     "height, `r = clamp(H/41, 8, 40)`, so the background estimate covers "
     "roughly the same physical area whatever the camera resolution.",
     ("sub", "Removing solid dark regions"),
     "A binder, a shadow or the dark surround of the sheet are solidly dark "
     "over a wide area, which no handwriting is. Regions where a 9 by 9 "
     "window is essentially entirely ink are detected and removed, with a "
     "small dilation so the whole region goes rather than just its core.",
   ]),

 "PaperRegionMask": dict(
   inputs="`g`, the page image. `P`, the line pitch in pixels.",
   output="A boolean mask that is true over the sheet of paper and false "
          "over the surroundings.",
   body=[
     "Finds the sheet inside the photograph, so that the desk, the binder "
     "and anything else around it can be excluded before line finding.",
     "The work is done on a 4 by 4 block average for speed, where the sheet "
     "is the largest bright connected region. Pixels below 45 per cent of "
     "the median block brightness are taken as surround; if almost none "
     "qualify, the photograph is all paper and the whole image is returned "
     "unmasked. Otherwise the largest component of the non-dark region is "
     "the sheet, selected by `bincount(...).argmax()`.",
     "The block-averaged mask is expanded back to full resolution with "
     "`np.kron`, which tiles each block value into a 4 by 4 patch of ones. "
     "Finally the mask is shrunk inward by about 0.4 of a line pitch using "
     "two one-dimensional dilations of its complement, which keeps the "
     "slightly soft paper edge from being read as ink.",
     "The pitch is used as the unit for that margin so the behaviour is the "
     "same on a page photographed close up and far away.",
   ]),

 "LinePitch": dict(
   inputs="`mask`, the page ink mask.",
   output="`(pitch, smoothedProfile)`, the spacing between lines in pixels "
          "and the smoothed row profile it was measured from.",
   body=[
     "Estimates the spacing between lines of writing, which is the unit "
     "almost every later threshold is expressed in. Three methods are tried "
     "in order of reliability, which is what makes the function robust on "
     "pages the first method cannot handle.",
     ("sub", "1. Gaps between profile peaks"),
     "Ink per row is smoothed with a gaussian of about four pixels and the "
     "local maxima above six per cent of the peak are taken as line centres; "
     "the pitch is the median gap between consecutive peaks. The median is "
     "used rather than the mean so that one unusually large gap, such as a "
     "paragraph break, does not drag the estimate. This works from two lines "
     "upward and tolerates irregular spacing, so it is tried first.",
     ("sub", "2. Autocorrelation"),
     "With fewer than two peaks there are no gaps to measure, so the profile "
     "is correlated with itself and the first strong peak in the "
     "autocorrelation at a lag above 15 pixels is taken as the period:",
     ("formula", ["ac[k] = sum over y of ( prof[y] - mean ) * ( prof[y+k] - mean )",
                  "pitch = smallest k > 15 with ac[k] >= 0.7 * max(ac)"]),
     "Subtracting the mean first is what makes this measure periodicity "
     "rather than overall brightness.",
     ("sub", "3. Letter size"),
     "On a page with a single line of writing there is no periodicity at "
     "all. The fallback measures the heights of the ink components and "
     "returns 2.2 times their median height, that ratio being the typical "
     "relationship between x-height and line spacing in handwriting. "
     "Components under 8 pixels of area or 3 pixels of height are excluded "
     "as noise.",
     "If there is no usable ink at all, 48 pixels is returned as a harmless "
     "default; with no ink, nothing downstream will find a line anyway.",
   ]),

 "LeastCostSeam": dict(
   inputs="`cost`, a 2-D array where each entry is the cost of the path "
          "passing through that pixel. `step`, how many rows the path may "
          "move between adjacent columns, 3 by default.",
   output="An array of length W giving the chosen row for each column.",
   cite="Bellman1954, Avidan2007, Arvanitopoulos2014",
   body=[
     "Finds the cheapest path from the left edge to the right edge of the "
     "page, used to cut between two lines of writing without slicing through "
     "a letter. This is seam carving [Avidan2007] applied to text line "
     "extraction [Arvanitopoulos2014], solved by dynamic programming "
     "[Bellman1954].",
     ("sub", "The recurrence"),
     "Let `acc[y, x]` be the cheapest total cost of any path that reaches "
     "pixel (y, x) from the left edge. Then",
     ("formula", ["acc[y, x] = cost[y, x]",
                  "          + min over d in [-step, step] of",
                  "              ( acc[y + d, x - 1] + 0.02*|d| )"]),
     "Each column depends only on the one before it, so one left-to-right "
     "sweep fills the table, and the minimising choice `d` is recorded in a "
     "back-pointer array. The path is then recovered by starting at the "
     "cheapest pixel in the last column and following the back-pointers "
     "right to left.",
     ("sub", "Why the small penalty on d"),
     "The `0.02*|d|` term charges a little for moving off straight ahead. "
     "Without it, among the many equally cheap paths through blank paper the "
     "algorithm would pick an arbitrary zig-zag; with it, a straight path is "
     "preferred and the seam only deviates where deviating actually avoids "
     "ink. The constant is small enough that it never outweighs crossing a "
     "stroke.",
     ("sub", "Why a seam rather than a straight cut"),
     "A straight horizontal cut between two lines passes through descenders "
     "from the line above and ascenders from the line below. A seam is free "
     "to wander by up to `step` rows per column, so it goes around them, "
     "which is the difference between a clean crop and a crop with the "
     "bottom of the previous line's 'g' stuck to the top of it.",
     "The candidate construction uses array shifts with `np.inf` padding so "
     "that all `2*step+1` options for a whole column are evaluated at once, "
     "making the sweep linear in the number of pixels with a small constant "
     "rather than a per-pixel Python loop.",
   ]),

 "SkewScore": dict(
   inputs="`mask`, the page ink mask. `angleDeg`, the trial angle.",
   output="A single float, higher when the lines of writing are more level "
          "at that angle.",
   cite="Postl1986, Baird1987",
   body=[
     "Scores how level the writing would be if the page were rotated by a "
     "trial angle. This is the projection profile criterion for skew "
     "estimation [Postl1986, Baird1987]: when a page is level, the ink of "
     "each line falls into a few rows and the row profile has tall narrow "
     "peaks; when it is tilted, the same ink is spread across many rows.",
     ("formula", ["k     = round( (y - cy)*cos(th) - (x - cx)*sin(th) + cy )",
                  "prof  = histogram of k over all ink pixels",
                  "score = sum of prof^2"]),
     "Only the row coordinate of the rotated position is needed, so the "
     "function bins ink pixels by that coordinate instead of rotating the "
     "image. A search over many angles therefore costs one pass over the ink "
     "per angle, with no interpolation, which made the skew search about "
     "thirty times faster than rotating the image for each trial.",
     ("sub", "A correction worth recording"),
     "An earlier version clipped `k` into the page height with `np.clip(k, "
     "0, h-1)`. That was wrong: at a large trial angle, ink that rotates off "
     "the page was piled into the first and last rows, creating an enormous "
     "artificial spike which, once squared, scored higher than any genuine "
     "alignment and selected an angle about four degrees away from correct. "
     "Shifting with `k -= k.min()` instead keeps every pixel in its own bin "
     "and restored exact agreement with the hand-measured angles on all 35 "
     "test pages.",
   ]),

 "Deskew": dict(
   inputs="`page`, the grey page image.",
   output="`(fullArray, halfArray, mask, ratio, bestAngle)`: the "
          "straightened page at full and half resolution, its ink mask and "
          "ratio image, and the angle that was applied.",
   cite="Postl1986",
   body=[
     "Straightens the page. The angle is found by maximising `SkewScore` "
     "over a range of trial angles, and the page is then rotated once by the "
     "winner.",
     "The search is coarse to fine: a wide sweep on a half-resolution copy "
     "to locate the neighbourhood, then a fine sweep around it. Because "
     "`SkewScore` works by binning coordinates, neither stage rotates the "
     "image; only the single final rotation does.",
     "The ink mask and ratio image are computed once here and passed out, "
     "rather than being recomputed by each later stage, because `InkMask` is "
     "one of the more expensive operations on a full page.",
     "On the 35 test pages whose skew was measured by hand this recovers the "
     "angle exactly where the true angle lies on the search grid, and to "
     "within the grid resolution otherwise, which is the expected behaviour "
     "for a quantised search.",
   ]),

 "RemovePrintedStructure": dict(
   inputs="`mask`, the page ink mask. `P`, the line pitch. Plus the paper "
          "region information.",
   output="The ink mask with the printed ruling, the binder edge and "
          "everything outside the sheet removed.",
   body=[
     "Removes what was printed on the page so that only what was written on "
     "it remains. Two steps.",
     "First the sheet is isolated with `PaperRegionMask`, which removes the "
     "desk and the binder in one go.",
     "Then ruled lines are removed with `LongRunMask`. A horizontal rule is "
     "ink that continues unbroken for far longer than a pen stroke, so the "
     "run-length threshold is set as a multiple of the line pitch: 1.25 "
     "pitches for horizontal rules and 1.4 pitches for the vertical binder "
     "edge. Each detection is widened slightly along its own axis with "
     "`AnyWithinRadius`, so a rule that is a fraction of a degree off "
     "horizontal is still removed along its whole length rather than in "
     "pieces, and the result is intersected back with the original mask so "
     "that only genuine ink can be deleted.",
   ]),

 "DropArtifactPieces": dict(
   inputs="`mask`, the ink mask. `P`, the line pitch, plus page geometry.",
   output="The mask with non-writing blobs removed.",
   body=[
     "Throws away connected components that cannot be handwriting. The tests "
     "are made on whole components, using `LabelComponents` and "
     "`PieceBounds`, which is the important design choice: a per-pixel test "
     "can remove half a letter, whereas a per-component test either keeps a "
     "letter or removes it.",
     "What gets removed: slivers against the page edge, which are the "
     "remains of the sheet boundary; components far larger than a line pitch "
     "in both axes, which are shadows or the binder; components that are "
     "extremely long and thin, which are leftovers of the ruling that "
     "survived the previous stage; and components too small to be a pen "
     "mark.",
     "All of the size tests are expressed as multiples of the line pitch "
     "rather than in pixels, so the same thresholds hold at any camera "
     "resolution.",
   ]),

# --------------------------------------------------- segmenter: line split
 "LineCentres": dict(
   inputs="`sm`, the smoothed row profile from `LinePitch`. `P`, the pitch.",
   output="A list of row indices, one per line of writing, in order down the "
          "page.",
   body=[
     "Locates the vertical centre of each line of writing as the peaks of "
     "the smoothed ink-per-row profile, with a minimum separation of 0.6 of "
     "a pitch enforced between accepted peaks.",
     "The separation rule is what prevents one line being counted twice. A "
     "single line of writing produces a broad peak with small bumps on it, "
     "from the uneven distribution of ascenders and descenders; without a "
     "minimum separation each bump becomes a line and the page is "
     "over-segmented. The threshold is a little over half a pitch because "
     "two genuine lines are never closer than that, by the definition of the "
     "pitch.",
   ]),

 "SeamCuts": dict(
   inputs="`mask`, the ink mask. `centres`, the line centre rows. `P`, the "
          "pitch.",
   output="A list of seams, each an array of one row index per column, "
          "giving the boundary between consecutive lines.",
   cite="Arvanitopoulos2014",
   body=[
     "Builds the boundary between each pair of neighbouring lines by running "
     "`LeastCostSeam` through the band between their centres, with the ink "
     "mask as the cost. The seam therefore follows the path of least ink "
     "from the left edge of the page to the right.",
     "Searching only within the band between two centres, rather than the "
     "whole page, keeps the cost down and also guarantees the seams come out "
     "in order and cannot cross one another.",
     "This is the step that makes descenders work. A descender from the line "
     "above hangs into the gap; a straight cut would slice it, while the seam "
     "deviates around it and leaves it attached to the line it belongs to.",
   ]),

 "_BandOwnership": dict(
   inputs="The ink components, the seam boundaries, and the page geometry.",
   output="An assignment from component to line index.",
   body=[
     "The first pass of ownership: each component is given to the line whose "
     "band contains the largest part of it, measured by pixel count rather "
     "than by the position of its centre.",
     "Using the majority rather than the centroid is what keeps a tall "
     "letter intact. The centroid of a letter with a long ascender can fall "
     "on the wrong side of a seam even though almost all of the letter is on "
     "the right side; majority ownership assigns it correctly, and because "
     "the whole component moves together, the letter is never cut in half.",
     "Three correction rules then follow, each handling a specific way this "
     "first guess goes wrong.",
   ]),

 "_DotsJoinedAbove": dict(
   inputs="The current ownership, the component measurements, and the pitch.",
   output="Ownership with dots reassigned to the line they belong to.",
   body=[
     "Correction rule one. Closing small gaps earlier in the pipeline can "
     "fuse the dot of an 'i' to a descender hanging down from the line "
     "above, so the two become one component and majority ownership gives "
     "the pair to the upper line.",
     "The rule detects a component that is small, sits high within its band, "
     "and whose ink is concentrated near the top, and returns it to the line "
     "below. The size and position tests are in units of the pitch, and the "
     "rule only fires where the geometry is unambiguous, so a genuine "
     "descender is not moved.",
   ]),

 "_RingsToStem": dict(
   inputs="The current ownership and the component measurements.",
   output="Ownership with ring-and-stem fusions resolved.",
   body=[
     "Correction rule two, the mirror image of the first. The same gap "
     "closing can weld the loop of a letter to the stem of a letter below "
     "it. Here the test looks for a ring-shaped blob sitting on top of a "
     "stem and assigns the pair to the line the stem belongs to, since the "
     "stem is the part that establishes which line is being written on.",
   ]),

 "_FragmentsToNeighbour": dict(
   inputs="The current ownership, the component measurements, and the "
          "x-height band of each line.",
   output="Ownership with stray fragments rehomed.",
   body=[
     "Correction rule three, the general catch. A component with no ink at "
     "all inside its assigned line's x-height band is not part of that "
     "line's writing, whatever the majority said: every real letter puts "
     "something in the band, because that is where letter bodies are.",
     "Such a component is moved to whichever neighbouring line it does reach "
     "into. This catches the cases the two specific rules do not, such as a "
     "comma or a stray accent that ended up in the wrong band.",
   ]),

 "AssignOwnership": dict(
   inputs="The ink components, the seams, and the page geometry.",
   output="The final assignment from component to line.",
   body=[
     "Runs the first guess and then the three corrections in order. The "
     "structure is deliberate: each rule is a separate function with a "
     "single responsibility, so that it can be read, tested or disabled on "
     "its own, and so that the sequence here reads as a statement of the "
     "method rather than as a block of conditionals.",
     "The rules are ordered from most specific to most general, so that a "
     "case with a known cause is handled by the rule written for it and only "
     "what remains falls through to the general test.",
   ]),

 "_BuildLineCrop": dict(
   inputs="One line's owned components, the page image, and the geometry.",
   output="A dictionary describing the finished crop, or `None` if what "
          "remains is too small to be a line of writing.",
   body=[
     "Turns one line's worth of owned ink into a finished crop. The ink "
     "owned by this line is assembled, the bounding box of that ink is "
     "taken, a small margin is added, and the grey image is cut to it.",
     "Cropping to the ink rather than to the band between the seams is what "
     "makes the output consistent: the band's height depends on where the "
     "neighbouring lines happened to be, while the ink's extent is a "
     "property of the writing itself. The downstream networks see a crop "
     "framed the same way every time.",
     "Returning `None` for an undersized result is the guard against a band "
     "that caught only a speck, and the caller drops those rather than "
     "passing an empty crop into the recogniser.",
   ]),

 "SegmentLines": dict(
   inputs="`page`, one photographed page as a 2-D grey array.",
   output="`(lines, debug, info)`. `lines` is a list of crop dictionaries in "
          "top-to-bottom order; `debug` carries intermediate images for "
          "inspection; `info` reports the line count, the measured pitch and "
          "the skew angle applied.",
   body=[
     "The entry point for the whole file and for functional units 2.1 and "
     "2.2. One call takes a photograph and returns one clean crop per line "
     "of writing. The body is a sequence of the stages above, which is why "
     "each stage was given its own function:",
     ("formula", ["Deskew                  straighten the page",
                  "RemovePrintedStructure  delete ruling and surround",
                  "DropArtifactPieces      delete non-writing blobs",
                  "LinePitch / LineCentres find where the lines are",
                  "SeamCuts                cut between them",
                  "AssignOwnership         decide who owns each piece",
                  "_BuildLineCrop          build one crop per line"]),
     "`info` is returned because the pitch and the applied angle are needed "
     "by callers for reporting, and because the line count is what the "
     "measurement harness compares against the transcript.",
     "**Measured performance.** On the 30 photographed pages with "
     "hand-typed transcripts, this produces exactly the expected number of "
     "lines on every page, 669 of 669 lines in total, in 4.4 to 9.2 seconds "
     "per page. The measured pitch ranged from 47 to 58 pixels and the "
     "corrected skew from -1.5 to +3.0 degrees across those pages.",
   ]),
}


#the remaining modules are written up in their own files, so no single file
#becomes unmanageable.  each one may add FUNCS, MODULES and REFERENCES.
for _extra in ("codedoc_recognise", "codedoc_style", "codedoc_synth",
               "codedoc_reproduce", "codedoc_train"):
    try:
        _m = __import__(_extra)
    except ModuleNotFoundError:
        continue
    FUNCS.update(getattr(_m, "FUNCS", {}))
    MODULES.update(getattr(_m, "MODULES", {}))
    REFERENCES.update(getattr(_m, "REFERENCES", {}))
