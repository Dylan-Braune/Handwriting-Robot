"""segment.py, part 4: the segmenter itself.

Everything in parts 1 to 3 was a general image tool.  This part is the
method: take a photograph of a page and hand back one clean crop per line
of writing.  The theory here is less about arithmetic and more about the
reasoning that turns a picture into a decision.
"""

FUNCS = {

 "InkMask": dict(
   inputs=[("gray", "the page as a 2-D array of grey values, 0 for black "
                    "and 255 for white, held as floating point numbers so "
                    "the division below does not round")],
   output="a pair. The first item is a boolean array the same shape as the "
          "page, true wherever the pixel is judged to be ink. The second is "
          "the **ratio image**, the page divided by its own local "
          "background, which later stages reuse to measure how dark each "
          "piece of ink is.",
   theory=[
     ("h", "What it is for"),
     ("p", "This is the first real decision in the whole pipeline: for "
           "every pixel of the photograph, is this pen or is this paper? "
           "Everything downstream, from finding the lines to cutting out "
           "letters, works on the answer. Getting it wrong in one direction "
           "loses strokes; getting it wrong in the other fills the page "
           "with shadows and paper texture that then get mistaken for "
           "writing."),
     ("h", "Background: why a single threshold is not enough"),
     ("p", "The obvious method is a global threshold: pick one grey value, "
           "call everything darker than it ink. That works for a scanner, "
           "where the lamp lights the page evenly. It fails badly for a "
           "photograph taken with a phone, because the illumination is "
           "not even. One corner of the page sits under a lamp and the "
           "opposite corner sits in shadow, and the page is often slightly "
           "curved so the middle is brighter than the edges. The result is "
           "that paper in the bright corner can be lighter than 220 while "
           "paper in the shadowed corner is darker than 150, and real pen "
           "ink in the bright corner is around 90. There is no single "
           "number that separates ink from paper everywhere at once: any "
           "threshold low enough to reject the shadowed paper also rejects "
           "the pale pen strokes, and any threshold high enough to catch "
           "the pale strokes also catches the shadow."),
     ("h", "The idea: divide out the illumination"),
     ("p", "The fix is to stop asking \"is this pixel dark?\" and ask "
           "instead \"is this pixel darker than the paper immediately "
           "around it?\". The illumination varies smoothly across the "
           "page, so over a small neighbourhood it is almost constant. If "
           "the brightness of a pixel is modelled as the true reflectance "
           "of the page at that point multiplied by the local illumination, "
           "then dividing by an estimate of that local illumination cancels "
           "it and leaves the reflectance, which is the thing that actually "
           "distinguishes pen from paper."),
     ("eq", r"\mathrm{ratio}[y,x] \;=\; \frac{\mathrm{gray}[y,x]}"
            r"{\max\big(\mathrm{BoxMean}(\mathrm{gray},r)[y,x],\,1\big)}",
      [(r"\mathrm{ratio}[y,x]",
        "the illumination-corrected value at this pixel; near 1 for paper, "
        "well below 1 for ink"),
       (r"\mathrm{gray}[y,x]", "the measured grey value at this pixel"),
       (r"\mathrm{BoxMean}(\mathrm{gray},r)",
        "the average grey value over a square window of radius r centred on "
        "each pixel, which serves as the estimate of the local illumination; "
        "see the entry for BoxMean"),
       ("r", "the radius of that window, in pixels"),
       (r"\max(\cdot,\,1)",
        "a guard so that a window which happens to average to zero cannot "
        "cause a division by zero")],
      None),
     ("p", "A ratio of 1 means the pixel is exactly as bright as its "
           "surroundings, which is what paper looks like. A ratio of 0.4 "
           "means the pixel is well under half as bright as the paper "
           "around it, which is what a pen stroke looks like. The key "
           "property is that this is now true in the bright corner and the "
           "shadowed corner alike, because the shadow divides out of both "
           "the numerator and the denominator."),
     ("h", "Choosing the window size, and the trap in it"),
     ("p", "The radius has to satisfy two opposing requirements. It must be "
           "large enough that the window is mostly paper even when it is "
           "centred on a stroke, because the denominator is supposed to "
           "estimate the paper brightness, not the ink. If the radius were "
           "the width of a pen stroke, then a window centred inside a thick "
           "stroke would be entirely ink, the estimated background would be "
           "as dark as the ink itself, the ratio would come out near 1, and "
           "the stroke would be declared paper. That failure has a name in "
           "this file: it is the \"solid ink\" case handled below. The "
           "radius must also be small enough that the illumination really "
           "is roughly constant across the window, which rules out making "
           "it enormous."),
     ("eq", r"r \;=\; \mathrm{clip}\!\left(\mathrm{round}"
            r"\!\left(\frac{H}{41}\right),\; 8,\; 40\right)",
      [("r", "the background window radius used above"),
       ("H", "the height of the page image in pixels"),
       (r"\mathrm{clip}(v,a,b)",
        "v forced into the range from a to b: the smaller of b and the "
        "larger of a and v"),
       ("41", "a divisor chosen so that the window comes out a few times "
              "the thickness of a pen stroke at the resolutions this "
              "pipeline works at"),
       ("8,\\;40", "hard floor and ceiling, so a very small or very large "
                   "input image still gets a sensible window")],
      None),
     ("p", "Scaling the radius with the image height rather than fixing it "
           "is what lets the same code handle a 1000-pixel-tall photograph "
           "and a 4000-pixel-tall one. A stroke in the larger photograph is "
           "four times as many pixels wide, so the window that must exceed "
           "it has to grow in step."),
     ("h", "Turning the ratio into a yes or no"),
     ("p", "The ratio image still has continuous values and a decision is "
           "needed. Otsu's method supplies the threshold, and it is "
           "unusually well suited here. Otsu's method assumes the values "
           "fall into two groups and picks the cut that separates them best "
           "in a precise statistical sense; see the entry for Otsu for the "
           "derivation. After the division the ratio image genuinely does "
           "have two groups, because the illumination that was smearing "
           "them together has been removed: a tall narrow spike just below "
           "1 for the paper, and a much smaller spread of low values for "
           "the ink. A pixel is ink when its ratio is below the threshold "
           "Otsu returns."),
     ("eq", r"\mathrm{mask}[y,x] \;=\; \big[\;\mathrm{ratio}[y,x] "
            r"\;<\; t_{\mathrm{Otsu}}(\mathrm{ratio})\;\big]",
      [(r"\mathrm{mask}[y,x]", "true when this pixel is taken to be ink"),
       (r"t_{\mathrm{Otsu}}(\mathrm{ratio})",
        "the threshold Otsu's method returns for the ratio image, that is, "
        "the value that best splits its two groups"),
       (r"[\;\cdot\;]",
        "square brackets around a condition mean 1 or true when the "
        "condition holds and 0 or false when it does not")],
      None),
     ("note", "Note that the threshold is computed from the ratio image, "
              "not from the original grey image. This is the whole point of "
              "the two-step design: a global method like Otsu's, which "
              "cannot adapt to position, is made to work on a photograph by "
              "first removing the position-dependent part of the signal."),
     ("h", "The solid-ink correction"),
     ("p", "One class of object defeats the division: a region of darkness "
           "much larger than the background window. The spine of a ring "
           "binder, a deep shadow along a fold, or the black edge of the "
           "desk behind the page are all far wider than 40 pixels. Inside "
           "such a region the window used for the denominator is entirely "
           "dark, so the estimated background is dark too, the ratio comes "
           "back near 1, and the interior is correctly rejected. The "
           "problem is the boundary, where the window straddles light and "
           "dark and the ratio dips, marking a band of false ink around "
           "every large dark object."),
     ("p", "These regions are found by their defining property: they are "
           "completely filled. Writing is not. A pen stroke is thin, so any "
           "small window placed on the page contains some paper even when "
           "it contains a stroke; a window inside a binder shadow contains "
           "nothing but shadow. Averaging the mask itself measures exactly "
           "this fill fraction."),
     ("eq", r"\mathrm{solid}[y,x] \;=\; \big[\;\mathrm{BoxMean}"
            r"(\mathrm{mask},\,4)[y,x] \;>\; 0.999\;\big]",
      [(r"\mathrm{solid}[y,x]",
        "true when this pixel lies in a completely filled dark region"),
       (r"\mathrm{BoxMean}(\mathrm{mask},4)",
        "the fraction of pixels marked as ink within a 9 by 9 window, "
        "obtained by averaging the mask treated as ones and zeros"),
       ("4", "the window radius, giving a 9 by 9 window: small enough that "
             "no ordinary pen stroke can fill it, large enough that noise "
             "does not"),
       ("0.999", "effectively a demand that every pixel in the window be "
                 "ink; it is not written as 1.0 only to be safe against "
                 "floating-point rounding in the average")],
      None),
     ("p", "The flagged cores are then grown slightly, with a second "
           "averaging test, and subtracted from the mask. Growing them is "
           "necessary because the false-ink band lies just outside the "
           "solid core, not inside it, so removing only the core would "
           "leave the band behind, which is the artefact being targeted."),
     ("h", "Why the ratio image is returned as well"),
     ("p", "Later stages need to ask how dark a piece of ink is, in order "
           "to tell a real pen stroke from a faint printed rule or a "
           "crease. The original grey values cannot answer that, because "
           "they still contain the illumination. The ratio image can, and "
           "it is already computed, so it is passed out rather than "
           "recomputed. DropArtifactPieces uses it for precisely this "
           "test."),
   ]),

 "PaperRegionMask": dict(
   inputs=[("g", "the page as a 2-D grey array in floating point"),
           ("P", "the line pitch in pixels, that is, the typical distance "
                 "between consecutive lines of writing; see LinePitch")],
   output="a boolean array the size of the page, true inside the sheet of "
          "paper and false outside it, shrunk slightly inward from the true "
          "paper edge.",
   theory=[
     ("h", "What it is for"),
     ("p", "A photograph of a page is rarely just the page. Around it there "
           "is desk, keyboard, hand, shadow and the dark gap at the edge of "
           "the sheet. None of that is writing, but much of it is darker "
           "than the paper and therefore survives into the ink mask. This "
           "function finds the outline of the sheet so that everything "
           "outside it can be discarded in one move, before any effort is "
           "spent trying to interpret it."),
     ("h", "The assumption being made"),
     ("p", "The method rests on one observation: of everything in the "
           "frame, the sheet of paper is both bright and large. The desk "
           "may be bright too, but it surrounds the page rather than "
           "forming one solid region with it, because a visible dark rim "
           "separates them, which is the shadow the sheet casts and the "
           "contrast at its cut edge. So the paper is the largest "
           "**connected** bright region, where connected means one can walk "
           "from any pixel of it to any other without leaving it."),
     ("h", "Step one: work small"),
     ("p", "The search is done on a copy of the page reduced by a factor of "
           "four in each direction, by averaging each 4 by 4 block of "
           "pixels into one. Two reasons. Speed: a sixteenth of the pixels "
           "means a sixteenth of the work, and the paper outline does not "
           "need pixel accuracy to be useful. Robustness: averaging "
           "suppresses noise and small features, so the result is dominated "
           "by the large-scale structure, which is what is being looked "
           "for. The reduction uses a block average rather than taking "
           "every fourth pixel, because taking every fourth pixel can land "
           "entirely in a gap and miss a thin dark rim completely, whereas "
           "an average over the block always registers it."),
     ("h", "Step two: find the dark surround"),
     ("eq", r"\mathrm{dark}[y,x] \;=\; \big[\; c[y,x] \;<\; "
            r"0.45 \cdot \mathrm{median}(c) \;\big]",
      [(r"\mathrm{dark}[y,x]",
        "true when this block of the shrunken page is part of the dark "
        "surround"),
       ("c", "the shrunken, locally averaged copy of the page"),
       (r"\mathrm{median}(c)",
        "the middle value of all the block brightnesses when sorted; it "
        "stands in for the brightness of the paper"),
       ("0.45", "how much darker than the paper a block must be before it "
                "counts as surround rather than page")],
      None),
     ("p", "The median is used rather than the mean because it is "
           "**robust**, meaning that a minority of extreme values barely "
           "moves it. In a photograph where a third of the frame is black "
           "desk, the mean brightness is dragged a long way down by the "
           "desk and no longer describes the paper at all, while the median "
           "still sits in the paper, because the paper is the single "
           "largest group of similar values. The threshold is a fraction of "
           "that median rather than an absolute grey level, which is again "
           "what makes it work under different lighting: in a dim "
           "photograph everything scales down together and the fraction is "
           "unchanged."),
     ("h", "The no-surround case"),
     ("p", "If less than 0.2 percent of the frame comes out dark, the "
           "conclusion is that there is no surround in the picture, because "
           "the page fills the whole frame or the image is a scan rather "
           "than a photograph. In that case the entire image is returned as "
           "paper. This matters because the \"largest bright region\" logic "
           "has nothing to find when there is no dark rim to separate "
           "regions in the first place, and forcing it to choose would pick "
           "an arbitrary piece of the page."),
     ("h", "Step three: the largest bright component"),
     ("p", "Connected-component labelling, described in the entry for "
           "LabelComponents, is run on the inverse of the dark mask, that "
           "is, on the bright blocks. It returns an array in which every "
           "bright block carries an integer label, with all the blocks of "
           "one connected region sharing a label. Counting how many blocks "
           "carry each label and taking the largest count identifies the "
           "sheet."),
     ("eq", r"\ell^{*} \;=\; \arg\max_{\ell \ge 1} \;"
            r"\big|\{\,(y,x) : \mathrm{lab}[y,x] = \ell \,\}\big|",
      [(r"\ell^{*}", "the label belonging to the sheet of paper"),
       (r"\mathrm{lab}[y,x]", "the component label of this block, with 0 "
                              "meaning it was not bright"),
       (r"\arg\max", "the value of the index at which the quantity is "
                     "largest, as opposed to the largest value itself"),
       (r"|\{\cdot\}|", "the number of elements in the set, here the number "
                        "of blocks carrying that label"),
       (r"\ell \ge 1", "labels start at 1, so starting the search at 1 "
                       "excludes the background")],
      None),
     ("h", "Step four: back to full size"),
     ("p", "The chosen region is a mask over blocks, and what is needed is "
           "a mask over pixels. Each block stands for a 4 by 4 square of "
           "the original, so the block mask is expanded by replacing every "
           "entry with a 4 by 4 square of copies of itself. This is a "
           "**Kronecker product** with a 4 by 4 block of ones, which is the "
           "standard name for exactly that operation, and in practice it is "
           "nearest-neighbour upsampling: a hard-edged enlargement with no "
           "smoothing, which is the correct choice because the value being "
           "enlarged is a yes-or-no flag and a smoothed version of it would "
           "be meaningless."),
     ("h", "Step five: shrink inward"),
     ("p", "Finally the mask is eroded, that is, every pixel within a "
           "radius of the paper boundary is removed from it, using a radius "
           "of 0.4 line pitches. The reason is that the boundary itself is "
           "the single least trustworthy part of the estimate. It was found "
           "on a coarse grid, the sheet's edge is darkened by shadow over a "
           "band several pixels wide, and the page may curl. Pulling the "
           "accepted region inward by a fraction of a line's height costs "
           "nothing, because people do not write right up to the cut edge "
           "of a sheet, and it reliably excludes the edge shadow that would "
           "otherwise be interpreted as a long horizontal stroke."),
     ("note", "Expressing the shrink in line pitches rather than pixels is "
              "a habit worth noticing, and it recurs throughout the "
              "segmenter. The pitch is the natural unit of the problem: it "
              "is set by the size of the writing, so a quantity measured in "
              "pitches means the same thing on a photograph taken close up "
              "as on one taken from further away."),
   ]),

 "LinePitch": dict(
   inputs=[("mask", "the ink mask of the page, a boolean 2-D array")],
   output="a pair. The first item is the line pitch, an integer number of "
          "pixels giving the typical vertical distance from one line of "
          "writing to the next. The second is the smoothed row profile, an "
          "array with one value per image row, which LineCentres reuses to "
          "locate the lines.",
   theory=[
     ("h", "What it is for, and why it comes first"),
     ("p", "Almost every decision later in the segmenter is a judgement of "
           "size: is this blob too tall to be a letter, is this gap wide "
           "enough to be a word break, is this dot close enough to that "
           "stem to belong to it. Expressing such rules in pixels would tie "
           "them to one camera at one distance. Expressing them as "
           "fractions of the line pitch makes them scale-free, because the "
           "pitch grows and shrinks with the writing. So the pitch is "
           "measured once, early, and becomes the ruler for everything "
           "else. The symbol P is used for it throughout the file."),
     ("h", "Background: what a projection profile is"),
     ("p", "A **projection profile** is what results from collapsing a 2-D "
           "image onto one axis by summing along the other. The horizontal "
           "projection profile used here has one number per image row: how "
           "many ink pixels that row contains."),
     ("eq", r"\mathrm{prof}[y] \;=\; \sum_{x} \mathrm{mask}[y,x]",
      [(r"\mathrm{prof}[y]", "the profile value for row y, that is, the "
                             "count of ink pixels in that row"),
       (r"\mathrm{mask}[y,x]", "1 where the pixel is ink, 0 where it is "
                               "not"),
       (r"\sum_{x}", "a sum over every column x of that row")],
      None),
     ("p", "The profile of a page of handwriting has a characteristic "
           "shape. Rows passing through the bodies of letters cross nearly "
           "every letter on the line and so collect a great deal of ink. "
           "Rows in the white space between two lines cross almost nothing "
           "and collect very little. The profile is therefore a run of "
           "peaks separated by troughs, one peak per line of writing, and "
           "the distance between consecutive peaks is the pitch. Using row "
           "projections to find text lines is a long-standing technique in "
           "document analysis \\cite{postl1986}."),
     ("p", "Columns are trimmed before summing: the leftmost 5 percent and "
           "everything past 97 percent of the width are ignored. The margins "
           "are where page edges, binder holes and the shadow at the spine "
           "live, and a vertical artefact there contributes to every row "
           "equally, which raises the whole profile and flattens the "
           "contrast between peak and trough that the method depends on."),
     ("h", "Smoothing, and why it is necessary"),
     ("p", "The raw profile is jagged. Within one line of writing the count "
           "jumps about from row to row according to which letters happen "
           "to have a horizontal stroke at that exact height, so a single "
           "line of text produces not one clean peak but a cluster of small "
           "ones. Counting those as separate lines would be wrong. "
           "Smoothing replaces each value with a weighted average of its "
           "neighbours, which merges the cluster into one broad peak while "
           "leaving the large gaps between lines intact."),
     ("eq", r"k[j] \;=\; \exp\!\left(-\tfrac{1}{2}"
            r"\left(\frac{j}{\sigma}\right)^{\!2}\right),"
            r"\qquad j = -12,\,\dots,\,12, \qquad \sigma = 4",
      [("k[j]", "the unnormalised weight given to a neighbour j rows away"),
       ("j", "the offset from the row being smoothed, running from -12 to "
             "+12"),
       (r"\sigma", "the standard deviation of the Gaussian in rows; it sets "
                   "how far the smoothing reaches, here 4 rows"),
       (r"\exp(\cdot)", "the exponential function, which here makes the "
                        "weight fall away smoothly and symmetrically with "
                        "distance")],
      None),
     ("eq", r"\mathrm{sm}[y] \;=\; \frac{\sum_{j} k[j]\,"
            r"\mathrm{prof}[y+j]}{\sum_{j} k[j]}",
      [(r"\mathrm{sm}[y]", "the smoothed profile at row y"),
       (r"\mathrm{prof}[y+j]", "the raw profile j rows away from y"),
       (r"\sum_{j} k[j]",
        "the sum of the weights, dividing by which keeps the smoothed "
        "profile on the same scale as the raw one, so that a flat input "
        "comes back unchanged instead of multiplied by the weight total")],
      None),
     ("p", "This operation is a **convolution**: slide a small set of "
           "weights along a signal, and at each position output the "
           "weighted sum of the values under it. The particular weights "
           "here form a Gaussian, or bell curve, which is the usual choice "
           "for smoothing because it is smooth itself and so does not "
           "introduce ripples of its own. A width of 4 rows is chosen to be "
           "comparable to the thickness of a pen stroke, which is the scale "
           "of the jaggedness being removed, and much smaller than the "
           "distance between lines, which is the structure being kept."),
     ("h", "Method one: the gaps between peaks"),
     ("p", "A row is a **local maximum**, or peak, when it is at least as "
           "large as the row above and strictly larger than the row below, "
           "and in addition exceeds 6 percent of the largest value anywhere "
           "in the profile. The last condition rejects the small bumps that "
           "noise produces in the white space between lines. The asymmetry, "
           "at least as large on one side and strictly larger on the other, "
           "is a detail that matters: if both tests were \"at least as "
           "large\", a flat-topped peak several rows wide would report every "
           "one of those rows as a separate peak."),
     ("p", "With two or more peaks found, the pitch is the **median** of "
           "the differences between consecutive peak positions."),
     ("eq", r"P \;=\; \mathrm{median}\big(\,y_{i+1} - y_{i} \;:\; "
            r"i = 1,\dots,m-1 \,\big)",
      [("P", "the line pitch in pixels"),
       ("y_i", "the row position of the i-th peak, in increasing order"),
       ("m", "how many peaks were found"),
       (r"\mathrm{median}", "the middle value of the differences once "
                            "sorted")],
      None),
     ("p", "The median rather than the mean, again for robustness. If the "
           "page contains a paragraph break, one of the gaps is two or "
           "three times the others. The mean would be pulled upward by that "
           "one value and would overstate the pitch for the whole page; the "
           "median ignores it entirely, since it only cares which value "
           "sits in the middle."),
     ("h", "Method two: autocorrelation"),
     ("p", "When fewer than two peaks are found there are no gaps to "
           "measure, and a different line of attack is needed. "
           "**Autocorrelation** asks how much a signal resembles a shifted "
           "copy of itself. For a repeating signal the resemblance is high "
           "when the shift equals the period of the repetition, because "
           "peaks then land on peaks."),
     ("eq", r"\mathrm{ac}[\tau] \;=\; \sum_{y} a[y]\,a[y+\tau],"
            r"\qquad a[y] \;=\; \mathrm{sm}[y] - \overline{\mathrm{sm}}",
      [(r"\mathrm{ac}[\tau]",
        "the autocorrelation at shift tau: large when the profile lines up "
        "with itself after being moved tau rows"),
       (r"\tau", "the shift, called the lag, measured in rows"),
       ("a[y]", "the profile with its average removed, so that the "
                "measurement responds to the pattern of variation rather "
                "than to the overall level"),
       (r"\overline{\mathrm{sm}}", "the mean of the smoothed profile")],
      None),
     ("p", "Subtracting the mean is essential. Without it every product in "
           "the sum is positive, the autocorrelation is dominated by the "
           "constant part of the signal, and it decreases steadily with lag "
           "simply because fewer terms overlap, so no peak corresponding to "
           "the period is visible. After subtraction the values straddle "
           "zero, and the sum is large only when peaks genuinely align with "
           "peaks and troughs with troughs."),
     ("p", "The first clear peak at a lag of at least 15 rows, reaching at "
           "least 70 percent of the maximum in the searched range, is taken "
           "as the pitch. The lower bound of 15 skips the trivial peak at "
           "zero lag, where any signal matches itself perfectly, along with "
           "the shoulder beside it caused by the smoothing. The search is "
           "also limited to a third of the signal length, because at large "
           "lags very few rows overlap and the estimate becomes noise."),
     ("h", "Method three: letter height"),
     ("p", "If even that fails, as it does for a crop containing a single "
           "line, the pitch is inferred from the size of the letters "
           "themselves. Connected components are labelled, each one's "
           "height is measured, pieces smaller than 8 pixels in area or "
           "under 3 pixels tall are discarded as noise, and the median "
           "height of what remains is multiplied by 2.2."),
     ("eq", r"P \;=\; \max\big(15,\; \mathrm{round}\,"
            r"(2.2 \cdot \mathrm{median}(h))\big)",
      [("P", "the line pitch in pixels"),
       ("h", "the heights of the accepted connected components"),
       ("2.2", "the ratio of line pitch to typical component height "
               "observed for ordinary handwriting"),
       ("15", "a floor, below which the estimate is treated as unusable")],
      None),
     ("p", "The factor 2.2 comes from the structure of writing. A typical "
           "connected piece of ink is roughly one x-height tall, the "
           "x-height being the height of a letter like x or o with no "
           "ascender or descender. A line of writing occupies the x-height "
           "plus the ascender zone above it plus the descender zone below, "
           "and lines are separated by a further gap, which together come "
           "to a little over twice the x-height. The estimate is coarse, "
           "but it is only reached when the two better methods have "
           "already failed."),
     ("h", "The final fallback"),
     ("p", "With no usable ink at all, 48 pixels is returned. This is an "
           "arbitrary number and is not expected to be right. Its purpose "
           "is to keep the caller running on a blank page instead of "
           "failing on a division by zero, and a blank page produces no "
           "crops regardless of what pitch is assumed, so the value cannot "
           "do any harm."),
     ("note", "The cascade of three methods, each a fallback for the last, "
              "is worth reading as a design pattern rather than as "
              "clutter. The first method is the most accurate but needs at "
              "least two lines; the second needs only a repeating pattern; "
              "the third needs only some ink. Each is tried in order of "
              "preference, so the accurate method is used whenever it "
              "applies and the crude ones only when nothing better is "
              "possible."),
   ]),

 "LeastCostSeam": dict(
   inputs=[("cost", "a 2-D array in which every entry is the price of "
                    "passing through that pixel; here, large where there is "
                    "ink and small where there is not"),
           ("step", "the largest number of rows the path may move up or "
                    "down when advancing one column, default 3")],
   output="an array with one entry per column, giving the row the path "
          "occupies in that column: the cheapest left-to-right route across "
          "the array.",
   theory=[
     ("h", "What it is for"),
     ("p", "Two lines of handwriting have to be separated from one another. "
           "A straight horizontal cut will not do it, because the "
           "descender of a g on the upper line hangs below the ascender of "
           "an l on the lower one, and any single row that clears the g "
           "also slices the l. What is needed is a wavy cut that dodges "
           "around the ink: it should dip below a descender and rise above "
           "an ascender, as long as it keeps moving left to right. That "
           "wavy cut is called a **seam**, and finding the cheapest one is "
           "what this function does. The technique is seam carving, "
           "introduced for resizing images \\cite{avidan2007} and adapted "
           "to the separation of handwritten text lines "
           "\\cite{arvanitopoulos2014}."),
     ("h", "The problem stated precisely"),
     ("p", "A seam is a choice of one row for each column, with the "
           "restriction that consecutive choices differ by no more than "
           "`step` rows, which is what makes the result a connected curve "
           "rather than a scatter of disconnected points. Its cost is the "
           "sum of the cost array along it. The task is to find the seam of "
           "least total cost."),
     ("eq", r"\mathrm{Cost}(y_1,\dots,y_W) \;=\; \sum_{x=1}^{W} "
            r"\mathrm{cost}[y_x,\,x], \qquad "
            r"|y_{x+1} - y_{x}| \le \mathrm{step}",
      [("y_x", "the row the seam occupies in column x"),
       ("W", "the number of columns"),
       (r"\mathrm{cost}[y,x]", "the price of the pixel at row y, column x"),
       (r"\mathrm{step}", "the maximum vertical move allowed between "
                          "neighbouring columns")],
      None),
     ("h", "Why brute force is impossible"),
     ("p", "Counting the candidates shows why a direct search is out of the "
           "question. With h rows to choose from in the first column and "
           "up to 2·step+1 choices at each of the remaining W−1 columns, "
           "the number of seams is about h(2·step+1)^(W−1). For a modest "
           "page half a size down, with 1200 columns and step 7, that is "
           "15 raised to the power 1199, a number with over a thousand "
           "digits. No computer will enumerate them."),
     ("h", "The idea that makes it easy: dynamic programming"),
     ("p", "**Dynamic programming** is a method for problems that can be "
           "solved by building up the answer one step at a time, where the "
           "best way to finish does not depend on how the start was "
           "reached \\cite{bellman1954}. That condition holds here. "
           "Suppose the cheapest way to reach row y of column x has been "
           "computed, for every row y. The cheapest way to reach a "
           "particular row of column x+1 must arrive from one of the rows "
           "of column x that is within `step` of it, and once it is known "
           "how cheaply each of those rows can be reached, the best arrival "
           "is just the smallest of those costs plus the price of the new "
           "pixel. Nothing about the earlier part of the route matters "
           "beyond its total cost. So the whole column can be solved from "
           "the previous column alone."),
     ("eq", r"A[y,x] \;=\; \mathrm{cost}[y,x] \;+\; "
            r"\min_{|d| \le \mathrm{step}} \Big( A[y-d,\,x-1] "
            r"\;+\; \lambda\,|d| \Big)",
      [("A[y,x]",
        "the accumulated cost: the price of the cheapest route that starts "
        "at the left edge and ends at row y of column x"),
       (r"\mathrm{cost}[y,x]", "the price of the current pixel"),
       ("d", "the vertical move made to arrive here, so the route came from "
             "row y−d of the previous column"),
       (r"\mathrm{step}", "the limit on that move"),
       (r"\lambda", "a small penalty per row of vertical movement, 0.02 in "
                    "this implementation"),
       (r"\min", "the smallest value over all the choices of d")],
      None),
     ("p", "The first column needs no recursion, since a route that ends "
           "there has visited only that one pixel, so A in column 1 is just "
           "the cost in column 1. Every later column follows from its "
           "predecessor by the formula above. The work is one pass over the "
           "array, with 2·step+1 comparisons per pixel, instead of the "
           "astronomical enumeration. That is the entire gain of dynamic "
           "programming: it replaces an exponential count of paths with a "
           "linear count of pixels."),
     ("h", "The movement penalty"),
     ("p", "The term λ|d| charges a small amount for each row of vertical "
           "movement. Without it, any two routes through equally empty "
           "space would cost the same, and the seam would wander about "
           "aimlessly in white space, picking an arbitrary one of the many "
           "equally cheap options. The penalty breaks those ties in favour "
           "of going straight, so that the seam moves only when moving buys "
           "it something, namely avoiding ink. At 0.02 per row it is far "
           "smaller than the cost of crossing a stroke, so it never "
           "persuades the seam to cut through a letter in order to stay "
           "level; it only decides between routes that are otherwise "
           "equivalent. This is a **regularisation** term, which is the "
           "general name for a small extra cost added to express a "
           "preference among solutions that the main objective cannot "
           "distinguish."),
     ("h", "Recovering the path"),
     ("p", "The accumulated cost gives the price of the best seam but not "
           "its route, so as each cell is filled the winning move is "
           "recorded in a second array. When the pass is complete, the "
           "lowest value in the final column is the end of the best seam, "
           "and following the stored moves backwards from there reconstructs "
           "the whole path, column by column, right to left. This is "
           "**backtracking**, and it is the standard companion to dynamic "
           "programming: the forward pass computes the value of the best "
           "solution, the backward pass reads off what that solution "
           "actually was."),
     ("h", "How it is made fast in practice"),
     ("p", "Written as a loop over rows inside a loop over columns this "
           "would be slow in Python. Instead, for each column, the "
           "candidate costs from all 2·step+1 possible moves are formed as "
           "whole shifted copies of the previous column, stacked into a "
           "small array, and the minimum is taken down the stack in one "
           "operation. Shifting a column by d rows and adding the penalty "
           "handles every row's d-th candidate at once. Only the loop over "
           "columns remains in Python, because columns are genuinely "
           "sequential: column x cannot be computed before column x−1."),
     ("note", "The shifted copies are filled with infinity where the shift "
              "runs off the end of the array. Infinity is the natural value "
              "for an impossible move, because taking a minimum "
              "automatically never selects it, with no need for a special "
              "case in the code."),
   ]),

 "SkewScore": dict(
   inputs=[("mask", "the ink mask of the page"),
           ("angleDeg", "the trial rotation angle in degrees")],
   output="a single number: how sharply the ink piles into rows when the "
          "page is considered at this angle. Larger is better aligned.",
   theory=[
     ("h", "What it is for"),
     ("p", "A photographed page is never perfectly square to the camera, so "
           "the lines of writing run slightly uphill or downhill. Every "
           "later stage assumes they are horizontal: the row profile is "
           "used to find lines, horizontal seams are used to separate them, "
           "and a crop is a horizontal strip. A tilt of even two degrees "
           "smears the profile enough to merge neighbouring lines. This "
           "function provides the measurement that lets the tilt be found: "
           "given a candidate angle, it scores how level the page would look "
           "at that angle."),
     ("h", "The principle: concentration"),
     ("p", "Consider the row profile again, the count of ink pixels per "
           "row. When the page is level, every letter on a line contributes "
           "to the same few rows, so those rows have very large counts and "
           "the rows between lines have almost none. When the page is "
           "tilted, the left end of a line sits higher than the right end, "
           "so the same ink is spread over many more rows, each with a "
           "smaller count. The total amount of ink is identical either way, "
           "since rotating the page does not create or destroy pixels. What "
           "changes is how **concentrated** the profile is. So the correct "
           "angle is the one that concentrates the profile most, and all "
           "that is needed is a number that measures concentration."),
     ("h", "Why the sum of squares measures concentration"),
     ("p", "Squaring, then summing, does it. Squaring rewards large values "
           "disproportionately: doubling a count quadruples its "
           "contribution. So for a fixed total, the sum of squares is "
           "largest when the total is piled into as few entries as possible "
           "and smallest when it is spread evenly."),
     ("eq", r"S(\theta) \;=\; \sum_{k} \mathrm{prof}_{\theta}[k]^{2}",
      [(r"S(\theta)", "the skew score at trial angle theta"),
       (r"\mathrm{prof}_{\theta}[k]",
        "the number of ink pixels that fall into row index k once the page "
        "is considered rotated by theta"),
       ("k", "the row index in the rotated frame")],
      None),
     ("p", "A numerical illustration makes the behaviour concrete. Suppose "
           "600 ink pixels belong to one line. Level, they might fall into "
           "4 rows of 150, giving a score of 4·150² = 90{,}000. Tilted, the "
           "same 600 might spread over 20 rows of 30, giving 20·30² = "
           "18{,}000. The totals match at 600 but the scores differ "
           "fivefold, and the larger score correctly identifies the level "
           "case. The sum of squares is used in the same spirit elsewhere "
           "in the project, where EstimateSlantDeg uses it to find the "
           "slant of letters rather than the skew of lines."),
     ("h", "Scoring an angle without rotating the image"),
     ("p", "Rotating the whole image for each candidate angle would mean "
           "interpolating every pixel, seventeen times over for the "
           "seventeen angles tried. That is unnecessary. Only the row index "
           "each ink pixel would land in is needed, not an image, and that "
           "index follows from one line of trigonometry. The standard "
           "rotation of a point about a centre gives the rotated row as"),
     ("eq", r"k \;=\; \mathrm{round}\Big( (y - c_y)\cos\theta "
            r"\;-\; (x - c_x)\sin\theta \;+\; c_y \Big)",
      [("k", "the row the pixel lands in after rotation"),
       ("y,\\,x", "the pixel's original row and column"),
       ("c_y,\\,c_x", "the centre of rotation, taken as the centre of the "
                      "image"),
       (r"\theta", "the rotation angle in radians, converted from the "
                   "degrees supplied"),
       (r"\cos,\,\sin", "the cosine and sine functions, which express how "
                        "much of each original axis contributes to the new "
                        "vertical position"),
       (r"\mathrm{round}", "rounding to the nearest whole number, since "
                           "rows are counted in whole pixels")],
      None),
     ("p", "Only the row is computed; the column is never needed, because "
           "the profile sums over columns anyway. Counting how many pixels "
           "land in each row is then a single tally of the k values, so the "
           "cost of scoring one angle is proportional to the number of ink "
           "pixels rather than to the number of pixels in the image. Since "
           "ink is a small fraction of a page, that is a large saving on "
           "top of avoiding the interpolation."),
     ("h", "A subtle trap: never clip the row index"),
     ("p", "Rotation moves some pixels outside the original image "
           "rectangle, so k can come out negative or beyond the image "
           "height. The obvious response, forcing k into the valid range, "
           "is badly wrong, and the code carries a comment saying so. "
           "Clipping piles every out-of-range pixel into the first or last "
           "row. That creates an enormous artificial count in one row, and "
           "because the score squares the counts, this fake spike scores "
           "higher than any genuine line of text. The effect grows with the "
           "angle, since larger angles push more pixels out of range, so "
           "the search would reliably select the most extreme angle "
           "offered. The code instead shifts the indices so the smallest "
           "becomes zero, which keeps every pixel in its own distinct row "
           "and lets the profile grow as tall as it needs to be."),
     ("note", "This is a good example of why a plausible-looking defensive "
              "measure can be worse than none. Clipping an index is "
              "usually the safe thing to do; here it silently converts a "
              "correct measurement into one that is guaranteed to give the "
              "wrong answer."),
   ]),

 "Deskew": dict(
   inputs=[("page", "the photograph, as a 2-D grey array or a 3-D colour "
                    "array of 8-bit values")],
   output="five items: the straightened full-resolution page, the "
          "straightened half-resolution copy, the ink mask of that copy, the "
          "ratio image, and the angle in degrees that was applied.",
   theory=[
     ("h", "What it is for"),
     ("p", "This is the first stage of the segmenter, and its job is to "
           "hand the rest of the pipeline a page whose lines of writing run "
           "horizontally, together with the derived images every later stage "
           "needs. It combines three things: reducing the page to grey, "
           "searching for the tilt angle, and applying the rotation."),
     ("h", "Step one: colour to grey"),
     ("p", "A colour photograph arrives as three values per pixel, for red, "
           "green and blue. The whole segmenter works on a single "
           "brightness value, so the three channels are averaged. A simple "
           "mean is used rather than one of the perceptual weightings that "
           "weight green more heavily, because the distinction being drawn "
           "is dark pen against light paper, which shows up in all three "
           "channels, and a plain mean has the advantage of treating blue "
           "and black ink alike."),
     ("h", "Step two: work at half size"),
     ("p", "Every decision in the segmenter is taken on a copy of the page "
           "reduced by a factor of two in each direction, which is a "
           "quarter of the pixels, so everything runs about four times "
           "faster. The decisions involved concern whole lines and whole "
           "letters, which are many pixels across, so halving the "
           "resolution does not blur the evidence they rest on. The full "
           "resolution page is kept because the final crops are cut from "
           "it, so the recogniser still sees every pixel the camera "
           "captured."),
     ("p", "The reduction averages each 2 by 2 block rather than taking "
           "every second pixel. Point sampling would risk a thin pale "
           "stroke falling between the samples and disappearing entirely, "
           "whereas any ink inside a block darkens that block's average and "
           "survives. This is the same argument that applies to the "
           "widening of a resampling filter when minifying, discussed in "
           "the entry for _Weights."),
     ("h", "Step three: search for the angle"),
     ("p", "The angle is found by a **grid search**, which means simply "
           "trying every candidate in a fixed list and keeping the best. "
           "The candidates run from −4 to +4 degrees in steps of 0.5, which "
           "is seventeen angles, each scored by SkewScore, with the highest "
           "score winning."),
     ("eq", r"\theta^{*} \;=\; \arg\max_{\theta \,\in\, \{-4,\,-3.5,\,"
            r"\dots,\,4\}} \; S(\theta)",
      [(r"\theta^{*}", "the chosen rotation angle in degrees"),
       (r"S(\theta)", "the concentration score at that angle, from "
                      "SkewScore"),
       (r"\arg\max", "the angle at which the score is largest")],
      None),
     ("p", "Two choices in that range deserve justification. The limit of 4 "
           "degrees reflects how a page is actually photographed: someone "
           "aiming a phone at a sheet on a desk is square to it within a "
           "few degrees, and a search that allowed 45 degrees would mostly "
           "offer opportunities to pick a wrong answer. The step of 0.5 "
           "degrees is chosen against the consequence of residual error. "
           "Over a page 1500 pixels wide, half a degree of tilt lifts one "
           "end relative to the other by 1500·tan(0.5°) ≈ 13 pixels, which "
           "is well under the height of a line of writing, so the remaining "
           "tilt cannot merge neighbouring lines. A finer step would cost "
           "more scoring for no benefit the later stages could use."),
     ("p", "The search runs on a quarter-size mask, halved again from the "
           "working copy, giving sixteen times fewer pixels to tally per "
           "angle. The angle of a whole page is a very coarse property, "
           "determined by the global arrangement of its lines, so it "
           "survives aggressive reduction intact."),
     ("h", "Step four: apply the rotation, and the fill values"),
     ("p", "The chosen angle is applied to both the full-size and half-size "
           "pages with Rotate. Rotating a rectangle leaves wedges at the "
           "corners that correspond to no original pixel, and these have to "
           "be filled with something. The fill is 255, white, because what "
           "lies just outside the frame of a photograph of a page is more "
           "page, and declaring it white means the later stages see paper "
           "rather than a black triangle that would read as a huge ink "
           "blob. Where a mask rather than a grey image is rotated the fill "
           "is 0 instead, for the same underlying reason: in a mask, 0 means "
           "no ink, so filling with 0 adds nothing that could be mistaken "
           "for writing."),
     ("h", "Step five: derive the mask"),
     ("p", "Finally InkMask is run on the straightened half-size page to "
           "produce the ink mask and the ratio image. Note the ordering: "
           "the mask is computed after rotation, not before. Rotating a "
           "mask would require interpolating yes-or-no values, which "
           "produces fractional ink along every stroke edge and forces an "
           "arbitrary second threshold to clean up. Rotating the grey image "
           "and then thresholding avoids that entirely, since interpolating "
           "grey values is exactly what interpolation is for."),
     ("note", "The ink mask is computed twice in this function, once before "
              "the rotation to drive the angle search and once after it for "
              "the output. The first is thrown away. That is deliberate: "
              "the search needs a mask to score, and the output must "
              "describe the rotated page, and no single mask can do both."),
   ]),

 "RemovePrintedStructure": dict(
   inputs=[("mask", "the ink mask of the page"),
           ("g", "the grey page at the same scale, used to locate the "
                 "sheet")],
   output="five items: the cleaned mask, the paper-region mask, the line "
          "pitch, how many ink pixels lay outside the paper, and how many "
          "were removed as printed structure.",
   theory=[
     ("h", "What it is for"),
     ("p", "A sheet of lined or gridded paper carries printing that was "
           "never written by hand: the ruled lines, the red margin line, "
           "the holes and the darkened spine where the page meets the "
           "binding. All of it is ink as far as the mask is concerned, and "
           "all of it is poison for the stages that follow. A ruled line "
           "runs the full width of the page, so it joins every letter on a "
           "line into one connected blob, which destroys the per-letter "
           "analysis that the cutter depends on, and it adds a huge "
           "horizontal stroke to every row profile. This function removes "
           "that printing while leaving the writing intact."),
     ("h", "The distinguishing property: length"),
     ("p", "The property that separates printing from writing is not "
           "darkness, since a ruled line can be as dark as a pencil stroke, "
           "and not thickness, since both are a pen's width. It is "
           "**length**. A ruled line crosses the whole page in one "
           "unbroken straight run. Handwriting contains straight pieces too, "
           "but short ones: the longest horizontal stroke in ordinary "
           "writing is the crossbar of a t or the join between two letters, "
           "neither more than a fraction of a line's height. So a threshold "
           "on run length, expressed in line pitches, separates the two "
           "reliably."),
     ("eq", r"\mathrm{flat} \;=\; \mathrm{LongRunMask}"
            r"\big(\mathrm{mask},\; \lfloor 1.25\,P \rfloor,\;"
            r"\text{horizontal}\big)",
      [(r"\mathrm{flat}", "the mask of pixels belonging to a long "
                          "horizontal run"),
       (r"\mathrm{LongRunMask}", "marks every pixel that lies in an "
                                 "unbroken run of set pixels at least the "
                                 "given length; see its own entry"),
       ("P", "the line pitch in pixels"),
       ("1.25", "the length threshold as a multiple of the pitch"),
       (r"\lfloor \cdot \rfloor", "rounding down to a whole number of "
                                  "pixels")],
      None),
     ("p", "The threshold of 1.25 pitches sits in a wide gap. A ruled line "
           "is tens of pitches long, and the longest horizontal stroke in "
           "handwriting is well under one pitch, since a letter is about "
           "half a pitch wide. Anything between about 1 and several pitches "
           "would give the same result, so the exact figure is not "
           "delicate. The vertical direction gets 1.4 pitches, slightly "
           "more, because a tall letter like a capital L or an f with a "
           "long descender genuinely can run more than a pitch vertically, "
           "and the extra margin keeps such letters out of the deletion."),
     ("h", "Tolerating a tilt"),
     ("p", "A ruled line on a page that is still a fraction of a degree off "
           "level is not a single row of pixels. It drifts downward as it "
           "crosses the page, so any one row contains only the piece of it "
           "that happens to lie in that row, and a run-length test applied "
           "row by row sees a series of short runs rather than one long "
           "one, and deletes nothing. The fix is to dilate the mask by one "
           "pixel vertically before the test, so that each row contains the "
           "line's pixels from the rows immediately above and below as well "
           "and the run becomes unbroken again."),
     ("p", "But dilating also thickens the writing, and a test applied to a "
           "thickened mask would mark pixels that were never ink. So the "
           "result is dilated back and then intersected with the original "
           "mask, which restricts the deletion to pixels that really were "
           "ink to begin with. In morphological terms the sequence dilate, "
           "test, dilate, intersect is a closing used as a detector rather "
           "than as a filter: the dilation makes the feature visible to the "
           "test, and the intersection confines the consequences to the "
           "original shape."),
     ("p", "The vertical pass uses a radius of 3 rather than 1 because the "
           "structures it targets, the shadow at the spine of a notebook "
           "and the edge of the sheet, are not straight. They wobble by "
           "several pixels across the height of the page, so a tolerance of "
           "one pixel is not enough to make them read as one continuous "
           "run."),
     ("h", "Masking off the surround first"),
     ("p", "Before any of this, PaperRegionMask locates the sheet and "
           "everything outside it is deleted from the mask in one "
           "intersection. This is done first because it is the cheapest and "
           "most decisive filter available: the desk, the hand and the "
           "shadow beyond the page account for a large share of the false "
           "ink on a photograph, and removing them wholesale means the "
           "subtler tests that follow are not distracted by them."),
     ("h", "What is returned, and why the counts"),
     ("p", "Besides the cleaned mask, the function returns the pitch and "
           "the paper mask, both already computed here and needed later, "
           "and two counts: how many ink pixels lay outside the paper and "
           "how many were deleted as printed structure. The counts feed the "
           "diagnostics dictionary that SegmentLines returns. They exist "
           "because a failure of this stage is otherwise invisible: if the "
           "paper mask is wrong, the ink-outside-paper count is enormous, "
           "and if the run-length thresholds are wrong for a given page, "
           "the removed-pixel count is far larger or smaller than expected. "
           "Reporting them makes a silent failure a visible one."),
   ]),

 "DropArtifactPieces": dict(
   inputs=[("mask", "the ink mask after the printed structure has been "
                    "removed"),
           ("ratio", "the ratio image from InkMask, used to judge how dark "
                     "each piece is"),
           ("P", "the line pitch in pixels"),
           ("H", "the mask height"),
           ("W", "the mask width")],
   output="five items: the mask with the artefacts deleted, the component "
         "label array, the number of components, the area of each, and how "
         "many pieces were dropped.",
   theory=[
     ("h", "What it is for"),
     ("p", "After the printing has been removed, what remains is mostly "
           "handwriting but not entirely. Creases cast faint shadows. The "
           "fold at the centre of a spread leaves a dark line. The cut edge "
           "of the sheet leaves a sliver. Fragments of the ruling survive "
           "where a letter crossed it and broke the run. This function "
           "examines the leftovers one piece at a time and deletes those "
           "that are not writing."),
     ("h", "Why judge whole pieces, not pixels"),
     ("p", "This is the central design decision and it is worth stating "
           "plainly. Every test in this function is applied to a connected "
           "component as a unit, and a component is either kept whole or "
           "deleted whole. The alternative, testing each pixel on its own "
           "merits, produces half-erased letters, and a half-erased letter "
           "is worse than either outcome: it is still there, so it still "
           "has to be interpreted, but it no longer looks like the letter "
           "it is. Deciding per piece means the worst case is a missing "
           "letter, which the recogniser handles as a gap, rather than a "
           "corrupted one, which it confidently misreads."),
     ("h", "Closing before labelling"),
     ("p", "The components are found not on the mask itself but on a closed "
           "copy, dilated by about a tenth of a pitch and then eroded by "
           "the same amount. The purpose is to make a letter that the "
           "threshold broke into two pieces, where the pen lifted or the "
           "ink ran thin, count as one piece for the purposes of these "
           "tests. Otherwise the two halves are judged separately and each "
           "half, being smaller and thinner than a letter, is more likely "
           "to be mistaken for an artefact. The tests then read their "
           "statistics from the original mask but group pixels by the "
           "labels from the closed one, so the closing influences the "
           "grouping without inflating the measurements."),
     ("h", "The geometric tests"),
     ("p", "Several tests follow from the simple fact that a letter has a "
           "characteristic size, and the pitch gives the scale for it."),
     ("items", [
       "**Too tall:** a piece more than 2.2 pitches high cannot be a "
       "letter, because that is taller than a whole line of writing "
       "including its ascender and descender zones.",
       "**Too thin:** a piece half a pitch wide or less, combined with "
       "another suspicious property, is a sliver rather than a letter.",
       "**A vertical bar:** a piece under 0.12 pitches wide and over 1.2 "
       "pitches tall is deleted outright. No letter has that aspect ratio; "
       "a printed margin rule does, and so does the fold at the centre of a "
       "page.",
       "**Touching an edge:** a piece reaching within half a pitch of the "
       "paper boundary is suspect, because writing does not start at the "
       "cut edge of a sheet. Touching the left edge is enough on its own, "
       "since the binding is there; touching the right or the top or bottom "
       "is only grounds for deletion in combination with being thin.",
       "**Far left of the text:** the fifth percentile of the left edges of "
       "all the letter-sized pieces estimates where the writing begins, and "
       "a piece lying more than 1.2 pitches to the left of that is in the "
       "margin, not in the text.",
     ]),
     ("p", "The fifth percentile, rather than the minimum, estimates the "
           "left margin of the text. A **percentile** is the value below "
           "which a given fraction of the data falls, so the fifth "
           "percentile is the value exceeded by 95 percent of the left "
           "edges. Using the minimum would let a single surviving artefact "
           "define the margin and so protect every other artefact to its "
           "right; using a percentile means one stray piece cannot shift "
           "the estimate."),
     ("h", "The darkness test"),
     ("p", "Geometry alone cannot separate a faint printed rule from a pen "
           "stroke of the same shape, so the ratio image is brought in. The "
           "mean ratio over a piece measures how dark it is relative to the "
           "paper around it, and the comparison is made against the same "
           "Otsu threshold that decided what counted as ink in the first "
           "place."),
     ("eq", r"\overline{\mathrm{ratio}}_{\ell} \;=\; \frac{1}{A_{\ell}}"
            r"\sum_{(y,x)\,\in\,\ell} \mathrm{ratio}[y,x], \qquad "
            r"\mathrm{faint}_{\ell} \;=\; \big[\;"
            r"\overline{\mathrm{ratio}}_{\ell} > 0.85\,t_{\mathrm{Otsu}}"
            r"\;\big]",
      [(r"\overline{\mathrm{ratio}}_{\ell}",
        "the mean ratio value over the pixels of piece l, that is, its "
        "average darkness relative to the local paper"),
       ("A_{\\ell}", "the number of pixels in piece l"),
       (r"\mathrm{ratio}[y,x]", "the illumination-corrected value at that "
                                "pixel, from InkMask"),
       (r"t_{\mathrm{Otsu}}", "the ink threshold on the ratio image"),
       ("0.85", "how close to the threshold a piece's average may come "
                "before it is called faint"),
       (r"\mathrm{faint}_{\ell}",
        "true when piece l only barely qualified as ink at all")],
      None),
     ("p", "The reasoning is that a real pen stroke is not marginal. It is "
           "far darker than the threshold, not just past it, so its mean "
           "ratio sits well below the threshold value. A piece whose "
           "average only just scrapes under is made of pixels that barely "
           "qualified, which is the signature of a crease, a shadow or a "
           "printed rule. Being faint is not grounds for deletion by "
           "itself, since a pencil stroke is faint too; it is combined with "
           "a shape test, so that a faint thin vertical thing goes but a "
           "faint letter-shaped thing stays."),
     ("p", "The per-piece mean is computed in a single pass by a weighted "
           "tally. Tallying the labels with the ratio values as weights "
           "sums the ratio over each piece, and dividing by the pixel count "
           "gives the mean, with no loop over pieces. The same trick "
           "appears throughout the segmenter, and the general pattern is "
           "worth recognising: a per-group sum is a weighted count indexed "
           "by group."),
     ("h", "The edge-shadow test"),
     ("p", "One composite test targets a specific and common artefact: the "
           "shadow of a fold or the curl at the top or bottom of the sheet. "
           "It is identified by five properties at once: it touches the top "
           "or bottom edge, it is no taller than 1.2 pitches, it is wider "
           "than a pitch, it is faint, and its **density** is below 0.15."),
     ("eq", r"\mathrm{density}_{\ell} \;=\; \frac{A_{\ell}}"
            r"{(\,\mathrm{bot}_{\ell}-\mathrm{top}_{\ell}+1\,)\cdot"
            r"(\,\mathrm{xmax}_{\ell}-\mathrm{xmin}_{\ell}+1\,)}",
      [(r"\mathrm{density}_{\ell}",
        "the fraction of this piece's bounding box that the piece actually "
        "fills"),
       ("A_{\\ell}", "the piece's pixel count"),
       (r"\mathrm{top}_{\ell},\,\mathrm{bot}_{\ell}",
        "the first and last rows the piece occupies"),
       (r"\mathrm{xmin}_{\ell},\,\mathrm{xmax}_{\ell}",
        "the first and last columns it occupies")],
      None),
     ("p", "Density distinguishes a sparse scatter from a compact shape. A "
           "word fills a reasonable share of its bounding box; a broken "
           "shadow stretching across the page fills almost none of it, "
           "because its box is wide and tall while its ink is a thin broken "
           "line. Requiring all five properties together is what makes the "
           "test safe: each one individually also describes some piece of "
           "real writing, but nothing in handwriting is simultaneously "
           "edge-touching, wide, short, faint and sparse."),
     ("note", "The background label is explicitly protected from deletion. "
              "Label 0 means \"not part of any piece\", and deleting it "
              "would erase everything that was not assigned to a component, "
              "which is a one-line guard against a catastrophic error."),
   ]),

 "LineCentres": dict(
   inputs=[("mask", "the cleaned ink mask"),
           ("H", "the mask height in pixels")],
   output="a pair: the line pitch, and a sorted list of the row positions "
          "of the lines of writing.",
   theory=[
     ("h", "What it is for"),
     ("p", "The pitch says how far apart the lines are; this function says "
           "where they actually are. Each entry of the returned list is the "
           "row at the heart of one line of writing, and the rest of the "
           "segmenter uses them as anchors: cuts are placed between "
           "consecutive centres, and a piece of ink is judged by how far "
           "above or below a centre it sits."),
     ("h", "Finding candidates"),
     ("p", "The smoothed row profile is reused from LinePitch, and the "
           "candidates are its local maxima, found by the same test: at "
           "least as large as the row above, strictly larger than the row "
           "below, and above 6 percent of the profile's peak. The rows "
           "where the ink piles up are the rows through the bodies of the "
           "letters, which is the right place to call the centre of a line."),
     ("h", "The real problem: suppressing duplicates"),
     ("p", "Smoothing does not remove every spurious peak. A line "
           "containing several tall ascenders can produce a secondary "
           "maximum a few rows above its main one, and a line of writing "
           "that drifts slightly can produce two. Accepting both would "
           "create two lines where there is one, and the consequence is "
           "severe: a cut would be placed between them, straight through "
           "the middle of the writing, and the line would be delivered to "
           "the recogniser as two half-height crops. So the candidates must "
           "be thinned."),
     ("h", "Greedy non-maximum suppression"),
     ("p", "The thinning is done by a method known as **non-maximum "
           "suppression**: sort the candidates by strength, take the "
           "strongest, discard anything too close to it, and repeat with "
           "what is left. It is called **greedy** because it commits to the "
           "best available option at each step and never revisits that "
           "choice."),
     ("eq", r"\mathcal{C} \;\leftarrow\; \mathcal{C} \cup \{c\} "
            r"\quad\text{if}\quad |c - o| \;\ge\; 0.6\,P "
            r"\;\;\text{ for every } o \in \mathcal{C}",
      [(r"\mathcal{C}", "the set of centres accepted so far, considered in "
                        "order of decreasing profile value"),
       ("c", "the candidate peak currently being considered"),
       ("o", "a centre already accepted"),
       ("P", "the line pitch in pixels"),
       ("0.6", "the exclusion radius as a fraction of the pitch")],
      None),
     ("p", "Processing the candidates strongest first is what makes the "
           "greedy rule correct rather than arbitrary. The strongest peak "
           "in a cluster is the one through the letter bodies, so it is "
           "accepted and the weaker peaks of the same line are then "
           "excluded by it. Had the candidates been processed top to bottom "
           "instead, a weak secondary peak slightly above a line would be "
           "accepted first and would then exclude the true centre below it, "
           "placing the line in the wrong row."),
     ("h", "Why 0.6 pitches"),
     ("p", "The exclusion radius has to exceed the width of a spurious "
           "cluster and fall short of the true spacing between lines. By "
           "definition the true spacing is one pitch, and the duplicate "
           "peaks of a single line lie within a fraction of a pitch of each "
           "other, since they are all within one line's x-height. A radius "
           "of 0.6 pitches is comfortably inside that gap: it cannot "
           "suppress a genuine neighbour, which is a full pitch away, and "
           "it does suppress a duplicate, which is much closer. Choosing "
           "slightly more than half a pitch, rather than exactly half, "
           "leaves margin for the pitch estimate itself being a little "
           "low."),
     ("note", "Accepted centres are collected in order of strength and only "
              "sorted into row order at the end. The sort matters because "
              "every later stage assumes the centres run down the page in "
              "sequence, pairing each with the next to place a cut between "
              "them."),
   ]),

 "SeamCuts": dict(
   inputs=[("mask", "the cleaned ink mask"),
           ("centres", "the row positions of the lines, from LineCentres"),
           ("P", "the line pitch in pixels"),
           ("H", "the mask height"),
           ("W", "the mask width")],
   output="three items: the cuts as a list of one row-per-column array "
          "each, the same cuts stacked into a single 2-D array, and L, the "
          "number of bands the cuts divide the page into.",
   theory=[
     ("h", "What it is for"),
     ("p", "With the lines located, boundaries have to be drawn between "
           "them. This function produces one boundary curve per gap, each "
           "given as the row it occupies in every column, so that the page "
           "is divided into L horizontal bands with one line of writing in "
           "each. The curves are seams rather than straight rows, for the "
           "reason given in the entry for LeastCostSeam: a straight cut "
           "between two lines of handwriting cannot avoid slicing through "
           "the descenders of the upper line or the ascenders of the lower "
           "one, because at some column they overlap vertically."),
     ("h", "The cost landscape"),
     ("p", "A seam needs a cost array, and the art of this function is in "
           "building one whose cheapest path is the cut that a person would "
           "draw. It has two terms."),
     ("eq", r"\mathrm{cost}[y,x] \;=\; 4.0 \cdot m[y,x] \;+\; "
            r"0.012 \left(\frac{y - y_{\mathrm{mid}}}"
            r"{g}\right)^{\!2}",
      [(r"\mathrm{cost}[y,x]", "the price of routing the cut through this "
                               "pixel"),
       ("m[y,x]", "1 where there is ink in the closed mask, 0 where there "
                  "is not"),
       ("4.0", "the price of crossing a single ink pixel"),
       ("y_{\\mathrm{mid}}", "the row midway between the two line centres "
                             "this cut separates"),
       ("g", "half the distance between those two centres, used to express "
             "the displacement in units of the gap rather than in pixels"),
       ("0.012", "the strength of the pull back toward the midline")],
      None),
     ("p", "The first term is the one that does the work: ink is expensive, "
           "so the cheapest route goes around letters. The second is a "
           "weak **quadratic penalty** on straying from the middle of the "
           "gap, quadratic meaning it grows with the square of the "
           "displacement, so small excursions are nearly free while large "
           "ones become progressively more costly. Its purpose is to settle "
           "what happens in genuinely empty space. Between two words there "
           "is no ink at all and every route is equally cheap on the first "
           "term alone, so without the second term the cut would wander to "
           "an arbitrary height and arrive at the next word from a silly "
           "position. The pull keeps it near the middle of the gap, ready "
           "to deflect in either direction when the next letter arrives."),
     ("p", "The ratio between the two coefficients is what matters, not "
           "either value alone. At 4.0 against 0.012, crossing one ink "
           "pixel costs as much as sitting about eighteen normalised units "
           "off the midline, which is far outside the gap. So the cut will "
           "always prefer a long detour through white space to cutting a "
           "stroke, and the midline term only ever acts as a tie-breaker. "
           "Expressing the displacement in units of the half-gap rather "
           "than in pixels keeps that balance the same for widely and "
           "narrowly spaced writing."),
     ("h", "Closing the mask first"),
     ("p", "The mask is closed, dilated then eroded by about 0.07 pitches, "
           "before being used as cost. The reason is a specific failure "
           "mode: handwriting has narrow gaps inside it, between the two "
           "strokes of an n or where a pen lifted, and a seam is happy to "
           "thread through such a gap, because threading it is free while "
           "going round costs ink. The cut then passes through the middle "
           "of a letter. Closing fills gaps narrower than the structuring "
           "element while leaving the outer shape of the stroke nearly "
           "unchanged, so the threading routes disappear but the cut still "
           "traces the true outline of the letter."),
     ("p", "Closing rather than plain dilation is the point. Dilation alone "
           "would also close the gaps, but it would leave every stroke "
           "thickened by the full radius, and the cut would then trace a "
           "bloated blob standing well clear of the real ink, wasting "
           "vertical space in both crops. The erosion that follows the "
           "dilation undoes the thickening in the places where it was not "
           "needed, which is exactly the property that distinguishes a "
           "closing from a dilation."),
     ("h", "The step limit"),
     ("p", "The seam here is allowed 7 rows of vertical movement per "
           "column, against the default of 3. A tall narrow descender, such "
           "as the tail of a y, demands a sharp deflection: the cut must "
           "drop below the tail and climb back within the few columns the "
           "tail occupies. With a small step limit the cut simply cannot "
           "turn fast enough and is forced through the tail. The cost of "
           "the larger limit is a little more work per pixel and a little "
           "more freedom to wander, the latter being what the midline "
           "penalty is there to restrain."),
     ("h", "The outer boundaries"),
     ("p", "Cuts exist only between consecutive centres, so the first line "
           "has no cut above it and the last none below. Two straight "
           "boundaries are added at 0.75 pitches beyond the outermost "
           "centres, clipped to the page. They are straight because there "
           "is no second line on the far side to avoid, so there is nothing "
           "for a seam to dodge, and 0.75 pitches is enough to clear the "
           "ascenders of the first line and the descenders of the last. The "
           "number of bands is then one less than the number of "
           "boundaries, which is why L is computed that way."),
   ]),

 "_BandOwnership": dict(
   inputs=[("mask", "the cleaned ink mask"),
           ("lab", "the component labels from DropArtifactPieces"),
           ("nc", "the number of components"),
           ("cuts", "the cuts as a list of per-column row arrays"),
           ("cutsArr", "the same cuts as one 2-D array"),
           ("L", "the number of bands"),
           ("P", "the line pitch"),
           ("H", "the mask height"), ("W", "the mask width"),
           ("centres", "the row positions of the lines")],
   output="six items: the ownership array assigning each ink pixel to a "
          "line, a copy of it taken before the later corrections, the band "
          "index of every pixel, the area of each component, and the row "
          "and column coordinates of the ink pixels.",
   theory=[
     ("h", "What it is for"),
     ("p", "The cuts divide the page into bands, but a band is not yet an "
           "answer. The question that must be settled for every ink pixel "
           "is which line of writing it belongs to, and the band it falls "
           "in is only the first guess at that. This function makes that "
           "guess and then applies several corrections to it, producing the "
           "ownership map that the crops are cut from."),
     ("h", "Computing the band index without a loop"),
     ("p", "Each cut is a curve, so asking which band a pixel is in means "
           "asking how many of the curves lie above it. That count is the "
           "band index directly, and it can be formed by adding up one "
           "comparison per cut over the whole image at once."),
     ("eq", r"\mathrm{region}[y,x] \;=\; \sum_{n=1}^{L-1} "
            r"\big[\; y \;\ge\; \mathrm{cuts}_{n}[x] \;\big]",
      [(r"\mathrm{region}[y,x]", "the band index of this pixel, counting "
                                 "from 0 at the top"),
        (r"\mathrm{cuts}_{n}[x]",
         "the row occupied by cut n in column x"),
       ("L", "the number of bands, so there are L−1 interior cuts"),
       (r"[\;\cdot\;]", "1 when the condition holds, 0 otherwise")],
      None),
     ("p", "The expression is read as: a pixel is below cut n, so it lies "
           "past that boundary, and summing those indicators over all the "
           "interior cuts counts the boundaries it has passed. The outer "
           "two boundaries are excluded from the sum, since nothing lies "
           "beyond them that needs numbering; instead they are used "
           "separately to discard ink above the first line and below the "
           "last."),
     ("h", "Why the band is only a guess"),
     ("p", "If every piece of ink lay wholly inside one band, the band "
           "index would be the final answer. Two things prevent that. "
           "First, a connected piece of ink can genuinely cross a cut: the "
           "descender of a g belongs to the upper line but hangs into the "
           "lower band, and the cut goes around the ink, so the piece ends "
           "up with pixels in both. Taking the band pixel by pixel would "
           "split that g in half, delivering its bowl to one crop and its "
           "tail to another, and neither fragment looks like a g. Second, "
           "small marks like the dot of an i are separate pieces that float "
           "well above their own line's body and can easily fall on the "
           "wrong side of a cut."),
     ("h", "The piece-by-band table"),
     ("p", "Both problems need the same information: how much of each piece "
           "lies in each band. That is a two-dimensional table, pieces "
           "against bands, and it is built with one tally by packing the "
           "two indices into a single number."),
     ("eq", r"\mathrm{key} \;=\; \ell \cdot L \;+\; b, \qquad "
            r"\mathrm{cnt}[\ell,\,b] \;=\; \big|\{\,(y,x) : "
            r"\mathrm{lab}[y,x]=\ell,\; \mathrm{region}[y,x]=b \,\}\big|",
      [(r"\mathrm{key}", "a single integer encoding the pair of a piece and "
                         "a band"),
       (r"\ell", "the piece's component label"),
       ("b", "the band index"),
       ("L", "the number of bands, used as the multiplier so that no two "
             "distinct pairs can collide"),
       (r"\mathrm{cnt}[\ell,b]",
        "how many pixels of piece l fall in band b")],
      None),
     ("p", "This is **index packing**, and the reason it works is place "
           "value, the same principle as writing a two-digit number: "
           "multiplying the first index by the number of possible second "
           "values guarantees that each pair maps to its own integer. A "
           "single tally over the packed keys then produces the whole "
           "table, and reshaping the result into an array of pieces by "
           "bands unpacks it. The alternative, looping over pieces and "
           "counting each one's pixels, would be far slower in Python for "
           "no gain."),
     ("h", "Assigning each piece: the topmost substantial band"),
     ("p", "A band holds a **substantial** share of a piece when it "
           "contains at least 15 percent of that piece's ink. Among the "
           "substantial bands, the piece is assigned to the topmost."),
     ("eq", r"\mathrm{major}(\ell) \;=\; \min\;\big\{\, b \;:\; "
            r"\mathrm{cnt}[\ell,b] \;\ge\; 0.15 \textstyle\sum_{b'} "
            r"\mathrm{cnt}[\ell,b'] \,\big\}",
      [(r"\mathrm{major}(\ell)", "the band piece l is assigned to"),
       (r"\mathrm{cnt}[\ell,b]", "the pixels of piece l in band b"),
       (r"\sum_{b'} \mathrm{cnt}[\ell,b']", "the piece's total ink"),
       ("0.15", "the share of its own ink a band must hold to count as "
                "substantial"),
       (r"\min", "the smallest such band index, that is, the highest on the "
                 "page")],
      None),
     ("p", "The 15 percent test exists to tell a genuine crossing from a "
           "trivial one. A piece whose ink is 95 percent in one band and 5 "
           "percent in the next merely touches the boundary, and treating "
           "it as spanning two lines would be wrong. A piece split 60 to 40 "
           "really does belong to both, and something must be decided."),
     ("p", "Choosing the topmost, rather than the band with the most ink, "
           "is a deliberate asymmetry. When a piece spans two lines it is "
           "usually because a descender from the upper line reaches down, "
           "or because two lines of writing touch. In either case the upper "
           "line is the one that is incomplete without the piece, since the "
           "shape originates there, while the lower line is merely "
           "intruded upon. Giving it to the upper line keeps the crop that "
           "would otherwise be damaged intact. If no band reaches 15 "
           "percent, which happens when a piece is spread thinly over three "
           "or more bands, the fallback is the band with the most ink."),
     ("h", "Small pieces: three graded rules"),
     ("p", "Small pieces are handled separately, because for them the band "
           "is unreliable and their size tells us what kind of thing they "
           "are. The thresholds are areas measured in P², the square of the "
           "pitch, which is the natural unit of area here: a letter "
           "occupies a region roughly a third of a pitch by a half, so "
           "letter-sized is a few hundredths of P²."),
     ("items", [
       "**Under 0.06 P², a stray mark.** It is compared with the "
       "letter-sized pieces assigned to its own band. If none sits within "
       "six pitches horizontally, there is no writing anywhere near it and "
       "it is discarded. Otherwise the median vertical centre of the "
       "nearby writing says where its line really is, and if the mark sits "
       "more than half a pitch below that it is moved down a band, or more "
       "than 0.7 pitches above it, up a band. A speck under 0.01 P² is "
       "discarded in that situation rather than moved, on the grounds that "
       "something that small and that far out of place is noise.",
       "**Between 0.06 and 0.15 P², a small but real piece.** Its own band "
       "and the two neighbours are considered, and it goes to whichever has "
       "writing whose median vertical centre is closest to its own. This is "
       "a nearest-centre rule, and it is used here rather than the band "
       "because a piece this size can easily fall on the wrong side of a "
       "cut while clearly belonging, by position, to one line.",
       "**A minor part of a larger piece.** A band holding a small, narrow "
       "part of a piece, under 0.12 P² and under 35 percent of the piece, "
       "no wider than 0.55 pitches, in a band adjacent to the piece's "
       "assigned one, is treated as a cut-off tail and follows the rest of "
       "the piece. The width condition is what distinguishes a tail from a "
       "whole word: a tail is narrow, and a wide part of a piece in another "
       "band is that band's own writing.",
     ]),
     ("p", "The median is used for the position of nearby writing rather "
           "than the mean, for the usual reason: one misplaced piece among "
           "the neighbours would drag a mean, while the median reports "
           "where the bulk of the line actually is."),
     ("h", "The dot rule"),
     ("p", "The last and most specific correction handles the dots of i and "
           "j. A dot floats clear of its own letter, often nearer to the "
           "line above than to its own line's body, so neither the band nor "
           "a nearest-centre test places it reliably. What does place it is "
           "the letter underneath it."),
     ("p", "A piece is a dot candidate when it is small, under 0.1 P², "
           "compact, no more than 0.45 pitches in either direction, and "
           "roughly round, its longer side no more than 2.2 times its "
           "shorter. The roundness test is what excludes a comma or an "
           "accent from being treated as a dot. The rows below the "
           "candidate are then scanned, up to 0.4 pitches down, for the "
           "first substantial piece appearing beneath it, and the dot is "
           "given to that piece's line, provided the geometry agrees: the "
           "line's writing must sit between 0.25 and 0.7 pitches below the "
           "dot, which is where a dot floats above its own line's body. "
           "That final check is what stops the rule firing when the piece "
           "found below happens to belong to a line further away."),
     ("h", "Writing the result"),
     ("p", "All of this produces, for each piece, either a band index or a "
           "marker meaning discard. The ownership array is then filled in "
           "one operation: a pixel takes its piece's assigned band if the "
           "piece was one of those the rules relocated, and its own band "
           "index otherwise. A copy is kept before the later corrections "
           "run, because those corrections need to know which pixels were "
           "already moved here, in order not to undo that work."),
   ]),

 "_DotsJoinedAbove": dict(
   inputs=[("own", "the ownership array, modified in place"),
           ("mask", "the cleaned ink mask"),
           ("lab", "the closed-mask component labels"),
           ("area", "the area of each closed-mask component"),
           ("region", "the band index of every pixel"),
           ("centres", "the line centres"), ("L", "the number of bands"),
           ("P", "the line pitch"), ("H", "the height"), ("W", "the width"),
           ("ys", "the row coordinates of the ink pixels"),
           ("xs", "their column coordinates")],
   output="four items describing the components of the unclosed mask: the "
          "label array, the count, the per-pixel labels and the areas. The "
          "ownership array is corrected in place.",
   theory=[
     ("h", "The problem it solves"),
     ("p", "The closing applied before labelling in DropArtifactPieces is "
           "useful but it has a side effect. If the dot of an i on one line "
           "happens to sit just below a descender from the line above, the "
           "closing can bridge the small gap between them and fuse the two "
           "into a single component. The dot then inherits whatever the "
           "fused piece was assigned, which is the upper line, and the "
           "lower line's crop loses its dot while the upper line's crop "
           "gains a stray mark."),
     ("h", "Why a second labelling is needed"),
     ("p", "The fix requires seeing the dot as its own object, which the "
           "closed labelling can no longer do. So the mask is labelled "
           "again, this time without closing. In the unclosed labelling the "
           "dot is separate from the descender, because the gap between "
           "them is real ink-free space; it was only the closing that "
           "joined them. The function therefore works with two labellings "
           "at once, the unclosed one to identify the dot and the closed "
           "one to recognise when a piece is in fact part of a larger "
           "structure."),
     ("h", "Identifying the dot"),
     ("p", "The same three-part test as before: small area, between 0.015 "
           "and 0.1 P², compact, no more than half a pitch across, and "
           "roughly round. The lower area bound is new and matters here, "
           "since it excludes single-pixel noise that would otherwise pass "
           "every other condition."),
     ("h", "Finding the stem and its line"),
     ("p", "The rows immediately below the candidate are scanned for the "
           "first other substantial piece, and the band of the pixel where "
           "the two meet is read from the region array. That gives the line "
           "the stem belongs to, which is the line the dot should belong "
           "to. The dot's own current assignment is read as the most common "
           "ownership value over its pixels."),
     ("h", "The three conditions for acting"),
     ("items", [
       "**The stem must be exactly one band below.** A fusion of this kind "
       "can only displace a mark by one line, since the closing bridges a "
       "gap of a few pixels. A discrepancy of two bands means something "
       "else is wrong, and acting on it would make things worse.",
       "**The fused piece must not be a whole word.** If the closed "
       "component containing the dot has an area of 0.2 P² or more, it is a "
       "word, not a dot welded to a descender. A genuine floating dot stays "
       "a small component even after closing, so a large closed area is "
       "evidence that the candidate is really part of a letter, such as the "
       "bowl of an a that the threshold happened to detach.",
       "**The dot must sit well below the upper line's centre.** At least "
       "0.45 pitches below it. A mark that close to the upper line's body "
       "really is part of the upper line, and the rule declines to move "
       "it.",
     ]),
     ("p", "Only the pixels currently assigned to the upper line are "
           "moved, not every pixel of the unclosed piece. This matters when "
           "a letter has broken apart in the unclosed mask, so that the "
           "candidate shares its unclosed component with ink that was "
           "already correctly assigned: moving the whole component would "
           "drag that correct ink along with it."),
     ("note", "The pattern of conditions here is characteristic of the "
              "whole ownership stage. Each correction rule is narrow on "
              "purpose, requiring several independent pieces of evidence to "
              "agree before it fires. A broad rule would fix more cases and "
              "break more, and a crop that is wrong is more costly than a "
              "crop that is merely imperfect, because the recogniser will "
              "read it confidently either way."),
   ]),

 "_RingsToStem": dict(
   inputs=[("own", "the ownership array, modified in place"),
           ("raw", "the unclosed component labels"),
           ("nr", "the number of unclosed components"),
           ("pr", "the unclosed label of each ink pixel"),
           ("Ar", "the area of each unclosed component"),
           ("lab", "the closed labels"), ("area", "the closed areas"),
           ("region", "the band index of every pixel"),
           ("centres", "the line centres"), ("P", "the line pitch"),
           ("H", "the height"), ("W", "the width"),
           ("ys", "the row coordinates of the ink pixels"),
           ("xs", "their column coordinates")],
   output="nothing is returned; the ownership array is corrected in place.",
   theory=[
     ("h", "The problem it solves"),
     ("p", "This is the mirror image of the previous correction. There, a "
           "dot on the lower line was welded to a descender from above and "
           "wrongly given to the upper line. Here the welding runs the "
           "other way: a small ring-shaped mark belonging to the lower "
           "line, such as the dot of an i written as a small circle, or the "
           "loop at the top of a letter, is fused to the stem it sits on, "
           "and the stem's line is the one it should follow. The two "
           "corrections exist separately because the evidence needed "
           "differs: finding a ring requires detecting a hole, which the "
           "dot rule does not do."),
     ("h", "Finding a candidate: walking up from the bottom"),
     ("p", "The candidate is not a whole component here but a part of one, "
           "so it cannot be found by labelling. It is found geometrically. "
           "For each component, the lowest ink pixel in each column is "
           "located, and those bottom points are examined in order from the "
           "lowest upward, since the ring sits at the top of a stem and the "
           "stem's foot is the lowest point of the structure."),
     ("p", "From each bottom point the algorithm walks upward, measuring at "
           "every row how wide the component is within a window around the "
           "column, and tracking the widest it has been. The walk stops "
           "when the width drops below 45 percent of that maximum. That "
           "point is the junction between the blob and the stem: a ring is "
           "wide, the stem below it is narrow, so the width collapses "
           "exactly where one ends and the other begins. This is a "
           "one-dimensional version of a shape-decomposition idea that "
           "recurs in image analysis: a narrow neck is where an object "
           "naturally divides."),
     ("h", "Validating the shape"),
     ("p", "Three conditions must hold before the candidate is accepted as "
           "a ring."),
     ("items", [
       "**Its width is between 0.2 and 0.45 pitches.** Narrower is a "
       "stroke, wider is a letter body.",
       "**It is roughly as tall as it is wide,** with a height-to-width "
       "ratio between 0.6 and 1.6. A ring is round; a stroke is not.",
       "**It encloses a hole.** This is the decisive test, and it is "
       "carried out by labelling the components of the inverse of the blob "
       "after padding it with a border. If the blob encloses a hole then "
       "the background splits into two components, the outside and the "
       "hole, so a count of fewer than two components means there is no "
       "hole and the candidate is a hook or a curl rather than a ring.",
     ]),
     ("p", "The padding before that test is not optional. Without a border "
           "of background all round, a blob touching the edge of the "
           "extracted window would let the outside connect to the hole "
           "along that edge, merging them into one component and hiding the "
           "hole. Padding guarantees the outside is a single region that "
           "surrounds the blob completely. Counting holes by labelling the "
           "background is a standard technique, and the number it yields is "
           "related to the **Euler number** of the shape, the classical "
           "topological measure of components minus holes."),
     ("h", "The decision"),
     ("p", "As before, the first other piece appearing below the ring "
           "identifies the line to move to, the two bands must be adjacent, "
           "the closed component must not be large enough to be a whole "
           "word, and the junction must lie no more than 0.45 pitches above "
           "the target line's centre. Only the ring's own pixels are "
           "reassigned, confined to the rows between the junction and the "
           "bottom point and to the columns the blob occupies, and only "
           "where they currently belong to the band being moved from."),
     ("h", "Why the examination is limited"),
     ("p", "At most 60 bottom points are examined per component, and "
           "accepting one point suppresses the columns within 0.35 pitches "
           "of it. Both are cost controls. A long word has hundreds of "
           "columns and the upward walk at each is not free, while the "
           "suppression prevents the same ring being found again from every "
           "column it spans. The cap is a deliberate compromise: a word "
           "with more than 60 distinct bottom points may have a ring "
           "missed, which costs one misplaced dot, whereas examining every "
           "column would slow the segmenter measurably on every page."),
   ]),

 "_FragmentsToNeighbour": dict(
   inputs=[("own", "the ownership array, modified in place"),
           ("ownBeforeEF", "ownership as it stood before the two dot "
                           "corrections, used to tell which pixels they "
                           "already moved"),
           ("mask", "the cleaned ink mask"),
           ("centres", "the line centres"), ("L", "the number of bands"),
           ("P", "the line pitch"), ("H", "the height"), ("W", "the width")],
   output="nothing is returned; the ownership array is corrected in place.",
   theory=[
     ("h", "The principle: a line's writing lies in its x-height band"),
     ("p", "This is the most general of the corrections, and it rests on a "
           "single observation. A line of handwriting is defined by the "
           "band of rows its letter bodies occupy, around its centre. "
           "Anything genuinely belonging to that line must have some ink "
           "there, because a letter has a body. A piece assigned to a line "
           "that has no ink at all within a quarter pitch of its centre is "
           "therefore not part of that line, whatever the cut said. It is a "
           "visitor: a descender hanging down from above, or an ascender "
           "reaching up from below."),
     ("h", "How the test is made"),
     ("p", "For each band in turn, the ink currently owned by that band is "
           "labelled into connected pieces, and for each piece the number "
           "of its pixels lying within 0.28 pitches of the band's centre is "
           "counted."),
     ("eq", r"\mathrm{inband}(c) \;=\; \big|\{\,(y,x) \in c \;:\; "
            r"|y - \mathrm{centre}_{n}| \;\le\; 0.28\,P \,\}\big|",
      [(r"\mathrm{inband}(c)",
        "how many pixels of piece c lie in the band's x-height zone"),
       ("c", "a connected piece of the ink currently owned by band n"),
       (r"\mathrm{centre}_{n}", "the row position of line n"),
       ("P", "the line pitch"),
       ("0.28", "the half-height of the x-height zone as a fraction of the "
                "pitch")],
      None),
     ("p", "A piece with a count of exactly zero, and an area of at least "
           "0.006 P² so that single-pixel noise is ignored, is a candidate "
           "for rehoming. The 0.28 figure is about half the x-height "
           "expressed in pitches, which follows from the factor of "
           "roughly 2.2 between pitch and x-height noted in LinePitch: half "
           "an x-height is about 0.23 pitches, and 0.28 allows a little "
           "margin for the centre estimate being slightly off."),
     ("h", "Deciding where it belongs: two routes"),
     ("p", "Having established that a fragment does not belong where it is, "
           "two independent kinds of evidence are tried for where it does."),
     ("items", [
       "**Adjacency.** The fragment is dilated by two pixels and the "
       "ownership of the surrounding ink is examined. If at least three of "
       "those neighbouring pixels belong to one other band, the fragment is "
       "physically touching that line's writing, which is strong evidence "
       "that it is part of it. The move is only made if that band is "
       "adjacent and lies on the correct side: a fragment below this line's "
       "centre may only move down, one above it only up. That consistency "
       "check is what prevents a chain of bad moves.",
       "**Position.** If no neighbouring band reaches three pixels, the "
       "fragment is isolated and adjacency says nothing. The fallback is to "
       "ask whether the fragment's own vertical centre falls inside a "
       "neighbouring line's x-height band, within 0.2 pitches of its "
       "centre, and if so to move it there. This is weaker evidence, which "
       "is why it is used only as a fallback and with a tighter tolerance.",
     ]),
     ("h", "Not undoing the previous corrections"),
     ("p", "The ownership array as it stood before the two dot corrections "
           "is passed in so that the pixels they moved can be identified, "
           "by comparing the two arrays. A small compact fragment that sits "
           "on pixels those rules already placed is skipped. The reason is "
           "direct: a dot deliberately placed above its line by the dot "
           "rule has no ink in its line's x-height band, by the very nature "
           "of being a dot, so it looks exactly like a fragment to this "
           "test and would be moved straight back. The exception is "
           "deliberately narrow, applying only to pieces under 0.04 P² and "
           "no more than 0.4 pitches across, so that a genuine tail loop "
           "that happens to overlap a placed dot is still free to move."),
     ("note", "Every move is recorded and applied only after the scan over "
              "all the bands is complete. Applying them as they are found "
              "would mean later tests read ownership that earlier moves had "
              "already changed, so the outcome would depend on the order "
              "the bands happened to be visited. Collecting first and "
              "applying afterwards makes the result independent of that "
              "order."),
   ]),

 "AssignOwnership": dict(
   inputs=[("mask", "the cleaned ink mask"),
           ("lab", "the component labels"), ("nc", "the component count"),
           ("cuts", "the cuts as a list"), ("cutsArr", "the cuts as an array"),
           ("L", "the number of bands"), ("centres", "the line centres"),
           ("P", "the line pitch"), ("H", "the height"), ("W", "the width")],
   output="the ownership array: for every pixel, the index of the line that "
          "owns it, or −1 for background and for ink that was discarded.",
   theory=[
     ("h", "What it is for"),
     ("p", "This function contains no logic of its own. It runs the band "
           "guess and then the three corrections, in order, and returns the "
           "result. Its value is organisational, and the organisation is "
           "the point worth explaining."),
     ("h", "Why the rules are separate functions"),
     ("p", "The ownership problem is where the segmenter is least "
           "principled and most empirical. The band guess follows from the "
           "cuts, but each correction after it exists because a specific "
           "failure was observed on real pages and diagnosed. Keeping each "
           "as its own function with its own name has three practical "
           "consequences. A rule can be read on its own, with its "
           "conditions in one place rather than interleaved with others. A "
           "rule can be disabled by commenting out one line here, which is "
           "how its contribution was measured. And a rule can be tested "
           "against the cases it was written for without running the whole "
           "page."),
     ("h", "Why the order is fixed"),
     ("p", "The sequence is not arbitrary. The band guess must come first "
           "because everything else corrects it. The two welding rules come "
           "next, and they are the most specific: each requires several "
           "independent conditions and fires rarely. The fragment rule "
           "comes last and is the most general, which is the right place "
           "for it, because a general rule run first would act on cases the "
           "specific rules would have handled better. The ordering "
           "principle, specific before general, is the same one that "
           "governs a cascade of fallbacks in LinePitch: apply the "
           "best-informed rule wherever it applies, and the broad rule only "
           "to what is left."),
     ("p", "The dependency between them is explicit in the arguments. The "
           "fragment rule receives the ownership array as it stood before "
           "the welding rules ran, precisely so it can recognise and "
           "respect their decisions. That is why the band guess returns "
           "that copy, rather than each rule working from whatever it finds."),
   ]),

 "_BuildLineCrop": dict(
   inputs=[("n", "which line to build, as a band index"),
           ("own", "the ownership array"),
           ("norm", "the illumination-corrected full-resolution page"),
           ("P", "the line pitch, in half-size pixels"),
           ("f", "the scale factor from half-size to full-size coordinates"),
           ("H", "the half-size height"), ("W", "the half-size width"),
           ("minInk", "the least ink a band must own to be a real line")],
   output="None if this band is not a usable line; otherwise a dictionary "
          "with the finished grey crop, a fixed-size copy for the "
          "recogniser, and the bounding box the crop came from.",
   theory=[
     ("h", "What it is for"),
     ("p", "All the analysis so far has been at half resolution and in "
           "terms of masks. This function turns one band's worth of "
           "ownership into the actual image the recogniser will read: a grey "
           "picture of one line of writing, cut from the full-resolution "
           "page, contrast-corrected, with the neighbouring lines removed."),
     ("h", "Rejecting a band that is not a line"),
     ("p", "Two tests reject a band outright. It must own at least minInk "
           "pixels, a threshold the caller derives from the median ink per "
           "line so that it adapts to the page, and its ink must span at "
           "least 0.2 pitches vertically. The height test is the more "
           "interesting one: a band containing a horizontal scatter of "
           "specks has plenty of ink but no vertical extent, and is a "
           "surviving artefact rather than writing. The floor of six pixels "
           "keeps the test meaningful on a very small image. The height "
           "demanded is deliberately well under a full x-height, because a "
           "short line containing only lowercase letters is still a real "
           "line."),
     ("h", "Trimming a distant stray"),
     ("p", "A band can own a legitimate line plus one mark far away, "
           "perhaps a page number the artefact rules did not catch. "
           "Cropping to all of it would produce an image mostly of blank "
           "paper with writing at one end, which is poor input for a "
           "recogniser. So the rows containing ink are examined, the gaps "
           "between them found, and the largest contiguous run of inked "
           "rows kept."),
     ("p", "The trim is only performed if it removes more than 0.8 pitches "
           "of height. That guard is what distinguishes a paragraph-sized "
           "gap from the ordinary small gaps inside a line of writing, "
           "which occur between the body of the line and a stray ascender. "
           "Without the guard, a line with an ordinary internal gap would "
           "have its ascenders trimmed away."),
     ("h", "From half-size decisions to full-size pixels"),
     ("p", "The bounding box of the kept ink is padded by 0.15 pitches, so "
           "that the writing does not touch the edge of the crop, and then "
           "multiplied by the scale factor to give the corresponding "
           "rectangle of the full-resolution page. This is where the two "
           "resolutions meet: every decision was made cheaply at half size, "
           "and the pixels are taken at full size."),
     ("h", "Building the alpha channel"),
     ("p", "The crop must show this line's ink and not the neighbouring "
           "lines' intrusions, so a per-pixel weight is needed saying how "
           "much of each pixel to show. That weight is called **alpha**, "
           "borrowing the standard graphics term for opacity, where 1 means "
           "fully shown and 0 fully replaced by white."),
     ("p", "It starts as the ownership mask for this line, enlarged to the "
           "crop size bilinearly. Bilinear rather than nearest-neighbour is "
           "a deliberate choice here, and for an unusual reason: the soft "
           "edge that interpolation produces is what makes the stroke "
           "boundary in the final crop look like a real pen edge rather "
           "than a staircase of hard pixels. The slight blur is the feature, "
           "not an artefact to be tolerated."),
     ("h", "Recovering ink the mask missed"),
     ("p", "The half-size mask inevitably misses some faint ink, and at "
           "full resolution that ink is visible. So a second, local "
           "threshold is computed, on the crop alone rather than the whole "
           "page, using the same divide-by-local-background method as "
           "InkMask but with a radius scaled to the crop's own height. "
           "Because the crop is one line of writing rather than a whole "
           "page, the contrast within it is far more uniform, so this local "
           "threshold can pick up strokes that the page-wide one could not."),
     ("p", "Those recovered pixels cannot simply be added, or the "
           "neighbouring lines' ink would come back with them. Two gates "
           "restrict them."),
     ("items", [
       "**The halo.** The line's own ink is dilated by about 0.2 pitches "
       "and the recovery is confined to that neighbourhood. The reasoning "
       "is that a stroke the mask missed lies next to strokes it caught, "
       "since it is part of the same writing, whereas ink from the line "
       "above or below is further away.",
       "**The ownership veto.** Any pixel the page-level ownership already "
       "gave to a different line is excluded, unconditionally. This gate is "
       "enlarged with nearest-neighbour rather than bilinear "
       "interpolation, precisely the opposite choice from the alpha "
       "channel, and for the opposite reason: ownership is a yes-or-no fact "
       "about which line a pixel belongs to, and a blurred version would "
       "half-assign a pixel to two lines at once.",
     ]),
     ("p", "The final alpha is the larger of the ownership alpha and the "
           "gated recovery, taken pixel by pixel, so recovery can only ever "
           "add ink and never remove it."),
     ("h", "Contrast stretching"),
     ("p", "The crop's grey values still span whatever range the "
           "photograph happened to produce, and the recogniser was trained "
           "on images with ink near black and paper near white. A "
           "**contrast stretch** maps the observed range onto the full "
           "range linearly."),
     ("eq", r"\mathrm{linear}[y,x] \;=\; \mathrm{clip}\!\left("
            r"\big(\mathrm{region}[y,x] - b\big)\cdot"
            r"\frac{255}{\max(w - b,\,1)},\; 0,\; 255\right)",
      [(r"\mathrm{linear}[y,x]", "the stretched grey value"),
       (r"\mathrm{region}[y,x]", "the illumination-corrected value at this "
                                 "pixel"),
       ("b", "the black point: the value that should become 0, taken as the "
             "40th percentile of the values under the ink"),
       ("w", "the white point: the value that should become 255, taken as "
             "the 75th percentile of the whole crop"),
       (r"\max(w-b,\,1)", "a guard against a degenerate crop in which the "
                          "two points coincide"),
       (r"\mathrm{clip}(\cdot,0,255)", "values forced into the valid grey "
                                       "range")],
      None),
     ("p", "The two reference points are percentiles rather than the "
           "minimum and maximum, and the choice is what makes the stretch "
           "stable. One unusually dark pixel, from a blob of ink or a speck "
           "of dust, would set a minimum-based black point and leave all "
           "the real ink grey. The 40th percentile of the ink pixels "
           "describes the typical darkness of a stroke, which is the value "
           "that should become black. Likewise the 75th percentile of the "
           "crop describes the paper, since paper is the majority of the "
           "area."),
     ("h", "The gamma correction"),
     ("p", "A linear stretch alone leaves faint strokes faint, because a "
           "stroke at 70 percent of the way from ink to paper maps to 70 "
           "percent of the way from black to white. A **gamma correction** "
           "applies a power to the darkness, which bends the mapping."),
     ("eq", r"d'[y,x] \;=\; \Big(\,d[y,x]\,\Big)^{\gamma}, \qquad "
            r"d[y,x] \;=\; \frac{255 - \mathrm{linear}[y,x]}{255}, "
            r"\qquad \gamma = 0.55",
      [("d[y,x]", "the darkness at this pixel, 0 for white and 1 for "
                  "black: the stretched value inverted and scaled"),
       ("d'[y,x]", "the corrected darkness"),
       (r"\gamma", "the gamma exponent, here 0.55")],
      None),
     ("p", "With an exponent below 1 the power function lies above the "
           "straight line on the interval from 0 to 1, so every darkness is "
           "increased, and increased most where it starts small. A faint "
           "stroke at darkness 0.3 becomes 0.3 raised to the power 0.55, "
           "which is about 0.51, nearly "
           "doubling, while ink already at 0.9 becomes 0.94, barely "
           "changing. That is precisely the desired behaviour: pull the "
           "faint strokes up to where the recogniser can see them, and "
           "leave the strong ink alone. The corrected darkness is then "
           "converted back to grey and composited with white through the "
           "alpha channel."),
     ("h", "Aligning the left margin"),
     ("p", "Finally the crop is shifted horizontally so that the first "
           "column containing ink sits the same distance from the left edge "
           "in every crop. The reason is that the recogniser's convolutional "
           "layers respond to position, so a line whose writing starts "
           "20 pixels in and one whose writing starts 80 pixels in present "
           "measurably different inputs for the same text. Standardising "
           "the left offset removes a variation that carries no "
           "information."),
     ("note", "The returned dictionary contains the crop twice: once at its "
              "natural size for display and for the letter cutter, and once "
              "squeezed onto the recogniser's fixed input canvas by "
              "FitFixedCanvas. Keeping both avoids any later stage having "
              "to guess which geometry it is looking at."),
   ]),

 "SegmentLines": dict(
   inputs=[("page", "the photograph of the page, grey or colour")],
   output="three items: the list of line crops in top-to-bottom order, a "
          "dictionary of raw intermediates for drawing diagnostic overlays, "
          "and a dictionary of summary statistics about the segmentation.",
   theory=[
     ("h", "What it is for"),
     ("p", "This is the entry point to the whole segmenter, and it is "
           "deliberately short. Every stage of the method is a named "
           "function above, so this function is a list of those calls in "
           "order, which means the sequence of operations can be read off "
           "directly instead of being reconstructed from a long body of "
           "code."),
     ("h", "The pipeline in order"),
     ("items", [
       "**Deskew.** Reduce to grey, make a half-size copy, find the tilt "
       "and rotate, and derive the ink mask and ratio image.",
       "**RemovePrintedStructure.** Locate the sheet, discard everything "
       "outside it, and delete the long straight runs that are ruling and "
       "page edges.",
       "**DropArtifactPieces.** Judge what remains piece by piece and "
       "delete the creases, slivers and shadows that are not writing.",
       "**LineCentres.** Locate each line of writing as a peak of the row "
       "profile, with duplicates suppressed.",
       "**SeamCuts.** Draw a least-ink seam between each pair of "
       "neighbouring lines, dividing the page into bands.",
       "**AssignOwnership.** Decide which line owns every ink pixel, "
       "starting from the band and applying the three corrections.",
       "**_BuildLineCrop,** once per band: cut the crop from the "
       "full-resolution page, mask it to this line's ink, correct its "
       "contrast and align it.",
     ]),
     ("h", "Why the crops come from a separate normalised image"),
     ("p", "The crops are cut not from the raw rotated page but from a "
           "full-resolution illumination-corrected version of it, built "
           "here. The background is estimated at half size, where the box "
           "average is cheap, then enlarged bilinearly to full size and "
           "divided out, with the same scale-aware radius that InkMask "
           "uses. Estimating the background at full resolution would cost "
           "four times as much for no benefit, because the illumination "
           "varies smoothly by definition, so a half-resolution estimate of "
           "it enlarged by interpolation is accurate to well within the "
           "precision that matters."),
     ("h", "Why the ink threshold adapts to the page"),
     ("p", "The minimum ink a band must own before it is accepted as a "
           "line is computed from the page itself: 12 percent of the median "
           "ink per band, with a floor of 60 pixels."),
     ("eq", r"\mathrm{minInk} \;=\; \max\big(60,\; 0.12 \cdot "
            r"\mathrm{median}\{\, I_n \,:\, I_n > 0 \,\}\big)",
      [(r"\mathrm{minInk}", "the ink threshold for accepting a band as a "
                            "line"),
       ("I_n", "the number of ink pixels owned by band n"),
       (r"\{\,I_n : I_n > 0\,\}", "the counts for the bands that own any "
                                  "ink at all, so that empty bands do not "
                                  "drag the median to zero"),
       ("0.12", "the fraction of a typical line's ink that a band must own"),
       ("60", "an absolute floor in pixels")],
      None),
     ("p", "An absolute threshold would be wrong in both directions: too "
           "high for small writing, where a real line has few pixels, and "
           "too low for a large photograph, where an artefact has many. "
           "Comparing against the median line on the same page sidesteps "
           "that, because it measures what a line looks like in this "
           "particular image. The fraction is set low, at 12 percent, "
           "because a short line containing one or two words is perfectly "
           "legitimate and must not be discarded; the threshold is aimed at "
           "bands containing a speck, not at short lines."),
     ("h", "Why three return values"),
     ("p", "The crops are the product. The other two exist because a "
           "segmenter that only returns its output is very hard to debug. "
           "The second dictionary carries the raw cuts, the ratio image and "
           "the paper mask, which the test code overlays on the page so "
           "that a bad cut can be seen rather than inferred. The third "
           "carries summary numbers: the angle applied, the pitch, what "
           "fraction of the frame was paper, how much ink lay outside it, "
           "how many straight pixels and artefact pieces were removed, and "
           "how many lines came out. Those numbers are how a failure gets "
           "localised: an angle at the edge of the search range, a pitch "
           "far from the usual, or a paper coverage near zero each point at "
           "a different stage, without any need to look at the image."),
     ("note", "The diagnostics deliberately leave the project's own "
              "evaluation results out of the pipeline code. They report "
              "what this page did, not what the method scores in general. "
              "That separation keeps the core code free of anything that "
              "would have to be revised every time the method is measured "
              "again."),
   ]),
}
