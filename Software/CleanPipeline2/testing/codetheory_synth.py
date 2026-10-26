"""synthesise.py, part 1: the fallback alphabet and choosing a shape.

This chapter covers writing new text in a stored hand.  Part 1 is the
built-in letterforms, the tuning constants, and the decision of which shape
to draw for each character.  Part 2, in codetheory_synth2.py, covers layout,
rasterising and the search that picks the best attempt.
"""

INTRO = [
 ("p", "This file does the reverse of style.py. Given a style profile and "
       "some text, it produces a pen path: an ordered list of strokes in "
       "millimetres that a plotter can draw, written in the hand the "
       "profile describes."),
 ("h", "The central tension of the whole project"),
 ("p", "Two goals pull against each other here, and most of the design is a "
       "response to that. The output should **look like the writer**, which "
       "argues for using their own stored letterforms exactly as extracted. "
       "And the output should **be readable**, which argues for clean, "
       "well-formed letters. These conflict because the stored letterforms "
       "are not clean: they were cut from joined-up writing at boundaries "
       "that often do not exist, so a proportion of them are fragments or "
       "carry part of a neighbour. Drawing those faithfully reproduces the "
       "writer's hand and some of the extraction's mistakes along with it."),
 ("p", "The file resolves this with a single number, called the "
       "**legibility lambda** and written as lam, running from 0 to 1. At 0 "
       "the writer's own shapes are always used. At 1 they are abandoned "
       "entirely in favour of a fallback letterform. In between, each glyph "
       "is swapped with a probability that depends both on lam and on how "
       "far that particular shape sits from what the letter should look "
       "like. The value is derived automatically per writer, so a hand "
       "whose library extracted cleanly keeps its own shapes and one whose "
       "library is mostly fragments does not."),
 ("h", "The three sources a letter can come from"),
 ("items", [
   "**own:** a variant cut from this writer's own pages. The most faithful "
   "and the least reliable.",
   "**core, the anchor:** a single clean letterform. Preferably the "
   "writer's own most typical variant of that letter, falling back to a "
   "built-in alphabet when even that is too rough.",
   "**fallback:** the built-in alphabet directly, used when the writer has "
   "no example of a character at all.",
 ]),
 ("p", "Every rendered glyph records which source it came from, and the "
       "counts are returned in the trajectory's metadata. That is what "
       "makes the trade-off measurable rather than a matter of impression: "
       "a page drawn mostly from the anchor is readable but is not really "
       "the writer's hand, and the counts say which happened."),
 ("h", "Units"),
 ("p", "Glyphs arrive in x-heights, as stored. Layout converts them to "
       "millimetres using the requested x-height size, and the rasteriser "
       "converts millimetres to pixels. So three coordinate systems appear "
       "in this file, in that order, and the conversions happen at exactly "
       "two points."),
]


FUNCS = {

 "SCRIPT_DIR": dict(
   output="the directory this file sits in.",
   theory=[
     ("h", "What it is for"),
     ("p", "Records the module's own location, from which a path to data "
           "beside the code could be built. The project's paths are "
           "otherwise defined centrally in paths.py, which is the better "
           "practice, since one definition serves the whole project."),
     ("note", "This constant is assigned and never read anywhere in the "
              "module. It is a remnant of an earlier layout, before the "
              "paths module existed. It is harmless, but a reader should "
              "not take its presence as evidence that the file locates its "
              "data this way."),
   ]),

 "CNN_DATA_DIR": dict(
   output="a path to a sibling directory of network data.",
   theory=[
     ("h", "What it is for"),
     ("p", "Points at a folder beside the pipeline holding network-related "
           "data, built by stepping up from this file's own location."),
     ("note", "Like the constant above, this is assigned and never read. "
              "Both are left in place rather than removed because removing "
              "a name is a change that has to be checked against every "
              "caller, and neither does any harm; but neither describes how "
              "the module actually finds anything."),
   ]),

 "_FB": dict(
   output="a dictionary mapping a character to a built-in printed "
          "letterform, as a list of strokes in x-heights.",
   theory=[
     ("h", "What it is and why it exists"),
     ("p", "A hand-written alphabet of 47 characters: the 26 lowercase "
           "letters, the ten digits, and eleven punctuation marks, each "
           "given as one or more strokes of points in the same x-height "
           "convention the extracted glyphs use. These are the letterforms "
           "of last resort."),
     ("p", "Three situations require them. A writer may have no example of "
           "a character at all, which is common: a page of ordinary English "
           "contains no q, z or semicolon. A writer's examples of some "
           "character may all have been rejected by the filters. And the "
           "legibility mechanism may decide a stored shape is too far from "
           "a letter to be drawn, in which case something must take its "
           "place."),
     ("h", "Why they are written out by hand"),
     ("p", "They could not be obtained any other way. A font is not usable: "
           "a font describes letters as filled outlines, and what is needed "
           "here is the path a pen travels along the centre of a stroke, "
           "which is a different object. Deriving centre lines from a font "
           "by thinning would reintroduce every artefact that the "
           "extraction pipeline exists to deal with, for letters that are "
           "supposed to be the clean reference. So the coordinates were "
           "laid out directly, in the same coordinate system as the stored "
           "glyphs, so that a fallback letter can be substituted for an "
           "extracted one with no conversion at all."),
     ("h", "The coordinate convention"),
     ("p", "A point of (0.4, 0.7) means four tenths of an x-height to the "
           "right of the glyph's origin and seven tenths of an x-height "
           "above the baseline. A y above 1.0 is in the ascender zone and a "
           "negative y is below the baseline, in the descender zone. Being "
           "in the same convention is what allows _BaselineGlyph to stretch "
           "these shapes to a particular writer's reach by rescaling only "
           "the parts outside the band."),
     ("note", "There are no uppercase letters in this dictionary. A capital "
              "is handled instead by taking the lowercase form and scaling "
              "it up to the writer's ascender height, which is what "
              "_GlyphSource does under the label `case`. That is correct "
              "for letters whose capital is a larger version of the "
              "lowercase, such as C, O, S and W, and is a visible "
              "approximation for the ones where it is not, such as A, E and "
              "R."),
   ]),

 "_CURSIVE": dict(
   output="a dictionary mapping each lowercase letter to a joined-up "
          "letterform.",
   theory=[
     ("h", "Why a second alphabet is needed"),
     ("p", "The printed alphabet above is wrong for a cursive hand, and not "
           "merely in appearance. A printed letterform begins and ends "
           "wherever is convenient, often at the top, and has no provision "
           "for a stroke arriving from the previous letter or leaving "
           "toward the next. Substituting one into joined writing produces "
           "a letter that cannot be connected: the join from the preceding "
           "letter has nowhere to attach, so either it is omitted, leaving "
           "a visible break in the middle of a word, or it is drawn to a "
           "point on the letter that no writer would join to."),
     ("p", "These forms are drawn to enter near the baseline on the left "
           "and leave near the baseline on the right, which is what a "
           "cursive letter does, so a ligature can be drawn to and from "
           "them naturally."),
     ("h", "Which alphabet is chosen"),
     ("p", "The choice is made per writer, from their measured "
           "connectedness: a hand at or above 0.5 gets the cursive forms "
           "and one below gets the printed ones. Using a single number "
           "measured from their own writing, rather than a setting, means "
           "the decision follows the evidence; see CORE_CURSIVE_CONN."),
     ("note", "This alphabet holds only the 26 lowercase letters, where the "
              "printed one also holds digits and punctuation. That is "
              "sufficient because joining only ever applies between two "
              "adjacent alphabetic characters, so a digit or a comma never "
              "needs a joinable form. Where a cursive writer needs one of "
              "those, the lookup falls through to the printed alphabet."),
   ]),

 "_FB_ADV": dict(
   output="1.0, the advance after a fallback letter, in x-heights.",
   theory=[
     ("h", "What an advance is"),
     ("p", "The **advance** is how far the pen origin moves to the right "
           "after drawing a character, before the next one is placed. It is "
           "not the same as the character's width: it includes the space "
           "that belongs between this letter and the next, so advancing by "
           "the width alone would butt every letter against its neighbour."),
     ("p", "For extracted glyphs the advance is measured from the cell the "
           "character was cut from, which is what preserves the writer's "
           "own spacing. A built-in letterform has no such measurement, so "
           "this constant supplies one: one x-height, which is "
           "approximately the advance of a lowercase letter in ordinary "
           "handwriting."),
     ("note", "It is used only when the writer's profile has no recorded "
              "advance for the character either. Where the profile does "
              "have one, even though the shape is a fallback, the writer's "
              "own figure is preferred, so the spacing stays theirs even "
              "when the letterform is not."),
   ]),

 "_SIMILAR": dict(
   output="a mapping from a character to another whose letterform can stand "
          "in for it.",
   theory=[
     ("h", "What it is for"),
     ("p", "Some characters are near-identical in handwriting, so one "
           "writer's example of the one can serve for the other. A capital "
           "I and a lowercase l are the same vertical stroke in most hands. "
           "A capital O and a zero differ by convention rather than by "
           "shape. A semicolon is a colon with a tail, an opening quote is "
           "a pair of apostrophes, and an exclamation mark is a vertical "
           "stroke with a dot."),
     ("h", "Where it sits in the order of preference"),
     ("p", "It is consulted after the writer's own examples of the "
           "character itself and after the opposite case of it, and before "
           "resorting to the anchor or the built-in alphabet. The ordering "
           "reflects how much of the writer's own hand each option "
           "preserves: their own l really is their own stroke, so borrowing "
           "it for an I is closer to their writing than substituting a "
           "designed letterform would be."),
     ("note", "The mapping is deliberately not symmetric in every case. An "
              "exclamation mark may borrow an l, but an l may not borrow an "
              "exclamation mark, because the dot would come with it. "
              "Substitution is only valid in the direction where nothing "
              "spurious is inherited."),
   ]),

 "_BaselineGlyph": dict(
   inputs=[("ch", "the character wanted"),
           ("profile", "the writer's style profile"),
           ("printOnly", "force the printed alphabet even for a cursive "
                         "hand")],
   output="a pair of the strokes and the advance, scaled to this writer's "
          "reach; or nothing if the character is in neither alphabet.",
   theory=[
     ("h", "What it is for"),
     ("p", "Fetches a built-in letterform and adapts it to the writer. "
           "Without the adaptation a substituted letter would be visibly "
           "foreign: a generic l reaching 1.75 x-heights dropped into a "
           "hand whose own ascenders reach 2.4 looks short, and in a hand "
           "that reaches only 1.4 it towers over its neighbours. The "
           "letterform itself is generic; its proportions should not be."),
     ("h", "Why only the zones outside the band are rescaled"),
     ("eq", r"y' \;=\; \begin{cases} "
            r"1 + (y-1)\,\max\!\big(0.35,\, \tfrac{\mathrm{asc}-1}{0.75}"
            r"\big) & y > 1 \\[4pt] "
            r"y \cdot \max\!\big(0.35,\, \tfrac{|\mathrm{desc}|}{0.55}"
            r"\big) & y < 0 \\[4pt] "
            r"y & 0 \le y \le 1 \end{cases}",
      [("y", "the built-in letterform's height at this point, in "
             "x-heights"),
       ("y'", "the adapted height"),
       (r"\mathrm{asc}", "this writer's measured ascender reach, in "
                         "x-heights above the baseline"),
       (r"\mathrm{desc}", "their measured descender reach, negative"),
       ("0.75", "the ascender reach above the band that the built-in forms "
                "were drawn with, so dividing by it gives a ratio of 1 for "
                "a writer matching them"),
       ("0.55", "the same for the descender zone"),
       (r"\max(0.35,\cdot)", "a floor, so a writer with very little "
                             "measured reach does not collapse the zone to "
                             "nothing")],
      None),
     ("p", "The x-height band itself, from 0 to 1, is left untouched, and "
           "that is the point of the piecewise form. The band is the "
           "reference against which everything is measured, so it is 1 by "
           "definition and rescaling it would be meaningless. What varies "
           "between writers is how far they reach *beyond* the band, and "
           "those are exactly the two zones the formula adjusts. A letter "
           "like an o, entirely inside the band, is therefore unchanged, "
           "while an l is stretched or compressed only above y = 1, leaving "
           "its body the same size as its neighbours'."),
     ("h", "Why there is a floor of 0.35"),
     ("p", "A writer whose measured ascender is close to 1.0, which happens "
           "when the measurement failed or when most of their ascender "
           "letters were rejected, would otherwise give a ratio near zero "
           "and flatten every ascender to the height of the band. An l "
           "indistinguishable from an e is worse than a slightly wrong "
           "l, so the floor keeps at least a third of the designed reach "
           "whatever the measurement says."),
     ("h", "How the advance is derived"),
     ("p", "The advance is the shape's own horizontal extent plus 0.12 "
           "x-heights of spacing, computed from the points rather than "
           "stored. Deriving it means the two alphabets need no advance "
           "table to be maintained alongside them, and the margin is added "
           "so that consecutive fallback letters do not touch."),
   ]),

 "UNIFORM_INK": dict(
   output="true, meaning every writer is drawn at the same pen width.",
   theory=[
     ("h", "Why the ink is uniform, and why that is not a limitation"),
     ("p", "The profile records the writer's own stroke width, so the "
           "renderer could vary the pen width per writer. By default it "
           "does not, and the reason is that the machine cannot. The "
           "plotter has one pen fitted in one holder; every stroke it draws "
           "is the same width. Rendering a writer with a thick pen when the "
           "machine will draw them with the standard one would produce an "
           "image unlike the thing being built."),
     ("p", "That matters because the rendered image is not only for "
           "display: it is what the two judging networks score during the "
           "best-of-N search. A search that optimised against images the "
           "machine cannot produce would be optimising the wrong quantity. "
           "So the renderer is made to match the hardware, and the search "
           "then measures what will actually be drawn."),
     ("note", "The non-uniform path is retained and is reachable by passing "
              "the flag, because it is the right choice when rendering a "
              "comparison against a real photographed page, where the real "
              "page does have the writer's own pen width. The default "
              "matches the machine; the option matches the camera."),
   ]),

 "UNIFORM_PEN_WIDTH_XH": dict(
   output="0.085, the uniform pen width as a fraction of the x-height.",
   theory=[
     ("h", "Why the width is relative to the x-height"),
     ("p", "A width in pixels or millimetres would be right at one writing "
           "size only. Expressed as a fraction of the x-height it scales "
           "with the writing, so text drawn large and small both look like "
           "the same pen, which is what a real pen does."),
     ("h", "Why it is thin"),
     ("p", "The code comment records that this value was reduced from 0.11, "
           "and the reason is specific: a heavy stroke fills small loops. "
           "The enclosed white space inside an a, e or o is only a fraction "
           "of an x-height across, and a pen drawn at 0.11 closes it, "
           "turning an e into a c with a filled top and an a into a blob. "
           "Those are precisely the confusions that cost character "
           "accuracy, so the thinner pen is chosen to keep the counters "
           "open."),
     ("p", "The trade-off is that a thinner stroke carries less ink, so a "
           "rendered line is lighter than a photographed one. That is "
           "addressed separately, by the soft edge below and by the ink "
           "density matching, rather than by thickening the pen."),
   ]),

 "UNIFORM_SOFT_FRAC": dict(
   output="0.30, the blur radius as a fraction of the pen width.",
   theory=[
     ("h", "Why a drawn stroke is blurred"),
     ("p", "A rasterised stroke has a hard edge: a pixel is ink or paper "
           "with nothing in between. A photographed pen stroke does not. "
           "Ink spreads slightly into the paper, the camera's optics are "
           "not perfect, and the sensor averages over each pixel's area, so "
           "a real stroke edge is a gradient a pixel or two wide."),
     ("p", "This is the domain-shift problem described under "
           "RenderRefStats. The networks judging the output were trained on "
           "photographs, so they have learned what a real stroke edge looks "
           "like. A hard-edged synthetic stroke is unlike anything in their "
           "training data, and their response to it is unreliable, which "
           "would corrupt the very scores the search is optimising. "
           "Blurring makes the rendered stroke statistically closer to a "
           "photographed one."),
     ("h", "Why the radius is a fraction of the pen width"),
     ("p", "Because the physical cause scales that way. The spread of ink "
           "and the softness of a stroke edge are roughly proportional to "
           "how much ink was laid down, so a thick stroke has a wider soft "
           "edge than a thin one. A fixed blur radius would over-soften a "
           "thin stroke, which at this pen width would close the small "
           "loops the thin pen was chosen to keep open, and under-soften a "
           "thick one."),
   ]),

 "JOIN_PROB_CAP": dict(
   output="1.0, the greatest probability with which two letters may be "
          "joined.",
   theory=[
     ("h", "What it would do"),
     ("p", "Joining is decided per pair of letters by a random draw against "
           "the writer's connectedness, so a hand measured at 0.8 joins "
           "about four pairs in five. This constant caps that probability, "
           "so a value below 1 would force even a fully joined hand to "
           "leave some pairs unjoined."),
     ("h", "Why it is set to 1, which disables it"),
     ("p", "At 1.0 the cap never binds and the writer's measured "
           "connectedness is used as it stands. It is kept as a named "
           "constant rather than removed because it is the natural place to "
           "intervene if fully joined output proves unreadable, and because "
           "the alternative of editing the join condition itself is a change "
           "in behaviour hidden inside a long function. A constant at its "
           "neutral value documents an available adjustment."),
     ("note", "Deciding each pair by a random draw, rather than by "
              "comparing connectedness against a threshold, is the "
              "substantive choice here. A threshold would make a hand "
              "either fully joined or fully separated, with nothing in "
              "between, whereas real writing of intermediate connectedness "
              "joins some pairs and not others. The draw reproduces that."),
   ]),

 "CORE_SLANT_CAP_DEG": dict(
   output="22.0, the greatest lean in degrees applied to a printed anchor "
          "letterform.",
   theory=[
     ("h", "What it is for"),
     ("p", "Every glyph is sheared to the writer's measured slant, so the "
           "page leans as they do. An anchor letterform is sheared too, "
           "because a perfectly upright substitute in a slanted hand is "
           "conspicuous. But it is sheared less, and this constant is the "
           "limit."),
     ("h", "Why the anchor is treated differently"),
     ("p", "The anchor exists to be readable; that is the whole reason it "
           "was substituted for the writer's own shape. Shearing degrades "
           "legibility, because a letterform drawn upright and then "
           "slanted is not the same as a letterform designed at that "
           "slant: the curves distort, loops narrow, and at a steep angle "
           "an o becomes a thin diagonal ellipse. Applying a 42-degree "
           "slant to the anchor would undo the substitution's purpose."),
     ("p", "So the anchor keeps the writer's lean up to a point and then "
           "stops. Twenty-two degrees is a visible lean, enough that the "
           "substituted letter does not stand upright among slanted "
           "neighbours, while being well short of the angle at which the "
           "shear starts closing loops. A cursive anchor is allowed 30 "
           "degrees, since those forms were drawn to be joined and already "
           "lean."),
     ("note", "This is one of several places where the anchor is "
              "deliberately held back from the writer's full style: it also "
              "receives less jitter and only a fraction of the baseline "
              "drift. The pattern is consistent, and the reasoning is "
              "always the same: style is applied to the anchor only as far "
              "as it does not cost the legibility that justified using it."),
   ]),

 "CORE_JITTER_FRAC": dict(
   output="0.15, the fraction of the usual random variation applied to an "
          "anchor letterform.",
   theory=[
     ("h", "What it is for"),
     ("p", "Each glyph is nudged, rotated and scaled slightly at random so "
           "that repeated letters are not identical; see _Jitter. An anchor "
           "receives only 15 percent of that, so it barely wobbles."),
     ("h", "Why"),
     ("p", "The same reasoning as the slant cap. Jitter exists to make the "
           "output look handwritten rather than printed, which is a "
           "cosmetic goal, and it costs a little legibility, which is why "
           "the amount is small to begin with. An anchor was substituted "
           "precisely because legibility was at risk for that character, so "
           "it is the glyph that can least afford the cost. Keeping a small "
           "fraction rather than none at all means an anchor among its "
           "neighbours is not conspicuously rigid."),
   ]),

 "COLLISION_MIN_GAP_XH": dict(
   output="0.05, the clear space forced between one glyph's ink and the "
          "next, in x-heights.",
   theory=[
     ("h", "The problem it solves"),
     ("p", "Placing glyphs by their advances assumes each glyph's ink fits "
           "inside the space its advance allots. That assumption fails "
           "regularly. A variant may be wider than the advance recorded for "
           "its character; the random scaling may enlarge it; the shear may "
           "push the top of a tall letter well to the right of its base. "
           "When it fails, the next glyph is drawn on top of the previous "
           "one, and two overlapping letters are usually unreadable as "
           "either."),
     ("h", "How it is detected and fixed"),
     ("p", "After a glyph is placed, its leftmost inked point is compared "
           "with the rightmost inked point of the previous glyph's body. If "
           "the gap is less than this constant, the pen is pushed right by "
           "the shortfall and the glyph is placed again at the new "
           "position."),
     ("eq", r"\mathrm{need} \;=\; x^{\mathrm{prev}}_{\max} \;+\; "
            r"0.05\,\mathrm{xh} \;-\; x^{\mathrm{cur}}_{\min}",
      [(r"\mathrm{need}", "how far the current glyph must be pushed right; "
                          "nothing is done when it is not positive"),
       (r"x^{\mathrm{prev}}_{\max}", "the rightmost inked point of the "
                                     "previous glyph's body stroke"),
       (r"x^{\mathrm{cur}}_{\min}", "the leftmost inked point of the glyph "
                                    "just placed"),
       (r"\mathrm{xh}", "the x-height in millimetres, converting the "
                        "constant into the working units")],
      None),
     ("p", "Measuring actual ink positions rather than advances is what "
           "makes this work: it tests the thing that matters, where the "
           "ink really landed, after every transformation has been applied. "
           "A check based on advances could not see a collision caused by "
           "the shear, since the shear does not change the advance."),
     ("h", "Why the correction only ever pushes right"),
     ("p", "Writing proceeds left to right and the previous glyph has "
           "already been committed, so the only available remedy is to move "
           "the current glyph on. Pulling it left would reintroduce the "
           "collision, and moving the previous one would require "
           "reconsidering everything before it. The cost of pushing right "
           "is a slightly wider than intended gap, which is a far smaller "
           "defect than an overlap."),
   ]),

 "_ELIGIBLE_VARIANTS": dict(
   output="5, how many of a character's stored variants may be chosen from.",
   theory=[
     ("h", "The problem it solves"),
     ("p", "Up to twelve variants of a letter are stored, and it might seem "
           "that drawing from all of them would give the most natural "
           "variety. It does the opposite, and the reason is worth stating "
           "carefully. A person forms a given letter fairly consistently: "
           "their a's differ in detail but share a shape. The stored "
           "variants are ranked by how close each sits to what the letter "
           "should look like, so the first few are that consistent shape "
           "and the tail is the unusual ones, including the mis-cuts that "
           "survived filtering."),
     ("p", "Sampling evenly over all twelve therefore draws from the tail "
           "as often as from the characteristic shape. The output varies "
           "more, but what it varies over is no longer the writer's hand: "
           "it is a mixture of their hand and the extraction's errors. The "
           "result reads as scrambled rather than natural, and the writer "
           "identification drops, because the signature being averaged away "
           "is exactly what the namer recognises."),
     ("h", "Why five"),
     ("p", "Enough variants that repeated letters in a line are visibly "
           "different, which one or two would not achieve, and few enough "
           "that all of them are drawn from the well-formed part of the "
           "ranking. The selection among those five is itself weighted "
           "toward the first, so even within the eligible set the most "
           "typical shape is drawn most often."),
     ("note", "This cap and the storage cap of twelve interact in a way "
              "worth knowing: variants ranked sixth and below are stored "
              "but can never be drawn. For this project's profiles that is "
              "a substantial fraction of everything stored. The storage is "
              "not wasted, since the whole set is what the medoid and the "
              "prototype are computed from, but raising the storage cap "
              "alone would not produce more variety in the output."),
   ]),

 "CORE_CURSIVE_CONN": dict(
   output="0.5, the connectedness at or above which a hand is treated as "
          "cursive.",
   theory=[
     ("h", "What it decides"),
     ("p", "Which of the two built-in alphabets a writer's anchor letters "
           "come from, and whether an anchor may be joined to its "
           "neighbours at all. A hand at or above 0.5 gets the cursive "
           "forms and may join them; below, the printed forms, which never "
           "join."),
     ("h", "Why the midpoint"),
     ("p", "Connectedness is the fraction of adjacent in-word letter pairs "
           "that were genuinely joined, so 0.5 means half. That is the "
           "natural place to divide, and it is not a tuned value: a hand "
           "that joins more pairs than it separates is cursive in the "
           "ordinary sense of the word, and one that separates more is "
           "not."),
     ("h", "Why a threshold is acceptable here, unlike for joining"),
     ("p", "A hard threshold was rejected for deciding whether to join a "
           "given pair, because real writing of intermediate connectedness "
           "joins some pairs and not others, which a threshold cannot "
           "express. Here a threshold is right, because the thing being "
           "chosen is which alphabet to draw from, and there are only two "
           "alphabets. There is no meaningful way to be half-way between "
           "the printed form of an a and the cursive one; a choice must be "
           "made, and the measured majority behaviour is the sound basis "
           "for making it."),
   ]),

 "JOIN_LEGIBILITY_COUPLING": dict(
   output="0.0, how much the legibility setting reduces joining.",
   theory=[
     ("h", "What it would do"),
     ("p", "Joins are a major cause of illegibility in synthetic cursive: a "
           "ligature drawn to the wrong point, or several in succession, "
           "turns a word into a tangle. This constant would let the "
           "legibility setting thin the joins out, so that a writer whose "
           "library is poor would also be drawn with fewer joins."),
     ("eq", r"\mathrm{conn}_{\mathrm{eff}} \;=\; \mathrm{conn} \cdot "
            r"\big(1 - \kappa \cdot \mathrm{lam}\big)",
      [(r"\mathrm{conn}_{\mathrm{eff}}", "the connectedness actually used "
                                         "when deciding whether to join"),
       (r"\mathrm{conn}", "the writer's measured connectedness"),
       (r"\kappa", "this constant, the coupling strength"),
       (r"\mathrm{lam}", "the legibility setting, from 0 to 1")],
      None),
     ("h", "Why it is zero"),
     ("p", "At zero the expression reduces to the measured connectedness "
           "and the coupling has no effect, which is the current "
           "behaviour. The code comment marks this as deliberate. The "
           "reasoning is that connectedness is the single most "
           "recognisable property of a hand, and the namer relies on it "
           "heavily: thinning the joins of a cursive writer makes the "
           "output more readable and markedly less like them. Since that "
           "writer's identification is already the weaker of the two "
           "measures, trading it away for text accuracy is the wrong "
           "direction."),
     ("note", "Like JOIN_PROB_CAP, this is a mechanism left in place at its "
              "neutral setting. Both are the result of trying the "
              "adjustment and finding it harmful, and leaving the constant "
              "named at zero records that the possibility was considered, "
              "which removing it would not."),
   ]),

 "IDEAL_ANCHOR_MAX_PRIORD": dict(
   output="0.30, how far from typical the writer's own best letterform may "
          "sit and still serve as the anchor.",
   theory=[
     ("h", "The decision it governs"),
     ("p", "When a glyph needs replacing, the first choice of replacement "
           "is the writer's *own* most typical version of that letter, not "
           "a built-in form. That keeps the substitution inside their hand. "
           "But their most typical version is only useful if it is itself a "
           "decent letter, and for a heavily cursive writer it may not be: "
           "if most of their a's were cut badly, the most typical of them "
           "is a typical bad cut. This constant is where that judgement is "
           "made, against the prior distance described in style.py."),
     ("h", "Why 0.30"),
     ("p", "The prior distance runs from 0 for a shape matching the "
           "cross-writer reference exactly upward. The same scale appears "
           "in the legibility thresholds, where 0.12 counts as clean and "
           "0.55 as unusable. At 0.30 the limit sits between those: a shape "
           "noticeably unlike the general idea of the letter is still "
           "accepted, because it is the writer's own and the alternative is "
           "a generic form, but one approaching the unusable range is not. "
           "It is deliberately permissive, since preferring the writer's "
           "own shape is the whole point of trying them first."),
   ]),

 "IDEAL_SOURCE": dict(
   output="'medoid', selecting which of the two stored fallback letterforms "
          "is used as the anchor.",
   theory=[
     ("h", "The two candidates"),
     ("p", "style.py stores two single letterforms per character. The "
           "**medoid** is the one real variant most typical of the writer's "
           "own set. The **prototype**, stored as idealGlyphs, is the "
           "per-point median of several aligned variants, which is cleaner "
           "than any real variant but is an average and so is not something "
           "the writer ever wrote."),
     ("h", "Why the medoid is chosen"),
     ("p", "It is real writing. An averaged shape has had the irregularity "
           "removed along with the noise, and that irregularity is part of "
           "what identifies the writer, so a page of prototypes is smoother "
           "and less recognisable. The medoid keeps the writer's own "
           "character while still being the best-formed example they "
           "produced."),
     ("p", "There is a second reason, visible in this project's "
           "measurements. Prototype-based anchors improved text accuracy "
           "slightly and cost more in writer identification than they "
           "gained, which is the wrong trade for a system whose purpose is "
           "to reproduce an individual hand. The constant is a string "
           "rather than a boolean so that switching between them is a "
           "one-word change, which is how that comparison was made."),
   ]),

 "_PRIOR_REJECT": dict(
   output="0.25, a prior distance beyond which a stored variant would not be "
          "offered.",
   theory=[
     ("h", "What it was for"),
     ("p", "A hard cut-off: a stored shape further than this from the "
           "cross-writer reference would simply not be considered for "
           "drawing, however the legibility setting stood."),
     ("h", "Why the approach it represents was superseded"),
     ("p", "A hard rejection is the wrong instrument for this decision, and "
           "comparing it with what replaced it is instructive. It admits no "
           "middle ground: a variant at 0.249 is used freely and one at "
           "0.251 is never used, although nothing distinguishes them. For a "
           "writer whose every variant of some letter sits above the line "
           "it rejects all of them, leaving no shape at all. And it cannot "
           "take account of how trustworthy the library is as a whole, so "
           "it treats a stray bad variant in a good library the same as a "
           "uniformly poor one."),
     ("p", "The mechanism that replaced it, the probabilistic swap in "
           "_SwapProb, addresses all three. It is graded rather than "
           "binary, so borderline shapes are used sometimes; it always "
           "leaves a shape available, because swapping requires a "
           "replacement to exist; and it is driven by a per-writer "
           "legibility value, so the same variant is treated differently "
           "depending on the company it keeps."),
     ("note", "This constant is declared and never read. It is recorded "
              "here as the earlier design rather than as live behaviour, "
              "because a reader finding it in the settings block would "
              "reasonably assume variants beyond 0.25 are excluded, and "
              "they are not."),
   ]),

 "Trajectory": dict(
   output="an object holding the pen path: a list of pen-down strokes in "
          "millimetres, plus metadata describing how it was produced.",
   theory=[
     ("h", "What it represents"),
     ("p", "The output of synthesis, and the thing the rest of the project "
           "consumes. It is a list of **pen-down paths**: each entry is a "
           "sequence of points the pen visits with its tip on the paper, "
           "and the gaps between consecutive entries are where the pen "
           "lifts. That representation carries exactly the information a "
           "plotter needs and nothing more."),
     ("h", "Why the pen-up moves are implicit"),
     ("p", "Nothing records the lifts explicitly. They are implied by the "
           "boundaries between strokes, because by definition the pen must "
           "rise at the end of one and fall at the start of the next. "
           "Storing them would duplicate information already present and "
           "create the possibility of the two disagreeing, which is a class "
           "of bug that cannot arise if the state is stored once."),
     ("h", "The coordinate convention, and the one place it changes"),
     ("p", "Coordinates are millimetres with y increasing upward, which is "
           "the natural orientation for describing writing and the "
           "convention the stored glyphs use. Images have y increasing "
           "downward, and some machines expect one or the other. Rather "
           "than taking a position on that, the trajectory states its own "
           "convention and each consumer converts: the rasteriser flips y "
           "as it draws, and the G-code writer flips it if the machine "
           "needs it. The flip therefore happens at the boundary, once per "
           "consumer, rather than being half-applied inside the "
           "trajectory."),
     ("h", "What the two methods report"),
     ("items", [
       "**Bounds** returns the smallest rectangle containing every point, "
       "which the rasteriser needs to size its canvas and the G-code "
       "writer needs to check the drawing fits the machine's travel.",
       "**PenTravelMm** returns the distance drawn with the pen down and "
       "the distance moved with it up. The first is how much ink the page "
       "needs; the second is wasted motion, and it is the quantity a "
       "stroke-ordering optimisation would try to reduce.",
     ]),
     ("note", "This is one of only two classes in the project, and it "
              "earns the form: it bundles data with two derived "
              "measurements of that same data, which is what a class is "
              "for. The rest of the pipeline is plain functions, because "
              "elsewhere there is no state that several operations need to "
              "share."),
   ]),

 "_GLYPH_CLEAN": dict(
   output="0.12, the prior distance at or below which a stored shape counts "
          "as clean.",
   theory=[
     ("h", "What it marks"),
     ("p", "The lower of the two reference points on the prior-distance "
           "scale used by the swap rule. A shape this close to the "
           "cross-writer reference for its letter is a well-formed example "
           "of that letter and is kept almost always, whatever the "
           "legibility setting says."),
     ("note", "Almost always, not always: even a clean shape carries a "
              "small swap probability proportional to lam. That is "
              "deliberate, because a library can be poor overall while "
              "containing some clean shapes, and a writer whose lam is high "
              "is one whose extraction is not trusted in general."),
   ]),

 "_GLYPH_GARBAGE": dict(
   output="0.55, the prior distance at or above which a stored shape is "
          "always replaced.",
   theory=[
     ("h", "What it marks"),
     ("p", "The upper reference point. A shape this far from the reference "
           "is not a recognisable instance of its letter at all: a "
           "fragment, or a cut carrying most of a neighbour. It is replaced "
           "unconditionally, with no regard to the legibility setting."),
     ("h", "Why this one case is unconditional"),
     ("p", "Every other decision in the swap rule is probabilistic, so this "
           "exception needs justifying. The argument is that lam expresses "
           "a trade-off between fidelity and legibility, and the trade-off "
           "only exists where there is something to trade. Drawing a "
           "fragment is not more faithful to the writer; it reproduces the "
           "extraction's mistake rather than their hand. With no fidelity "
           "to be gained, there is nothing to weigh against legibility, and "
           "the shape goes."),
   ]),

 "_COMMIT_LAM": dict(
   output="0.85, the legibility setting at or above which the library is "
          "abandoned entirely.",
   theory=[
     ("h", "What it does"),
     ("p", "Above this setting every stored shape is replaced by the "
           "anchor, so the writer's extracted library contributes nothing "
           "and the output is built from anchor letterforms throughout."),
     ("h", "Why there is a hard commitment point at all"),
     ("p", "Without it, a writer at lam 0.95 would still draw a few "
           "percent of glyphs from their library, and those few would be "
           "drawn from a library that the lam value itself declares "
           "untrustworthy. The result is the worst of both: a page that is "
           "mostly anchor letterforms, so it does not look much like the "
           "writer, with a scattering of fragments in it, so it is not "
           "clean either. Committing fully above 0.85 at least produces a "
           "consistent page."),
     ("p", "This case genuinely arises. A heavily cursive hand has no "
           "honest place to cut between letters, so most of its variants "
           "are fragments, its median prior distance is high, and "
           "AutoLegibility returns a value near 1. For such a writer the "
           "output is drawn almost entirely from anchor forms, which is the "
           "correct response to a library that could not be extracted, and "
           "it is also the mechanism behind this project's measured result "
           "that heavily cursive hands reproduce less faithfully than clean "
           "ones."),
   ]),

 "_SwapProb": dict(
   inputs=[("lam", "the legibility setting, from 0 to 1"),
           ("priorD", "this shape's distance from the cross-writer "
                      "reference")],
   output="the probability, from 0 to 1, that this shape is replaced by the "
          "anchor.",
   theory=[
     ("h", "What it is for"),
     ("p", "This small function is where the project's central trade-off is "
           "actually resolved. For one glyph, given how far it sits from "
           "what the letter should look like and how much the library as a "
           "whole is trusted, it returns the chance that the writer's own "
           "shape is set aside in favour of a clean one."),
     ("h", "The four cases, and the order they are tested in"),
     ("eq", r"P_{\mathrm{swap}} \;=\; \begin{cases} "
            r"1 & d \ge 0.55 \\[3pt] "
            r"1 & \mathrm{lam} \ge 0.85 \\[3pt] "
            r"0 & \mathrm{lam} \le 0.001 \\[3pt] "
            r"0.06\,\mathrm{lam} & d \le 0.12 \\[3pt] "
            r"\mathrm{clip}\big(\mathrm{lam}\,(0.55 + 0.9\,b),\,0,\,1\big) "
            r"& \text{otherwise} \end{cases}",
      [(r"P_{\mathrm{swap}}", "the returned swap probability"),
       ("d", "the shape's prior distance"),
       (r"\mathrm{lam}", "the legibility setting"),
       ("b", "how bad the shape is on a 0 to 1 scale, defined below"),
       ("0.55", "the base swap rate for a shape of middling quality"),
       ("0.9", "how much worse quality raises that rate"),
       (r"\mathrm{clip}(\cdot,0,1)", "the result held to a valid "
                                     "probability")],
      None),
     ("p", "The order matters because the cases overlap. Garbage is tested "
           "first, so a hopeless shape is replaced even at lam 0. Full "
           "commitment is next, so above 0.85 everything goes regardless of "
           "quality. Only then is lam 0 tested, which keeps every shape "
           "that survived the first test: this is what makes lam 0 mean "
           "\"faithful reproduction, defects and all\" rather than \"no "
           "filtering whatever\"."),
     ("h", "The interpolation in the middle range"),
     ("eq", r"b \;=\; \mathrm{clip}\!\left( "
            r"\frac{d - 0.12}{0.55 - 0.12},\; 0,\; 1 \right)",
      [("b", "the badness of the shape: 0 at the clean threshold, 1 at the "
             "garbage threshold"),
       ("d", "the shape's prior distance"),
       ("0.12,\\;0.55", "the clean and garbage reference points"),
       (r"\mathrm{clip}(\cdot,0,1)", "held in range, although the earlier "
                                     "cases already guarantee d lies "
                                     "between the two")],
      None),
     ("p", "This maps the prior distance onto a 0 to 1 scale between the "
           "two reference points, which is what lets a single expression "
           "cover the whole middle range. The swap probability is then lam "
           "times a factor running from 0.55 at the clean end to 1.45 at "
           "the garbage end, clipped to 1. So at a given lam, a worse shape "
           "is swapped more often, and at a given quality, a less trusted "
           "library is swapped more often. The two inputs act "
           "multiplicatively, which is the behaviour wanted: neither alone "
           "should force a swap."),
     ("h", "Why a probability and not a decision"),
     ("p", "Returning a probability rather than a yes or no is what "
           "produces a mixed page, and a mixed page is the point. A "
           "threshold rule would draw every instance of a given character "
           "from the same source, so a letter whose shape sits near the "
           "line would be either always the writer's or always the anchor. "
           "With a probability, some instances are theirs and some are not, "
           "which both preserves more of the hand and spreads the risk: one "
           "bad instance of a letter in a line is recoverable from context, "
           "where every instance being bad is not."),
     ("note", "Because the draw is made per glyph from a seeded generator, "
              "the same text and seed give the same page. That is what "
              "makes the best-of-N search meaningful, since each attempt "
              "differs only by its seed, and what makes a result "
              "reproducible rather than something that happened once."),
   ]),

 "_AUTO_LAM_GOOD": dict(
   output="0.24, the median prior distance at or below which a library is "
          "trusted completely.",
   theory=[
     ("h", "What it marks"),
     ("p", "The lower end of the scale AutoLegibility maps onto. A writer "
           "whose variants have a median prior distance at or below this "
           "gets lam 0, meaning their own shapes are used throughout."),
     ("h", "Why the median across the whole library"),
     ("p", "The quantity being judged is the trustworthiness of the "
           "extraction as a whole, not of any one glyph, which the swap "
           "rule already handles per glyph. The median over every stored "
           "variant of every character answers the right question: is this "
           "writer's library, in general, made of letters? Using the median "
           "rather than the mean keeps the answer from being set by the "
           "worst few."),
   ]),

 "_AUTO_LAM_BAD": dict(
   output="0.32, the median prior distance at or above which a library is "
          "not trusted at all.",
   theory=[
     ("h", "What it marks"),
     ("p", "The upper end of the same scale. At or above this, lam is 1, "
           "the library is abandoned, and the output is drawn from anchor "
           "letterforms."),
     ("h", "Why the window is so narrow"),
     ("p", "The two thresholds are only 0.08 apart, which makes the mapping "
           "steep: a small change in median prior distance moves lam a long "
           "way. That is deliberate, and it reflects what the measurements "
           "show. Across this project's writers the median prior distance "
           "clusters tightly, around 0.18 to 0.24 for the hands that "
           "extracted cleanly and around 0.31 for the two most cursive. The "
           "thresholds are placed on either side of that gap, so the "
           "mapping separates the populations that actually exist rather "
           "than spreading a gentle gradient over a range where no writer "
           "falls."),
     ("note", "The consequence is that most writers receive a lam of "
              "exactly 0 or exactly 1, with few in between. That is "
              "appropriate for a measurement with two clear groups, but it "
              "does mean the intermediate behaviour of the swap rule is "
              "exercised mainly by the repair mechanism, which raises lam "
              "for individual characters rather than for the writer."),
   ]),

 "AutoLegibility": dict(
   inputs=[("profile", "the writer's style profile")],
   output="the legibility setting for that writer, from 0 to 1.",
   theory=[
     ("h", "What it is for"),
     ("p", "Derives the legibility setting from the profile instead of "
           "requiring it to be chosen. The question it answers is: how far "
           "can this writer's extracted library be trusted?"),
     ("eq", r"\mathrm{lam} \;=\; \mathrm{clip}\!\left( "
            r"\frac{\mathrm{median}\{d_g\} - 0.24}{0.32 - 0.24},\; 0,\; 1"
            r"\right)",
      [(r"\mathrm{lam}", "the returned legibility setting"),
       ("d_g", "the prior distance of stored variant g, over every variant "
               "of every character in the library"),
       (r"\mathrm{median}", "the middle value of those distances"),
       ("0.24,\\;0.32", "the good and bad reference points above"),
       (r"\mathrm{clip}(\cdot,0,1)", "held within the valid range, which is "
                                     "what makes the mapping flat outside "
                                     "the two reference points")],
      None),
     ("h", "Why this works, and what it is really measuring"),
     ("p", "The chain of reasoning behind it is the substantive part. The "
           "prior distance measures how far a stored shape sits from what "
           "its letter should look like. A library whose shapes sit far "
           "from that is a library of bad cuts. Bad cuts come from "
           "boundaries placed where no honest boundary exists. And that "
           "happens exactly when the writing is joined up, because in "
           "cursive the pen does not lift between letters, so there is no "
           "place to cut that does not damage something."),
     ("p", "So the median prior distance is, indirectly, a measure of how "
           "cursive the hand is, arrived at by measuring the consequence "
           "rather than the cause. That is why it works as a trust signal: "
           "it responds to the thing that actually makes an extraction "
           "unreliable. And it is why heavily cursive writers end up drawn "
           "from anchor forms, which is the mechanism behind this project's "
           "finding that such hands reproduce less faithfully."),
     ("h", "Why an empty library returns 1"),
     ("p", "With no variants there is nothing to measure and nothing to "
           "draw from, so the anchor is the only available source and lam 1 "
           "states that. Returning 0 instead would claim a library worth "
           "trusting, and the caller would then try to draw from one that "
           "does not exist."),
     ("note", "A profile may carry an explicit legibilityLambda, which "
              "callers prefer over this function. That is how a value found "
              "by experiment for a particular writer can override the "
              "derived one, and it is why the stored lam column appears in "
              "this project's profile tables even though it is usually "
              "empty."),
   ]),

 "_CoreGlyph": dict(
   inputs=[("ch", "the character wanted"),
           ("profile", "the writer's style profile")],
   output="the anchor letterform as strokes, advance, lead and whether it "
          "is a cursive form; or nothing if no anchor exists.",
   theory=[
     ("h", "What the anchor is"),
     ("p", "The clean letterform that a stored variant is swapped for. The "
           "function implements a preference order with two steps, and the "
           "order is the whole content of it."),
     ("h", "First choice: the writer's own most typical letterform"),
     ("p", "The medoid stored in the profile is tried first, accepted if "
           "its prior distance is within IDEAL_ANCHOR_MAX_PRIORD. This "
           "matters more than it may appear. An anchor is by definition "
           "drawn when the writer's chosen variant was not trusted, so "
           "anchors account for a large share of a cursive writer's page. "
           "If the anchor were always a built-in form, such a page would be "
           "generic handwriting with the writer's spacing, and the namer "
           "would not identify it. Preferring their own medoid keeps the "
           "substitution inside their hand: it is still a letter they "
           "actually wrote, merely the best-formed one rather than a "
           "random one."),
     ("p", "This project's experiments bear that out directly. Extending "
           "the anchor-selection improvements to more writers raised writer "
           "identification substantially for the most cursive hand, and the "
           "gain came from better anchor letterforms rather than from any "
           "change to how variants were chosen."),
     ("h", "Second choice: the built-in alphabet"),
     ("p", "If the medoid is absent or too rough, _BaselineGlyph supplies a "
           "built-in form, cursive or printed according to the writer's "
           "connectedness, scaled to their reach. This is a real loss of "
           "fidelity and is accepted only because the alternative is "
           "drawing a shape that is not a letter."),
     ("h", "Why the advance is taken as a maximum"),
     ("eq", r"\mathrm{adv} \;=\; \max\big(\,\mathrm{adv}_{\mathrm{stored}},"
            r"\;\; \mathrm{lead} + \mathrm{width} + 0.04 \,\big)",
      [(r"\mathrm{adv}", "the advance actually used"),
       (r"\mathrm{adv}_{\mathrm{stored}}", "the advance recorded with the "
                                           "letterform"),
       (r"\mathrm{lead}", "the gap from the cell's left edge to where the "
                          "ink starts"),
       (r"\mathrm{width}", "the horizontal extent of the ink"),
       ("0.04", "a small margin so two letters do not touch")],
      None),
     ("p", "The stored advance came from the cell the character was "
           "originally cut from, and that cell can be narrower than the ink "
           "it contains, since a cut can leave ink spilling past its own "
           "boundary. Advancing by the stored figure alone would then place "
           "the next letter on top of this one. Taking whichever of the two "
           "is larger guarantees the advance clears the ink while "
           "preserving the writer's own spacing wherever it is sufficient."),
   ]),

 "_GlyphSource": dict(
   inputs=[("profile", "the writer's style profile"),
           ("ch", "the character to draw"),
           ("rng", "the seeded random generator"),
           ("prevExitY", "the height the pen left the previous letter at, "
                         "or nothing if not joining"),
           ("lam", "the legibility setting for this glyph"),
           ("forceCore", "force the anchor regardless of the swap draw"),
           ("forcePrint", "force the printed alphabet, the last resort")],
   output="four things: the strokes, the advance, the lead, and a label "
          "saying which source the shape came from.",
   theory=[
     ("h", "What it is for"),
     ("p", "Given one character, decide what to draw. This is where the "
           "legibility trade-off, the variant ranking and the fallback "
           "chain all meet, and the label it returns is what makes the "
           "outcome auditable."),
     ("h", "The order of preference"),
     ("items", [
       "**forcePrint**, when set, returns the printed alphabet immediately. "
       "It is the last resort invoked by the repair loop for a character "
       "the reader still cannot read.",
       "**The writer's own variants** of this exact character, ranked and "
       "sampled as described below, then possibly swapped for the anchor. "
       "Labelled `own`, or `core` if swapped.",
       "**The opposite case** of the character. A capital drawn from the "
       "writer's lowercase is scaled up to their ascender height. Labelled "
       "`case`.",
       "**A similar character** from the _SIMILAR table, such as an l for "
       "an I. Labelled `similar`.",
       "**The anchor**, when the writer has nothing for this character at "
       "all. Labelled `core`.",
       "**The built-in alphabet** directly, labelled `fallback`.",
       "**Nothing**, labelled `blank`: no strokes, but the advance is still "
       "applied, so an unavailable character leaves a gap rather than "
       "closing up the line.",
     ]),
     ("p", "The order descends by how much of the writer's own hand each "
           "option retains, which is the right criterion for a system whose "
           "purpose is to reproduce an individual. Note that the swap "
           "check is applied in the first three branches, not only the "
           "first: a borrowed or case-converted shape is still a stored cut "
           "and can still be a bad one."),
     ("h", "How a variant is chosen: weighted sampling"),
     ("p", "The candidates are ranked by prior distance, cut to the "
           "eligible few, and one is drawn with probability proportional to "
           "a weight."),
     ("eq", r"w_i \;=\; \frac{1}{1 + 1.4\,i} \;\cdot\; "
            r"\exp\!\left(-\left(\frac{\Delta y_i}{0.45}\right)^{\!2}"
            r"\right)",
      [("w_i", "the weight given to the i-th ranked candidate"),
       ("i", "its position in the ranking, 0 for the most typical"),
       ("1.4", "how steeply the weight falls with rank"),
       (r"\Delta y_i", "the difference between this variant's entry height "
                       "and the height the pen left the previous letter "
                       "at"),
       ("0.45", "the tolerance on that difference, in x-heights"),
       (r"\exp(-(\cdot)^2)", "a Gaussian falloff, applied only when joining; "
                             "otherwise this factor is absent")],
      None),
     ("p", "The first factor implements the preference for typical shapes: "
           "the top-ranked variant is weighted 1, the second about 0.42, "
           "the fifth about 0.15. So the characteristic shape is drawn most "
           "often while the others still appear, which is what gives "
           "repeated letters their variation. A reciprocal falloff is "
           "gentler than an exponential one, which would make the top "
           "variant dominate so heavily that the others might as well not "
           "be eligible."),
     ("h", "Why the entry height matters when joining"),
     ("p", "The second factor is the more interesting one. When this letter "
           "will be joined to the previous one, a ligature must be drawn "
           "from where the pen left that letter to where this one begins. "
           "If the two heights differ a great deal, the ligature has to "
           "climb or dive, which is what produces the tangled joins that "
           "make synthetic cursive unreadable. Since several variants are "
           "available and they differ in where they begin, the one whose "
           "entry height is closest to the pen's current height can be "
           "preferred, and the join is then short and natural."),
     ("p", "This is a small instance of a general principle worth naming: "
           "where a choice exists anyway, it costs nothing to make it with "
           "regard to context. The variant selection had to pick one of "
           "five; biasing that pick toward the one that joins well "
           "improves the output without any extra drawing or searching."),
     ("h", "Why the variant's own advance is used"),
     ("p", "Each variant carries its own advance, and that is used rather "
           "than the character's median. The code comment gives the reason: "
           "a wide variant advanced by a narrow median would collide with "
           "the next letter. The median describes the character across all "
           "its instances, but what is being drawn is one particular "
           "instance, and the spacing must suit the ink actually placed."),
     ("note", "The label returned is counted in the trajectory's metadata, "
              "giving a breakdown of how many glyphs came from each source. "
              "That breakdown is how the fidelity-legibility trade-off is "
              "measured in this project: a page reported as mostly `core` "
              "is readable but is largely not the writer's own letterforms, "
              "and the counts say so plainly where looking at the output "
              "would not."),
   ]),
}


#the rest of the chapter, in order
for _part in ("codetheory_synth2",):
    try:
        _m = __import__(_part)
    except ModuleNotFoundError:
        continue
    FUNCS.update(_m.FUNCS)
