"""recognise.py, part 1: the alphabet, the checkpoints, and the layers.

This chapter covers the two trained networks that read a line of writing
and say who wrote it.  Part 1 is the vocabulary, the checkpoint loader and
every layer the networks are built from.  Part 2, in
codetheory_recognise2.py, covers the recurrent half, CTC, and the two
network classes themselves.
"""

INTRO = [
 ("p", "Two trained neural networks live in this file. The first, called "
       "the reader throughout, takes a crop of one line of handwriting and "
       "returns the text written on it. The second, the namer, takes the "
       "same kind of crop and returns which of the enrolled writers "
       "produced it. Both run forward only; the code that trained them is "
       "in train.py."),
 ("h", "Background: what a neural network is, in the sense used here"),
 ("p", "A **neural network** is a function built by stacking simple "
       "operations, most of them a multiplication by a table of numbers "
       "followed by a simple non-linear squashing. The tables are called "
       "**weights** or **parameters**, and they are not written by hand: "
       "they are found by training, which means repeatedly adjusting them "
       "so that the network's output on examples moves closer to the known "
       "correct answer. A **layer** is one of those operations. Running the "
       "layers in order from input to output is the **forward pass**, which "
       "is all this file does; computing how each weight should change is "
       "the **backward pass**, which lives in train.py."),
 ("p", "Nothing in this file uses a machine-learning framework. Every "
       "layer is a few lines of array arithmetic, which is possible "
       "because a forward pass really is nothing more than that. The point "
       "of a framework is mostly to supply the backward pass "
       "automatically, and that was derived by hand for this project."),
 ("h", "The shape convention"),
 ("p", "Arrays flowing through the networks carry four axes in a fixed "
       "order, written throughout as (B, C, H, W)."),
 ("items", [
   "**B, the batch:** how many separate images are being processed at "
   "once. It is 1 everywhere in this file, since crops are read one at a "
   "time, but the axis is kept because the training code uses many.",
   "**C, the channels:** how many separate images the layer is carrying "
   "per input. The input crop has one channel, its greyness. After the "
   "first convolution it has as many channels as that layer had filters, "
   "each one a map of where some particular stroke feature was found.",
   "**H and W, height and width:** the spatial axes, in pixels.",
 ]),
 ("p", "The recurrent half of the reader uses a different convention, "
       "(T, B, F), where T is a position along the line, B is still the "
       "batch and F is a vector of features at that position. The "
       "transition between the two conventions happens inside the reader's "
       "forward method, and it is the single most important structural "
       "step in the network; it is explained there."),
]


FUNCS = {

 "CHARSET": dict(
   output="a string listing every character the reader is able to output, "
          "in the order of their slot numbers.",
   theory=[
     ("h", "What it is for"),
     ("p", "A network cannot emit a letter. It emits numbers, and something "
           "must fix which number means which letter. This string is that "
           "agreement: the character at position i of the string is what "
           "output slot i+1 of the network stands for. It is a constant "
           "rather than a setting because it was fixed when the network was "
           "trained; changing it now would silently relabel every output."),
     ("h", "What is in it, and what is deliberately not"),
     ("p", "It holds the 26 lowercase letters, the 26 uppercase letters, "
           "the ten digits, the space, and eleven punctuation marks, giving "
           "74 characters and so 75 output slots once the blank is counted. "
           "Upper and lower case are separate slots rather than being "
           "folded together, because the reader is required to reproduce "
           "capitalisation, and a letterform really is different between "
           "the two cases, so merging them would ask one slot to cover two "
           "unrelated shapes."),
     ("p", "What is absent is as deliberate. There are no accented letters "
           "and no currency or mathematical symbols, because none appear in "
           "the training data, and an output slot that is never trained is "
           "worse than no slot at all: it occupies capacity and can still "
           "win a column by accident."),
     ("h", "Why the space is an ordinary character"),
     ("p", "The space is in the set like any letter. This is worth noticing "
           "because it is not obvious: a space is the absence of ink, so it "
           "might seem to need no symbol. But the reader must distinguish "
           "a gap between words from a gap between the strokes of one "
           "letter, and that distinction is a judgement about the writing, "
           "not a measurement of white space. Making the space a trainable "
           "character lets the network learn where word boundaries fall "
           "from examples, rather than forcing a threshold on gap width to "
           "be chosen by hand."),
     ("note", "Slot 0 does not appear in this string. It is reserved for "
              "the CTC blank, which is not a character at all but a signal "
              "meaning \"nothing is being emitted here\". Its role is "
              "explained in the entry for DecodeCtc, and it is the reason "
              "every index in the two lookups below is shifted by one."),
   ]),

 "CHAR_TO_IDX": dict(
   output="a dictionary mapping each character of the charset to its output "
          "slot number, starting at 1.",
   theory=[
     ("h", "What it is for"),
     ("p", "Training and alignment both need the opposite direction from "
           "reading. Given the known text of a line, each character must "
           "become the slot number the network is supposed to favour, so "
           "that the network's output can be compared with it. This "
           "dictionary performs that conversion."),
     ("h", "Why a dictionary and not a search"),
     ("p", "The same answer could be had by searching the charset string "
           "for the character each time, but a dictionary is a **hash "
           "table**, meaning it computes a position directly from the key "
           "rather than scanning, so a lookup costs the same whether the "
           "set has ten entries or ten thousand. Since every character of "
           "every training label passes through this lookup, the difference "
           "is worth having for free."),
     ("h", "The offset by one"),
     ("p", "Slot numbers start at 1 rather than 0 because slot 0 belongs to "
           "the blank. Every use of this dictionary therefore inherits the "
           "offset, which is why the alignment code builds its label "
           "sequence from these values directly and never adds one of its "
           "own."),
     ("note", "A character absent from the charset has no entry, and the "
              "callers rely on that: alignment and training both filter the "
              "label text through a membership test and silently drop "
              "anything unknown. The alternative, substituting a "
              "placeholder, would teach the network to emit that "
              "placeholder for an unfamiliar mark instead of leaving it "
              "out."),
   ]),

 "IDX_TO_CHAR": dict(
   output="a dictionary mapping an output slot number back to its "
          "character.",
   theory=[
     ("h", "What it is for"),
     ("p", "This is the reverse lookup, used when the network's output is "
           "turned back into readable text. The decoder picks a winning "
           "slot per column and this dictionary says what letter that slot "
           "stands for."),
     ("h", "Why it is derived rather than written out"),
     ("p", "It is constructed by reversing the forward dictionary, not by "
           "listing the pairs again. Writing both by hand would create two "
           "places that must agree, and a disagreement between them would "
           "not raise an error; it would quietly produce text in which some "
           "letters are substituted for others, which is exactly the kind "
           "of fault that is hard to notice and harder to trace. Deriving "
           "one from the other makes the agreement structural."),
     ("note", "Slot 0 has no entry here either, which is correct: the blank "
              "is not a character and must never appear in decoded text. "
              "The decoder removes blanks before consulting this "
              "dictionary, and also passes a default of the empty string, "
              "so that an out-of-range slot contributes nothing rather than "
              "raising an error in the middle of reading a page."),
   ]),

 "WEIGHTS_DIR": dict(
   output="the directory holding every trained checkpoint file.",
   theory=[
     ("h", "What it is for"),
     ("p", "It names one place where trained weights live, taken from the "
           "paths module so that the location is defined once for the whole "
           "project. The value matters less than the fact that it is a "
           "single named constant: every checkpoint below is built from it, "
           "so moving the weights folder is one edit in paths.py rather "
           "than a search through several files."),
   ]),

 "INPUT_HEIGHT": dict(
   output="64, the crop height in pixels that both networks were trained "
          "on.",
   theory=[
     ("h", "Why the height is fixed and the width is too"),
     ("p", "A convolution can be applied to an image of any size, so it may "
           "seem that a network should accept any crop. The constraint "
           "comes from what the network learned. During training every crop "
           "was 64 pixels tall, so the filters learned the size a letter "
           "appears at when a line of writing is 64 pixels high, and the "
           "layer that collapses the height assumes that much to collapse. "
           "Presenting a 128-pixel crop would show the network letters at "
           "twice the scale its filters were tuned for, and the features "
           "they detect would simply not be found."),
     ("h", "Why 64 specifically"),
     ("p", "It is a compromise between detail and cost. A line of "
           "handwriting has to be tall enough that the difference between "
           "an a and an o survives, which needs a few tens of pixels "
           "across the x-height, and the crop must also hold the ascender "
           "and descender zones above and below it. Against that, every "
           "extra row multiplies the work in all the convolutional layers. "
           "A power of two is convenient because the network halves the "
           "height twice with pooling and then collapses what is left, so "
           "64 divides cleanly where 60 or 70 would leave a remainder to be "
           "discarded."),
   ]),

 "INPUT_WIDTH": dict(
   output="640, the crop width in pixels that both networks were trained "
          "on.",
   theory=[
     ("h", "Why the width is fixed as well"),
     ("p", "The width constraint is weaker in principle than the height "
           "one, because the reader treats width as the axis along which it "
           "reads and could in principle accept any length. It is fixed "
           "because training in batches requires every crop in a batch to "
           "be the same shape, and because a fixed width makes the mapping "
           "from an output column back to a pixel position a single scale "
           "factor, which is what FrameXToPixel relies on."),
     ("h", "The ten-to-one ratio"),
     ("p", "640 is ten times the height, which reflects the shape of a line "
           "of writing: a line is long and thin. After two halvings the "
           "width is 160 columns, and since the reader emits one set of "
           "character scores per remaining column, that sets the maximum "
           "length of text it can represent. The CTC method requires at "
           "least one column per character and in practice rather more, so "
           "160 columns comfortably covers the sixty or seventy characters "
           "a line of handwriting holds."),
   ]),

 "_TEXT_CANDIDATES": dict(
   output="a list of reader checkpoint filenames, searched in order.",
   theory=[
     ("h", "What it is for"),
     ("p", "Several reader checkpoints exist, produced by different "
           "training runs. This list names them in the order they are to be "
           "preferred, and the first one present on disk is the one that "
           "gets used. The arrangement means the module runs on whatever "
           "subset of the checkpoints a given copy of the project happens "
           "to have, instead of failing when one specific file is absent."),
     ("h", "What the three names mean"),
     ("items", [
       "**scratch:** trained from randomly initialised weights on the "
       "public database alone.",
       "**joint:** trained on the public database and this project's own "
       "photographed pages together.",
       "**hf:** trained from a checkpoint obtained elsewhere and then "
       "adapted, so not trained from scratch within this project.",
     ]),
     ("h", "Why the order of this list is a substantive decision"),
     ("p", "Because the first existing file wins, this list alone decides "
           "which network actually reads a page, with no further input from "
           "the caller. That makes the ordering a functional choice rather "
           "than a cosmetic one, and it is worth being explicit that the "
           "checkpoints are not interchangeable: measured on the same 669 "
           "photographed lines, the scratch and joint checkpoints differ by "
           "a very large margin in character accuracy, the joint one being "
           "far stronger. Whichever name stands first in this list is "
           "therefore the one that determines the accuracy a user sees."),
     ("note", "A list searched in preference order is a reasonable pattern, "
              "but it hides which choice was made behind the contents of a "
              "directory. Anything depending on the reader's accuracy "
              "should confirm which file was actually loaded, which is why "
              "the reader records its checkpoint path as an attribute."),
   ]),

 "DEFAULT_TEXT_WEIGHTS": dict(
   output="the path of the reader checkpoint chosen from the candidate "
          "list, or the first candidate if none exists.",
   theory=[
     ("h", "How the choice is made"),
     ("p", "The candidate names are tested in order and the first that "
           "exists on disk is taken. If none exists, the first name is "
           "returned anyway rather than nothing at all."),
     ("h", "Why a missing file is reported late rather than early"),
     ("p", "Returning a path that does not exist looks like an oversight "
           "but is deliberate. If this constant were allowed to be empty, "
           "every later use would need to check for that, and the "
           "eventual error would complain about an empty value rather than "
           "a missing file. Returning the expected path means the failure "
           "surfaces when a network is actually constructed, as an error "
           "naming the exact file that could not be found, which is the "
           "message someone can act on."),
     ("note", "Because this is evaluated once when the module is first "
              "imported, a checkpoint added to the directory afterwards is "
              "not picked up until the program restarts. Any code wanting a "
              "specific checkpoint should pass its path explicitly rather "
              "than relying on this constant."),
   ]),

 "_AUTHOR_NPZ": dict(
   output="the path of the preferred writer-naming checkpoint.",
   theory=[
     ("h", "What it is for"),
     ("p", "The first of three candidate checkpoints for the namer, "
           "trained from scratch on the writers enrolled in this project. "
           "It is preferred when present because it was trained within the "
           "project on the current set of writers, so its output slots "
           "match the writer list the rest of the pipeline uses."),
   ]),

 "_SHAPE": dict(
   output="the path of the shape-based writer-naming checkpoint.",
   theory=[
     ("h", "What makes this one different"),
     ("p", "It was trained on crops that had been stroke-normalised, "
           "meaning redrawn at a single constant pen width. The reason that "
           "matters is explained in full under StrokeNormalize: a namer "
           "trained on raw crops can identify a writer by how dark or thick "
           "their pen is, which is a property of the pen and not of the "
           "handwriting. Removing pen width forces the network to judge "
           "letter shapes instead."),
     ("p", "This is the checkpoint used to judge the robot's own output, "
           "and that use is why it exists. Drawn output has a single "
           "uniform line width by construction, since a plotter draws every "
           "stroke with the same pen. A namer that had learned to use ink "
           "thickness would therefore be judging the plotter rather than "
           "the style, and would answer the wrong question."),
   ]),

 "_W1": dict(
   output="the path of the ink-based writer-naming checkpoint.",
   theory=[
     ("h", "What makes this one different"),
     ("p", "It was trained on raw grey crops, with pen width and darkness "
           "left in. That makes it the right choice for identifying the "
           "writer of a real photographed page, where the pen is part of "
           "the evidence: a person's usual pen, and the pressure they "
           "write with, genuinely are characteristic of them. It is the "
           "checkpoint the live server uses on a photograph."),
     ("note", "The project therefore keeps two namers that answer the same "
              "question from different evidence, and which to use depends "
              "on what the input is. Measured on the same crops the "
              "ink-based one scores higher, which is expected, because it "
              "is allowed to use information the shape-based one has been "
              "deliberately denied. The higher score is not grounds for "
              "preferring it on drawn output, where that extra information "
              "does not exist."),
   ]),

 "DEFAULT_AUTHOR_WEIGHTS": dict(
   output="whichever of the three namer checkpoints was found, in order of "
          "preference.",
   theory=[
     ("h", "How the choice is made"),
     ("p", "The scratch-trained checkpoint is taken if it exists, "
           "otherwise the shape-based one, otherwise the ink-based one. "
           "The same preference-order idea as the reader's list, written as "
           "nested conditions because there are only three."),
     ("h", "Why the fallback order ends with the ink model"),
     ("p", "The ink-based checkpoint is last, which makes it the one "
           "reached when nothing else is present. That is the safe end of "
           "the order for a general-purpose default: the commonest use of "
           "the namer is on a photographed page, which is precisely the "
           "input the ink model is correct for. Code that needs the "
           "shape-based model, such as the evaluation of drawn output, "
           "names its checkpoint explicitly rather than trusting this "
           "default."),
   ]),

 "LoadStateDictAny": dict(
   inputs=[("checkpointPath", "path to a saved .npz checkpoint file")],
   output="a pair: a dictionary mapping parameter names to arrays, and the "
          "writer name-to-slot mapping if the file carries one, otherwise "
          "nothing.",
   theory=[
     ("h", "What it is for"),
     ("p", "A trained network is just a collection of named arrays, and "
           "this function reads that collection off disk. Its value is that "
           "it is the only place in the file that knows how a checkpoint is "
           "stored, so every network below receives a plain dictionary and "
           "none of them contains any file handling."),
     ("h", "Background: what a state dictionary is"),
     ("p", "A **state dictionary** is the standard way to store a trained "
           "network: a mapping from the name of each parameter to its "
           "values. A name like `stage1.0.weight` encodes a position in the "
           "network, here the weight array of layer 0 of the block called "
           "stage1. The network's structure is not stored at all; it is "
           "written in code, and loading consists of looking up the names "
           "that code expects. That is why the forward methods below are "
           "full of string keys: those strings are the contract between the "
           "training script that wrote the file and the code that reads "
           "it."),
     ("h", "The .npz format"),
     ("p", "An `.npz` file is a zip archive of individual arrays, each "
           "stored under a name. That suits a state dictionary exactly, and "
           "it needs no library beyond numpy, which is the constraint this "
           "project works under."),
     ("h", "Why pickling is refused"),
     ("p", "The file is opened with object pickling disabled. "
           "**Pickling** is a mechanism for storing arbitrary Python "
           "objects, and restoring one executes code from the file, so a "
           "checkpoint able to contain pickled objects is able to run "
           "anything when loaded. Refusing it restricts the file to plain "
           "numeric arrays, which is all a set of weights needs to be. It "
           "also acts as a check on the files themselves: a checkpoint that "
           "will not load without pickling is storing something other than "
           "arrays, and that is worth discovering rather than tolerating."),
     ("h", "Keys beginning with a double underscore"),
     ("p", "Names starting with a double underscore are skipped when "
           "building the parameter dictionary, because they hold metadata "
           "rather than weights. One is read specially: a list of writer "
           "names, which is converted into a mapping from each name to its "
           "output slot."),
     ("p", "Carrying that mapping inside the checkpoint matters more than "
           "it appears. The namer outputs one score per slot, and a slot "
           "means nothing without knowing which writer it stands for. If "
           "the correspondence were kept separately, in a configuration "
           "file or in code, then a checkpoint trained on a different set "
           "of writers, or on the same writers in a different order, would "
           "load without complaint and attribute every page to the wrong "
           "person. Storing the names with the weights makes that mismatch "
           "impossible."),
   ]),

 "Conv2d": dict(
   inputs=[("x", "the input, shaped (batch, channels, height, width)"),
           ("weight", "the filters, shaped (out channels, in channels, "
                      "filter height, filter width)"),
           ("bias", "one number added per output channel, or nothing"),
           ("stride", "how many pixels the filter advances between "
                      "positions, default 1"),
           ("padding", "how many rows and columns of zeros to add round "
                       "the input before filtering, default 1")],
   output="the filtered result, shaped (batch, out channels, output height, "
          "output width).",
   theory=[
     ("h", "What a convolution is, from scratch"),
     ("p", "A **convolution** takes a small grid of numbers, called a "
           "**filter** or **kernel**, places it over a patch of the image, "
           "multiplies each filter number by the image value beneath it, "
           "and adds the products into one output number. Then it moves "
           "along and does it again, covering every position. The result is "
           "a new image, one value per position, recording how strongly the "
           "pattern in the filter was present there."),
     ("eq", r"y[o,\,p,\,q] \;=\; b[o] \;+\; \sum_{c=0}^{C-1} "
            r"\sum_{u=0}^{k_h-1} \sum_{v=0}^{k_w-1} "
            r"w[o,\,c,\,u,\,v] \cdot x[c,\;p\,s+u,\;q\,s+v]",
      [("y[o,p,q]", "the output value for filter o at row p, column q"),
       ("b[o]", "the bias, one learned number added to every position of "
                "output channel o"),
       ("c", "which input channel, running over all C of them"),
       ("u,\\,v", "the position inside the filter"),
       ("k_h,\\,k_w", "the filter's height and width, 3 by 3 everywhere in "
                      "this project"),
       ("w[o,c,u,v]", "the learned filter weight at that position"),
       ("x[c,\\,y,\\,x]", "the input value in channel c"),
       ("s", "the stride, how far the filter moves between output "
             "positions")],
      None),
     ("p", "Read the triple sum as: for one output position, visit every "
           "cell of the filter in every input channel, multiply by the "
           "input underneath, and total it. The sum over channels is the "
           "part that is easy to overlook and important: a filter is not a "
           "flat grid but a stack as deep as the input has channels, so it "
           "combines evidence from every feature map below it rather than "
           "looking at one."),
     ("h", "Why this is the right operation for handwriting"),
     ("p", "Two properties make convolution suited to images. It is "
           "**local**: each output depends only on a small patch, which "
           "matches the fact that a stroke, a corner or a loop is a local "
           "arrangement of ink. And it is **translation-equivariant**, "
           "meaning that moving the input moves the output by the same "
           "amount without changing it: the same filter detects a vertical "
           "stroke wherever on the line it occurs. A plain fully connected "
           "layer has neither property; it would have to learn what a "
           "vertical stroke looks like separately at every position, "
           "needing enormously more parameters and far more examples."),
     ("h", "What padding is for"),
     ("p", "A 3 by 3 filter cannot be centred on a pixel in the outermost "
           "row, because part of it would hang off the edge, so without "
           "intervention each convolution would shrink the image by one "
           "pixel on every side. **Padding** adds a border of zeros so the "
           "filter can reach the edge, and a padding of 1 with a 3 by 3 "
           "filter keeps the output exactly the same size as the input. "
           "With twelve convolutions stacked in this network, the "
           "unpadded alternative would eat twelve pixels off the height, "
           "which is a fifth of a 64-pixel crop. Zero is the correct pad "
           "value because the input has been arranged so that zero means "
           "no ink; see TensorFromResized."),
     ("eq", r"H_{\mathrm{out}} \;=\; \left\lfloor "
            r"\frac{H + 2\,\mathrm{pad} - k_h}{s} \right\rfloor + 1",
      [("H_{\\mathrm{out}}", "the output height"),
       ("H", "the input height"),
       (r"\mathrm{pad}", "the padding added on each side"),
       ("k_h", "the filter height"),
       ("s", "the stride"),
       (r"\lfloor \cdot \rfloor", "rounding down, because a filter "
                                  "position that does not fit entirely "
                                  "inside the padded input is not used")],
      None),
     ("h", "The im2col trick, and why it is used"),
     ("p", "Written literally the definition is a loop over output "
           "positions inside a loop over filters, which in Python would be "
           "unusably slow. The standard remedy is **im2col**, short for "
           "image to columns \\cite{chellapilla2006}: rearrange the input "
           "so that every patch the filter will see becomes one column of a "
           "large matrix. The convolution then becomes a single matrix "
           "multiplication, because that is exactly what a matrix product "
           "computes, a dot product of every row of one matrix with every "
           "column of the other."),
     ("eq", r"Y \;=\; W_{\mathrm{flat}} \, X_{\mathrm{cols}}, \qquad "
            r"W_{\mathrm{flat}} \in \mathbb{R}^{\,O \times (C k_h k_w)}, "
            r"\qquad X_{\mathrm{cols}} \in "
            r"\mathbb{R}^{\,(C k_h k_w) \times (H_{\mathrm{out}} "
            r"W_{\mathrm{out}})}",
      [("Y", "the output, as a matrix of one row per filter and one column "
             "per output position"),
       (r"W_{\mathrm{flat}}", "the filters flattened so each becomes one "
                              "row of length C times the filter area"),
       (r"X_{\mathrm{cols}}", "the patches, each flattened into one column "
                              "in the same order"),
       ("O", "the number of filters, that is, output channels"),
       (r"\mathbb{R}^{\,m \times n}", "the set of real-valued matrices with "
                                      "m rows and n columns; this notation "
                                      "just states each array's shape")],
      None),
     ("p", "The gain is not from doing less arithmetic, since the number of "
           "multiplications is unchanged. It is that matrix multiplication "
           "is handed to **BLAS**, a long-standing library of linear "
           "algebra routines written to use the processor's vector "
           "instructions and its cache well. A matrix product runs "
           "perhaps two orders of magnitude faster than the equivalent "
           "Python loops, and reformulating the problem to reach it is "
           "worth more than any amount of tuning the loops."),
     ("h", "Building the columns without copying"),
     ("p", "The patches are not assembled by copying. Neighbouring patches "
           "overlap heavily, a 3 by 3 filter at stride 1 sharing six of "
           "nine pixels with its neighbour, so a literal im2col would "
           "duplicate most of the input nine times over. Instead the "
           "columns are produced as a **strided view**, which needs some "
           "explanation."),
     ("p", "A numpy array is a flat block of memory plus a set of "
           "**strides**, one per axis, each saying how many bytes to step "
           "to advance by one along that axis. Indexing is then just "
           "arithmetic on those strides. Supplying a different shape and a "
           "different set of strides therefore produces an array that reads "
           "the same memory in a different pattern, with no data copied. "
           "Here a six-axis view is constructed whose axes are batch, "
           "channel, filter row, filter column, output row and output "
           "column, so that the patches appear laid out ready to be "
           "reshaped into columns. The strides for the two output axes are "
           "the input's own row and column strides multiplied by the "
           "stride, which is what makes stepping to the next output "
           "position step the right distance through the image."),
     ("note", "A strided view of overlapping patches reports a shape far "
              "larger than the memory behind it, because the same bytes "
              "appear at many positions. That is safe to read and would be "
              "meaningless to write to. The reshape that follows does force "
              "one copy, since the pattern cannot be expressed as strides "
              "alone, but it is one copy instead of nine."),
     ("h", "The bias"),
     ("p", "One learned number per output channel is added to every "
           "position of that channel. It lets a filter shift its whole "
           "response up or down, which matters because the squashing that "
           "follows treats zero specially: without a bias, a filter could "
           "only ever decide between zero and its raw response, whereas "
           "with one it can set the level at which it starts responding at "
           "all. Adding one number to a whole channel is written as a "
           "reshape to four axes and a plain addition, relying on "
           "**broadcasting**, numpy's rule that an axis of length 1 is "
           "repeated as needed to match the other operand."),
   ]),

 "BatchNorm2d": dict(
   inputs=[("x", "the input, shaped (batch, channels, height, width)"),
           ("weight", "the learned scale, one per channel"),
           ("bias", "the learned shift, one per channel"),
           ("runningMean", "the mean of each channel recorded during "
                           "training"),
           ("runningVar", "the variance of each channel recorded during "
                          "training"),
           ("eps", "a small constant guarding the division, default 1e-5")],
   output="the input with each channel standardised and then rescaled, same "
          "shape as the input.",
   theory=[
     ("h", "The problem it solves"),
     ("p", "Stack a dozen layers and the scale of the numbers flowing "
           "through them drifts. Each layer multiplies by its weights, so "
           "if those weights are on average slightly larger than the scale "
           "that preserves magnitude, the values grow layer by layer; "
           "slightly smaller, and they shrink toward nothing. Worse, the "
           "scale a layer sees changes during training, because the layers "
           "beneath it are themselves being changed. A layer is therefore "
           "chasing a moving target, which forces a small learning rate and "
           "slow training. **Batch normalisation** \\cite{ioffe2015} fixes "
           "the scale explicitly: it standardises each channel so that it "
           "arrives at the next layer with a known mean and spread."),
     ("h", "Background: mean, variance and standardising"),
     ("p", "The **mean** is the average value. The **variance** is the "
           "average of the squared distances from the mean, a measure of "
           "how spread out the values are, and its square root is the "
           "**standard deviation**, which is in the same units as the "
           "values themselves. To **standardise** is to subtract the mean "
           "and divide by the standard deviation, which leaves a quantity "
           "with mean 0 and standard deviation 1 regardless of what scale "
           "it started on."),
     ("eq", r"\hat{x}[c] \;=\; \frac{x[c] - \mu[c]}"
            r"{\sqrt{\sigma^{2}[c] + \epsilon}}",
      [(r"\hat{x}[c]", "the standardised value in channel c"),
       ("x[c]", "the incoming value"),
       (r"\mu[c]", "the mean of channel c, recorded during training"),
       (r"\sigma^{2}[c]", "the variance of channel c, also recorded during "
                          "training"),
       (r"\epsilon", "a small constant, 0.00001, added so that a channel "
                     "with no variation cannot cause a division by zero")],
      None),
     ("h", "Why the scale and shift are put back"),
     ("p", "Forcing every channel to mean 0 and spread 1 would be too "
           "strong a constraint, because for some features the useful "
           "information is precisely that they are usually large or usually "
           "near zero. So two learned numbers per channel are applied "
           "afterwards, a scale and a shift, letting the network restore "
           "any mean and spread it finds useful, including the original "
           "one. The point is not that the statistics are fixed at 0 and 1; "
           "it is that they are now set by two parameters the network "
           "controls directly, instead of emerging from the accumulated "
           "behaviour of every layer below."),
     ("eq", r"y[c] \;=\; \gamma[c] \, \hat{x}[c] \;+\; \beta[c]",
      [("y[c]", "the output value in channel c"),
       (r"\gamma[c]", "the learned scale for that channel, held in the "
                      "argument called weight"),
       (r"\beta[c]", "the learned shift, held in the argument called bias"),
       (r"\hat{x}[c]", "the standardised value from above")],
      None),
     ("h", "Training-time and inference-time behaviour differ"),
     ("p", "The method takes its name from how the statistics are obtained "
           "during training: from the current **batch**, the group of "
           "examples being processed together. That cannot be done when "
           "reading one crop. A single example's own mean and variance are "
           "not a useful description of anything, and worse, the answer for "
           "one image would depend on which other images happened to "
           "accompany it, which is unacceptable at inference time."),
     ("p", "The resolution is that training also keeps a running average of "
           "the statistics across all batches, which estimates the mean and "
           "variance over the whole dataset. Those stored values are what "
           "this function uses. So the layer behaves differently in "
           "training and in use, and that difference is intended: in "
           "training it standardises by the batch, in use by fixed numbers "
           "read from the checkpoint. This is why `runningMean` and "
           "`runningVar` are arguments here while a training "
           "implementation computes them."),
     ("h", "Folding four operations into two"),
     ("p", "Written directly the layer is a subtraction, a division, a "
           "multiplication and an addition at every one of the hundreds of "
           "thousands of values in the array. But at inference time all "
           "four constants are fixed, so the whole thing is a linear "
           "function and can be rearranged into one multiplication and one "
           "addition."),
     ("eq", r"y \;=\; \underbrace{\frac{\gamma}{\sqrt{\sigma^{2}+"
            r"\epsilon}}}_{\textstyle a} \; x \;+\; \underbrace{\left("
            r"\beta - \frac{\gamma\,\mu}{\sqrt{\sigma^{2}+\epsilon}}"
            r"\right)}_{\textstyle b}",
      [("a", "the combined scale, computed once per channel"),
       ("b", "the combined shift, computed once per channel"),
       (r"\gamma,\,\beta", "the learned scale and shift"),
       (r"\mu,\,\sigma^{2}", "the stored mean and variance"),
       (r"\epsilon", "the guard constant")],
      None),
     ("p", "The two combined constants are computed once per channel, over "
           "an array of a few hundred numbers, and then applied to the full "
           "array in one multiply and one add. The square root, in "
           "particular, happens once per channel rather than once per "
           "pixel. This rearrangement is exact, not an approximation, and "
           "it is the whole reason the implementation is two lines."),
     ("h", "Why it is per channel and not per pixel"),
     ("p", "There is one mean and one variance for each channel, shared by "
           "every pixel in it. That follows from what a channel is: the map "
           "of where one particular filter responded. A filter's typical "
           "response size is a property of that filter, the same wherever "
           "on the image it fires, so one statistic describes the whole "
           "map. Normalising per pixel would instead remove the differences "
           "between positions, which is the information the map exists to "
           "carry."),
   ]),

 "Relu": dict(
   inputs=[("x", "any array")],
   output="the array with every negative value replaced by zero.",
   theory=[
     ("h", "Why a network needs this at all"),
     ("p", "Convolution and the linear layer are both **linear** "
           "operations, meaning the output is a weighted sum of the inputs. "
           "Composing linear operations gives another linear operation: two "
           "matrix multiplications are equivalent to one by the product of "
           "the matrices. So a stack of a hundred linear layers computes "
           "exactly what a single one could, and depth buys nothing. "
           "Something non-linear must sit between them, and that is what an "
           "**activation function** is for."),
     ("eq", r"\mathrm{ReLU}(x) \;=\; \max(x,\,0) \;=\; "
            r"\begin{cases} x & x > 0 \\ 0 & x \le 0 \end{cases}",
      [(r"\mathrm{ReLU}(x)", "the output, named for rectified linear unit"),
       ("x", "the input value"),
       (r"\max(x,0)", "the larger of x and zero"),
       (r"\begin{cases}\end{cases}",
        "case notation: the two lines give the value in each of the two "
        "situations listed on the right")],
      None),
     ("h", "Why this particular function"),
     ("p", "ReLU is nearly the simplest non-linear function there is, and "
           "that is much of its appeal, but it has three concrete "
           "advantages over the smooth S-shaped functions that preceded "
           "it. Its derivative is 1 for every positive input, so a gradient "
           "passing back through many layers is not repeatedly multiplied "
           "by a number smaller than one, which is what made deep networks "
           "with sigmoid activations so hard to train. It produces exact "
           "zeros, so a layer's output is **sparse**, with many units "
           "silent on any given input, which tends to give cleaner and more "
           "separable features. And it costs one comparison, against an "
           "exponential for the alternatives."),
     ("h", "What it means in this network"),
     ("p", "After a convolution, a positive value means the filter's "
           "pattern was found and a negative one means something like its "
           "opposite was. ReLU discards the negatives entirely, which is to "
           "say the layer reports where each feature *was* present and "
           "stays silent about where it was not. The next layer therefore "
           "builds on evidence of presence rather than on a signed score, "
           "which is the behaviour that makes stacked filters compose into "
           "detectors for larger shapes: strokes into corners, corners into "
           "loops, loops into letters."),
     ("note", "A unit whose input is always negative outputs zero always, "
              "and since the derivative of ReLU is zero there, nothing ever "
              "pushes it back. This is known as a dying unit, and it is "
              "ReLU's one real drawback. It is a training-time concern, "
              "handled in train.py by careful initialisation, and has no "
              "effect on the forward pass here."),
   ]),

 "Sigmoid": dict(
   inputs=[("x", "any array")],
   output="the array squashed into the range 0 to 1.",
   theory=[
     ("h", "What it is for"),
     ("p", "Where ReLU is used to introduce non-linearity, the sigmoid is "
           "used when a number has to be interpreted as a **proportion**: "
           "how much of something to let through, on a scale from none to "
           "all. In this file its only use is in the LSTM, whose gates each "
           "need exactly that."),
     ("eq", r"\sigma(x) \;=\; \frac{1}{1 + e^{-x}}",
      [(r"\sigma(x)", "the sigmoid of x, a value strictly between 0 and 1"),
       ("x", "the input, any real number"),
       ("e", "the base of the natural logarithm, about 2.71828"),
       ("e^{-x}", "the exponential of minus x: large when x is very "
                  "negative, near zero when x is large and positive")],
      None),
     ("h", "Why the shape is right for a gate"),
     ("p", "Follow the formula at three points. For a large positive x the "
           "exponential is near zero, so the result approaches 1, fully "
           "open. For a large negative x the exponential is huge, so the "
           "result approaches 0, fully closed. At zero the exponential is "
           "exactly 1 and the result is exactly one half. The curve is "
           "smooth and increases everywhere, so the gate can take any "
           "intermediate position, and because it is smooth it has a "
           "derivative everywhere, which training requires. The S shape "
           "also means it saturates: once a gate is firmly open or firmly "
           "closed, moderate changes in the input barely move it, which is "
           "the stability a memory mechanism needs."),
     ("note", "For a very large negative input the exponential overflows "
              "and numpy issues a warning while returning infinity, whose "
              "reciprocal is correctly zero. The answer is right; only the "
              "warning is spurious. The implementation is left in its "
              "direct form because the alternative formulation adds a "
              "branch to every element of every gate at every time step to "
              "suppress a warning that misrepresents nothing."),
   ]),

 "MaxPool2d": dict(
   inputs=[("x", "the input, shaped (batch, channels, height, width)"),
           ("kernel", "the window size, default 2"),
           ("stride", "how far the window moves, default 2")],
   output="the input reduced by the stride in both directions, each output "
          "value being the largest in its window.",
   theory=[
     ("h", "What it is for"),
     ("p", "**Pooling** shrinks the spatial size of a feature map by "
           "summarising each small window as a single number. With a 2 by 2 "
           "window and a stride of 2, the windows do not overlap and the "
           "map comes out half as tall and half as wide, a quarter of the "
           "values."),
     ("eq", r"y[c,\,p,\,q] \;=\; \max_{0 \le u,v < k} \; "
            r"x[c,\; p\,s+u,\; q\,s+v]",
      [("y[c,p,q]", "the pooled value in channel c at row p, column q"),
       ("k", "the window size"),
       ("s", "the stride"),
       ("u,\\,v", "the position within the window"),
       (r"\max", "the largest of the values in the window")],
      None),
     ("h", "Three reasons it is done"),
     ("items", [
       "**Cost.** A quarter of the values means a quarter of the work in "
       "every layer that follows. Since this network pools twice, the "
       "deepest and widest convolutional block runs on a sixteenth of the "
       "original area.",
       "**Receptive field.** The patch of the original image that "
       "influences one value, called its **receptive field**, grows slowly "
       "if only 3 by 3 convolutions are stacked, by two pixels per layer. "
       "Halving the resolution doubles the area each later filter covers in "
       "original pixels, which is how the network gets from detecting "
       "strokes to detecting whole letters without needing very many "
       "layers.",
       "**Small-shift tolerance.** Taking the largest value in a window "
       "gives the same answer wherever in that window the feature "
       "occurred, so the output is unchanged by a shift of a pixel. For "
       "handwriting this is exactly right: the same letter written a pixel "
       "higher is the same letter.",
     ]),
     ("h", "Why the maximum rather than the average"),
     ("p", "After ReLU, a value is evidence that a feature is present, and "
           "zero is silence. The maximum answers \"was this feature found "
           "anywhere in this window\", which is the useful question about "
           "evidence. An average would dilute a strong local response by "
           "the surrounding silence, so a thin stroke correctly detected in "
           "one position would be reported as a weak response over a wider "
           "area. Averaging has its place, and this file uses it later "
           "where the question genuinely is about a typical level rather "
           "than a presence."),
     ("h", "Implementation, and the discarded remainder"),
     ("p", "The windows are formed as a strided view, by the same mechanism "
           "described under Conv2d, with two extra axes for position within "
           "the window; taking the maximum over those two axes then pools "
           "everything at once. An odd final row or column, which cannot "
           "fill a window, is dropped before the view is built. Discarding "
           "input is acceptable here because the loss is a single row or "
           "column at the edge of a padded crop, where there is no ink, "
           "while the alternative of padding would introduce a window "
           "containing invented values whose maximum could be reported as a "
           "detected feature."),
   ]),

 "AdaptiveMaxPoolHeight": dict(
   inputs=[("x", "the input, shaped (batch, channels, height, width)")],
   output="the same array with its height collapsed to 1, keeping the "
          "largest value in each column of each channel.",
   theory=[
     ("h", "What it is for, and why it is the pivot of the whole network"),
     ("p", "This one line is where the reader stops being an image network "
           "and becomes a sequence network. Before it, the data is a stack "
           "of two-dimensional feature maps. After it, each column of the "
           "image is a single vector of features, and the line is a "
           "sequence of such vectors, left to right. That is the form the "
           "LSTM and CTC both require, and this is the step that produces "
           "it."),
     ("eq", r"y[c,\,0,\,q] \;=\; \max_{0 \le p < H} \; x[c,\,p,\,q]",
      [("y[c,0,q]", "the collapsed value for channel c at column q, the "
                    "height axis now being of length 1"),
       ("x[c,p,q]", "the feature value at row p of that column"),
       ("H", "the feature map height at this point in the network, which "
             "is 16 after two poolings of a 64-pixel crop"),
       (r"\max_{0 \le p < H}", "the largest value down the whole column")],
      None),
     ("h", "Why collapsing the height is legitimate"),
     ("p", "It looks like throwing away a great deal, and it would be for a "
           "general image. For a line of handwriting it is justified by the "
           "geometry of the problem. The crop contains exactly one line of "
           "text, so vertical position within it carries no information "
           "about *which* character is present, only about which part of a "
           "character a pixel belongs to. What identifies the character at "
           "a given horizontal position is the set of features found "
           "anywhere in that column. Taking the maximum down each column "
           "asks precisely that: was this feature present anywhere at this "
           "horizontal position."),
     ("p", "The word **adaptive** means the output size is specified rather "
           "than the window: here the output height is 1 whatever the input "
           "height, so the window is as tall as the input. That is why the "
           "function needs no parameters, and why it would behave "
           "identically on a crop of a different height, unlike the fixed "
           "pooling above."),
     ("note", "A maximum, not an average, for the same reason as before: "
              "after ReLU these values are evidence of presence, and a "
              "feature found in two rows out of sixteen should be reported "
              "as found, not diluted by the fourteen silent rows. This is "
              "the choice that lets an ascender and the body of a letter in "
              "the same column both register."),
   ]),

 "AdaptiveAvgPoolWidth": dict(
   inputs=[("x", "the input, shaped (batch, channels, height, width)")],
   output="the array averaged down axis 2, with that axis kept at length 1.",
   theory=[
     ("h", "What it is for"),
     ("p", "The namer, unlike the reader, has to produce one answer for the "
           "whole crop rather than one per column. The identity of a writer "
           "is a property of the line as a whole, not of a position along "
           "it. So at the end of the namer's backbone the remaining axis is "
           "collapsed by averaging, turning a sequence of feature vectors "
           "into a single feature vector describing the whole line."),
     ("eq", r"y[c] \;=\; \frac{1}{N} \sum_{q=0}^{N-1} x[c,\,q]",
      [("y[c]", "the pooled value for channel c"),
       ("x[c,q]", "the feature value at position q"),
       ("N", "the number of positions being averaged over")],
      None),
     ("h", "Why averaging is right here and maximum was right before"),
     ("p", "The question has changed, and so the summary must. The reader "
           "asks whether a feature occurs at a given position, which is a "
           "question about presence, answered by a maximum. The namer asks "
           "how characteristic a feature is of this writing, which is a "
           "question about prevalence, answered by an average. A writer who "
           "makes a particular join in most of their letters should score "
           "differently from one who makes it once; a maximum would report "
           "both identically, since both have it somewhere. Averaging is "
           "what makes the feature vector describe a habit rather than an "
           "occurrence."),
     ("p", "This construction is known as **global average pooling**, and "
           "beyond matching the question it has a practical benefit: the "
           "output size depends only on the number of channels, not on the "
           "length of the line. The classifier head that follows therefore "
           "has a fixed input size regardless of how long the crop was."),
     ("note", "The function name says width and the code averages axis 2, "
              "which is the height axis in the four-axis convention. There "
              "is no inconsistency in effect, because by the time it is "
              "called the height has already been collapsed and the array "
              "has been reduced to three axes, so axis 2 is the sequence "
              "position. The name describes the intent and the index "
              "describes the array as it actually arrives."),
   ]),

 "Linear": dict(
   inputs=[("x", "the input, whose last axis is the feature vector"),
           ("weight", "the weight matrix, shaped (outputs, inputs)"),
           ("bias", "one number added per output, or nothing")],
   output="the input multiplied by the transposed weight matrix, with the "
          "bias added; the last axis becomes the number of outputs.",
   theory=[
     ("h", "What it is"),
     ("p", "The **fully connected** or **linear** layer, the most basic "
           "layer there is: every output is a weighted sum of every input, "
           "plus a constant. Where a convolution looks at a local patch and "
           "reuses one filter everywhere, a linear layer looks at the whole "
           "input vector and has an independent weight for each "
           "input-output pair."),
     ("eq", r"y[j] \;=\; b[j] \;+\; \sum_{i=0}^{n-1} W[j,\,i] \, x[i]",
      [("y[j]", "the j-th output"),
       ("b[j]", "the bias for that output"),
       ("W[j,i]", "the weight connecting input i to output j"),
       ("x[i]", "the i-th input"),
       ("n", "the number of inputs")],
      None),
     ("h", "Why the matrix is transposed"),
     ("p", "The weight array is stored with one row per output, which is "
           "the convention the training framework used, while the input has "
           "its features along its last axis. Multiplying the input by the "
           "transpose pairs each input vector with each row of weights, "
           "giving one output per row. Transposing is free in numpy, being "
           "only a reordering of the strides, so this costs nothing and "
           "keeps the stored weights in the layout the checkpoint was "
           "written with."),
     ("h", "Where it is used in each network"),
     ("p", "In the reader it is the final step, turning the LSTM's output "
           "at each column into one score per character. The weighted sum "
           "is the right operation there because any feature of the column "
           "may bear on any character. In the namer it appears twice, as a "
           "hidden layer with a ReLU between and then a final layer "
           "producing one score per writer, which is enough to combine the "
           "pooled features non-linearly before the decision."),
     ("note", "The multiplication applies to the last axis and leaves every "
              "earlier axis alone, so the same call works on a single "
              "vector, on a batch of them, or on the reader's three-axis "
              "sequence of columns. That is broadcasting again, and it is "
              "why no reshaping surrounds either use."),
   ]),

 "LogSoftmax": dict(
   inputs=[("x", "an array of raw scores"),
           ("axis", "which axis the scores to be compared lie along")],
   output="the log of the softmax along that axis: values at most zero, "
          "whose exponentials sum to one.",
   theory=[
     ("h", "Background: turning scores into probabilities"),
     ("p", "A network's final layer produces one unbounded number per "
           "class, called a **logit**. These are not probabilities: they "
           "can be negative, and they need not sum to anything in "
           "particular. The **softmax** converts them into probabilities by "
           "exponentiating each, which makes them positive, and dividing by "
           "the total, which makes them sum to one."),
     ("eq", r"p_{k} \;=\; \frac{e^{x_{k}}}{\sum_{j} e^{x_{j}}}",
      [("p_k", "the probability assigned to class k"),
       ("x_k", "the raw score, or logit, for class k"),
       (r"\sum_{j} e^{x_{j}}", "the sum of the exponentials over every "
                               "class, which makes the outputs total 1")],
      None),
     ("p", "Exponentiating is what makes the conversion monotone and "
           "amplifying: a score two units higher becomes a probability "
           "about 7.4 times larger, so softmax preserves the ranking while "
           "sharpening the differences."),
     ("h", "Why the logarithm is taken"),
     ("p", "CTC multiplies many probabilities together, one per column of "
           "a line, and a line has well over a hundred columns. A product "
           "of a hundred numbers each around 0.1 is around ten to the power "
           "minus one hundred, far below the smallest number a double can "
           "represent, so it becomes exactly zero and all information is "
           "lost. This is **underflow**. Working with logarithms removes it "
           "entirely, because the logarithm of a product is the sum of the "
           "logarithms, and a sum of a hundred moderate negative numbers is "
           "an ordinary number."),
     ("eq", r"\log\!\Big(\prod_{t} p_{t}\Big) \;=\; \sum_{t} \log p_{t}",
      [(r"\prod_{t} p_{t}", "the product of the per-column probabilities "
                            "along a path"),
       (r"\sum_{t} \log p_{t}", "the equivalent sum of their logarithms"),
       ("p_t", "the probability at column t")],
      None),
     ("h", "The log-sum-exp trick"),
     ("p", "Computing the logarithm of the softmax naively would "
           "exponentiate first, and a logit of 800 overflows to infinity "
           "before any logarithm is taken. The remedy uses the fact that "
           "softmax is unchanged by subtracting a constant from every "
           "score, since the constant's exponential cancels between "
           "numerator and denominator. Subtracting the largest score makes "
           "the biggest exponential exactly 1 and every other smaller, so "
           "nothing can overflow."),
     ("eq", r"\log p_{k} \;=\; (x_{k} - m) \;-\; \log \sum_{j} "
            r"e^{\,x_{j} - m}, \qquad m \;=\; \max_{j} x_{j}",
      [(r"\log p_k", "the log probability of class k, the function's "
                     "output"),
       ("x_k", "the raw score for class k"),
       ("m", "the largest score, subtracted from every score before any "
             "exponential is taken"),
       (r"\log \sum_j e^{x_j - m}", "the log-sum-exp term, which is the "
                                    "normalisation expressed in log space")],
      None),
     ("p", "Underflow in the other direction is also harmless here. A score "
           "far below the maximum exponentiates to zero, contributing "
           "nothing to the sum, which is the correct contribution for a "
           "class of negligible probability. The maximum term is always "
           "present and equals 1, so the sum can never be zero and its "
           "logarithm is always defined. This rearrangement is known as the "
           "**log-sum-exp trick**, and it is standard wherever "
           "probabilities are handled in log space."),
     ("note", "Every output of this function is at most zero, because a "
              "probability is at most one and the logarithm of a number at "
              "most one is at most zero. Seeing negative numbers in the "
              "reader's output is therefore expected, not a fault, and a "
              "value near zero means near-certainty."),
   ]),

 "Softmax": dict(
   inputs=[("x", "an array of raw scores"),
           ("axis", "which axis the scores to be compared lie along")],
   output="probabilities along that axis: non-negative and summing to one.",
   theory=[
     ("h", "What it is for"),
     ("p", "The same conversion as above without the logarithm, used where "
           "an actual probability is wanted for a person to read rather "
           "than a log probability for further arithmetic. In this file its "
           "only use is in the namer, where the output is a confidence per "
           "writer that gets reported."),
     ("h", "The same stabilisation, for the same reason"),
     ("p", "The maximum is subtracted before exponentiating here too. The "
           "result is mathematically identical, since the subtraction "
           "cancels, but it cannot overflow: the largest exponential is 1 "
           "rather than potentially enormous. The cost is one pass to find "
           "the maximum and one subtraction, which is negligible against "
           "the exponentials themselves."),
     ("h", "Why the namer uses this and the reader uses the log version"),
     ("p", "The distinction follows from what happens next. The namer's "
           "output is compared and reported directly, so a probability is "
           "what is wanted. The reader's output is summed along thousands "
           "of paths by CTC, so log probabilities are what is wanted. "
           "Converting between them afterwards would be possible but would "
           "reintroduce exactly the underflow that working in log space "
           "exists to avoid."),
     ("note", "A confidence from this function is a probability over the "
              "enrolled writers only, and it necessarily sums to one across "
              "them. It therefore cannot express \"none of these\": a crop "
              "from somebody not enrolled is still assigned to whichever "
              "enrolled writer resembles it most, with a confidence that "
              "may be high. This is a property of the output layer, not a "
              "fault in the training, and any use of the namer on an "
              "unknown writer has to account for it."),
   ]),

 "ConvBlockForward": dict(
   inputs=[("x", "the input feature maps"),
           ("sd", "the loaded state dictionary of parameters"),
           ("prefix", "the name prefix identifying this block in the "
                      "checkpoint, such as stage1"),
           ("nLayers", "how many convolution, normalise, activate groups "
                       "this block contains")],
   output="the feature maps after that many groups have been applied.",
   theory=[
     ("h", "What it is for"),
     ("p", "Both networks are built from the same repeated unit: a "
           "convolution, then batch normalisation, then ReLU. This function "
           "applies that unit a given number of times, so the unit is "
           "written once rather than twelve times in each network."),
     ("h", "Why that particular order"),
     ("p", "The order is not arbitrary and each adjacency has a reason. "
           "Normalisation comes after the convolution because its purpose "
           "is to fix the scale of the convolution's output, which it "
           "cannot do before that output exists. ReLU comes after "
           "normalisation because the normalisation needs to see the signed "
           "values to standardise them meaningfully; applied to an already "
           "rectified signal, with its mass piled at zero, the mean and "
           "variance would describe mostly how many units were silent. And "
           "ReLU must come before the next convolution, since the whole "
           "point of an activation is to break the chain of linear "
           "operations."),
     ("p", "A further consequence of this order is that the convolution's "
           "bias is redundant: the normalisation immediately subtracts a "
           "mean, which removes any constant the bias added. It is kept "
           "because the checkpoint stores it and the trained values must be "
           "applied as they were learned, but it is worth knowing that it "
           "has no effect on the result."),
     ("h", "How a layer is located in the checkpoint"),
     ("p", "Within a block the layers are numbered in threes, so group i "
           "occupies positions 3i for the convolution and 3i+1 for the "
           "normalisation, the activation having no parameters and so no "
           "position of its own. The names are built by joining the prefix "
           "and those numbers, which is how a checkpoint saved by the "
           "training framework names its layers. The pattern is "
           "effectively a contract, and the next entry describes what "
           "happens when it is relied on across different files."),
     ("h", "The three stages, and why they get deeper"),
     ("items", [
       "**stage1, two groups,** on the full-resolution crop, where a filter "
       "sees a 3 by 3 patch of actual pixels and so can only detect the "
       "smallest features: an edge, a stroke, the direction of a line.",
       "**stage2, four groups,** after one pooling, where features combine "
       "into corners, junctions and curves.",
       "**stage3, six groups,** after a second pooling, where a filter's "
       "receptive field covers a large part of a letter and the features "
       "become letter-like.",
     ]),
     ("p", "The number of groups increases with depth because the number of "
           "distinguishable patterns does. There are few ways to arrange "
           "ink in a 3 by 3 patch and a great many ways to arrange strokes "
           "into a letter, so the later stages need more capacity. Placing "
           "the deepest stage after two poolings is also what makes this "
           "affordable: it runs on a sixteenth of the original area, so its "
           "six groups cost less than stage1's two."),
   ]),
}


#the rest of the chapter, in order
for _part in ("codetheory_recognise2",):
    try:
        _m = __import__(_part)
    except ModuleNotFoundError:
        continue
    FUNCS.update(_m.FUNCS)
