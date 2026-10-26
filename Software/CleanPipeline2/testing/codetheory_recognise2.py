"""recognise.py, part 2: memory, CTC, and the two networks.

The recurrent half of the reader, the decoding and alignment that turn its
output back into words, the accuracy measures, and the two network classes.
"""

FUNCS = {

 "LstmForward": dict(
   inputs=[("x", "the input sequence, shaped (time, batch, features)"),
           ("weightIh", "the weights applied to the input, holding all four "
                        "gates stacked"),
           ("weightHh", "the weights applied to the previous output, also "
                        "all four gates stacked"),
           ("biasIh", "the bias on the input side"),
           ("biasHh", "the bias on the recurrent side"),
           ("hiddenSize", "the width of the memory and output vectors")],
   output="a pair: the output at every time step, shaped (time, batch, "
          "hidden), and the final hidden and cell state.",
   theory=[
     ("h", "The problem: a column is not enough"),
     ("p", "After the convolutional backbone the line has become a sequence "
           "of feature vectors, one per column. Classifying each column on "
           "its own would be hopeless. A single narrow column of a "
           "handwritten line is often ambiguous even to a person: the left "
           "half of an a and the left half of a d are the same stroke, and "
           "which it is depends on what follows. A smudge is identifiable "
           "only from the letters around it. What is needed is a layer "
           "whose output at each column depends on the whole line, and "
           "that is what a **recurrent** layer provides."),
     ("h", "Background: what recurrence means"),
     ("p", "A **recurrent neural network** processes a sequence one step at "
           "a time, carrying a vector of numbers from each step to the "
           "next. That carried vector is the **hidden state**, and it is "
           "the network's summary of everything seen so far. At each step "
           "the layer combines the new input with the incoming hidden state "
           "to produce a new hidden state, which is also its output for "
           "that step. The same weights are used at every step, so the "
           "layer has a fixed number of parameters however long the "
           "sequence is."),
     ("h", "Why a plain recurrent layer is not used"),
     ("p", "In a plain recurrent layer the hidden state is overwritten at "
           "every step, as a weighted combination of the old state and the "
           "new input passed through a squashing function. Information "
           "therefore survives only by being repeatedly multiplied by the "
           "recurrent weights. Over a hundred and sixty steps that product "
           "either shrinks to nothing or grows without bound, depending on "
           "whether the weights are on average below or above the scale "
           "that preserves magnitude. These are the **vanishing** and "
           "**exploding gradient** problems, and the practical consequence "
           "is that a plain recurrent layer cannot learn to connect events "
           "more than a few steps apart. For reading a line of writing that "
           "is useless, since the evidence resolving an ambiguous letter can "
           "be a whole word away."),
     ("h", "The LSTM idea: a cell you add to rather than overwrite"),
     ("p", "The **long short-term memory** layer \\cite{hochreiter1997} "
           "solves this by keeping a second vector, the **cell state**, "
           "whose default behaviour is to persist unchanged. It is not "
           "recomputed each step; it is modified, by having some of its "
           "contents erased and some new contents added. Because the "
           "default is to carry forward, information can survive "
           "arbitrarily many steps without being multiplied by anything, "
           "which is precisely what the plain layer could not do."),
     ("p", "What to erase, what to add and what to reveal are decided by "
           "three **gates**, each a vector of numbers between 0 and 1 "
           "produced by a sigmoid. A gate value of 0 blocks completely, 1 "
           "lets everything through, and anything between is a proportion. "
           "A fourth quantity, the candidate, supplies the contents that "
           "might be added."),
     ("eq", r"\begin{aligned}"
            r"i_t &= \sigma\big(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}\big) \\[2pt]"
            r"f_t &= \sigma\big(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}\big) \\[2pt]"
            r"g_t &= \tanh\big(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}\big) \\[2pt]"
            r"o_t &= \sigma\big(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}\big)"
            r"\end{aligned}",
      [("i_t", "the **input gate** at step t: how much of the candidate to "
               "write into the cell"),
       ("f_t", "the **forget gate**: how much of the existing cell to keep"),
       ("g_t", "the **candidate**, the new contents that might be written; "
               "it uses tanh rather than a sigmoid because it is a value to "
               "be stored, which may usefully be negative, not a proportion"),
       ("o_t", "the **output gate**: how much of the cell to reveal as this "
               "step's output"),
       ("x_t", "the input vector at step t"),
       ("h_{t-1}", "the hidden state carried in from the previous step"),
       ("W_{i\\ast},\\, W_{h\\ast}", "the learned weights applied to the "
                                     "input and to the previous hidden "
                                     "state respectively"),
       ("b_{i\\ast},\\, b_{h\\ast}", "the corresponding learned biases"),
       (r"\sigma", "the sigmoid, squashing to between 0 and 1; see its own "
                   "entry"),
       (r"\tanh", "the hyperbolic tangent, squashing to between −1 and 1")],
      "hochreiter1997"),
     ("p", "Every one of the four lines has the same form: a weighted sum "
           "of the new input, plus a weighted sum of the previous output, "
           "plus biases. They differ only in their weights and in the "
           "squashing applied. That uniformity is what lets the "
           "implementation compute all four in a single matrix "
           "multiplication, described below."),
     ("h", "The two state updates"),
     ("eq", r"c_t \;=\; f_t \odot c_{t-1} \;+\; i_t \odot g_t, "
            r"\qquad h_t \;=\; o_t \odot \tanh(c_t)",
      [("c_t", "the cell state after step t: the long-term memory"),
       ("c_{t-1}", "the cell state carried in from the previous step"),
       ("h_t", "the hidden state after step t, which is also this step's "
               "output"),
       (r"\odot", "elementwise multiplication: each position of one vector "
                  "multiplied by the matching position of the other, as "
                  "opposed to a matrix product"),
       ("f_t,\\, i_t,\\, g_t,\\, o_t", "the gates and candidate from "
                                       "above")],
      "hochreiter1997"),
     ("p", "The cell update is the heart of the method and is worth reading "
           "slowly. It is an addition, not a replacement. If the forget "
           "gate is 1 and the input gate is 0 at some position, that "
           "position of the cell passes through completely untouched, for "
           "as many steps as those gate values persist. No weight matrix is "
           "applied to it along the way, so it neither shrinks nor grows. "
           "That is the path along which information travels a long "
           "distance, and it is what the name long short-term memory "
           "refers to: a short-term memory that can be made to last."),
     ("p", "The hidden state is then a filtered view of the cell. The tanh "
           "keeps it in a bounded range, and the output gate decides how "
           "much of each position to expose. So the layer can hold "
           "something in the cell for many steps while revealing nothing "
           "about it, and then reveal it when it becomes relevant. For "
           "reading handwriting that is directly useful: the shape of an "
           "ambiguous stroke can be held until the letter that disambiguates "
           "it arrives."),
     ("h", "How all four gates are computed at once"),
     ("p", "The four weight matrices are stored stacked into one array, "
           "four times the hidden size tall. One matrix multiplication "
           "therefore produces all four pre-activation vectors end to end, "
           "and the implementation simply slices the result into four "
           "equal blocks in the order input, forget, candidate, output. "
           "That ordering is not a choice made here; it is the convention "
           "of the framework the weights were trained with, and slicing in "
           "a different order would silently use the forget gate's weights "
           "as the input gate's. One large matrix multiplication is also "
           "considerably faster than four small ones, for the BLAS reasons "
           "given under Conv2d."),
     ("h", "Why there are two biases"),
     ("p", "Mathematically a single bias would do, since adding two "
           "constants is adding one constant. Both are kept because the "
           "framework that trained the weights stores both, and the trained "
           "values must be applied as they were learned. Their sum is what "
           "has any effect."),
     ("h", "Why this loop cannot be vectorised"),
     ("p", "Every other layer in this file processes its whole input at "
           "once. This one cannot, and the reason is structural rather than "
           "a shortcoming of the implementation: step t needs the hidden "
           "and cell state from step t−1, so the steps are genuinely "
           "sequential. Within a step everything is already whole-array "
           "arithmetic across the batch and the hidden positions. This is "
           "the same situation as the column loop in LeastCostSeam, and the "
           "same conclusion holds: the loop is over the one axis that is "
           "irreducibly ordered."),
     ("note", "Both states start as vectors of zeros. Zero is the natural "
              "choice for \"nothing seen yet\", and it has the useful "
              "property that an all-zero cell with a forget gate near 1 "
              "stays empty until something is deliberately written into "
              "it."),
   ]),

 "BidirectionalLstmForward": dict(
   inputs=[("x", "the input sequence, shaped (time, batch, features)"),
           ("paramsFwd", "the weights and biases for the forward pass"),
           ("paramsBwd", "the weights and biases for the backward pass"),
           ("hiddenSize", "the width of each direction's output")],
   output="the two directions' outputs joined along the feature axis, so "
          "twice as wide as one direction.",
   theory=[
     ("h", "Why one direction is not enough"),
     ("p", "An LSTM reading left to right has, at each column, seen "
           "everything to the left and nothing to the right. For "
           "handwriting that is half the available evidence and often the "
           "less useful half. Consider the stroke that begins both an a and "
           "a d: what distinguishes them is the ascender that comes "
           "afterwards. A left-to-right reader at the start of that letter "
           "cannot know. It will often get the answer right in the end, "
           "because the output at the later column can be correct, but its "
           "representation of the earlier columns is impoverished, and the "
           "earlier columns are where CTC may want to place the character."),
     ("h", "The construction"),
     ("p", "The remedy is to run a second, completely separate LSTM over "
           "the sequence reversed, then put its outputs back into forward "
           "order and join the two side by side. Each position then carries "
           "a summary of everything to its left, from the forward pass, and "
           "a summary of everything to its right, from the backward pass. "
           "The two have independent weights throughout; nothing is "
           "shared."),
     ("eq", r"y_t \;=\; \big[\; \overrightarrow{h}_t \;;\; "
            r"\overleftarrow{h}_t \;\big]",
      [("y_t", "the combined output at position t, of width twice the "
               "hidden size"),
       (r"\overrightarrow{h}_t", "the forward LSTM's hidden state at "
                                 "position t, summarising positions 1 to t"),
       (r"\overleftarrow{h}_t", "the backward LSTM's hidden state at the "
                                "same position, summarising positions t to "
                                "the end"),
       (r"[\;\cdot\;;\;\cdot\;]", "concatenation: the two vectors laid end "
                                  "to end, not added or combined")],
      None),
     ("p", "Concatenation rather than addition is the right choice because "
           "the two summaries answer different questions and the layer "
           "above should be free to weigh them differently. Adding them "
           "would force every feature to mean the same thing in both "
           "directions and would discard which direction it came from."),
     ("h", "Why this is legitimate here and not always"),
     ("p", "Reading the future is only possible because the whole line is "
           "available before any output is needed. A system transcribing "
           "speech as it is spoken could not do this, since the future has "
           "not happened yet. Handwriting recognition on a captured image "
           "has no such constraint: the entire line was photographed at "
           "once. Bidirectional recurrence is standard in handwriting "
           "recognition for exactly this reason, and the architecture this "
           "network follows is built on it \\cite{shi2017}."),
     ("note", "Reversing a numpy array along its first axis is a view, not "
              "a copy, so running the backward pass costs nothing extra in "
              "memory. The output of the backward pass is reversed again so "
              "that position t of both halves refers to the same column; "
              "forgetting that second reversal would align each position "
              "with its mirror image and produce fluent-looking nonsense."),
   ]),

 "MultilayerBidirectionalLstm": dict(
   inputs=[("x", "the input sequence"),
           ("layerParams", "a list of parameter sets, one per layer, each "
                           "holding a forward and a backward set"),
           ("hiddenSize", "the width of each direction in each layer")],
   output="the sequence after passing through every layer in turn.",
   theory=[
     ("h", "What stacking buys"),
     ("p", "Each layer's output sequence becomes the next layer's input "
           "sequence. The reader uses two such layers. The gain is the same "
           "as from depth anywhere in a network: the first layer works on "
           "the convolutional features directly and can form short-range "
           "relationships between neighbouring columns, such as grouping the "
           "strokes that make up one letter; the second works on the "
           "first's output, where each position already summarises a "
           "neighbourhood, and can form relationships between those "
           "summaries, such as relating a letter to the word around it."),
     ("h", "Why the second layer's input is twice as wide"),
     ("p", "Because the first layer concatenates two directions, its output "
           "has twice the hidden size, and the second layer's input weights "
           "are shaped to accept that. This is handled implicitly here: the "
           "weights come from the checkpoint and already have the right "
           "shape, so the function needs no knowledge of the widths and the "
           "same loop would work for any number of layers."),
     ("note", "Two layers is the depth the architecture specifies "
              "\\cite{shi2017}. More would add parameters and sequential "
              "work for a task where the first layer already covers a "
              "letter and the second a word, which is the range the "
              "evidence actually spans."),
   ]),

 "ResizeLineImageFixed": dict(
   inputs=[("img", "the line crop as a grey image"),
           ("height", "the target height, default 64"),
           ("width", "the target width, default 640"),
           ("padValue", "the value to fill unused canvas with, default 255 "
                        "for white")],
   output="the crop fitted onto a canvas of exactly the trained size.",
   theory=[
     ("h", "What it is for"),
     ("p", "The networks accept exactly one input size, for the reasons "
           "given under INPUT_HEIGHT. A crop cut from a page is whatever "
           "size that line of writing happened to be. This function "
           "reconciles the two, by delegating to FitFixedCanvas in "
           "segment.py."),
     ("h", "Why the aspect ratio is preserved"),
     ("p", "The important property, explained in full in the entry for "
           "FitFixedCanvas, is that a single scale factor is applied to "
           "both axes, chosen so the crop fits inside the canvas, and the "
           "remainder is padded with white rather than stretched. "
           "Stretching each axis independently to fill the canvas would "
           "change the proportions of the letters, and proportion is "
           "precisely what distinguishes some letters from one another and "
           "one writer's hand from another's. A tall narrow l stretched to "
           "fill a wide canvas becomes something no letter looks like, and "
           "the filters trained on correctly proportioned letters would not "
           "find their features in it."),
     ("note", "White, not black, is the pad value, because at this point "
              "the image is still in its natural form of dark ink on light "
              "paper. The inversion happens in the next function, and "
              "padding after inversion would require black instead. Getting "
              "this the wrong way round surrounds every crop with a solid "
              "block of apparent ink."),
   ]),

 "TensorFromResized": dict(
   inputs=[("img", "the fitted grey crop")],
   output="a floating-point array shaped (1, 1, height, width), scaled and "
          "inverted into the range the networks were trained on.",
   theory=[
     ("h", "What it is for"),
     ("p", "An image is 8-bit integers from 0 to 255 with white high. A "
           "network expects floating-point numbers of a particular scale "
           "and sign convention. This function performs that conversion, in "
           "three steps that must match exactly what was done during "
           "training."),
     ("eq", r"x \;=\; \frac{\big(1 - \frac{g}{255}\big) - 0.5}{0.5} "
            r"\;=\; 1 - \frac{2g}{255}",
      [("x", "the value handed to the network, running from −1 for paper to "
             "+1 for black ink"),
       ("g", "the original grey value, 0 for black and 255 for white"),
       (r"\frac{g}{255}", "the grey value scaled to the range 0 to 1"),
       (r"1 - \cdot", "the inversion, making ink high and paper low"),
       (r"\frac{\cdot - 0.5}{0.5}", "the standardisation, mapping the range "
                                    "0 to 1 onto −1 to +1")],
      None),
     ("h", "Why invert"),
     ("p", "After inversion, ink is a large number and blank paper is "
           "zero before standardising. This matters because of padding: "
           "every convolution pads with zeros, and zero must therefore mean "
           "nothing rather than something. With the original convention "
           "zero means black, so the padding round the edge of every crop "
           "would read as a border of solid ink and the first filters would "
           "spend their response on an artefact of the padding."),
     ("h", "Why standardise to a range centred on zero"),
     ("p", "Networks train better when their inputs are centred near zero "
           "and of a magnitude near one, for the same reasons batch "
           "normalisation exists. Inputs in the hundreds would make the "
           "first layer's outputs correspondingly large and would demand "
           "very small initial weights to compensate. Mapping to −1 to +1 "
           "puts the input on the scale the initialisation assumed."),
     ("h", "Why the two extra axes"),
     ("p", "The array is given the shape (1, 1, height, width) to match the "
           "four-axis convention described in the chapter introduction: one "
           "image in the batch, one channel in that image. The axes are "
           "necessary even though both lengths are 1, because every layer "
           "indexes by position and would otherwise misinterpret which axis "
           "is which."),
     ("note", "These three steps are not a design decision that can be "
              "revisited independently; they are a reproduction of what the "
              "training code did. A network is only correct on inputs "
              "prepared the way its training inputs were, and a mismatch "
              "here, such as a forgotten inversion, produces a network that "
              "loads without error and returns confident nonsense."),
   ]),

 "PenWidthXh": dict(
   output="0.14, the pen width that stroke normalisation redraws at, "
          "expressed as a fraction of the x-height.",
   theory=[
     ("h", "Why the width is given in x-heights and not in pixels"),
     ("p", "A width in pixels would mean something different on every crop, "
           "since crops differ in resolution and writers differ in size. "
           "Expressed as a fraction of the x-height it means the same "
           "thing everywhere: a stroke whose thickness is about a seventh "
           "of the height of a letter body. The actual pixel width is "
           "computed per crop from its measured x-height, which is what "
           "makes the normalisation resolution-free."),
     ("h", "Why this value"),
     ("p", "It has to be thick enough that the redrawn writing looks like "
           "writing rather than a wire diagram, since the network judging "
           "it was trained on strokes of a realistic thickness, and thin "
           "enough that the counters, the enclosed white spaces inside "
           "letters like a, e and o, survive. At about a seventh of the "
           "x-height a counter remains open, while a noticeably thicker pen "
           "would fill the loop of an e and make it indistinguishable from "
           "a c with a closed top. That failure is not hypothetical for "
           "this project; distinguishing those two shapes has been a real "
           "source of error in the glyph library."),
   ]),

 "StrokeNormalize": dict(
   inputs=[("gray", "a line crop as a grey image"),
           ("penWidthXh", "the pen width to redraw at, in x-heights, "
                          "defaulting to the constant above")],
   output="the writing redrawn at one constant pen width as ink on white, "
          "or nothing if the crop has no usable ink.",
   theory=[
     ("h", "The problem it solves"),
     ("p", "A network asked to identify a writer from a raw crop will use "
           "whatever is most reliably distinctive, and that is not "
           "necessarily the handwriting. Pen width and ink darkness are "
           "extremely easy to measure and happen to be consistent within "
           "each writer's pages, because each writer used one pen. A "
           "network can reach high accuracy on such a dataset while having "
           "learned nothing about letterforms at all. This is a "
           "**confound**: a variable correlated with the answer that is not "
           "the thing being studied."),
     ("p", "The confound becomes a real failure precisely where the project "
           "needs the namer most. The robot draws with one pen, so every "
           "writer's reproduced output has identical stroke width. A namer "
           "relying on width therefore has nothing to go on and is being "
           "asked a question it was never really trained for. Removing pen "
           "width from the input forces the network to use shape, which is "
           "the property that does survive reproduction."),
     ("h", "The method: thin, then re-ink"),
     ("p", "The writing is reduced to its centre line and then redrawn at a "
           "chosen width. Whatever the original thickness, the output "
           "thickness is the same, so the information is genuinely "
           "destroyed rather than merely de-emphasised."),
     ("items", [
       "**Binarise.** The crop is thresholded into ink and paper by "
       "BinarizeLine.",
       "**Measure the x-height.** CoreBand finds the band of rows holding "
       "the letter bodies, and its height is the x-height, which sets the "
       "scale for the pen width.",
       "**Thin.** Skeletonize reduces the ink to a one-pixel-wide centre "
       "line by the Zhang-Suen method \\cite{zhangsuen1984}, which removes "
       "boundary pixels repeatedly while preserving connectivity, so the "
       "skeleton has the same topology as the original writing.",
       "**Re-ink.** The skeleton is dilated by a square of the chosen "
       "width, which draws a stroke of that width centred on the centre "
       "line, and the result is inverted back to dark ink on white paper.",
     ]),
     ("h", "Why the pen width is forced to be odd"),
     ("eq", r"k \;=\; 2\left\lfloor \frac{w}{2} \right\rfloor + 1, "
            r"\qquad w \;=\; \mathrm{clip}\big(\mathrm{round}"
            r"(0.14 \cdot \mathrm{xh}),\; 2,\; 9\big)",
      [("k", "the structuring element size actually used, always odd"),
       ("w", "the pen width in pixels before being made odd"),
       (r"\mathrm{xh}", "the measured x-height in pixels"),
       ("0.14", "the pen width as a fraction of the x-height"),
       (r"\mathrm{clip}(\cdot,2,9)", "the width held between 2 and 9 "
                                     "pixels, so a tiny crop still gets a "
                                     "visible stroke and a large one does "
                                     "not get a blob"),
       (r"\lfloor \cdot \rfloor", "rounding down")],
      None),
     ("p", "A dilation by an even-sized square has no centre cell, so the "
           "redrawn stroke is offset by half a pixel from the skeleton it "
           "was drawn around. Over a whole line that shifts the writing "
           "slightly and, worse, shifts it in a direction that depends on "
           "the rounding, so two crops of the same writing could be offset "
           "differently. Forcing the size odd puts the pen exactly on the "
           "centre line."),
     ("h", "The x-height fallback"),
     ("p", "If no usable band is found, or the band is under four pixels "
           "tall, the x-height is taken as a third of the crop height "
           "instead, with a floor of eight pixels. A crop is built around "
           "one line of writing whose body occupies roughly the middle "
           "third, so that is a reasonable estimate when measurement "
           "fails, and it keeps the function returning something usable "
           "rather than propagating a nonsensical width."),
     ("note", "Two cases return nothing at all: a crop with no ink, and one "
              "whose skeleton is empty. Returning nothing rather than a "
              "blank image is deliberate, because the caller must be able "
              "to tell \"normalisation failed\" from \"this line is blank\" "
              "and fall back to the raw crop. ClassifyImage does exactly "
              "that."),
   ]),

 "DecodeCtc": dict(
   inputs=[("logProbs", "the reader's output, shaped (time, batch, "
                        "classes), holding a log probability per character "
                        "per column")],
   output="a list of strings, one per item in the batch.",
   theory=[
     ("h", "The problem CTC exists to solve"),
     ("p", "The reader produces one set of character scores per column, "
           "around 160 of them, while the line contains perhaps 40 "
           "characters. Nothing says which columns correspond to which "
           "character, and the correspondence differs for every line and "
           "every writer: a wide m covers many columns, a narrow i covers "
           "few. Training a network to produce a short sequence from a long "
           "one would ordinarily require knowing that correspondence, which "
           "would mean marking by hand where every letter begins and ends "
           "on every training line. For tens of thousands of lines that is "
           "not feasible."),
     ("p", "**Connectionist temporal classification** "
           "\\cite{graves2006} removes the requirement. It lets the network "
           "be trained on the text alone, with no indication of where the "
           "characters sit, by defining a rule that maps any per-column "
           "output onto a shorter string, and then training the network so "
           "that the total probability of all column-outputs mapping to the "
           "correct string is high. The alignment is never specified; it is "
           "whatever the network finds convenient."),
     ("h", "The collapsing rule"),
     ("p", "The rule has two steps, applied in this order: merge runs of "
           "the same character into one, then delete the blanks."),
     ("eq", r"\mathcal{B}\big(\pi\big) \;:\; \text{merge repeats, then "
            r"remove blanks}",
      [(r"\mathcal{B}", "the collapsing map, which takes a sequence of one "
                        "label per column and returns a string"),
       (r"\pi", "a **path**: one label chosen for each column, including "
                "possibly the blank"),
       (r"\text{blank}", "the extra label, slot 0, which stands for "
                         "\"emitting nothing here\"")],
      "graves2006"),
     ("h", "Why the blank is necessary"),
     ("p", "The blank is the part that makes the scheme work, and its "
           "purpose is easiest to see from what goes wrong without it. If "
           "the rule were only \"merge repeats\", the word `letter` could "
           "never be produced, because its two consecutive t's would merge "
           "into one. The blank provides a separator: a path reading "
           "t, t, blank, t, t collapses first to t, blank, t and then, with "
           "the blank removed, to tt. So a blank between two runs of the "
           "same character means a genuine doubled letter, while its "
           "absence means one letter held across several columns."),
     ("items", [
       "**a, a, a, b, b** collapses to **ab**, one a and one b.",
       "**a, blank, a** collapses to **aa**, a doubled letter.",
       "**a, a, blank, b** collapses to **ab**, the blank being harmless "
       "between different letters.",
     ]),
     ("p", "The blank also gives the network somewhere to put the columns "
           "between letters, the gaps and joins where no character is being "
           "written. In practice a trained CTC network emits blank for the "
           "large majority of columns and spikes briefly on a character "
           "where it is confident, which is why its output looks sparse."),
     ("h", "Greedy decoding, and what it approximates"),
     ("p", "What is wanted is the single most probable *string*, which "
           "requires summing over every path that collapses to it, since "
           "many paths give the same string. That is expensive. This "
           "function instead takes the most probable *path*, by picking the "
           "best-scoring character independently in each column, and "
           "collapses that. This is **greedy** or **best-path** decoding."),
     ("eq", r"\pi^{*}_t \;=\; \arg\max_{k} \; \log p_t(k), "
            r"\qquad \hat{y} \;=\; \mathcal{B}\big(\pi^{*}\big)",
      [(r"\pi^{*}_t", "the chosen label for column t"),
       (r"\log p_t(k)", "the reader's log probability for character k at "
                        "column t"),
       (r"\arg\max_k", "the character index with the highest score"),
       (r"\hat{y}", "the decoded string"),
       (r"\mathcal{B}", "the collapsing map defined above")],
      "graves2006"),
     ("p", "The approximation is not exact: the most probable path need not "
           "collapse to the most probable string, because a string whose "
           "probability is spread over many moderately good paths can beat "
           "one concentrated in a single excellent path. In practice the "
           "difference is small for a well-trained network, whose output is "
           "confident enough that one path dominates, and the measured "
           "accuracy of 96 to 99 percent on real crops is obtained with "
           "this decoder. A beam search would recover the difference at "
           "several times the cost."),
     ("note", "Because the comparison within a column is only between "
              "scores in that same column, the logarithm makes no "
              "difference to which character wins. The log probabilities "
              "could be replaced by raw logits here and the output would be "
              "identical. They are log probabilities because the same array "
              "is used by the alignment, which does need them."),
   ]),

 "CtcForcedAlign": dict(
   inputs=[("logProbs", "the reader's output for one line, shaped (time, "
                        "classes)"),
           ("text", "the known text of that line")],
   output="a list pairing each character with the span of columns it "
          "occupies, together with a confidence; or nothing if the text "
          "cannot be aligned.",
   theory=[
     ("h", "What it is for, and why the project needs it"),
     ("p", "Decoding answers \"what does this line say\". This answers a "
           "different question: given that the line says this, *where* is "
           "each character. That is what the glyph extractor needs. To "
           "build a library of a writer's letterforms, the pipeline must cut "
           "individual letters out of their line crops, and to do that it "
           "must know which columns of the crop hold which letter. No "
           "separate letter-finding step is needed, because the reader "
           "already contains that information implicitly in its per-column "
           "scores; forced alignment extracts it."),
     ("h", "Why it is called forced"),
     ("p", "The text is not being recognised; it is given. The algorithm is "
           "**forced** to produce an alignment consistent with that text, "
           "and among all such alignments it finds the most probable. So it "
           "cannot return the wrong letters, only a wrong idea of where "
           "they are. This is a much easier problem than recognition and "
           "correspondingly more reliable, which is why the project uses "
           "known transcripts for glyph extraction rather than the reader's "
           "own output."),
     ("h", "The extended label sequence"),
     ("p", "The first step is to write out the target as a sequence that "
           "allows blanks everywhere they are permitted: a blank, then each "
           "character followed by a blank."),
     ("eq", r"\mathrm{ext} \;=\; \big(\,\varnothing,\; \ell_1,\; "
            r"\varnothing,\; \ell_2,\; \varnothing,\; \dots,\; "
            r"\ell_n,\; \varnothing \,\big), \qquad S \;=\; 2n+1",
      [(r"\mathrm{ext}", "the extended sequence of states the alignment "
                         "moves through"),
       (r"\varnothing", "the blank label, slot 0"),
       (r"\ell_i", "the slot number of the i-th character of the text"),
       ("n", "the number of characters"),
       ("S", "the length of the extended sequence, hence the number of "
             "states")],
      "graves2006"),
     ("p", "Every valid path through the columns can now be described as a "
           "non-decreasing walk through these states: at each column the "
           "path is in some state, and as columns advance the state index "
           "only ever stays put or increases. Characters sit at odd "
           "positions and the optional blanks at even ones. A path that "
           "skips a blank writes two characters back to back; a path that "
           "lingers in a blank puts a gap between them."),
     ("h", "The allowed moves"),
     ("p", "From state s at the previous column, three moves are "
           "considered: stay at s, advance to s from s−1, or advance to s "
           "from s−2. The third, the skip, is what allows the optional "
           "blanks to be passed over, and it carries one restriction."),
     ("items", [
       "**Stay.** The same label continues across another column: a wide "
       "letter, or a long gap.",
       "**Advance by one.** Move to the next state, either from a character "
       "into its following blank or from a blank into the next character.",
       "**Skip by two.** Jump over a blank, placing two characters in "
       "consecutive columns. Permitted only when the target state is a "
       "character and that character differs from the one two states back.",
     ]),
     ("p", "The restriction on the skip is the whole reason the blanks were "
           "inserted. If the two characters either side of a blank are the "
           "same letter, skipping the blank would make a doubled letter "
           "collapse back into one, by the merge rule. So for a doubled "
           "letter the blank between them is compulsory, and the skip is "
           "forbidden. A skip onto a blank is likewise meaningless, since "
           "it would be jumping over a character. Both cases are enforced "
           "by setting the candidate score to a very large negative number, "
           "which the maximisation then never selects, which is the same "
           "device as the infinities in LeastCostSeam."),
     ("h", "The recurrence"),
     ("p", "This is dynamic programming again, with the same structure as "
           "the seam, but maximising a log probability instead of "
           "minimising a cost. Working in log space means the "
           "multiplication of probabilities along a path becomes an "
           "addition, as established under LogSoftmax."),
     ("eq", r"\alpha[t,s] \;=\; \log p_t\big(\mathrm{ext}[s]\big) \;+\; "
            r"\max\Big( \alpha[t-1,s],\; \alpha[t-1,s-1],\; "
            r"\alpha[t-1,s-2] \Big)",
      [(r"\alpha[t,s]", "the score of the best path that reaches state s at "
                        "column t"),
       (r"\log p_t(\mathrm{ext}[s])", "the reader's log probability, at "
                                      "column t, of the label belonging to "
                                      "state s"),
       (r"\alpha[t-1,\,\cdot\,]", "the best scores at the previous column, "
                                  "for the three states a path may arrive "
                                  "from"),
       (r"\max", "the best of those three arrivals; the third is excluded "
                 "when the skip is not allowed")],
      "graves2006"),
     ("p", "The initial column may only be in state 0, the leading blank, "
           "or state 1, the first character, since a path cannot have "
           "progressed further than one state in one column. At the final "
           "column the path must have reached either the last state, the "
           "trailing blank, or the state before it, the last character; "
           "whichever of those two scores higher is the end of the best "
           "path."),
     ("h", "Why maximum and not sum"),
     ("p", "Training CTC sums over all paths, because the quantity being "
           "maximised is the total probability of the target string. "
           "Alignment takes the maximum instead, because the quantity "
           "wanted is one specific path: the single most probable way the "
           "text could be laid out across the columns. Taking the maximum "
           "of a sequence of choices like this is the **Viterbi** algorithm "
           "\\cite{viterbi1967}, long used for exactly this purpose of "
           "recovering the most likely state sequence behind an observed "
           "signal."),
     ("h", "From the path to column spans"),
     ("p", "Backtracking the recorded moves gives the state occupied at "
           "every column. The i-th character of the text lives in state "
           "1+2i, so the columns whose state equals that number are the "
           "columns that character occupies, and the first and last of them "
           "are its span. A character whose state is never occupied gets no "
           "span, which happens when the aligner had to pass through it in "
           "a single column shared with its neighbours; the caller handles "
           "that by skipping the glyph rather than cutting a zero-width "
           "one."),
     ("h", "The confidence"),
     ("eq", r"\mathrm{conf} \;=\; \frac{1}{T} \sum_{t=1}^{T} "
            r"\log p_t\big(\mathrm{ext}[\mathrm{path}[t]]\big)",
      [(r"\mathrm{conf}", "the returned confidence: the mean log "
                          "probability per column along the chosen path"),
       ("T", "the number of columns"),
       (r"\mathrm{path}[t]", "the state the best path occupies at column t"),
       (r"\log p_t(\cdot)", "the reader's log probability for that state's "
                            "label at that column")],
      None),
     ("p", "Dividing by the number of columns is what makes the figure "
           "comparable between lines. A total would favour short lines "
           "simply because fewer negative terms are added, so the mean is "
           "the per-column average, with zero meaning perfect certainty and "
           "more negative meaning less. The glyph extractor uses it to "
           "reject lines whose alignment is not trustworthy enough to cut "
           "letters from, which matters because a confidently wrong "
           "alignment would poison the library with mislabelled glyphs."),
     ("note", "Two guards return nothing rather than a bad answer: a text "
              "with no recognisable characters, and a line with fewer "
              "columns than characters. The second is a genuine "
              "impossibility rather than a difficulty, since every "
              "character needs at least one column of its own, so no "
              "alignment exists at all."),
   ]),

 "FrameXToPixel": dict(
   inputs=[("canvasX", "a horizontal position on the network's input "
                       "canvas"),
           ("rawW", "the width of the original crop in pixels"),
           ("origH", "the height of the original crop in pixels")],
   output="the corresponding position in the original crop, clamped to lie "
          "within it.",
   theory=[
     ("h", "What it is for"),
     ("p", "An alignment span is measured in network columns, but a glyph "
           "has to be cut from the original crop in original pixels. This "
           "function reverses the geometric transformation that got the "
           "crop onto the canvas, so a column index becomes a pixel "
           "position."),
     ("h", "Reversing the fit"),
     ("p", "The fit applied one scale factor to both axes, chosen as the "
           "largest that lets the crop fit inside the canvas, which is the "
           "smaller of the two ratios. Undoing it is division by that same "
           "factor."),
     ("eq", r"s \;=\; \min\!\left( \frac{H_{\mathrm{in}}}{H_{\mathrm{orig}}},"
            r"\; \frac{W_{\mathrm{in}}}{W_{\mathrm{raw}}} \right), "
            r"\qquad x_{\mathrm{orig}} \;=\; "
            r"\mathrm{clip}\!\left( \frac{x_{\mathrm{canvas}}}{s},\; 0,\; "
            r"W_{\mathrm{raw}} \right)",
      [("s", "the scale factor that was applied when fitting the crop"),
       (r"H_{\mathrm{in}},\, W_{\mathrm{in}}", "the network's canvas height "
                                               "and width, 64 and 640"),
       (r"H_{\mathrm{orig}},\, W_{\mathrm{raw}}", "the original crop's "
                                                  "height and width"),
       (r"x_{\mathrm{canvas}}", "the position on the canvas"),
       (r"x_{\mathrm{orig}}", "the corresponding position in the original "
                              "crop"),
       (r"\mathrm{clip}(\cdot,0,W_{\mathrm{raw}})", "the result held inside "
                                                    "the crop")],
      None),
     ("h", "Why the smaller ratio, and why no offset"),
     ("p", "Taking the smaller of the two ratios is what guarantees the "
           "crop fits on both axes; the larger would overflow one of them. "
           "There is no offset term because the fit places the crop at the "
           "top left of the canvas and pads to the right and below, so the "
           "origins coincide. Had the crop been centred, this conversion "
           "would need to subtract the left margin first."),
     ("h", "Why the result is clamped"),
     ("p", "A column index can map slightly beyond the crop, because the "
           "network's columns are a coarser grid than the original pixels "
           "and the last one may extend past the real content into the "
           "padding. Clamping keeps the returned position usable as an "
           "index without the caller checking. The value is returned as a "
           "floating-point number rather than an integer because the caller "
           "may want to interpolate between pixels, and rounding here would "
           "discard precision that was correctly computed."),
   ]),

 "PaperCRNNNumpy": dict(
   inputs=[("checkpointPath", "which checkpoint to load, defaulting to the "
                              "one chosen above"),
           ("hiddenSize", "the LSTM width, default 256")],
   output="an object holding the loaded weights, whose forward method maps "
          "a crop to log probabilities per character per column.",
   theory=[
     ("h", "What a CRNN is"),
     ("p", "A **convolutional recurrent neural network** "
           "\\cite{shi2017} is the standard architecture for reading a line "
           "of text, and it is a direct expression of the structure of the "
           "problem. Reading a line requires two different kinds of "
           "reasoning: recognising a shape, which is local and spatial, and "
           "assembling shapes into words, which is sequential and "
           "long-range. Convolutions do the first, recurrence does the "
           "second, and CTC joins them to text without requiring the "
           "letters to be located first. Each of the three parts is doing "
           "the job it is suited to."),
     ("h", "The layer sequence, and what happens to the data"),
     ("items", [
       "**stage1, two convolution groups,** on a (1, 64, 640) input, "
       "finding edges and stroke directions.",
       "**Pool by two,** halving to 32 by 320.",
       "**stage2, four groups,** finding corners, junctions and curves.",
       "**Pool by two,** halving again to 16 by 160.",
       "**stage3, six groups,** finding letter-scale structure, ending with "
       "128 channels.",
       "**Collapse the height,** taking the maximum down each column, "
       "giving 128 features per column.",
       "**Transpose to (160, 1, 128),** making width the sequence axis.",
       "**Two bidirectional LSTM layers,** each position gaining a summary "
       "of the whole line to its left and right.",
       "**A linear layer,** turning each position's features into 75 "
       "character scores.",
       "**Log softmax,** making those scores log probabilities.",
     ]),
     ("h", "The transpose is the key structural step"),
     ("p", "Moving the width axis to the front is what converts an image "
           "into a sequence. Before it, the array is an image with a width; "
           "after it, the array is a sequence of 160 feature vectors, and "
           "the word width no longer appears. Everything downstream, the "
           "recurrence and CTC alike, is defined on sequences and would be "
           "meaningless on an image. The 160 columns become CTC's time axis, "
           "and the fact that they derive from horizontal position is what "
           "makes CTC's monotonic left-to-right assumption correct for this "
           "problem: writing proceeds left to right, so the alignment "
           "between columns and characters never needs to go backwards."),
     ("h", "Why the weights are unpacked in the constructor"),
     ("p", "The LSTM parameters are gathered into a structured list once, "
           "when the object is built, rather than looked up by name on "
           "every call. Reading a page means running the forward pass once "
           "per line, so repeating a few dozen dictionary lookups per line "
           "would be harmless in cost, but gathering them once means the "
           "key names appear in exactly one place. A checkpoint missing a "
           "parameter then fails at construction, naming the key, rather "
           "than part way through reading a page."),
     ("note", "The reverse-direction parameters are stored under keys "
              "ending in `_reverse`, which is the naming convention of the "
              "framework the weights were trained with. The forward "
              "direction's keys are built with a helper while the reverse "
              "ones are written out, which is a small asymmetry of style in "
              "otherwise identical work."),
   ]),

 "ReadText": dict(
   inputs=[("image", "a line crop as a grey image"),
           ("model", "a loaded reader")],
   output="the text read from that crop, as a string.",
   theory=[
     ("h", "What it is for"),
     ("p", "The three steps of reading in one call: fit the crop to the "
           "canvas, convert it to the network's input convention, run the "
           "forward pass, and decode. It exists so that no caller has to "
           "remember the order, and so that there is a single place where "
           "the preparation and the decoding are paired with each other."),
     ("h", "Why that matters"),
     ("p", "The preparation steps are not independently meaningful. Fitting "
           "without inverting, or inverting without standardising, produces "
           "an array the network will accept and misread. Keeping them "
           "together in one function makes the complete sequence the only "
           "thing a caller can invoke, which removes a class of error that "
           "would otherwise be silent."),
     ("note", "Only the first element of the decoded batch is returned, "
              "since the batch axis is always 1 here. The batch machinery "
              "is retained throughout the layers because the training code "
              "uses it, and because removing it would mean two versions of "
              "every layer."),
   ]),

 "Levenshtein": dict(
   inputs=[("a", "one string"), ("b", "the other string")],
   output="the smallest number of single-character insertions, deletions "
          "and substitutions that turn one string into the other.",
   theory=[
     ("h", "What it is for"),
     ("p", "Scoring a reader requires a way to say how wrong an answer is. "
           "Counting matching positions does not work, because a single "
           "missing letter shifts everything after it and would be scored "
           "as if every later letter were also wrong. What is needed is a "
           "measure that understands insertion and deletion, and that is "
           "the **edit distance**, or **Levenshtein distance** "
           "\\cite{levenshtein1966}: the least number of single-character "
           "edits needed to turn one string into the other."),
     ("items", [
       "**Insertion:** add a character. `cat` to `cart` is one edit.",
       "**Deletion:** remove a character. `cart` to `cat` is one edit.",
       "**Substitution:** replace a character. `cat` to `cot` is one edit.",
     ]),
     ("h", "The recurrence"),
     ("p", "This is dynamic programming once more. Let the problem be "
           "solved for every pair of prefixes: how many edits turn the "
           "first i characters of one string into the first j of the other. "
           "The answer for a larger pair follows from three smaller ones, "
           "because the last edit must be one of the three kinds."),
     ("eq", r"D[i,j] \;=\; \min \begin{cases} "
            r"D[i-1,\,j] + 1 & \text{delete } a_i \\ "
            r"D[i,\,j-1] + 1 & \text{insert } b_j \\ "
            r"D[i-1,\,j-1] + \big[\,a_i \neq b_j\,\big] & "
            r"\text{match or substitute} \end{cases}",
      [("D[i,j]", "the edit distance between the first i characters of a "
                  "and the first j characters of b"),
       ("a_i,\\, b_j", "the i-th and j-th characters of the two strings"),
       (r"[\,a_i \neq b_j\,]", "1 when the two characters differ and 0 when "
                               "they match, so a match is free and a "
                               "substitution costs one"),
       (r"\min", "the cheapest of the three ways to arrive")],
      "levenshtein1966"),
     ("p", "The boundary cases are immediate: turning an empty string into "
           "a prefix of length j takes j insertions, and the reverse takes "
           "i deletions. From there the table fills row by row, and the "
           "bottom right corner is the answer for the whole strings."),
     ("h", "Two implementation economies"),
     ("p", "Only the previous row is kept, not the whole table. Each row "
           "depends only on the one before it, so storing all of them would "
           "use memory proportional to the product of the lengths for no "
           "purpose; this version uses memory proportional to the shorter "
           "string alone. That is possible because the distance itself is "
           "wanted and not the sequence of edits, which would require the "
           "full table to backtrack through, exactly as the seam does."),
     ("p", "The function also swaps its arguments so the shorter string "
           "controls the row length. The distance is symmetric, since every "
           "edit has an inverse of the same cost, so swapping cannot change "
           "the answer, and it makes the retained row as short as "
           "possible."),
   ]),

 "CharAcc": dict(
   inputs=[("pred", "the predicted text"), ("truth", "the correct text")],
   output="the fraction of characters correct, between 0 and 1.",
   theory=[
     ("h", "How it is defined"),
     ("p", "The edit distance counts errors; accuracy expresses them as a "
           "proportion of the length of the correct text."),
     ("eq", r"\mathrm{CharAcc} \;=\; \max\!\left(0,\; 1 - "
            r"\frac{D(\mathrm{pred},\,\mathrm{truth})}"
            r"{|\mathrm{truth}|}\right)",
      [(r"\mathrm{CharAcc}", "the character accuracy"),
       ("D", "the edit distance between the two strings"),
       (r"|\mathrm{truth}|", "the number of characters in the correct text"),
       (r"\max(0,\,\cdot)", "the floor at zero, explained below")],
      None),
     ("h", "Why it is divided by the truth length"),
     ("p", "The denominator is the length of the correct text, not of the "
           "prediction and not the longer of the two. That choice makes the "
           "figure answer \"what fraction of what should have been written "
           "came out right\", which is the question of interest, and it "
           "makes the measure comparable across lines of different lengths. "
           "Dividing by the prediction length instead would let a system "
           "improve its score by predicting less."),
     ("h", "Why the floor at zero"),
     ("p", "The edit distance can exceed the length of the correct text, "
           "when the prediction is much longer than the truth: turning a "
           "two-character truth into a fifty-character prediction takes "
           "around fifty edits, giving a raw score around −24. Negative "
           "accuracy is meaningless and, more practically, averaging it "
           "across a page would let one runaway line dominate the mean and "
           "produce a negative overall accuracy. The floor caps the damage "
           "at zero, which is the honest statement that nothing on that "
           "line was right."),
     ("note", "The empty-truth case is handled separately: an empty "
              "prediction scores 1, anything else scores 0. Without it the "
              "division would be by zero. This is the right convention, "
              "since correctly reading a blank line as blank is a success "
              "and reading text off a blank line is a complete failure."),
   ]),

 "WordAcc": dict(
   inputs=[("pred", "the predicted text"), ("truth", "the correct text")],
   output="the fraction of the correct text's words that appear, in order, "
          "in the prediction.",
   theory=[
     ("h", "Why a second measure"),
     ("p", "Character accuracy can be high while the text is still hard to "
           "read, because one wrong character per word leaves every word "
           "wrong. Word accuracy is the stricter measure and is closer to "
           "whether a person could use the output: a word counts only if "
           "every character in it is right. The gap between the two figures "
           "is informative, since a large gap means errors are spread thinly "
           "across many words rather than concentrated."),
     ("h", "Longest common subsequence"),
     ("p", "The implementation counts how many of the correct words appear "
           "in the prediction in the same order, which is the **longest "
           "common subsequence** of the two word lists. A **subsequence** "
           "takes items in order but allows gaps, so it tolerates both "
           "missing and spurious words without penalising the words around "
           "them."),
     ("eq", r"L[i,j] \;=\; \begin{cases} "
            r"L[i-1,\,j-1] + 1 & t_i = p_j \\[2pt] "
            r"\max\big(L[i-1,\,j],\; L[i,\,j-1]\big) & t_i \neq p_j "
            r"\end{cases}",
      [("L[i,j]", "the length of the longest common subsequence of the "
                  "first i true words and the first j predicted words"),
       ("t_i", "the i-th word of the correct text"),
       ("p_j", "the j-th predicted word"),
       (r"\max", "the better of skipping a word on either side when the two "
                 "do not match")],
      None),
     ("eq", r"\mathrm{WordAcc} \;=\; \frac{L[m,n]}{m}",
      [(r"\mathrm{WordAcc}", "the word accuracy"),
       ("L[m,n]", "the length of the longest common subsequence of the full "
                  "word lists"),
       ("m", "the number of words in the correct text"),
       ("n", "the number of predicted words")],
      None),
     ("h", "Why case is ignored"),
     ("p", "Words are compared with case folded, so `The` matches `the`. "
           "That is a deliberate relaxation: a capitalisation error is a "
           "real error and is counted in full by the character measure, but "
           "counting it again as a whole wrong word would double-count one "
           "mistake in the measure meant to indicate readability. A reader "
           "is not impeded by a wrong capital."),
     ("note", "Words are split on whitespace, so punctuation stays attached "
              "to the word it touches and a word followed by a comma must "
              "have that comma right to count. That is strict, and it is "
              "part of why the measured word accuracies sit well below the "
              "character accuracies."),
   ]),

 "AuthorClassifierCNNNumpy": dict(
   inputs=[("checkpointPath", "which checkpoint to load, defaulting to the "
                              "namer chosen above")],
   output="an object holding the weights and the writer name mapping, whose "
          "forward method maps a crop to one raw score per writer.",
   theory=[
     ("h", "What it is for"),
     ("p", "The namer answers a **classification** question: given this "
           "crop, which of the enrolled writers produced it. Unlike the "
           "reader it produces one answer for the whole crop, not a "
           "sequence, and so it needs no recurrence and no CTC."),
     ("h", "What it shares with the reader, and why"),
     ("p", "The convolutional backbone is identical: the same three stages "
           "with the same number of groups, the same two poolings, the same "
           "collapse of the height. That is not laziness. The features "
           "useful for recognising a letter and the features useful for "
           "recognising a hand are largely the same features: stroke "
           "direction, curvature, how strokes join, the proportions of a "
           "letter. The two tasks diverge only in what is done with those "
           "features, and that is exactly where the two networks differ."),
     ("h", "Where it differs"),
     ("items", [
       "**No recurrence.** Writer identity is not a sequential property. "
       "There is no analogue of using a later letter to disambiguate an "
       "earlier one, because the answer is a single global label.",
       "**Average pooling over position.** The remaining sequence axis is "
       "averaged away, turning the line into one feature vector. Averaging "
       "rather than maximising, for the reason given in that entry: the "
       "question is how prevalent a trait is, not whether it occurred.",
       "**A two-layer head.** A linear layer, a ReLU, then a second linear "
       "layer producing one score per writer. The non-linearity between "
       "them lets the decision depend on combinations of pooled features "
       "rather than on each independently, which a single layer could not "
       "express.",
       "**Raw scores out.** The forward pass returns logits, leaving the "
       "softmax to the caller, so that the same output can be used for a "
       "decision, which needs only the ranking, or for a confidence, which "
       "needs the normalisation.",
     ]),
     ("h", "The dropout layer that is not there"),
     ("p", "The head's parameters are numbered 0 and 3, with 1 and 2 "
           "missing. Position 1 is the ReLU, which has no parameters, and "
           "position 2 is **dropout**, a training-time technique that "
           "randomly sets a fraction of the values to zero at each step "
           "\\cite{srivastava2014}. Its purpose is to prevent the network "
           "relying on any one feature, since that feature may be absent, "
           "which forces redundancy and reduces overfitting. At inference "
           "time it is disabled entirely, so the forward pass steps from 0 "
           "to 3 and the gap in the numbering is the trace of a layer that "
           "exists only during training."),
     ("note", "The writer name mapping is read from the checkpoint "
              "alongside the weights and kept on the object. This is what "
              "lets an output slot be reported as a person's name, and "
              "keeping the two together is what prevents a checkpoint "
              "trained on a different writer list from silently "
              "misattributing every page."),
   ]),

 "ClassifyImage": dict(
   inputs=[("image", "a line crop as a grey image"),
           ("model", "a loaded namer"),
           ("authorMapping", "an optional name-to-slot mapping, "
                             "unused by this function"),
           ("strokeNormalise", "whether to redraw at a constant pen width "
                               "first, default true")],
   output="a pair: the index of the most likely writer, and the full array "
          "of probabilities across all writers.",
   theory=[
     ("h", "What it is for"),
     ("p", "The namer's equivalent of ReadText: the whole identification in "
           "one call, from a crop to a writer and a set of confidences."),
     ("h", "The normalisation switch"),
     ("p", "The one real decision in this function is whether to "
           "stroke-normalise first, and it must match the checkpoint being "
           "used. A network trained on normalised crops must be given "
           "normalised crops, and one trained on raw crops must be given "
           "raw crops. Passing the wrong kind does not fail; it produces "
           "confident answers from a network looking at a distribution of "
           "inputs it has never seen. The default is to normalise, which "
           "suits the shape-based checkpoint used for judging drawn output, "
           "and the ink-based checkpoint is used with the switch turned "
           "off."),
     ("p", "If normalisation returns nothing, because the crop has no "
           "usable ink or no skeleton, the raw crop is used instead. That "
           "is a compromise: the network is then seeing the wrong kind of "
           "input. It is preferred to failing outright because the "
           "alternative on a marginal crop is no answer at all, and the "
           "caller can weigh a low-confidence answer. The condition is rare, "
           "since a crop that produces no skeleton has essentially no "
           "writing in it."),
     ("h", "From scores to an answer"),
     ("p", "The logits are turned into probabilities by softmax and the "
           "largest is taken as the prediction. Both are returned, because "
           "they answer different questions: the index says who, and the "
           "full array says how sure and who the runners-up were. The "
           "second is what the synthesis search uses, since it needs a "
           "continuous score to optimise rather than a discrete decision."),
     ("note", "As noted under Softmax, these probabilities are normalised "
              "across the enrolled writers and so cannot express that the "
              "crop belongs to none of them. A page from an unenrolled "
              "writer still receives a confident-looking answer. The "
              "`authorMapping` argument is accepted and not used, the "
              "mapping being carried on the model itself; it remains in the "
              "signature for callers that pass it positionally."),
   ]),
}
