"""Write-up for recognise.py: reading the text, and naming the writer."""

MODULES = {
 "recognise.py": dict(intro=[
   "**Functional units 3 and 4, requirements R3 and R4.** Two trained "
   "networks live here and both run forward only. The reader turns a line "
   "crop into a string of characters; the namer turns a line crop into a "
   "probability for each writer. Training is in train.py and is not needed "
   "to use either network.",

   ("The problem the reader has to solve is worth stating before any code. "
    "A line of handwriting is an image of some width; the text is a sequence "
    "of characters. Nothing tells you where one character ends and the next "
    "begins, and in joined-up writing there often is no such place. So you "
    "cannot simply cut the image into letters and classify each one, which "
    "is how early systems worked and why they failed on cursive. The modern "
    "answer, and the one used here, is to avoid segmenting at all: slide a "
    "window along the line, emit a probability distribution over characters "
    "at every position, and then use a loss function that considers every "
    "possible way those positions could spell the target text. That is "
    "Connectionist Temporal Classification [Graves2006], and the "
    "architecture built around it is the CRNN [Shi2017]."),

   ("The three stages follow from that. **Convolutions** reduce the image to "
    "a sequence of feature vectors, one per horizontal position, each "
    "describing the stroke shapes in a vertical slice. **A bidirectional "
    "LSTM** then reads that sequence in both directions, so the decision at "
    "each position can use the letters on both sides of it, which is what "
    "lets context resolve an ambiguous shape. **CTC decoding** collapses the "
    "per-position output into a string."),

   ("The namer is the same backbone with the sequence part removed. Writer "
    "identity does not depend on reading order, so there is nothing for a "
    "recurrent layer to contribute; the features are pooled across the whole "
    "line and turned into one score per writer. The project trains two "
    "copies of it: one on raw grey crops, which can use how dark and heavy "
    "the pen is, and one on stroke-normalised crops, which cannot and so "
    "must judge letter shape. The second is the one used to score "
    "synthesised output, because synthesised output has no real ink."),
 ]),
}

FUNCS = {

 "LoadStateDictAny": dict(
   inputs="A path to a saved `.npz` checkpoint.",
   output="A plain dictionary of named numpy arrays.",
   body=[
     "Checkpoints in this project were written by several training scripts "
     "over the life of the work, and they do not all use the same key "
     "names. Rather than let that spread into every function that touches "
     "weights, this tries each known layout and returns one normalised "
     "dictionary.",
     "The value of doing it here is that the rest of the file can assume a "
     "single naming convention, and that adding a differently-saved "
     "checkpoint later means changing one function.",
   ]),

 "Conv2d": dict(
   inputs="`x`, a batch of feature maps shaped (batch, channels, height, "
          "width). `w`, the filter bank shaped (outChannels, inChannels, "
          "kh, kw). `b`, one bias per output channel. `stride`, `padding`.",
   output="A batch of feature maps with `outChannels` channels and a height "
          "and width set by the stride and padding.",
   cite="Chellapilla2006",
   body=[
     ("sub", "What a convolution is for"),
     "A convolution looks for the same small pattern everywhere in the "
     "image. One filter is a small grid of weights; sliding it over the "
     "image and recording how strongly it responds at each position gives a "
     "map of where that pattern occurs. A bank of filters therefore turns "
     "an image into a stack of maps, one per pattern.",
     "This is the right tool for handwriting for two reasons. A stroke "
     "junction looks the same wherever on the page it appears, so the same "
     "weights should be reused everywhere, which is far fewer parameters "
     "than a fully connected layer and far less to overfit. And stacking "
     "these layers builds a hierarchy: the first learns edges at various "
     "angles, the next combines edges into curves and corners, the next "
     "combines those into the parts that distinguish letters.",
     ("sub", "How it is computed here"),
     "Done naively this is six nested loops and is unusably slow in Python. "
     "The standard trick is to rewrite it as a single matrix "
     "multiplication [Chellapilla2006]. Every patch the filter will see is "
     "copied out as a column of a large matrix, the filters are flattened "
     "into rows of another, and one matrix product computes every output "
     "position for every filter at once:",
     ("formula", ["out = W_flat @ X_cols + b",
                  "   W_flat :  (outChannels) x (inChannels*kh*kw)",
                  "   X_cols :  (inChannels*kh*kw) x (positions)"]),
     "The cost is that patches overlap, so the column matrix repeats data "
     "and uses more memory than the image. That trade is worth making "
     "because the matrix product runs in optimised compiled code inside "
     "numpy while the loop version would run in the interpreter.",
     "Padding adds a border of zeros so that a filter can be centred on "
     "edge pixels, which keeps the output the same size as the input and "
     "stops the image shrinking with every layer. Stride is how far the "
     "filter hops between positions; a stride above one reduces the output "
     "size and is a cheap way of downsampling.",
   ]),

 "BatchNorm2d": dict(
   inputs="`x`, a batch of feature maps. `gamma`, `beta`, the learned scale "
          "and shift per channel. `mean`, `var`, the statistics recorded "
          "during training. `eps`, a small constant.",
   output="The normalised, rescaled feature maps, same shape as the input.",
   cite="Ioffe2015",
   body=[
     ("sub", "The problem it solves"),
     "In a deep stack, every layer has to cope with whatever distribution "
     "the layer below currently produces, and that distribution shifts "
     "throughout training as the lower weights change. Each layer is "
     "therefore chasing a moving target, which forces a small learning rate "
     "and slows everything down. Batch normalisation fixes the "
     "distribution: each channel is standardised to zero mean and unit "
     "variance before the next layer sees it [Ioffe2015].",
     ("formula", ["xhat = ( x - mean ) / sqrt( var + eps )",
                  "out  = gamma * xhat + beta"]),
     ("sub", "Why the scale and shift are needed"),
     "Forcing every channel to zero mean and unit variance would be a "
     "restriction, not just a convenience: it would prevent a layer from "
     "learning that a feature should be large, or mostly off. The learned "
     "`gamma` and `beta` give the parameters back, so the layer can undo "
     "the normalisation if that is genuinely what is best. What has been "
     "removed is the *dependence on the layer below's drift*, not the "
     "expressiveness.",
     ("sub", "Why the stored statistics"),
     "During training the mean and variance come from the batch at hand. At "
     "inference there may be only one image, and its own statistics are not "
     "representative, so the values recorded during training are used "
     "instead. That is why this forward-only version takes `mean` and `var` "
     "as inputs rather than computing them: using batch statistics here "
     "would make a crop's prediction depend on what else happened to be in "
     "the batch with it.",
     "`eps` guards the square root against a channel that was constant "
     "during training and therefore has zero recorded variance.",
   ]),

 "Relu": dict(
   inputs="`x`, any array.",
   output="`x` with negative values replaced by zero.",
   body=[
     "The non-linearity. Without one, a stack of linear layers collapses: "
     "the composition of linear maps is a single linear map, so a hundred "
     "layers would have exactly the expressive power of one. Something "
     "non-linear between them is what makes depth worth anything.",
     ("formula", ["relu(x) = max(0, x)"]),
     "The rectifier is preferred to the smooth sigmoid-shaped functions "
     "used historically because its gradient is exactly 1 wherever it is "
     "active. A sigmoid saturates: for large inputs its gradient approaches "
     "zero, so in a deep stack gradients are multiplied by small numbers "
     "many times over and effectively vanish before reaching the early "
     "layers. The rectifier does not attenuate in its active region, which "
     "is what made deep stacks trainable in practice.",
     "Its cost is that a unit whose input is always negative passes no "
     "gradient and can stop learning entirely. In this network that is "
     "mitigated by batch normalisation, which keeps each channel's "
     "distribution centred so that units are rarely pushed permanently "
     "negative.",
   ]),

 "Sigmoid": dict(
   inputs="`x`, any array.",
   output="Values squashed into the open interval (0, 1).",
   body=[
     "Used for the LSTM gates, where a value is needed that behaves like a "
     "proportion: 0 means let nothing through, 1 means let everything "
     "through.",
     ("formula", ["sigmoid(x) = 1 / ( 1 + exp(-x) )"]),
     "The saturation that makes this a poor choice for a deep stack's "
     "activations is exactly what makes it right for a gate: a gate should "
     "be able to commit to fully open or fully closed, and the flat tails "
     "are what let it do so stably.",
   ]),

 "MaxPool2d": dict(
   inputs="`x`, a batch of feature maps. The pooling window and stride.",
   output="Smaller feature maps holding the largest value in each window.",
   body=[
     "Reduces resolution by keeping only the strongest response in each "
     "window. Two things are gained. The obvious one is cost: halving both "
     "axes quarters the work of every layer above. The subtler one is "
     "tolerance to position: after pooling, a feature that moves by a pixel "
     "or two produces the same output, so the network is not required to "
     "learn the same letter separately at every possible offset.",
     "Maximum rather than average is used because these maps encode "
     "evidence that a pattern is present. The strongest evidence in a "
     "neighbourhood is the informative summary; averaging would dilute a "
     "clear detection with the surrounding blank.",
   ]),

 "AdaptiveMaxPoolHeight": dict(
   inputs="`x`, a batch of feature maps.",
   output="Feature maps with the height collapsed to one row.",
   body=[
     "Squashes the vertical axis away so that what remains is a sequence "
     "along the line, which is what the recurrent part needs: one feature "
     "vector per horizontal position.",
     "Taking the maximum down each column means the question being asked is "
     "'does this pattern occur anywhere in this vertical slice', which is "
     "the right question for reading. Whether the crossbar of a 't' sits "
     "three pixels higher in one sample than another should not change the "
     "letter, and collapsing by maximum makes the representation ignore "
     "that.",
     "It is 'adaptive' in that the output height is fixed at one whatever "
     "the input height, so crops that survive earlier stages at slightly "
     "different heights still produce a compatible sequence.",
   ]),

 "AdaptiveAvgPoolWidth": dict(
   inputs="`x`, a batch of feature maps. The target width.",
   output="Feature maps resampled to a fixed number of columns.",
   body=[
     "Used by the writer namer, not the reader. Writer identity is a "
     "property of the whole line, so the horizontal axis is reduced to a "
     "fixed size and then flattened into the final layer.",
     "Averaging is correct here where maximum was correct for the reader's "
     "height pooling, because the question is different. The reader asks "
     "whether a pattern is present somewhere; the namer asks what this "
     "writing is like on average. A single unusual letter should not "
     "dominate the judgement of a hand, which a maximum would allow.",
   ]),

 "Linear": dict(
   inputs="`x`, a batch of vectors. `w`, `b`, the weight matrix and bias.",
   output="`x @ w.T + b`, a batch of vectors of the output width.",
   body=[
     "The fully connected layer: every input contributes to every output "
     "with its own weight. In both networks this is the last step, turning "
     "the features into the actual answer — a score per character in the "
     "reader, a score per writer in the namer.",
     "It appears only at the end because it has no notion of position: it "
     "would need a separate weight for the same stroke at every location, "
     "which is both enormous and unable to generalise. Convolutions do the "
     "position-dependent work; this combines their conclusions.",
   ]),

 "LogSoftmax": dict(
   inputs="`x`, an array of scores along the last axis.",
   output="Log probabilities along that axis.",
   cite="Graves2006",
   body=[
     "Converts arbitrary real-valued scores into probabilities, and returns "
     "their logarithms:",
     ("formula", ["softmax(x)_i   = exp(x_i) / sum_j exp(x_j)",
                  "logsoftmax(x)_i = x_i - log sum_j exp(x_j)"]),
     ("sub", "Why logs"),
     "CTC multiplies probabilities along a path hundreds of steps long. In "
     "ordinary arithmetic that product underflows to zero almost "
     "immediately and all information is lost. In logs, multiplication "
     "becomes addition, which cannot underflow, so the whole of CTC is done "
     "in log space and this layer is what feeds it.",
     ("sub", "Why the maximum is subtracted first"),
     "`exp` of a large score overflows. Subtracting the maximum score "
     "before exponentiating leaves the result unchanged, because a constant "
     "factor cancels between numerator and denominator, while guaranteeing "
     "the largest exponent is `exp(0) = 1`. This is the standard stable "
     "formulation and without it a confident network produces `inf` and "
     "then `nan`.",
   ]),

 "Softmax": dict(
   inputs="`x`, an array of scores.",
   output="Probabilities summing to one along the last axis.",
   body=[
     "The same conversion without the logarithm, used by the writer namer, "
     "where the output is wanted as a directly readable confidence per "
     "writer rather than as something to be multiplied along a path.",
     "The exponential is what makes this different from simply dividing by "
     "the sum: it keeps everything positive regardless of the sign of the "
     "scores, and it widens the gap between the leader and the rest, so a "
     "modest advantage in score becomes a clear advantage in probability.",
   ]),

 "ConvBlockForward": dict(
   inputs="The input maps and the parameters of one block.",
   output="The block's output maps.",
   body=[
     "The repeated unit of the backbone: convolution, then batch "
     "normalisation, then rectifier, sometimes followed by pooling. Written "
     "once because the backbone is this block several times over.",
     "The order matters. Normalising after the convolution and before the "
     "rectifier means the rectifier receives a centred distribution, so "
     "roughly half of its inputs are positive and the unit is in its useful "
     "region. Normalising after the rectifier would standardise a quantity "
     "that is already clipped at zero, which is a less useful thing to "
     "standardise.",
   ]),

 "LstmForward": dict(
   inputs="The input sequence, the weight matrices for input and hidden "
          "connections, and the biases.",
   output="The sequence of hidden states, one per step.",
   cite="Hochreiter1997",
   body=[
     ("sub", "Why a recurrent layer at all"),
     "After the convolutions, each horizontal position has a feature vector "
     "describing the strokes near it. Deciding which letter that is often "
     "requires more than the local evidence: a smudged letter may be "
     "readable only from the word it is in, and in joined writing the "
     "boundary between two letters is genuinely ambiguous in isolation. A "
     "recurrent layer carries information along the line, so a decision at "
     "one position can draw on what came before it.",
     ("sub", "Why LSTM rather than a plain recurrent layer"),
     "A plain recurrent layer multiplies its state by the same matrix at "
     "every step. Over a long line that repeated multiplication either "
     "shrinks the signal towards zero or grows it without bound, so "
     "information from early in the line never reaches the end and the "
     "gradient that should train it never reaches back. The LSTM "
     "[Hochreiter1997] solves this by keeping a separate cell state that is "
     "*added to* rather than multiplied, with learned gates controlling what "
     "enters, what is kept, and what is read out.",
     ("sub", "The gates"),
     "At each step, the input and the previous hidden state together "
     "produce four quantities, three of them squashed by a sigmoid into the "
     "range 0 to 1 and used as proportions:",
     ("formula", ["i = sigmoid( Wi x + Ui h_prev + bi )    how much to take in",
                  "f = sigmoid( Wf x + Uf h_prev + bf )    how much to keep",
                  "g = tanh   ( Wg x + Ug h_prev + bg )    the candidate value",
                  "o = sigmoid( Wo x + Uo h_prev + bo )    how much to read out",
                  "",
                  "c = f * c_prev + i * g                  the new cell state",
                  "h = o * tanh(c)                         the output"]),
     "The line `c = f * c_prev + i * g` is the whole point. When the forget "
     "gate is near 1 and the input gate near 0, the cell state passes "
     "through a step essentially unchanged, so information can survive "
     "arbitrarily many steps. That same additive path is what carries the "
     "gradient backwards intact during training.",
     ("sub", "Implementation note"),
     "The four gates' weights are stored as one stacked matrix, ordered "
     "input, forget, cell, output, which matches the convention used by the "
     "major frameworks. That is not an arbitrary choice: it means one "
     "matrix product computes all four pre-activations at once, and it "
     "means a checkpoint saved by other tooling loads here unchanged.",
   ]),

 "BidirectionalLstmForward": dict(
   inputs="The sequence and the parameters of both directions.",
   output="The concatenation of the forward and backward hidden states at "
          "every position.",
   cite="Hochreiter1997, Shi2017",
   body=[
     "Runs one LSTM left to right and a second, with its own weights, right "
     "to left, then joins their outputs at each position.",
     "The reason is that reading is not causal. When deciding what the "
     "third letter of a word is, the letters after it are as informative as "
     "the letters before it — a shape that could be 'c' or 'e' is often "
     "settled by what follows. A forward-only layer cannot use that; a "
     "bidirectional one gives every position a summary of the entire line "
     "on both sides.",
     "This is affordable precisely because recognition is offline. The "
     "whole line exists before reading starts, so there is no difficulty in "
     "running a pass from the end.",
   ]),

 "MultilayerBidirectionalLstm": dict(
   inputs="The sequence and a list of per-layer parameters.",
   output="The final layer's output sequence.",
   body=[
     "Stacks bidirectional layers, each reading the sequence produced by "
     "the one below. The first layer works with local stroke features; "
     "layers above it work with increasingly summarised descriptions, which "
     "is the same depth argument as for convolutions, applied along the "
     "sequence instead of across the image.",
   ]),

 "ResizeLineImageFixed": dict(
   inputs="`gray`, a line crop.",
   output="The crop on the fixed canvas the networks were trained with.",
   body=[
     "The gateway every crop passes through before either network sees it. "
     "It delegates to `FitFixedCanvas`, so the aspect ratio is preserved "
     "and the unused area is filled with paper.",
     "Why preserving aspect ratio is not optional: both networks are "
     "reading letter proportion. Stretching a short crop to fill a wide "
     "canvas makes the writing wider relative to its height, which is "
     "precisely the kind of difference the writer namer is trained to treat "
     "as a different person.",
   ]),

 "TensorFromResized": dict(
   inputs="A fitted grey image.",
   output="A float array shaped (1, 1, height, width), scaled for the "
          "network.",
   body=[
     "Adds the batch and channel axes the layers expect and converts the "
     "pixel range to the one used during training. Getting the scaling "
     "wrong does not error — it produces confident nonsense, because every "
     "learned weight was fitted on a particular input range.",
   ]),

 "PenWidthXh": dict(
   inputs="Not a function; the constant used by `StrokeNormalize`.",
   output="0.14, a pen width as a fraction of the x-height.",
   body=[
     "The single width that stroke normalisation redraws everything at. It "
     "is expressed relative to the x-height rather than in pixels so that "
     "the normalised appearance is the same whatever the size of the crop.",
   ]),

 "StrokeNormalize": dict(
   inputs="A grey line crop.",
   output="A crop in which the writing has been redrawn at one constant pen "
          "width.",
   body=[
     ("sub", "The purpose"),
     "To separate *what was written* from *what it was written with*. Ink "
     "darkness, pen width and pressure are powerful clues to who wrote "
     "something, and a classifier given raw crops will happily use them — "
     "which is excellent when identifying a photographed page, and useless "
     "when judging a drawing made by a gantry holding one pen.",
     ("sub", "How"),
     "The ink is thresholded, thinned to its one-pixel centreline, and then "
     "redrawn by dilating that centreline to a fixed width set by "
     "`PenWidthXh` relative to the measured x-height. What survives is "
     "stroke geometry: the path the pen took, the proportions, the slant, "
     "the curvature. What is destroyed is everything about the ink itself.",
     ("sub", "Why it matters to this project"),
     "The writer namer trained on normalised crops is the only one of the "
     "two that can fairly judge synthesised handwriting, since synthesised "
     "handwriting has no real ink. The measured gap between the two models "
     "is visible in the results: on real photographed crops the ink-based "
     "model reaches 99 to 100 per cent while the shape-based model reaches "
     "94 to 100, and on individual crops the two can disagree sharply — one "
     "of the writers has a crop the ink model identifies with 100 per cent "
     "confidence and the shape model with 48 per cent.",
   ]),

 "DecodeCtc": dict(
   inputs="`logProbs`, a (time, classes) array of log probabilities.",
   output="The decoded string.",
   cite="Graves2006",
   body=[
     ("sub", "The problem"),
     "The network emits a distribution over characters at every horizontal "
     "position — typically 160 positions for a line of perhaps 40 "
     "characters. Turning that into text means deciding which positions "
     "belong to the same letter, without being told.",
     ("sub", "The blank, and why it is necessary"),
     "CTC adds an extra class, the blank, meaning 'no character here' "
     "[Graves2006]. Decoding is then two rules applied in order: collapse "
     "runs of the same character, then delete the blanks.",
     ("formula", ["a a a _ a a   ->  'aa'      (blank separates the two a's)",
                  "a a a a a a   ->  'a'       (no blank, so one letter)",
                  "_ b b _ _ e   ->  'be'"]),
     "Without the blank there would be no way to write a doubled letter, "
     "since any repeat would collapse to one. The blank is what lets the "
     "network say 'this is a new letter' as distinct from 'this is still "
     "the same letter'.",
     ("sub", "Greedy decoding"),
     "This function takes the most likely class at each position "
     "independently and then applies the two rules. That is not guaranteed "
     "to find the most likely *string*, because many position-sequences can "
     "spell the same text and their probabilities should really be added "
     "before comparing. A beam search would account for that. Greedy "
     "decoding is used here because it is far cheaper and, on a well "
     "trained network, almost always agrees — the measured character "
     "accuracy with greedy decoding is 96 to 99 per cent per writer on real "
     "crops.",
   ]),

 "CtcForcedAlign": dict(
   inputs="`logProbs`, the (time, classes) output. `text`, the transcript "
          "that is already known.",
   output="`(alignment, confidence)` where alignment gives the column span "
          "of each character, or `None` if the text cannot be aligned.",
   cite="Graves2006, Viterbi1967",
   body=[
     ("sub", "A different question from decoding"),
     "Decoding asks 'what does this say'. Forced alignment asks 'given that "
     "it says this, where is each letter'. The second question is the one "
     "the glyph extractor needs: to cut a writer's letters out of their "
     "pages, it must know which columns belong to which character, and the "
     "transcript is already known.",
     ("sub", "How it works"),
     "Because the text is known, there is exactly one sequence of characters "
     "to consider; what is unknown is the timing. The target is expanded "
     "with blanks between every character and at both ends, and then the "
     "single best path through the trellis that spells it is found by "
     "dynamic programming, keeping at each step the best running score of "
     "reaching that state [Viterbi1967]:",
     ("formula", ["best[t][s] = logProb[t][label[s]]",
                  "           + max( best[t-1][s],",
                  "                  best[t-1][s-1],",
                  "                  best[t-1][s-2] if label[s] is not blank",
                  "                                 and differs from label[s-2] )"]),
     "The three predecessors are the three legal moves: stay on the current "
     "symbol, advance by one, or skip a blank to reach the next real "
     "character. The skip is forbidden between two identical characters, "
     "because then the blank between them is what distinguishes 'aa' from "
     "'a' and must not be bypassed.",
     "Taking the maximum gives the single most likely alignment, which is "
     "what is wanted here. Training instead sums over all alignments, which "
     "is the same recurrence with addition in place of maximum.",
     "The returned confidence is the score of that best path, and callers "
     "use it to reject lines the reader could not align well; the profile "
     "builder keeps only glyphs from lines above a confidence percentile, "
     "on the grounds that a boundary from a line the reader could not read "
     "is not a boundary worth cutting on.",
   ]),

 "FrameXToPixel": dict(
   inputs="A horizontal position in network-output coordinates, plus the "
          "original crop's width and height.",
   output="The corresponding column in the original crop.",
   body=[
     "Alignment gives positions in the network's own coordinate system, "
     "which is the fixed canvas after pooling. To cut a glyph out of the "
     "original crop, that has to be mapped back through every size change "
     "that was applied: the scale factor used to fit the canvas, the "
     "left-flush placement, and the horizontal reduction caused by pooling.",
     "This is the true inverse of the forward path rather than an "
     "approximation, which matters because an error here moves every "
     "character boundary by the same amount and would bias every glyph in "
     "the library in the same direction.",
     "This function is also the subject of a bug fixed during the work: it "
     "was previously imported from a module that did not define it, so the "
     "extractor failed at call time while importing cleanly.",
   ]),

 "PaperCRNNNumpy": dict(
   inputs="Constructed with an optional `checkpointPath`; defaults to the "
          "first reader checkpoint found on disk.",
   output="An object holding the weights, whose `forward` returns log "
          "probabilities shaped (time, batch, classes).",
   cite="Shi2017, Graves2006",
   body=[
     "The reader, assembled. It loads the weights and runs the stages in "
     "order: the convolutional backbone to describe stroke shapes, height "
     "pooling to turn the image into a sequence, the bidirectional LSTM "
     "stack to read along it with context, a linear layer to produce one "
     "score per character per position, and a log-softmax to make those "
     "into log probabilities.",
     "**Which checkpoint is loaded is the single most consequential "
     "configuration choice in the project.** The default search order "
     "prefers a checkpoint trained from scratch over one trained jointly, "
     "and measured on the same 669 real line crops the difference is 22 to "
     "26 per cent character accuracy against 96 to 99 per cent. Both files "
     "exist on disk and both load without complaint; nothing in the output "
     "indicates which was used.",
   ]),

 "ReadText": dict(
   inputs="`img`, a grey line crop. `reader`, a loaded `PaperCRNNNumpy`.",
   output="The recognised string.",
   body=[
     "The whole reading path in one call: fit the crop to the canvas, "
     "convert it to a tensor, run the network, decode greedily. Most of the "
     "pipeline and all of the measurement code uses this rather than the "
     "pieces.",
   ]),

 "Levenshtein": dict(
   inputs="Two strings.",
   output="The minimum number of single-character edits that turn one into "
          "the other.",
   cite="Levenshtein1966",
   body=[
     "The standard edit distance [Levenshtein1966], and the basis of every "
     "accuracy figure in the project. Comparing strings by equality would "
     "throw away all information about how close a wrong answer was; edit "
     "distance measures it.",
     "It is computed by dynamic programming over a table where cell (i, j) "
     "is the cheapest way to turn the first i characters of one string into "
     "the first j of the other:",
     ("formula", ["D[i][j] = min( D[i-1][j]   + 1,          delete",
                  "               D[i][j-1]   + 1,          insert",
                  "               D[i-1][j-1] + (a[i] != b[j]) )  substitute"]),
     "Each cell depends only on its three predecessors, so one pass fills "
     "the table in O(n*m) time; the implementation keeps only the previous "
     "row, which reduces the memory to O(min(n, m)).",
   ]),

 "CharAcc": dict(
   inputs="`got`, the recognised string. `want`, the truth.",
   output="A fraction between 0 and 1.",
   cite="Levenshtein1966",
   body=[
     "Character accuracy, defined from the edit distance:",
     ("formula", ["acc = 1 - editDistance(got, want) / max(1, len(want))"]),
     "Normalising by the length of the *truth* rather than of the output is "
     "the convention, and it is the conservative choice: a recogniser that "
     "emits far too much text is penalised for every spurious character, "
     "whereas normalising by the output length would let it hide the "
     "insertions.",
     "The measure can in principle go negative, if the output is longer "
     "than the truth and wrong throughout; callers clamp where that would "
     "be confusing.",
   ]),

 "WordAcc": dict(
   inputs="`got`, `want`.",
   output="The fraction of whitespace-separated words that match exactly.",
   body=[
     "A stricter companion to character accuracy. One wrong letter fails the "
     "whole word, so the figure is always lower, and the gap between the "
     "two is informative: errors spread thinly across many words give a "
     "high character accuracy and a poor word accuracy, while errors "
     "concentrated in a few words leave word accuracy comparatively intact.",
     "In this project the pair reads 96 to 99 per cent character against 85 "
     "to 97 per cent word on real crops, and around 79 per cent character "
     "against 41 per cent word on synthesised lines, which says that the "
     "synthesiser's errors are scattered rather than clustered.",
   ]),

 "AuthorClassifierCNNNumpy": dict(
   inputs="Constructed with a `checkpointPath`.",
   output="An object holding the weights and the writer-name-to-slot "
          "mapping it was trained with.",
   body=[
     "The writer namer. The same convolutional backbone as the reader, then "
     "pooling across the whole line, then a linear layer giving one score "
     "per writer, then a softmax.",
     "There is no recurrent layer because the question has no sequence to "
     "it. Identity is a property of the handwriting as a whole, so pooling "
     "the features and classifying is the natural form; a recurrent layer "
     "would add parameters without adding anything the task needs.",
     "The object carries its own name-to-slot mapping, which is essential: "
     "slot numbers are assigned during training, and a caller that assumed "
     "an ordering would silently attribute each crop to the wrong person.",
   ]),

 "ClassifyImage": dict(
   inputs="`img`, a crop. `model`, a loaded namer. `strokeNormalise`, "
          "whether to redraw at one pen width first.",
   output="`(index, probabilities)` — the most likely writer's slot and the "
          "full distribution.",
   body=[
     "Runs the namer on one crop. The `strokeNormalise` flag selects which "
     "question is being asked, and it must match the checkpoint: the "
     "shape-trained weights expect normalised input and the ink-trained "
     "weights expect raw grey.",
     "The full distribution is returned, not just the winner, because "
     "almost every use needs more than the top answer. Synthesis scores "
     "itself on the probability assigned to the *intended* writer, which "
     "may not be the winner; the measurement harness reports the margin "
     "between the true writer and the best competitor, which is the honest "
     "way to show that an identification was nearly a mistake.",
   ]),
}
