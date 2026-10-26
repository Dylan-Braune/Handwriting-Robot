"""Write-up for train.py: how the two networks were trained."""

MODULES = {
 "train.py": dict(intro=[
   "Everything needed to *make* a checkpoint, and nothing needed to use one. "
   "recognise.py runs the trained networks forward; this file holds the "
   "backward pass of every layer, the losses, the optimisers, the datasets, "
   "the augmentation and the two training runs.",

   ("Why the backward passes are written out by hand. Training a network "
    "means computing, for every weight, how the loss would change if that "
    "weight changed — the gradient. Frameworks obtain this automatically by "
    "recording every operation and replaying it in reverse. With no "
    "framework, each layer's derivative has to be derived and implemented "
    "explicitly, which is the bulk of this file. The results were checked "
    "against numerically estimated gradients, which agreed to about 1e-11; "
    "that check is the only practical defence against a sign error in a "
    "derivative, because a wrong gradient does not crash, it simply trains "
    "to a worse answer."),

   ("The chain rule is the organising principle. If a layer computes y from "
    "x, and we are given dL/dy, the layer must produce dL/dx to pass down "
    "and dL/dW for its own weights. Every backward function below has that "
    "same shape, and the forward functions save whatever intermediate values "
    "their backward counterpart will need — which is why there are separate "
    "`...ForwardTrain` versions: the inference forward passes throw those "
    "away."),

   ("Two networks are trained here. The reader is trained with CTC loss, "
    "because the labels say what was written but not where. The writer namer "
    "is trained with ordinary cross entropy, because each crop has exactly "
    "one correct answer. The namer is trained twice, once on raw grey crops "
    "and once on stroke-normalised ones, producing the ink-based and "
    "shape-based checkpoints."),
 ]),
}


def _simple(summary, *paras):
    return dict(body=[summary] + list(paras))


FUNCS = {

 # ---------------------------------------------------------- layer forward
 "Conv2dForward": dict(
   inputs="The input maps, the filter bank, the bias, stride and padding.",
   output="`(out, cache)` — the output maps and the values backward needs.",
   cite="Chellapilla2006",
   body=[
     "The same unfolded matrix product as the inference version, but it also "
     "returns the column matrix of extracted patches.",
     "That matrix is kept because the weight gradient needs it again, and "
     "rebuilding it during the backward pass would double the cost of the "
     "most expensive operation in the network. Memory is traded for time, "
     "which is the usual bargain in training.",
   ]),

 "Conv2dBackward": dict(
   inputs="`dOut`, the gradient arriving from above, and the cache.",
   output="`(dX, dW, dB)` — gradients for the input, the filters and the "
          "bias.",
   cite="Chellapilla2006",
   body=[
     "The derivative of a convolution. Because the forward pass is a matrix "
     "product, the backward pass is two more matrix products:",
     ("formula", ["dW = dOut_flat @ X_cols.T",
                  "dX_cols = W_flat.T @ dOut_flat",
                  "dB = sum of dOut over batch and spatial positions"]),
     "The weight gradient is the output gradient against the same patches "
     "that produced the output, which is the chain rule applied to a product: "
     "the derivative with respect to one factor is the other factor.",
     "The input gradient comes back as a column matrix and has to be folded "
     "back into image shape. This is the subtle part: patches overlap, so a "
     "single input pixel appears in several columns and receives a "
     "contribution from each. Those contributions must be **added**, because "
     "the pixel influenced the output through every patch it took part in. "
     "Overwriting instead of accumulating here is a classic error and "
     "produces gradients that are wrong by a factor that varies across the "
     "image.",
   ]),

 "BatchNorm2dForwardTrain": dict(
   inputs="The input maps, gamma, beta, the running statistics, and the "
          "momentum for updating them.",
   output="`(out, cache)`, and the running statistics are updated in place.",
   cite="Ioffe2015",
   body=[
     "Normalises using the statistics of the batch at hand rather than the "
     "stored ones, which is what makes batch normalisation work during "
     "training: the whole point is to standardise against the distribution "
     "that is actually arriving.",
     "It simultaneously maintains an exponential moving average of the mean "
     "and variance for use at inference time:",
     ("formula", ["running = (1 - momentum) * running + momentum * batch"]),
     "That average is why a single image can be classified later: at "
     "inference there is no batch to take statistics from, and using one "
     "image's own statistics would make its prediction depend on itself in a "
     "way training never did.",
   ]),

 "BatchNorm2dBackward": dict(
   inputs="`dOut` and the cache.",
   output="`(dX, dGamma, dBeta)`.",
   cite="Ioffe2015",
   body=[
     "The most involved derivative in the file, because the normalisation "
     "couples every item in the batch: the mean and variance depend on all "
     "of them, so changing one input changes the normalisation of all the "
     "others.",
     "The gradient therefore has three contributions — the direct path "
     "through the standardised value, and two indirect paths through the "
     "batch mean and the batch variance:",
     ("formula", ["dxhat = dOut * gamma",
                  "dVar  = sum( dxhat * (x - mean) ) * -0.5 * (var+eps)^-1.5",
                  "dMean = sum( dxhat ) * -1/sqrt(var+eps)",
                  "        + dVar * mean( -2*(x - mean) )",
                  "dX    = dxhat / sqrt(var+eps)",
                  "        + dVar * 2*(x - mean)/N",
                  "        + dMean / N"]),
     "The N in the denominators is the number of elements the statistics were "
     "taken over, which for a 2-D batch norm is batch times height times "
     "width, not batch alone — each channel's statistics pool every spatial "
     "position. Getting that count wrong scales the gradient and is not "
     "visible except as slower training.",
     "dGamma and dBeta are simple sums, since gamma and beta enter linearly.",
   ]),

 "ReluForward": dict(
   inputs="`x`.", output="`(out, cache)` where the cache is the mask of "
          "positive inputs.",
   body=["Keeps the mask rather than the input, because that is all backward "
         "needs and it is a byte per element instead of a float."]),

 "ReluBackward": dict(
   inputs="`dOut`, the mask.", output="`dX`.",
   body=[
     "The derivative of `max(0, x)` is 1 where the input was positive and 0 "
     "where it was not, so the gradient is passed through the same mask:",
     ("formula", ["dX = dOut * (x > 0)"]),
     "At exactly zero the function is not differentiable; the convention is "
     "to take 0, and since an exact zero is measure-zero in floating point "
     "the choice has no practical consequence.",
   ]),

 "MaxPool2dForward": dict(
   inputs="The input maps and the window.", output="`(out, cache)` with the "
          "position of each winner.",
   body=["Records which element won each window, because that is where the "
         "gradient has to go back to."]),

 "MaxPool2dBackward": dict(
   inputs="`dOut` and the cache.", output="`dX`.",
   body=[
     "The output depended on exactly one input per window, so the gradient "
     "goes entirely to that element and every other element in the window "
     "receives zero. This is why max pooling is said to route gradient: it "
     "selects a path rather than spreading influence.",
     "If two elements tie, the recorded index breaks the tie arbitrarily, "
     "which is correct — the function is not differentiable at a tie and "
     "either choice is a valid subgradient.",
   ]),

 "AdaptiveMaxPoolHeightForward": dict(
   inputs="The input maps.", output="`(out, cache)`.",
   body=["Collapses the height by taking the maximum down each column, "
         "recording which row won."]),

 "AdaptiveMaxPoolHeightBackward": dict(
   inputs="`dOut`, the cache.", output="`dX`.",
   body=["Routes the gradient back to the winning row of each column, zero "
         "elsewhere, for the same reason as max pooling."]),

 "AdaptiveAvgPoolWidthForward": dict(
   inputs="The input maps and the target width.", output="`(out, cache)`.",
   body=["Averages groups of columns down to a fixed count, recording the "
         "group boundaries."]),

 "AdaptiveAvgPoolWidthBackward": dict(
   inputs="`dOut`, the cache.", output="`dX`.",
   body=[
     "Every element in a group contributed equally to that group's average, "
     "so the gradient is divided equally among them:",
     ("formula", ["dX[i] = dOut[group(i)] / size(group(i))"]),
     "Averaging spreads gradient where maximum routes it, which is the "
     "backward reflection of the difference discussed in recognise.py: the "
     "namer's averaging means every part of the line trains the weights, "
     "while the reader's maximum means only the strongest evidence does.",
   ]),

 "LinearForward": dict(
   inputs="`x`, `w`, `b`.", output="`(out, cache)`.",
   body=["Keeps the input, which the weight gradient needs."]),

 "LinearBackward": dict(
   inputs="`dOut`, the cache.", output="`(dX, dW, dB)`.",
   body=[
     "The derivatives of a matrix product and a bias:",
     ("formula", ["dX = dOut @ W",
                  "dW = dOut.T @ X",
                  "dB = sum of dOut over the batch"]),
     "Each is the chain rule on a bilinear form: differentiating with "
     "respect to one operand leaves the other.",
   ]),

 "DropoutForward": dict(
   inputs="`x`, the drop probability, and a random generator.",
   output="`(out, mask)`.",
   cite="Srivastava2014",
   body=[
     "Randomly zeroes a fraction of the values during training "
     "[Srivastava2014]. The purpose is to stop the network leaning on any "
     "single feature: if a feature might be absent on any given pass, the "
     "network has to build representations that survive without it, which in "
     "effect trains an ensemble of thinned networks that share weights.",
     "The survivors are scaled up by `1/(1-p)` so that the expected total "
     "passed on is unchanged. Without that, the activations arriving at the "
     "next layer would be systematically smaller during training than at "
     "inference, and every layer above would be calibrated for the wrong "
     "scale.",
   ]),

 "DropoutBackward": dict(
   inputs="`dOut`, the mask.", output="`dX`.",
   body=["The gradient follows the same mask and the same scaling, since a "
         "dropped unit contributed nothing to the output."]),

 "ConvBlockForwardTrain": dict(
   inputs="The input and one block's parameters.",
   output="`(out, cache)` for the whole block.",
   body=["Runs convolution, batch norm and rectifier with their caches, "
         "bundling them so the backward pass has one entry point per "
         "block."]),

 "ConvBlockBackwardTrain": dict(
   inputs="`dOut` and the block cache.",
   output="Gradients for the input and for every parameter in the block.",
   body=["Applies the three backward passes in reverse order — rectifier, "
         "then batch norm, then convolution — which is the chain rule "
         "applied to a composition."]),

 # ------------------------------------------------------------ lstm, train
 "LstmForwardTrain": dict(
   inputs="The input sequence and the parameters.",
   output="`(hidden sequence, cache)` where the cache holds every gate value "
          "at every step.",
   cite="Hochreiter1997",
   body=[
     "The same recurrence as the inference version, but every intermediate "
     "is retained: the four gate activations, the cell state and the output "
     "at each step.",
     "All of it is needed. The gradient at step t depends on the gate values "
     "at step t, and recomputing them would mean re-running the recurrence "
     "from the start for every step, which is quadratic in the sequence "
     "length. For a 160-step line that is the difference between training "
     "and not.",
   ]),

 "LstmBackward": dict(
   inputs="The gradient of the hidden sequence and the cache.",
   output="Gradients for the input sequence and for every weight.",
   cite="Hochreiter1997, Pascanu2013",
   body=[
     ("sub", "Backpropagation through time"),
     "The recurrence is unrolled, so the layer is effectively a very deep "
     "network whose layers share weights. The gradient is therefore computed "
     "by walking backwards from the last step to the first, carrying two "
     "running gradients: one for the hidden state and one for the cell "
     "state, because each step fed both into the next.",
     ("formula", ["at each step, going backwards:",
                  "  dh += gradient arriving from above at this step",
                  "  do  = dh * tanh(c)",
                  "  dc += dh * o * (1 - tanh(c)^2)",
                  "  df  = dc * c_prev",
                  "  di  = dc * g",
                  "  dg  = dc * i",
                  "  dc_prev = dc * f",
                  "  then back through each gate's own non-linearity"]),
     "The line `dc_prev = dc * f` is the counterpart of the additive cell "
     "update, and it is why the LSTM trains over long sequences: the "
     "gradient flowing back through the cell path is multiplied by the "
     "forget gate rather than by a weight matrix, so when the gate is near 1 "
     "it passes back essentially undiminished.",
     "Because weights are shared across steps, every step contributes to the "
     "same weight gradients, which are accumulated across the whole backward "
     "walk rather than assigned.",
     "Even with the cell path, gradients over a long line can grow, which is "
     "what `ClipGradNorm` exists to contain [Pascanu2013].",
   ]),

 "BidirectionalLstmForwardTrain": dict(
   inputs="The sequence and both directions' parameters.",
   output="`(output, cache)` for both directions.",
   body=["Runs both directions with their caches kept separately, since the "
         "two have independent weights and independent gradients."]),

 "BidirectionalLstmBackward": dict(
   inputs="The output gradient and the cache.",
   output="Gradients for the input and both directions' weights.",
   body=[
     "The forward output is the concatenation of the two directions, so the "
     "arriving gradient is split along that axis and each half sent back "
     "through its own direction — the backward-reading half travelling "
     "forwards in time.",
     "Both directions read the same input sequence, so their input gradients "
     "are **summed**: the input influenced the output through both paths.",
   ]),

 "MultilayerBidirectionalLstmForwardTrain": dict(
   inputs="The sequence and per-layer parameters.",
   output="`(output, caches)` for the stack.",
   body=["Each layer reads the one below and keeps its own cache."]),

 "MultilayerBidirectionalLstmBackward": dict(
   inputs="The output gradient and the caches.",
   output="Gradients for the input and for every layer.",
   body=["Walks down the stack in reverse, each layer's input gradient "
         "becoming the output gradient of the layer beneath it."]),

 # ---------------------------------------------------------------- losses
 "LogSumExp": dict(
   inputs="An array and an axis.",
   output="`log(sum(exp(x)))` along that axis, computed stably.",
   body=[
     "Adding probabilities that are stored as logarithms. CTC needs this "
     "constantly, because it sums over alternative alignments while working "
     "in log space to avoid underflow.",
     ("formula", ["logsumexp(x) = m + log sum exp(x - m),   m = max(x)"]),
     "Factoring out the maximum is what makes it stable: without it, `exp` "
     "of a large log-probability overflows to infinity, and `exp` of a very "
     "negative one underflows to zero, either of which destroys the sum. "
     "With it, the largest term is exactly `exp(0) = 1` and the rest are "
     "between 0 and 1.",
   ]),

 "CtcLoss": dict(
   inputs="The log probabilities, the target label sequences, and their "
          "lengths.",
   output="`(loss, gradient)` with respect to the network's output.",
   cite="Graves2006",
   body=[
     ("sub", "The problem being solved"),
     "The label says what was written but not where. For a line of 160 "
     "output positions and a 40-character target there are astronomically "
     "many ways the characters could be distributed across the positions. "
     "CTC defines the probability of the label as the **sum over all of "
     "them** [Graves2006]:",
     ("formula", ["P(label | image) = sum over all alignments A that",
                  "                   collapse to label, of P(A | image)"]),
     "and the loss is its negative logarithm. Summing rather than picking "
     "one alignment is what removes the need for character-level "
     "segmentation in the training data, which is the whole reason the "
     "architecture can be trained on nothing but line images and "
     "transcripts.",
     ("sub", "Why it is computable"),
     "The number of alignments is exponential, but they share structure, so "
     "the sum is obtained by dynamic programming. The target is expanded "
     "with blanks between every character and at both ends, giving `2n+1` "
     "states, and a forward variable alpha[t][s] is defined as the total "
     "probability of all paths that have consumed t output positions and "
     "reached state s:",
     ("formula", ["alpha[t][s] = p[t][label[s]] * (  alpha[t-1][s]",
                  "                                + alpha[t-1][s-1]",
                  "                                + alpha[t-1][s-2] )",
                  "",
                  "the s-2 term only when label[s] is not blank and",
                  "label[s] != label[s-2]"]),
     "The three predecessors are the three legal moves — stay, advance, or "
     "skip a blank — and the restriction on the skip is what forces a blank "
     "to remain between two identical characters, since without it 'aa' "
     "could not be distinguished from 'a'.",
     ("sub", "The gradient"),
     "A symmetric backward variable beta is computed from the end, and the "
     "product alpha*beta at a state gives the total probability of all "
     "complete paths through it. Summing that over the states that emit a "
     "given character gives how much that character's output unit "
     "contributed, and the gradient with respect to the network's output is",
     ("formula", ["dL/dy[t][k] = p[t][k] - ( sum of alpha*beta over states",
                  "                          emitting k ) / P(label)"]),
     "which is the familiar form 'prediction minus target', where the target "
     "is the posterior probability that position t was character k given the "
     "label. That is the same shape as the gradient of cross entropy, which "
     "is not a coincidence: both come from a softmax output with a "
     "likelihood objective.",
     ("sub", "Implementation"),
     "The whole recursion is done in log space with `LogSumExp` in place of "
     "addition and ordinary addition in place of multiplication. In linear "
     "space the products underflow to zero within a few dozen steps and the "
     "loss becomes infinite.",
   ]),

 "CrossEntropyLoss": dict(
   inputs="Logits and integer class labels.",
   output="`(loss, gradient)`.",
   body=[
     "The loss for the writer namer, where each crop has exactly one correct "
     "answer and no alignment problem.",
     ("formula", ["loss = -log( softmax(logits)[correct] )",
                  "dL/dlogits = softmax(logits) - onehot(correct)"]),
     "The gradient is 'prediction minus truth', which is as clean as it is "
     "because the softmax and the logarithm compose: the exponential in the "
     "softmax and the logarithm in the loss cancel, leaving a linear "
     "expression. Implementing the two separately and composing them "
     "numerically would be both slower and less stable.",
     "The loss is computed from the log-softmax rather than by taking the "
     "logarithm of a probability, so that a confident wrong answer gives a "
     "large finite loss instead of the infinity that `log(0)` would "
     "produce.",
   ]),

 # ------------------------------------------------------------ optimisers
 "RMSprop": dict(
   inputs="Parameters, gradients, the running average of squared gradients, "
          "the learning rate, decay and epsilon.",
   output="None; updates in place.",
   cite="Tieleman2012",
   body=[
     "Scales each weight's step by how noisy that weight's gradient has "
     "recently been [Tieleman2012]:",
     ("formula", ["v = decay*v + (1 - decay)*g^2",
                  "w = w - lr * g / ( sqrt(v) + eps )"]),
     "The reasoning: different weights in a deep network see gradients of "
     "wildly different magnitudes, and a single global learning rate is "
     "therefore either too large for some or too small for others. Dividing "
     "by the root mean square of recent gradients makes the step size "
     "roughly scale-free, so one learning rate suits the whole network.",
     "`eps` prevents division by zero for a weight whose gradient has been "
     "consistently zero.",
   ]),

 "AdamW": dict(
   inputs="Parameters, gradients, the two moment estimates, the step count, "
          "the learning rate, betas, epsilon and the weight decay.",
   output="None; updates in place.",
   cite="Kingma2015, Loshchilov2019",
   body=[
     "Momentum and per-weight scaling together [Kingma2015], with weight "
     "decay applied separately [Loshchilov2019].",
     ("formula", ["m = b1*m + (1 - b1)*g            first moment",
                  "v = b2*v + (1 - b2)*g^2          second moment",
                  "mhat = m / (1 - b1^t)            bias correction",
                  "vhat = v / (1 - b2^t)",
                  "w = w - lr * mhat/(sqrt(vhat)+eps) - lr*wd*w"]),
     ("sub", "The two moments"),
     "The first is momentum: averaging recent gradients smooths out the "
     "noise from individual batches and carries the update through small "
     "local irregularities. The second is the RMSprop scaling.",
     ("sub", "Why bias correction"),
     "Both averages start at zero, so early in training they are biased "
     "towards zero and the steps would be far too small. Dividing by "
     "`1 - beta^t` compensates exactly, and since beta^t decays the "
     "correction fades away as the averages warm up.",
     ("sub", "Why the decay is separate"),
     "In the original Adam, weight decay was added into the gradient, which "
     "means it then passed through the per-weight scaling and so decayed "
     "weights unevenly — weights with small gradients were decayed much "
     "harder. AdamW applies it directly to the weight instead "
     "[Loshchilov2019], so decay means the same thing for every weight.",
   ]),

 "ReduceLrOnPlateau": dict(
   inputs="The current metric, the best so far, patience and factor.",
   output="The possibly reduced learning rate.",
   body=[
     "Cuts the learning rate when validation stops improving for a number of "
     "epochs. The logic is that a plateau usually means the step size is now "
     "too large to settle into the minimum the optimiser is circling, so "
     "reducing it lets progress resume.",
     "It reacts to measured behaviour rather than to a fixed schedule, which "
     "is its advantage: no guess is needed in advance about when the run "
     "will stall.",
   ]),

 "CosineAnnealingLr": dict(
   inputs="The step number, the total steps, and the base rate.",
   output="The learning rate for this step.",
   body=[
     "Eases the learning rate down a cosine curve over the run:",
     ("formula", ["lr = base * 0.5 * ( 1 + cos( pi * step/total ) )"]),
     "The shape is the point. It stays near the base rate for a while, "
     "letting the run explore, then falls smoothly to nearly zero at the "
     "end, which lets the weights settle. A step-wise schedule produces a "
     "visible jolt in the loss at each drop; the cosine has no "
     "discontinuity.",
   ]),

 "ClipGradNorm": dict(
   inputs="The gradients and the maximum norm.",
   output="The scaling that was applied.",
   cite="Pascanu2013",
   body=[
     "Caps how large a single update may be [Pascanu2013]:",
     ("formula", ["total = sqrt( sum of g^2 over all parameters )",
                  "if total > maxNorm:  g *= maxNorm / total"]),
     "Scaling all gradients by one factor preserves the direction of the "
     "update and changes only its length, which is what distinguishes this "
     "from clipping each gradient independently — that would change the "
     "direction and move the weights somewhere the gradient never pointed.",
     "It matters most for the recurrent layer, where backpropagation through "
     "160 time steps occasionally produces a very large gradient. Without "
     "clipping, one such batch can undo an entire epoch of progress.",
   ]),

 # ------------------------------------------------------------------- data
 "EncodeCrop": dict(**_simple(
   "Packs a crop into bytes for the on-disk cache, as PNG data.",
   "Caching extracted line crops matters because extracting them from pages "
   "is far slower than training on them, and the same crops are read once "
   "per epoch.")),
 "DecodeCrop": dict(**_simple(
   "Unpacks a cached crop back into an array.")),
 "ReadLabelLines": dict(**_simple(
   "Reads a page's typed transcript, one line of text per line of writing.",
   "Lines marked as unusable are filtered out by the caller rather than "
   "here, so that the file on disk stays a faithful record of what was "
   "written.")),
 "ClusterRows": dict(**_simple(
   "Groups row indices that sit close together into clusters.",
   "Used to turn a set of detected rule-line rows into a smaller set of "
   "actual rules, since one printed line is several rows of dark pixels.")),
 "FindRuleLineClusters": dict(**_simple(
   "Finds the printed ruling on a form by looking for rows that are dark "
   "across most of their width.",
   "A printed rule spans the page; a line of handwriting does not reach the "
   "same proportion of the width in any single row, so the test separates "
   "them without needing to know where the rules are.")),
 "RemoveStraightFormLines": dict(**_simple(
   "Deletes those rules from the page image.",
   "Removing the ruling before line extraction stops it being mistaken for "
   "ink and dragged into the crops, where the recogniser would try to read "
   "it.")),
 "DetectPageBounds": dict(**_simple(
   "Finds the sheet inside the photograph, so that the surroundings are "
   "excluded before anything else runs.")),
 "MergeUndersizedFragments": dict(**_simple(
   "Joins detected line regions too small to be a line on their own into "
   "their neighbours.",
   "A line whose ink is sparse can be detected as two or three thin bands; "
   "merging them restores one line rather than producing several unusable "
   "crops.")),
 "MergeOrSplitToExpectedLineCount": dict(
   inputs="The detected regions and the number of transcript lines.",
   output="Regions adjusted to match that count.",
   body=[
     "The transcript says how many lines the page has, which is ground truth "
     "the detector does not have. When the detector disagrees, the closest "
     "merge or split is applied to reconcile them.",
     "This is legitimate for building **training** data, where the goal is "
     "correctly paired crops and labels and the transcript is known. It "
     "would not be legitimate at inference, where the text is what is being "
     "discovered — and it is not used there: the segmenter in segment.py has "
     "no access to a transcript and its measured line counts are therefore "
     "an honest test.",
   ]),
 "EstimateLinePitch": dict(**_simple(
   "Estimates line spacing on a page, by the same profile-peak reasoning as "
   "`LinePitch` in segment.py.")),
 "SplitLinesByPeriodicity": dict(**_simple(
   "Splits a detected block that is really several evenly spaced lines.",
   "Two touching lines of writing can be detected as one tall region; if the "
   "ink within it is periodic at the page's line pitch, it is divided at the "
   "troughs.")),
 "ExtractLinePatches": dict(**_simple(
   "Cuts the page into one image per detected line.")),
 "EncodeText": dict(
   inputs="A transcript line.", output="A list of label indices.",
   body=["Maps characters to their slots in the charset, which is the form "
         "CTC needs. Characters outside the charset are dropped, so the "
         "label and the image stay consistent with what the network can "
         "possibly output."]),

 "FindPeaks": dict(
   inputs="A profile, a minimum distance between peaks, and a minimum "
          "prominence.",
   output="The indices of accepted peaks.",
   body=[
     "Finds peaks in a 1-D profile with a prominence test, replacing the "
     "library function previously used.",
     ("sub", "What prominence means"),
     "A local maximum is not necessarily a peak: a small bump on the side of "
     "a large hill is a local maximum but not a separate feature. Prominence "
     "measures how far you must descend from a candidate before you can "
     "reach any higher ground:",
     ("formula", ["walk left and right until the profile rises above",
                  "  the candidate, or the data ends",
                  "prominence = peak - max( lowest point on the left ,",
                  "                         lowest point on the right )"]),
     "A bump on a hillside has small prominence because you barely descend "
     "before climbing higher; a genuine separate peak has large prominence. "
     "This is the standard definition and it is the behaviour the library "
     "implementation provides.",
     ("sub", "A bug worth recording"),
     "The first implementation defaulted its floor parameter to zero, which "
     "collapsed the test to 'is the value positive'. Every caller passes the "
     "**negated** profile when looking for valleys, so every value was "
     "negative and the function returned no peaks at all, where the library "
     "version returned sixteen. Defaulting the floor to negative infinity "
     "fixed it, and the two then agreed on all six test profiles.",
   ]),

 # ---------------------------------------------------------- augmentation
 "MyAffine": dict(**_simple(
   "Applies shift, rotation, scale and shear in one combined transform.",
   "Composing the transforms into a single matrix and sampling once avoids "
   "the compounding blur that applying them in sequence would cause, since "
   "each resampling step loses a little sharpness.")),
 "AugmentShear": dict(**_simple(
   "Leans the writing a little more or less than it was.",
   "Slant varies between samples of the same person's writing, so a "
   "recogniser should not depend on it; varying it in training is what "
   "teaches that.")),
 "AugmentRotate": dict(**_simple(
   "Tilts the line slightly, representing a page that was not placed quite "
   "straight.")),
 "FindPerspectiveCoeffs": dict(
   inputs="Four source corners and four destination corners.",
   output="The eight coefficients of the homography mapping one to the "
          "other.",
   body=[
     "A homography has eight unknowns, and each corner correspondence gives "
     "two equations, so four corners determine it exactly. The eight "
     "equations are assembled into a linear system and solved:",
     ("formula", ["x' = (a*x + b*y + c) / (g*x + h*y + 1)",
                  "y' = (d*x + e*y + f) / (g*x + h*y + 1)",
                  "",
                  "rearranged, each corner gives two linear equations",
                  "in the eight unknowns a..h"]),
     "The rearrangement is what makes it linear: multiplying through by the "
     "denominator turns a ratio of linear forms into a linear equation in "
     "the coefficients.",
   ]),
 "AugmentPerspective": dict(**_simple(
   "Applies such a transform, as if the page were photographed off square.",
   "This is the augmentation that most directly matches how the project's "
   "own data is collected, since the pages are photographed by hand rather "
   "than scanned.")),
 "PerspectiveJitterPoints": dict(**_simple(
   "Chooses slightly displaced corners to map to, bounding the displacement "
   "so the result stays a plausible photograph rather than an extreme "
   "distortion.")),
 "AugmentElastic": dict(**_simple(
   "Bends the strokes locally, using the smoothed random displacement field "
   "described under `Elastic` in segment.py.")),
 "AugmentLineImage": dict(**_simple(
   "Picks some subset of the augmentations at random for one crop.",
   "Applying all of them every time would push samples far from the real "
   "distribution; applying a random subset keeps most samples close to "
   "realistic while still covering the space.")),
 "AUGMENTATIONS": dict(**_simple(
   "Which augmentations are enabled and how strong each is.",
   "Collected in one place so a training run's augmentation policy is a "
   "single readable object rather than scattered constants.")),

 # -------------------------------------------------------------- datasets
 "LineImageCache": dict(**_simple(
   "Keeps extracted line crops on disk so that a rerun does not repeat the "
   "extraction.",
   "Extraction involves segmenting every page; training reads the same crops "
   "once per epoch for many epochs, so the cache is the difference between "
   "minutes and hours per run.")),
 "IAMLineDatasetRaw": dict(
   inputs="The dataset root and a cache directory.",
   output="An object exposing every line as a sample with its text and a "
          "holdout flag.",
   cite="Marti2002",
   body=[
     "Walks the public handwriting database [Marti2002], pairs each line "
     "image with its transcript, and marks which lines are held out.",
     "The holdout decision is made here, once, from a fixed seed, which is "
     "what makes it trustworthy: if the split were made independently "
     "wherever it was needed, a line could be held out for one purpose and "
     "trained on for another, and the resulting accuracy figure would be "
     "meaningless. Profiles are built only from non-holdout lines for the "
     "same reason.",
   ]),
 "TrainEvalView": dict(**_simple(
   "The same dataset with augmentation switched on or off.",
   "Validation must see unaugmented data, or the measured accuracy "
   "describes performance on distorted crops rather than on real ones.")),
 "TextCollateFn": dict(
   inputs="A list of samples.",
   output="A batch: padded images, concatenated targets, target lengths and "
          "the original texts.",
   body=[
     "Assembles a batch for the reader. Crops differ in width, so they are "
     "padded to the widest in the batch; the targets are concatenated into "
     "one flat array with their lengths recorded, which is the layout CTC "
     "expects.",
     "Flat targets plus lengths is used rather than a padded rectangle "
     "because the loss has to know exactly where each label starts and ends "
     "— padding the labels would require a padding symbol that CTC would "
     "then try to align.",
     "**Note.** This function and the writer-namer version below were both "
     "originally named `CollateFn`, so the second shadowed the first and the "
     "calls to `TextCollateFn` and `AuthorCollateFn` elsewhere in the file "
     "could never have resolved. They were renamed to match those calls.",
   ]),
 "AuthorLabeledView": dict(**_simple(
   "The same crops, labelled by writer instead of by text.",
   "Reusing one dataset for both tasks guarantees the two networks see "
   "exactly the same images, so a difference in their behaviour is a "
   "difference in the task and not in the data.")),
 "AuthorCollateFn": dict(**_simple(
   "Assembles a batch for the writer namer: padded images and one integer "
   "label each.",
   "Simpler than the reader's version because there is no sequence and "
   "therefore no alignment to describe.")),
 "AddPersonalSamples": dict(**_simple(
   "Mixes the photographed writers in with the public database.",
   "The public database supplies the volume needed to learn handwriting in "
   "general; the photographed pages supply the specific hands the project "
   "has to reproduce. Training on both is what lets a model trained mostly "
   "on other people's writing still identify these five.",
   "This function contains two unresolved references, to a segmentation "
   "module and a page-listing helper that no longer exist under those "
   "names, inherited from an earlier arrangement of the code.")),
 "TekliaLineDataset": dict(**_simple(
   "A second public source of line images, exposed through the same "
   "interface as the first.",
   "Presenting both behind one interface means the training loop does not "
   "need to know which corpus a batch came from.")),
 "MAX_SAFE_LABEL_CHARS": dict(**_simple(
   "The longest label CTC can represent given the number of output "
   "positions.",
   "CTC needs at least one position per character and an extra for each "
   "blank between repeats, so a label longer than roughly the number of "
   "positions is impossible to align and would contribute an infinite loss. "
   "Such samples are excluded rather than allowed to poison a batch.")),
 "HIDDEN_SIZE": dict(**_simple(
   "How much state the recurrent layer carries.",
   "Sets the capacity of the memory: too small and long-range context cannot "
   "be held, too large and the layer overfits and trains slowly.")),
 "WEIGHTS_DIR": dict(**_simple("Where checkpoints are written.")),
 "DEFAULT_TEXT_CACHE_DIR": dict(**_simple("Cache location for the reader's "
   "crops.")),
 "DEFAULT_AUTHOR_CACHE_DIR": dict(**_simple("Cache location for the namer's "
   "crops.")),
 "DEFAULT_AUTHOR_DATA_DIR": dict(**_simple("Where the namer's source pages "
   "live.")),
 "DEFAULT_TEXT_DATA_DIR": dict(**_simple("Where the reader's source pages "
   "live.")),

 # ------------------------------------------------- init, save, load, loop
 "InitConv": dict(
   inputs="The filter shape and a generator.",
   output="Randomly initialised filters.",
   cite="Glorot2010, He2015",
   body=[
     "Starting weights are drawn with a variance set by the number of inputs "
     "feeding each output:",
     ("formula", ["std = sqrt( 2 / fanIn ),   fanIn = inChannels*kh*kw"]),
     "The reasoning is about signal scale through depth. If weights are too "
     "small the activations shrink layer by layer until nothing distinguishes "
     "them; too large and they grow until the non-linearities saturate. "
     "Scaling by the fan-in keeps the variance of the output roughly equal "
     "to the variance of the input [Glorot2010], and the factor of 2 is the "
     "correction for the rectifier, which discards half the distribution "
     "[He2015].",
     "This matters more without a framework than with one: there is no "
     "default to fall back on, and a badly scaled start simply fails to "
     "train with no obvious error.",
   ]),
 "InitBatchNorm": dict(**_simple(
   "Scale at one, shift at zero, running statistics empty.",
   "Starting at scale 1 and shift 0 makes the layer an identity apart from "
   "the normalisation, so it begins by doing exactly the job it was added "
   "for and learns any departure from there.")),
 "InitLinear": dict(**_simple(
   "Starting weights for a fully connected layer, scaled by fan-in on the "
   "same reasoning as `InitConv`.")),
 "InitLstmDirection": dict(**_simple(
   "Starting weights for one LSTM direction, with the four gate blocks laid "
   "out in the order the forward pass expects.")),
 "InitConvBlock": dict(**_simple("One convolution block's parameters.")),
 "InitBackboneParams": dict(**_simple(
   "The whole shared backbone, built from the block initialisers.",
   "Both networks call this, so both start from the same architecture and a "
   "comparison between them is about the head and the task rather than "
   "about the feature extractor.")),
 "TrainableParams": dict(**_simple(
   "Every array the optimiser is allowed to update.",
   "Listed explicitly so that the running batch-norm statistics, which are "
   "updated by their own rule and not by gradient descent, cannot be "
   "accidentally included.")),
 "SaveCheckpoint": dict(**_simple(
   "Writes the weights and the run state to a compressed .npz.",
   "Saving the optimiser state and the step count as well as the weights is "
   "what makes a run resumable rather than only reusable.")),
 "LoadCheckpointMeta": dict(**_simple(
   "Reads back what a checkpoint records about itself: the step, the metric "
   "and the configuration.",
   "Needed to decide whether a candidate checkpoint is an improvement on the "
   "best so far.")),
 "LoadParamsInto": dict(**_simple(
   "Loads saved weights into a freshly built network, checking that the "
   "shapes match.",
   "A silent shape mismatch would be the worst possible failure, since it "
   "would produce a network that runs and returns confident nonsense.")),
 "AppendLog": dict(**_simple(
   "Adds one line to the training log.",
   "Appending rather than rewriting means an interrupted run keeps its "
   "history.")),
 "BackboneForwardTrain": dict(**_simple(
   "Runs the shared convolutional backbone, keeping every cache.")),
 "BackboneBackwardTrain": dict(**_simple(
   "Propagates gradients back down through the backbone, block by block in "
   "reverse.")),
 "EvalBackboneForward": dict(**_simple(
   "The same forward pass with no caches, for validation.",
   "Skipping the caches saves a large amount of memory, which is what allows "
   "validation to run with a larger batch than training.")),
 "IterateBatches": dict(**_simple(
   "Yields shuffled batches for one pass over a dataset.",
   "Shuffling every epoch matters: a fixed order means the optimiser sees "
   "the same sequence of gradients each epoch and can lock into a cycle "
   "rather than converging.")),

 # ------------------------------------------------------- the two trainers
 "InitTextModelParams": dict(**_simple(
   "Builds a fresh reader: backbone, LSTM stack and output layer.")),
 "BuildLstmLayerParams": dict(**_simple(
   "Assembles the per-layer weight dictionaries for the LSTM stack.")),
 "TextModelForward": dict(**_simple(
   "Backbone, height pooling, LSTM stack, linear layer, log-softmax, with "
   "every cache retained for the backward pass.")),
 "TextModelBackward": dict(**_simple(
   "The full reverse path from the CTC gradient back to the first "
   "convolution.",
   "This is the function the numerical gradient check validates end to end, "
   "which is the only way to be confident that a long hand-derived chain has "
   "no sign error in it.")),
 "EvalTextForward": dict(**_simple(
   "Forward only, for validation and for measurement.")),
 "TextNumpyBatch": dict(**_simple(
   "Converts a collated batch into the arrays the forward pass expects.")),
 "ValidateTextModel": dict(
   inputs="The model, the holdout data and a batch size.",
   output="Character and word accuracy on held-out lines.",
   body=[
     "Decodes every held-out crop and compares against its transcript with "
     "the edit-distance measures.",
     "It reports accuracy rather than loss because loss is not comparable "
     "across runs — it depends on label lengths and on the blank's "
     "distribution — whereas character accuracy means the same thing "
     "regardless. The checkpoint kept as best is chosen on this figure.",
   ]),
 "BuildPagesSplit": dict(**_simple(
   "Splits pages into training and holdout with the fixed seed.",
   "Splitting by page rather than by line is the stronger choice, since two "
   "lines from one page share pen, paper, lighting and session, and a "
   "line-level split therefore lets a near-duplicate of a training sample "
   "appear in the holdout.")),
 "BuildTekliaSplit": dict(**_simple(
   "The same split for the second corpus, which already provides its own "
   "division and so needs no further splitting.")),
 "TrainTextModel": dict(
   inputs="The data sources, hyperparameters and output paths.",
   output="None; writes checkpoints and a log.",
   body=[
     "The reader's training loop: for each batch, forward, CTC loss, "
     "backward, clip, optimiser step; at the end of each epoch, validate, and "
     "save if the validation accuracy improved.",
     "Saving only on improvement is what makes the 'best' checkpoint "
     "meaningful, and validating on data split by page is what makes the "
     "figure believable.",
     "**Measured result.** The checkpoint produced by joint training reads "
     "the project's own writers at 96.5 to 99.1 per cent character accuracy "
     "and 85 to 97 per cent word accuracy across 669 real line crops. A "
     "separate checkpoint trained from scratch reaches only 22 to 26 per "
     "cent on the same crops; both files sit in the weights directory and "
     "both load without complaint, so which one a program picks up is "
     "determined entirely by the search order in recognise.py.",
   ]),
 "InitAuthorModelParams": dict(**_simple(
   "Builds a fresh writer namer: the same backbone, pooling, and a linear "
   "layer with one output per writer.")),
 "AuthorModelForward": dict(**_simple(
   "Backbone, pooling across the line, linear layer, with caches.")),
 "AuthorModelBackward": dict(**_simple(
   "The reverse path for the namer, which is shorter than the reader's "
   "because there is no recurrent layer.")),
 "EvalAuthorForward": dict(**_simple("Forward only, for validation.")),
 "AuthorNumpyBatch": dict(**_simple(
   "Converts a collated batch into arrays for the namer.")),
 "ValidateAuthorModel": dict(**_simple(
   "Top-1 accuracy on held-out crops.",
   "Accuracy alone is reported here, but downstream measurement also looks "
   "at the margin between the true writer and the best competitor, since an "
   "identification that was nearly a mistake and one that was confident both "
   "count as correct.")),
 "TrainAuthorModel": dict(
   inputs="The data, hyperparameters and whether to stroke-normalise.",
   output="None; writes checkpoints.",
   body=[
     "The namer's training loop, structurally the same as the reader's with "
     "cross entropy in place of CTC.",
     "It is run twice. Once on raw grey crops, producing the ink-based "
     "model, which may use pen weight and ink density. Once on "
     "stroke-normalised crops, producing the shape-based model, which "
     "cannot, and so must judge letter geometry.",
     "**Measured result.** On real photographed crops the ink model reaches "
     "99.2 to 100 per cent per writer and the shape model 94.4 to 100 per "
     "cent. The shape model is the lower of the two precisely because it has "
     "been denied the easier cue, and it is the one used to judge "
     "synthesised output, which has no real ink to measure.",
   ]),
}
