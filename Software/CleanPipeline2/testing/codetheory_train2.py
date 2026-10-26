"""train.py, part 2: the recurrent gradient, the losses, and the optimisers.

Backpropagation through time, the CTC forward-backward algorithm, and the
rules that turn a gradient into a weight update.
"""

FUNCS = {

 "LstmForwardTrain": dict(
   inputs=[("x", "the input sequence, shaped (time, batch, features)"),
           ("weightIh", "the input weights, all four gates stacked"),
           ("weightHh", "the recurrent weights"),
           ("biasIh", "the input bias"), ("biasHh", "the recurrent bias"),
           ("hiddenSize", "the width of the hidden and cell state")],
   output="a pair: the output at every step, and a cache holding every "
          "intermediate value at every step.",
   theory=[
     ("h", "What it adds to the inference version"),
     ("p", "The same gate arithmetic as LstmForward in recognise.py, with "
           "one addition that dominates its memory use: at every time step "
           "it records the input, the two incoming states, all four gate "
           "activations and the squashed cell value."),
     ("h", "Why so much must be kept"),
     ("p", "The backward pass needs each of them, and the reason is that "
           "the LSTM's updates are products of the gates with the states. "
           "Differentiating a product requires both factors, so the "
           "gradient with respect to the forget gate needs the previous "
           "cell value and the gradient with respect to the previous cell "
           "needs the forget gate. None of it can be recomputed cheaply "
           "either, because recomputing the state at step t requires "
           "replaying every step before it."),
     ("eq", r"\mathrm{memory} \;\approx\; T \times B \times "
            r"\big(\,\mathrm{inputSize} + 6H\,\big)",
      [(r"\mathrm{memory}", "the number of values cached, per LSTM "
                            "direction per layer"),
       ("T", "the sequence length, 160 columns here"),
       ("B", "the batch size"),
       (r"\mathrm{inputSize}", "the width of the input at that layer"),
       ("H", "the hidden size, 256"),
       ("6H", "the six vectors kept per step: two incoming states, four "
              "gates, and the squashed cell, less the one that is "
              "recomputed")],
      None),
     ("p", "With two layers and two directions that is eight such caches. "
           "At a hidden size of 256 and 160 steps it comes to a few million "
           "values per batch item, which is why the text model trains with "
           "small batches. This is the characteristic cost of "
           "**backpropagation through time**: the whole history of the "
           "sequence must be held in memory until the backward pass has "
           "consumed it."),
     ("note", "The step caches are kept in a plain list indexed by time, "
              "built forwards and consumed backwards, which is the same "
              "arrangement as the convolution block's cache list and for "
              "the same reason."),
   ]),

 "LstmBackward": dict(
   inputs=[("dOutputs", "the gradient with respect to the output at every "
                        "step"),
           ("cache", "the per-step values and the weights")],
   output="five gradients: with respect to the input sequence, both weight "
          "matrices and both biases.",
   theory=[
     ("h", "Backpropagation through time"),
     ("p", "An LSTM applies the same weights at every step, so the loss "
           "depends on each weight through all 160 steps. By the "
           "shared-parameter rule the weight gradient is the sum over the "
           "steps, and the state gradients must be carried backwards "
           "through them, because the state at step t influenced the output "
           "at every later step. The procedure is **backpropagation through "
           "time**: walk the steps in reverse, accumulating weight "
           "gradients and carrying state gradients."),
     ("h", "The two gradients carried backwards"),
     ("p", "At each step two quantities arrive from the step after it: the "
           "gradient with respect to the hidden state and the gradient with "
           "respect to the cell state. The hidden gradient is combined with "
           "the gradient arriving directly from the layer above, because "
           "the hidden state at step t is used twice, as that step's output "
           "and as the next step's input."),
     ("eq", r"\mathrm{d}h_t \;=\; \frac{\partial L}{\partial y_t} \;+\; "
            r"\mathrm{d}h_{t+1}^{\mathrm{carried}}",
      [(r"\mathrm{d}h_t", "the total gradient with respect to the hidden "
                          "state at step t"),
       (r"\frac{\partial L}{\partial y_t}", "the gradient arriving from the "
                                            "layer above for this step's "
                                            "output"),
       (r"\mathrm{d}h_{t+1}^{\mathrm{carried}}", "the gradient the "
                                                 "following step computed "
                                                 "with respect to its "
                                                 "incoming hidden state")],
      None),
     ("p", "Adding the two is the single most important line of the "
           "function. Omitting the carried term computes the gradient as "
           "though each step's output were independent, which trains "
           "something that works tolerably on short sequences and never "
           "learns any long-range dependency, which is the entire reason an "
           "LSTM was chosen."),
     ("h", "Differentiating the two state updates"),
     ("p", "The updates are h = o · tanh(c) and c = f · c_prev + i · g, and "
           "everything else follows from differentiating them. Each gate's "
           "gradient is the gradient of whatever it multiplied, and then "
           "each is passed back through its own squashing function."),
     ("eq", r"\begin{aligned} "
            r"\mathrm{d}o &= \mathrm{d}h \odot \tanh(c_t) \\[2pt] "
            r"\mathrm{d}c &= \mathrm{d}c^{\mathrm{carried}} + "
            r"\mathrm{d}h \odot o \odot \big(1 - \tanh^2(c_t)\big) \\[2pt] "
            r"\mathrm{d}f &= \mathrm{d}c \odot c_{t-1} \\[2pt] "
            r"\mathrm{d}i &= \mathrm{d}c \odot g \\[2pt] "
            r"\mathrm{d}g &= \mathrm{d}c \odot i \\[2pt] "
            r"\mathrm{d}c_{t-1} &= \mathrm{d}c \odot f "
            r"\end{aligned}",
      [(r"\mathrm{d}o,\,\mathrm{d}f,\,\mathrm{d}i,\,\mathrm{d}g",
        "the gradients with respect to the four gate outputs"),
       (r"\mathrm{d}c", "the gradient with respect to this step's cell "
                        "state"),
       (r"\mathrm{d}c_{t-1}", "the gradient carried back to the previous "
                              "cell state"),
       ("c_t,\\,c_{t-1}", "the cell states at this and the previous step"),
       (r"\odot", "elementwise multiplication"),
       (r"1-\tanh^2", "the derivative of the hyperbolic tangent")],
      None),
     ("p", "Every line is the product rule: the gradient with respect to "
           "one factor is the incoming gradient times the other factor. The "
           "cell gradient has two contributions because the cell is used "
           "twice, revealed through the hidden state at this step and "
           "carried forward to the next, which is the same double-use "
           "argument as for the hidden state."),
     ("h", "The gradient path that makes long memory possible"),
     ("p", "The last line is worth isolating. The gradient carried back to "
           "the previous cell state is the current cell gradient multiplied "
           "by the forget gate, and by nothing else: no weight matrix is "
           "applied along this path. So if the forget gate is near 1, the "
           "gradient passes backwards through that step essentially "
           "unchanged, and it can do so for many steps in succession "
           "without shrinking."),
     ("p", "That is the precise sense in which the LSTM solves the "
           "vanishing gradient problem. A plain recurrent layer multiplies "
           "the carried gradient by its recurrent weight matrix at every "
           "step, so after a hundred steps it has been multiplied by that "
           "matrix a hundred times and is either negligible or enormous. "
           "The LSTM's cell path multiplies only by gate values the network "
           "itself controls, and it can learn to set them near 1 where "
           "information must survive."),
     ("h", "Through the squashing functions"),
     ("p", "Each gate gradient is then passed back through its own "
           "activation to reach the pre-activation value, which is what the "
           "weights actually produced. For the three sigmoid gates the "
           "derivative is the output times one minus the output, and for "
           "the candidate it is one minus the output squared. Both are "
           "written in terms of the output rather than the input, which is "
           "why the cache holds the gate activations and not the "
           "pre-activations."),
     ("h", "The weight gradients"),
     ("p", "The four pre-activation gradients are concatenated back into "
           "the stacked layout the weights use, and the weight gradients "
           "are then matrix products of that stack with the step's input "
           "and previous hidden state. They are accumulated across steps "
           "with addition rather than assignment, which is the "
           "shared-parameter sum. Both biases receive the same gradient, "
           "since both were added to the same quantity, which is the "
           "backward counterpart of the inference note that only their sum "
           "has any effect."),
     ("note", "The input gradient is written per step into a preallocated "
              "array rather than appended, because the steps are visited in "
              "reverse and the result must come back in forward order. "
              "Appending and reversing would also work; writing by index "
              "makes the ordering explicit at the point it matters."),
   ]),

 "BidirectionalLstmForwardTrain": dict(
   inputs=[("x", "the input sequence"),
           ("paramsFwd", "the forward direction's parameters"),
           ("paramsBwd", "the backward direction's parameters"),
           ("hiddenSize", "the width per direction")],
   output="a pair: the two directions concatenated, and a cache holding "
          "both directions' caches.",
   theory=[
     ("h", "What it does"),
     ("p", "Runs both directions as in the inference version and keeps both "
           "caches. Each direction is a complete independent LSTM with its "
           "own weights, so each has its own full set of per-step values, "
           "and the memory cost is doubled."),
     ("note", "The backward direction is run on a reversed view of the "
              "input, which is free, and its output is reversed again to "
              "align positions. The reversals must be undone consistently "
              "in the backward pass too, which is the one real "
              "opportunity for error in an otherwise mechanical function."),
   ]),

 "BidirectionalLstmBackward": dict(
   inputs=[("dout", "the gradient with respect to the concatenated output"),
           ("cache", "both directions' caches and the hidden size")],
   output="the gradient with respect to the input, and both directions' "
          "weight gradients.",
   theory=[
     ("h", "Splitting the gradient"),
     ("p", "The forward pass concatenated the two directions along the "
           "feature axis, so the incoming gradient is split at the same "
           "point: the first half belongs to the forward direction and the "
           "second to the backward one. Concatenation and splitting are "
           "exact inverses, so no arithmetic is involved."),
     ("h", "Why the two input gradients are added"),
     ("p", "Both directions read the *same* input sequence, so the input is "
           "a shared quantity and by the shared-parameter rule its gradient "
           "is the sum of the two directions' contributions. This is easy "
           "to get wrong by treating them as separate inputs and returning "
           "only one, which halves the gradient reaching the convolutional "
           "backbone and makes it train at half rate with no visible "
           "symptom."),
     ("h", "The reversal, undone"),
     ("p", "The backward direction's gradient must be reversed before being "
           "handed to its LSTM backward pass and reversed again afterwards, "
           "mirroring the two reversals of the forward pass. The invariant "
           "to hold on to is that every array crossing the boundary of this "
           "function is in forward time order, and the reversed order "
           "exists only inside it."),
   ]),

 "MultilayerBidirectionalLstmForwardTrain": dict(
   inputs=[("x", "the input sequence"),
           ("layerParams", "a list of per-layer parameters"),
           ("hiddenSize", "the width per direction")],
   output="a pair: the final output, and the list of per-layer caches.",
   theory=[
     ("h", "What it does"),
     ("p", "Applies each bidirectional layer in turn, collecting the "
           "caches. The same build-forwards, consume-backwards arrangement "
           "as the convolution block, one level up: a list of layer caches, "
           "each of which holds two direction caches, each of which holds a "
           "list of step caches."),
   ]),

 "MultilayerBidirectionalLstmBackward": dict(
   inputs=[("dout", "the gradient with respect to the final output"),
           ("caches", "the per-layer caches")],
   output="the gradient with respect to the input, and every layer's weight "
          "gradients.",
   theory=[
     ("h", "What it does"),
     ("p", "Walks the layers in reverse, passing each layer's input "
           "gradient to the layer below and collecting the weight "
           "gradients. There is no arithmetic of its own; it is the "
           "sequencing that makes the chain rule hold across the stack."),
     ("note", "This completes the recurrent half. The gradient arriving "
              "here came from the text head, and the gradient leaving goes "
              "to the height-collapsing pool, which converts it back from a "
              "sequence gradient into an image gradient. Those two "
              "conversions, forward and backward, are what join the "
              "convolutional and recurrent halves of the reader."),
   ]),

 "LogSumExp": dict(
   inputs=[("a", "one array of log-space values"),
           ("b", "another of the same shape")],
   output="the logarithm of the sum of their exponentials, computed without "
          "overflow.",
   theory=[
     ("h", "What it is for"),
     ("p", "CTC adds probabilities together, and the probabilities are held "
           "as logarithms to avoid underflow, as explained under "
           "LogSoftmax. But a logarithm turns a *product* into a sum, not a "
           "*sum* into anything simple: adding two probabilities held as "
           "logarithms requires this function."),
     ("eq", r"\log(e^a + e^b) \;=\; m + \log\!\big(e^{a-m} + "
            r"e^{b-m}\big), \qquad m = \max(a,b)",
      [("a,\\,b", "the two log-space values being combined"),
       ("m", "the larger of the two, factored out"),
       ("e^{a-m}", "at most 1, since a is at most m, so it cannot "
                   "overflow")],
      None),
     ("p", "The identity follows from factoring the larger exponential out "
           "of the sum and taking the logarithm. Its value is numerical: "
           "the exponentials in the rearranged form are at most 1, so "
           "nothing can overflow, while a direct computation would "
           "exponentiate a log probability of −800 and get zero, or of a "
           "large positive number and get infinity."),
     ("h", "The infinity case, and why it needs handling"),
     ("p", "A forbidden transition is represented by negative infinity, "
           "meaning zero probability. When both arguments are that, the "
           "maximum is negative infinity and the rearranged expression "
           "computes infinity minus infinity, which is not a number. The "
           "answer should be negative infinity: the sum of two "
           "impossibilities is still impossible."),
     ("p", "So the result is selected explicitly in that case. The code "
           "comment notes that numpy evaluates both branches of the "
           "selection eagerly, so the invalid arithmetic still happens and "
           "still warns even though its result is discarded, and the "
           "warning is suppressed for that reason. Suppressing a warning is "
           "usually a bad sign; here it is suppressing a warning about a "
           "value that has already been correctly replaced, and the "
           "suppression is scoped narrowly to this one expression."),
     ("note", "Preserving negative infinity exactly, rather than using a "
              "large negative number, matters for correctness: an "
              "impossible path must contribute exactly nothing, and a "
              "finite stand-in contributes a tiny amount that accumulates "
              "over 160 steps and can make a forbidden alignment the "
              "winner."),
   ]),

 "CtcLoss": dict(
   inputs=[("rawLogits", "the network's raw output, shaped (time, batch, "
                         "classes)"),
           ("targets", "the label sequences, one per batch item")],
   output="a pair: the mean loss over the batch, and the gradient with "
          "respect to the raw output.",
   theory=[
     ("h", "What is being computed"),
     ("p", "This is the heart of training the reader. The label says what "
           "was written and not where, so the loss cannot compare the "
           "output to a target column by column. Instead CTC "
           "\\cite{graves2006} defines the probability of the target text "
           "as the total probability of *every* alignment of it to the "
           "columns, and the loss is the negative logarithm of that."),
     ("eq", r"L \;=\; -\log P(\mathbf{y} \mid \mathbf{x}) \;=\; "
            r"-\log \!\!\sum_{\pi \,\in\, \mathcal{B}^{-1}(\mathbf{y})} "
            r"\prod_{t=1}^{T} p_t\big(\pi_t\big)",
      [("L", "the loss for one example"),
       (r"\mathbf{y}", "the target text"),
       (r"\mathbf{x}", "the input image"),
       (r"\pi", "one alignment: a choice of label for each column"),
       (r"\mathcal{B}^{-1}(\mathbf{y})", "the set of every alignment that "
                                         "collapses to the target text, by "
                                         "the merge-then-remove-blanks rule "
                                         "described under DecodeCtc"),
       ("p_t(\\pi_t)", "the network's probability for the label this "
                       "alignment chose at column t"),
       ("T", "the number of columns")],
      "graves2006"),
     ("p", "The negative logarithm is the standard form for a loss built "
           "from a probability: maximising the probability is minimising "
           "its negative logarithm, and the logarithm turns the product "
           "over columns into a sum, which is both numerically stable and "
           "what makes the gradient manageable."),
     ("h", "Why the sum cannot be computed directly"),
     ("p", "The number of alignments is enormous. A 40-character label "
           "across 160 columns has on the order of 10 to the power 40 valid "
           "alignments, so enumerating them is as impossible as "
           "enumerating the seam paths in segment.py, and for the same "
           "reason. The remedy is also the same: dynamic programming over a "
           "quantity that can be built up column by column."),
     ("h", "The forward variable"),
     ("eq", r"\alpha_t(s) \;=\; \Big(\alpha_{t-1}(s) + "
            r"\alpha_{t-1}(s-1) + \alpha_{t-1}(s-2)\Big) \cdot "
            r"p_t\big(\mathrm{ext}[s]\big)",
      [(r"\alpha_t(s)", "the total probability of every alignment that "
                        "reaches state s of the extended label sequence by "
                        "column t"),
       (r"\mathrm{ext}[s]", "the label at position s of the extended "
                            "sequence, built as blank, label, blank, "
                            "label, … as described under CtcForcedAlign"),
       ("p_t(\\cdot)", "the network's probability for that label at column "
                       "t"),
       (r"\alpha_{t-1}(s-2)", "the skip term, excluded when the skip is "
                              "forbidden")],
      "graves2006"),
     ("p", "The recurrence is the same topology as the forced alignment: "
           "the same three transitions and the same restriction that a skip "
           "may not land on a blank or repeat the label two states back. "
           "The one crucial difference is that alignment took the maximum "
           "of the three arrivals and this takes the **sum**."),
     ("h", "Why sum here and maximum there"),
     ("p", "This is the distinction worth being clear about. Alignment "
           "wants one particular layout, the most probable, so it takes the "
           "best arrival. Training wants the total probability of the text "
           "however it is laid out, so it adds the arrivals: every "
           "alignment that reaches this state contributes, and their "
           "probabilities add because they are mutually exclusive ways of "
           "the same thing happening."),
     ("p", "Summing is also what makes CTC trainable at all. If training "
           "used the maximum it would be committing to one alignment and "
           "pushing the network toward it, which would be arbitrary early "
           "on when the network has no idea where anything is. Summing lets "
           "the network be rewarded for any layout that spells the right "
           "text, and the alignment emerges from training rather than being "
           "imposed."),
     ("p", "In log space the sum is performed by LogSumExp, applied twice "
           "to combine the three terms, which is why that function exists."),
     ("h", "The backward variable"),
     ("eq", r"\beta_t(s) \;=\; \sum_{s' \in \{s,\,s+1,\,s+2\}} "
            r"\beta_{t+1}(s') \cdot p_{t+1}\big(\mathrm{ext}[s']\big)",
      [(r"\beta_t(s)", "the total probability of completing the label from "
                       "state s at column t onwards"),
       ("s'", "the states reachable from s, the mirror of the forward "
              "transitions"),
       ("p_{t+1}(\\cdot)", "the probability of the label emitted on "
                           "arrival")],
      "graves2006"),
     ("p", "The forward variable measures the probability of everything "
           "*before* a point and the backward variable the probability of "
           "everything *after* it. Both are needed because the gradient "
           "requires knowing, for each column and each label, the total "
           "probability of all the complete alignments passing through that "
           "point, and that is the product of the two."),
     ("h", "The total probability, and the loss"),
     ("p", "The total probability of the label is the forward variable at "
           "the final column, in either of the two states a valid alignment "
           "may end in, the last character or the trailing blank, summed. "
           "The loss is its negative logarithm."),
     ("h", "The gradient, and why it takes this remarkably simple form"),
     ("eq", r"\frac{\partial L}{\partial z_t(k)} \;=\; "
            r"p_t(k) \;-\; \frac{1}{P(\mathbf{y}\mid\mathbf{x})} "
            r"\!\!\sum_{s \,:\, \mathrm{ext}[s] = k} \!\! "
            r"\alpha_t(s)\,\beta_t(s)",
      [(r"\frac{\partial L}{\partial z_t(k)}", "the gradient with respect "
                                               "to the raw score for class "
                                               "k at column t"),
       ("p_t(k)", "the network's current probability for that class, from "
                  "the softmax"),
       (r"\alpha_t(s)\beta_t(s)", "the total probability of all complete "
                                  "alignments passing through state s at "
                                  "column t"),
       (r"\sum_{s : \mathrm{ext}[s]=k}", "summed over every state of the "
                                         "extended sequence that emits "
                                         "class k, since a character may "
                                         "appear several times"),
       (r"P(\mathbf{y}\mid\mathbf{x})", "the total probability of the "
                                        "label, which normalises the "
                                        "product into a proportion")],
      "graves2006"),
     ("p", "The second term is called the **occupation**: among all the "
           "alignments that spell the target, the proportion that have "
           "class k at column t. So the gradient is the difference between "
           "what the network currently predicts for that class and what it "
           "*should* predict given the target. Where the network already "
           "agrees with the occupation the gradient is zero; where it "
           "predicts too much of a class the gradient is positive and that "
           "score is pushed down."),
     ("p", "That this comes out as a simple difference is the same "
           "simplification that makes the cross-entropy gradient a "
           "difference, and it arises for the same reason: the softmax and "
           "the logarithm in the loss are inverse operations, so "
           "differentiating their composition cancels the awkward terms. It "
           "is why the gradient is computed with respect to the raw logits "
           "rather than with respect to the log probabilities, even though "
           "the loss was defined on the latter: taking the whole softmax "
           "and loss together is far simpler than either separately."),
     ("h", "Why there is a loop over the batch"),
     ("p", "Each example has its own label and so its own length of "
           "extended sequence, so the dynamic programming tables are "
           "different shapes and cannot be computed as one array. The loop "
           "over the batch is therefore unavoidable, and inside it the "
           "loops over time and over states remain in Python too, which "
           "makes this the slowest function in the file. The small loop "
           "over states checking the skip restriction could be vectorised "
           "by precomputing the mask once per example instead of once per "
           "column, which is the obvious remaining optimisation."),
     ("note", "Both the loss and the gradient are divided by the batch "
              "size, so the loss is a mean over examples rather than a "
              "total. That makes it comparable between batch sizes, which "
              "matters because the learning rate is tuned against the "
              "gradient's magnitude: without the division, doubling the "
              "batch size would double the gradient and effectively double "
              "the learning rate."),
   ]),

 "CrossEntropyLoss": dict(
   inputs=[("logits", "the raw scores, one row per example"),
           ("targetIdx", "the correct class index for each example")],
   output="a pair: the mean loss, and the gradient with respect to the "
          "logits.",
   theory=[
     ("h", "What it is for"),
     ("p", "The loss for the writer namer, which is an ordinary "
           "classification: one correct answer per example, and its "
           "position is known, so none of the CTC machinery is needed."),
     ("eq", r"L \;=\; -\frac{1}{B}\sum_{b=1}^{B} "
            r"\log p_b\big(c_b\big)",
      [("L", "the mean loss over the batch"),
       ("B", "the batch size"),
       ("c_b", "the correct class for example b"),
       ("p_b(c_b)", "the probability the network assigned to that class, "
                    "from the softmax")],
      None),
     ("h", "Why the negative logarithm of the correct class"),
     ("p", "**Cross-entropy** for a single correct answer reduces to the "
           "negative logarithm of the probability assigned to it. Its shape "
           "is what makes it a good loss: a probability of 1 gives a loss "
           "of 0, a probability of 0.5 gives about 0.69, and as the "
           "probability approaches zero the loss grows without bound. So "
           "being confidently wrong is punished far more than being "
           "uncertain, which is the behaviour wanted from a classifier that "
           "reports confidences."),
     ("h", "The gradient"),
     ("eq", r"\frac{\partial L}{\partial z_b(k)} \;=\; \frac{1}{B}"
            r"\Big( p_b(k) - [\,k = c_b\,] \Big)",
      [(r"\frac{\partial L}{\partial z_b(k)}", "the gradient with respect "
                                               "to the raw score for class "
                                               "k of example b"),
       ("p_b(k)", "the predicted probability for that class"),
       (r"[\,k = c_b\,]", "1 for the correct class and 0 for every other"),
       (r"\frac{1}{B}", "the batch mean, matching the loss")],
      None),
     ("p", "The gradient is the predicted distribution minus the target "
           "distribution, the latter being 1 on the correct class and 0 "
           "elsewhere. The code expresses it exactly that way: copy the "
           "probabilities and subtract one from the correct class. Every "
           "wrong class is pushed down in proportion to how much it was "
           "predicted, and the right one is pushed up in proportion to how "
           "much it was missed."),
     ("p", "The same cancellation as in CTC produces this. Differentiating "
           "the softmax alone gives a full matrix of terms, and "
           "differentiating the logarithm gives a reciprocal, and composing "
           "them cancels almost everything. This is the standard reason "
           "softmax and cross-entropy are implemented as one unit rather "
           "than two layers."),
     ("note", "The log-sum-exp stabilisation is applied here too, the "
              "maximum being subtracted before exponentiating, for the "
              "reason given under LogSoftmax. It costs one pass and removes "
              "any possibility of overflow on a large logit."),
   ]),

 "RMSprop": dict(
   output="an optimiser object holding the parameters and a running average "
          "of squared gradients.",
   theory=[
     ("h", "The problem plain gradient descent has"),
     ("p", "A single learning rate must suit every weight, and it cannot. "
           "Different weights have gradients of wildly different "
           "magnitudes, by orders of magnitude between layers and between a "
           "bias and a convolution weight. A rate small enough to be stable "
           "for the largest gradients is far too small for the smallest, so "
           "some weights barely move while others oscillate."),
     ("h", "The idea: divide by the typical gradient size"),
     ("p", "**RMSprop** \\cite{tieleman2012} keeps, for each weight, a "
           "running average of its squared gradient, and divides the step "
           "by the square root of that. A weight whose gradients are "
           "habitually large is divided by a large number and takes a small "
           "step; one with habitually small gradients takes a comparatively "
           "large one. The effect is that every weight moves at a "
           "comparable rate regardless of its gradient's scale."),
     ("eq", r"v \;\leftarrow\; \alpha\,v + (1-\alpha)\,g^2, \qquad "
            r"w \;\leftarrow\; w - \eta\,\frac{g}{\sqrt{v} + \epsilon}",
      [("v", "the running average of the squared gradient, per weight"),
       (r"\alpha", "how much of the old average to keep, 0.99 here"),
       ("g", "the current gradient for this weight"),
       ("w", "the weight"),
       (r"\eta", "the learning rate"),
       (r"\epsilon", "a small constant, 1e-8, so a weight with no gradient "
                     "history does not divide by zero")],
      None),
     ("p", "The square root is what makes the division dimensionally "
           "sensible: the running average is of squared gradients, so its "
           "square root is in the same units as a gradient, and the ratio "
           "is dimensionless. The name is from that root-mean-square. At "
           "0.99 the average covers roughly the last hundred steps, so it "
           "describes the gradient's recent typical size rather than this "
           "step's value, which is what makes the scaling stable."),
     ("note", "Weight decay here is added to the gradient, which is the "
              "classical formulation. The entry for AdamW explains why that "
              "interacts badly with the per-weight scaling and what the "
              "alternative is."),
   ]),

 "AdamW": dict(
   output="an optimiser object holding the parameters and two running "
          "averages per weight.",
   theory=[
     ("h", "What it adds to RMSprop"),
     ("p", "**Adam** \\cite{kingma2015} keeps a running average of the "
           "gradient itself as well as of its square, and uses the "
           "smoothed gradient for the step direction. The first is "
           "**momentum**: averaging successive gradients cancels the part "
           "that fluctuates between batches and keeps the part that is "
           "consistent, so the step follows the persistent downhill "
           "direction rather than the noise of whichever examples happened "
           "to be in this batch."),
     ("eq", r"\begin{aligned} "
            r"m &\leftarrow \beta_1 m + (1-\beta_1)\,g \\[2pt] "
            r"v &\leftarrow \beta_2 v + (1-\beta_2)\,g^2 \\[2pt] "
            r"\hat{m} &= \frac{m}{1-\beta_1^{\,t}}, \qquad "
            r"\hat{v} = \frac{v}{1-\beta_2^{\,t}} \\[2pt] "
            r"w &\leftarrow w - \eta\left(\frac{\hat{m}}"
            r"{\sqrt{\hat{v}}+\epsilon} + \lambda w\right) "
            r"\end{aligned}",
      [("m", "the running average of the gradient, the first moment"),
       ("v", "the running average of its square, the second moment"),
       (r"\beta_1", "the decay for the first, 0.9: an average over roughly "
                    "ten steps"),
       (r"\beta_2", "the decay for the second, 0.999: roughly a thousand "
                    "steps"),
       ("t", "the step number, counted from 1"),
       (r"\hat{m},\,\hat{v}", "the bias-corrected averages"),
       (r"\lambda", "the weight decay coefficient"),
       (r"\eta", "the learning rate")],
      "kingma2015"),
     ("h", "Why the averages must be bias-corrected"),
     ("p", "Both averages start at zero, which is not a neutral starting "
           "point but a strong claim that the gradient is zero. After one "
           "step the first moment is only one tenth of the observed "
           "gradient, and the second is one thousandth of its square, so "
           "early steps would be far too small and, worse, badly "
           "misproportioned between the two, since they are biased by "
           "different amounts."),
     ("p", "The correction divides each by one minus its decay raised to "
           "the step number, which is exactly the total weight the average "
           "has accumulated so far. At step 1 the first moment is divided "
           "by 0.1, recovering the full gradient; by step 100 the divisor "
           "is nearly 1 and the correction fades away. So the optimiser "
           "behaves sensibly from its very first step, which matters "
           "because the first steps are taken from random weights where "
           "large corrections are most needed."),
     ("h", "Why the W, and what it fixes"),
     ("p", "**Weight decay** shrinks every weight slightly at each step, "
           "which discourages large weights and reduces overfitting. "
           "Classically it is implemented by adding a multiple of the "
           "weight to its gradient, which for plain gradient descent is "
           "exactly equivalent."),
     ("p", "With Adam it is not equivalent, and that is the point of "
           "AdamW \\cite{loshchilov2019}. A decay term added to the "
           "gradient passes through the division by the square-root average "
           "like any other gradient, so a weight with large gradients has "
           "its decay divided down and is barely regularised, while one "
           "with small gradients is decayed heavily. The strength of the "
           "regularisation therefore varies per weight in a way nobody "
           "intended, and it depends on the gradient history rather than on "
           "the weight."),
     ("p", "The fix is to apply the decay to the weight directly, outside "
           "the scaled step, which is the separate term in the update "
           "above. Every weight is then shrunk by the same proportion, "
           "which is what weight decay is supposed to mean. This is called "
           "**decoupled** weight decay, and the W in the name stands for "
           "it."),
     ("note", "Writing the decay inside the same parenthesis as the scaled "
              "step means it is multiplied by the learning rate, so the "
              "effective decay changes as the learning rate is annealed. "
              "The original formulation decouples it from the learning rate "
              "as well. The difference matters mainly late in training, "
              "when the rate has fallen and the decay falls with it."),
   ]),

 "ReduceLrOnPlateau": dict(
   output="a scheduler object that lowers the learning rate when progress "
          "stalls.",
   theory=[
     ("h", "Why the learning rate should change during training"),
     ("p", "A large rate is right early on, when the weights are far from "
           "any good configuration and large steps make rapid progress. It "
           "is wrong later, when the weights are close and a large step "
           "jumps past the minimum and bounces around it. So the rate "
           "should start large and end small, and a **scheduler** is what "
           "changes it."),
     ("h", "This scheduler's rule"),
     ("p", "It watches a metric, usually the validation loss, and remembers "
           "the best value seen. Each epoch that fails to beat the best "
           "counts as a bad epoch; once the bad epochs exceed the patience, "
           "the learning rate is multiplied by the factor and the count "
           "resets. A new best resets it too."),
     ("h", "Why it reacts rather than following a schedule"),
     ("p", "The advantage over a fixed schedule is that it needs no "
           "estimate of how long training will take. It lowers the rate "
           "exactly when progress has stopped, which is the evidence that "
           "the current rate is now too large, whatever epoch that happens "
           "to be. The disadvantage is that it only reacts after the "
           "patience has elapsed, so several epochs are spent at a rate "
           "already known to be too large."),
     ("h", "Why the patience is needed"),
     ("p", "Validation loss is noisy, because it is measured on a finite "
           "set and the weights change between epochs. A single epoch "
           "failing to improve means little. Requiring five consecutive "
           "failures distinguishes a genuine plateau from the ordinary "
           "fluctuation of a metric that is still improving overall. Too "
           "little patience lowers the rate on noise and ends training "
           "early at a rate too small to make progress."),
     ("note", "The floor on the learning rate stops the repeated halving "
              "from reaching zero, at which point training would continue "
              "indefinitely without changing anything. It is a way of "
              "bounding wasted time rather than a property of the method."),
   ]),

 "CosineAnnealingLr": dict(
   output="a scheduler object that eases the learning rate down a cosine "
          "curve.",
   theory=[
     ("h", "The schedule"),
     ("eq", r"\eta_t \;=\; \frac{\eta_0}{2}\left(1 + "
            r"\cos\!\left(\frac{\pi t}{T}\right)\right)",
      [(r"\eta_t", "the learning rate at epoch t"),
       (r"\eta_0", "the initial learning rate"),
       ("t", "the current epoch"),
       ("T", "the total number of epochs planned"),
       (r"\cos", "the cosine, which runs from 1 at the start to −1 at the "
                 "end over half a period")],
      None),
     ("p", "At epoch 0 the cosine is 1 and the rate is the full initial "
           "value; at the final epoch it is −1 and the rate is zero, with a "
           "smooth decrease between."),
     ("h", "Why a cosine rather than a straight line"),
     ("p", "The shape is what recommends it. The cosine is flat at both "
           "ends and steepest in the middle, so training spends a while at "
           "nearly the full rate, then drops briskly, then spends a while "
           "at a very small rate. Both plateaux are useful: the first makes "
           "rapid early progress, and the second lets the weights settle "
           "precisely into a minimum rather than continuing to step past "
           "it. A linear decay spends equal time at every rate, including "
           "the middling ones that are neither fast nor precise."),
     ("h", "The trade-off against the plateau scheduler"),
     ("p", "This one requires knowing the total number of epochs in "
           "advance, since the schedule is defined against it. In exchange "
           "it never wastes epochs at a rate that has become too large, "
           "because it lowers the rate continuously rather than waiting for "
           "evidence. Having both available lets the choice be made per "
           "training run according to whether the length is known."),
   ]),

 "ClipGradNorm": dict(
   inputs=[("grads", "the gradient dictionary, modified in place"),
           ("maxNorm", "the greatest allowed total magnitude, default 5.0")],
   output="the total gradient magnitude before any clipping.",
   theory=[
     ("h", "The problem it addresses"),
     ("p", "Recurrent networks occasionally produce an enormous gradient. A "
           "particular batch, often one with an unusually long or badly "
           "labelled line, can give gradients orders of magnitude larger "
           "than usual, and a single step with such a gradient moves the "
           "weights so far that everything learned so far is destroyed. "
           "Training then has to recover from near-random weights, and "
           "hours of work are lost to one batch. This is the **exploding "
           "gradient** problem."),
     ("h", "The remedy"),
     ("eq", r"\|g\| \;=\; \sqrt{\sum_{k} \sum_{i} g_k[i]^2}, "
            r"\qquad g \;\leftarrow\; g \cdot "
            r"\min\!\left(1,\; \frac{\tau}{\|g\| + \delta}\right)",
      [(r"\|g\|", "the total gradient norm: every parameter's gradient "
                  "treated as one long vector"),
       ("g_k[i]", "element i of the gradient for parameter k"),
       (r"\tau", "the maximum norm allowed, 5.0"),
       (r"\delta", "a tiny constant preventing a division by zero"),
       (r"\min(1,\cdot)", "so the scaling only ever shrinks")],
      "pascanu2013"),
     ("p", "If the total magnitude exceeds the limit, every gradient is "
           "multiplied by the same factor so that the total comes to "
           "exactly the limit. The published remedy for exploding gradients "
           "is precisely this \\cite{pascanu2013}."),
     ("h", "Why the whole network is scaled together"),
     ("p", "This is the part worth understanding. The norm is computed over "
           "every parameter at once, and one factor is applied to all of "
           "them, rather than each being clipped separately. The reason is "
           "that the gradient's *direction* is the useful information: it "
           "says which way the loss decreases. Scaling everything by one "
           "factor preserves that direction exactly and changes only the "
           "distance travelled. Clipping each parameter independently would "
           "shorten some components and not others, which rotates the "
           "direction into one that is not a descent direction at all, so "
           "the step might increase the loss."),
     ("h", "Why the norm is returned"),
     ("p", "The function returns the magnitude it measured, before any "
           "clipping, so the training loop can log it. That number is one "
           "of the most informative diagnostics available during training: "
           "a norm that is steady and comfortably below the limit means "
           "training is healthy, one that is clipped on most batches means "
           "the learning rate or the limit is wrong, and a sudden spike "
           "identifies the batch that caused it. Without the return value, "
           "clipping would silently conceal the very instability it is "
           "protecting against."),
   ]),
}
