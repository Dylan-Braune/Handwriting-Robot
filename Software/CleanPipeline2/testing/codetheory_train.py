"""train.py, part 1: how a network learns, and the layer gradients.

Every layer appears twice in this file, as a forward pass that also saves
what the backward pass will need, and as a backward pass that turns the
gradient of the output into the gradient of the input and of the weights.
Parts 2 and 3 cover the recurrent layers, the losses, the optimisers, the
data handling and the training loops.
"""

INTRO = [
 ("p", "This is where the networks are trained. recognise.py runs them "
       "forward; everything that works out how the weights should change "
       "is here, derived by hand and written in numpy. No automatic "
       "differentiation is used, which is the main reason this file is the "
       "largest in the project."),
 ("h", "Background: what training actually is"),
 ("p", "A network starts with random weights and gives nonsense answers. "
       "Training is a loop: run some examples through, measure how wrong "
       "the output is with a **loss function**, work out how each weight "
       "would have to change to make the loss smaller, and change it a "
       "little in that direction. Repeat for days."),
 ("p", "The quantity that says how a weight should change is the "
       "**gradient**: the rate at which the loss changes as that weight "
       "changes. If the gradient with respect to some weight is positive, "
       "increasing that weight increases the loss, so the weight should "
       "decrease. Moving every weight a small step against its gradient is "
       "**gradient descent**, and the size of that step is the **learning "
       "rate**."),
 ("eq", r"w \;\leftarrow\; w \;-\; \eta \, "
        r"\frac{\partial L}{\partial w}",
  [("w", "one weight of the network"),
   (r"\eta", "the learning rate: how big a step to take"),
   ("L", "the loss, a single number saying how wrong the output was"),
   (r"\frac{\partial L}{\partial w}",
    "the **partial derivative** of the loss with respect to that weight: "
    "how much the loss changes per unit change in the weight, holding "
    "every other weight fixed"),
   (r"\leftarrow", "assignment: the weight is replaced by the value on the "
                   "right")],
  None),
 ("h", "Background: the chain rule and backpropagation"),
 ("p", "A network has millions of weights and the loss depends on all of "
       "them through many layers, so computing each gradient separately is "
       "out of the question. What makes it feasible is the **chain rule** "
       "of calculus: if a quantity depends on another through an "
       "intermediate step, the derivatives multiply."),
 ("eq", r"\frac{\partial L}{\partial x} \;=\; "
        r"\frac{\partial L}{\partial y} \cdot "
        r"\frac{\partial y}{\partial x}",
  [("L", "the loss"),
   ("y", "a layer's output"),
   ("x", "that layer's input"),
   (r"\frac{\partial L}{\partial y}",
    "the gradient arriving from the layer above: how the loss changes with "
    "this layer's output"),
   (r"\frac{\partial y}{\partial x}",
    "the layer's own derivative: how its output changes with its input"),
   (r"\frac{\partial L}{\partial x}",
    "the gradient to pass down to the layer below")],
  None),
 ("p", "So each layer needs to know only two things: the gradient of the "
       "loss with respect to its own output, which the layer above hands "
       "it, and how its own output depends on its input and weights, which "
       "is local knowledge. It multiplies them, keeps the weight gradients, "
       "and passes the input gradient down. Starting at the loss and "
       "working backwards layer by layer computes every gradient in the "
       "network in one sweep, at about the cost of one forward pass. That "
       "is **backpropagation**."),
 ("h", "Why every function here comes in pairs"),
 ("p", "A layer's backward pass needs values from its forward pass. The "
       "gradient through a ReLU depends on which inputs were positive; the "
       "gradient through a convolution depends on what the input patches "
       "were. So each forward function returns its output together with a "
       "**cache** of exactly what its backward partner will need, and the "
       "backward function takes that cache. The pairing is the whole "
       "structure of the first part of this file."),
 ("p", "The naming convention follows from it: a function named Forward "
       "here differs from the same-named function in recognise.py only in "
       "returning that cache, and sometimes in behaving differently during "
       "training, as batch normalisation and dropout do."),
 ("h", "A note on notation"),
 ("p", "Throughout, a quantity written with a leading d, such as `dout` or "
       "`dx`, means the gradient of the loss with respect to that quantity, "
       "not a small change in it. So `dx` has exactly the same shape as x, "
       "because there is one gradient per element. This is the usual "
       "convention in hand-written backpropagation code and it is worth "
       "fixing in mind before reading any of what follows."),
]


FUNCS = {

 "Conv2dForward": dict(
   inputs=[("x", "the input, shaped (batch, channels, height, width)"),
           ("weight", "the filters"),
           ("bias", "one number per output channel, or nothing"),
           ("padding", "zeros added round the input, default 1")],
   output="a pair: the filtered result, and the cache the backward pass "
          "needs.",
   theory=[
     ("h", "What it adds to the inference version"),
     ("p", "The arithmetic is identical to Conv2d in recognise.py, "
           "including the im2col rearrangement and the strided view, both "
           "explained in that entry. The only difference is that this one "
           "keeps a cache."),
     ("h", "What is cached, and why each item is needed"),
     ("items", [
       "**The input's shape,** not the input itself. The backward pass "
       "builds a gradient array of that shape and never needs the values.",
       "**The weights,** because the input gradient is formed by "
       "multiplying the output gradient by them.",
       "**The flattened columns,** which is the significant one. The "
       "weight gradient is the output gradient multiplied by exactly these "
       "patches, so keeping them avoids rebuilding them, which would mean "
       "redoing the strided view and the reshape.",
       "**The padding and the filter and output sizes,** so the backward "
       "pass can fold the column gradient back into image shape without "
       "recomputing the geometry.",
     ]),
     ("h", "The cost of caching, and why it is accepted"),
     ("p", "The columns are the memory cost of training. A 3 by 3 filter "
           "over three channels makes each patch 27 numbers, and there is "
           "one patch per output position, so the cached columns are "
           "roughly nine times the size of the input feature map. Every "
           "convolution in the network holds its own for the whole forward "
           "pass, since none can be released until the backward pass has "
           "used it. That, rather than the weights, is what sets the "
           "largest batch size that fits in memory, and it is why training "
           "needs far more memory than inference."),
     ("note", "No bias-related value is cached. The bias gradient is just "
              "the output gradient summed over every position, which needs "
              "nothing from the forward pass, because adding a constant has "
              "a derivative of one."),
   ]),

 "Conv2dBackward": dict(
   inputs=[("dout", "the gradient of the loss with respect to this layer's "
                    "output"),
           ("cache", "what the forward pass saved")],
   output="three gradients: with respect to the input, the weights and the "
          "bias.",
   theory=[
     ("h", "Where the three formulas come from"),
     ("p", "The forward pass, in matrix form, is the flattened weights "
           "multiplied by the columns. Everything below follows from "
           "differentiating a matrix product, so it is worth stating that "
           "rule first: if Y = W X, then the gradient with respect to W is "
           "the output gradient times X transposed, and the gradient with "
           "respect to X is W transposed times the output gradient. Both "
           "the shapes and the transposes follow from requiring the result "
           "to have the same shape as the thing differentiated."),
     ("h", "The weight gradient"),
     ("eq", r"\frac{\partial L}{\partial W} \;=\; \sum_{b} "
            r"\frac{\partial L}{\partial Y_b} \; X_b^{\mathsf{T}}",
      [(r"\frac{\partial L}{\partial W}", "the gradient for the filters, "
                                          "same shape as the filters"),
       (r"\frac{\partial L}{\partial Y_b}", "the output gradient for batch "
                                            "item b, as a matrix of output "
                                            "channels by positions"),
       ("X_b", "the cached columns for that item"),
       (r"{}^{\mathsf{T}}", "matrix transpose"),
       (r"\sum_b", "summed over the batch")],
      None),
     ("p", "The sum over the batch is not an approximation or an average "
           "applied afterwards; it is what the derivative is. The same "
           "filter was applied to every item in the batch and at every "
           "position, so the loss depends on each weight through all of "
           "them, and the chain rule adds the contributions. This is the "
           "general rule for a shared parameter: wherever one value is used "
           "in several places, its gradient is the sum over those places."),
     ("h", "The input gradient, and the fold back into image shape"),
     ("eq", r"\frac{\partial L}{\partial X} \;=\; "
            r"W^{\mathsf{T}} \, \frac{\partial L}{\partial Y}",
      [(r"\frac{\partial L}{\partial X}", "the gradient with respect to the "
                                          "columns"),
       ("W", "the flattened filters"),
       (r"\frac{\partial L}{\partial Y}", "the output gradient")],
      None),
     ("p", "That gives a gradient with respect to the *columns*, which is "
           "not what is wanted: the layer below needs a gradient with "
           "respect to the image. Converting one to the other is the only "
           "part of this function that is not a matrix product, and it is "
           "the subtle part."),
     ("p", "The reason it is subtle is that the forward view was "
           "overlapping. One input pixel appears in up to nine patches, so "
           "it influenced up to nine output positions, and by the chain "
           "rule its gradient is the sum of the gradients from all of them. "
           "The loop over the nine filter offsets does exactly that: for "
           "each offset, the corresponding slice of the column gradient is "
           "*added* into the padded image gradient at the shifted position. "
           "Nine additions over whole arrays accumulate every "
           "contribution."),
     ("p", "The accumulation is the whole point, and it is why the forward "
           "trick cannot be run in reverse. The forward pass could use a "
           "strided view because reading the same memory from several "
           "places is harmless; writing to it from several places is not, "
           "since the writes would overwrite rather than accumulate. So the "
           "backward direction must be an explicit addition, and the "
           "operation is known as **col2im**, the counterpart of im2col."),
     ("h", "Finally, removing the padding"),
     ("p", "The gradient was accumulated into an array the size of the "
           "padded input, and the layer below knows nothing of the "
           "padding, so the border is discarded. That is correct rather "
           "than merely convenient: the padding was constant zeros, not "
           "data, so the gradient with respect to it has nowhere to go."),
     ("h", "The bias gradient"),
     ("eq", r"\frac{\partial L}{\partial b_o} \;=\; \sum_{b,p,q} "
            r"\frac{\partial L}{\partial Y[b,o,p,q]}",
      [(r"\frac{\partial L}{\partial b_o}", "the gradient for the bias of "
                                            "output channel o"),
       (r"\sum_{b,p,q}", "summed over the batch and both spatial axes")],
      None),
     ("p", "One bias is added to every position of its channel, so by the "
           "shared-parameter rule its gradient is the sum of the output "
           "gradient over all those positions. No cache is needed because "
           "the derivative of adding a constant is one."),
   ]),

 "BatchNorm2dForwardTrain": dict(
   inputs=[("x", "the input"), ("gamma", "the learned scale per channel"),
           ("beta", "the learned shift per channel"),
           ("runningMean", "the stored mean, updated in place"),
           ("runningVar", "the stored variance, updated in place"),
           ("momentum", "how fast the stored statistics follow the batches, "
                        "default 0.1"),
           ("eps", "the guard on the division, default 1e-5")],
   output="a pair: the normalised and rescaled output, and the cache.",
   theory=[
     ("h", "How this differs from the inference version"),
     ("p", "The inference version, in recognise.py, standardises using "
           "statistics stored in the checkpoint. This one standardises "
           "using the current batch's own mean and variance, and updates "
           "the stored ones as a side effect."),
     ("eq", r"\mu_c \;=\; \frac{1}{N} \sum_{b,p,q} x[b,c,p,q], "
            r"\qquad \sigma^2_c \;=\; \frac{1}{N} \sum_{b,p,q} "
            r"\big(x[b,c,p,q] - \mu_c\big)^2",
      [(r"\mu_c", "the mean of channel c over this batch"),
       (r"\sigma^2_c", "its variance"),
       ("N", "how many values that is: the batch size times the height "
             "times the width"),
       (r"\sum_{b,p,q}", "summed over the batch and both spatial axes, but "
                         "not over channels")],
      None),
     ("p", "The statistics are per channel and pooled over everything else, "
           "for the reason given in the inference entry: a channel is the "
           "map of where one filter responded, and that filter's typical "
           "response size is one property however many positions and "
           "examples it is measured over."),
     ("h", "Why using the batch is what makes it work"),
     ("p", "Standardising by the batch is not a convenience; it is the "
           "mechanism. The purpose of the layer is to fix the scale a layer "
           "presents to the next one, and during training that scale is "
           "changing constantly because the layers below are being updated. "
           "Measuring it fresh from the current batch tracks those changes "
           "immediately, where stored statistics would always describe an "
           "earlier state of the network."),
     ("h", "The running statistics, and why they are an exponential "
           "average"),
     ("eq", r"\hat{\mu} \;\leftarrow\; (1-m)\,\hat{\mu} \;+\; m\,\mu_c",
      [(r"\hat{\mu}", "the stored mean, kept for inference"),
       ("m", "the momentum, 0.1"),
       (r"\mu_c", "this batch's mean")],
      None),
     ("p", "This is an **exponential moving average**: each update keeps 90 "
           "percent of the old value and takes 10 percent from the new "
           "observation, so the stored figure is a weighted average over "
           "recent batches with older ones fading away. It is used rather "
           "than a plain average over all batches for two reasons. It needs "
           "no count and no accumulator, only the current value. And the "
           "fading is correct here: early batches were produced by a "
           "network whose weights were nearly random, so their statistics "
           "do not describe the trained network and should be forgotten."),
     ("h", "Why the stored variance is the unbiased estimate"),
     ("eq", r"\sigma^2_{\mathrm{unbiased}} \;=\; \sigma^2_{\mathrm{biased}} "
            r"\cdot \frac{N}{N-1}",
      [(r"\sigma^2_{\mathrm{biased}}", "the variance computed with the mean "
                                       "measured from the same data"),
       (r"\sigma^2_{\mathrm{unbiased}}", "the corrected estimate of the "
                                         "population variance"),
       ("N", "the number of values")],
      None),
     ("p", "Measuring the variance about a mean estimated from the same "
           "sample systematically underestimates it, because the sample "
           "mean is by construction the point about which the squared "
           "deviations are smallest. **Bessel's correction**, dividing by "
           "N−1 rather than N, compensates. The biased form is used to "
           "normalise this batch, since that is the quantity that actually "
           "standardises these values, while the corrected form is stored, "
           "since that is an estimate of the variance of all data. The code "
           "comment notes this matches the reference framework's behaviour, "
           "which matters because checkpoints must be interchangeable."),
     ("h", "What is cached"),
     ("p", "The standardised values, the standard deviation, the scale and "
           "the element count. Notably the input itself is not cached: the "
           "backward derivation below is written entirely in terms of the "
           "standardised values, which is both less memory and fewer "
           "operations than recovering the input."),
   ]),

 "BatchNorm2dBackward": dict(
   inputs=[("dout", "the gradient with respect to this layer's output"),
           ("cache", "what the forward pass saved")],
   output="three gradients: with respect to the input, the scale and the "
          "shift.",
   theory=[
     ("h", "Why this one is genuinely hard"),
     ("p", "Every other layer in this file has a local derivative: the "
           "output at one position depends on the input at one position, "
           "or a few. Batch normalisation does not. The mean and variance "
           "are computed from *every* value in the channel, so changing one "
           "input changes the mean, which changes the normalisation of "
           "every other value in the channel as well. Each input therefore "
           "affects its own output directly and every other output "
           "indirectly, and the gradient must account for all three "
           "routes."),
     ("h", "The three paths"),
     ("items", [
       "**Directly,** through its own standardised value.",
       "**Through the mean,** since the input contributed to the mean, and "
       "the mean is subtracted from every value in the channel.",
       "**Through the variance,** since the input contributed to the "
       "variance, and the variance divides every value in the channel.",
     ]),
     ("p", "Differentiating each path and collecting the terms gives an "
           "expression that simplifies considerably, which is the form the "
           "code uses."),
     ("eq", r"\frac{\partial L}{\partial x} \;=\; "
            r"\frac{1}{N\sigma}\left( N \, \frac{\partial L}"
            r"{\partial \hat{x}} \;-\; \sum \frac{\partial L}"
            r"{\partial \hat{x}} \;-\; \hat{x} \sum \left( "
            r"\frac{\partial L}{\partial \hat{x}} \, \hat{x} \right) "
            r"\right)",
      [(r"\frac{\partial L}{\partial x}", "the gradient to pass down"),
       (r"\frac{\partial L}{\partial \hat{x}}",
        "the gradient with respect to the standardised values, which is the "
        "incoming gradient multiplied by the learned scale"),
       (r"\hat{x}", "the cached standardised values"),
       (r"\sigma", "the cached standard deviation"),
       ("N", "the number of values per channel"),
       (r"\sum", "summed over the batch and both spatial axes, per "
                 "channel")],
      None),
     ("p", "The three terms correspond to the three paths, and each has a "
           "readable meaning. The first is the direct effect. The second "
           "subtracts the average gradient over the channel, which is the "
           "correction for the mean: it ensures the gradients sum to zero, "
           "so the layer below cannot be pushed in a direction that would "
           "merely shift the whole channel, since such a shift would be "
           "normalised away anyway. The third subtracts the component "
           "aligned with the standardised values themselves, which is the "
           "correction for the variance: it removes any push that would "
           "merely scale the channel, for the same reason."),
     ("p", "So the layer passes down only the part of the gradient that is "
           "not a pure shift or a pure rescaling of the channel. That is "
           "the deeper statement of what batch normalisation does: it makes "
           "the network's behaviour independent of the scale and offset of "
           "each channel, and the backward pass correspondingly refuses to "
           "pass down gradients that would only change those."),
     ("h", "The scale and shift gradients"),
     ("eq", r"\frac{\partial L}{\partial \gamma} \;=\; \sum "
            r"\frac{\partial L}{\partial y} \, \hat{x}, \qquad "
            r"\frac{\partial L}{\partial \beta} \;=\; \sum "
            r"\frac{\partial L}{\partial y}",
      [(r"\gamma,\, \beta", "the learned scale and shift"),
       (r"\frac{\partial L}{\partial y}", "the incoming gradient"),
       (r"\hat{x}", "the standardised values"),
       (r"\sum", "summed over everything but the channel axis")],
      None),
     ("p", "These two are straightforward by comparison. The output is the "
           "scale times the standardised value plus the shift, so the "
           "derivative with respect to the scale is the standardised value "
           "and with respect to the shift is one, each summed over every "
           "position the parameter was used at, by the shared-parameter "
           "rule."),
     ("note", "The code comment attributes the simplified form to the "
              "standard derivation taught in the Stanford CS231n course "
              "notes, which is where this arrangement of the terms is "
              "usually given. The project's numerical gradient check exists "
              "largely to confirm that this function in particular is "
              "right, since it is the one most easily got wrong and an "
              "error in it produces slow or unstable training rather than "
              "an obvious failure."),
   ]),

 "ReluForward": dict(
   inputs=[("x", "any array")],
   output="a pair: the rectified array, and a mask of which entries were "
          "positive.",
   theory=[
     ("h", "What is cached and why it is cheap"),
     ("p", "The forward arithmetic is the same clamping as in recognise.py. "
           "What it adds is a boolean mask recording which inputs were "
           "positive, which is all the backward pass needs. A boolean array "
           "is one byte per element against eight for a double, so this is "
           "the cheapest cache of any layer, which matters because ReLU "
           "appears after every convolution."),
     ("note", "The mask could be derived from the output instead, since the "
              "output is positive exactly where the input was. Caching it "
              "explicitly costs a little memory and keeps the backward pass "
              "independent of what the caller did with the output, which "
              "would otherwise be a hidden requirement that the output not "
              "be modified in place."),
   ]),

 "ReluBackward": dict(
   inputs=[("dout", "the gradient with respect to the output"),
           ("cache", "the positivity mask")],
   output="the gradient with respect to the input.",
   theory=[
     ("h", "The derivative"),
     ("eq", r"\frac{\partial L}{\partial x} \;=\; "
            r"\frac{\partial L}{\partial y} \cdot "
            r"\begin{cases} 1 & x > 0 \\ 0 & x \le 0 \end{cases}",
      [(r"\frac{\partial L}{\partial x}", "the gradient to pass down"),
       (r"\frac{\partial L}{\partial y}", "the gradient arriving"),
       ("x", "the original input, consulted only through the cached mask")],
      None),
     ("p", "ReLU is the identity where the input was positive and constant "
           "zero where it was not, so its derivative is one or zero "
           "accordingly. The backward pass is therefore the incoming "
           "gradient with the inactive positions zeroed: the gradient "
           "passes unchanged through the units that fired and stops dead at "
           "the ones that did not."),
     ("h", "Why this is the property that makes deep networks trainable"),
     ("p", "A derivative of exactly 1 on the active path is what "
           "distinguishes ReLU from the smooth activations it replaced. "
           "The sigmoid's derivative is at most 0.25 and much smaller when "
           "saturated, so a gradient crossing twelve sigmoid layers is "
           "multiplied by twelve such factors and arrives at the first "
           "layer effectively zero. That is the vanishing gradient problem, "
           "and it is why networks of this depth were impractical before "
           "ReLU. With a factor of exactly 1 the gradient reaches the first "
           "layer at full strength."),
     ("note", "The derivative at exactly zero does not exist, since the "
              "function has a corner there. Taking it as zero is "
              "conventional and the choice has no practical consequence: an "
              "input of exactly zero is a measure-zero event in floating "
              "point, and either choice gives a valid subgradient."),
   ]),

 "MaxPool2dForward": dict(
   inputs=[("x", "the input"), ("kernel", "the window size, default 2"),
           ("stride", "the step between windows, default 2")],
   output="a pair: the pooled output, and the cache recording which element "
          "won each window.",
   theory=[
     ("h", "What must be cached, and why"),
     ("p", "The forward arithmetic is as in recognise.py. What training "
           "requires in addition is the **position** of the winner in each "
           "window, not merely its value. The backward pass must send the "
           "gradient to the element that was passed through, and only the "
           "position identifies it."),
     ("p", "The position is stored as an index into the flattened window, a "
           "number from 0 to 3 for a 2 by 2 window, rather than as a pair of "
           "coordinates. A single index is one array instead of two and can "
           "be used directly by the indexed-write operation the backward "
           "pass uses to place the gradient."),
     ("note", "This is why the forward pass flattens the window axes and "
              "takes the maximum by finding the winning index and then "
              "reading it, instead of simply taking a maximum over two "
              "axes. The inference version does the latter, which is "
              "simpler; here the index is needed anyway, so the value is "
              "obtained by reading it."),
   ]),

 "MaxPool2dBackward": dict(
   inputs=[("dout", "the gradient with respect to the pooled output"),
           ("cache", "the winning indices and the geometry")],
   output="the gradient with respect to the input.",
   theory=[
     ("h", "The routing rule"),
     ("p", "Max pooling passes one element of each window through "
           "unchanged, so its derivative with respect to that element is "
           "one and with respect to every other element of the window is "
           "zero. The gradient is therefore routed entirely to the winner "
           "and the rest of the window receives nothing."),
     ("eq", r"\frac{\partial L}{\partial x[b,c,i,j]} \;=\; "
            r"\begin{cases} \dfrac{\partial L}{\partial y[b,c,p,q]} & "
            r"(i,j) \text{ was the maximum of window } (p,q) \\[6pt] "
            r"0 & \text{otherwise} \end{cases}",
      [("x[b,c,i,j]", "an input element"),
       ("y[b,c,p,q]", "the pooled output of the window containing it"),
       (r"(p,q)", "which window")],
      None),
     ("p", "An input that lost its window has no influence on the output at "
           "all, so changing it slightly changes nothing and its gradient "
           "is genuinely zero. That is a real consequence of pooling rather "
           "than an approximation: only a quarter of the inputs to a 2 by 2 "
           "pooling receive any gradient on a given example, which is part "
           "of why the layers below a pooling train more slowly."),
     ("h", "Why no accumulation is needed here"),
     ("p", "The code comment notes that the stride equals the kernel, so "
           "the windows do not overlap and each input belongs to exactly "
           "one window. The gradient blocks can therefore be written "
           "straight into position rather than added. With overlapping "
           "windows an input could win several, and the contributions would "
           "have to be accumulated as they are in the convolution's "
           "backward pass."),
     ("note", "The transpose before the final reshape interleaves the "
              "window axes back with the output axes, so that the blocks "
              "land in the right places. Getting this wrong produces a "
              "gradient that is correct in magnitude and scrambled in "
              "position, which trains to something, slowly and wrongly, "
              "without any error being raised. It is exactly the kind of "
              "fault the numerical gradient check exists to catch."),
   ]),

 "AdaptiveMaxPoolHeightForward": dict(
   inputs=[("x", "the input")],
   output="a pair: the height collapsed to one row by taking the maximum, "
          "and the winning row indices.",
   theory=[
     ("h", "What is cached"),
     ("p", "Which row won in each column of each channel. The reasoning is "
           "the same as for ordinary max pooling: the gradient goes "
           "entirely to the element that was passed through, so its "
           "position must be known."),
     ("p", "This is the layer that converts the image into a sequence, as "
           "described in the inference entry, so it is also the point where "
           "the gradient converts back: the sequence gradient coming from "
           "the recurrent layers becomes an image gradient here, placed "
           "into the rows that won."),
   ]),

 "AdaptiveMaxPoolHeightBackward": dict(
   inputs=[("dout", "the gradient with respect to the collapsed output"),
           ("cache", "the input shape and the winning rows")],
   output="the gradient with respect to the full-height input.",
   theory=[
     ("h", "What it does"),
     ("p", "Creates a zero array of the original shape and writes the "
           "incoming gradient into the winning row of each column. Every "
           "other row gets nothing."),
     ("h", "What that means for training"),
     ("p", "Of the sixteen rows the feature map has at this point, exactly "
           "one per column receives any gradient on a given example. So "
           "each column of each channel trains from one row at a time, "
           "though which row varies between examples and over the course of "
           "training as the weights change. The effect is that the layer "
           "below learns to make its strongest response correct, rather "
           "than learning anything about the rows that lose, which is the "
           "behaviour a maximum should produce: the network is being asked "
           "whether a feature appears anywhere in the column, so only the "
           "best evidence matters."),
   ]),

 "AdaptiveAvgPoolWidthForward": dict(
   inputs=[("x", "the input")],
   output="a pair: the average along axis 2, and the length averaged over.",
   theory=[
     ("h", "Why the cache is a single number"),
     ("p", "An average needs nothing from its input to be differentiated, "
           "only how many values it was taken over, which is one integer. "
           "That is the cheapest cache in the file, and the contrast with "
           "max pooling is instructive: a maximum depends on *which* value "
           "won and so must remember it, whereas an average treats every "
           "value identically and so has nothing to remember."),
   ]),

 "AdaptiveAvgPoolWidthBackward": dict(
   inputs=[("dout", "the gradient with respect to the pooled output"),
           ("cache", "how many values were averaged")],
   output="the gradient spread evenly back over those values.",
   theory=[
     ("h", "The derivative"),
     ("eq", r"y \;=\; \frac{1}{N}\sum_{k} x_k \quad\Longrightarrow\quad "
            r"\frac{\partial y}{\partial x_k} \;=\; \frac{1}{N} "
            r"\quad\text{for every } k",
      [("y", "the pooled output"),
       ("x_k", "the k-th input averaged over"),
       ("N", "how many there were"),
       (r"\Longrightarrow", "implies")],
      None),
     ("p", "Each input contributes one N-th of the average, so each "
           "receives one N-th of the gradient. The backward pass is "
           "therefore the incoming gradient repeated across the axis and "
           "divided by the length."),
     ("h", "The contrast with maximum pooling, and what it means for "
           "training"),
     ("p", "Where max pooling routes the whole gradient to one element and "
           "nothing to the rest, average pooling spreads it thinly over all "
           "of them. Every position therefore receives gradient on every "
           "example, which is why the writer namer's backbone trains more "
           "evenly than the reader's. It is also the right behaviour for "
           "what that layer means: the namer asks how prevalent a feature "
           "is across the line, so every position genuinely contributes to "
           "the answer and every position should be told how it "
           "contributed."),
   ]),

 "LinearForward": dict(
   inputs=[("x", "the input, with features along the last axis"),
           ("weight", "the weight matrix"), ("bias", "the bias, or nothing")],
   output="a pair: the result, and the cache holding the input and the "
          "weights.",
   theory=[
     ("h", "What is cached and why both items are needed"),
     ("p", "Both the input and the weights, because the backward pass needs "
           "each for a different gradient: the weight gradient is formed "
           "from the input and the input gradient is formed from the "
           "weights. This is the only layer here that caches its full input "
           "values, which is affordable because a linear layer sits at the "
           "end of the network where the data is small, a vector of a few "
           "hundred numbers rather than a feature map."),
   ]),

 "LinearBackward": dict(
   inputs=[("dout", "the gradient with respect to the output"),
           ("cache", "the input and the weights")],
   output="three gradients: with respect to the input, the weights and the "
          "bias.",
   theory=[
     ("h", "The three gradients"),
     ("eq", r"\frac{\partial L}{\partial W} \;=\; "
            r"\frac{\partial L}{\partial Y}^{\mathsf{T}} X, \qquad "
            r"\frac{\partial L}{\partial X} \;=\; "
            r"\frac{\partial L}{\partial Y} \, W, \qquad "
            r"\frac{\partial L}{\partial b} \;=\; \sum_{\text{rows}} "
            r"\frac{\partial L}{\partial Y}",
      [(r"\frac{\partial L}{\partial Y}", "the incoming gradient, one row "
                                          "per example"),
       ("X", "the cached input, one row per example"),
       ("W", "the weight matrix, one row per output"),
       (r"{}^{\mathsf{T}}", "transpose"),
       (r"\sum_{\text{rows}}", "summed over the examples")],
      None),
     ("p", "All three follow from differentiating a matrix product, as in "
           "the convolution. The transposes are determined by the shapes: "
           "there is only one arrangement of each product that yields an "
           "array the same shape as the quantity being differentiated, "
           "which is a useful check when deriving such expressions by "
           "hand."),
     ("h", "Why the leading axes are flattened"),
     ("p", "The code comment explains the one wrinkle. The input may carry "
           "more than a batch axis before its feature axis: the reader's "
           "text head applies this layer to a sequence, shaped time by "
           "batch by features, so there are two leading axes. The weight "
           "gradient must sum over every one of those positions, since the "
           "same weights were used at each."),
     ("p", "Flattening every leading axis into one reduces any such case to "
           "the two-dimensional form, after which the single matrix product "
           "performs the sum automatically. The input gradient needs no "
           "flattening, because it has the same leading axes as the output "
           "and the product acts only on the last; it is reshaped back to "
           "the input's shape at the end. This is what lets one "
           "implementation serve both the namer's single vector and the "
           "reader's sequence."),
   ]),

 "DropoutForward": dict(
   inputs=[("x", "the input"), ("p", "the fraction to drop"),
           ("training", "whether this is a training pass"),
           ("rng", "the seeded random generator")],
   output="a pair: the input with some entries zeroed and the rest scaled "
          "up, and the mask; or the input unchanged outside training.",
   theory=[
     ("h", "What dropout is for"),
     ("p", "**Overfitting** is when a network learns the training examples "
           "rather than the pattern in them, so it performs well on what it "
           "has seen and poorly on anything new. One way it happens is that "
           "the network comes to rely on a particular combination of "
           "features that happens to be reliable in the training data. "
           "**Dropout** \\cite{srivastava2014} prevents that by randomly "
           "removing a fraction of the values at each training step, so no "
           "feature can be depended on: it may not be there. The network is "
           "forced to spread its evidence across several features, which "
           "generalises better."),
     ("h", "Why the survivors are scaled up"),
     ("eq", r"\mathrm{mask} \;=\; \frac{[\,u \ge p\,]}{1 - p}, "
            r"\qquad u \sim \mathrm{Uniform}(0,1)",
      [(r"\mathrm{mask}", "what the input is multiplied by, elementwise"),
       ("u", "a random number drawn uniformly between 0 and 1, one per "
             "element"),
       ("p", "the drop probability"),
       (r"[\,u \ge p\,]", "1 when the element survives and 0 when it is "
                          "dropped"),
       (r"\frac{1}{1-p}", "the compensating scale applied to the "
                          "survivors")],
      None),
     ("p", "Dropping a fraction p of the values reduces the sum reaching "
           "the next layer by that fraction, so without compensation the "
           "layer below would see a smaller signal during training than at "
           "inference, and everything calibrated for one would be wrong for "
           "the other. Dividing the survivors by 1−p restores the expected "
           "total: a fraction 1−p of the values survive and each is "
           "multiplied by 1/(1−p), so the expectation is unchanged."),
     ("h", "Why the scaling is done here rather than at inference"),
     ("p", "The compensation could instead be applied by scaling everything "
           "down by 1−p at inference, which is how dropout was originally "
           "described. Putting it in the training path instead is called "
           "**inverted dropout**, and it is now standard for a practical "
           "reason: inference then requires no dropout-related code at all. "
           "The layer simply vanishes outside training, which is exactly "
           "what recognise.py does, where the gap in the parameter "
           "numbering is the only trace of it."),
     ("note", "Outside training the input is returned unchanged with no "
              "mask. The backward pass accepts that absent mask and passes "
              "the gradient through untouched, which keeps the two halves "
              "consistent without either needing to know whether the other "
              "is in training mode."),
   ]),

 "DropoutBackward": dict(
   inputs=[("dout", "the gradient with respect to the output"),
           ("cache", "the mask, or nothing")],
   output="the gradient with respect to the input.",
   theory=[
     ("h", "The derivative"),
     ("p", "Dropout multiplies elementwise by a mask, and the derivative of "
           "multiplying by a constant is that constant, so the backward "
           "pass multiplies by the same mask. The mask includes the scaling "
           "factor, so the gradient is scaled by it too, which is "
           "necessary: the forward pass amplified the survivors, so their "
           "influence on the loss is amplified by the same factor."),
     ("p", "A dropped element receives exactly zero gradient, which is "
           "correct rather than approximate. It genuinely had no effect on "
           "the output for this example, so the loss does not depend on it "
           "at all this step. It will receive gradient on steps where it "
           "survives, which is how a unit that is dropped half the time "
           "still trains, from half the examples."),
   ]),

 "ConvBlockForwardTrain": dict(
   inputs=[("x", "the input feature maps"),
           ("params", "the parameter dictionary"),
           ("prefix", "the block's name prefix"),
           ("nLayers", "how many groups the block contains")],
   output="a pair: the output, and a list of per-group caches.",
   theory=[
     ("h", "What it adds to the inference version"),
     ("p", "The same convolution, normalise, activate repetition, with the "
           "batch-statistics version of the normalisation and with every "
           "group's three caches collected into a list."),
     ("h", "Why the caches are kept in order"),
     ("p", "The list is built front to back as the groups are applied, and "
           "the backward pass walks it in reverse. That correspondence is "
           "the whole structure of backpropagation through a sequence of "
           "layers: the gradient must pass through them in the opposite "
           "order to the data, so the last cache created is the first one "
           "needed."),
     ("note", "The running statistics are updated in place inside the "
              "parameter dictionary rather than returned. They are not "
              "parameters in the usual sense, since no gradient is computed "
              "for them and no optimiser touches them, but they must be "
              "saved in the checkpoint, so they live alongside the "
              "parameters and are excluded from the trainable set by name."),
   ]),

 "ConvBlockBackwardTrain": dict(
   inputs=[("dout", "the gradient with respect to the block's output"),
           ("caches", "the per-group caches from the forward pass"),
           ("prefix", "the block's name prefix"),
           ("nLayers", "how many groups")],
   output="a pair: the gradient with respect to the block's input, and a "
          "dictionary of weight gradients keyed as the parameters are.",
   theory=[
     ("h", "The reversal"),
     ("p", "The groups are walked in reverse, and within each group the "
           "three layers are walked in reverse too: activation, then "
           "normalisation, then convolution. Each step takes the gradient "
           "from the step before it, so the gradient flows back through the "
           "block in exactly the opposite order to the way the data flowed "
           "forward."),
     ("h", "Why the gradients are keyed by parameter name"),
     ("p", "The weight gradients are collected into a dictionary using the "
           "same names the parameters themselves use. That is what lets the "
           "optimiser be written without any knowledge of the network's "
           "structure: it iterates over the parameters and looks up each "
           "one's gradient by name. The names are therefore a contract "
           "between three separate things, the initialiser, the backward "
           "pass and the checkpoint format, and a mismatch in any of them "
           "fails with a missing key rather than silently training the "
           "wrong array."),
     ("note", "The variable holding the gradient is reassigned at each step "
              "rather than being given a new name per layer. That reads "
              "oddly but is exactly right: there is only ever one gradient "
              "travelling backwards at a time, and naming its successive "
              "values separately would suggest they coexist."),
   ]),
}


#the rest of the chapter, in order
for _part in ("codetheory_train2", "codetheory_train3"):
    try:
        _m = __import__(_part)
    except ModuleNotFoundError:
        continue
    FUNCS.update(_m.FUNCS)
