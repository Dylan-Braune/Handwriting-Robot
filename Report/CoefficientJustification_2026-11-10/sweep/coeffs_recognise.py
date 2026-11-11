#u23539195 coefficient justification: the stroke-normalisation and network-input coefficients
import probes as P


def C(**kw):
    return kw


SN_RUN = dict(page=P.PAGE,lineIdx=P.LINE,author=P.AUTHOR,
              authorWeights=P.AUTHOR_WEIGHTS)


ENTRIES = [

C(key="pen_width_xh",
  title="Pen width the writer classifier's input is redrawn at",
  symbol=r"$w_{\text{pen}}$",
  subsystem="Writer classification",
  module="recognise", where="StrokeNormalize",
  default=0.17, values=[0.09,0.13,0.17,0.21,0.25],
  patches=lambda v: [("recognise","PenWidthXh = 0.17","PenWidthXh = %s" % v)],
  task="stroke_norm", params=SN_RUN,
  panel=dict(kind="image",key="norm",aspect=7.0),
  metrics=["pred","trueProb","correct","inkFrac"],
  trendMetrics=["trueProb","inkFrac"],
  source=r"Ink width is itself a writer-discriminative feature: Bulacu and "
         r"Schomaker build writer identification on edge- and contour-based "
         r"distributions that stroke thickness enters directly "
         r"\cite{bulacu2007}, and Brink \textit{et al.} show pen pressure and "
         r"stroke width carry writer identity \cite{brink2012}. Thinning to a "
         r"centreline is Zhang and Suen \cite{zhang1984}, with the correctness "
         r"analysis of Lam \textit{et al.} \cite{lam1992}.",
  reason=r"That literature is the \emph{reason} for this coefficient rather than "
         r"an endorsement of it. Because stroke width is writer-discriminative, "
         r"a classifier trained on raw crops can learn the pen and the paper "
         r"instead of the hand: in this project each writer's pages were "
         r"photographed in one sitting with one pen, so pen width is almost a "
         r"label. Thinning every crop to a one-pixel centreline and redrawing it "
         r"at a single fixed width deletes that shortcut and forces the network "
         r"onto letter shape, which is what the system needs, since the "
         r"synthesised output is drawn by one machine with one pen and would "
         r"otherwise never match any writer. $w_{\text{pen}}$ is then only the "
         r"width that reinstates enough ink for the convolutions to see, and it "
         r"is set in x-heights so it is the same relative weight for every "
         r"writer.",
  understanding=r"The redrawn width is "
                r"$\mathrm{clip}(\mathrm{round}(w_{\text{pen}}\,x_h),2,9)$ "
                r"pixels, forced odd so the pen sits centred on the centreline. "
                r"Too thin and the strokes break up under the $2\times2$ "
                r"pooling; too thick and neighbouring strokes merge and the "
                r"counters of \texttt{a}, \texttt{e} and \texttt{o} fill in, "
                r"which destroys exactly the shape information the normalisation "
                r"was meant to preserve.",
  alternatives=r"Not normalising at all, which leaves the pen-width shortcut in "
               r"place; normalising by a measured per-writer stroke width, which "
               r"preserves the shortcut deliberately; or feeding the skeleton "
               r"itself, which is one pixel wide and is largely destroyed by the "
               r"two pooling stages.",
  worked=r"On the probe crop $x_h = 42$ px, so "
         r"$0.17 \times 42 = 7.1 \to 7$, already odd, and the centreline is "
         r"dilated with a $7\times7$ square. At $w_{\text{pen}} = 0.09$ the "
         r"width is $4 \to 5$ and at $0.25$ it is $9$, the clip limit.",
  observed=r"On the probe line the classifier named Owen correctly at every "
           r"one of the five widths, with probability between $0.9995$ and "
           r"$1.000$. Two things follow, and both are worth saying plainly. "
           r"First, the quantisation means the five values do not give five "
           r"distinct tests: $\mathrm{round}(0.09 \times 42) = 4 \to 5$ and "
           r"$\mathrm{round}(0.13 \times 42) = 5$ both redraw at $5$ px, and "
           r"$0.21$ and $0.25$ both reach the clip at $9$ px, so the sweep "
           r"really exercises three widths, $5$, $7$ and $9$ px, and the "
           r"recorded ink fractions ($0.089$, $0.128$, $0.166$) come in matching "
           r"pairs. Second, on this line the decision is not sensitive to the "
           r"width at all. That is a genuine result and it means the sweep does "
           r"\emph{not} on its own justify $0.17$ over $0.13$ or $0.21$: the "
           r"justification rests on the physical argument above and on the "
           r"train--test match recorded in the checkpoint name, not on this "
           r"measurement. Establishing a best width would need the classifier "
           r"retrained at each one and scored over the held-out set.",
  notes=r"The deployed checkpoint is named \texttt{author\_shape\_11w\_pw170}, "
        r"that is pen width $0.170$, which records that the weights were trained "
        r"with this same value. Changing $w_{\text{pen}}$ at inference therefore "
        r"creates a train--test mismatch as well as a visual change, and the "
        r"sweep measures both at once. The checkpoint actually loaded here is "
        r"\texttt{author\_shape\_10new\_weights.npz}, because the one the "
        r"submitted code names is not present on disk.",
  ),

C(key="pen_width_clip_hi",
  title="Upper clip on the redrawn pen width, in pixels",
  symbol=r"$w_{\max}$",
  subsystem="Writer classification",
  module="recognise", where="StrokeNormalize",
  default=9, values=[3,5,9,15,25],
  patches=lambda v: [("recognise",
      "w = int(np.clip(round(penWidthXh*xh),2,9))",
      "w = int(np.clip(round(penWidthXh*xh),2,%s))" % v)],
  task="stroke_norm", params=SN_RUN,
  panel=dict(kind="image",key="norm",aspect=7.0),
  metrics=["pred","trueProb","correct","inkFrac"],
  trendMetrics=["trueProb","inkFrac"],
  source=r"Clipping a derived kernel size to a safe range is ordinary defensive "
         r"practice in morphological pipelines \cite{haralick1987}.",
  reason=r"The redrawn width is derived from a measured x-height, and that "
         r"measurement can fail: a crop with no usable core band falls back to "
         r"$\max(8,h/3)$, which on a tall crop gives a large x-height and hence "
         r"an absurd pen. The clip bounds the damage in both directions. The "
         r"upper limit of $9$ px is about one fifth of a typical $42$ px "
         r"x-height, beyond which the counters of letters begin to fill.",
  understanding=r"For the probe crop the natural width is $7$ px, so the clip "
                r"only binds once $w_{\max}$ is set below that. The sweep "
                r"therefore shows the clip taking over: at $w_{\max} = 3$ and "
                r"$5$ the pen is forced thinner than the coefficient asks for.",
  alternatives=r"Rejecting the crop outright when the band measurement fails, "
               r"which loses a line that may well be readable, or clipping the "
               r"x-height rather than the width.",
  worked=r"$\mathrm{round}(0.17 \times 42) = 7$, inside $[2,9]$, so the clip is "
         r"inactive at the submitted setting on this crop.",
  ),

C(key="input_height",
  title="Height of the fixed canvas both networks read",
  symbol=r"$H_{\text{in}}$",
  subsystem="Network input",
  module="segment", where="InputHeight",
  default=64, values=[32,48,64,96,128],
  patches=lambda v: [("segment","InputHeight,InputWidth = 64,640",
                      "InputHeight,InputWidth = %s,640" % v)],
  task="page_segment",
  params=dict(page=P.PAGE,lines=[P.LINE],score=True,
              textWeights=P.TEXT_WEIGHTS,scoreLines=6),
  panel=dict(kind="image",key="crop%d" % P.LINE,aspect=7.0),
  metrics=["charAcc","nLines"],
  trendMetrics=["charAcc"],
  source=r"A fixed input height with width scaled to preserve aspect ratio is "
         r"the CRNN convention \cite[p.~2298]{shi2017}; Puigcerver "
         r"\cite{puigcerver2017} uses the same arrangement for handwritten line "
         r"recognition, and the IAM line images this was trained against are "
         r"distributed in that form \cite{marti2002}.",
  reason=r"The height must be large enough that an x-height survives the two "
         r"$2\times2$ pooling stages of the backbone. At $H_{\text{in}} = 64$ a "
         r"line crop is placed so the writing occupies roughly the middle third, "
         r"so the x-height is about $20$ px, which is $5$ px after pooling: "
         r"enough for a $3\times3$ convolution to see a letter's shape. Halving "
         r"the height would leave about $2$ px, which is not. The value is also "
         r"the one the published architectures use, so the layer sizes and the "
         r"receptive field arithmetic carry over.",
  understanding=r"This is \textbf{not a free parameter of the deployed system}. "
                r"The trained checkpoints were produced at $64 \times 640$, and "
                r"the convolution stack has no fully connected layer over height "
                r"only because the height is collapsed by an adaptive max "
                r"pool, so a different height does run, but against weights that "
                r"never saw it. The sweep therefore measures a train--test "
                r"mismatch, and the accuracy it reports away from $64$ is a "
                r"statement about that mismatch, not about the merits of another "
                r"input size. Choosing a different height honestly would require "
                r"retraining.",
  alternatives=r"Variable-height input with per-batch padding, which complicates "
               r"batching for no gain on single-line crops; or $32$ px, used by "
               r"the original CRNN for scene text \cite{shi2017}, which is too "
               r"small for handwriting with ascenders and descenders in the same "
               r"crop.",
  worked=r"$64$ px with the writing in the middle third gives about $21$ px of "
         r"x-height; after \texttt{MaxPool2d} twice this is $21/4 \approx 5$ px "
         r"entering stage three.",
  ),

C(key="input_width",
  title="Width of the fixed canvas, which sets the number of CTC columns",
  symbol=r"$W_{\text{in}}$",
  subsystem="Network input",
  module="segment", where="InputWidth",
  default=640, values=[320,480,640,960,1280],
  patches=lambda v: [("segment","InputHeight,InputWidth = 64,640",
                      "InputHeight,InputWidth = 64,%s" % v)],
  task="page_segment",
  params=dict(page=P.PAGE,lines=[P.LINE],score=True,
              textWeights=P.TEXT_WEIGHTS,scoreLines=6),
  panel=dict(kind="image",key="crop%d" % P.LINE,aspect=7.0),
  metrics=["charAcc","nLines"],
  trendMetrics=["charAcc"],
  source=r"CTC requires at least as many input columns as label symbols, and "
         r"strictly more where a label repeats a character, because a blank must "
         r"separate the repeat \cite[pp.~370--372]{graves2006}.",
  reason=r"The backbone pools width twice, so the number of CTC time steps is "
         r"$T = W_{\text{in}}/4$. That is the hard constraint: a line of $n$ "
         r"characters needs $T \ge n$ at the very least, and in practice several "
         r"columns per character for the alignment to be stable. At "
         r"$W_{\text{in}} = 640$, $T = 160$ columns, and the code records the "
         r"resulting safety margin explicitly as "
         r"$\texttt{MAX\_SAFE\_LABEL\_CHARS} = \lfloor 160 \times 0.75 \rfloor "
         r"= 120$ characters, which is longer than any line on these pages. "
         r"Width also sets how much the crop is shrunk: a wide crop fitted to a "
         r"narrow canvas loses resolution.",
  understanding=r"Because the canvas fit preserves aspect ratio and is flush "
                r"left, a shorter canvas does not squash the writing; it cuts it "
                r"off. So reducing $W_{\text{in}}$ truncates the line, and the "
                r"accuracy drop in the sweep is mostly text falling off the right "
                r"edge, not a subtler representational effect. As with the "
                r"height, the weights were trained at $640$.",
  alternatives=r"Per-line variable width with bucketed batching, the usual choice "
               r"for large-scale training, or splitting long lines into "
               r"overlapping windows and stitching the transcripts.",
  worked=r"The probe crop is $2170 \times 156$ px, aspect $13.9$. The canvas fit "
         r"scale is $\min(64/156,\,640/2170) = \min(0.410,0.295) = 0.295$, so "
         r"width binds and the crop lands $640 \times 46$ px, leaving $18$ rows "
         r"of padding.",
  ),

C(key="lstm_hidden",
  title="Hidden size of each LSTM direction",
  symbol=r"$n_h$",
  subsystem="Network input",
  module="recognise", where="PaperCRNNNumpy",
  default=256, values=[64,128,256,384,512],
  patches=lambda v: [("recognise",
      "def __init__(self,checkpointPath=None,hiddenSize=256):",
      "def __init__(self,checkpointPath=None,hiddenSize=%s):" % v)],
  task="page_segment",
  params=dict(page=P.PAGE,lines=[P.LINE],score=True,
              textWeights=P.TEXT_WEIGHTS,scoreLines=4),
  panel=dict(kind="image",key="crop%d" % P.LINE,aspect=7.0),
  metrics=["charAcc"],
  trendMetrics=["charAcc"],
  skipFigure=True,
  source=r"The LSTM cell is Hochreiter and Schmidhuber \cite{hochreiter1997}; "
         r"reading the sequence both ways is Schuster and Paliwal "
         r"\cite{schuster1997}; the two-layer bidirectional stack over a "
         r"convolutional backbone with CTC is the CRNN of \cite[p.~2298]{shi2017} "
         r"and the handwriting system of Graves \textit{et al.} "
         r"\cite[p.~855]{graves2009}.",
  reason=r"$256$ per direction, concatenated to $512$, is the size used by the "
         r"published CRNN this architecture follows. It is the one architectural "
         r"number in the reader that is taken from the literature rather than "
         r"derived here, and it is kept because the weights exist at that size.",
  understanding=r"\textbf{This cannot be swept at inference.} The checkpoint "
                r"stores gate matrices of shape $4n_h \times n_{\text{in}}$ and "
                r"$4n_h \times n_h$; changing $n_h$ makes those shapes "
                r"inconsistent and the forward pass fails outright. The sweep is "
                r"included so that failure is on the record: four of the five "
                r"runs are expected to raise, and the one that does not is the "
                r"submitted value. Answering what $n_h$ is worth requires "
                r"retraining at each size, which was not done.",
  alternatives=r"A smaller recurrent stack, which would cut the inference cost "
               r"that dominates on the ODROID; or a transformer encoder over the "
               r"same columns, which removes the sequential dependency but needs "
               r"far more data than eleven writers provide.",
  worked=r"Each direction holds $4n_h(n_{\text{in}}+n_h)+8n_h$ parameters. "
         r"Layer~0 sees $128$ channels, so it holds "
         r"$4 \times 256 \times (128+256)+2048 = 395\,264$ weights per "
         r"direction.",
  ),

]
