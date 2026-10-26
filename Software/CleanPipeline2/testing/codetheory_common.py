r"""Preamble, front matter, glossary and bibliography for the theory manual.

The theory for each module lives in codetheory_<module>.py as:

    INTRO = [ ("p", "paragraph"), ("h", "heading"), ... ]
    FUNCS = { "Name": dict(
        inputs = [("argName", "what it means")],
        output = "what comes back and what it means",
        theory = [ ("h", "What it is for"),
                   ("p", "prose"),
                   ("eq", r"a = b + c", [("a","meaning"),("b","meaning")],
                    "citeKey or None"),
                   ("items", ["bullet", "bullet"]),
                   ("code", ["literal", "lines"]),
                   ("note", "an aside in italics") ] ) }
"""

PREAMBLE = r"""\documentclass[11pt,a4paper,oneside]{report}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[margin=2.4cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{enumitem}
\usepackage{booktabs}
\usepackage{listings}
\usepackage{xcolor}
\usepackage[hidelinks]{hyperref}
\usepackage{titlesec}

% parskip.sty is not installed here, so the same effect is set by hand:
% paragraphs separated by a blank line rather than by an indent
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.7em plus 0.1em minus 0.1em}

\titleformat{\chapter}[hang]{\LARGE\bfseries}{\thechapter}{0.6em}{}
\titlespacing*{\chapter}{0pt}{0pt}{18pt}
\setcounter{secnumdepth}{2}
\setcounter{tocdepth}{1}

\definecolor{sigbg}{gray}{0.95}
\lstdefinestyle{sig}{basicstyle=\ttfamily\small,backgroundcolor=\color{sigbg},
  frame=none,xleftmargin=0pt,framexleftmargin=4pt,breaklines=true,
  aboveskip=6pt,belowskip=6pt,literate={_}{{\_}}1}
\lstdefinestyle{plain}{basicstyle=\ttfamily\footnotesize,frame=none,
  xleftmargin=1.5em,breaklines=true,aboveskip=6pt,belowskip=6pt,
  literate={_}{{\_}}1}

\title{\Huge Handwriting Reproduction Robot\\[6pt]
       \Large Theory Manual for the Software}
\author{D.R. Braune \\ u23539195 \\[4pt]
        University of Pretoria \\ EPR400 Final Year Project}
\date{}

\begin{document}
\maketitle
\tableofcontents
"""

FRONT = r"""
\chapter{How to use this manual}

This manual explains every function in the handwriting pipeline. For each one
it gives the declaration exactly as it appears in the code, what each input
means, what comes out, and then the theory: what the function is for, what
problem it solves, how it solves it, and why that method was chosen rather
than another.

The intention is that nothing is assumed. Where a term from mathematics,
image processing or machine learning appears for the first time it is
explained in place, and the most common terms are collected in the glossary
immediately after this chapter. Where a formula is given, every symbol in it
is defined underneath, and if the method comes from a published source that
source is named with the formula rather than only in the bibliography.

\paragraph{Conventions used throughout the software.}
These are worth reading once, because they are assumed by every chapter.

\begin{itemize}\setlength\itemsep{2pt}
  \item \textbf{Images} are rectangular grids of numbers, stored as
        \texttt{numpy} arrays and indexed \texttt{[row, column]}, which is
        \texttt{[y, x]}. The row index increases \emph{downwards}, which is
        the normal convention for images because that is the order a screen
        or a printer draws them in.
  \item \textbf{Pen paths} are lists of \texttt{(x, y)} points measured in
        millimetres, with y increasing \emph{upwards}, which is the normal
        convention for plotters and for ordinary graph paper. The pipeline
        therefore contains two coordinate systems whose vertical axes point
        in opposite directions. The conversion happens in exactly two places
        and both are pointed out where they occur.
  \item \textbf{Letter shapes} are stored in units of the x-height. The
        x-height is the height of the body of a lower case letter such as
        \texttt{a} or \texttt{o}, measured from the line the writing sits on.
        Storing a letter in these units, rather than in pixels, means the
        same stored letter can be drawn at any physical size.
  \item \textbf{No libraries.} Apart from \texttt{numpy}, which provides
        arithmetic on arrays of numbers, every algorithm in this manual is
        implemented in the files described here. There is no image
        processing library and no machine learning framework.
\end{itemize}

\chapter{Glossary}

Terms used in this manual, in plain language. They are grouped by subject
rather than alphabetically, because reading a group in order explains more
than reading a definition on its own.

\section{Arrays and arithmetic}
\begin{description}[style=nextline,leftmargin=1.4em]
  \item[Array] A grid of numbers. A one-dimensional array is a list, a
        two-dimensional array is a table, and a greyscale image is a
        two-dimensional array where each number is the brightness of one
        pixel, usually from 0 for black to 255 for white.
  \item[Element-wise] An operation applied to every number in an array
        independently. Adding 10 to an image, element-wise, makes every pixel
        ten levels brighter.
  \item[Vectorised] Written so that an operation is applied to a whole array
        at once rather than by looping over the elements one at a time in
        Python. This matters for speed: the loop runs inside compiled code
        instead of in the interpreter, which is typically between ten and a
        hundred times faster.
  \item[Cumulative sum, or prefix sum] An array where each entry is the total
        of everything up to that point. For the input $(2, 3, 1, 4)$ the
        cumulative sum is $(2, 5, 6, 10)$. Its usefulness is that the total
        of any stretch of the original can then be found by one subtraction,
        because the total from $i$ to $j$ equals the entry at $j$ minus the
        entry at $i$.
  \item[Mask] An array of true and false values, used to mark which pixels
        belong to something. An ``ink mask'' is true at every pixel judged to
        be ink and false elsewhere.
  \item[Dot product] For two lists of equal length, the sum of the products
        of corresponding entries. If both lists are first scaled so that
        their own length is one, the dot product equals the cosine of the
        angle between them considered as arrows, which is why it is used in
        this project to measure how similar two shapes are.
  \item[Norm] The length of a list of numbers considered as an arrow from the
        origin, which is the square root of the sum of the squares of its
        entries.
  \item[Interpolation] Estimating a value between known values. If a
        transformed image needs the brightness at a position halfway between
        two pixels, interpolation supplies it as a blend of the neighbours.
  \item[Median] The middle value when a set of numbers is sorted. It is used
        throughout this project in preference to the mean (the ordinary
        average) because it is insensitive to a small number of extreme
        values: one badly wrong measurement among eleven good ones moves the
        mean but not the median.
  \item[Percentile] The value below which a given share of the data falls.
        The 40th percentile is the value that 40 per cent of the data lies
        below. The median is the 50th percentile.
  \item[Variance and standard deviation] Measures of how spread out a set of
        numbers is. The variance is the average of the squared distances from
        the mean; the standard deviation is its square root, which has the
        same units as the data.
\end{description}

\section{Images and shapes}
\begin{description}[style=nextline,leftmargin=1.4em]
  \item[Threshold] A brightness level used to divide an image into two
        groups, typically ink and paper. ``Thresholding'' is applying it.
  \item[Histogram] A count of how many pixels fall at each brightness level.
        It describes an image's distribution of tones without reference to
        where anything is.
  \item[Morphology] A family of operations that change the shapes in a mask
        rather than the values in an image: growing shapes, shrinking them,
        filling small holes, removing small specks.
  \item[Structuring element, or kernel] The small window a morphological or
        filtering operation looks through. A square kernel of radius 2 is a
        five-by-five window centred on each pixel.
  \item[Dilation and erosion] Growing and shrinking shapes. Dilation turns a
        pixel on if anything nearby is on; erosion turns it off unless
        everything nearby is on.
  \item[Closing] A dilation followed by an erosion by the same amount. The
        net effect is that small gaps are bridged while the overall thickness
        of a shape is preserved.
  \item[Connected component] A group of pixels that touch one another,
        treated as a single object. A letter, a dot over an \texttt{i} and a
        speck of dust are each a connected component.
  \item[Connectivity] Which neighbours count as touching. Four-connectivity
        counts only the pixels above, below, left and right;
        eight-connectivity also counts the four diagonals, so two pixels
        meeting at a corner belong to the same component.
  \item[Skeleton, or centreline] A one-pixel-wide line running down the
        middle of a thick stroke. Thinning a pen stroke to its skeleton
        recovers approximately the path the pen travelled.
  \item[Projection profile] The result of adding up the ink in each row, or
        in each column, producing a one-dimensional summary of a
        two-dimensional image. Lines of text appear as peaks in the row
        profile.
  \item[Polyline] A path described by an ordered list of points, understood
        to be joined by straight segments.
  \item[Aspect ratio] The ratio of width to height.
  \item[Bounding box] The smallest upright rectangle containing a shape.
\end{description}

\section{Networks and training}
\begin{description}[style=nextline,leftmargin=1.4em]
  \item[Neural network] A function with a very large number of adjustable
        numbers, called weights, arranged in layers. Each layer takes the
        output of the previous one and performs a simple operation; the
        composition can represent complicated relationships. Training is the
        process of adjusting the weights so the function gives the right
        answers on examples.
  \item[Weight and bias] The adjustable numbers. A weight multiplies an
        input; a bias is added afterwards. Together they are the parameters.
  \item[Layer] One stage of a network.
  \item[Feature map] The output of a convolutional layer: an array showing,
        for each position in the image, how strongly some learned pattern was
        detected there.
  \item[Channel] One such map. A layer producing 64 channels is detecting 64
        different patterns.
  \item[Convolution] Sliding a small grid of weights over an image and
        recording the response at every position, which detects the same
        pattern wherever it occurs.
  \item[Activation function] A simple non-linear function applied after a
        layer, such as setting negatives to zero. Without one, stacking
        layers would be pointless, because a chain of purely linear
        operations is equivalent to a single linear operation.
  \item[Pooling] Reducing the size of a feature map by summarising each small
        window, typically by its largest value, which saves work and makes
        the result tolerant to small shifts.
  \item[Logit] A raw score produced by the final layer, before it is
        converted into a probability. Logits may be any real number, positive
        or negative.
  \item[Softmax] The conversion from logits to probabilities: exponentiate
        each score and divide by the total, which forces them to be positive
        and to sum to one.
  \item[Loss function] A single number measuring how wrong the network's
        output is on a training example. Training consists of reducing it.
  \item[Gradient] The collection of derivatives of the loss with respect to
        every weight: for each weight, the rate at which the loss would
        change if that weight changed. It is what tells training which
        direction to move each weight.
  \item[Chain rule] The rule of calculus for differentiating a composition of
        functions, which is what makes it possible to compute the gradient of
        a loss with respect to a weight many layers away from it.
  \item[Backpropagation] Applying the chain rule layer by layer from the loss
        back to the first layer, so that every weight's gradient is obtained
        in one backward pass.
  \item[Forward pass, backward pass] Computing the output from the input, and
        computing the gradients from the loss, respectively.
  \item[Batch] A group of examples processed together, so that one weight
        update is based on their average gradient rather than on a single
        example, which is both faster and less noisy.
  \item[Epoch] One complete pass through the training data.
  \item[Learning rate] How far each weight is moved per update.
  \item[Optimiser] The rule that turns gradients into weight updates.
  \item[Overfitting] Learning the training examples rather than the general
        pattern, so that performance on new data is poor. It is detected by
        measuring on data the network was not trained on.
  \item[Holdout, or validation set] Data deliberately excluded from training
        and kept for measurement, which is the only way an accuracy figure
        means anything.
  \item[Checkpoint] A saved copy of all the weights, so a trained network can
        be reloaded later.
  \item[Recurrent layer] A layer that processes a sequence one position at a
        time while carrying information forward, so that a decision at one
        position can depend on earlier ones.
  \item[Inference] Running a trained network forward to get an answer, as
        opposed to training it.
\end{description}

\section{This project's own terms}
\begin{description}[style=nextline,leftmargin=1.4em]
  \item[Glyph] One stored example of one character in one writer's hand: a
        set of pen strokes, in x-height units, with measurements describing
        how much space it occupies.
  \item[Variant] One of several stored glyphs for the same character by the
        same writer, since nobody writes a letter identically twice.
  \item[Prior distance] A number measuring how far a stored glyph's shape
        sits from the typical shape of that character pooled across all
        writers. Low means a well-formed letter; high means the shape was
        probably cut out wrongly.
  \item[Anchor, or fallback letterform] A clean, readable shape used in place
        of a stored variant when the stored variants are too damaged to be
        legible. Preferably the writer's own most typical version of the
        letter.
  \item[Legibility lambda] A number between 0 and 1 controlling how often the
        anchor is used in place of the writer's own shapes, and therefore
        where the output sits between being readable and being recognisably
        that person's hand.
  \item[Connectedness] The measured fraction of neighbouring letter pairs
        within a word that the writer joins without lifting the pen.
  \item[Ligature] The joining stroke between two letters written without
        lifting the pen.
  \item[Trajectory] The pipeline's representation of a drawing: a list of pen
        strokes in millimetres.
  \item[G-code] The text format understood by the machine, consisting of
        lines such as \texttt{G1 X10 Y20 F900}, meaning ``move in a straight
        line to this position at this feed rate''.
  \item[Endstop] A switch at the end of an axis's travel, used to find a
        known position and to prevent the machine driving into its own
        frame.
\end{description}
"""

BACK = r"""
\bibliographystyle{ieeetr}
\bibliography{refs}
\end{document}
"""

BIB = r"""
@article{otsu1979,
  author  = {Otsu, Nobuyuki},
  title   = {A Threshold Selection Method from Gray-Level Histograms},
  journal = {IEEE Transactions on Systems, Man, and Cybernetics},
  volume  = {9}, number = {1}, pages = {62--66}, year = {1979}
}
@article{crow1984,
  author  = {Crow, Franklin C.},
  title   = {Summed-Area Tables for Texture Mapping},
  journal = {ACM SIGGRAPH Computer Graphics},
  volume  = {18}, number = {3}, pages = {207--212}, year = {1984}
}
@article{zhangsuen1984,
  author  = {Zhang, T. Y. and Suen, C. Y.},
  title   = {A Fast Parallel Algorithm for Thinning Digital Patterns},
  journal = {Communications of the ACM},
  volume  = {27}, number = {3}, pages = {236--239}, year = {1984}
}
@article{he2008,
  author  = {He, Lifeng and Chao, Yuyan and Suzuki, Kenji},
  title   = {A Run-Based Two-Scan Labeling Algorithm},
  journal = {IEEE Transactions on Image Processing},
  volume  = {17}, number = {5}, pages = {749--756}, year = {2008}
}
@article{tarjan1975,
  author  = {Tarjan, Robert Endre},
  title   = {Efficiency of a Good But Not Linear Set Union Algorithm},
  journal = {Journal of the ACM},
  volume  = {22}, number = {2}, pages = {215--225}, year = {1975}
}
@inproceedings{postl1986,
  author    = {Postl, Wolfgang},
  title     = {Detection of Linear Oblique Structures and Skew Scan in
               Digitized Documents},
  booktitle = {Proc. 8th International Conference on Pattern Recognition},
  pages     = {687--689}, year = {1986}
}
@article{vinciarelli2001,
  author  = {Vinciarelli, Alessandro and Luettin, Juergen},
  title   = {A New Normalization Technique for Cursive Handwritten Words},
  journal = {Pattern Recognition Letters},
  volume  = {22}, number = {9}, pages = {1043--1050}, year = {2001}
}
@inproceedings{arvanitopoulos2014,
  author    = {Arvanitopoulos, Nikolaos and S{\"u}sstrunk, Sabine},
  title     = {Seam Carving for Text Line Extraction on Color and Grayscale
               Historical Manuscripts},
  booktitle = {Proc. 14th International Conference on Frontiers in
               Handwriting Recognition},
  pages     = {726--731}, year = {2014}
}
@article{avidan2007,
  author  = {Avidan, Shai and Shamir, Ariel},
  title   = {Seam Carving for Content-Aware Image Resizing},
  journal = {ACM Transactions on Graphics},
  volume  = {26}, number = {3}, year = {2007}
}
@article{bellman1954,
  author  = {Bellman, Richard},
  title   = {The Theory of Dynamic Programming},
  journal = {Bulletin of the American Mathematical Society},
  volume  = {60}, number = {6}, pages = {503--515}, year = {1954}
}
@book{wolberg1990,
  author    = {Wolberg, George},
  title     = {Digital Image Warping},
  publisher = {IEEE Computer Society Press}, year = {1990}
}
@inproceedings{simard2003,
  author    = {Simard, Patrice Y. and Steinkraus, Dave and Platt, John C.},
  title     = {Best Practices for Convolutional Neural Networks Applied to
               Visual Document Analysis},
  booktitle = {Proc. 7th International Conference on Document Analysis and
               Recognition},
  pages     = {958--963}, year = {2003}
}
@article{douglaspeucker1973,
  author  = {Douglas, David H. and Peucker, Thomas K.},
  title   = {Algorithms for the Reduction of the Number of Points Required to
             Represent a Digitized Line or its Caricature},
  journal = {Cartographica},
  volume  = {10}, number = {2}, pages = {112--122}, year = {1973}
}
@inproceedings{graves2006,
  author    = {Graves, Alex and Fern{\'a}ndez, Santiago and Gomez, Faustino
               and Schmidhuber, J{\"u}rgen},
  title     = {Connectionist Temporal Classification: Labelling Unsegmented
               Sequence Data with Recurrent Neural Networks},
  booktitle = {Proc. 23rd International Conference on Machine Learning},
  pages     = {369--376}, year = {2006}
}
@article{hochreiter1997,
  author  = {Hochreiter, Sepp and Schmidhuber, J{\"u}rgen},
  title   = {Long Short-Term Memory},
  journal = {Neural Computation},
  volume  = {9}, number = {8}, pages = {1735--1780}, year = {1997}
}
@article{shi2017,
  author  = {Shi, Baoguang and Bai, Xiang and Yao, Cong},
  title   = {An End-to-End Trainable Neural Network for Image-Based Sequence
             Recognition and Its Application to Scene Text Recognition},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume  = {39}, number = {11}, pages = {2298--2304}, year = {2017}
}
@inproceedings{chellapilla2006,
  author    = {Chellapilla, Kumar and Puri, Sidd and Simard, Patrice},
  title     = {High Performance Convolutional Neural Networks for Document
               Processing},
  booktitle = {Proc. 10th International Workshop on Frontiers in Handwriting
               Recognition}, year = {2006}
}
@inproceedings{ioffe2015,
  author    = {Ioffe, Sergey and Szegedy, Christian},
  title     = {Batch Normalization: Accelerating Deep Network Training by
               Reducing Internal Covariate Shift},
  booktitle = {Proc. 32nd International Conference on Machine Learning},
  pages     = {448--456}, year = {2015}
}
@inproceedings{kingma2015,
  author    = {Kingma, Diederik P. and Ba, Jimmy},
  title     = {Adam: A Method for Stochastic Optimization},
  booktitle = {Proc. 3rd International Conference on Learning
               Representations}, year = {2015}
}
@inproceedings{loshchilov2019,
  author    = {Loshchilov, Ilya and Hutter, Frank},
  title     = {Decoupled Weight Decay Regularization},
  booktitle = {Proc. 7th International Conference on Learning
               Representations}, year = {2019}
}
@misc{tieleman2012,
  author = {Tieleman, Tijmen and Hinton, Geoffrey},
  title  = {Lecture 6.5 --- RMSProp: Divide the Gradient by a Running Average
            of its Recent Magnitude},
  note   = {COURSERA: Neural Networks for Machine Learning}, year = {2012}
}
@article{levenshtein1966,
  author  = {Levenshtein, Vladimir I.},
  title   = {Binary Codes Capable of Correcting Deletions, Insertions, and
             Reversals},
  journal = {Soviet Physics Doklady},
  volume  = {10}, number = {8}, pages = {707--710}, year = {1966}
}
@article{viterbi1967,
  author  = {Viterbi, Andrew J.},
  title   = {Error Bounds for Convolutional Codes and an Asymptotically
             Optimum Decoding Algorithm},
  journal = {IEEE Transactions on Information Theory},
  volume  = {13}, number = {2}, pages = {260--269}, year = {1967}
}
@article{bresenham1965,
  author  = {Bresenham, Jack E.},
  title   = {Algorithm for Computer Control of a Digital Plotter},
  journal = {IBM Systems Journal},
  volume  = {4}, number = {1}, pages = {25--30}, year = {1965}
}
@article{srivastava2014,
  author  = {Srivastava, Nitish and Hinton, Geoffrey and Krizhevsky, Alex and
             Sutskever, Ilya and Salakhutdinov, Ruslan},
  title   = {Dropout: A Simple Way to Prevent Neural Networks from
             Overfitting},
  journal = {Journal of Machine Learning Research},
  volume  = {15}, pages = {1929--1958}, year = {2014}
}
@inproceedings{glorot2010,
  author    = {Glorot, Xavier and Bengio, Yoshua},
  title     = {Understanding the Difficulty of Training Deep Feedforward
               Neural Networks},
  booktitle = {Proc. 13th International Conference on Artificial Intelligence
               and Statistics}, pages = {249--256}, year = {2010}
}
@inproceedings{he2015,
  author    = {He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  title     = {Delving Deep into Rectifiers: Surpassing Human-Level
               Performance on ImageNet Classification},
  booktitle = {Proc. IEEE International Conference on Computer Vision},
  pages     = {1026--1034}, year = {2015}
}
@inproceedings{pascanu2013,
  author    = {Pascanu, Razvan and Mikolov, Tomas and Bengio, Yoshua},
  title     = {On the Difficulty of Training Recurrent Neural Networks},
  booktitle = {Proc. 30th International Conference on Machine Learning},
  pages     = {1310--1318}, year = {2013}
}
@article{marti2002,
  author  = {Marti, Urs-Viktor and Bunke, Horst},
  title   = {The IAM-Database: An English Sentence Database for Offline
             Handwriting Recognition},
  journal = {International Journal on Document Analysis and Recognition},
  volume  = {5}, pages = {39--46}, year = {2002}
}
"""
