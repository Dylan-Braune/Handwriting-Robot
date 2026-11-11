"""Apply checked source locators and explicit manual-check flags to Design only."""
from pathlib import Path
BASE=Path(__file__).resolve().parents[2]/'Report/DesignRevision_2026-11-01'

def edit(rel, pairs):
 p=BASE/rel;t=p.read_text(encoding='utf-8')
 for old,new in pairs:
  if old not in t: raise ValueError(f'Missing target in {rel}: {old[:90]}')
  t=t.replace(old,new)
 p.write_text(t,encoding='utf-8')

edit('3_Main_Report/design/3_3_hardware.tex',[
 (r'~\cite{teikari2016}',r'~\cite[p.~6, Fig.~4]{teikari2016}'),
 # Place the thesis citation against its actual Cartesian-example claim.
 (r'and a pen up/down mechanism. Complex kinematic equations and joint-dependent position errors are avoided by using separate $x$ and $y$ movements for reproduction~\cite[p.~6, Fig.~4]{teikari2016}.',
  r'and a pen up/down mechanism, consistent with the Cartesian arrangement illustrated by Teikari~\cite[p.~6, Fig.~4]{teikari2016}. Complex kinematic equations and joint-dependent position errors are avoided by using separate $x$ and $y$ movements for reproduction.'),
 (r'next step~\cite{a4988ds}',r'next step~\cite[p.~9]{a4988ds}'),
 (r'inputs~\cite{a4988ds}',r'inputs~\cite[p.~6, Table~1]{a4988ds}'),
 (r'~\cite{acarnley2002}',r'~\cite{acarnley2002}\designsourceflag{acarnley2002}'),
 (r'~\cite{bresenham1965}',r'~\cite[p.~28]{bresenham1965}'),
 (r'~\cite{bc337ds}',r'~\cite{bc337ds}\designsourceflag{bc337ds}'),
 (r'approximate base current.',r'approximate base current using the standard low-side resistor relation~\cite[section ``Why NPN Devices Are Used in Low-Side Switches'']{rohmlowside2026}.'),
 (r'the switching stage. A diode across the motor,',r'the switching stage~\cite[section ``Protection and Wiring Techniques'']{rohmlowside2026}. A diode across the motor,'),
 (r'\ref{eq:des-current}~\cite{a4988ds}',r'\ref{eq:des-current}~\cite[p.~9]{a4988ds}'),
])
edit('3_Main_Report/design/3_4_segmentation.tex',[
 (r'~\cite{crow1984}',r'~\cite{crow1984}\designsourceflag{crow1984}'),
 (r'~\cite{otsu1979}',r'~\cite{otsu1979}\designsourceflag{otsu1979}'),
 (r'~\cite{postl1986}',r'~\cite{postl1986}\designsourceflag{postl1986}'),
 (r'~\cite{avidan2007}',r'~\cite[PDF p.~3, Sec.~3]{avidan2007}'),
 (r'~\cite{rosenfeld1966}',r'~\cite[pp.~476--479]{rosenfeld1966}'),
])
edit('3_Main_Report/design/3_5_classification.tex',[
 (r'~\cite{kizilirmak2023}',r'~\cite[pp.~4--5]{kizilirmak2023}'),
 (r'~\cite{chellapilla2006}',r'~\cite{chellapilla2006}\designsourceflag{chellapilla2006}'),
 (r'~\cite{ioffe2015}',r'~\cite[PDF pp.~4--5, Alg.~1]{ioffe2015}'),
 (r'~\cite{hochreiter1997}',r'~\cite[p.~5, Eqs.~(9)--(11)]{graves2013}'),
 (r'Each is computed from the current feature vector and previous hidden state.',r'Each is computed from the current feature vector and previous hidden state. The implementation omits the additional cell-to-gate peephole connections shown in the cited source.'),
 (r'without requiring character positions~\cite{graves2006}',r'without requiring character positions~\cite[PDF pp.~2--3]{graves2006}'),
 (r'through the alignment lattice~\cite{graves2006}',r'through the alignment lattice~\cite[PDF pp.~4--6]{graves2006}'),
 (r'~\cite{srivastava2014}',r'~\cite[pp.~1931, 1934]{srivastava2014}'),
 (r'~\cite{zhang1984}',r'~\cite[p.~237]{zhang1984}'),
 (r'~\cite{pytorchctc}',r'~\cite[CTCLoss, ``Parameters'']{pytorchctc}'),
 (r'its per-sample loss is $-\log p_y$, with logit derivative $p_k-\mathbf{1}[k=y]$.',r'its per-sample loss is $-\log p_y$, with logit derivative $p_k-\mathbf{1}[k=y]$~\cite[p.~6, Eqs.~(13)--(14)]{graves2013}.'),
 (r'calculated by forward and backward recurrences through the alignment lattice',r'calculated by forward and backward recurrences through the alignment lattice'),
 (r'The derivative used by the implementation is then Equation~\ref{eq:des-ctcgrad}.',r'The derivative used by the implementation is then Equation~\ref{eq:des-ctcgrad}, with the occupancy convention obtained by adapting the CTC forward--backward derivation~\cite[PDF pp.~5--6]{graves2006}.'),
])
edit('3_Main_Report/design/3_6_reproduction.tex',[
 (r'~\cite{viterbi1967,graves2006}',r'by replacing path summation in the CTC lattice with maximisation~\cite[PDF pp.~4--5]{graves2006}'),
 (r'~\cite{zhang1984}',r'~\cite[p.~237]{zhang1984}'),
 (r'~\cite{douglas1973}',r'~\cite{douglas1973}\designsourceflag{douglas1973}'),
 (r'$d_{\mathrm{edit}}$ is Levenshtein distance~\cite{levenshtein1966}',r'$d_{\mathrm{edit}}$ is the minimum number of insertions, deletions and substitutions, using the reader error convention~\cite[PDF p.~2, Sec.~2.1]{graves2006}'),
])
edit('Appendices/design_implementation_details.tex',[
 (r'This appendix records the derivative conventions, image operations and parameter meanings needed to inspect the numerical implementation in more detail.',r'This appendix records the derivative conventions, image operations and parameter meanings needed to inspect the numerical implementation in more detail~\cite[\texttt{train.py}, \texttt{segment.py}, \texttt{style.py}]{braunepipeline2026}.'),
 (r'Equation~\ref{eq:app-bnback} gives the input derivative.',r'Equation~\ref{eq:app-bnback} gives the input derivative, obtained by simplifying the batch-normalisation chain-rule expressions~\cite[PDF p.~4]{ioffe2015}.'),
 (r'Inverted dropout multiplies by the cached Bernoulli mask divided by $1-p$ during training, and is the identity at inference.',r'The implementation uses inverted dropout: it multiplies by the cached Bernoulli mask divided by $1-p$ during training and is the identity at inference. This moves the expected-output scaling from test time to training time relative to the original presentation~\cite[pp.~1931, 1934]{srivastava2014}.'),
 (r'The LSTM backward pass carries both the hidden derivative from the next step and a separate cell derivative.',r'The LSTM backward pass differentiates the implemented recurrence in Equation~\ref{eq:des-lstm}; full backpropagation through time for this cell form is described by Graves~\cite[pp.~4--5]{graves2013}. It carries both the hidden derivative from the next step and a separate cell derivative.'),
 (r'The numerical CTC forward score uses the recurrence in Equation~\ref{eq:app-ctcforward}.',r'The numerical CTC forward score uses the recurrence in Equation~\ref{eq:app-ctcforward}, written in log space from the CTC path recursion~\cite[PDF p.~4, Sec.~4.1]{graves2006}.'),
 (r'The reader uses an uncentred RMSprop accumulator.',r'The reader uses an uncentred RMSprop accumulator, following the moving-square-gradient principle~\cite[PDF p.~29]{tieleman2012}. The code selects its own averaging coefficient and adds the stated decay term.'),
 (r"The writer's AdamW instead uses separate parameter decay, as shown in Equation~\ref{eq:app-adamw}.",r"The writer's AdamW instead uses separate parameter decay, as shown in Equation~\ref{eq:app-adamw}~\cite[p.~2, Alg.~1]{kingma2015}\cite[p.~3, Alg.~2]{loshchilov2019}."),
 (r'Zhang--Suen thinning inspects the eight neighbours',r'The thinning conditions follow Zhang and Suen~\cite[p.~237]{zhang1984}. The method inspects the eight neighbours'),
 (r'Near-closed paths are first split at a distant point',r'The code adapts the polyline reduction method~\cite{douglas1973}\designsourceflag{douglas1973}. Near-closed paths are first split at a distant point'),
])
edit('3_Main_Report/3_Design_and_Implementation.tex',[
 (r'\input{3_Main_Report/design/3_1_summary}',r"Implementation-specific equations, parameter values and pseudocode describe the project's CleanPipeline3 source~\cite{braunepipeline2026}; borrowed methods are cited at their point of use. Worked numerical examples are derived here unless identified as recorded measurements.\par"+'\n\n'+r'\input{3_Main_Report/design/3_1_summary}'),
])
print('Applied source locators and unresolved-source flags.')
