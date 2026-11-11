from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'Report/DesignVisualRevision_2026-11-01'
d=json.loads((HERE/'Evidence/bench.json').read_text(encoding='utf-8'))
cam=json.loads((HERE/'Evidence/camera_bench.json').read_text(encoding='utf-8'))
seg=HERE/'3_Main_Report/design/3_4_segmentation.tex'
s=seg.read_text(encoding='utf-8')
start=s.index('The verification comparison in Figure~')
end=s.index('\\begin{figure}',start)
s=s[:start]+r'''Figure~\ref{fig:des-seg-variation} uses three actual photographs of the same three-line handwritten page from the camera roll. Their shadows and page presentation differ; no illuminance or physical rotation angle was recorded. The unchanged native images were processed through the line-axis decision and segmentation, without manually correcting an unsuccessful orientation.

The striped-shadow capture produced three crops. The diffuse-shadow capture produced four, so matching the writing visually remains necessary even under milder-looking illumination. The third capture was left on the wrong line axis: its two gap scores were 0.379 and 0.400, below the 1.35 turn-ratio requirement, and segmentation produced 28 slices. Reaching the $-4^\circ$ deskew limit could not correct a missed quarter turn. These cases expose a sparse-page limitation of row-gap orientation; they do not establish a lighting operating range or a success rate. An acquisition guide or a separately verified orientation method would address the ambiguity before fine deskewing.

The fuller Owen page in Figure~\ref{fig:des-seg-stages} produced 20 crops at native resolution, matching its 20 labelled text lines. That count checks gross splitting, not every boundary pixel. Appendix~\ref{app:des-visual-tests} records the capture filenames, extended digital perturbations and independent algorithm checks, including failures after additional image reduction.
''' +s[end:]
s=s.replace('Effect of capture variation on conditioning and line separation for the same page.','Actual repeated photographs and segmentation diagnostics for the same sparse page. Capture C retains the unsuccessful axis decision.')
seg.write_text(s,encoding='utf-8')
p=HERE/'3_Main_Report/design/3_5_classification.tex';s=p.read_text(encoding='utf-8')
s=s.replace('A confirmed numerical comparison','A confirmed numerical comparison')
needle='An additional check can compare an implemented derivative with the central difference in Equation~'
pos=s.index(needle)
s=s[:pos]+r'''A separate, newly run small-lattice check enumerated all $4^5=1024$ frame paths for a repeated-label target and compared their summed probability with the numerical CTC loss. The absolute loss difference was $4.44\times10^{-16}$; central differences over all 20 logits gave a maximum absolute derivative difference of $3.63\times10^{-11}$. This verifies that finite example, including the required blank between repeated labels. Its construction and limits appear in Appendix~\ref{app:des-visual-tests}; it is not a learned-model accuracy measurement.

'''+s[pos:]
s=s.replace('; impossible alignments follow the implementation\'s guard','')
p.write_text(s,encoding='utf-8')
p=HERE/'3_Main_Report/design/3_6_reproduction.tex';s=p.read_text(encoding='utf-8')
s=s.replace('1.60 rulings','1.25 rulings')
needle='A low prior distance expresses resemblance'
# Add provenance next to alignment, without implying archived crops came from this new page run.
pos=s.index('The initial boundaries are refined')
s=s[:pos]+r'''The worked example in Figure~\ref{fig:des-glyph-alignment} uses an archived original Owen line crop, visually matched to the first line of its page transcript, with the joint reader checkpoint. The reader recovered the full line, and the extraction routine then generated the displayed boundaries and normalised paths. Correct text does not certify a correct individual cut: small punctuation and neighbouring strokes still require inspection. The archived crop is identified separately from the new whole-page segmentation run in the evidence manifest.

'''+s[pos:]
pos=s.index('The same models guide selection')
s=s[:pos]+r'''In the worked Owen example, three pre-repair trials use the fixed text ``Writing with style.'', seeds 2026, 3003 and 3980, and the selected eleven-writer shape model with an explicit width fraction of 0.17. Candidate 1 has $(A,P,J)=(0.947,0.579,0.718)$, while candidate 3 has the same character score but $(P,J)=(0.464,0.623)$. The combined criterion therefore selects candidate 1. Its reader output begins with a lower-case ``w'', which accounts for one edit rather than perfect text recovery. This is an inspectable scoring example before repair, not a claim that the full default search or the physical system attained those scores.

'''+s[pos:]
pos=s.index('The representation\'s main advantage')
s=s[:pos]+r'''Appendix~\ref{app:des-visual-tests} shows the same sentence generated from four stored profiles alongside paths independently parsed from their emitted G-code. Nine controlled conversion fixtures additionally preserved path vertex counts with no clipping and a maximum coordinate discrepancy of $0.0004$\,mm, consistent with writing coordinates to three decimal places. Joins and ring closure were disabled only for those correspondence fixtures so that the vertices could be paired; the four profile illustrations used the normal configuration. Neither check measures the mechanism's positional accuracy.

'''+s[pos:]
p.write_text(s,encoding='utf-8')

rows=[]
for writer in ['Owen','Dylan','Yeukita']:
    v=[x for x in d['segmentation'] if x['writer']==writer]
    rows.append(writer+' & '+str(v[0]['expected_label_lines'])+' & '+' & '.join(str(x['nLines']) for x in v)+r' \\')
body=r'''\section{Subsystem visual evidence and independent software checks}
\label{app:des-visual-tests}
This appendix records newly run offline checks and illustrations made from existing project data. It supplements the Design explanations; the integrated qualification tests against the proposal remain in Results. The evidence scripts use the current processing modules, record source and input SHA-256 hashes, and write their outputs into this revision's \texttt{Evidence} directory. No training or physical motion was performed. Diagnostic settings are stated explicitly rather than inherited silently from server defaults.

\subsection{Actual capture variation and input resolution}
The three camera-roll inputs in Figure~\ref{fig:des-seg-variation} are \texttt{WIN\_20261028\_23\_23\_30\_Pro.jpg}, \texttt{WIN\_20261028\_23\_28\_34\_Pro.jpg} and \texttt{WIN\_20261028\_23\_32\_40\_Pro.jpg}. Each stored image is $3264\times2448$ pixels. The page visibly contains three handwritten text rows. Automatic axis decisions were retained, including the failure in the third capture. The observed crop counts were 4, 3 and 28, respectively. Shadows are visually different, but neither lux levels nor physical angles were measured. These observations therefore explain a design limitation rather than qualifying operation across a specified lighting range.

The native Owen01 page ($2448\times3264$) produced 20 crops for 20 labelled lines. A separate exploratory bench first fitted each of three original pages inside $1600\times1600$ pixels. From each reduced page it produced an unchanged view, a synthetic left-to-right multiplier from 0.42 to 1.00, a synthetic $+3^\circ$ rotation using bicubic interpolation on an expanded white canvas, and a quarter turn. Each was passed through the same automatic axis and segmentation functions. Table~\ref{tab:des-resample-cases} retains every case, including unsuccessful ones.
\begin{table}[!htbp]\centering\small
\begin{tabular}{lrrrrr}\hline
Page & Label lines & Reduced & Gradient & $+3^\circ$ & $90^\circ$ \\\hline
'''+'\n'.join(rows)+r'''
\hline\end{tabular}
\caption{Exploratory crop counts after extra input reduction and digital perturbation. Equal counts do not establish correct ink ownership.}
\label{tab:des-resample-cases}
\end{table}
The reduced Owen case produced 19 crops and lost the usable first-line crop, whereas the native page produced 20. Its rotated reduced image produced only 11. The gradient image happened to restore the count to 20, which is not evidence that adding a shadow improves segmentation: thresholds, interpolation and component geometry changed together. These cases show why nominally scale-relative thresholds do not guarantee resolution invariance. A suitable follow-up would compare matched stroke ownership at controlled resolutions and improve capture guidance; the present bench does not supply pixel-level annotations or attribute every error to one operation.

\subsection{Independent component and seam references}
The run-based eight-connected component labeller was compared with an independently written flood-fill reference on 100 binary masks of size $9\times11$, using NumPy seed 2026 and foreground probabilities spanning 0.1--0.8. The test compared the full same-component relation between foreground pixel pairs, not arbitrary numerical label values. All 100 partitions agreed. Small random masks exercise diagonal contact and label equivalences, but do not demonstrate large-image speed or prove every possible topology.

For the seam solver, 20 random $4\times6$ cost arrays (seed 83) were checked against exhaustive enumeration of all row sequences with adjacent row changes of at most one. Both objectives included the implementation's $0.02|\Delta y|$ penalty. The maximum objective discrepancy was '''+f"${d['seam_reference']['maximum_cost_error']:.1g}$"+r'''. The exhaustive reference tests optimal cost on those arrays; it does not establish that the chosen image-derived cost always follows the correct handwritten line boundary.

\begin{designalgorithm}
\caption{Independent partition and seam verification}
\label{alg:des-reference-checks}
\prequire{Fixed random seeds, mask cases and small seam-cost arrays.}
\begin{designpseudo}
\pstate{0}{\textbf{for each} binary mask $M$ \textbf{do}}
\pstate{1}{$A\leftarrow\operatorname{LabelComponents}(M)$; $B\leftarrow\operatorname{FloodFill8}(M)$}
\pstate{1}{\textbf{assert} $[A_p=A_q]=[B_p=B_q]$ for every foreground pair $(p,q)$}
\pstate{0}{\textbf{end for}}
\pstate{0}{\textbf{for each} small cost array $C$ \textbf{do}}
\pstate{1}{$p^*\leftarrow\operatorname{LeastCostSeam}(C,1)$}
\pstate{1}{$\mathcal P\leftarrow$ all paths with adjacent row differences at most one}
\pstate{1}{$E(p)\leftarrow\sum_x C(p_x,x)+0.02\sum_x|p_x-p_{x-1}|$}
\pstate{1}{Record $|E(p^*)-\min_{p\in\mathcal P}E(p)|$}
\pstate{0}{\textbf{end for}}
\pstate{0}{\textbf{return} partition agreements and seam-cost discrepancies}
\end{designpseudo}
\end{designalgorithm}

\subsection{CTC probability and derivative check}
The loss check used float64 logits of shape $5\times1\times4$, blank index 0 and target $[1,1]$. All 1024 frame paths were enumerated. Consecutive repeats were collapsed before blanks were removed, and only paths yielding the target contributed to its probability. This independently exercises the repeated-label restriction that a single uninterrupted run of label 1 cannot encode two labels. The path sum gave negative log likelihood 3.875974036731708; the dynamic programme gave 3.875974036731709. The absolute difference was $4.44\times10^{-16}$.

All 20 logits were perturbed individually with central-difference step $10^{-5}$. The maximum absolute discrepancy from \texttt{train.py::CtcLoss} was $3.63\times10^{-11}$. Logits were drawn after the fixed-seed component-mask draws in the bench; the script preserves that order. These checks concern the implemented CTC loss and derivative on a finite feasible target. They do not test impossible alignments, every transcript length, trained recognition accuracy or the full CNN--LSTM backward pass.

\begin{designalgorithm}
\caption{Exhaustive CTC likelihood and finite-difference check}
\label{alg:des-ctc-reference}
\prequire{Small logits $z$, target $\ell$, blank index and step $\varepsilon$.}
\begin{designpseudo}
\pstate{0}{$(L,g)\leftarrow\operatorname{CtcLoss}(z,\ell)$; $p\leftarrow\operatorname{Softmax}(z)$; $Z\leftarrow0$}
\pstate{0}{\textbf{for each} frame path $\pi\in\{0,\ldots,C-1\}^T$ \textbf{do}}
\pstate{1}{\textbf{if} $\mathcal B(\pi)=\ell$ \textbf{then} $Z\leftarrow Z+\prod_t p_t(\pi_t)$}
\pstate{0}{\textbf{end for}}
\pstate{0}{Record $|L+\log Z|$}
\pstate{0}{\textbf{for each} logit index $i$ \textbf{do}}
\pstate{1}{$g_i^{\rm ref}\leftarrow[L(z+\varepsilon e_i)-L(z-\varepsilon e_i)]/(2\varepsilon)$}
\pstate{1}{Record $|g_i-g_i^{\rm ref}|$}
\pstate{0}{\textbf{end for}}
\pstate{0}{\textbf{return} loss discrepancy and maximum derivative discrepancy}
\end{designpseudo}
\end{designalgorithm}

\subsection{Stored profiles, emitted files and physical page}
Figure~\ref{fig:des-style-files} uses the final Owen, Dylan, Yeukita and IAM 150 profiles with the same text, seed 2026, requested x-height 4\,mm and jitter 0.15. The left column shows generated pen paths. The right column reconstructs pen-down paths from each emitted G-code file with the independent parser; each panel is fitted separately for legibility. The normal converter was used, including its path ordering, joins and near-ring closure. All four generated jobs had coordinates within the configured bed bounds. The figures illustrate profile-dependent geometry and file conversion, rather than measuring similarity to a human writer.
\begin{figure}[!htbp]\centering
\includegraphics[width=0.98\linewidth]{Figures/design-style-gcode-examples.pdf}
\caption{Same text generated from four final profiles and reconstructed from emitted G-code. These are software path drawings.}
\label{fig:des-style-files}
\end{figure}

Nine controlled fixtures combined scale factors 1, 3 and 30 with line offsets 0, 8.4 and 25.2\,mm. Ring closure and joining were disabled, minimum-segment filtering was set to zero, and the independently parsed paths were paired with \texttt{ToMachine}'s output. Each fixture retained path and vertex counts, with no clamped points. Maximum coordinate error was $0.0004$\,mm, within the $0.0005$\,mm per-coordinate rounding bound from three-decimal emission. This comparison checks serialization after placement; it does not independently prove that the placement equation itself is correct or that a motor reached a coordinate.

\begin{designalgorithm}
\caption{Controlled G-code serialization correspondence}
\label{alg:des-file-reference}
\prequire{Known multi-path trajectory and configurations with joins and closure disabled.}
\begin{designpseudo}
\pstate{0}{\textbf{for each} fixture configuration $c$ \textbf{do}}
\pstate{1}{$P\leftarrow\operatorname{ToMachine}(T,c)$}
\pstate{1}{$f\leftarrow\operatorname{WriteGcode}(T,c)$; $Q\leftarrow\operatorname{ParseGcode}(f)$}
\pstate{1}{Compare path counts and corresponding vertex counts}
\pstate{1}{Record $\max_i\|P_i-Q_i\|_\infty$ and clamped-point count}
\pstate{0}{\textbf{end for}}
\pstate{0}{\textbf{return} count agreements, rounding discrepancies and clipping records}
\end{designpseudo}
\end{designalgorithm}

Figure~\ref{fig:des-physical-page} is separately identified as a photograph of actual robot output, from the existing Owen reproduction sheet. The line crops are newly obtained from that stored photograph. This closes the visual explanation from a physical page to the segmentation interface, without presenting the earlier drawing as a newly executed hardware experiment. No exact same-job correspondence with Figure~\ref{fig:des-style-files} is claimed.
\begin{figure}[!htbp]\centering
\includegraphics[width=0.95\linewidth]{Figures/design-physical-page-example.pdf}
\caption{Existing photographed robot output and newly recovered line crops. The photograph is distinct from the software renders above.}
\label{fig:des-physical-page}
\end{figure}

\subsection{Candidate-scoring configuration and scope}
The three Owen trials in Figure~\ref{fig:des-candidates} use the joint reader and selected \texttt{author\_shape\_11\_pw170\_weights.npz} model. Candidate images are rendered with uniform ink, then stroke-normalised explicitly at width 0.17 before writer inference. This bench selects checkpoints and preprocessing directly; it does not certify that an older server default selects the same configuration. The diagnostic score uses the implemented character-accuracy function and requested-writer posterior before repair. The exact readings and full-precision scores are retained in \texttt{Evidence/bench.json}. No candidate, page or test outcome was substituted to manufacture a passing result.
\FloatBarrier
'''
(HERE/'Appendices/design_subsystem_evidence.tex').write_text(body,encoding='utf-8')
p=HERE/'design_preview.tex';s=p.read_text(encoding='utf-8').replace('\\end{document}','\\clearpage\n\\input{Appendices/design_subsystem_evidence}\n\\end{document}');p.write_text(s,encoding='utf-8')
print('Integrated new evidence and appendix')
