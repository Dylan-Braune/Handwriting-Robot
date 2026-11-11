from pathlib import Path
exec(Path('tmp/compact_design_a.py').read_text().split('# Segmentation.')[0])
o=source('3_6_reproduction.tex')
collect(o,['For the \\texttt{V} in that Owen line','The refined right boundary','A traced point on this','For unit-length descriptor','In the generated Owen sentence','The final Owen profile','For the recorded Owen candidate','In the worked Owen example','The generated Owen trajectory','For this same generated file','Figure~\\ref{fig:vt-timeparts} applies'],'Glyph extraction and reproduction calculations')
s=r'''\subsection{Style extraction and handwriting reproduction}
\label{sec:des-reproduction}
Recognition returns text, but the gantry needs pen paths. The reproduction stage extracts reusable glyphs from labelled handwriting, stores each writer's shape and spacing measurements, and assembles them into new text.

\subsubsection{Choice of reproduction representation}
A glyph profile reuses the few labelled pages available for each writer. Haines et al. used approximately four A4 pages in their standard glyph-based experiments~\cite[p.~4]{haines2016}. The explicit paths can be inspected, resized and mapped to the gantry without training another generator.

The photographs contain no measured pen order, speed or pressure, unlike the trajectories required by recurrent synthesis~\cite[p.~18]{graves2013}. Image generators can use few style references, but still need a separate training corpus. One-DM's IAM experiment used 339 writers and four RTX 3090 GPUs~\cite[p.~9]{dai2024}. The chosen glyph approach fits the available data and compute, with weaker handling of letter shapes that change with their neighbours.

\subsubsection{Character alignment and glyph extraction}
\label{sec:des-alignment}
A source line has a known transcript but no character-position labels. The reader supplies frame probabilities, and forced alignment selects the highest-scoring path consistent with that transcript. It uses the CTC transition rules, replacing path summation with maximisation~\cite[pp.~4--5]{graves2006}.
'''+eq(o,'eq:des-viterbi')+r'''Here $\ell'$ is the blank-interleaved transcript and $\mathcal P(s)$ is the permitted predecessor set for state $s$. Backtracking gives each character's frame interval. A frame edge $t$ maps approximately to source position $(640t/T)/a$, using the inverse canvas scale $a$. Several plausible boundaries can still give the correct transcript.

The horizontal ink profile estimates the body band. Its first and last rows above 45\% of the maximum define the band top and baseline $b$, with difference $h_x$ giving estimated x-height. Bands below three pixels are rejected. Initial character cuts are refined towards thin connectors or blank columns using the project-specific cost
'''+eq(o,'eq:des-cutcost')+r'''where $\overline p_x$ is smoothed column ink, the denominator is its positive-count 60th percentile, and $y_{\rm top}(x)$ is the highest ink row. The height term favours shallow connectors near the baseline. A true blank column has zero cost. Cuts remain ordered, with minimum cell width $\max(1,0.08h_x)$, so neighbouring boundaries cannot collapse onto the same gap.

Figure~\ref{fig:des-alignment} follows the archived first Owen line because its crop and transcript can be traced through recognition and extraction. It shows the initial frame locations, refined cuts and resulting glyphs. The worked path and cut-cost substitutions are in Appendix~\ref{app:worked-design-calculations}.
'''+fig(o,'fig:des-alignment','Character alignment and refined glyph cuts.')+r'''\subsubsection{Skeleton tracing and pen path construction}
Thinning converts ink area into a one-pixel centreline~\cite[p.~237]{zhang1984}. The complete line is thinned before cutting it into cells, avoiding an artificial thinning boundary at every letter. A graph tracer splits paths at junctions, prunes short spurs and rejoins compatible ends below a $75^\circ$ continuation angle. Douglas--Peucker simplification removes unnecessary vertices within a bounded deviation~\cite{douglas1973}.

Isolated dots are recovered as short paths. Eligible detached crossbars are assigned to their letter rather than split across cells or counted as cursive joins. The resulting points are stored relative to left cut $x_L$, baseline $b$, x-height $h_x$ and slant $\phi$, using a shear transform~\cite{vinciarelli2001}.
'''+eq(o,'eq:des-glyphcoords')+r'''The vertical coordinate increases upwards. The trim shift $\delta$ is removed from the stored horizontal path and retained as lead-in spacing. Slant is removed for storage and reapplied at placement. This separates the letter shape from its source size and cell spacing.

Figure~\ref{fig:vt-tracing} shows the Owen fragment before and after thinning and tracing. Across its full line, simplification reduces 3738 vertices to 974 at 0.6 source-pixel tolerance. The graph gives geometric paths, while the writer's original stroke order remains unknown. The coordinate substitution is retained in Appendix~\ref{app:worked-design-calculations}.
'''+fig(o,'fig:vt-tracing','Ink, skeleton and simplified pen paths.')+r'''\subsubsection{Glyph selection and library quality}
Alignment can place a cut inside a letter even when the text is correct. The profile therefore filters unreliable lines and glyphs using width, height, stroke count, path length and allocated character advance. Ordinary widths are limited to 2.1 x-heights and wide letters to 2.9. Character-specific distributions provide further checks without forcing every letter to the same size.

A normalised $12\times12$ occupancy descriptor compares shape. A pooled character reference identifies malformed cuts, while pairwise distances within a writer's pool identify a representative real variant, called the medoid. A separate denoised prototype combines compatible strokes. The ordinary pool holds up to twelve variants per character. Algorithm~\ref{alg:des-profile} shows where these checks enter profile construction.
'''+block(o,'designalgorithm','alg:des-profile')+r'''Figure~\ref{fig:des-profile-example} compares the nine stored \texttt{a} variants for Owen, Dylan and Yeukita. The cloud shows their variation, the medoid is an actual example, and the prototype is a combined shape. The distance bars explain which variant is most typical within each writer. The exact descriptor and medoid calculations are retained in Appendix~\ref{app:worked-design-calculations}.
'''+fig(o,'fig:des-profile-example','Glyph variants and representatives for three writers.')+r'''Missing characters can use case-related shapes, similar forms or fallback glyphs. Source tags distinguish these choices from observed personal letterforms. They allow text generation to continue, but a fallback does not establish that the character was captured in that writer's style.

\subsubsection{Writer style measurement and storage}
\label{sec:des-profile}
The profile stores slant, x-height, ascender and descender extent, character advance, word gap and connection fraction. Slant is estimated by finding the shear that concentrates ink into columns. Dimensions are normalised by x-height to reduce dependence on photograph resolution.

Connectedness is the fraction of eligible adjacent non-space boundaries crossed by a traced centreline, motivated by contour-based writer features~\cite[p.~701]{bulacu2007}.
'''+eq(o,'eq:des-connectedness')+r'''The Owen example has 3 crossings among 39 eligible boundaries, giving 0.077. This is a property of the source ink, not a measured pen-lift sequence. Glyph paths and style measurements are stored together under the writer identity for later requests.

\subsubsection{Text layout and connected handwriting synthesis}
\label{sec:des-layout}
The synthesiser selects glyph variants, applies the stored style and places words within the available line width. Character advance and word gap come from the profile. Words wrap to the next line when the remaining width is insufficient.

For stored point $(u,v)$ and lead $l$, physical x-height $h_x^{\rm mm}$ converts the glyph to millimetres. Slant shear $s$ and baseline drift $d$ are applied about placement origin $(p_x,p_y)$.
'''+eq(o,'eq:des-placement')+r'''Small translations, rotations and scale changes provide variation. Drift changes gradually with persistence 0.85 and is bounded to $\pm0.25$ x-heights. A nominal 0.05 x-height clearance limits collisions. Better-ranked variants are sampled more often, with compatible entry heights preferred for joining.

The automatic anchoring strength $\lambda$ depends on median stored prior distance $\widetilde d$.
'''+eq(o,'eq:des-anchor')+r'''Lower-distance pools retain more personal variants. Larger distances increase reliance on an acceptable representative and then a generic form if needed. This is a heuristic, with source tags recording the resulting glyph choice.

Joining uses the writer's measured connection fraction, but is suppressed below 0.45. A join also requires a compatible source and gap no greater than 0.80 x-heights. A shallow connector joins the body strokes without attaching to dots or crossbars. A rendered probe applies bounded slant and extent corrections before synthesis. The placement and anchoring examples are retained in Appendix~\ref{app:worked-design-calculations}.

\subsubsection{Candidate selection for legibility and writer identity}
\label{sec:des-selection}
Different variant choices can change readability and writer identity. Each candidate is rendered at constant ink width and scored by the reader and stroke-normalised writer classifier. Character score $A$ and requested-writer probability $P$ form joint score $J$.
'''+eq(o,'eq:des-jointscore')+r'''The harmonic mean penalises a candidate weak in either score. The recorded Owen candidate has $A=18/19=0.95$ and $P=0.58$, giving $J=0.72$. Algorithm~\ref{alg:des-joint} gives the bounded search and repair sequence. A repair is retained only when the complete candidate score improves.
'''+block(o,'designalgorithm','alg:des-joint')+r'''Figure~\ref{fig:des-candidates} compares the three recorded trials of ``Writing with style.'' using the same profile and settings. Candidate 1 is selected because its writer score is stronger while retaining the same character score as candidate 3. The full values are in Appendix~\ref{app:worked-design-calculations}.
'''+fig(o,'fig:des-candidates','Candidate paths and their selection scores.')+r'''The scoring models guide this choice, so their confidence does not independently prove physical style similarity. More trials also add inference time and cannot recover a missing source letterform.

\subsubsection{Machine coordinate mapping and G-code generation}
\label{sec:des-gcode}
The selected trajectory is fitted inside the calibrated workspace. Ruled placement adjusts each line to the chosen ruling, while free placement retains its synthesised pitch. Character size and line spacing remain separate so that a tall descender need not change both.

Trajectory bounds $(x_0,y_0,x_1,y_1)$, origin $(o_x,o_y)$, margins $m_x,m_y$ and scale $a$ define the mapping. Offset $d_y$ selects a later starting line and $\Delta_y$ adjusts line spacing.
'''+eq(o,'eq:des-machinecoords')+r'''Both stored and machine coordinates use upward-increasing $y$. Clamping is a final travel limit and can distort a letter. Robust extent fitting and margins should keep the generated text inside the bed before execution. Nearby path segments may be simplified and compatible strokes joined, using a default join distance of 0.6\,mm.

Figure~\ref{fig:des-gcode-excerpt} gives the opening commands of the generated Owen example. It selects millimetres, absolute coordinates, a raised-pen travel and the first drawing movements. The full point mapping and file interpretation are extended in Appendix~\ref{app:worked-design-calculations}.
'''+fig(o,'fig:des-gcode-excerpt','Opening commands of the Owen example.')+r'''Travel feed is restated after drawing because feed is modal. Pen dwell adds 0.35\,s per transition to the switch-confirmed actuation. For drawing and travel distances $D_d,D_u$, feeds $F_d,F_u$ and $N_p$ alternating pen transitions, the nominal budget is
'''+eq(o,'eq:des-timebudget')+r'''The earlier traced file gives 32.82\,s, including 19.60\,s dwell. Figure~\ref{fig:vt-timeparts} shows a separate four-profile calculation for the same sentence. Dwell contributes about 61 to 65\% of these nominal budgets. This identifies a motion-side improvement to test, while actual completion also includes synthesis, actuation and pulse-loop overhead.
'''+fig(o,'fig:vt-timeparts','Drawing, travel and pen-dwell budgets.')+r'''\subsubsection{Reproduction verification and design limitations}
Checks follow source ink, stored glyphs, assembled paths and emitted G-code. The independent file parser in Appendix~\ref{app:des-visual-tests} recovers the generated paths. Nine controlled fixtures preserve vertex counts with maximum coordinate discrepancy 0.0004\,mm, consistent with three-decimal coordinate emission. Physical drawing tests then check the mechanism rather than file conversion alone.

The explicit glyph representation makes source errors and scaling inspectable. Its main limits are context-dependent letterforms, missing characters and the absence of natural pen dynamics. Multi-character units and recorded trajectories could address these losses, but need additional reliable data. Physical qualification remains in Results.
\FloatBarrier
'''
(B/'3_Main_Report/design/3_6_reproduction.tex').write_text(s,encoding='utf-8')
oldm=json.loads((R/'tmp/final_report_review/moved_calculations.json').read_text());oldm.extend(moved)
(R/'tmp/final_report_review/moved_calculations.json').write_text(json.dumps(oldm,indent=2),encoding='utf-8')
print('Reproduction condensed without removing its subsections, formulas or figures.')


