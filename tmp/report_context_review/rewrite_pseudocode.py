from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'Report/DesignVisualRevision_2026-11-01'
blocks={
'alg:des-stepping':r'''
\prequire{Signed axis counts; pulse delay; connected GPIO interface.}
\begin{designpseudo}
\pstate{0}{$(N_M,N_m)\leftarrow$ major and minor absolute counts; set directions}
\pstate{0}{$e\leftarrow\lfloor N_M/2\rfloor$}
\pstate{0}{\textbf{for} $i\leftarrow1$ \textbf{to} $N_M$ \textbf{do}}
\pstate{1}{\textbf{if} abort requested or end stop confirmed \textbf{then}}
\pstate{2}{\textbf{return} interrupted, current position counters}
\pstate{1}{\textbf{end if}}
\pstate{1}{Pulse major axis; update its signed counter}
\pstate{1}{$e\leftarrow e-N_m$}
\pstate{1}{\textbf{if} $e<0$ \textbf{then}}
\pstate{2}{Pulse minor axis; update its signed counter; $e\leftarrow e+N_M$}
\pstate{1}{\textbf{end if}}
\pstate{0}{\textbf{end for}}
\pstate{0}{\textbf{return} completed, current position counters}
\end{designpseudo}''',
'alg:des-segment':r'''
\prequire{Grey page on its chosen line axis.}
\pensure{Ordered full-resolution crops, coordinates and diagnostics.}
\begin{designpseudo}
\pstate{0}{$(G_f,G_h,M,R,\theta)\leftarrow\operatorname{Deskew}(G)$}
\pstate{0}{$M\leftarrow\operatorname{RemovePrintedStructure}(M,G_h)$}
\pstate{0}{$(M,\mathcal C)\leftarrow\operatorname{DropArtifactPieces}(M,R)$}
\pstate{0}{$(P,\mathbf c)\leftarrow\operatorname{LineCentres}(M)$}
\pstate{0}{$\mathbf b\leftarrow\operatorname{SeamCuts}(M,\mathbf c,P)$}
\pstate{0}{$O\leftarrow\operatorname{AssignOwnership}(M,\mathcal C,\mathbf b,\mathbf c,P)$}
\pstate{0}{$\mathcal L\leftarrow\varnothing$; normalise full-resolution brightness}
\pstate{0}{\textbf{for each} line index $n$ \textbf{do}}
\pstate{1}{$C_n\leftarrow\operatorname{BuildLineCrop}(n,O,G_f,P)$}
\pstate{1}{\textbf{if} $C_n$ has sufficient ink and height \textbf{then}}
\pstate{2}{Append $C_n$, its fixed-canvas view and bounding box to $\mathcal L$}
\pstate{1}{\textbf{end if}}
\pstate{0}{\textbf{end for}}
\pstate{0}{$\mathcal L\leftarrow\operatorname{DropSolidCrops}(\mathcal L)$; renumber retained crops}
\pstate{0}{\textbf{return} $\mathcal L$, seams, ratio image and diagnostics}
\end{designpseudo}''',
'alg:des-ownership':r'''
\prequire{Ink mask $M$, components $\mathcal C$, seams, centres and pitch $P$.}
\begin{designpseudo}
\pstate{0}{\textbf{for each} component $C\in\mathcal C$ \textbf{do}}
\pstate{1}{Count overlap $a_{C,n}$ with each seam band $n$}
\pstate{1}{$S\leftarrow\{n:a_{C,n}\geq0.15\sum_j a_{C,j}\}$}
\pstate{1}{Choose topmost $S$ as reference owner, or maximum overlap if $S=\varnothing$}
\pstate{1}{Correct eligible narrow tails and small pieces; reject remote specks}
\pstate{1}{Associate compact dots with plausible stems using pitch-scaled guards}
\pstate{0}{\textbf{end for}}
\pstate{0}{$O\leftarrow$ corrected ownership, preserving large-piece seam assignments}
\pstate{0}{\textbf{for each} raw component near a neighbouring line \textbf{do}}
\pstate{1}{\textbf{if} detached-dot or enclosed-ring guards pass \textbf{then}}
\pstate{2}{Transfer eligible pixels to the supported stem's line; protect correction}
\pstate{1}{\textbf{end if}}
\pstate{0}{\textbf{end for}}
\pstate{0}{Reassign residual fragments lacking their own body-band support}
\pstate{0}{\textbf{return} $O$; excluded pixels retain no line index}
\end{designpseudo}''',
'alg:des-ctc':r'''
\prequire{Logits $z\in\mathbb R^{T\times B\times C}$ and encoded targets.}
\begin{designpseudo}
\pstate{0}{$\log p\leftarrow\operatorname{LogSoftmax}(z)$; $L\leftarrow0$; $g\leftarrow0$}
\pstate{0}{\textbf{for each} batch item $b$ \textbf{do}}
\pstate{1}{$\ell'\leftarrow$ target interleaved with blanks}
\pstate{1}{Initialise $\alpha,\beta$ to $-\infty$; admit first blank and first label}
\pstate{1}{\textbf{for} $t\leftarrow1$ \textbf{to} $T-1$ \textbf{do}}
\pstate{2}{\textbf{for each} state $s$ \textbf{do}}
\pstate{3}{$\alpha_{t,s}\leftarrow\log p_t(\ell'_s)+\operatorname{LSE}_{r\in\mathcal P(s)}\alpha_{t-1,r}$}
\pstate{2}{\textbf{end for}}
\pstate{1}{\textbf{end for}}
\pstate{1}{$\log Z\leftarrow\operatorname{LSE}$ of the admissible final states}
\pstate{1}{Initialise final $\beta$ to zero; recurse backward with next-frame emissions}
\pstate{1}{$q_{t,k}\leftarrow\sum_{s:\ell'_s=k}\exp(\alpha_{t,s}+\beta_{t,s}-\log Z)$}
\pstate{1}{$L\leftarrow L-\log Z$; $g_{t,b,k}\leftarrow(p_t(k)-q_{t,k})/B$}
\pstate{0}{\textbf{end for}}
\pstate{0}{\textbf{return} $L/B,g$; impossible alignments follow the implementation's guard}
\end{designpseudo}''',
'alg:des-train':r'''
\prequire{Training batch, task, parameters $\theta$ and optimiser state.}
\begin{designpseudo}
\pstate{0}{$(X,y)\leftarrow$ task-specific preparation and training augmentation}
\pstate{0}{$(F,\mathcal K)\leftarrow\operatorname{BackboneForward}(X,\theta)$; retain caches}
\pstate{0}{\textbf{if} task is reader \textbf{then}}
\pstate{1}{$(z,\mathcal K_h)\leftarrow\operatorname{BiLSTMHead}(F)$; $(L,g)\leftarrow\operatorname{CTC}(z,y)$}
\pstate{0}{\textbf{else}}
\pstate{1}{$(z,\mathcal K_h)\leftarrow\operatorname{WriterHead}(\operatorname{MeanWidth}(F))$}
\pstate{1}{$(L,g)\leftarrow\operatorname{CrossEntropy}(z,y)$}
\pstate{0}{\textbf{end if}}
\pstate{0}{$g_F\leftarrow\operatorname{HeadBackward}(g,\mathcal K_h)$}
\pstate{0}{$\nabla\theta\leftarrow\operatorname{BackboneBackward}(g_F,\mathcal K)$}
\pstate{0}{\textbf{if} task is writer \textbf{then} clip total gradient norm to $5$}
\pstate{0}{$(\theta,\text{state})\leftarrow\operatorname{OptimiserStep}(\theta,\nabla\theta,\text{state})$}
\pstate{0}{\textbf{if} epoch is complete \textbf{then}}
\pstate{1}{Validate with fixed preparation, stored normalisation and dropout disabled}
\pstate{1}{Update schedule; save progress; replace best only if its criterion improves}
\pstate{0}{\textbf{end if}}
\pstate{0}{\textbf{return} $L,\theta,\text{state}$}
\end{designpseudo}''',
'alg:des-profile':r'''
\prequire{Labelled original lines $\mathcal L$, trained reader and writer identifier.}
\begin{designpseudo}
\pstate{0}{$\mathcal G\leftarrow\varnothing$; $\mathcal S\leftarrow\varnothing$}
\pstate{0}{\textbf{for each} $(G,\ell)\in\mathcal L$ \textbf{do}}
\pstate{1}{$A\leftarrow\operatorname{ForcedAlign}(G,\ell)$; estimate baseline, x-height and slant}
\pstate{1}{\textbf{if} alignment or body band is unusable \textbf{then continue}}
\pstate{1}{Refine shared increasing cuts; thin full line; trace skeleton paths}
\pstate{1}{Split paths at cuts; assign crossbars; recover untraced dots}
\pstate{1}{Normalise glyph coordinates; record advance, lead and boundary crossings}
\pstate{1}{Append glyphs to $\mathcal G$ and line measurements to $\mathcal S$}
\pstate{0}{\textbf{end for}}
\pstate{0}{Cache raw extraction; reject low-confidence lines relative to this writer}
\pstate{0}{\textbf{for each} character's glyph pool $\mathcal G_c$ \textbf{do}}
\pstate{1}{Apply size, height-class, path-length and cell-consistency gates}
\pstate{1}{Refine size range; apply cross-writer prior and within-writer consensus}
\pstate{1}{Rank and smooth retained variants; construct medoid and eligible prototype}
\pstate{0}{\textbf{end for}}
\pstate{0}{\textbf{return} persistent profile containing $\mathcal G$ and robust summaries of $\mathcal S$}
\end{designpseudo}''',
'alg:des-joint':r'''
\prequire{Text, writer profile, reader, shape classifier, seed $s$ and trial budget $K$.}
\begin{designpseudo}
\pstate{0}{$J^*\leftarrow-1$; $R\leftarrow\varnothing$; determine anchoring strength}
\pstate{0}{\textbf{for} $k\leftarrow0$ \textbf{to} $\max(1,K)-1$ \textbf{do}}
\pstate{1}{$T\leftarrow\operatorname{SynthesizeText}(\text{text},\text{profile},s+977k)$}
\pstate{1}{$I\leftarrow\operatorname{RenderUniformInk}(T)$; $(A,P,J)\leftarrow\operatorname{Score}(I)$}
\pstate{1}{\textbf{if} $J>J^*$ \textbf{then} $(T^*,I^*,J^*)\leftarrow(T,I,J)$}
\pstate{1}{\textbf{if} $J^*\geq0.99$ \textbf{then break}}
\pstate{0}{\textbf{end for}}
\pstate{0}{\textbf{for} at most three configured repair rounds \textbf{do}}
\pstate{1}{\textbf{if} $J^*\geq0.995$ \textbf{then break}}
\pstate{1}{$W\leftarrow$ up to two weak aligned positions not already in $R$}
\pstate{1}{\textbf{if} $W=\varnothing$ \textbf{then break}}
\pstate{1}{$R'\leftarrow R\cup W$; regenerate with stronger anchors at $R'$ and seed $s$}
\pstate{1}{Render and rescore the complete trial trajectory}
\pstate{1}{\textbf{if} $J>J^*$ \textbf{then} commit $R'$ and the new best trajectory}
\pstate{0}{\textbf{end for}}
\pstate{0}{\textbf{return} $T^*$ with selection and accepted-repair metadata}
\end{designpseudo}''',
'alg:des-machine':r'''
\prequire{Selected trajectory $T$ and calibrated workspace configuration.}
\begin{designpseudo}
\pstate{0}{Measure bounds; determine available width and height after margins and offset}
\pstate{0}{$a\leftarrow$ uniform fit scale; apply robust height cap when ruled fitting is enabled}
\pstate{0}{Group paths by text row; determine ruled or free baseline shifts}
\pstate{0}{$\mathcal P\leftarrow$ machine coordinates from Equation~\ref{eq:des-machinecoords}}
\pstate{0}{Decimate short segments; optimise path order within rows}
\pstate{0}{Clamp remaining out-of-bounds points; record their count}
\pstate{0}{Emit absolute millimetre mode and initial pen state}
\pstate{0}{\textbf{for each} path $p\in\mathcal P$ \textbf{do}}
\pstate{1}{Close eligible near-ring; join previous path if configured gap permits}
\pstate{1}{\textbf{if} a new pen-down path is needed \textbf{then}}
\pstate{2}{Raise pen; travel to start; lower pen; emit configured dwells}
\pstate{1}{\textbf{end if}}
\pstate{1}{Emit drawing moves with modal drawing feed; accumulate ink and travel budgets}
\pstate{0}{\textbf{end for}}
\pstate{0}{Raise pen; save file, placement and separate timing budgets}
\pstate{0}{\textbf{return} file path and job metadata}
\end{designpseudo}'''
}
# Preserve captions and labels; replace only algorithm bodies.
seen=[]
for p in (HERE/'3_Main_Report/design').glob('*.tex'):
    s=p.read_text(encoding='utf-8')
    def replace(m):
        block=m.group(0);label=re.search(r'\\label\{([^}]+)\}',block).group(1)
        if label not in blocks:raise ValueError(label)
        seen.append(label)
        prefix=block[:re.search(r'\\label\{[^}]+\}',block).end()]+'\n'
        return prefix+blocks[label].strip()+'\n\\end{designalgorithm}'
    s=re.sub(r'\\begin\{designalgorithm\}.*?\\end\{designalgorithm\}',replace,s,flags=re.S)
    p.write_text(s,encoding='utf-8')
print('Converted',seen)
