from pathlib import Path
import re,json
P=Path('Report/SubmissionRevision_2026-11-02'); R=P/'3_Main_Report'; B=Path('tmp/final_report_review/baseline/3_Main_Report')
s=(R/'4_Results.tex').read_text()
# Condense summary while keeping its heading and label.
a=s.index('Table~\\ref{tab:res-summary}'); b=s.index('\\subsection{Qualification tests}')
s=s[:a]+r'''Table~\ref{tab:res-summary} compares the measured outcomes with the Project Proposal. The qualification tests below give the procedures and evidence.
\begin{table}[!htbp]\centering\small
\begin{tabular}{p{.27\linewidth}p{.54\linewidth}p{.12\linewidth}}\toprule
Requirement & Measured outcome & Assessment\\\midrule
R1. At least 85\% character accuracy and style correlation & Physical character accuracy 84.41\%. Seven of eleven writers exceed 85\%. Shape correlation 0.697. & Below pooled targets\\[4pt]
R2. At least 95\% text and writer accuracy & Reader 94.06\%. Writer identification 153/156, or 98.08\%. & Writer target met\\[4pt]
R3. At most 4 s per character for reproduction & All 25 completion-confirmed ODROID jobs pass. Worst 3.875 s per character. & Met in tested jobs\\[4pt]
R4. Classification within $2+0.025x$ s & Two ODROID requests take 46.26 and 55.49 s, against 14.50 and 26.55 s allowed. & Not met\\[4pt]
R5. At most 0.5 mm positional error & Seven measured feature lengths have a maximum absolute error of 0.37 mm. Absolute placement and repeatability need further measurements. & Dimensional check met\\[4pt]
R6. At least ten stored styles & Eleven profiles contain glyph mappings and are exercised in physical reproduction. & Met\\[4pt]
Field conditions. Well-lit, stable bench & Indoor captures and drawings demonstrate the fixed setup. No calibrated lux or vibration series was recorded. & Observed setup only\\\bottomrule
\end{tabular}
\caption{Results against the proposal.}\label{tab:res-summary}
\end{table}
\FloatBarrier

'''+s[b:]
# Replace each qualification step's body with a concise version. Preserve every heading.
starts=list(re.finditer(r'\\reportqt\{qt:r(\d)\}\{[^\n]+',s))
parts={}
for i,m in enumerate(starts):
 end=starts[i+1].start() if i+1<len(starts) else s.index('\\subsubsection{Qualification under')
 parts[int(m[1])]=s[m.start():end]
def getstep(q,title):
 st=parts[q]; pat=r'\\resstep\{'+re.escape(title)+r'\}\s*'; m=re.search(pat,st); end=re.search(r'\\resstep\{',st[m.end():]); return st[m.end():m.end()+end.start()] if end else st[m.end():]
def replace(q,title,body):
 global s
 old=getstep(q,title); s=s.replace(old,body.strip()+'\n\n',1)
O='Objectives of the test or experiment'; E='Equipment used'; T='Test setup and experimental parameters'; F='Steps followed in the test or experiment'; M='Results or measurements'; V='Observations'; A='Statistical analysis'
# Shared method section.
a=s.index('The final evaluation uses');b=s.index('\\reportqt{qt:r1}')
orig=s[a:b]; eq=re.search(r'\\begin\{equation\}.*?\\end\{equation\}',orig,re.S)[0]
s=s[:a]+r'''The evaluation uses six IAM writers (150, 152, 153, 384, 552 and 588) and five personal contributors (A, B, O, T and Y). Fourteen physical pages provide 184 transcript-paired lines. A separate model comparison includes one additional unpaired crop and therefore has 185 crops.

The gantry produced the pages under ODROID control. Saved images were scored on the development host. Embedded timings come from the preserved ODROID log. The selected joint reader and eleven-writer classifier use the configuration described in Design, with writer stroke width 0.17. Appendix~\ref{app:res-evidence} identifies the measured checkpoints and source records.

Character accuracy uses the summed Levenshtein distance $d_{\rm edit}$ between recognised text $\hat t_i$ and reference $t_i$, following the character-error convention~\cite[p.~2]{graves2006}.
'''+eq+r'''
The physical result is $100(1-853/5471)=84.41\%$. Writer accuracy is the proportion of correctly identified lines. Page pooling is reported separately. Word match counts equal words at corresponding positions, so it differs from word edit accuracy. Full page and line records are retained in Appendix~\ref{app:res-page-records}.

Wilson intervals describe line-classification uncertainty~\cite[pp.~209--212]{wilson1927}. Lines share writers and capture conditions, so these are descriptive intervals. Resampling whole writer groups provides a separate sensitivity check on the physical character and style scores.

'''+s[b:]
replace(1,O,'Measure legibility and writer-style retention of new digital text after physical reproduction. Compare character accuracy and geometric style correlation with the two 85\\% targets.')
replace(1,E,'The calibrated gantry, ODROID N2+, IMX179 camera, stored profiles and selected models. Saved photographs are $3264\\times2448$ pixels.')
replace(1,T,'Each writer reproduces a common paragraph, a pangram and available source text. Transcripts follow the photographed line breaks. The final stored glyph mappings, greedy CTC decoding and 0.17 writer normalisation are used.')
replace(1,F,r'Generate and draw each job, photograph it, pair the segmented lines with transcripts, then save reader edits and writer predictions. Sum the counts by page, text block and writer. Extract the same geometric descriptors from original and reproduced lines for the style comparison.')
body=getstep(1,M)
# Keep its tables and figures but remove long prose following figures and simplify preceding introduction.
bb=body.index('The automated reader')
body=body[:bb]+r'''Overall character accuracy is 84.41\%, positional word match is 45.97\% and writer identification is 159/184 (86.41\%). All fourteen recorded page pools are correctly attributed. Seven writers exceed 85\% character accuracy. The complete Owen page in Figure~\ref{fig:res-owen-page} connects the line examples to a physical sheet. The full fourteen-page set and page scores are in Appendix~\ref{app:res-page-records}.
\begin{figure}[!htbp]\centering
\includegraphics[height=.43\textheight,keepaspectratio]{Figures/PhysicalPages/AuthorO.jpg}
\caption{Owen's reproduced page.}\label{fig:res-owen-page}
\end{figure}

Figure~\ref{fig:res-reproduced-confusion} compares the physical-output writer predictions. Each row sums to 100\%, giving each writer equal visual weight despite different line counts. The row labels retain the actual sample sizes. This normalisation changes the display, not the measured 159 correct predictions.
\begin{figure}[!htbp]\centering
\includegraphics[width=.82\linewidth]{Figures/results_reproduced_confusion.pdf}
\caption{Writer identification on reproduced lines, row percentages.}\label{fig:res-reproduced-confusion}
\end{figure}

The geometric comparison uses 156 original and 184 physical lines across all eleven writers. Nine descriptors include ink density and stroke width. Seven shape descriptors omit these pen-dependent quantities. Table~\ref{tab:res-physical-style} gives the correlations. Original and reproduced sentences are not matched, so this measures writer-level geometry across the available samples.
\input{3_Main_Report/results_tables/style_physical}

Slant correlation is 0.991, ascender extent 0.817, character width 0.798 and word gap 0.486. Separately standardising the seven features in each domain gives pooled shape correlation 0.697. Using original-domain scaling for both gives 0.611. Appendix~\ref{app:res-physical-style} retains the definitions and full descriptor values.
'''
replace(1,M,body)
replace(1,V,r'The pooled character score is 0.59 percentage points below target. Writer scores range from 60.97\% to 97.01\%. Legibility and identity measure different properties. T is identified on every line but has the lowest character score, while 150 scores 96.75\% character accuracy with 53.85\% identification. More separated original handwriting is associated with better reproduced character accuracy across these writers (Spearman $\rho=0.845$), without isolating the cause of the difference.')
replace(1,A,r'The Wilson interval for 159/184 identifications is 80.7\% to 90.6\%. Resampling eleven whole writers 10\,000 times gives a central 95\% range of 77.82\% to 90.27\% for character accuracy and 0.555 to 0.827 for shape correlation. These describe sensitivity to the observed writer set. Neither pooled target is met by the observed point estimate.')
replace(2,O,r'Evaluate reading and writer identification separately against 95\%, using the recorded validation partitions.')
replace(2,E,'Saved IAM and personal line images, reference transcripts, selected checkpoints and the scoring tools. This experiment does not require gantry movement.')
replace(2,T,r'The reader population has 976 public validation lines (42\,072 characters) and 100 photographed lines (3\,675 characters). The writer population has 156 lines, using held-out IAM pages and a seeded personal-line split. Personal validation can share page backgrounds with training. These are validation results because checkpoint selection may have used the same split.')
replace(2,F,r'Check the corpus partitions, read each line with greedy CTC decoding, classify the normalised writer crop and retain the edits, reference counts and predictions. Aggregate the reader populations and writer confusion counts separately.')
replace(2,M,r'''Table~\ref{tab:res-r2-reader} reports 94.06\% pooled character accuracy for the selected joint reader. The scratch comparison reaches 94.39\% on the same 1\,076 lines. The training-route comparison is in Table~\ref{tab:dis-training}.
\input{3_Main_Report/results_tables/r2_reader}
The writer model correctly identifies 153/156 lines (98.08\%) and all eleven pooled writer groups. Figure~\ref{fig:res-holdout} shows the row-normalised predictions, with actual line counts beside each writer. The only three errors are predictions of Y.
\begin{figure}[!htbp]\centering
\includegraphics[width=.78\linewidth]{Figures/results_holdout_confusion.pdf}
\caption{Writer validation predictions, row percentages.}\label{fig:res-holdout}
\end{figure}''')
replace(2,V,r'The writer point estimate exceeds target by 3.08 percentage points. The selected reader is 0.94 points below target and recovers 221 complete lines without edits. One incorrect writer prediction has 83.8\% confidence, showing that a high posterior alone cannot guarantee identity.')
replace(2,A,r'The writer Wilson interval is 94.5\% to 99.3\%. The result demonstrates a point estimate above 95\%, with a stronger pooled decision across the available lines. Character accuracy is an edit-distance ratio, so a binomial interval on individual characters is not assigned.')
replace(3,O,r'Compare total generation and drawing time with the proposal allowance of $4x$ seconds for $x$ characters.')
replace(3,E,'ODROID N2+, gantry and the server timer. The preserved log supplies the measured totals and completion status.')
replace(3,T,'Retain positive-character jobs with a drawing stage and an explicit program-end marker. Exclude stopped jobs, commissioning shapes and entries without confirmed completion. Generation time is carried into the recorded drawing-request total.')
replace(3,F,r'Associate each timing block with its terminal status, retain completed jobs, calculate $t/x$ and compare the recorded total $t$ with $4x$. Group the retained records by text length.')
replace(3,M,r'''Table~\ref{tab:res-r3-confirmed} summarises 25 completed requests. All pass. The worst is $166.61/43=3.875$ s per character, leaving 5.39 s. At 378 characters, nine totals range from 507.57 to 1085.01 s against 1512 s allowed. Figure~\ref{fig:res-r3} and the complete records are retained in Appendix~\ref{app:res-speed}.
\input{3_Main_Report/results_tables/r3}''')
replace(3,V,r'The shortest tested length has a minimum margin of 3.1\%, compared with 28.2\% for the longest. Timing therefore depends on both fixed generation work and the drawn path. Completion and timing do not replace the separate fidelity checks.')
replace(3,A,r'All 25 observations pass, with a descriptive Wilson interval of 86.7\% to 100\%. The records cover six lengths from 43 to 378 characters, with one to nine observations at each length. They do not establish a timing envelope for shorter requests.')
replace(4,O,r'Compare the complete embedded classification request with $2+0.025x$ seconds and identify the stages responsible for the difference.')
replace(4,E,'ODROID server log and stage timer. Supporting acceleration and resolution experiments use saved pages on the development host.')
replace(4,T,'The two ODROID requests log 500 characters over 16 lines and 982 over 22 lines. Totals include loading, conditioning, segmentation, cropping and both networks. Allowances use those logged counts. Camera acquisition outside the timer is not included.')
replace(4,F,'Read the totals and component stages, calculate the allowances and compare them. In separate host experiments, compare sequential and batched network execution on identical saved pages, then check the consequences of reducing image resolution.')
replace(4,M,r'''Table~\ref{tab:res-r4-times} gives the ODROID measurements. Neither request meets R4. Networks account for 67.6\% of their combined time. Figure~\ref{fig:res-r4} and the supporting comparisons are in Appendix~\ref{app:res-speed}.
\input{3_Main_Report/results_tables/r4}
The host network comparison takes 359.8 s sequentially and 130.3 s with equivalent batched modules, a factor of 2.76. Reader text agrees on every evaluated line. This changes both backend and batching. Halving image resolution is faster but loses lines and produces only 33.8\% mean text agreement, so it is unsuitable as an equivalent-output replacement.''')
replace(4,V,r'The measured totals are 3.19 and 2.09 times their allowances. Even projecting the host network factor onto these requests, with unchanged image processing, gives approximately 27.05 and 30.82 s. Both remain above target. The projection identifies further work rather than an embedded pass.')
replace(4,A,'Two requests show two observed overruns. They do not establish a general timing distribution. The fourteen-page host comparison supports an optimisation route, but its embedded gain requires a direct ODROID measurement.')
replace(5,O,'Check the calibrated geometry against the 0.5 mm dimensional limit and distinguish feature lengths from absolute placement and repeatability.')
replace(5,E,'Calibrated gantry, pen, paper and a metal non-digital vernier caliper. Its model and least count are not recorded.')
replace(5,T,'One 100 mm square and one triangle of 80 mm base and nominal 40 mm height were drawn after calibration. Four square sides and three triangle sides were measured. Pen width and caliper reading affect the measurement.')
replace(5,F,'Execute the geometric jobs, measure each side and subtract its commanded length. Compare the errors with 0.5 mm and inspect the unequal triangle sides for geometric asymmetry.')
# Move apex derivation to existing appendix.
old=getstep(5,M); ap=old[old.index('Place the measured'):]
Path('tmp/final_report_review/apex_calculation.tex').write_text(ap)
replace(5,M,r'''Table~\ref{tab:res-r5-lengths} retains all seven readings. Mean absolute square-side error is $(0.30+0.20+0.10+0.10)/4=0.175$ mm. The largest difference is $|56.20-\sqrt{40^2+40^2}|=0.369$ mm. Figure~\ref{fig:des-shapetest} shows the actual specimen beside its commanded geometry. Figure~\ref{fig:res-geometry} and the triangle diagnostic are retained in Appendix~\ref{app:res-geometry-calculation}.
\input{3_Main_Report/results_tables/r5}''')
replace(5,V,'Every measured feature length is within 0.5 mm of nominal. The unequal triangle slopes suggest geometric asymmetry, but these measurements do not separate rail alignment, scale, pen width and reading uncertainty.')
replace(5,A,'There is one specimen of each shape, so variation between sides is not repeated-draw variability. Length checks leave translation undetected. These results support dimensional accuracy, while absolute bed coordinates and repeated-word overlays remain separate checks.')
# R6 already concise. Remove excessive detail from field conditions.
a=s.index('Indoor captures and physical drawings',s.index('\\subsubsection{Qualification under'))
s=s[:a]+r'''The indoor captures and physical drawings demonstrate the fixed bench arrangement. The capture variations in Section~\ref{sec:des-segverify} support operation under different photographed conditions. A calibrated illuminance and vibration series is not present, so no numerical coverage of the full field-condition range is assigned.
\FloatBarrier
'''
# Captions shortened after explanatory prose.
s=re.sub(r'\\caption\{Physical reproduction by writer\..*?\}',r'\\caption{Physical reproduction scores by writer.}',s,flags=re.S)
s=re.sub(r'\\caption\{Stored crops from photographed robot pages.*?\}',r'\\caption{Examples from four physical writing styles.}',s,flags=re.S)
s=s.replace('Their different lengths and content are retained in the analysis.','')
(R/'4_Results.tex').write_text(s)
# two minor Design corrections
p=R/'design/3_4_segmentation.tex';t=p.read_text();t=t.replace('a=0.29','a\\simeq0.295');p.write_text(t)
p=R/'design/3_6_reproduction.tex';t=p.read_text();needle='\\subsubsection{';# insert in medoid discussion without altering headings
m=re.search(r'[^\n]*medoid[^\n]*',t)
if m and 'd(g,c)' not in t:
 t=t[:m.end()]+r' The descriptor distance is $d(g,c)=1-v_g^Tr_c$ for unit-normalised glyph descriptor $v_g$ and class reference $r_c$. The medoid minimises summed within-class distances.'+t[m.end():]
p.write_text(t)
print('Results words',len(s.split()))
