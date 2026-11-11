from pathlib import Path
from pypdf import PdfReader,PdfWriter
import shutil,json,hashlib
b=Path('Report/IntegratedReport_2026-11-01');v=b.parent/'DesignVisualTests_2026-11-01';out=b/'Figures/VisualTests';out.mkdir(exist_ok=True)
selected=['T25_seam_cost_field','T27_ownership_corrections','T32_feature_stack','T36_ctc_lattice','T38_gradient_check','T40_stroke_width_sweep','T41_skeleton_trace_simplify','T43_glyph_coordinates','T45_variant_pool_medoid','T49_connectedness','T52_bresenham_staircase','T56_time_budget']
for n in selected:
 if n.startswith('T56'):continue
 p=PdfReader(v/'figures'/(n+'.pdf')).pages[0];stamps=[]
 def visit(t,cm,tm,f,size):
  if 5.3<size<5.5 and t.strip():stamps.append(cm[5]+tm[5])
 p.extract_text(visitor_text=visit)
 assert stamps,n
 cut=max(stamps)+8;p.mediabox.lower_left=(0,cut);p.cropbox.lower_left=(0,cut)
 w=PdfWriter();w.add_page(p);w.write(out/(n+'.pdf'))
e=b/'VisualTestEvidence';e.mkdir(exist_ok=True)
for folder in ['scripts','data']:shutil.copytree(v/folder,e/folder,dirs_exist_ok=True)
shutil.copy2(v/'MANIFEST.md',e/'ORIGINAL_MANIFEST.md')
backup=Path('tmp/report_context_review/pre_visual_additions');backup.mkdir(exist_ok=True)
paths=['3_Main_Report/design/3_3_hardware.tex','3_Main_Report/design/3_4_segmentation.tex','3_Main_Report/design/3_5_classification.tex','3_Main_Report/design/3_6_reproduction.tex','Appendices/design_subsystem_evidence.tex']
for q in paths:shutil.copy2(b/q,backup/Path(q).name)
def fig(stem,label,caption,width='0.96'):
 return '\n\\begin{figure}[!htbp]\\centering\n\\includegraphics[width='+width+'\\linewidth]{Figures/VisualTests/'+stem+'.pdf}\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\end{figure}\n\n'
def before(s,heading,text):
 needle='\\subsubsection{'+heading+'}';assert needle in s,heading;return s.replace(needle,text+needle,1)
p=b/paths[0];s=p.read_text(encoding='utf-8')
s=before(s,'Pen lift mechanism and drive circuit',r'''Figure~\ref{fig:vt-stepping} makes the integer pulse distribution visible on another segment from a generated job, with count changes $(+6,-28)$. The major axis advances at every iteration, while the accumulator distributes the six minor-axis pulses. The plotted deviation from the ideal line reaches 0.5 microsteps, equivalent to $0.5/80=0.00625$\,mm at the nominal scale. This is the quantisation of the commanded path, not the mechanical error of the pen.
'''+fig('T52_bresenham_staircase','fig:vt-stepping','Integer stepping and command-space deviation for one generated move; evaluated offline.'))
p.write_text(s,encoding='utf-8')
p=b/paths[1];s=p.read_text(encoding='utf-8')
s=before(s,'Ink ownership and classifier crop preparation',r'''Figure~\ref{fig:vt-seamfield} shows why the curved boundary is useful on Abhinav01 at native page resolution. Among that page's gaps, the first example was selected for the largest reduction in crossed ink: the midpoint cut intersects 64 working-image ink pixels, while the seam intersects one. The second was selected for the largest departure from the midpoint; its seam moves by up to 22 rows and crosses no ink, compared with 44 pixels for the straight cut. These selected examples expose the effect of the cost function rather than establishing an average segmentation improvement.
'''+fig('T25_seam_cost_field','fig:vt-seamfield','Seam paths and local cost fields on two explicitly selected gaps from Abhinav01.'))
s=before(s,'Segmentation verification and limitations',r'''The three largest ownership corrections on Abhinav01 are displayed in Figure~\ref{fig:vt-ownership}. Colour identifies the assigned line, and the boxes locate pixels whose ownership changes after the component rules. Across the page, 462 pixels change assignment; the three boxed components contain 115, 78 and 73 pixels. The examples make the detached-piece and protruding-stroke corrections inspectable. No manually labelled pixel-ownership reference was supplied, so the reassignment count measures the action of the rules rather than their accuracy.
'''+fig('T27_ownership_corrections','fig:vt-ownership','Initial seam-band ownership and corrected component ownership on the same page.'))
p.write_text(s,encoding='utf-8')
p=b/paths[2];s=p.read_text(encoding='utf-8-sig')
s=before(s,'Text recognition using sequence modelling and CTC',r'''The arrays behind the architecture are shown in Figure~\ref{fig:vt-featurestack}, using the joint reader on the archived Owen line. Each row shows three channels selected for their activation strength at that stage; the maps are independently scaled for visibility, so their darkness is not a comparison of absolute response across layers. After the two spatial pooling stages, the final tensor is $128\times16\times160$. Taking the maximum over height gives $128\times1\times160$: 160 ordered feature columns, each containing 128 values. This links the convolutional representation to the sequence input without treating an output frame as a complete character.
'''+fig('T32_feature_stack','fig:vt-featurestack','Feature maps and height pooling from a real line passed through the selected joint reader.','0.90'))
s=before(s,'Writer identification and stroke normalisation',r'''Figure~\ref{fig:vt-lattice} compares the two paths through the same reader output. Greedy decoding selects the largest probability at each frame, whereas forced alignment restricts the path to the supplied transcript, following the CTC conventions already defined above~\cite[pp.~2--4]{graves2006}. On this particular line, the decoded string matches the known transcript exactly; the blank class accounts for about 46\% of the mean frame probability. The repeated letters in ``Vloggers'' illustrate why separated emission runs are needed. This is one worked recognition example, not a replacement for the pooled character-accuracy results.
'''+fig('T36_ctc_lattice','fig:vt-lattice','Forced alignment, greedy decoding and blank probability for one known Owen transcript.'))
s=s.replace('The selected checkpoint stores its width as 0.17.',r'Appendix~\ref{app:vt-width} illustrates the actual integer dilation kernels selected by the sweep fractions; nearby fractions can coincide on a particular crop. The selected checkpoint stores its width as 0.17.',1)
s=s.replace('\\subsubsection{First principles and framework implementations}',r'\subsubsection{First principles and framework implementations}',1)
s=s.replace('\\subsubsection{Classification verification and limitations}',r'Additional convolution and CTC derivative checks are illustrated in Appendix~\ref{app:vt-gradient}; they test numerical operations separately from recognition accuracy.'+'\n\n'+r'\subsubsection{Classification verification and limitations}',1) if '\\subsubsection{Classification verification and limitations}' in s else s
p.write_text(s,encoding='utf-8')
p=b/paths[3];s=p.read_text(encoding='utf-8')
s=before(s,'Glyph selection and library quality',r'''Figure~\ref{fig:vt-tracing} follows a real Owen line fragment from ink area to centreline and then to simplified paths. Across the complete traced line, 85 paths contain 3\,738 vertices before simplification and 974 afterwards, using the extraction tolerance of 0.6 source pixels. This removes 2\,764 vertices, approximately 74\%, before later glyph placement. Path colours and start markers also expose the separate pen strokes. The order is inferred from static ink and does not recover the writer's original stroke timing. Appendix~\ref{app:vt-glyphgeometry} connects the resulting points to the stored glyph frame and connectedness count.
'''+fig('T41_skeleton_trace_simplify','fig:vt-tracing','Real ink, its thinned centreline and the traced paths after bounded simplification.'))
s=s.replace("shows actual variants from Dylan's final library alongside its representative forms.","compares the final nine-variant \\texttt{a} pools for Owen, Dylan and Yeukita alongside their representative forms.")
s=s.replace('Figures/design-glyph-profile-example.pdf','Figures/VisualTests/T45_variant_pool_medoid.pdf')
s=s.replace('Actual glyph variants and representative forms in the final writer profile.','Variant clouds, actual medoids, denoised prototypes and within-writer distance totals for three final profiles.')
s=s.replace('This final \\texttt{a} pool contains nine variants. The four displayed examples retain recognisable loop geometry while differing in closure and curvature.','Each displayed \\texttt{a} pool contains nine variants. Their overlaid paths retain recognisable loop geometry while differing in closure and curvature; the distance bars show why a particular real example is selected.')
s=before(s,'Reproduction verification and design limitations',r'''Figure~\ref{fig:vt-timeparts} applies the same budget calculation to a separate offline generation of ``Writing with style.'' in four final profiles. The Owen job contains 58 pen transitions: its recorded components are 9.249\,s drawing, 3.198\,s travel and $58(0.35)=20.3$\,s dwell, giving 32.747\,s in total. Thus dwell contributes about 62\% of this nominal budget; across the four jobs it contributes approximately 61--65\%. The different path and transition counts distinguish these jobs from the earlier worked file. The breakdown identifies a drawing-side optimisation opportunity, but shortening dwell would require a physical test that the pen still clears and settles reliably. These estimates exclude the execution costs stated above and are not additional R3 timing measurements.
'''+fig('T56_time_budget','fig:vt-timeparts','Nominal drawing, travel and dwell budgets for the same sentence generated from four final profiles.'))
p.write_text(s,encoding='utf-8')
p=b/paths[4];s=p.read_text(encoding='utf-8')
s+=r'''
\clearpage
\subsection{Visual checks of numerical training operations}\label{app:vt-gradient}
The retained visual-test record contains 249 scalar perturbation comparisons: 73 sampled convolution-weight entries, 73 bias comparisons, 73 input entries and all 30 logits of a small CTC lattice. The sampled entries need not be distinct parameters. Central differences use a step of $10^{-6}$ for the convolution and $10^{-5}$ for CTC, following the numerical-check principle in Equation~\ref{eq:des-fd}. Figure~\ref{fig:vt-grad} compares the resulting numerical derivatives with the implemented backward operations. Relative difference is the absolute discrepancy divided by the sum of the two derivative magnitudes, with a $10^{-12}$ denominator floor.

The largest relative difference is $7.50\times10^{-7}$, among the sampled convolution inputs; the largest absolute input discrepancy is $7.26\times10^{-9}$. The CTC relative difference is at most $1.69\times10^{-9}$. These checks support the tested local derivatives under the fixed fixtures. They do not exercise every layer, parameter, floating-point regime or complete training run.
'''.replace('eq:des-fd','eq:des-numgrad')+fig('T38_gradient_check','fig:vt-grad','Analytic and central-difference gradients for sampled convolution quantities and a complete small CTC logit array.')
s+=r'''\subsection{Stroke-width quantisation in the validation sweep}\label{app:vt-width}
Figure~\ref{fig:vt-width} shows four real crops processed at the four sweep fractions. The image transformation depends on the resulting odd integer dilation kernel as well as the fraction. For Owen and Abhinav, 0.06 and 0.09 both select $k=3$ on the illustrated crop; for Dylan, 0.09 and 0.11 both select $k=5$. Identical kernels give identical normalised images for that same crop. This does not imply that two settings produce identical training datasets, since x-height varies across crops. The validation bars reproduce the supplied training sweep, not a new accuracy measurement. Its best observed setting remains 0.17, while a single run at each setting does not isolate width effects from training variability.
'''+fig('T40_stroke_width_sweep','fig:vt-width','Real-crop normalisation kernels alongside the previously reported eleven-writer validation sweep.')
s+=r'''\subsection{Glyph coordinates and measured connectedness}\label{app:vt-glyphgeometry}
Figure~\ref{fig:vt-coordinates} shows the actual Owen \texttt{V} used in the coordinate calculation in Design: left boundary 146 pixels, baseline 107 pixels, x-height 38 pixels and slant $-4.5^\circ$. The stored path uses the baseline as its origin, measures height upwards and expresses lengths in x-heights. The lead is stored separately from the path so that source-cell spacing is not confused with the first surviving ink point.
'''+fig('T43_glyph_coordinates','fig:vt-coordinates','The same glyph in image coordinates and in its stored, de-slanted coordinate frame.')
s+=r'''Figure~\ref{fig:vt-connectedness} makes the connectedness definition inspectable on the same line. Three of its 39 eligible adjacent non-space boundaries are crossed by a traced centreline, giving $3/39=0.076923$. This is the line-level example already used in Design. A crossing records connected source ink under the extraction convention; it does not reveal the writer's natural pen-lift sequence or establish a connection fraction for every page by itself.
'''+fig('T49_connectedness','fig:vt-connectedness','Eligible character boundaries and the three observed centreline crossings in one original line.')
s+='\\FloatBarrier\n';p.write_text(s,encoding='utf-8')
labels='\n'.join(p.read_text(encoding='utf-8-sig') for p in b.rglob('*.tex'))
assert '\\label{eq:des-numgrad}' in labels,'check finite difference label'
review=['# Review of all 41 design visual-test figures','', 'Reviewed every PNG against its supplied manifest, generation scripts and recorded data. Added figures explain offline subsystem behaviour; no new physical qualification result is claimed. Original front matter, proposal, Conclusion and user block diagrams are unchanged.','', '## Included', '']
for n in selected:review.append('- '+n+': '+('appendix detail' if n.startswith(('T38','T40','T43','T49')) else 'Design explanation')+'.')
review+=['','## Remaining figures','']
for p in sorted((v/'figures').glob('*.png')):
 if p.stem in selected:continue
 reason='Useful supporting illustration, but adds detail already covered by the selected figures or existing report.'
 if p.stem.startswith(('T20','T48','T50')):reason='Synthesis is plotted with an inverted vertical direction; excluded to avoid a misleading illustration.'
 if p.stem.startswith('T15'):reason='Keep the author\'s supplied software diagram.'
 if p.stem.startswith('T51'):reason='Existing candidate figure and numerical trace already explain this; replacing it would mix separately generated scores.'
 if p.stem.startswith('T54'):reason='Ordering reduces travel for Owen but increases it for Dylan and Yeukita; does not support a general travel-reduction claim.'
 if p.stem.startswith('T53'):reason='Existing text explains the directional timing penalty; figure contains an obsolete equation-number legend.'
 if p.stem.startswith('T60'):reason='Existing report already separates emitted-file reconstruction from physical-page evidence.'
 review.append('- '+p.stem+': '+reason)
review+=['','## Asset handling','Original plots remain untouched. Included PDF copies have only their embedded footer caption/provenance area cropped, replacing stale equation numbers with manuscript captions. T56 is replotted directly from the retained motion_results.json to remove overlapping plot labels; no data changed. Raw records, original manifest and generation scripts are bundled in VisualTestEvidence.','']
(e/'REVIEW.md').write_text('\n'.join(review),encoding='utf-8')
print('Added/replaced 12 visual-test figures; full review recorded.')
