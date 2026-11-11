from pathlib import Path
import json,re,shutil
ROOT=Path.cwd();B=ROOT/'Report/ResultsDiscussionRevision_2026-11-02';D=ROOT/'tmp/results_discussion_revision';E=D/'evidence'
for name in ['design_implementation_details','design_subsystem_evidence','results_evidence']:
 p=B/'Appendices'/f'{name}.tex'
 if not (D/f'original_{name}.tex').exists():shutil.copy2(p,D/f'original_{name}.tex')
a=(D/'original_design_implementation_details.tex').read_text();b=(D/'original_design_subsystem_evidence.tex').read_text();c=(D/'original_results_evidence.tex').read_text()
data=json.loads((E/'report_data.json').read_text());geo=data['geometry'];gan=data['gantry']

def block(text,kind,label):
 pat=r'\\begin\{'+kind+r'\}.*?\\end\{'+kind+r'\}'
 return next(m.group(0) for m in re.finditer(pat,text,re.S) if '\\label{'+label+'}' in m.group(0))
def sections(text):
 m=list(re.finditer(r'\\subsection\{([^}]+)\}',text));return {q.group(1):text[q.start():m[i+1].start() if i+1<len(m) else len(text)] for i,q in enumerate(m)}
def table(headers,rows,caption,label,widths=None,long=False):
 cols=widths or ('l'+'r'*(len(headers)-1));env='longtable' if long else 'tabular'
 out=('' if long else '\\begin{table}[!htbp]\\centering\\small\n')+'\\begin{'+env+'}{'+cols+'}\n'
 if long:out+='\\caption{'+caption+'}\\label{'+label+'}\\\\\n'
 out+='\\toprule '+' & '.join(headers)+'\\\\\\midrule\n'
 if long:out+='\\endfirsthead\n\\toprule '+' & '.join(headers)+'\\\\\\midrule\\endhead\n'
 out+='\n'.join(' & '.join(str(x) for x in row)+'\\\\' for row in rows)+'\n\\bottomrule\\end{'+env+'}\n'
 if not long:out+='\\caption{'+caption+'}\\label{'+label+'}\\end{table}\n'
 return out
def fig(path,caption,label,width='\\linewidth'):
 return '\\begin{figure}[H]\\centering\n\\includegraphics[width='+width+']{'+path+'}\n\\caption{'+caption+'}\\label{'+label+'}\\end{figure}\n'

# Only existing worked numerical substitutions, referenced procedures and interfaces.
asections=sections(a);worked=asections['Worked values from the implemented system']
worked=worked.split('\\subsubsection{Hardware calculation detail}')[0]
# Keep the numerical evidence and remove the surrounding teaching/interpretation.
parts=[]
for paragraph in worked.split('\n\n'):
 if paragraph.lstrip().startswith('\\'):parts.append(paragraph);continue
 sentences=re.split(r'(?<=[.!?])\s+(?=[A-Z])',paragraph)
 keep=[s for s in sentences if re.search(r'\d',s)]
 if keep:parts.append(' '.join(keep))
worked='\n\n'.join(parts)
A='\\section{Extended design calculations and connections}\n\\label{app:des-implementation}\n'
A+='Numerical substitutions extend the calculations referenced in Design.\n'+worked+'\n'
A+='\\subsection{Additional recorded numerical checks}\\label{app:des-expanded-checks}\\label{app:des-real-calculations}\n'
# Preserve the actual 3x3 patch and recorded traces, not an appendix derivation.
real=sections(b)['Real-input calculation traces']
for tab in re.finditer(r'\\begin\{table\}.*?\\end\{table\}',real,re.S):A+=tab.group(0)+'\n'
A+=table(['Recorded check','Values'],[
['Convolution patch storage','$368640\\times4/2^{20}=1.40625$ MiB'],
['CTC finite-difference losses','$0.4703364611173033$, $0.4703403997759956$'],
['Perturbation and derivative','$10^{-4}$, $-0.0196932934615$'],
['Analytic derivative and difference','$-0.0196932934309$, $3.06\\times10^{-11}$'],
['Switch confirmation intervals','$6+6=12$ ms']], 'Recorded calculation values.','tab:app-numeric-values',widths='p{.43\\linewidth}p{.49\\linewidth}')
A+='The switch algorithm in Design uses the following recorded confirmation duration.\n\\begin{equation}6\\,\\mathrm{ms}+6\\,\\mathrm{ms}=12\\,\\mathrm{ms}.\\label{eq:gan-latency}\\end{equation}\n'
A+='\\subsection{Extended training procedure}\\label{app:des-training-procedures}\nThe following procedure is the one referenced in Design.\n'+block(a,'designalgorithm','alg:des-train')+'\n'
A+='\\subsection{Representation and connection records}\n'+block(a,'table','tab:des-contracts')+'\n'
A+=asections['Controller pin and supply planning']+'\n\\FloatBarrier\n'
(B/'Appendices/design_implementation_details.tex').write_text(A)

bs=sections(b)
BB='\\section{Extended design figures and subsystem evidence}\\label{app:des-visual-tests}\n'
BB+='\\subsection{Recorded capture and seam checks}\n'
for label in ['tab:des-resample-cases','tab:app-seam-sweep']:
 BB+=block(b,'table',label)+'\n'
BB+='\\label{app:des-seam-sweep}\n'
BB+='\\subsection{Numerical verification records}\\label{sec:des-netverify}\n'
for heading in ['Visual checks of numerical training operations','Stroke-width quantisation in the validation sweep','Glyph coordinates and measured connectedness']:
 text=bs[heading]
 for f in re.finditer(r'\\begin\{figure\}.*?\\end\{figure\}',text,re.S):
  if 'T49' in f.group(0):continue
  BB+='The following figure extends the corresponding Design example.\n'+f.group(0)+'\n'
BB+='\\subsection{Extended figures from the Design section}\\label{app:extended-design-figures}\n'
for f in re.finditer(r'\\begin\{figure\}.*?\\end\{figure\}',bs['Extended figures from the Design section'],re.S):
 BB+='The complete figure extends the panels retained in Design.\n'+f.group(0)+'\n'
BB+='\\FloatBarrier\n';(B/'Appendices/design_subsystem_evidence.tex').write_text(BB)

C='\\section{Extended qualification results}\\label{app:res-evidence}\n'
C+='\\subsection{Measurement records and populations}\\label{app:res-provenance}\n'
C+=table(['Record','Identity or scope'],[
['Physical classifier evidence','184 paired lines, 14 pages, 5471 reference characters'],
['Evaluated reader','joint\\_best, SHA-256 prefix \\texttt{336802d535992323}'],
['Evaluated writer','11\\_pw170, \\texttt{1884fb86467538eb}'],
['Numerical comparison reader','scratch\\_best, \\texttt{2add315ded078de5}'],
['Numerical comparison writer','11w\\_pw170\\_scratch, \\texttt{b23a709a30144f67}'],
['Checkpoint training provenance','The evaluated joint/pw170 weights use the recorded library training route. Numerical comparison models are listed separately.'],
['Execution records','Physical motion and request timings are ODROID records. Saved-image scoring and geometric analyses are offline verification on the development host.']], 'Identities and scope of the retained measurements.','tab:app-checkpoints',widths='p{.34\\linewidth}p{.58\\linewidth}')
C+='\\subsection{Original-handwriting validation}\\label{app:res-validation}\n'
C+=block(c,'table','tab:app-reader-breakdown')+'\n'
C+='Figure~\\ref{fig:app-validation-confusion} retains the complete validation writer predictions.\n'+fig('Figures/rev_validation_confusion.pdf','Validation writer predictions, row percentages.','fig:app-validation-confusion')
C+='\\subsection{Physical classifier records}\\label{app:res-page-records}\n'
C+='\\input{3_Main_Report/results_tables/r1_writers}\n\\input{3_Main_Report/results_tables/r1_blocks}\n'
C+='Figure~\\ref{fig:res-reproduced-confusion} retains the physical writer predictions.\n'+fig('Figures/rev_physical_confusion.pdf','Physical writer predictions, row percentages.','fig:res-reproduced-confusion')
C+=block(c,'longtable','tab:app-all-lines')+'\n'
C+='\\subsection{Matched geometric verification}\\label{app:res-geometric}\n'
C+='Both sides of each comparison use the same original reference bank. Tables retain ordinary-average and nearest-reference results separately.\n'
interval=geo['ordinary_interval'];closestint=geo['closest_interval']
C+=table(['Quantity','Ordinary average','Nearest reference'],[
['Equal-author mean normalised score',f"{geo['ordinary_equal_author']:.2f}\\%",f"{geo['closest_equal_author']:.2f}\\%"],
['Conditional resampling range',f"{interval[0]:.2f} to {interval[1]:.2f}\\%",f"{closestint[0]:.2f} to {closestint[1]:.2f}\\%"],
['Physical words','83','83'],['Writer/token groups','77','77'],['Writers','11','11']], 'Geometric verification summary.','tab:app-geometric-summary',widths='lrr')
rows=[[r['label'],r['n'],f"{r['natural']:.2f}",f"{r['robot']:.2f}",f"{r['relative']:.2f}",f"{r['closest_relative']:.2f}"] for r in geo['per_writer']]
C+=table(['Writer','Words','Genuine F / \\%','Robot F / \\%','Relative / \\%','Nearest / \\%'],rows,'All writers in the repeated-word comparison.','tab:app-geometric-writers')
nat=json.loads((ROOT/'Report/NaturalVariationBaseline_2026-11-02/results.json').read_text());ww=[r for r in nat['words'] if r['mode']=='cross_page' and r['method']=='word'];labels={r['writer']:r['label'] for r in geo['per_writer']}
def esc(s):return s.replace('&','\\&').replace('_','\\_').replace('%','\\%')
rows=[[labels[r['writer']],esc(r['token']),f"{100*r['natural_pair_mean_F']:.2f}",f"{100*r['robot_pair_mean_F']:.2f}",f"{100*r['pair_mean_score_ratio']:.2f}"] for r in ww]
C+=table(['Writer','Word','Genuine F / \\%','Robot F / \\%','Relative / \\%'],rows,'All 83 repeated-word results using ordinary reference averages.','tab:app-geometric-words',widths='llrrr',long=True)
full=json.loads((ROOT/'Report/GeometricFidelity_AllAuthors_2026-11-02/results.json').read_text())
rows=[[labels[w],z['scored_words'],z['eligible_proposals'],f"{100*z['line']['mean_F']:.2f}",f"{100*z['word']['mean_F']:.2f}"] for w,z in full['summary'].items()]
C+=table(['Writer','Scored','Eligible','Line height / \\%','Word height / \\%'],rows,'Full word-test coverage and absolute F-scores.','tab:app-geometric-coverage')
C+='The baseline subset has 83 of the 141 comparable words. Twenty-five of 166 eligible words have no accepted exact original reference. Full crop IDs, bank membership and all original-to-original and robot-to-original pair scores accompany the evidence package.\n'
C+='\\subsection{Complete confirmed reproduction timings}\\label{app:res-speed}\n'+block(c,'longtable','tab:app-complete-timings')+'\n'
C+='\\subsection{Classification stage records and supporting benchmark}\\label{app:res-classification-time}\n\\input{3_Main_Report/results_tables/r4}\n'
C+=table(['Recorded comparison','Outcome'],[
['Fourteen-page network benchmark','359.8 s sequential, 130.3 s batched reference backend, factor 2.76. Recorded reader output agrees on every line.'],
['Image-resolution experiment','247 versus 182 detected lines. No line-count match among 12 comparable successful pairs. Two reduced pages fail segmentation.'],
['Measurement scope','Both supporting comparisons are development-host experiments. They are not measured ODROID speed gains.']], 'Supporting optimisation measurements.','tab:app-acceleration',widths='p{.35\\linewidth}p{.57\\linewidth}')
C+='\\subsection{Ten-sample dimensional records}\\label{app:res-geometry-calculation}\n'
rows=[[r['feature'],f"{r['mean_error']:+.3f}",f"{r['sample_sd']:.3f}",f"{r['min_error']:+.2f}",f"{r['max_error']:+.2f}"] for r in gan['per_feature']]
C+=table(['Feature','Mean / mm','Sample SD / mm','Min / mm','Max / mm'],rows,'Repeated feature-length error statistics, ten readings per feature.','tab:app-gantry-features')
raw=list(__import__('csv').DictReader((E/'gantry_70_measurements.csv').open()))
rows=[[r['sample'],r['feature'],r['nominal_mm'],f"{float(r['measured_mm']):.2f}",f"{float(r['error_mm']):+.2f}"] for r in raw]
C+=table(['Sample','Feature','Nominal / mm','Measured / mm','Error / mm'],rows,'All 70 author-supplied dimensional readings.','tab:app-gantry-all',widths='rlrrr',long=True)
rows=[[r['sample'],f"{r['mean_absolute']:.3f}",f"{r['rmse']:.3f}",f"{r['max_absolute']:.2f}"] for r in gan['per_sample']]
C+=table(['Sample','Mean absolute / mm','RMS / mm','Largest absolute / mm'],rows,'Error summary for each sample.','tab:app-gantry-samples')
C+='\\subsection{Stored style inventory}\\label{app:res-inventory}\n'
C+=table(['Writer','Character mappings','Physical pages'],[[r['label'],r['glyph_keys'],r['pages']] for r in data['style_inventory']],'Stored profiles and physical-page coverage.','tab:app-style-inventory')
C+='\\subsection{Supplied first-principles training records}\\label{app:training-records}\n'
C+='The tables retain the displayed console records. The writer uses 1031 training and 155 validation lines, eleven writers and 0.17 stroke normalisation. The reader records 6481 public and 569 personal training lines, and 976 public and 100 personal held-out lines.\n'
C+=block(c,'longtable','tab:app-author-log')+'\n'+block(c,'longtable','tab:app-reader-log')+'\n'
C+='\\subsection{Complete physical-page photographs}\n'
physical_section=sections(c)['Complete physical-page records']
for m in re.finditer(r'\\begin\{figure\}.*?\\end\{figure\}',physical_section,re.S):
 C+='The following original photographs extend the classifier summaries.\n'+m.group(0)+'\n'
C+='\\FloatBarrier\n'
(B/'Appendices/results_evidence.tex').write_text(C)
gs=gan
(B/'3_Main_Report/results_tables/gantry_summary.tex').write_text(table(['Recorded quantity','Value'],[['Geometric samples','10'],['Measured feature lengths','70'],['Lengths within 0.5 mm','70/70'],['Mean absolute length error',f"{gs['mean_absolute_mm']:.3f} mm"],['RMS length error',f"{gs['rmse_mm']:.3f} mm"],['Largest absolute length error','0.49 mm']],'Ten-sample dimensional summary.','tab:res-gantry-summary',widths='lr'))
# The main reader table retains the selected measurement; implementation comparison
# remains in the protected Design and extended appendix record.
(B/'3_Main_Report/results_tables/r2_reader.tex').write_text(table(['Population','Lines','Characters','Accuracy / \\%'],[['IAM validation',976,42072,'94.12'],['Personal validation',100,3675,'93.33'],['Pooled',1076,45747,'94.06']],'Reader accuracy on the recorded validation populations.','tab:res-r2-reader'))
for name in ['results','discussion']:
 shutil.copy2(D/f'{name}.tex',B/'3_Main_Report'/('4_Results.tex' if name=='results' else '5_Discussion.tex'))
print('Results, Discussion and three evidence-only appendices installed')
