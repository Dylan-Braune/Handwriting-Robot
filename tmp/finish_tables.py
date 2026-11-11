from pathlib import Path
import re
P=Path('Report/SubmissionRevision_2026-11-02');R=P/'3_Main_Report'
p=P/'references.bib';s=p.read_text();s=s.replace('?seq_no=2','?seq\\_no=2');p.write_text(s)
# Keep full per-writer reader breakdown in the appendix and use population totals in main Results.
p=R/'results_tables/r2_reader.tex';s=p.read_text();full=s.replace('tab:res-r2-reader','tab:app-reader-breakdown');start=full.index('\\caption{');end=full.index('\\label',start);full=full[:start]+r'\caption{Reader character accuracy by validation population.}'+full[end:]
a=P/'Appendices/results_evidence.tex';t=a.read_text();i=t.index('\\subsection{Physical reproduction by writer}');t=t[:i]+r'''\subsection{Reader validation breakdown}
Table~\ref{tab:app-reader-breakdown} extends the population totals in QT~2. The personal contributors are retained separately, using the same reader checkpoints and greedy decoder.
'''+full+'\n'+t[i:];a.write_text(t)
p.write_text(r'''\begin{table}[!htbp]\centering\small
\begin{tabular}{lrrrr}\toprule
Population & Lines & Characters & Joint / \% & Scratch / \%\\\midrule
IAM validation & 976 & 42072 & 94.12 & 94.39\\
Personal validation & 100 & 3675 & 93.33 & 94.42\\
Pooled & 1076 & 45747 & 94.06 & 94.39\\\bottomrule
\end{tabular}
\caption{Reader character accuracy on validation lines.}\label{tab:res-r2-reader}
\end{table}
''')
p=R/'4_Results.tex';s=p.read_text();s=s.replace('The training-route comparison is in Table~\\ref{tab:dis-training}.','The training-route comparison is in Table~\\ref{tab:dis-training}. Table~\\ref{tab:app-reader-breakdown} retains the individual personal-writer scores.');p.write_text(s)
# Short table captions. Details remain in their introducing paragraphs.
caps={'r1_blocks.tex':'Physical reproduction scores by text block.','r1_writers.tex':'Physical reproduction scores by writer.','r3.tex':'Completed ODROID reproduction jobs by text length.','r4.tex':'Recorded ODROID classification-stage timings.','r5.tex':'Seven measured feature lengths after calibration.','style_physical.tex':'Physical-to-original geometric correlations.'}
for name,caption in caps.items():
 p=R/'results_tables'/name;s=p.read_text();a=s.index('\\caption{')+9;d=1
 for b in range(a,len(s)):
  d+=(s[b]=='{')-(s[b]=='}')
  if not d:break
 s=s[:a]+caption+s[b:];s=s.replace('A: paragraph','Paragraph').replace('B: pangram','Pangram').replace('C: source text','Source text');p.write_text(s)
# Compact main Design table captions without changing their data.
repls={
'Implemented convolutional backbone shared in architecture by the text and writer networks. The two models have separately trained parameters.':'Convolutional stages and feature dimensions.',
'Archived float64 implementation comparison from the CleanPipeline2 verification script. This is a controlled numerical check, not a final-model recognition result.':'Archived float64 operation comparison.',
}
for p in (R/'design').glob('*.tex'):
 s=p.read_text()
 for a,b in repls.items():s=s.replace(a,b)
 p.write_text(s)
