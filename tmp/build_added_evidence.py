from pathlib import Path
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil,csv
P=Path('Report/SubmissionRevision_2026-11-02');E=Path('Report/ReportWorkingArchive_2026-11-01_VisualTests/ResultsEvidence');src=Path('Software/CleanPipeline3/ReportResults/FullReproducedPages')
order=['150','152','153','384','552','588','abhinav','dylan','owen','thiya','yeukita'];alias=dict(zip(order,['150','152','153','384','552','588','A','B','O','T','Y']))
lines=json.loads((E/'measurements/outputs/blocks/lines.json').read_text());hold=json.loads((E/'measurements/outputs/holdout_namer/lines.json').read_text()); metrics=json.loads((E/'REVIEWED_METRICS.json').read_text())
for data,name,title in [(lines,'results_reproduced_confusion','Physical reproduced lines'),(hold,'results_holdout_confusion','Original validation lines')]:
 c=np.zeros((11,11),int)
 for row in data:
  reverse={v:k for k,v in alias.items()};c[order.index(reverse.get(row['writer'],row['writer'])),order.index(reverse.get(row['named'],row['named']))]+=1
 n=c.sum(1);pct=100*c/n[:,None];assert np.allclose(pct.sum(1),100)
 fig,ax=plt.subplots(figsize=(7.4,6.5),layout='constrained');im=ax.imshow(pct,cmap='Blues',vmin=0,vmax=100)
 ax.set_xticks(range(11),[alias[w] for w in order]);ax.set_yticks(range(11),[f'{alias[w]} (n={v})' for w,v in zip(order,n)])
 ax.set_xlabel('Predicted writer');ax.set_ylabel('Reference writer');ax.tick_params(labelsize=9)
 for y in range(11):
  for x in range(11):
   if c[y,x]:ax.text(x,y,f'{pct[y,x]:.1f}',ha='center',va='center',fontsize=8,color='white' if pct[y,x]>55 else '#18334b')
 cb=fig.colorbar(im,ax=ax,shrink=.78);cb.set_label('Percentage of reference-writer lines')
 fig.savefig(P/'Figures'/f'{name}.pdf');plt.close(fig)
 print(name,'correct',np.trace(c),'total',c.sum(),'balanced',np.diag(pct).mean())
 np.savetxt(P/'Evidence'/f'{name}_counts.csv',c,delimiter=',',fmt='%d',header=','.join(alias.values()),comments='')
 np.savetxt(P/'Evidence'/f'{name}_row_percent.csv',pct,delimiter=',',fmt='%.6f',header=','.join(alias.values()),comments='')
# Full original page images without altering pixel data.
F=P/'Figures/PhysicalPages';F.mkdir(exist_ok=True)
pages=sorted({r['page'] for r in lines})
for page in pages:shutil.copy2(src/(page+'.jpg'),F/(page+'.jpg'))
def esc(x):
 for a,b in [('\\',r'\textbackslash{}'),('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#'),('$',r'\$')]:x=x.replace(a,b)
 return x
out=r'''\subsection{Complete physical-page records}\label{app:res-page-records}
Table~\ref{tab:app-pages} retains all fourteen pages. Character accuracy pools edit counts within each page. Writer identification uses its paired lines. Text and timing records have no verified one-to-one job link, so the timing table below is kept separate.
\begin{longtable}{lrrrrr}
\caption{All physical pages and paired-line scores.}\label{tab:app-pages}\\
\toprule Page & Lines & Characters & Edits & Character / \% & Writer / \%\\\midrule\endfirsthead
\toprule Page & Lines & Characters & Edits & Character / \% & Writer / \%\\\midrule\endhead
'''
for page in pages:
 d=[r for r in lines if r['page']==page];n=sum(r['refChars'] for r in d);ed=sum(r['dist'] for r in d);correct=sum(r['writer']==r['named'] for r in d)
 out+=f'{page} & {len(d)} & {n} & {ed} & {100*(1-ed/n):.2f} & {100*correct/len(d):.2f}'+r'\\'+'\n'
out+=r'\bottomrule\end{longtable}'+'\n'
for i in range(0,len(pages),2):
 pair=pages[i:i+2];label=f'fig:app-physical-pages-{i//2+1}'
 out+='\n'+r'Figure~\ref{'+label+r'} preserves the photographed pages '+', '.join(pair)+'. The scores come from their paired line records, not a visual estimate.\n'
 out+=r'\begin{figure}[!htbp]\centering'+'\n'
 for page in pair:out+=r'\begin{minipage}{.48\linewidth}\centering\includegraphics[width=\linewidth,height=.55\textheight,keepaspectratio]{Figures/PhysicalPages/'+page+r'.jpg}\par '+page+r'\end{minipage}\hfill'+'\n'
 out+=r'\caption{Physical pages '+', '.join(pair)+r'.}\label{'+label+r'}\end{figure}\FloatBarrier'+'\n'
out+=r'''
\subsection{Paired-line scoring record}
Table~\ref{tab:app-all-lines} gives every paired line, its reference length, edit distance and writer decision. Line numbering starts at one for each page. Reference and recognised text are retained verbatim in the accompanying evidence CSV so that these summaries can be recalculated.
\begin{longtable}{lrrrrl}
\caption{All 184 paired physical lines.}\label{tab:app-all-lines}\\
\toprule Page & Line & Characters & Edits & Character / \% & Predicted writer\\\midrule\endfirsthead
\toprule Page & Line & Characters & Edits & Character / \% & Predicted writer\\\midrule\endhead
'''
for r in lines:out+=f"{r['page']} & {r['line']+1} & {r['refChars']} & {r['dist']} & {r['char']:.2f} & {alias[r['named']]}"+r'\\'+'\n'
out+=r'\bottomrule\end{longtable}'+'\n'
out+=r'''
\subsection{Complete confirmed reproduction timings}
Table~\ref{tab:app-complete-timings} retains the 25 completed, positive-character records used in QT~3. The line number identifies the timing block in the preserved ODROID log. Synthesis and drawing are component stages. The total also includes the other logged stages and rounding, so it is not replaced by their sum.
\begin{longtable}{rrrrrr}
\caption{All confirmed ODROID reproduction timings.}\label{tab:app-complete-timings}\\
\toprule Log line & Characters & Synthesis / s & Drawing / s & Total / s & s per character\\\midrule\endfirsthead
\toprule Log line & Characters & Synthesis / s & Drawing / s & Total / s & s per character\\\midrule\endhead
'''
records=[r for r in metrics['reproduction_timing']['records'] if r['status']=='complete' and r['characters']>0];assert len(records)==25
for r in records:out+=f"{r['log_line']} & {r['characters']} & {r['synthesis']:.2f} & {r['drawing']:.2f} & {r['total']:.2f} & {r['total']/r['characters']:.3f}"+r'\\'+'\n'
out+=r'\bottomrule\end{longtable}'+'\n'
p=P/'Appendices/results_evidence.tex';p.write_text(p.read_text()+'\n'+out)
with (P/'Evidence/physical_paired_lines.csv').open('w',newline='',encoding='utf8') as f:
 w=csv.DictWriter(f,fieldnames=list(lines[0]));w.writeheader();w.writerows(lines)
# Training records transcribed from the user's console excerpts, retain all completed reported epochs.
author='''1 1.979 .477 20.1
2 1.393 .542 19.7
3 1.178 .581 19.9
4 .985 .342 21.6
5 .817 .723 22.1
6 .726 .594 22.7
7 .730 .439 22.1
8 .642 .594 21.7
9 .578 .735 20.3
10 .531 .877 20.1
11 .532 .665 20.8
12 .499 .806 20.4
13 .465 .639 20.6
14 .404 .897 20.6
15 .442 .594 20.7
16 .373 .632 20.1
17 .313 .845 20.2
18 .378 .942 20.6
19 .265 .865 19.4
20 .344 .865 19.5
21 .294 .903 19.8
22 .319 .923 19.7
23 .250 .890 19.4
24 .233 .935 19.4
25 .211 .897 20.2
26 .213 .948 22.2
27 .208 .903 21.7
28 .210 .916 21.8
29 .193 .858 20.4
30 .159 .942 19.5
31 .154 .955 19.6
32 .167 .942 19.6
33 .154 .948 19.5
34 .143 .961 19.5
35 .108 .955 19.5
36 .107 .955 19.5
37 .092 .923 19.5
38 .121 .916 19.5
39 .108 .955 19.5
40 .125 .955 19.6
41 .068 .968 19.5
42 .106 .968 19.9
43 .071 .968 19.5
44 .106 .961 19.9
45 .076 .968 20.8
46 .066 .968 21.0
47 .082 .968 20.9'''
reader='''7 45.339 44.488 70.17 66.56 122.3
8 24.865 18.597 87.62 88.54 120.1
9 18.473 40.667 73.19 75.48 118.6
10 15.332 17.133 88.89 90.07 118.7
11 12.862 13.298 91.43 91.40 111.6
12 11.047 13.509 91.34 92.00 111.3
13 9.818 15.953 89.91 92.16 111.7
14 8.549 13.378 91.65 92.65 111.6
15 7.642 14.057 91.44 92.46 111.5
16 7.045 11.663 92.89 93.09 112.3'''
(P/'Evidence/supplied_first_principles_author_log.txt').write_text('Epoch Loss ValAccuracy Minutes\n'+author+'\nTraceback followed epoch 47. No completed epoch 50 supplied.\n')
(P/'Evidence/supplied_first_principles_reader_log.txt').write_text('Epoch TrainLoss ValLoss ValCharPercent PersonalCharPercent Minutes\n'+reader+'\nEpoch 17 batch 375 running loss 5.997. No completed epoch 17 supplied.\n')
out=r'''
\subsection{Supplied first-principles training records}\label{app:training-records}
The following tables transcribe the console excerpts supplied by the author. They extend the training-route comparison in Design. The writer run uses 1031 training and 155 validation lines, eleven writers and 0.17 stroke width. Its output is \texttt{author\_shape\_11w\_pw170\_scratch\_weights.npz}. Table~\ref{tab:app-author-log} retains the displayed values. The excerpt ends with an interruption after epoch 47, rather than a completed 50-epoch run.
\begin{longtable}{rrrr}
\caption{Supplied numerical writer training log.}\label{tab:app-author-log}\\
\toprule Epoch & Loss & Validation / \% & Time / min\\\midrule\endfirsthead
\toprule Epoch & Loss & Validation / \% & Time / min\\\midrule\endhead
'''
for line in author.splitlines():
 ep,loss,val,t=line.split();out+=f'{ep} & {float(loss):.3f} & {100*float(val):.1f} & {t}'+r'\\'+'\n'
out+=r'\bottomrule\end{longtable}'+'\n'
out+=r'''
The reader excerpt resumes at epoch 7 after loading 90 of 90 tensors from its numerical checkpoint. Its data preparation reports 6481 public and 569 personal training lines, with 976 public and 100 personal held-out lines. Table~\ref{tab:app-reader-log} records the completed epochs shown. Epoch 17 is only a partial batch report. These excerpts describe training progress. The later saved-checkpoint accuracy in QT~2 comes from direct evaluation and is not inferred from an unfinished log.
\begin{longtable}{rrrrrr}
\caption{Supplied numerical reader training log.}\label{tab:app-reader-log}\\
\toprule Epoch & Train loss & Val loss & Val char / \% & Personal / \% & Time / min\\\midrule\endfirsthead
\toprule Epoch & Train loss & Val loss & Val char / \% & Personal / \% & Time / min\\\midrule\endhead
'''
for line in reader.splitlines():out+=' & '.join(line.split())+r'\\'+'\n'
out+=r'\bottomrule\end{longtable}'+'\n'
p=P/'Appendices/results_evidence.tex';p.write_text(p.read_text()+'\n'+out)

