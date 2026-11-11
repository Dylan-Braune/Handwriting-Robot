from pathlib import Path
import re,json
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'Report/DesignPolished_2026-11-01'
app=B/'Appendices/design_implementation_details.tex'
extra=[];moves=[]
def move_env(p,env,label,replacement):
 t=p.read_text(encoding='utf-8')
 matches=[m for m in re.finditer(r'\\begin\{'+env+r'\}.*?\\end\{'+env+r'\}',t,re.S) if '\\label{'+label+'}' in m[0]]
 assert len(matches)==1,label
 m=matches[0];extra.append(m[0]);moves.append(label);t=t[:m.start()]+replacement+t[m.end():]
 p.write_text(t,encoding='utf-8')
p=B/'3_Main_Report/design/3_5_classification.tex'
extra.append(r'\subsection{Detailed training procedures}\label{app:des-training-procedures}'+'\n'+r'The main chapter retains the model, loss, design choices and representative checks. The following pseudocode records the detailed recurrences and update sequence~\cite[train.py]{braunepipeline2026}.')
move_env(p,'designalgorithm','alg:des-ctc',r'Algorithm~\ref{alg:des-ctc} in Appendix~\ref{app:des-training-procedures} gives the full forward--backward procedure with explicit blank/repeat transitions.')
move_env(p,'designalgorithm','alg:des-train',r'The complete numerical batch/update procedure appears in Algorithm~\ref{alg:des-train} in Appendix~\ref{app:des-training-procedures}; the calculations remain explicit in the source~\cite[train.py]{braunepipeline2026}.')
extra.append(r'\subsection{Expanded numerical calculation checks}\label{app:des-expanded-checks}')
t=p.read_text(encoding='utf-8')
start=t.index('For one image, let $X_c$');end=t.index('\\subsubsection{Network verification',start)
extra.append(t[start:end]);t=t[:start]+r'The patch-matrix weight/input derivatives, their worked substitution and the numerical training-memory calculation are given in Appendix~\ref{app:des-expanded-checks}~\cite[train.py, Conv2dBackward]{braunepipeline2026}.'+'\n\n'+t[end:]
start=t.index('The finite-difference reference uses Equation');end=t.index('The main design limitations',start)
extra.append(t[start:end]);t=t[:start]+r'The finite-difference reference, numerical substitution and its treatment of non-smooth operations are recorded in Equation~\ref{eq:des-gradcheck} in Appendix~\ref{app:des-expanded-checks}.'+'\n\n'+t[end:]
p.write_text(t,encoding='utf-8')
extra.append(r'\subsection{Implementation and interface reference tables}\label{app:des-reference-tables}'+'\n'+r'These tables preserve supporting implementation distinctions and representation contracts. The main chapter explains the corresponding choices and transitions~\cite[recognise.py; train.py; style.py; web/server.py]{braunepipeline2026}.')
move_env(p,'table','tab:des-implementations',r'The operation-by-operation comparison is retained in Table~\ref{tab:des-implementations}, Appendix~\ref{app:des-reference-tables}.')
p=B/'3_Main_Report/design/3_6_reproduction.tex'
move_env(p,'table','tab:des-styleparameters',r'Table~\ref{tab:des-styleparameters} in Appendix~\ref{app:des-reference-tables} gives the stored quantities, their units and their use in synthesis.')
p=B/'3_Main_Report/design/3_7_integration.tex'
move_env(p,'table','tab:des-contracts',r'The input/output contracts, units and coordinate conventions are listed in Table~\ref{tab:des-contracts}, Appendix~\ref{app:des-reference-tables}.')
# Keep summary as an actual concise summary; the final architecture explains the change.
p=B/'3_Main_Report/design/3_1_summary.tex';t=p.read_text(encoding='utf-8')
start=t.index('The recognition design uses complete');end=t.index('\\FloatBarrier',start)
summary=t[start:end];g=re.search(r'\\designgap\{([^}]+)\}',summary)
t=t[:start]+t[end:];p.write_text(t,encoding='utf-8')
p=B/'3_Main_Report/design/3_2_system.tex';t=p.read_text(encoding='utf-8')
pos=t.index("The proposal's functional units")
addition=r'''The final design reads complete lines rather than requiring isolated letters at FU 2.2. Character boundaries are instead recovered with known transcripts during profile construction. This avoids committing to uncertain cuts before recognising connected handwriting~\cite[recognise.py, PaperCRNNNumpy; style.py, ExtractLineGlyphs]{braunepipeline2026}.

'''
t=t[:pos]+addition+t[pos:];p.write_text(t,encoding='utf-8')
if g:extra.append(r'\subsection{Submission completion record}'+'\n'+r'\designgap{'+g[1]+'}')
t=app.read_text(encoding='utf-8')+'\n'+'\n\n'.join(extra)+'\n\\FloatBarrier\n'
app.write_text(t,encoding='utf-8')
(B/'APPENDIX_RELOCATION.json').write_text(json.dumps({'moved_labels':moves,'reason':'Preserve worked calculations, citations and detailed procedures while keeping the main design readable.'},indent=2),encoding='utf-8')
# Fix record citations: provenance of selected parameters is not established by code alone.
p=B/'3_Main_Report/design/3_5_classification.tex';t=p.read_text(encoding='utf-8')
t=t.replace('the provenance of the selected weights remains the framework training route.~\\cite[recognise.py; train.py; trainlibrary.py]{braunepipeline2026}',r'the provenance of the selected weights remains the framework training route~\cite[training records]{brauneevidence2026}\cite[recognise.py; train.py; trainlibrary.py]{braunepipeline2026}.')
p.write_text(t,encoding='utf-8')
print(moves)
