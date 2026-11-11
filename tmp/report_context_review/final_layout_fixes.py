from pathlib import Path
import re,json
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'Report/DesignPolished_2026-11-01'
p=B/'Appendices/design_subsystem_evidence.tex';t=p.read_text(encoding='utf-8')
start=t.index('The three Owen trials in Figure');end=t.index('\\subsection{Post-design',start)
t=t[:start]+r'''The Owen trials use the joint reader and eleven-writer \texttt{pw170} checkpoint, with explicit 0.17 normalisation and uniform-ink rendering. They score character accuracy and requested-writer posterior before repair, independently of inherited server defaults. Exact inputs, readings and full-precision scores are retained in \texttt{Evidence/bench.json}~\cite[subsystem manifests]{brauneevidence2026}.

'''+t[end:]
start=t.index('The seam weights were examined');end=t.index('\\begin{table}',start)
t=t[:start]+r'''A new one-at-a-time check examines the existing seam weights. Owen01 and Yeukita01 are reduced to a maximum dimension of 1600 pixels, then passed through orientation, conditioning and line-centre estimation. The first, middle and last gaps on each page are selected by index before inspecting outputs. Eight settings share these six gaps, step limit seven and ink weight four: midpoint weight varies at step penalty 0.02, then step penalty varies at midpoint weight 0.012. This is post-design sensitivity, not a joint search or recovered historical tuning~\cite[\texttt{Evidence/coefficient\_sensitivity.json}]{brauneevidence2026}.

The parameterised reference matches all six unmodified default paths. Table~\ref{tab:app-seam-sweep} reports raw-ink intersections, total row movement and mean midpoint distance in working pixels; the midpoint is a preference, not a labelled correct boundary~\cite[\texttt{run\_coefficient\_sensitivity.py}]{brauneevidence2026}\cite[segment.py, LeastCostSeam and SeamCuts]{braunepipeline2026}.

'''+t[end:]
start=t.index('Increasing the midpoint weight');t=t[:start]+r'''Increasing midpoint weight from zero to 0.012 reduces mean offset from 4.055 to 0.363 pixels, while row movement rises from 35 to 42 pixels. Doubling the weight gives 0.326 pixels offset. Increasing step penalty has the opposite trade-off. Every setting retains two raw-mask and three closed-mask intersections. These cases therefore show competing effects without establishing a unique optimum, eliminating intersections or measuring recognition accuracy. Input/code hashes and all 48 runs are supplied; no pipeline coefficient was changed~\cite[\texttt{Evidence/coefficient\_sensitivity.json}]{brauneevidence2026}.
\FloatBarrier
'''
# Show the exact tie value rather than inconsistent three-place rounding.
data=json.loads((B/'Evidence/coefficient_sensitivity.json').read_text())
for r in data['summary']:
 old=f"{r['midpoint_weight']:.3f}&{r['step_penalty']:.3f}&{r['raw_ink_intersections']}&{r['vertical_step_sum']}&{r['mean_midpoint_offset_pixels']:.3f}"
 new=f"{r['midpoint_weight']:.3f}&{r['step_penalty']:.3f}&{r['raw_ink_intersections']}&{r['vertical_step_sum']}&{r['mean_midpoint_offset_pixels']:.4f}"
 assert old in t;t=t.replace(old,new)
p.write_text(t,encoding='utf-8')
p=B/'Appendices/design_implementation_details.tex';t=p.read_text(encoding='utf-8').replace(r'\subsection{Submission completion record}',r'\FloatBarrier'+'\n'+r'\subsection{Submission completion record}')
p.write_text(t,encoding='utf-8')
# Recover actual current counts after moving support into the appendix.
files=[p for f in ['3_Main_Report','Appendices'] for p in (B/f).rglob('*.tex')]
des='\n'.join(p.read_text(encoding='utf-8') for p in files)
sites=re.findall(r'\\cite(?:\[([^\]]*)\])?\{([^}]+)\}',des)
audit=json.loads((B/'SOURCE_AUDIT_COUNTS.json').read_text())
audit.update(citation_sites_total=len(sites),external_citation_sites=sum(any(k.strip() not in {'braunepipeline2026','brauneevidence2026'} for k in ks.split(',')) for _,ks in sites),citations_with_locators=sum(bool(loc) for loc,_ in sites))
(B/'SOURCE_AUDIT_COUNTS.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
