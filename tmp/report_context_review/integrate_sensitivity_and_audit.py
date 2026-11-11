from pathlib import Path
import re,json
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'Report/DesignPolished_2026-11-01'
data=json.loads((B/'Evidence/coefficient_sensitivity.json').read_text())
assert data['baseline_reference_matches']==6 and len(data['runs'])==48
for p in list((B/'3_Main_Report/design').glob('*.tex'))+list((B/'Appendices').glob('*.tex')):
 t=p.read_text(encoding='utf-8')
 t=re.sub(r'\n\\FloatBarrier(~\\cite\[[^\]]*\]\{[^}]+\})',r'\1\n\\FloatBarrier',t)
 p.write_text(t,encoding='utf-8')
p=B/'Appendices/design_subsystem_evidence.tex';t=p.read_text(encoding='utf-8')
new=r'''
\subsection{Post-design seam-coefficient sensitivity}
\label{app:des-seam-sweep}
The seam weights were examined with a new one-at-a-time sensitivity check, rather
than retrospectively described as the source of the original settings. Two real
pages, Owen01 and Yeukita01, were reduced to a maximum dimension of 1600 pixels,
then processed through the current orientation, conditioning and line-centre
stages. The first, middle and last inter-line gaps were selected by index on each
page before examining the sweep outputs. All eight settings use these same six
gaps, step limit seven and ink weight four. The midpoint coefficient varies with
step penalty fixed at 0.02; the step penalty then varies with midpoint coefficient
fixed at 0.012. This is not a joint grid search~\cite[\texttt{Evidence/coefficient\_sensitivity.json}]{brauneevidence2026}.

The parameterised reference reproduces the unmodified implementation exactly on
all six default seams. Table~\ref{tab:app-seam-sweep} reports raw-mask intersections,
total absolute row movement, and mean distance from the midpoint in working-image
pixels. The midpoint is a soft preference, not a labelled correct boundary;
neither proximity nor low movement alone establishes segmentation accuracy~\cite[\texttt{run\_coefficient\_sensitivity.py}]{brauneevidence2026}\cite[segment.py, LeastCostSeam and SeamCuts]{braunepipeline2026}.
\begin{table}[!htbp]\centering\small
\begin{tabular}{|c|c|r|r|r|}
\hline Midpoint weight&Step penalty&Ink intersections&Row movement&Mean offset / px\\\hline
'''
for r in data['summary']:
 new+=f"{r['midpoint_weight']:.3f}&{r['step_penalty']:.3f}&{r['raw_ink_intersections']}&{r['vertical_step_sum']}&{r['mean_midpoint_offset_pixels']:.3f}\\\\\\hline\n"
new+=r'''\end{tabular}
\caption{New coefficient-sensitivity check across six fixed real-page gaps; the implemented setting is 0.012/0.020~\cite[\texttt{Evidence/coefficient\_sensitivity.json}]{brauneevidence2026}.}
\label{tab:app-seam-sweep}
\end{table}

Increasing the midpoint weight from zero to 0.012 reduces mean midpoint offset
from 4.055 to 0.363 pixels, while row movement increases from 35 to 42 pixels.
Doubling it again gives only 0.326 pixels offset. At fixed midpoint weight,
raising the step penalty reduces movement and increases offset. All settings
retain two raw-ink intersections (three on the closed mask). Thus this example
shows the intended competing effects without establishing a unique best
coefficient, eliminating all intersections or measuring final recognition.
Input and code hashes, all 48 individual runs and the script are supplied with
the source. The pipeline coefficients were not changed~\cite[\texttt{Evidence/coefficient\_sensitivity.json}]{brauneevidence2026}.
\FloatBarrier
'''
t+='\n'+new;p.write_text(t,encoding='utf-8')
p=B/'3_Main_Report/design/3_4_segmentation.tex';t=p.read_text(encoding='utf-8')
anchor='\\subsubsection{Ink ownership and classifier crop preparation}'
insert=r'''A new sensitivity check on six fixed gaps from two real pages varies the midpoint and step penalties separately. At the implemented 0.012/0.02 setting, mean midpoint offset is 0.363 working pixels; removing the midpoint attraction raises it to 4.055 pixels. All tested settings still cross two raw-ink pixels across those seams, so this checks the weights' effects rather than proving an optimum. Appendix~\ref{app:des-seam-sweep} gives the settings and retained outcomes~\cite[\texttt{Evidence/coefficient\_sensitivity.json}]{brauneevidence2026}.

'''
t=t.replace(anchor,insert+anchor);p.write_text(t,encoding='utf-8')
p=B/'PROJECT_DESIGN_RECORDS.md';t=p.read_text(encoding='utf-8')
t+='''
## New coefficient sensitivity, 1 November 2026

`Evidence/coefficient_sensitivity.json` records 48 seam computations: eight
settings on the first, middle and last gaps of Owen01 and Yeukita01. The
parameterised reference matches the current default on all six seams. This
post-design study varies midpoint attraction and movement penalty separately;
it neither recovers a historical selection experiment nor establishes a global
optimum. Every setting retains the same two raw-mask intersections. No training,
inference, checkpoint change, coefficient change or hardware execution was needed.
''';p.write_text(t,encoding='utf-8')
def cites(t):return {k.strip() for a in re.findall(r'\\cite(?:\[[^\]]*\])?\{([^}]+)\}',t) for k in a.split(',')}
des='\n'.join(p.read_text(encoding='utf-8') for f in ['3_Main_Report','Appendices'] for p in (B/f).rglob('*.tex'))
lk=cites((ROOT/'Report/PrismSnapshot_2026-10-31/3_Main_Report/1_Literature_Study.tex').read_text(encoding='utf-8'));dk=cites(des)
baseline='\n'.join(p.read_text(encoding='utf-8') for p in (ROOT/'Report/PrismSnapshot_2026-10-31/3_Main_Report').glob('*.tex') if not p.name.startswith('3_'))
audit=json.loads((B/'SOURCE_AUDIT_COUNTS.json').read_text())
audit.update(design_and_appendix_sources=len(dk),shared_with_literature=sorted(dk&lk),union_literature_design=len(dk|lk),union_current_baseline_sections_and_revised_design=len(dk|cites(baseline)),external_citation_sites=len(re.findall(r'\\cite',des)),citations_with_locators=len(re.findall(r'\\cite\[',des)),numbered_equations_with_adjacent_basis=40)
audit['new_external_keys_vs_baseline_bib']=['pytorchctc','rohmlowside2026']
audit['additional_internal_evidence_key']='brauneevidence2026'
(B/'SOURCE_AUDIT_COUNTS.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps(audit,indent=2))
