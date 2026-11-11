from pathlib import Path
import re
R=Path(__file__).resolve().parents[2];B=R/'Report/IntegratedReport_2026-11-01'
replacements={
 'Prototype pen-motor switching circuit supplied with the draft. Supply and protection details require confirmation against the installed circuit.':'Prototype pen-motor switching circuit. The supply arrangement and protection limitations are discussed below.',
 'The square-and-triangle comparison already included in the draft is useful for this purpose':'The square-and-triangle comparison provides this geometric check',
 'Final run settings require their own training records.':'The table describes implemented defaults rather than a recovered record of every completed training run.',
 'these fallbacks should be counted when interpreting a nominally shape-based dataset.':'these cases retain their original crop rather than a successfully normalised shape.',
 "retains the proposal's exact environmental conditions for confirmation during the final demonstration.":"compares the retained evidence with the proposal's exact environmental conditions.",
}
for p in [*B.glob('3_Main_Report/**/*.tex'),*B.glob('Appendices/*.tex')]:
 t=p.read_text(encoding='utf-8')
 for a,b in replacements.items():t=t.replace(a,b)
 t=re.sub(r'(\\cite(?:\[[^\]]*\])?\{[^}]+\})(?=\\cite)',r'\1,~',t)
 p.write_text(t,encoding='utf-8')
print('Remaining editorial draft/confirmation phrasing replaced with factual descriptions.')
