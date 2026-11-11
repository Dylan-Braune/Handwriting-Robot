from pathlib import Path
import json,re,hashlib,math
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'Report/DesignPolished_2026-11-01'
p=B/'3_Main_Report/design/3_5_classification.tex';s=p.read_text(encoding='utf-8')
start=s.index('For the complete labelled line');end=s.index('\n\nHere $p_t',start)
s=s[:start]+r"For the complete labelled line ``1.9. Vloggers using fame to support charities is a positive way'', the recorded forward--backward total is $\log Z=-0.47033843$. Equation~\ref{eq:des-ctc} gives $\mathcal L_{\rm CTC}=0.47033843$ and $Z\simeq0.6247908$. This loss was evaluated offline from the trained reader's actual logits and known transcript, without updating its weights."+s[end:]
start=s.index('The word ``Vloggers');end=s.index('\n\n\\subsubsection',start)
s=s[:start]+r"The word ``Vloggers'' in this actual transcript contains adjacent \texttt{g}s. They require separate emission runs with a blank between them: a direct skip between identical label states would collapse to one character. The same restriction is retained during forced alignment in Section~\ref{sec:des-alignment}; training sums admissible paths, whereas extraction selects one."+s[end:]
p.write_text(s,encoding='utf-8')

records=json.loads((B/'Evidence/real_calculations.json').read_text(encoding='utf-8'))
R=records['records'];assert R['reader']['alphabet_count']==74 and R['reader']['head_weight_shape']==[75,512]
for name,digest in records['inputs'].items():
    p=Path(name);p=p if p.is_absolute() else ROOT/p
    assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,name
for name,digest in records['code_sha256'].items():assert hashlib.sha256((ROOT/'Software/CleanPipeline3'/name).read_bytes()).hexdigest()==digest
assert abs(R['brightness']['G']/R['brightness']['background']-R['brightness']['ratio'])<1e-12
assert abs(R['seam']['local_cost']+R['seam']['predecessor_cost']+R['seam']['movement_cost']-R['seam']['accumulated'])<1e-12
assert abs(R['ctc']['p']-R['ctc']['q']-R['ctc']['gradient'])<1e-12
assert abs(R['finite_difference']['analytic']-R['finite_difference']['estimate'])<1e-9
assert R['timing']['returned']['clipped']==0
assert abs(sum(R['timing'][k] for k in ['drawS','travelS','dwellS'])-32.82231421802138)<1e-9

eqs=json.loads((B/'EQUATION_AUDIT.json').read_text(encoding='utf-8'))
trace_keys={'ratio':'brightness','integral':'brightness','otsu':'otsu','skew':'skew_trials','seam':'seam','seamcost':'seam_cost','fit':'fit','conv':'conv','bn':'bn','lstm':'lstm','ctc':'ctc','ctcgrad':'ctc','strokenorm':'stroke_width','viterbi':'viterbi','cutcost':'cut','glyphcoords':'glyph_point','connectedness':'line_connectedness','placement':'placement','anchor':'anchoring','jointscore':'existing bench.json candidates, first trial','machinecoords':'machine','timebudget':'timing','pagevote':'vote','ctcforward':'ctc','thinning':'thinning','gradcheck':'finite_difference','pulsetime':'motion_segment'}
omitted={'bnback','lstmback','rmsprop','adamw','convback'}
for e in eqs:
    key=e['equation'].split('-')[-1]
    if key in trace_keys:e['numerical_example']='Real input execution trace';e['trace_record']=trace_keys[key]
    elif key in omitted:e['numerical_example']='Invented scalar example removed; cited formula/derivation retained. No historical training update claimed.'
    elif key=='torque':e['numerical_example']='Actual pulley coefficient only; full numeric torque margin withheld pending measured load inputs.'
    elif key=='calibration':e['numerical_example']='Configured usable travel calculated; no homing counts invented.'
    elif key=='scaleerror':e['numerical_example']='Actual configured-geometry sensitivity coefficient; no measured uncertainty invented.'
    elif key=='current':e['numerical_example']='Conditional component-rating bound using actual phase rating and tentative sense resistance; measured Vref still pending.'
    else:e['numerical_example']='Substitution using actual configured component values; assumptions stated.'
(B/'EQUATION_AUDIT.json').write_text(json.dumps(eqs,indent=2)+'\n',encoding='utf-8')
p=B/'README.md';s=p.read_text(encoding='utf-8')
s=s.replace('All forty numbered equations retain numerical worked values.','All forty numbered equations remain. Generic scalar examples have been replaced with real-input traces or removed where no actual run values were available.')
s=s.replace('All forty numbered equations and their worked values\nremain.','All forty numbered equations remain; their real-input trace or derivation basis is recorded in `EQUATION_AUDIT.json`.')
s=re.sub(r'Current reading copy after the research-citation correction:.*','Current pagination is recorded in `VALIDATION.json`.',s)
s+='''
## Real system calculations

The manuscript no longer uses `Calculated example:` labels or invented scalar
values as design evidence. Read-only traces use Owen01, its archived first original
line, the actual trained reader, the selected eleven-writer classifier, and final
Owen/Dylan profiles. The generated Owen sentence is a new offline job, not a
historical physical execution. The trace catches and corrects the reader-head
count (74 alphabet characters plus blank = 75 outputs) and uses the actual job
feeds (900/2400 mm/min). Generic optimizer/derivative scalars with no real run
record were removed; the supported theory remains. Hardware gaps were not filled
with invented homing counts, load coefficients or Vref measurements.

Run `trace_real_calculations.py` to reproduce the execution traces. Full-precision
records, code/model/input hashes and settings are in `Evidence/real_calculations.json`.
No optimizer step, training epoch or hardware command is involved.
'''
p.write_text(s,encoding='utf-8')
p=B/'SOURCE_AUDIT.md';s=p.read_text(encoding='utf-8').replace('Constructed\ncalculation examples remain labelled and follow the equation they evaluate; they\ndo not need another citation to the source file after every numerical substitution.','Numerical substitutions now use real-input execution traces or actual component/configuration values; where no real run record exists, invented scalar examples were removed. Research citations remain with the underlying method, without reintroducing a code citation after each substitution.')
s=s.replace('All forty numbered equations and their numerical substitutions remain.','All forty numbered equations remain. Generic numeric examples have been replaced by real-input traces where applicable; others were removed when real historical or measurement values were unavailable.')
s+='''
## Real calculation correction

See Appendix B.7 and `Evidence/real_calculations.json` for the new traces. They
support numerical substitutions using real stored images, checkpoint parameters
and generated trajectory/G-code values. These are newly run offline traces, not
recovered historical design tuning, physical performance tests or new training.
No scholarly reference is substituted for that provenance. Formulas without
actual run inputs retain their cited derivation rather than an invented example.
'''
p.write_text(s,encoding='utf-8')
p=B/'SOURCE_AUDIT_COUNTS.json';c=json.loads(p.read_text(encoding='utf-8'));c.pop('numbered_equations_with_worked_values',None);c['numbered_equations']=40;c['numerical_substitution_policy']='Real-input traces or actual configuration; unavailable historical/measurement inputs not invented.';p.write_text(json.dumps(c,indent=2)+'\n',encoding='utf-8')
p=B/'FULL_REVIEW.md';s=p.read_text(encoding='utf-8')
s=s.replace('Retained numerical examples for all forty numbered equations; existing hardware\n  substitutions are retained.','Retained all forty numbered equations; actual system substitutions and trace values\n  replace the generic numerical examples where applicable.')
s=s.replace('All 40 numbered equations, 11 algorithms, original sweep and new 48-run\ncoefficient check remain.','All 40 numbered equations, 11 algorithms, original sweep and new 48-run\ncoefficient check remain.')
s+='''
## Real calculation pass

All generic `Calculated example:` paragraphs were replaced by actual input traces
or removed when no historical or measured inputs were available. Read-only trace
records establish the ratio/integral/Otsu calculation, angle selection, seam cost,
crop fit, real network activations, CTC loss/gradient and finite difference, alignment,
glyph cut/coordinates, line connection count, anchoring, placement, machine mapping,
motion-count conversion, generated distance/dwell budget and two-crop writer vote.
The full manifest records exact scope, input/model/code hashes and precision.
These calculations show operation, not independent physical-system qualification.
The 75-output reader head and actual 900/2400 mm/min generated-job feeds were
corrected against runtime values. Historical homing counts, torque margin and Vref
remain unfilled. No models, final profiles, pipeline code or hardware were changed.
'''
p.write_text(s,encoding='utf-8')
p=ROOT/'Report/REPORT_COLLABORATION_CONTEXT.md'
with p.open('a',encoding='utf-8') as f:f.write('''

## Real calculation preference — 1 November 2026

The user rejects repeated `Calculated example:` paragraphs with arbitrary scalar
inputs. Use actual system dimensions/configuration, recorded experiments or values
captured in read-only processing of the existing pages/profiles/models. Explain
where each number was used, without presenting a new trace as historical tuning
or physical proof. If actual input values are unavailable, retain the formula and
its research/derived basis without inventing an example. All generic labels have
been removed from the current DesignPolished manuscript. The new manifest is
Evidence/real_calculations.json and reproducible script trace_real_calculations.py.
Actual reader alphabet is 74 characters; head is 75 outputs including blank
(confirmed 75x512 head and 160x1x75 output), correcting the previous 73/74 statement
and architecture diagram. The offline generated Owen job uses configured feeds
900/2400 mm/min, 56 transitions and a 32.8223 s nominal geometry/dwell budget.
No training epochs, optimizer steps, profile/checkpoint writes or physical calls
were run. Earlier synthetic numerical unit checks remain labelled test fixtures.
''')

for folder in ['3_Main_Report','Appendices']:
    for p in (B/folder).rglob('*.tex'):
        data=p.read_bytes();assert not any(x<32 and x not in [10,13] for x in data),p
        assert b'Calculated example:' not in data
print('Real-input manifest hashes and numeric identities checked; documentation updated.')
