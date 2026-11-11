from pathlib import Path
import re,json,hashlib
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'Report/DesignPolished_2026-11-01'
files=[p for folder in ['3_Main_Report','Appendices'] for p in (B/folder).rglob('*.tex')]
for p in files:
 t=p.read_text(encoding='utf-8')
 # Integrate citations for equations already containing substitutions into their prose.
 pattern=r'(\\begin\{equation\}.*?\\end\{equation\})\nImplementation/input basis: (\\cite(?:\[[^\]]*\])?\{[^}]+\})\n'
 def integrate(m):return m[2]+'\n'+m[1]+'\n'
 t=re.sub(pattern,integrate,t,flags=re.S)
 # Avoid a source citation dangling after a blue author-completion note.
 t=re.sub(r'\n\\designgap\{([^}]+)\}(~\\cite(?:\[[^\]]*\])?\{[^}]+\})',r'\2\n\\designgap{\1}',t)
 p.write_text(t,encoding='utf-8')
des='\n'.join(p.read_text(encoding='utf-8') for p in files)
audit=json.loads((B/'SOURCE_AUDIT_COUNTS.json').read_text())
sites=re.findall(r'\\cite(?:\[([^\]]*)\])?\{([^}]+)\}',des)
audit['citation_sites_total']=len(sites)
audit['external_citation_sites']=sum(any(k.strip() not in {'braunepipeline2026','brauneevidence2026'} for k in keys.split(',')) for _,keys in sites)
audit['citations_with_locators']=sum(bool(loc) for loc,_ in sites)
(B/'SOURCE_AUDIT_COUNTS.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
equations=json.loads((B/'EQUATION_AUDIT.json').read_text())
for e in equations:
 hits=[p for p in files if '\\label{'+e['equation']+'}' in p.read_text(encoding='utf-8')]
 assert len(hits)==1;e['file']=hits[0].relative_to(B).as_posix()
(B/'EQUATION_AUDIT.json').write_text(json.dumps(equations,indent=2),encoding='utf-8')
p=B/'FULL_REVIEW.md';t=p.read_text(encoding='utf-8').replace('eight main pseudocode blocks','five main pseudocode blocks plus six appendix procedures').replace('eight in Design and three independent checking procedures in the appendix','five in Design and six in the appendices (including three independent checking procedures)')
t+='''
## Final organisation after the calculation pass

The complete CTC and training-batch procedures, convolution-backward and
finite-difference examples, motion-file procedure and detailed implementation,
style-field and interface tables were retained in the supporting appendix. Main
Design keeps the network/CTC model, physical design, principal image/path
equations, worked values and five directly explanatory algorithms. Eleven total
algorithms and forty total equations remain. All relocation references compile.
The sweep inventory distinguishes included studies from older artifacts whose
metadata still needs recovery.
''';p.write_text(t,encoding='utf-8')
p=B/'README.md';t=p.read_text(encoding='utf-8').replace('- Added eight numbered algorithm blocks, representation contracts, explicit CTC','- Added numbered algorithm blocks, representation contracts, explicit CTC')
t+='''
Detailed CTC/training and motion-file pseudocode and the supporting reference
tables are now in Appendix A. Main Design retains five algorithms; the complete
package contains eleven. All forty numbered equations and their worked values
remain. See `SWEEP_INVENTORY.md` for included studies and older artifacts lacking
enough metadata for a controlled comparison.
''';p.write_text(t,encoding='utf-8')
# Reproducibility of the newly executed study.
sweep=json.loads((B/'Evidence/coefficient_sensitivity.json').read_text())
assert len(sweep['runs'])==48 and sweep['baseline_reference_matches']==6
for name,digest in sweep['inputs'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
for name,digest in sweep['code_sha256'].items():assert hashlib.sha256((ROOT/'Software/CleanPipeline3'/name).read_bytes()).hexdigest()==digest
print({'citation_sites':audit['citation_sites_total'],'external_sites':audit['external_citation_sites'],'equations':len(equations),'new_sweep_hashes':'verified'})
