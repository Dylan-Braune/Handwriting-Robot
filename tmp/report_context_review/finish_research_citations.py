from pathlib import Path
import re,json
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'Report/DesignPolished_2026-11-01'
files=[*sorted((B/'3_Main_Report').rglob('*.tex')),*sorted((B/'Appendices').glob('*.tex'))]
for p in files:
    s=p.read_text(encoding='utf-8').replace(r'project\textquotesingle s',"project's").replace(r'writer\textquotesingle s',"writer's")
    p.write_text(s,encoding='utf-8')
text='\n'.join(p.read_text(encoding='utf-8') for p in files)
assert not any(k in text for k in ['braunepipeline2026','brauneevidence2026'])
def cited(t):return {k.strip() for g in re.findall(r'\\cite(?:\[[^\]]*\])?\{([^}]+)\}',t) for k in g.split(',')}
baseline=ROOT/'Report/PrismSnapshot_2026-10-31'
lit=cited((baseline/'3_Main_Report/1_Literature_Study.tex').read_text(encoding='utf-8'))
other=cited('\n'.join(p.read_text(encoding='utf-8') for p in (baseline/'3_Main_Report').glob('*.tex') if p.name!='3_Design_and_Implementation.tex'))
design=cited(text)
sites=re.findall(r'\\cite(?:\[([^\]]*)\])?\{([^}]+)\}',text)
pending=sorted(set(re.findall(r'\\designsourceflag\{([^}]+)\}',text)))
counts={'literature_sources':len(lit),'design_and_appendix_sources':len(design),'shared_with_literature':sorted(lit&design),'union_literature_design':len(lit|design),'union_current_baseline_sections_and_revised_design':len(other|design),'pending_manual_source_checks':pending,'external_citation_sites':len(sites),'citations_with_locators':sum(bool(loc) for loc,_ in sites),'citation_sites_total':len(sites),'numbered_equations_with_worked_values':40,'internal_citation_sites':0,'removed_internal_citation_sites':285,'policy':'Research citations for borrowed methods; project derivations, configuration and experiments identified as own work; occasional prose pointers to implementation.'}
(B/'SOURCE_AUDIT_COUNTS.json').write_text(json.dumps(counts,indent=2)+'\n',encoding='utf-8')

# Retain the verified primary-source table, replacing the superseded self-citation policy.
old=(B/'SOURCE_AUDIT.md').read_text(encoding='utf-8')
verified=old.split('| Key | Inspected material and in-text locator | Claim supported and scope |')[1].split('Relevant complete source pages')[0]
unverified=old.split('| Key | Citation site | What must be checked | Why left flagged |')[1].split('These are **referenced but unverified')[0]
unverified='\n'.join(line for line in unverified.splitlines() if '`crow1984`' not in line)
audit='''# Research citation audit — 1 November 2026

## Current policy, following the user's correction

The manuscript uses research papers and relevant manufacturer/standards material
to support borrowed methods and external technical facts. The 285 repeated code
and internal-record citation sites have been removed, as have their two bibliography
entries. They have not been mechanically replaced with unrelated papers.

Project-specific parameter values, circuit construction, measured observations,
training records, sweeps, heuristics and coordinate transforms are described as
this project's work. Their supporting files remain in the source package. A paper
cannot establish what was wired, trained or measured in this project. Constructed
calculation examples remain labelled and follow the equation they evaluate; they
do not need another citation to the source file after every numerical substitution.

The few module pointers in ordinary prose are reading aids, not numbered scholarly
references. The detailed evidence-manifest paths are introduced once in the evidence
appendix and retained with the raw results. Main-text figure, table and algorithm
captions no longer carry repetitive self-citations.

Verified printed page numbers remain in the in-text cite commands. Author copies
without printed pagination explicitly use PDF page numbers. Unpaginated manufacturer
or framework documentation uses section headings. Unavailable source/page checks
retain `page to verify` and a visible blue source flag; no page was invented.

## Verified research and technical sources

| Key | Inspected material and in-text locator | Claim supported and scope |'''+verified+'''
| `bradley2007` | Author-hosted six-page reading copy, PDF pp. 2–3, Secs. 2.2–3, Eq. (2); complete pages rendered and read | Local-average illumination comparison and four-corner integral-image sum. Project forms a ratio and uses Otsu; it does not claim to reproduce Bradley's threshold rule. Replaces the unavailable Crow citation at this site. |
| `brink2012` | Author-hosted journal copy, printed p. 162; complete page rendered and read | Ink width is useful for writer identification and depends on pen and style. Does not prove which features the initial project CNN used. |
| `haralick1987` | Author-hosted original scan, printed pp. 533 and 541, Defs. 1 and 36; complete pages rendered and read | Binary dilation and closing, including closing narrow breaks. Project kernel-size choices remain engineering choices. |
| `haines2016` | Author-preprint PDF p. 4, Sec. 4 | Approximately four A4 sample pages in that study; no universal sample minimum. |
| `dai2024` | ECCV paper, printed p. 9 | One generation reference distinguished from the 339-writer training corpus and four-GPU setup; no ODROID timing inferred. |

Graves 2013 was additionally inspected at printed p. 18 for online pen positions
and end-of-stroke markers. The project has offline photographs, so temporal order
is not measured. Haines and One-DM already appear in the user's Literature Review.
The three newly used research entries above were already in the supplied bibliography;
their reading-copy URLs have been recorded. No additional bibliography entry was needed.

## Referenced sources still requiring the user's manual check

| Key | Citation site | What must be checked | Why left flagged |'''+unverified+'''

Crow is no longer a pending manuscript source: Bradley and Roth supply an inspected
explanation of the exact rectangular-sum identity used here.

## Equations and original contribution

The established CTC, LSTM, batch-normalisation, thinning, RMSprop, Adam/AdamW,
seam dynamic programme and driver-current relation retain their research or
manufacturer citations next to the explanation. Matrix derivatives and the central
finite-difference expression now include their algebraic derivation. The remaining
project geometry, scale conversions, weighted vote, connector/selection heuristics
and time budgets are explicitly presented as derived calculations or project definitions.
This avoids attributing a custom formula or empirical sweep result to an unrelated paper.

All forty numbered equations and their numerical substitutions remain. Eleven
algorithm blocks, the recorded width sweep and the 48-run seam-coefficient study
remain unchanged in content and measured results. The equation audit records the
basis of each formula separately from a code locator.

## Current citation counts

'''+f"The Design and appendices cite **{len(design)}** distinct external sources at **{len(sites)}** citation sites. The union with the current baseline main-report sections is **{len(other|design)}** sources. **{len(pending)}** source/page checks remain flagged. There are **zero** code/internal-record bibliography citations in the manuscript.\n"
(B/'SOURCE_AUDIT.md').write_text(audit,encoding='utf-8')

eqs=json.loads((B/'EQUATION_AUDIT.json').read_text(encoding='utf-8'))
bases={
'integral':'Bradley and Roth, PDF p. 2 Eq. (2); exclusive-prefix indexing adaptation',
'ratio':'Local-mean principle: Bradley and Roth, PDF pp. 2–3; project ratio/Otsu combination',
'otsu':'Otsu 1979, page to verify', 'skew':'Postl 1986, page to verify; project squared-projection objective',
'seam':'Avidan and Shamir, PDF p. 3 Sec. 3; project orientation and step penalty',
'conv':'Kizilirmak and Yanikoglu pp. 4–5; matrix formulation Chellapilla, page to verify',
'bn':'Ioffe and Szegedy, PDF pp. 4–5 Alg. 1', 'lstm':'Graves 2013 p. 5 Eqs. (9)–(11), with omitted peepholes stated',
'ctc':'Graves 2006, PDF pp. 2–3', 'ctcgrad':'Graves 2006, PDF pp. 5–6; stated occupancy adaptation',
'strokenorm':'Zhang and Suen p. 237; Haralick p. 533; project width rule',
'viterbi':'Maximum-path adaptation of Graves 2006 CTC lattice, PDF pp. 4–5',
'convback':'Chain-rule derivation shown; matrix convolution reference Chellapilla, page to verify',
'gradcheck':'Taylor-expansion derivation shown, with smoothness and fixed-mask limitations',
'basecurrent':'ROHM low-side switch article, stated section; Ohm law with stated assumed VBE',
'current':'A4988 rev. 8 datasheet p. 9',
'bnback':'Ioffe and Szegedy, PDF p. 4; simplified channel derivative',
'lstmback':'Chain rule on cited LSTM recurrence; Graves 2013 pp. 4–5',
'ctcforward':'Graves 2006, PDF p. 4 Sec. 4.1; log-space adaptation',
'rmsprop':'Tieleman/Hinton, PDF p. 29; project coefficient and decay explicitly stated',
'adamw':'Kingma/Ba p. 2 Alg. 1; Loshchilov/Hutter p. 3 Alg. 2',
'thinning':'Zhang and Suen, p. 237'}
for e in eqs:
    suffix=e['equation'].split('-')[-1]
    e['basis']=bases.get(suffix,'Project definition or geometric/dimensional calculation; input values and assumptions stated in surrounding prose')
    e.pop('locator',None)
(B/'EQUATION_AUDIT.json').write_text(json.dumps(eqs,indent=2)+'\n',encoding='utf-8')

p=B/'README.md';s=p.read_text(encoding='utf-8')
start=s.index('All forty numbered equations now have a nearby')
end=s.index('The original four-width training sweep',start)
s=s[:start]+'''All forty numbered equations retain numerical worked values. Borrowed methods
use research citations with verified page locators where available. Seven source/page
checks remain visibly flagged. Repeated citations to the project's code and records
have been removed throughout; occasional code-module pointers remain in prose.
Own design choices, derivations and experiments are identified as project work,
with evidence provenance retained in the source package. See `SOURCE_AUDIT.md`.

'''+s[end:]
s=re.sub(r'Final reading-copy pagination:.*', 'Final reading-copy pagination is recorded in `VALIDATION.json`; full-report import changes it.',s)
p.write_text(s,encoding='utf-8')
p=B/'FULL_REVIEW.md';s=p.read_text(encoding='utf-8').replace('eight unverified source/page checks','seven unverified source/page checks').replace('**Sources:** eight flagged references','**Sources:** seven flagged references')
s=s.replace('Added immediate implementation/input citations and numerical examples for all\n  forty numbered equations; existing hardware substitutions are retained.', 'Retained numerical examples for all forty numbered equations; existing hardware\n  substitutions are retained. The research citation policy below supersedes the earlier code-citation pass.')
s=s.replace('Added in-text citations to implementation explanations, project records and\n  algorithm/figure/table captions. Unpaginated source uses function/section\n  locators; unavailable external pages say **page to verify** inside `\\cite`.', 'Cited borrowed methods with research and technical sources. Repeated project-code\n  and record citations were removed at the user\'s request. Unavailable external\n  pages say **page to verify** inside `\\cite`; own calculations remain identified.')
s+='''
## Research citation correction requested by the author

Removed 285 repeated code/internal-record citation sites and both self-reference
bibliography entries throughout Design and both appendices. Retained occasional
module pointers as reading aids. Borrowed methods retain research citations;
custom costs, transforms, heuristics and measurements are clearly the project's
own work. Checked and added existing bibliography sources Bradley/Roth,
Brink et al. and Haralick et al. with page locators. The verified integral-image
source replaces the unverified Crow citation, leaving seven page/source checks.
All 40 numbered equations, 11 algorithms, original sweep and new 48-run
coefficient check remain. No experimental result was changed by this pass.
'''
p.write_text(s,encoding='utf-8')
p=ROOT/'Report/REPORT_COLLABORATION_CONTEXT.md'
with p.open('a',encoding='utf-8') as f:f.write('''

## Citation preference correction — 1 November 2026

The user clarified that citations should refer to existing research papers and
technical sources, not repeated locations in project code. The current DesignPolished
manuscript removes all 285 code/internal-record citation sites and both bibliography
self-reference entries. Preserve this policy: occasional code-module pointers in
ordinary prose are welcome, but no code citation after each paragraph or calculation.
Research citations retain page numbers; own design definitions, derivations,
parameters and experimental results are identified as project work, without falsely
attributing them to an external paper. Raw provenance stays in the source package.
Bradley/Roth, Brink and Haralick were checked and cited using existing bibliography
entries. Bradley replaces the inaccessible Crow source for integral-image sums;
seven external source/page flags now remain. This supersedes the earlier context
entry that requested internal code/evidence citations throughout the manuscript.
''')
print(json.dumps(counts,indent=2))
