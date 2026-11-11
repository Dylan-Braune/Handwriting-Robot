from pathlib import Path
import json,re,hashlib
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'Report/DesignPolished_2026-11-01'
validation=json.loads((B/'VALIDATION.json').read_text(encoding='utf-8'))
note=f'''

## Full Design review, citations and worked calculations — 1 November 2026

Latest working/delivery folder: `Report/DesignPolished_2026-11-01/`. Earlier
DesignRevision and DesignVisualRevision folders are preserved. Delivery names:
`output/pdf/Handwriting_Robot_Design_Polished_2026-11-01.pdf` and
`output/Handwriting_Robot_Design_Polished_Prism_Source_2026-11-01.zip`.
Full requirement/coverage review: `FULL_REVIEW.md`; source/page audit:
`SOURCE_AUDIT.md`; pending/available sweep records: `SWEEP_INVENTORY.md`.

The user supplied the actual reproduction rationale: a few offline photographed
pages per writer, no measured pen chronology, and avoiding an additional
first-principles generative training task on the ODROID. Main Design now supports
this with existing Literature Review keys Haines (inspected preprint PDF p. 4),
Graves 2013 (p. 18), and One-DM (p. 9). It explicitly distinguishes per-writer
reference samples from a generator's broad training corpus; do not reinstate the
universal claim that every learned generator needs many pages per new writer.
No ODROID generator training-time measurement exists.

Latest user instruction places strong priority on visible in-text `\\cite`
commands for claims and equations, with page numbers wherever applicable, and
actual numerical substitutions after equations. All forty numbered equations
now have numerical examples/input substitutions and nearby source citations.
Borrowed methods retain inspected primary page locators. Project implementation
facts cite unpaginated code/file/function locators; supplied measurements and
training facts cite accompanying unpublished project records. Eight unavailable
source-page checks remain `page to verify` inside citations plus blue flags.
Do not invent page numbers. Two additional outside publications were reused from
the existing Literature Review; one internal evidence reference was added.
The full current baseline/Design reference union is 41.

The user also requested all available parameter sweeps and authorised additional
offline coefficient tests. The recorded four-width eleven-writer training sweep
is retained. A new seam sensitivity study used eight settings on six preselected
gaps from Owen01/Yeukita01, 48 runs. All six default paths exactly match the
unmodified source. It varies midpoint and movement penalties separately, retains
two raw-ink intersections at every setting, and does NOT prove a unique optimum
or historical selection rationale. Input/code hashes and all outputs are in
`Evidence/coefficient_sensitivity.json`; the script is supplied. Older nTries and
rendering sweep images were found but lack adequate recovered run metadata for a
controlled final-model comparison; do not manufacture results from filenames.

Final preview recorded {validation['design_pages']} Design pages, three reference
pages and {validation['appendix_pages']} supporting appendix pages
({validation['pages']} total), with the original class unchanged. Eleven algorithms
remain: five main explanations and six appendix procedures. Extended training,
derivatives, file-conversion pseudocode and reference tables moved to Appendix A
to retain complexity without bloating the main narrative. Integration was tightened.
Main Design still has six intentional figure slots and eight completion notes.

The full review assumes planned diagrams are complete but does not certify full
submission readiness: actual hardware values, some model/data provenance,
source-page verification, completion declarations and AMS source upload remain
conditions. Appendix 4's literal hardware/software-heading wording is acknowledged
in FULL_REVIEW; the user's explicit preference for three software-stage headings
was followed. Do not claim the source never requests a Software design heading.

Code corrections in prose include the glyph's post-trim horizontal shift/lead,
operational body-band estimate, twelve ordinary variants plus possible prototype,
the crop-filter all-rejected fallback and the orientation score ratio. Final
framework-trained model provenance, NumPy trainer/inference and selected 0.17
eleven-writer configuration remain distinct. No pipeline code, models, profiles,
training, hardware execution, deployment, commit or push was changed/performed.
'''
p=ROOT/'Report/REPORT_COLLABORATION_CONTEXT.md'
assert '## Full Design review, citations and worked calculations' not in p.read_text(encoding='utf-8')
with p.open('a',encoding='utf-8') as f:f.write(note)
p=B/'README.md';s=p.read_text(encoding='utf-8')
s+=f"\nFinal reading-copy pagination: {validation['design_pages']} Design pages, three reference pages and {validation['appendix_pages']} appendix pages ({validation['pages']} total). Page numbers are local to this preview; full-report import changes them.\n"
p.write_text(s,encoding='utf-8')
print({'context_updated':True,'design_pages':validation['design_pages']})
