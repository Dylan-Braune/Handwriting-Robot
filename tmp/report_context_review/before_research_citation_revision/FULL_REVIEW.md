# Full Design review — 1 November 2026

## Overall assessment

The chapter explains the complete implemented chain: photographed page → line
crops → text/writer; labelled lines → glyph paths and profile; new text → selected
trajectory → G-code → calibrated GPIO execution. It contains the core mathematics,
implementation decisions, worked values, local verification and material limits.
It is substantially developed rather than an outline or a function catalogue.

This is a **coverage review assuming the planned diagrams are completed**. It is
not an unconditional submission-compliance certificate. Missing installed hardware
values, eight unverified source/page checks, model/data provenance gaps and final
submission declarations cannot be resolved by assuming a figure exists. Those
conditions remain explicitly marked. No grade or physical performance is inferred.

## Appendix 4 check

The source is the actual APPENDIX4_2026.pdf, especially printed/PDF pp. 12–17,
checked against the annotated final-template notes. Each entry below distinguishes
the content from the remaining evidence/submission condition.

| Requirement | Present treatment | Condition / outcome |
|---|---|---|
| Core technical detail and own contribution, pp. 12–14 | Processing, numerical forward/backward operations, extraction, synthesis, geometry and motion receive the space; interface support is brief | Covered; framework-trained final weights are explicitly distinguished from numerical implementation |
| Compulsory **3.1 Design summary**, p. 14 | Exact heading; opening sentence; three-column task/implementation/completion-and-location table | Present; final completion declarations must match actual submission |
| Deliverables from proposal §4 line 5 and tasks in §6.1, pp. 14–15 | Grouped rows include research, capture/conditioning, separation, networks, training, maps/storage, synthesis/control, mechanism, integration and checks | Present as grouped tasks; integrated qualification belongs in Results rather than an invented Design completion claim |
| Compulsory **3.2 System block diagram**, p. 15 | Exact heading; hardware and processing architectures, operating narrative, stored representations, final functional-unit mapping | Covered assuming the hardware diagram is completed |
| Hardware section and complete circuit, p. 15 | 3.3, whole-system circuit slot plus pen/switch details, component operation and calculations | Figure must show actual values and readable text at least 6 pt at final scale; Vref/load/supply facts remain open |
| Hardware design equations/calculations, pp. 13, 15–16 | Pulley geometry, resolution, operating speed, static/dynamic load model, pulse delay, base current, calibration/error budget, driver current | Present with substitutions; illustrative assumptions do not verify installed torque or current margin |
| Software operation and flow diagrams, pp. 15–16 | Processing overview, architecture, three stages in 3.4–3.6, integration in 3.7; five main pseudocode blocks plus six appendix procedures | Functional content covered; see heading choice below |
| Source submitted separately on AMS, p. 16 | Explicit source-submission sentence in 3.7 | User must actually submit executable source separately; this edit does not upload it |
| Necessary mathematics rather than only function descriptions, p. 13 | Forty numbered equations across main text and supporting appendix, plus inline definitions, tensor dimensions and numerical examples | Covered at report scope; not a proof of every source-code line |
| Logical storyline and all relevant calculations, p. 16 | Hardware constraints precede the three representation changes; detailed derivatives/field dictionary in appendix | Covered; appendices must be imported with Design |
| Motivation, parameter significance and assumptions, p. 17 | Line-based recognition, shape judge, small offline glyph library, compute scope, joining/selection trade-offs, default-vs-selected settings | Covered; original rationale for an unverified experimental tracer or motion scheduler is not invented |
| Experiments/simulation explaining decisions, pp. 13, 17 | Recorded width selection, controlled numerical checks, real page examples, real candidate scoring, path round-trip checks, new seam sensitivity | Present; new sensitivity is labelled post-design and does not establish historical tuning or a global optimum |
| Typical 30–40 pages, p. 17 | Original class/font/margins retained; latest user requested citations and calculations expanded the content | Actual final count is recorded in VALIDATION.json; full-report compression remains necessary |

**Heading choice:** Appendix 4 pp. 15–16 explicitly calls for Hardware/Software
design headings, while also allowing project-specific organisation. The user
explicitly requested the three processing-stage headings and no generic Software
design heading. This revision follows that request. It is inaccurate to claim that
Appendix 4 says no software heading is needed. Software design content is present
throughout 3.4–3.7; exact heading compliance remains a deliberate presentation choice.
The two expressly compulsory numbered headings 3.1 and 3.2 are retained verbatim.

## Proposal contribution and task mapping

The approved-proposal snapshot is the unchanged PDF in the Prism export. Its
functional description and §4 contribution rows were read directly; the root
PDF text extraction mostly contains the blank form and cannot replace that source.

| Original contribution / §6.1 task | Where now explained | Limit retained |
|---|---|---|
| Investigate networks, feature extraction, conditioning, separation and trajectory methods | Literature Review; 3.4–3.6 rationale | Literature itself is not newly rewritten in this pass |
| Implement CNN, conditioning and separation | 3.4 and 3.5; numerical implementation appendix | Final supplied models originated in framework training |
| Simulate/evaluate classifiers on existing datasets and investigate settings | 3.5 corpus, architecture, training, numerical checks and four-width sweep | No invented layer-count search or missing matched final training experiment |
| Construct capture rig and embed algorithms | 3.3.6 and 3.7 | Mount dimensions/adapter details need records; Colab training is not embedded training |
| Test classifier on the embedded platform with captured pages | Local image checks explain processing; formal timing/accuracy belongs in Results | Windows software evidence is not ODROID qualification |
| Design CAD gantry and construct it | 3.3 mechanics, custom parts/assembly figure, drive and calibration | Final illustration should identify actual custom parts/dimensions; an assumed image does not establish a missing CAD record |
| Implement trajectory maps and motor control | 3.6 alignment, tracing, library, layout and conversion; 3.3 drive/pen | Library geometry is not natural measured pen chronology |
| Integrated gantry/classifier tests | 3.7 interface checks, conversion and physical-output illustration; Results for requirements | No QTP or unmeasured full-system result inserted into Design |
| Efficient/reliable reproduction and persistent multi-writer maps, §4 | Ordering/joins, nominal timing, limits, profile format and writer mapping | Nominal feed, counts and confidence do not establish physical accuracy or speed |

The final line-based reader is explained as a change to the proposal's early
character-isolation formulation, not silently described as identical to it.

## Sreekumar and Aston presentation review

Sreekumar's u23531003 report is inspiration, not an additional assessment rubric.
Useful practices retained here are: identify the actual first-principles work;
explain mathematics beside implementation; substitute numbers into calculations;
use actual intermediate system outputs; compare a model/reference with the
implementation; explain meaningful limitations; and place extended technical
detail in an appendix. Another project's components, measured values or design
reasons are not borrowed. Not every experiment in a different project applies here.

Aston's draft informs the indented, numbered pseudocode style: inputs/outputs,
assignment, conditions, loops and returns. The revision has eleven algorithm
blocks, five in Design and six in the appendices (including three independent checking procedures).
Real crops, actual stored profiles and actual generated paths make those algorithms
concrete. Unsupported diagrams or incomplete measurements in either reference
report do not establish facts about this robot.

## Important accuracy/polish changes in this pass

- Added the author's genuine glyph rationale, with existing Literature Review
  sources: Haines preprint PDF p. 4; Graves p. 18; One-DM p. 9. It distinguishes
  reference data for a new writer from generator training data/compute.
- Preserved the user's mechanics/motor voice and seven-subsection structure.
- Removed repetitive review-like caveats while retaining engineering limitations.
- Corrected the post-trim horizontal glyph shift and separate lead; operational
  body-band estimate; twelve ordinary variants plus possible prototype; and
  all-rejected crop-filter fallback.
- Corrected the orientation comparison to a score ratio 1.054 against 1.35.
- Kept final framework weight provenance, explicit numerical trainer/inference,
  selected 0.17 preprocessing and eleven-writer population distinct.
- Kept restored/archived crop evidence separate from the failed reduced-page run.
- Added immediate implementation/input citations and numerical examples for all
  forty numbered equations; existing hardware substitutions are retained.
- Added in-text citations to implementation explanations, project records and
  algorithm/figure/table captions. Unpaginated source uses function/section
  locators; unavailable external pages say **page to verify** inside `\cite`.
- Added a new reproducible seam-coefficient study: eight settings, six fixed
  gaps, 48 runs. It shows trade-offs but no unique optimum or eliminated crossing.

## Completion register even with diagrams assumed finished

1. **Hardware:** actual Vref and confirmed sense resistor; motor startup/running
   load, 3.3 V capacity and adapter rating; torque/starting-rate evidence at the
   installed drive setting; confirm mass; actual wiring/decoupling and omission
   of the diode; capture dimensions and light measurements if claimed.
2. **Data/model:** reader split counts and training host/epochs where recoverable;
   source-page provenance for profiles; matched final numerical/framework
   comparison; final checkpoint/map/preprocessing consistency. Defaults cannot
   fill missing historical run metadata.
3. **Sources:** eight flagged references need the author's manual source/page
   check. Verified page numbers must not be inferred from a bibliographic range.
4. **Submission:** truthful completed/incomplete labels, actual separate AMS code
   upload, import both appendices and rebuild the complete report. Confirm the
   final approved proposal version before assembly; preserve its specifications.
5. **Figures:** assumed for this review; six assets are still physically pending
   in the package. Their production briefs remain in FIGURES_AND_EVIDENCE.md.

No pipeline code, models or final profiles were changed. No training, deployment,
physical motion, commit or push occurred. New offline seam computations and the
earlier retained subsystem evidence are explicitly identified with provenance.

## Final organisation after the calculation pass

The complete CTC and training-batch procedures, convolution-backward and
finite-difference examples, motion-file procedure and detailed implementation,
style-field and interface tables were retained in the supporting appendix. Main
Design keeps the network/CTC model, physical design, principal image/path
equations, worked values and five directly explanatory algorithms. Eleven total
algorithms and forty total equations remain. All relocation references compile.
The sweep inventory distinguishes included studies from older artifacts whose
metadata still needs recovery.
