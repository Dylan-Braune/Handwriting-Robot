# Report collaboration context

## Expanded poster overview, 6 November 2026

Latest delivery is output/Poster_Overview_2026-11-06/
BrauneDR_23539195_Poster_Overview.pptx and .pdf, plus Poster_Preview.png.
Only Project overview changed, expanding the goal of retaining individual style
and the sequence from photographed pages through separate recognition networks,
labelled glyph profiles and G-code to ODROID-controlled physical drawing. The
final render was compared with the prior poster and differs only in the overview
text area. Template, pictures, subsystem descriptions and results are unchanged.
Final package/layout/table/chart checks pass. Builder is
 tmp/poster_rebuild_2026-11-06/build/overview.mjs.


## Technical poster descriptions, 6 November 2026

Latest poster is output/Poster_Technical_2026-11-06/
BrauneDR_23539195_Poster_Technical.pptx and .pdf, with Poster_Preview.png.
Only the four descriptive paragraphs and the gantry heading changed from the
restored-template version. Gantry block is now titled Gantry Plotting. Descriptions
name local background normalisation, Otsu thresholding, projection profiles and
seams, Zhang and Suen thinning, CNN, two BiLSTM layers, CTC decoding and alignment,
writer stroke normalisation, glyph profiles, Bresenham stepping and A4988 drivers.
Font size is reduced slightly to fit. Original header and footer remain pixel
identical, all 12 image frames/crops unchanged, results unchanged. Final package,
layout, chart/table and font checks pass. Builder is
 tmp/poster_rebuild_2026-11-06/build/technical_blocks.mjs.


## Original poster template restored, 6 November 2026

Latest delivery is output/Poster_Template_Restored_2026-11-06/
BrauneDR_23539195_Poster_Template_Restored.pptx and .pdf plus Poster_Preview.png.
The user clarified that only the central white canvas may be edited. The original
header, footer, separator and author/title block are restored and verified pixel
identical against the supplied template render. All 12 source image frames and
crops remain unchanged. Summaries use smaller text and simple complete sentences.
Text validation uses the documented first-principles score 94.39%, shown 94.4%.
The writer and physical-page scores are unchanged existing records, not relabelled
as new scratch physical tests. Result highlights now show mean absolute length
error 0.297 mm (70 measurements) and mean job-normalised completion time 2.43 s per
character (25 confirmed completed jobs, including generation and physical drawing).
The pooled character-weighted ratio 2.15466 s/char is not the displayed statistic.
No tests were rerun. Layout, font, native chart/table and workbook checks pass.
Builder is tmp/poster_rebuild_2026-11-06/build/restore_template.mjs. The source
poster and previous versions are preserved.


## User-arranged poster polished, 6 November 2026

The user provided a newly arranged Downloads/Poster Template.pptx. The supplied
slide was imported and edited, filling headings, short summaries and result
highlights. All twelve image frames, crops and image-byte hashes were verified
unchanged. Original university artwork is retained. Four subsections are Image
processing, Recognition, Motion control and Handwriting synthesis. A native
parameter table and recognition comparison chart fill the reserved spaces. The
existing physical writer confusion image stays in place. Relative shape score
86.7% is labelled against natural variation. Maximum measured length error
0.49 mm and 25/25 completed jobs within 4 s/character are highlighted.
Classification timing remains above target. No new data processing occurred.

Deliveries are output/Poster_Updated_2026-11-06/
BrauneDR_23539195_Poster_Polished.pptx and .pdf plus Poster_Preview.png.
Package, layout, native table/chart, workbook and Arial checks pass. PDF is a
high-resolution visual export from the finalized PPTX render. Authoring source
is tmp/poster_rebuild_2026-11-06/build/polish_updated.mjs. Source poster and earlier
versions are preserved. Report wording and experimental data are unchanged.


## Poster rebuild with four image panels, 6 November 2026

The user requested a fresh poster following their example image, with rounded
section blocks. The new A1 layout places the description and functional flow at
the top, four image-led subsystem panels in the middle and results at the bottom.
The panels are page processing, text and writer recognition, handwriting
reproduction, and gantry/pen control. The recognition image area is deliberately
blank for a genuine screenshot, as the user allowed. Other panels use real page
boundary images, matched original/robot word photos and the actual gantry photo.
Black/navy, blue, silver and white styling. Results are unchanged.

Deliveries are output/Poster_Rebuilt_2026-11-06/
BrauneDR_23539195_Poster_Rebuilt.pptx and .pdf, plus Poster_Preview.png.
The editable PowerPoint passed package, layout, Arial-font, chart-workbook and
import checks and was rendered after finalization. The PDF is a high-resolution
visual export of that final render. Authoring source is
 tmp/poster_rebuild_2026-11-06/build/poster.mjs. Previous posters and the six-slide
presentation are preserved. No report text, experimental data or code was changed.


## Poster and five-minute presentation mocks, 6 November 2026

The user requested visually appealing editable mocks in black, silver and blue,
with a full functional flow and subsystem mini visual tests. Two PowerPoint files
and matching PDF previews were created in output/Poster_and_Presentation_2026-11-06.

- BrauneDR_23539195_Poster_Mock_v2.pptx and .pdf, one A1 portrait page at the
  supplied template dimensions. Header, author and supervisor are updated; original
  UP branding artwork is retained. Five subsystem rows cover segmentation,
  recognition, glyph extraction, synthesis/G-code and motion. Each has real visual
  evidence. Editable flow diagrams and two native charts are included.
- BrauneDR_23539195_Presentation_Mock.pptx and .pdf, six numbered 16:9 slides.
  Context, full flow, preprocessing, glyphs/motion, measured performance and
  timing/demo plan. Speaker notes total 280 seconds (4 minutes 40 seconds) and
  include sources. The six-slide presentation was also built in the live session.

Authoring source is tmp/poster_mock_2026-11-06/build/content.mjs. The runner uses
@oai/artifact-tool and validates native charts with embedded editable workbooks.
The poster's superseded first mock is preserved privately, not in the delivery folder.
Final PowerPoint exports passed package/layout/font/chart checks and were rendered.
PDFs are high-resolution visual previews made from verified final slide renders,
because the tool's direct PDF exporter fails on a missing standard-fonts module.
The PowerPoint text, diagrams and charts remain editable. No native PowerPoint
application opening was claimed or performed. All plots and figures use existing
project data. No inference, scoring, training or gantry run occurred.

The supplied exam guidance distinguishes a five-minute introduction, questioning,
and a separate ten-minute demo. The poster event is informal and A1 portrait is
preferred. The public poster upload is required even for competition entrants.
Do not claim numerical or embedded compliance beyond the recorded report results.
The latency chart clearly labels a conditional laptop estimate, not a measured pass.

## Latest QTP2, laptop timing and conclusion revision, 2 November 2026

QTP1 remains unchanged. QTP2 was restored from the supplied SubmissionRevision ZIP,
including validation partitions/caveats, both reader comparison columns, 221 exact
lines, the 83.8% confidence example and author confusion matrix. The matrix is
shown in QTP2 rather than duplicated in Appendix C.

Discussion is now three pages with all required Appendix 4 headings. Conclusion
is completed and reconciled to the current 0.49 mm maximum error and 86.68%
relative geometric score. Earlier notes saying Conclusion is unchanged are stale.

QTP4 now includes thirteen laptop pages, each with three recorded repetitions,
from Report/_dev/qtp_data/qtp4_results.json and qtp4_log.txt. Both text and author
stage times are given. All thirteen page means exceed their full-request budgets.
The separate fourteen-page batched-library benchmark's 2.76 network factor is
applied conditionally to selected laptop network stages with unchanged image time.
L3, L4 and L5 would fit at 30.63, 35.59 and 39.70 seconds if that factor transfers.
These are estimates, not new measured passes or demonstrated embedded gains.
ODROID measurements remain separately labelled. No tests, inference or training
were rerun. The new evidence ZIP includes raw laptop records and the page index.

All twelve VisualTests PDFs were replaced byte-for-byte from
Report/SubmissionRevision_2026-11-02/Figures/VisualTests as requested. Pre-Results
wording remains unchanged. These larger revised figures move Results to page 50.
Main report totals 67 pages, Results 14, Discussion 3, Conclusion 1. Complete PDF
is 112 pages. Local pdfLaTeX/BibTeX succeeds without undefined references or
text overflow. Changed sections and supplied figures were rendered and checked.
The app preview still exceeds its 20 MiB limit.

Current source remains Report/ResultsDiscussionRevision_2026-11-02/full_report.tex.
Latest exports are output/pdf/Handwriting_Robot_Final_Revision_2026-11-02.pdf,
output/Handwriting_Robot_Final_Revision_Source_2026-11-02.zip,
output/Handwriting_Robot_Final_Revision_Evidence_2026-11-02.zip and
output/Final_Revision_Notes_2026-11-02.md. The source ZIP contains 100 verified
compile inputs plus README and hash manifest. No commit, push or deployment.

## QTP1 and shorter visual appendix, latest revision on 2 November 2026

The user requested no more data processing or experiments. Only the report was
edited, using existing scores and pictures. QTP1 now explicitly includes the text
reader and author classifier, restores the full eleven-writer table including exact
word match, moves the existing row-normalised physical author confusion matrix
into QTP1, and retains the current geometric test and examples. The duplicate
physical classifier bar plot and three redundant main-text plots were removed.
No scores were changed or recomputed.

Appendix C is now Qualification evidence. Long 184-line, 83-word and 70-reading
tables were removed from the PDF, along with repetitive summaries. The complete
transcribed records remain in the new evidence ZIP. Four existing word comparisons
for Dylan, Abhinav, Thiya and writer 153 were added as visual evidence, reusing their
original images and stored nearest-reference F-scores. Labels were retypeset at
10.5 pt Times New Roman without running image analysis or scoring. Original genuine
versus genuine and robot versus original examples are clearly distinguished from
the ordinary-reference writer averages. Short training-log extracts and all fourteen
physical-page photographs remain. Design appendices A and B are unchanged.

Latest working source remains Report/ResultsDiscussionRevision_2026-11-02.
The main report is now 66 pages, Results 12, all appendices 22 (nine fewer than the
previous delivery), and the complete PDF 110 pages. Pre-Results wording, Discussion
and Conclusion are unchanged from the previous delivery. The earlier Conclusion
inconsistencies remain outside scope. Local compilation and rendered-page checks
pass. The built-in app preview still exceeds its 20 MiB limit.

Latest deliveries are output/pdf/Handwriting_Robot_QTP1_Visual_Revision_2026-11-02.pdf,
output/Handwriting_Robot_QTP1_Visual_Source_2026-11-02.zip,
output/Handwriting_Robot_QTP1_Visual_Evidence_2026-11-02.zip and
output/QTP1_Visual_Revision_Notes_2026-11-02.md. Prefer these over the earlier exports.

## Results, Discussion and evidence revision, 2 November 2026

Latest supplied draft is Downloads/SubmissionRevision_2026-11-02.zip, SHA-256
ab2ae73580ae6917bfa40b0e0e7495fa0a456a839a96dd4f035143c380810d6e.
The revised working copy is Report/ResultsDiscussionRevision_2026-11-02,
main file full_report.tex. The older SubmissionRevision folder and the open
standalone segmentation document were left untouched.

The user authorised changes only to Results, Discussion and appendices. All
pre-Results wording remains unchanged. The only repairs in a Design source are
two missing dollar delimiters in reproduction pseudocode. A separate build
compatibility file handles an undefined \inP command and XeTeX microtype settings.
Do not infer permission to rewrite protected sections from this revision.

After being asked about checkpoint/host provenance, the user said to use the
existing results. Retain their documented setups. Do not describe the archived
library-trained physical-page scores as new scratch-model tests or offline
Windows geometry as an ODROID execution measurement.

QTP1 now uses classifier scores and a direct geometric natural-variation baseline,
not the older descriptor-correlation experiment. Equal-writer ordinary-reference
geometric ratio is 86.68% on 83 cross-page repeated words from eleven writers,
with a conditional word-type bootstrap range 83.65 to 89.65%. Closest-reference
ratio is 87.89%. Neither is recognition accuracy or a Pearson correlation.

QTP5 uses the user's ten new square/triangle samples, 70 feature lengths, max
absolute error 0.49 mm, MAE 0.297 mm and RMS 0.327 mm. Do not infer absolute bed
position or measurement uncertainty from those lengths. Main physical reader
84.41%, writer 159/184 (86.41%), page pools 14/14. Original validation remains
reader 94.06%, writer 153/156 (98.08%). There are 25 explicitly completed ODROID
reproduction timings. Classification totals remain 46.26 s and 55.49 s.

The unchanged Conclusion still has old 0.37 mm and 0.697 figures. It was outside
scope and needs reconciliation before submission. Protected Design's 97.51%
scratch writer claim differs from archived 149/156 (95.51%) evaluation and the
supplied training log's 96.8% on another 155-line validation set. Keep those
populations distinct and do not silently substitute values.

Final local pdfLaTeX/BibTeX build succeeds without undefined references, citation
warnings or overfull boxes. Main report Literature through Conclusion is 67
pages, Results 13 and Discussion 7. Total PDF is 120 pages including front matter,
references and appendices. Built-in preview compilation reaches the 20 MiB
preview limit. All eleven new blue/silver plots use embedded Times New Roman
at a printed floor of 10 pt. PDF pages and chart layouts were visually checked.

Deliveries are output/pdf/Handwriting_Robot_Results_Discussion_Revision_2026-11-02.pdf,
output/Handwriting_Robot_Results_Discussion_Source_2026-11-02.zip,
output/Handwriting_Robot_Results_Evidence_2026-11-02.zip and
output/Results_Discussion_Revision_Notes_2026-11-02.md. Source ZIP contains only
98 build inputs plus README and a hash manifest. Evidence ZIP includes the native
geometric banks, original classifier/timing provenance, 70 measurements, new
statistics, plot script and protected-source audit. No training, gantry run,
commit, push or deployment was performed.

## Submission code folder, 2 November 2026

`Software/SubmissionCode_2026-11-02` is the cleaned code for submission, matching
the final design sections. Eight files: paths, segment (3.4), camera (3.4),
recognise and train (3.5), style and synthesise (3.6), reproduce (3.3 and 3.6).
Every algorithm is numpy and the standard library; no Pillow, Torch, Flask or
SciPy anywhere, and no fallback to one.

Two device libraries are kept, at the hardware boundary only, and the user
confirmed this split: gpiod for the gantry pins in reproduce.py, and OpenCV for
the camera in camera.py, where it opens the USB Video Class device and writes a
frame to disk. Neither does arithmetic on an image. The OpenCV file read that used
to sit inside train.py is NOT in that category and was replaced: decoding a file
already on disk is Huffman decoding plus an inverse cosine transform, so train.py
now carries the first-principles PNG and JPEG decoders taken from CleanPipeline2.

Excluded: trainlibrary.py and strokenorm_* (Torch), the Flask server and the
glyph-sheet renderer, curate.py and extractiam.py (Pillow), retrace.py and
timing.py (measuring instruments, not the reproduction path). Removed inside the kept files: the
superseded junction-crossing tracer, SynthesizeLegible, the disabled feature
switches and their dead branches, the manual keyboard/test-circle loop in
reproduce.py, ParseGcode, and unreferenced constants.

Verified against the original over one page, profile and seed on 33 stage
fingerprints, including the md5 of the G-code file: all identical except one
reporting key removed with the setting it described. The decoder swap changes JPEG
pixels by at most one grey level; over 112 photographed lines mean character
accuracy went 98.06 to 98.12 per cent. 8458 lines to 6378 for the same seven files.

CleanPipeline3 was left exactly as it was. Nothing was committed or pushed.

## Figure typography and legibility pass, 2 November 2026

Applies to `Report/SubmissionRevision_2026-11-02`. No text, number, claim, table
or citation changed. Only figures, the scripts that draw them and the
`\includegraphics` widths. Nothing was committed or pushed.

Every generated figure in the report is now set in **Times New Roman**, maths
included, matching `mathptmx` in `epr400.cls`; **every size is 2 pt larger than
before**, with a 10 pt floor on the smallest text rather than Appendix 4's 8 pt;
and all 29 generated figures are included at `width=\linewidth` rather than at
0.78 to 0.99, so LaTeX no longer rescales them and the sizes set in a figure are
the sizes on the page. Three generators were changed:
`Report/DesignVisualTests_2026-11-01/scripts/vt_*.py`,
`Report/DesignResultsDiscussion_2026-11-01/run_visual_bench.py` and
`make_results_figures.py` and `make_design_examples.py` in the same folder, plus
`tmp/redraw_confusion.py` for the two row-normalised confusion matrices.
`tmp/install_figures.py` copies the results in and `tmp/audit_figures.py` checks
the finished PDFs for font and printed size.

The six figures drawn by `run_visual_bench.py` keep their 720 pt canvas, which
LaTeX scales by 455.24/720; their type is set that factor larger so it prints at
the stated size. They were in Helvetica at an effective 6.3 to 6.9 pt before.

Crops of handwriting are contrast-stretched for print, drawn at equal aspect and
cut to windows that fill a panel. Three figures were previously stretching their
handwriting about fourfold because a band height was divided by the band count
twice. Two seam overlays drawn in red were changed to dark blue, which the
project's figure palette requires.

`design-seam-two-panels` and `design-owen-first-features` were mediabox crops of
the full T25 and T32 figures. They are now drawn by the same functions with an
`only` argument, so the short Design version and the long appendix version come
from one code path and one dataset.

**Page budget.** The figures are larger, so Literature through Conclusion is now
**67 pages**, at the top of the stated 65 to 67 maximum, against 63 before. The
full PDF is 139 pages, from 133. The build is clean: no overfull boxes, no
float-too-large warnings, no undefined or multiply defined references. If pages
must be recovered, the levers are the Design-section figure heights, chiefly T36
at 8.21 in and T41 at 6.98 in; letting those float instead of `[H]` was tried and
changed the page count by nothing, so that placement was put back as it was.

## Classifier-free matched-text geometric study, 2 November 2026

Separate analysis only, `Report/GeometricFidelity_2026-11-02/FINDINGS.md`. No report
LaTeX or deployed code changed. Compared the source-text blocks of all five personal
writers with the corresponding original handwritten sentences. Used preserved line
crops, exact case-sensitive transcript matching, pre-score word-boundary review and
fixed geometric processing. Four small source-annotation corrections are recorded.
No recognition model, writer classifier, training or gantry operation was used.

74 word pairs and 65 unambiguously separated character pairs were evaluated. At the
prespecified 0.10 x-height distance tolerance, mean centreline F-scores are Owen
87.09 percent (25 chars in 7 words), Yeukita 83.26 (19), Dylan 73.97 (18), Abhinav
48.78 (3, too small for a writer-wide claim), and Thiya unavailable (0 separable).
All 65 pool to a mean 80.57 percent. Owen's pooled stroke-point F-score is 85.28
percent. Owen strict foreground IoU is 39.59 and fixed-width IoU 56.99 percent.
Do not relabel F-score as IoU, character transcription accuracy, a count of correct
characters or a Pearson style-correlation coefficient. Only 17/25 Owen characters
individually have F>=85; the reported 87.09 is the mean score, not a pass fraction.

Owen wrong-character control averages 52.24 percent. On 17 cases with a same-letter
other-writer reference, true-writer mean is 88.04 and control mean 74.88. Independent
cKDTree checks agree with raster distance-transform scores on all 65 pairs. Word
resampling gives Owen descriptive 95-percent range 82.72 to 91.83. This is bounded
same-source reconstruction evidence on a separable-character subset, not proof all
writers or future texts meet 85. Whole-word scores are much lower and retained in
full, with normalisation/layout limitations documented.

The proposal's actual demonstration wording counts visibly matching characters.
Prepared anonymised full physical-block read-back sheets and an unscored response
CSV in `blind_readback/`. Its separate answer key must be withheld from readers.
`score_blind_readback.py` requires every line and computes edit-based per-writer
accuracy after actual responses exist. No human scores have been invented. Human
style judgement remains separate from read-back. Source references include Chen
et al. (2022), p.5, adaptive IoU, and the official Tanks and Temples F-score definition.

Deliverables `output/Handwriting_Geometric_Comparison_2026-11-02.zip` and
`output/Handwriting_Geometric_Findings_2026-11-02.md`. Figures, all pairs, controls,
annotation records, input hashes and exact as-run script copies are in the analysis
folder/package. This study did not edit the report. The last delivered main report was 63 pages.


## Latest full-report revision, 2 November 2026

The latest full-report task used the user's newly attached
`C:/Users/braun/Downloads/IntegratedReport_2026-11-01.zip`, not an older generated
working folder. The revised main is `Report/SubmissionRevision_2026-11-02/full_report.tex`.
Keep that revision separate from earlier reports. Literature and Approach remain
byte-for-byte unchanged. All main subsection, subsubsection and paragraph headings
are retained. Hardware keeps the supplied wording, restored CAD and torque/speed
sequence, with limited calculation moves and figure explanations.

The main report is 63 pages from Literature through Conclusion. Design is 37 pages.
The full PDF is 133 pages including front matter, the unchanged approved proposal,
references and expanded appendices. There are 48 printed references. The user's
65 to 67 page maximum applies ONLY to Literature through Conclusion. Less is welcome.

Details and full calculations are in Appendix A. Full seam/cost and convolution
feature figures are in Appendix B, with two seam panels and the input plus first
feature row retained in Design. Results has row-normalised writer confusion matrices,
with each row summing to 100 percent and actual line counts shown. No resampling to
invent equal class counts. Appendix C retains all fourteen physical page photographs,
184 paired-line scores, 25 explicitly completed ODROID timings and the supplied
numerical training-log tables. Full raw data and scripts are in the separate evidence ZIP.

The selected joint reader is 94.06 percent; the numerical comparison is 94.39 percent.
The selected writer gives 153/156, the numerical comparison 149/156. Preserve accurate
checkpoint provenance once in the appendix. Never relabel reference-trained results
as first-principles results or host timings as ODROID measurements. The blanket AI
and library declarations were corrected to reflect the documented workflow. Unsigned
fields remain unsigned. No new physical tests or training were performed.

Source folder contains only 97 compile dependencies/build instructions. Unused inputs
were moved into `tmp/final_report_review/unused-source-assets`, not deleted. Build with
`python Report/SubmissionRevision_2026-11-02/build_report.py`. The app compiler timed
out setting up its TeX environment. Local pdfLaTeX and BibTeX compiled successfully.
The separate open `output/Page_Segmentation_Revised_2026-11-02.tex` was not edited.

Deliverables are `output/pdf/Handwriting_Robot_Concise_Revision_2026-11-02.pdf`,
`output/Handwriting_Robot_Concise_Source_2026-11-02.zip`,
`output/Handwriting_Robot_Supporting_Evidence_2026-11-02.zip` and
`output/Report_Revision_Notes_2026-11-02.md`. Read the notes for the unresolved source,
caliper, Vref, ethics and photographic checks. The protected Approach still overstates
encoder-free accuracy, and Literature retains its existing selected-weight wording.
Do not silently rewrite protected sections to conceal those issues.

The user is sleeping and explicitly requested no more questions. Proceed using the
existing record, keep simple wording and concise logical explanations, and leave
unmeasured claims out rather than inventing evidence.


## Latest hardware preference and standalone replacement, 2 November 2026

The user rejected the heavily expanded hardware wording and structure. For hardware,
prefer the newly pasted original draft in attachment 439b6e90-13d1-4e86-8b97-8780e1a2ae55.
Restore its six substantive headings, mechanics voice, CAD, speed rationale, 950 g
load/torque sequence, pen/switch/calibration algorithms and G-code explanation. Keep
new command-space figure in the G-code subsection. Preserve the user's own A4988
wording where technically valid; do not expand into a new generic engineering review.
Latest delivered standalone: output/Hardware_Restored_From_Your_Draft_2026-11-02/3_3_hardware_restored.tex
and matching ZIP. The assembled report is NOT changed; insert only when asked.
The inherited pull-out curve (7 N cm at 60 rpm; 21 N cm at 120 rpm) and friction 0.004
remain conditional, not verified. Correct pulley radius gives 0.0237 N cm steady-load
estimate and conditional ratio about 688. 4000 mm/min is sizing headroom, not actual
selected operating feed. Never restore the false unconditional 560-fold guarantee.


## Current authoritative report: full assembly with visual tests and clean compile folder

As of 1 November 2026, use `Report/IntegratedReport_2026-11-01/full_report.tex`.
It includes unchanged original front matter, approved proposal and blank Conclusion,
plus the revised main report and appendices. The BiLSTM diagram is the user asset;
the pin map is in Appendix A. Reviewed all 41 DesignVisualTests figures and included
12 (8 Design, 4 appendix), with bounded explanations; no new physical qualification
results claimed. Full visual-test assembly: 128 PDF pages.

The report folder now contains ONLY 68 compile dependencies plus README.md and a
build_report.py that compiles full_report.tex into .build/. Do not use the archived
report_preview entry point or regenerate figures as part of compilation. The 536
nonessential files (evidence, checkpoints, scripts, unused assets, old build files,
audits and completion list) were MOVED, not deleted, into the sibling folder
`Report/ReportWorkingArchive_2026-11-01_VisualTests`. Its CLEANUP_MANIFEST.json records
each moved path. VisualTestEvidence/REVIEW.md there records decisions for all 41 images.
No project code, source visual-test folder, datasets or checkpoints outside the
report working folder were changed. Build succeeded after cleanup.

Latest delivery: output/pdf/Handwriting_Robot_Full_Report_Visuals_2026-11-01.pdf
and output/Handwriting_Robot_Clean_Compile_Source_2026-11-01.zip.
Older scope/page-count entries below are historical and superseded by this note.


## Current revision: Literature and Approach integrated, 1 November 2026

The latest working report is `Report/IntegratedReport_2026-11-01/`, built from
the previous Design/Results/Discussion revision and the author's original
Prism Literature/Approach. Sections 1-5 are now included, with references and
appendices. Literature is 6 pages and Approach 2 pages. Main text is 75 pages;
the final whole-report 70-page cap still requires compression and Conclusion
and front matter remain outside this assembly. Do not claim full submission readiness.

The Literature preserves the author's explanatory voice where accurate, removes
Intended contribution, adds topic-level project applications, and retains a brief
Application of the literature summary. Approach follows the proposal's functional
units and the final hardware/segmentation/classification/reproduction arrangement.
Unsupported multiplied-accuracy and all-no-library claims were removed.

User's latest citation preference: only source and page number(s) in text; no
chapter, section, algorithm, equation or PDF qualifiers. Where a page is not
verified, cite the source alone. One bibliography has 44 cited entries. Existing
Ma and Hemanth bibliographic records were corrected, not duplicated. All 42 numbered
equations have a source/derivation basis in EQUATION_SOURCE_AUDIT.md. Own formulas
are not falsely attributed to papers. No code citations were introduced.

All inline editorial comments/blue notes and absent-figure placeholders were
removed from the manuscript and preserved in AUTHOR_COMPLETION_LIST.md. Unknown
measurements remain factual limitations, not invented completions. Both user
hardware/software diagrams remain byte-for-byte unchanged and retain their
numbered explanations underneath. Measurements and model choices remain as
previously recorded. No training, gantry operation, deployment, commit or push.

Delivery: `output/pdf/Handwriting_Robot_Report_Sections_1-5_2026-11-01.pdf`,
`output/Handwriting_Robot_Report_Sections_1-5_Source_2026-11-01.zip`, and
`output/Handwriting_Robot_Author_Completion_List_2026-11-01.md`.



Delivery completed: PDF `output/pdf/Handwriting_Robot_Design_Results_Discussion_2026-11-01.pdf`; source/evidence ZIP `output/Handwriting_Robot_Report_Polished_2026-11-01.zip`. 93 PDF pages: Design 46, Results 15, Discussion 8, then references/appendices. This is a Sections 3-5 review, not the complete submission. Clean build and reference checks passed; revised sections and changed Design pages visually inspected.

## Hardware diagram explanation updated, 1 November 2026

Section 3.2 now places five brief numbered explanations UNDER the unchanged
hardware diagram: 1 External power, 2 External interfaces, 3 Stepper motor
control, 4 Cartesian gantry movement, 5 Pen up/down control. This matches
the author's diagram labels and the software-diagram presentation.

## Software diagram supplied, 1 November 2026

The author supplied the final software diagram; it is included unchanged as
Evidence/user_software_diagram_input.png in the integrated revision. Section 3.2
uses brief explanations tied to its numbered groups: 1 Segmentation,
2 Reproduction, 3 Training, 4 Classification. Do not replace it with the older
three-lane diagram brief. The software-figure placeholder is resolved.

## Latest task update: integrated Results and Discussion, 1 November 2026

The current working revision is `Report/DesignResultsDiscussion_2026-11-01/`.
It incorporates the current Design (including the author's original hardware
diagram unchanged), new Results and Discussion, references and appendices.
Entry point `report_preview.tex`; full current instructions and evidence are in
that directory's README. Older Design-only revisions and v6 remain preserved.

The user selected the **library-trained joint reader** for the demo, with the
**library-trained eleven-writer pw170 classifier at width 0.17**. Compare the
NumPy-trained models separately. Joint reader pooled character accuracy is
94.06%; evaluated numerical reader is 94.39%. Selected writer 153/156=98.08%;
numerical writer 149/156=95.51%. Newly supplied training excerpts show a NumPy
writer best displayed val accuracy .968 by epoch41 on 155 lines and numerical
reader epoch durations about111--122min. Do not attribute model-score differences
to float32 versus float64 without a controlled experiment; do not infer completed
epochs from an interrupted excerpt.

A new offline physical-style analysis uses all184 reproduced lines and156
original validation lines across all 11 writers, no exclusions. It yields .6967
seven-shape-feature correlation and .7127 for allnine, with .9911 slant. Full
inputs, code hashes and rows are in ResultsEvidence/physical_style/results.json.
The older .714 value is a different ten-writer **software-render** study and
seven shape features, not nine or final physical pages. Keep them distinct.

R3 primary reporting uses25 positive-character console records with explicit
program-end confirmation, all within4s/character; worst3.875. Claude's earlier36
uses a speed-based filter which retains at least one explicitly stopped job.
Original archive is preserved. User authorises excluding stopped jobs. Timing
completion and quality/clamping are separate questions. R4 times46.26/55.49s are
ODROID; 2.76x batched-framework comparison is development-host evidence, not a
measured target speed-up. Do not claim all offline measurements were on ODROID.

User confirms allseven geometry values from the final calibrated gantry, with a
metal non-digital vernier caliper; make/model/least count to follow. More readings
and repeated-word overlay are planned; leave placeholders, never predict values.
Wider lighting operation is author-reported, with numeric lux records pending.

User confirms signed consent from allfive contributors. Retain signatures
privately, optionally a redacted template in appendix; institutional ethics
clearance/exemption is still requested separately. Current build estimate about
R4000, ODROID borrowed at zero cost and returned. Later listed clipboard90,
paper130, gel pen30, estimated training electricity500 sumR750; inclusion in
approximate total unconfirmed. Ledger image Downloads/BUDGET.jpg is preserved.

Four verified checkpoints were copied to uniquely named **evidence** files under
ResultsEvidence/checkpoints to make the report reproducible. No existing runtime
checkpoint path, pipeline code, training, hardware motion, deployment, commit or
push was changed/performed. Current glyph libraries remain final as requested;
do not discuss manual curation as a feature. The report review covers Sections3--5;
final assembly of the whole report requires a separate overall70-page budget pass.

Recorded from the user's request and initial source review on 31 October 2026.
This is persistent repository context, not a claim of automatic Prism access or
account-wide memory. Update it when the user supplies a new draft or resolves facts.

## Current task and editorial preferences

- The handwriting robot project is effectively complete according to the user.
  The immediate task is to produce an excellent final report, especially Design
  and implementation. Begin by understanding the project, requirements and drafts.
- The project was developed with Claude and other agents. Do not turn older
  assertions of unaided authorship into report facts. Attribution must reflect
  what actually happened; drafting the final declarations needs factual input.
- Preserve the user's own wording and the structure of the supplied Prism draft.
  Improve grammar, technical correctness, explanation, evidence and concision.
  The user identified their own wording as the Literature review before Intended
  contributions, the Plotter designs and mechanics description, and the first two
  paragraphs of the stepper motor subsection. Preserve those passages closely and
  use them as the voice reference for new text. Larger changes to other AI-written
  material are acceptable while retaining the current draft's structure.
- Latest scope clarification: focus on Design now; Results and the rest of the
  report will be addressed later.
- Subsequent outline request: v6 is optional supporting material and may be
  substantially departed from. Organise software around the three main code stages,
  without requiring a generic Software design heading. Supplementary variable
  descriptions and code analysis may go in the appendix. Keep essential design
  calculations, rationale and representative verification in the main text.
- Present the final design and meaningful engineering trade-offs, not a diary of
  incidental implementation bugs (the user specifically rejects the frozen-backbone
  anecdote as a narrative focus). Explain material limitations, their causes,
  remedies and actual reasons remedies were not implemented. Do not invent those
  reasons or disguise unmet requirements. Highlight strengths with evidence.
- The user accepted the overall outline in `Report/PROPOSED_DESIGN_OUTLINE.md`
  and requested one further flow review. The refinement separates shared training
  procedure, first-principles/framework implementations, and verification in 3.5.
- Latest user clarification: an ink-based writer classifier initially worked well;
  a stroke-normalised classifier was then needed to assess gantry output fairly
  and reduce reliance on ink cues. The user says the stroke-normalised classifier
  is now used everywhere. This meaningful design decision belongs in 3.5.4, with
  a short cross-reference when explaining synthesis candidate scoring in 3.6.6.
  This is user-reported current state, not confirmation that the older local server
  preprocessing defaults have been brought into alignment.
- User says first-principles and library classifiers have both been trained and
  run similarly, and can obtain further results from Claude when needed. Explain
  architecture/recipe once, implementation routes separately, and numerical and
  trained-model validation as distinct evidence.
- Avoid a report that reads like a function catalogue or general textbook. Explain
  the design decisions, relevant mathematics, implementation and verification.
- Remove repetition and bloat without discarding engineering that earns credit.
- For this drafting stage the user permits roughly 1–2 pages over a section's
  upper guideline when needed. Prefer the guideline when achievable comfortably.
  Do not let older notes demanding an immediate hard compression override this.
  The final submission's overall 70-page cap remains a separate eventual constraint.
- The user will manually supply Prism exports. There is no live sync with Prism.
- Current software is `Software/CleanPipeline3`. The user pushes to GitHub and
  runs/tests the server on an ODROID N2+ controlling the gantry.
- No request to change the pipeline, train models, operate hardware, commit or
  push was made in this onboarding task.

## Draft authority and locations

1. **Current manually edited draft:**
   `C:/Users/braun/Downloads/ERPReport2026u23539195.zip`.
   An unchanged extracted copy is in `Report/PrismSnapshot_2026-10-31/`.
   Treat that directory as the baseline snapshot; make future revisions separately.
2. **Supporting AI-assisted draft:** `Report/FinalReport_v6/`.
   It contains much more technical prose, figures and newer results, but is not
   automatically the master. Older v2–v5 directories are historical.
3. **Assessment authority:** approved Project Proposal and Appendix 4. Check source
   PDFs rather than treating old condensed notes as exact quotations.
4. **Implementation authority:** current CleanPipeline3 code. A report, explanation,
   README or handover does not establish that a feature is actually deployed.
5. **Measurement authority:** original outputs, scans, logs, equipment and exact
   checkpoint/configuration that produced the result. Preserve their provenance.

The Prism snapshot has developed Literature and Approach text and substantial
Hardware material. Most subsequent Design subsections are still planning notes.
Results contains the test skeleton and empty summary; Discussion and Conclusion
are mostly empty headings. Do not confuse these with v6's populated later sections.

## Sources found and reviewed

### Official assessment and proposal

- `C:/Users/braun/Downloads/APPENDIX4_2026.pdf` — 32 pages, revised 4 September 2026.
- `C:/Users/braun/Downloads/FinalReportTemplate_NOTES.pdf` — 18 pages; annotated
  template with required headings, table formats and explanations.
- `C:/Users/braun/Downloads/FinalReportTemplate_2026.doc` — located, not parsed in
  the initial review; use the annotated PDF and original class for present work.
- `ProjectProposal2026DylanFinalRev0 - Dylan Braune-1.pdf` — root proposal, 8 pages.
- `Report/PrismSnapshot_2026-10-31/2_Project_Definition/project_proposal.pdf` —
  8-page proposal with extractable requirements and visible reviewer annotations.
  It differs as a file from the root proposal. Confirm the approved AMS version
  before final assembly; do not modify the specifications.
- `Report/REPORT_REQUIREMENTS.md` — useful condensed reference, but contains
  interpretations and overstatements (e.g. presenting some example wording as exact).

### Project state and writing preferences

- `PROJECT_HANDOVER_CONTEXT.md` — broad state and 31 October results, including
  models, datasets, tests, timings, known defects and open deployment choices.
- `handover.md` — older report comparison and torque discussion. Some conclusions
  are technically unsound or outdated. Do not accept its praise or claims uncritically.
- `PROJECT_CONTEXT.md` — older project scope summary; some working rules conflict
  with later user practice and handovers. The current user request takes priority.
- `Report/ProjectHistoryDigest.md` — useful historical development narrative; its
  module paths and results predate CleanPipeline3.
- `Report/PersonalSummaryOfWhatIWantInMyReport.docx` — user's content wishlist,
  read through extracted text, including meaningful explanatory figures and tests.
- `Report/ReportContentPlan.md` and `Report/_dev/REWRITE_RULES.md` — historical
  plans/agent briefs, not new authorization or a substitute for this request.
- `Report/FinalReport_v6/REFERENCE_AUDIT.md` — tracks citation corrections and which
  source locators were actually checked. Do not invent locators or assume the old
  audit validates a newly written claim.
- `C:/Users/braun/Downloads/Final_Report_Grading_Rubric.docx` — inspected, but it
  discusses MARV subsystems and appears to concern a different assignment. Do not
  apply its marks or structure as this project's rubric without confirmation.

### Reference student report

- Likely intended reference: `C:/Users/braun/Downloads/23531003_Report_v2.pdf`,
  A. Sreekumar, analogue/hybrid neural network for digit recognition, 111 pages.
- User explicitly confirmed Sreekumar's u23531003 report is the intended reference.
- Relevant design passages and page layouts were reviewed directly, not solely
  through prior agent comparisons. Useful practices: worked numerical examples,
  quantified alternatives, model-to-measurement comparisons, and evidence explaining
  why a design changed. Borrow practices, not wording or unsupported conclusions.
- `Report/FinalReport_v6/COMPARISON_23531003.md` gives a detailed historical critique.
  Its counts, page lengths and claims need refreshing for the current Prism draft.

### Code explanations and live code

- `Report/CodeExplained_INDEX.md` maps old module names to CleanPipeline3.
- `Report/CodeExplained_A_Segmentation.md`, `_B_TrainingClassification.md`,
  `_C_ReproductionGcode.md`, `_D_GantryHardware.md` explain the four major blocks.
  Initial review covered architecture and selected detailed sections; these are
  reference material to reread at function level during revision, not fully audited.
- The explainers originated against CleanPipeline v1. Segmentation has drifted most;
  the index explicitly lists new functions it does not explain. Even an assertion
  that the mathematics is unchanged needs checking when a claim matters.
- `Software/CleanPipeline2/testing/results/CodeTheory/main.pdf` — **335 pages**,
  detailed per-function theory manual. Its source chapter `.tex` files are beside it.
  Initial review indexed its contents and read selected segmentation and alignment
  sections; it has NOT been exhaustively read or verified cover to cover.
- `Report/FullPipelineRun/` contains verification DOCX files and inventories for
  follow-up checks. They were located, not exhaustively reviewed initially.
- `Software/CleanPipeline3/README.md` is useful but already contradicts current
  server model selection; prefer code for current behavior.

## System understanding

The read path conditions a photographed page, separates text lines, recognises text
with a CNN–BiLSTM–CTC reader, and identifies the writer with a CNN. The offline path
uses known transcripts and CTC forced alignment to locate glyphs, traces skeletons
into pen paths, filters variants and records writer-specific style and glyph data.
The write path assembles new text from those glyphs, applies layout/style/joins,
selects candidates using reader and writer scores, emits G-code, and directly drives
the gantry through the ODROID's GPIO using A4988 STEP/DIR inputs.

Current modules: `segment.py`, `recognise.py`, `train.py`, `style.py`,
`synthesise.py`, `reproduce.py`, `paths.py`, `timing.py`, and `web/server.py`.
`trainlibrary.py` is a separate Torch training route. NumPy inference does not prove
that a deployed checkpoint was trained without a framework.

Reader and writer models use a 12-convolution backbone (2/4/6 layers, 32/64/128
channels). Reader preserves the width axis, uses two bidirectional LSTM layers and
CTC; writer classification collapses width to a feature vector and classifies it.
Attribute the published architecture correctly; do not claim invented algorithms.

Mechanics: Cartesian GT2-belt gantry, one motor per axis, cross-shaft drives both
Y-side belts, linear bearings, and a one-direction geared cam motor for pen lift
with a microswitch. Code assumes 200 full steps/revolution and 1/16 microstepping.
The user has now confirmed a 1.8-degree motor, consistent with 200 full steps.

### Hardware facts supplied by the user after initial review

- Motor: **42BYGH40-01**, NEMA 17, 40 mm body length, **1.8-degree step angle**,
  **1.68 A per phase**, described by the user as a 12–24 V unit.
- The user operates the motor-driver supply at **24 V**.
- The user reports **0.35 A system supply draw**, with the supply's current limit
  set to **2.11 A** while running. Operating state during the 0.35 A reading has
  not been specified. Do not call this a peak or worst-case current.
- Keep supply draw, supply current-limit setting, rated motor phase current and
  A4988 regulated phase-current setting distinct. These are not interchangeable.
  The supply current limit does not establish the A4988's phase-current setting.
- These are user-supplied facts, not newly measured or datasheet-verified values.

### Hardware diagram and pin-map update, 1 November 2026

- The user supplied a hand-drawn hardware overview and asked for it to appear in
  Design with a separate pin/voltage interconnect table. After an initial redraw,
  the user explicitly requested their original diagram; the report now includes
  that image unchanged from `Report/DesignPolished_2026-11-01/Evidence/`.
- The pin table distinguishes ODROID physical header pins from `/dev/gpiochip0`
  line offsets read from `Software/CleanPipeline3/reproduce.py`. It separates 3.3 V
  STEP/DIR logic from 24 V VMOT and the driver's chopped winding outputs. The
  24 V supply current limit is not identified as the motor phase-current setting.
- User-reported details: 8 MP USB autofocus camera with IMX179 sensor, 78-degree
  field of view and stated 3264x2488 resolution; actual ODROID adapter output is
  unknown; the ODROID's dedicated 3.3 V pin powers the pen motor; GPIO controls
  the BC337 stage; 10 kOhm pull-downs are reported; no flyback diode; A4988
  sense-resistor marking is tentatively R10 and Vref is unmeasured. Keep these
  uncertainties explicit until installed wiring and component markings are checked.

## Requirements to trace through the report

| Requirement | Proposal target |
|---|---|
| R1 | Reproduce arbitrary digital text in a trained style; 85% character accuracy and 85% stylistic correlation |
| R2 | Decode text and identify writer; 95% overall accuracy |
| R3 | Reproduction within 4 seconds per character |
| R4 | Classification within 2 + 0.025 times character count, seconds |
| R5 | Absolute positional error at most 0.5 mm |
| R6 | Train/store at least 10 writer styles permanently |

Field conditions: 400–1000 lux and stable, level, vibration-free support. The
proposal's repeated-word demonstration is repeatability evidence; it does not alone
establish absolute accuracy (the reviewer explicitly flagged this).

Appendix 4 prioritises the technical Design section and a connected engineering
argument. Mandatory Design summary and System block diagram, and Results summary
and integrated qualification tests, must map to the proposal. Main-report guideline
lengths are Lit 4–8, Approach 1–2, Design 30–40, Results 10–15, Discussion 6–8;
Conclusion 1–4. Sections' guidelines are not all independent hard maxima. Overall
main-report cap is 70, excluding preamble/proposal, references and appendix.

Part 3 uses no first person. Include relevant equations with defined symbols and
text references, complete circuits with component values/calculations, readable
figures/tables referenced in prose, and short purposeful code excerpts. Distinguish
subsystem verification in Design from end-to-end qualification in Results.

## Verified conflicts and items to check before reuse

1. Prism draft says both 400 and 200 full steps/revolution. The user resolved this:
   42BYGH40-01, 1.8 degrees, hence **200 full steps/revolution**, at a 24 V driver
   supply. Correct the 400-step claim when revising the working draft.
2. The displayed RPM equation multiplies by 60 although the substituted feed is
   already mm/min. The numeric 100 rpm is consistent with 4000/40 without that
   factor. Correct units and expression together.
3. Prism torque calculation uses a measured shaft/bearing radius for belt torque;
   use the belt pitch radius when converting belt force into pulley torque.
   Its printed numerical product is inconsistent too. Neither the old 560-fold
   nor v6's 680-fold friction-only margin proves safe abrupt starting or sufficient
   acceleration torque. Check inertia, drive conditions and actual friction.
4. v6's full-torque treatment also appears to include rotary inertia explicitly
   and inside its effective mass, risking double counting. Do not transfer blindly.
5. Distinguish driver current chopping from the motor stepping sequence. Describe
   open-loop step counts as commanded position, not guaranteed physical position.
6. Local server currently sets `PAGE_NAMER = SHAPE_MODEL` and
   `PAGE_NAMER_STROKE_NORM = False`. Check training preprocessing and exact installed
   checkpoint before describing the deployed writer classifier.
7. `paths.py` lists 13 writers, whereas the handover describes an 11-writer test
   configuration excluding 151 and 551. Do not conflate stored profiles, classifier
   classes, evaluated writers and deployed writers.
   The user clarified that **11 tested writers** are the Results population and
   intended demonstration population. The server has not properly been tested on
   13, but the user explicitly requests that this website-testing status not be
   inserted into the report. Respect that scope; report the tested 11 without
   asserting validation of 13. Do not change the code as part of this clarification.
8. Current reader candidate order prefers joint weights before scratch weights.
   Older checkpoint quality claims conflict. Ask what is actually installed.
9. Draft says no endpoint interrupts running jobs; current server has
   `/api/gantry/stop`, and `BresenhamMove` checks `abortEvent`. This draft statement
   is stale. Review current stop semantics rather than copying old safety claims.
10. Nominal calibration travel values in draft (204/262 mm) differ from local code
    (200.6/261.3 mm). Do not present nominal/default values as current measurements.
11. Step loop sleeps for each axis pulse separately; diagonal execution timing
    differs from a simple distance/feed prediction. A handover attributes an overhead
    factor to acceleration although the driver has no acceleration ramp. That causal
    explanation needs evidence; Python/GPIO timing and axis sequencing also matter.
12. A debounce calculation does not itself measure stopping distance or latency.
    Inspect whether motion pauses while switch confirmation sleeps occur.
13. Explainer B incorrectly treats 2L+1 extended CTC states as a universal minimum
    number of timesteps. Blank states may be skipped between different labels.
    Verify the correct feasibility condition in primary CTC sources before writing.
14. Never equate writer posterior confidence, classification accuracy, and stylistic
    correlation without a justified operational definition. The selection networks
    also judge generated candidates, so that evaluation is not independent.
15. Handover has 14 scanned reproduced pages/184 lines for 11 writers; its evaluation
    checkpoints differ from local server defaults. Those numbers are reported history,
    not newly reproduced results. Separate laptop evaluations and ODROID timings.
16. Prism static reference audit found nine missing targets:
    `alg:gan-step`, `eq:gan-margin-eq`, `eq:gan-radius`, `fig:des-hw`, `fig:des-sw`,
    `fig:gan-repeat`, `sec:design`, `sec:gan-circuits`, `tab:des-fumap`.
    `new` and `mustdo` are used without definitions found in snapshot `.tex`/class.
    This was a static check, not a compilation test.

## Clarifications received after the initial review

- Reference report: Sreekumar's u23531003, confirmed.
- Voice reference: early Literature review before Intended contributions, Plotter
  mechanics, and the first two stepper paragraphs, confirmed.
- Evaluation and intended demonstration population: 11 tested writers, confirmed.
  The user says the results remain unchanged; defer Results work for now.
- Motor model, step angle, supply voltage and supply current readings: recorded above.
- Exact deployed checkpoint provenance/preprocessing was not resolved by the
  writer-count clarification. It may be checked when a specific Design claim
  depends on it; do not repeatedly ask broad deployment questions now.

Later, before final factual claims, resolve training provenance and computer used,
the actual approved proposal copy, driver sense resistance/Vref/current, supplies,
installed protection components, and measured hardware verification. Ask in focused
batches when working on those subsections; do not dump the entire hardware checklist
on the user at once.

## Work completed in the initial review

Extracted the supplied ZIP into a separate snapshot; extracted reference texts and
indexed the 335-page manual in `tmp/report_context_review/`; reviewed assessment
content, current draft, handovers and selected code explanations/live functions;
visually inspected representative template, assessment and reference-report pages;
recorded preferences and conflicts here. No report prose or pipeline code was
changed. No models or hardware were run. Existing uncommitted edits were preserved.

Next work: revise Design within the current draft's structure, section by section,
using v6 and the explanations as supporting material. Start with the hardware prose
already written by the user, then build the undeveloped software subsections around
actual methods, purposeful worked examples, and existing verification evidence.

## Classifier comparison evidence located during outline refinement

`Software/CleanPipeline2/testing/results/trainvs/` contains `equivalence.json`,
`compare_both.json`, and `compare_both.csv`. Inspected their metadata and selected
entries; have not yet traced the producing script or verified applicability to the
final CleanPipeline3 checkpoints. Do not request all comparison results as though
none exist, or present these older tests as final-model measurements.

- `equivalence.json` records float64 forward/loss/gradient comparisons; numerical
  metric definitions and treatment of near-zero gradients need checking.
- `compare_both.json` records a float32 text-training comparison on 500 training
  lines, 40 validation and 40 test lines, 15 epochs, batch 8, seed 7. This is a
  small controlled experiment, not a full final-model benchmark.
- Needed before final comparison claims: matched held-out text and writer results
  for the relevant final checkpoints, exact split/preprocessing/metric definitions,
  training/inference environment and separately measured timings if speed is claimed.
- For the ink-to-shape decision: ideally compare both writer models on identical
  real-handwriting and reproduced-writing samples, with uncertainty/sample counts,
  and state whether the reproduced samples were independent of candidate selection.

## Design drafting on 1 November 2026

The user authorised continuing the Design revision. Ask for diagrams and example
images when needed, continue independent writing while waiting, and pause only if
the missing material actually blocks progress. Subsystem illustrations and design
verification belong here; full-system results and QTPs stay in Results.

A first complete Design draft is in `Report/DesignRevision_2026-11-01/`, with
seven subsection files under `3_Main_Report/design/`, a compilable Design-only
`design_preview.tex`, and `design_support.tex` for import into Prism. The snapshot,
v6, other report sections and pipeline code were not modified by this revision.
The supplied class file was copied without modification (SHA256 checked).

The draft preserves the user's mechanics and opening stepper explanations while
correcting technical errors. It includes explicit numerical operations and design
choices, the ink-to-shape rationale, distinct training implementations, archived
numerical verification, and subsystem examples. Missing facts are blue completion
notes; unavailable figures have labelled slots, not fabricated images/results.

`FIGURES_AND_EVIDENCE.md` in the revision directory records every missing asset,
exact filename, required contents and pending factual inputs. **User confirmed the
important hardware system overview will be drawn later with proper guidance.**
Do not forget it or ask the user to produce it immediately. A detailed layout brief
has been saved in that register. Other pending questions concern the full circuit,
camera/gantry photographs, matched lighting/rotation samples, processing overview
and a worked glyph-alignment/join example.

One new computational illustration uses existing Owen/Yeukita crop segments and
live segmentation primitives at width fraction 0.14 to show original, binary,
skeleton and fixed-width images. No model weights were loaded and no training or
hardware was run. This illustration does not establish the final deployed width.

The numerical comparison producer `CleanPipeline2/testing/torchtrain.py::DoCheck`
was inspected, and archived forward/loss/gradient numbers were included with the
test scope and near-zero-gradient exclusion explained. It is still an archived
implementation check, not newly measured final-model performance.

Build uses direct `pdflatex` and `bibtex`, because local latexmk needs unavailable
Perl. Final delivery includes a PDF reading copy, source ZIP and import instructions.
The draft is not submission-ready until the pending figures, measurements and
configuration facts are resolved. Do not treat a successful build as evidence that
all technical claims or sources have been independently audited.

## Hardware and figure clarifications received after first Design delivery

The user now wants a consolidated figure checklist for later, with placeholders
retained in the report. The updated `DesignRevision_2026-11-01/FIGURES_AND_EVIDENCE.md`
contains four drawings, two photographs and seven worked illustration items
(one stroke-normalisation illustration already exists). Offer proper guidance
when the user returns to drawing the important hardware overview.

User-supplied facts, not independently measured in this session:

- Camera: 8 MP USB autofocus webcam, 78 degree field of view, CMOS IMX179,
  supplied resolution 3264 x 2488. Actual software capture resolution is unknown.
- ODROID uses its supplied wall adapter; exact DC output rating is unknown.
  Do not infer it from the two-prong mains cable.
- All 3.3 V supply connections use the ODROID dedicated 3.3 V power pin.
  GPIO supplies control signals, not motor power. Pen motor uses that 3.3 V rail
  and a GPIO-controlled transistor; steppers use the external supply via drivers.
- User tentatively recalls sense-resistor marking R10 (0.1 ohm). Keep tentative;
  actual Vref was not supplied and cannot be established from Rsense alone.
- No flyback diode is fitted. 10 kOhm pull-down resistors are fitted, but their
  precise locations and specifically a transistor base pull-down remain unknown.

The user offers `Software/CleanPipeline3/ReportResults/FullReproducedPages/` and
recommends Owen, IAM 150, Yeukita and Dylan as clear, distinct examples. Use actual
reproductions where appropriate. For alignment and glyph/profile extraction,
use the original handwritten input and transcript rather than pretending that
re-extraction from robot output represents the original construction procedure.

These facts/checklist were saved after delivery; the previously compiled Design
PDF and source ZIP still need the factual updates in the next report revision.

## Second Design audit and user provenance clarification, 1 November

The user requested a full coverage/explanation pass against Appendix 4, Sreekumar,
Aston's `C:/Users/braun/Downloads/aston_not_finished.pdf`, and the actual code,
assuming all planned pictures will be supplied. Use real subsystem examples and
algorithm blocks to improve readability without losing mathematics or engineering
depth. Do not inflate the report with a catalogue of every helper function.

The revision folder now contains eight algorithm blocks, a real saved Dylan glyph
inspection example, an importable implementation appendix and a detailed
`DESIGN_COVERAGE_AUDIT.md`. Source paths and algorithmic operations were inspected;
this is not a proof of every line of code or a full audit of every inherited source.

New authoritative user-supplied records:

- Final reader: `paper_cnn_bilstm_ctc_joint_best.npz`, trained on joint Teklia IAM
  lines plus five photographed writers. PyTorch-trained; pure NumPy inference.
- Historical shipped shape weights: `author_shape_10new_weights.npz`, 13 classes,
  validation 0.9708, epoch and training width not recorded. Do not infer width from
  the module default 0.14 or compare its score directly with the 11-class sweep.
- Selected 11-writer model: `author_shape_11_pw170_weights.npz`, recorded width
  0.17, best saved epoch 43, validation 0.9935; PyTorch on Google Colab T4.
- Width sweep: 0.06 → epoch 38 / 0.9806; 0.09 → 43 / 0.9742;
  0.11 → 39 / 0.9806; 0.17 → 43 / 0.9935. Each had a 50-epoch budget.
- 11-writer dataset: 517 IAM + 669 photographed = 1186 lines; 1031 train,
  155 holdout. 13-writer historical dataset: 689 IAM, 1358 total; 1187/171.
- Public writer holdout: last labelled page per writer. Personal split: seeded
  15% of lines, seed 0. Reader counts and historical training computer/epochs/run
  settings remain unknown. CPU inference from a log is not a verified host record.
- Supplied production checkpoints were framework-trained. Do not restore any
  no-machine-learning-library claim for training them. The NumPy trainer exists
  and has archived numerical verification; its existence is distinct from which
  implementation produced the supplied weights. No retraining is authorised.
- User confirms the cleaned (artifact filenames `*_curated`) profiles ARE the
  final profiles. Report them simply as final profiles. Explain inspection as a
  preparation step without presenting original/curated demo versions or suggesting
  mixed profile populations. File provenance may retain actual filenames privately.

Code review found an additional substantive distinction: `train.py::CtcLoss`
averages sequence losses over batch, while `trainlibrary.py` uses default target-
length-normalised CTCLoss mean reduction. The archived verification script matches
reduction explicitly. State the comparison scope rather than claiming identical
training settings. Official PyTorch CTCLoss documentation was checked and cited.

The default skeleton tracer first splits at graph junctions but then prunes spurs
and greedily rejoins compatible ends; do not describe it as one lift per junction.
The retrace route bypasses models/synthesis but retains image/tracing/conversion
errors; it is not a pure independent mechanics measurement against original ink.

The supplied final weight provenance and cleaned profile interpretation supersede
older pending notes above. Hardware facts are incorporated into the second PDF.

### Latest steering during the second pass

- The user will remove manual curation features. Take the currently saved glyph
  libraries as fixed final demo inputs, without describing manual curation as an
  operating feature or presenting original/curated demo options. A real final-
  profile figure may show existing variants and representatives. Do not change
  pipeline code to remove features; the user said they will handle that.
- The user explicitly confirms CURRENT normalisation thickness is 0.17, chosen
  from the sweep. This resolves the normalisation-width question for the described
  final configuration; older default/shipped metadata remain historical.
- Sourcing is especially important: imitate Sreekumar's in-text source locators
  beside the relevant formulas/facts. Use verified printed pages, explicitly
  labelled PDF pages for unnumbered reading copies, and exact headings for truly
  unpaginated online sources. Never invent a page.
- User explicitly authorises flagging sources that cannot be inspected for their
  own manual verification. The report now has exact-site blue source/page flags
  and `SOURCE_AUDIT.md`, not a claim that those sources were read.
- Prefer the user's already-read Literature Review sources where they actually
  support the claim, reuse existing bibliography keys and avoid gratuitous growth
  beyond roughly 50–60 sources. The current baseline/revised main-report union is
  40 cited keys; only two new external keys were introduced in this pass, plus one
  internal source-code record. Final Results/Discussion may change that count.

### Reviewed second-pass delivery

Completed and packaged the second Design pass on 1 November 2026:
`output/pdf/Handwriting_Robot_Design_Reviewed_2026-11-01.pdf` and
`output/Handwriting_Robot_Design_Reviewed_Prism_Source_2026-11-01.zip`.
The preview has 39 Design pages, two reference pages and four supporting appendix
pages. All 45 rendered pages were inspected; the build has no undefined references
or overfull warnings. The source archive contains 44 files and passes its CRC check.
Hashes of the 17 inspected processing modules match the coverage inventory.
Outstanding: 12 requested figure assets, blue completion notes for missing facts
or evidence, and eight referenced sources requiring the user's manual page check.
The detailed audit and drawing briefs are in the revision folder. This is a review
draft, not a claim that pending evidence or full-report integration is complete.

## Visual evidence and pseudocode pass, 1 November 2026

The user explicitly authorised offline use of their personal/IAM data, stored
profiles, camera roll, reproduced pages, G-code generators and new test benches
to explain and verify subsystems. Extended evidence belongs in the appendix;
representative examples belong in Design. Formal system qualification remains in
Results. This expands the earlier read-only evidence scope to actual offline
processing/inference, not training, deployment or physical motion. The user then
emphasised: use real rotation/lighting photographs if available; never force data
or results. Leave unsupported items pending for later capture.

New folder: `Report/DesignVisualRevision_2026-11-01/`; previous reviewed delivery
is preserved. New preview: 42 Design pages + two references + nine appendix pages.
Eight existing algorithms were converted to line-numbered, indented pseudocode
with explicit assignments, conditions, loops and returns, using Aston's inspected
format as presentation inspiration. Three independent verification procedures were
added to the appendix (11 algorithm blocks total).

Filled six figure slots: processing overview; network architecture; native Owen01
segmentation; actual camera-roll comparison; original Owen crop alignment; three
pre-repair Owen candidate scores. Added four-profile software/G-code path renders
and an existing physical Owen robot-page photograph with newly recovered crops.

Newly run evidence, with exact inputs/settings/hashes in `Evidence/bench.json`
and `Evidence/camera_bench.json`:
- Native Owen01: 20 crops for 20 labelled rows; count is not pixel accuracy.
- Twelve extra reduced/synthetic page cases retain failures. Reduced Owen01 lost
  its first usable crop; the alignment example uses a separately identified,
  visually matched original archived crop and its known transcript.
- Real camera-roll photographs `WIN_20261028_23_23_30_Pro.jpg`,
  `WIN_20261028_23_28_34_Pro.jpg`, `WIN_20261028_23_32_40_Pro.jpg`: same sparse
  three-row page, different shadows/presentation. Native pixel arrays processed
  without manually fixing failed orientation. Counts 4, 3 and 28. The third
  missed its quarter turn; gap scores 0.379/0.400. No lux/physical angle measured.
- 100 eight-connected partitions agree with independent BFS; 20 tiny seam
  optima agree with exhaustive enumeration; repeated-label CTC probability from
  1024 paths agrees to 4.44e-16, all 20 logit derivatives to 3.63e-11.
- Nine serialization fixtures preserve vertices without clipping, max coordinate
  error 0.0004 mm. Joins/closure disabled only for those paired fixtures.
- Candidate illustration explicitly loads joint reader + eleven-writer pw170
  checkpoint from Downloads and uses 0.17 preprocessing. Three fixed seeds,
  before repair, not full server search or independent physical qualification.
- A trial instrumented for a joining figure produced no joins; no passing example
  was manufactured. Joining/filtering figures remain pending, along with four
  hardware illustrations and missing measurements.

Corrected previously misreported current ruling extent default from 1.60 to 1.25.
All inspected pipeline hashes remain unchanged. No training, checkpoint/profile
mutation, hardware execution, commit or push occurred. Eight external source/page
verification flags remain; no new external source was needed for these examples.


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

Latest user instruction places strong priority on visible in-text `\cite`
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

Final preview recorded 43 Design pages, three reference
pages and 13 supporting appendix pages
(59 total), with the original class unchanged. Eleven algorithms
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

Rebuilt delivery after citation correction: 42 Design pages + 3 references + 13 appendix pages = 58 total. All rendered pages checked; no undefined references or overfull warnings. Stable output PDF and source ZIP updated. Prior delivery retained in tmp/report_context_review/before_research_citation_revision.


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

Real-calculation delivery checked: 43 Design + 3 references + 14 appendix pages = 60. All rendered pages reviewed; no overfull boxes, undefined references or missing citation keys. Existing six figure placeholders and eight completion notes remain explicit. Generic scalar examples were replaced or removed, with full-precision real traces supplied. The traced single-crop first-convolution patch storage is 1.40625 MiB, not a measured peak.

System diagram plan: Appendix 4 p. 15 permits separate hardware and software block diagrams or a combined view and requires an accompanying systems-level explanation of intent and operation. Use separate views for legibility. Figure placeholders are `design-hardware-overview.pdf` and `design-software-block-diagram.pdf` in Design §3.2. Hardware brief separates USB, GPIO/switch signals, power and mechanical coupling; labels Y cross-shaft accurately; ODROID adapter voltage and common-ground wiring remain to be verified; reported 2.11 A supply setting must not be mistaken for measured A4988 phase current; pen motor has no flyback diode. Software brief has three lanes (offline preparation, ODROID page reading, ODROID writing), stored weights/cache/profile/job objects, proposal FU labels, best-of-N score feedback, and G-code file boundary. Read and write are separate requests. General figure text must be ≥8 pt (Appendix 4 p. 28); circuit component labels ≥6 pt (p. 15). A network-operated ODROID must not be described as lacking an HDMI connector.


## Extended five-writer reference-bank geometry, 2 November 2026

Separate offline analysis in `Report/GeometricFidelity_Extended_2026-11-02/`.
No report LaTeX, application code, checkpoint or hardware changed. Final package
`output/Handwriting_Five_Writer_Geometry_2026-11-02.zip`, explanation
`output/Handwriting_Five_Writer_Geometry_Findings_2026-11-02.md`.

193 visually checked separable physical source-block characters, nearest original
variant from fixed page-diverse/source-ordered banks of up to eight same-writer,
same-letter examples. Bank membership and quality exclusions set before scores.
Same 48-pixel x-height, isotropic scale, +/-5-pixel translation, primary centreline
F at 0.10 x-height as earlier study. No tolerance tuning. 708 accepted reference
examples. One additional physical Yeukita q remains unassessed without a reference.

Dylan 36 chars 85.14 percent, Yeukita 42 chars 95.45, Owen 64 chars 94.53,
Abhinav 14 chars 75.10, IAM 150 37 chars 96.24. Equal-writer mean 89.29 percent,
character-weighted mean 91.90. This is optimistic nearest-reference geometric fit,
NOT IoU, transcription accuracy, a count of correct characters, Pearson style
correlation or proof of the proposal's 85-percent specification. All low scores
retained. Abhinav's three c examples average 86.77 but his full mean remains 75.10.

Original 65 physical masks verified byte-for-byte unchanged, scoring with the bank
raises their pooled mean from 80.57 to 91.11. This isolates the bank-search effect.
Independent cKDTree checks reproduce all 193 winning F scores exactly. Conditional
word-group resampling gives 87.45 to 91.06 for equal-five mean, not generalisation.

Best-pair gallery plus strongest/middle/weakest and all-character sheets saved for
each writer. Explicitly post-selected distinguishing gallery uses Dylan e,
Yeukita s, Owen n, Abhinav s and 150 i. Their own-minus-other margins remain positive
against all four rival same-letter banks, each comparison using equal bank sizes.
This is exploratory example selection, not held-out author identification.
Sources checked online are Tanks and Temples tutorial Appendix B (nearest-point F)
and Chen et al. (2022), p. 5 (IoU and adaptive stroke-width rationale). No human readback
results exist. Frozen masks, image toolkit, scripts, scores and hashes are packaged.


## Quick connected-word extension, 2 November 2026

Offline exploratory test in Report/GeometricFidelity_Cursive_2026-11-02. No report or application edits. Eight Thiya whole words and seventeen Abhinav whole words, nearest same-word original from fixed banks. Unchanged line-x-height F means are18.79 and44.26 percent. Separate isotropic total-word-height normalisation gives55.18 and67.38 percent, not comparable to earlier character means or the original tolerance in physical units. Best diagnostic Thiya but71.14, Abhinav of96.16 and one the86.47. Five of seventeen Abhinav words exceed.85 under the diagnostic, zero Thiya. No85-percent cursive accuracy claim. Source crops show spacing, scale and join differences, with causes not isolated. All25 scores, bank comparisons and PNG/PDF figures saved. Output Handwriting_Cursive_Quick_Test_2026-11-02.zip and matching findings MD. Previous five-writer evidence unchanged.


## All eleven authors, two whole-word normalisations, 2 November 2026

Separate full offline extension in Report/GeometricFidelity_AllAuthors_2026-11-02.
Both methods applied to the same 141 whole words across all 11 authors with physical
reproduction pages, out of 166 eligible block-C words across 44 lines. Remaining 25
have no accepted exact case/punctuation same-writer reference and are unassessed,
not failures. Only the source passages were compared, not generic blocks A/B.
Original-bank selection precedes scores, up to 8 variants by page/source order.
983 candidate occurrences, 502 selected, 22 visual rejections, 480 accepted references.
All 166 physical proposals retained, including poor strokes/joins. Mean results
are optimistic nearest-reference shape F, not accuracy, IoU or style correlation.

Writer, scored/eligible, line-xheight F%, whole-word-height F%.
150 14/15 71.29 89.88; 152 13/15 60.00 74.54; 153 8/15 49.82 77.93;
384 13/14 49.71 75.76; 552 11/19 49.52 71.41; 588 11/13 41.17 68.93;
Abhinav 18/20 45.71 67.74; Dylan 10/11 47.06 81.13; Owen 19/19 52.04 76.72;
Thiya 8/9 18.79 55.18; Yeukita 16/16 60.85 83.36.
Equal-writer means 49.63/74.78. Word-weighted means 51.25/75.48.
Nine/141 line-height and 44/141 word-height scores individually reach F=.85.
Only 150 has a whole-word mean above 85. Whole-word normalisation changes the height
reference for the 4.8-pixel tolerance, so it must remain separately labelled.
Do not treat the earlier 89.29 character-subset mean as this whole-word test.

All 282 selected pairs (141 words x 2 methods) exactly verified against the original
raster scorer and independent cKDTree distances. Frozen masks/toolkit, all bank
scores, unsupported-word list, protocol, pre-score review, 11 per-writer comparison
figures, complete audit sheets, overview, scripts and hashes saved. No training,
recognition model inference, robot operation or report/application edits.
Output Handwriting_All_Authors_Both_Methods_2026-11-02.zip (14.6 MB), matching MD.


## Measured natural-writing baseline, 2 November 2026

New separate experiment Report/NaturalVariationBaseline_2026-11-02. User requested
comparison with authors' own natural word variation rather than changing thresholds.
Uses frozen accepted all-author masks, no new score-selected crops or tolerance.
Cross-page matched banks compare each genuine probe and robot probe against exactly
the same other-page references. Self comparisons forbidden, zero identical-gray
reference duplicates found. Both fixed normalisations and leave-one-occurrence-out
secondary policy retained. No report, runtime, model or hardware changes.

Primary word-height cohort is 83 physical words, 77 writer/token groups, all 11
writers. Natural nearest-reference mean F=86.7935%, robot=76.4703%, equal-writer
ratio=88.1061% OF THE NATURAL BASELINE. Ordinary mean of all reference pairings
(no best selection) gives natural=79.4474%, robot=68.9964%, ratio=86.8454%.
These are descriptive relative agreement ratios, NOT accuracy, Pearson r, counts
of acceptable words, theoretical ceilings or proof of the proposal's 85% target.
Individual ratios can exceed 100%; retain uncapped. Keep both raw scores visible.

Per writer cross-page word-height ratio percentages (physical word counts).
150 97.26 (8), 152 83.63 (8), 153 82.90 (6), 384 88.99 (6), 552 83.54 (8), 588 91.36 (5),
Abhinav 82.40 (12), Dylan 93.83 (10), Owen 97.50 (5), Thiya 70.34 (4), Yeukita 95.09 (11).
Line-x-height relative nearest-reference result 81.05%, not hidden. Same-page repeats
allowed in secondary leave-one-out give 88.52% word-height ratio on 87 words.
Conditional word-type bootstrap 10k range 85.16–90.83% for primary ratio, 83.68–89.73%
for ordinary-pair ratio. Fixed banks/writers, not population/specification guarantee.

Baseline cohort has shorter common words (3.63-character mean vs 4.73 across prior
141 words). 83 is a subset of 141, not replacement full-page result. Missing repeats
limit coverage, especially Owen 5 and Thiya 4. Genuine/genuine tests 4480 ordered
pairs across norms and policies, 1828 matched folds. 162 genuine pairs (every bank,
both norms) exactly checked against full raster and independent cKDTree. Earlier
282 robot winning pairs separately validated. No neural modules used.

Output Handwriting_Natural_Variation_Baseline_2026-11-02.zip (~6.1 MB) and matching
MD, under output/. Full scores, matched bank memberships, per-author tables, ratio
derivation with real values, conditional ranges, 11 comparison figures, two overview
charts, scatter, masks, scripts and hashes saved. Suggested bounded report wording
in FINDINGS.md. No actual new Pearson correlation metric was computed.
