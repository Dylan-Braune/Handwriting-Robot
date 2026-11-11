# Report revision notes, 2 November 2026

## Scope and size

The baseline was the supplied `IntegratedReport_2026-11-01.zip`.
The main report is 63 pages from Literature through Conclusion, compared with
82 pages in the supplied baseline. Design is 37 pages, compared with 50.
The complete PDF is 133 pages including the proposal, front matter, references
and expanded evidence appendices. There are 48 printed references.

All existing main subsection, subsubsection and paragraph headings are retained.
Literature and Approach are byte-for-byte unchanged. The approved proposal PDF
and its wrapper are unchanged. Hardware prose is retained, with the permitted
calculation reductions, CAD restoration and brief figure explanations.

## Changes

- Simplified the explanations from segmentation onwards while retaining general
  formulas, variables, design choices and worked values.
- Moved detailed substitutions into Appendix A, with links from the main text.
- Retained two seam panels and Owen's first feature row in Design. Full originals
  are in Appendix B. Other Design figures and algorithms remain.
- Added panel labels and a short functional table for the eight CAD parts.
- Put explanations before every Design and Results figure. Main figures use fixed
  placement so they follow their introducing text. Captions are shorter.
- Added row-normalised writer matrices for original and physical lines. Each
  reference-writer row sums to 100 percent and shows its actual line count.
  No lines were duplicated or discarded to force equal class sizes.
- Retained 159/184 physical line identifications and 153/156 original identifications.
  The unweighted physical diagonal mean is 85.431 percent, distinct from the
  line-pooled 86.413 percent. The report retains the original pooled score.
- Added a complete Owen photograph in Results, all fourteen physical pages in
  Appendix C, all 184 paired-line scores and all 25 completed ODROID timings.
  Photo pixels are unchanged. LaTeX rotates their display to reading orientation.
- Kept the training-route comparison in Design and transcribed the supplied
  first-principles logs in Appendix C, including the incomplete end of each excerpt.
- Completed the acronyms and all four existing Conclusion headings.
- Corrected the blanket AI and software-library declarations. They previously
  contradicted the documented assistance and implementation. Signatures remain blank.

## Evidence and interpretation

No new physical trials, training runs or gantry operation were performed. New
matrices and page summaries are arithmetic regroupings of the saved prediction
records. All measured accuracies and times are retained. Checkpoint provenance is
recorded in the appendix rather than repeatedly discussed in Results.

The physical page accuracy records and embedded timing records have no verified
one-to-one page/job mapping. They remain separate tables. Development-host timing
improvements are not presented as measured ODROID passes. Feature-length accuracy
is not relabelled as absolute coordinate accuracy. Missing lux measurements and
word overlays were not invented.

## Checks to finish yourself

- Add the complete gantry and measurement-setup photographs. The current 63-page
  main report leaves four pages below the 67-page maximum.
- Confirm the fitted motor specification and the exact `stepperds` source. The
  inherited bibliography record does not identify a supplier or document revision.
  The torque curve and friction assumption remain explicitly conditional.
- Check the full Dziedzic paper for the 10 to 60 mm/s speed range and its page.
  The publisher/DOI record was verified, but the full text was unavailable.
  DOI https://doi.org/10.1016/j.forsciint.2026.112896
- Check the GA12-N20 variant supplier specifications against the installed motor.
  The restored reference is the existing Phipps Electronics listing. Check the
  DML31P listing against the installed switch as well.
- Other existing source page numbers were retained, not newly verified against
  every source PDF. No new page number was invented for unavailable material.
- Obtain the caliper model and least count, fitted Vref and institutional clearance
  or exemption reference. Consent signatures should remain private supporting records.
- Read the corrected declarations against your university's required wording
  before signing. This revision does not certify institutional compliance.
- The protected Approach still says step counts assure accuracy without an
  encoder. That overstates open-loop accuracy. It was left because you expressly
  protected Approach, while Design states the missed-step assumption correctly.
- The protected Literature retains its existing sentence distinguishing selected
  reference-trained weights. It was not silently edited or contradicted elsewhere.

## Validation

The report builds with installed pdfLaTeX and BibTeX. The final build has no
undefined citations, missing references, duplicate labels or overfull boxes.
The main heading hierarchy and retained figure/formula labels were checked against
the baseline. Literature, Approach and proposal preservation were checked separately.
Figures and pages were rendered for visual review. The app compiler timed out
while downloading its TeX environment, so the successful local compiler supplied
this PDF. No tab or the separate open segmentation document was changed.
