# Final QTP2, timing and conclusion revision, 2 November 2026

QTP1 is unchanged. QTP2 is restored from the supplied SubmissionRevision_2026-11-02.zip, including the partition caveat, both reader columns, exact-line count, confidence example, author confusion matrix and statistical explanation. The confusion matrix uses the existing readable row-normalised figure and is not repeated in the appendix.

Discussion is shortened to three pages while retaining all compulsory Appendix 4 headings. Conclusion is completed and now reflects the current 0.49 mm maximum dimensional error and 86.68% natural-variation-relative geometric score. The older 0.697 descriptor correlation is no longer quoted in the Conclusion.

QTP4 adds existing laptop records from Report/_dev/qtp_data/qtp4_results.json, thirteen pages with three runs each. The table separates image preparation, text recognition, author classification, complete time and allowance. All thirteen page means exceed the allowance. The Windows/Intel processor details are from that record. ODROID measurements remain separately labelled.

The existing fourteen-page batched-library benchmark measures 359.8 s against 130.3 s for combined networks, a factor of 2.76. Applying that factor to the laptop network stages with unchanged image time gives conditional totals of 30.63 s, 35.59 s and 39.70 s for L3, L4 and L5, within their respective budgets. These are projections, not measured passes or proof of an ODROID speed gain. L1 remains above budget. The benchmark changed both backend and batching, and only reader-output equivalence was established.

All twelve PDFs in the working report's Figures/VisualTests folder now match Report/SubmissionRevision_2026-11-02/Figures/VisualTests byte for byte. No new image-processing experiment, inference run, benchmark or training was performed. Pre-Results wording was preserved, with the requested figure replacements only. QTP1 is unchanged.

The main report is 67 pages from Literature through Conclusion. Results occupies fourteen pages, Discussion three and Conclusion one. The complete PDF is 112 pages including front matter, references and appendices. Local pdfLaTeX/BibTeX compilation succeeds without undefined references or overflowing text. Changed pages and the supplied figure replacements were visually checked. The built-in preview remains limited to 20 MiB.

The earlier protected Design comparison table still contains its original writer-training claims. That text was outside this request and was not changed. Consult the previous provenance notes before comparing training-validation and separately evaluated checkpoint scores.
