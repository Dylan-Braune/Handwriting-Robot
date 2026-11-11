from pathlib import Path
p=Path('Report/SubmissionRevision_2026-11-02/3_Main_Report/4_Results.tex');s=p.read_text();s=s.replace('Figure~\\ref{fig:res-physical} provides examples of actual robot output alongside both strong and more difficult cases.','Figure~\\ref{fig:res-physical} provides examples of actual robot output alongside both strong and more difficult cases. Strip labels give each writer\'s aggregate character accuracy. The dashed 85\\% line in Figure~\\ref{fig:res-writers} marks the character target, with identity reported separately.');p.write_text(s)
p=Path('Report/SubmissionRevision_2026-11-02/Appendices/results_evidence.tex');s=p.read_text().replace('Figure~\\ref{fig:app-physical-style} shows the actual writer means.','Figure~\\ref{fig:app-physical-style} shows the actual writer means. Each point represents one writer, and dashed lines mark equal feature values.');p.write_text(s)
p=Path('Report/REPORT_COLLABORATION_CONTEXT.md');s=p.read_text();note='''## Latest full-report revision, 2 November 2026

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

'''
if '## Latest full-report revision, 2 November 2026' not in s:s=s.replace('# Report collaboration context\n','# Report collaboration context\n\n'+note,1)
p.write_text(s)
