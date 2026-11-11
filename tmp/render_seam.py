from pathlib import Path
import pypdfium2 as pdfium
p=Path('Report/SubmissionRevision_2026-11-02/Figures/VisualTests/T25_seam_cost_field.pdf');d=pdfium.PdfDocument(str(p));d[0].render(scale=1.2).to_pil().save('tmp/final_report_review/render/full-seam.png')
