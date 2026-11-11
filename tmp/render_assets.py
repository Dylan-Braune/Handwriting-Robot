from pathlib import Path
import pypdfium2 as pdfium
P=Path('Report/SubmissionRevision_2026-11-02');out=Path('tmp/final_report_review/render');out.mkdir(exist_ok=True)
for name in ['design-seam-two-panels','design-owen-first-features','results_reproduced_confusion']:
 d=pdfium.PdfDocument(str(P/'Figures'/f'{name}.pdf'));d[0].render(scale=1.5).to_pil().save(out/f'{name}.png')
