from pathlib import Path
import json, shutil, zipfile, hashlib

root = Path(__file__).resolve().parents[2]
base = root / 'Report/DesignResultsDiscussion_2026-11-01'
review_path = base / 'INTEGRATED_REVIEW.json'
review = json.loads(review_path.read_text(encoding='utf-8'))
review['visual_review'] = {
    'status': 'complete for revised sections and changed Design pages; hardware and software diagram descriptions checked separately',
    'rendered_pages': review['pdf_pages'],
    'inspected_pages': 'Latest hardware and software diagrams and explanations pages 2-4; other sections inherited from preceding visually reviewed build',
    'findings': 'No clipping, overlapping text, broken tables or unresolved reference markers in inspected pages. Intentional pending figures and author completion notes remain.'
}
review_path.write_text(json.dumps(review, indent=2)+'\n', encoding='utf-8')
out = root / 'output'
(out / 'pdf').mkdir(parents=True, exist_ok=True)
pdf = out / 'pdf/Handwriting_Robot_Design_Results_Discussion_2026-11-01.pdf'
shutil.copy2(base / 'report_preview.pdf', pdf)
archive = out / 'Handwriting_Robot_Report_Polished_2026-11-01.zip'
excluded = {'.aux', '.bbl', '.blg', '.out', '.toc', '.synctex.gz', '.pyc'}
files = []
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(base.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts:
            continue
        rel = p.relative_to(base)
        if len(rel.parts) == 1 and (p.name.startswith('report_preview.') and p.suffix != '.tex'):
            continue
        if p.suffix in excluded:
            continue
        z.write(p, rel.as_posix())
        files.append(rel.as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert 'ResultsEvidence/measurements/raw/odroid_server_log.txt' in z.namelist()
    assert len([n for n in z.namelist() if n.startswith('ResultsEvidence/checkpoints/') and n.endswith('.npz')]) == 4
assert pdf.read_bytes() == (base / 'report_preview.pdf').read_bytes()
context = root / 'Report/REPORT_COLLABORATION_CONTEXT.md'
text = context.read_text(encoding='utf-8')
note = ('\nDelivery completed: PDF `output/pdf/Handwriting_Robot_Design_Results_Discussion_2026-11-01.pdf`; '
        'source/evidence ZIP `output/Handwriting_Robot_Report_Polished_2026-11-01.zip`. '
        f"{review['pdf_pages']} PDF pages: Design {review['design_pages']}, Results {review['results_pages']}, Discussion {review['discussion_pages']}, then references/appendices. "
        'This is a Sections 3-5 review, not the complete submission. '
        'Clean build and reference checks passed; revised sections and changed Design pages visually inspected.\n')
if note.strip() not in text:
    context.write_text(text.replace('## Latest task update:', note+'\n## Latest task update:', 1), encoding='utf-8')
print(json.dumps({'pdf': str(pdf), 'zip': str(archive), 'files': len(files), 'zip_bytes': archive.stat().st_size, 'pdf_sha256': hashlib.sha256(pdf.read_bytes()).hexdigest()}, indent=2))
