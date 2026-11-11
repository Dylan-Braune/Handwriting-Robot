from pathlib import Path
import hashlib
import json
import shutil
import zipfile

root = Path(__file__).resolve().parents[2]
revision = root / 'Report/DesignRevision_2026-11-01'
validation_path = revision / 'VALIDATION.json'
validation = json.loads(validation_path.read_text(encoding='utf-8'))
validation['visual_review'] = 'Complete: all 45 final rendered pages inspected; no clipping, overlap or broken cross-references observed. Draft completion and source flags are intentional.'
validation_path.write_text(json.dumps(validation, indent=2) + '\n', encoding='utf-8')

inventory = json.loads((revision / 'CODE_COVERAGE_INVENTORY.json').read_text(encoding='utf-8'))
for name, record in inventory.items():
    path = root / 'Software/CleanPipeline3' / name
    if not path.exists():
        raise FileNotFoundError(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], name

output = root / 'output'
(output / 'pdf').mkdir(parents=True, exist_ok=True)
pdf = output / 'pdf/Handwriting_Robot_Design_Reviewed_2026-11-01.pdf'
shutil.copy2(revision / 'design_preview.pdf', pdf)
archive = output / 'Handwriting_Robot_Design_Reviewed_Prism_Source_2026-11-01.zip'
allowed = {'.tex', '.bib', '.cls', '.md', '.json', '.py', '.png', '.jpg', '.jpeg', '.pdf'}
included = []
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as package:
    for path in sorted(revision.rglob('*')):
        if path.is_file() and path.suffix.lower() in allowed and path.name != 'design_preview.pdf':
            package.write(path, path.relative_to(revision).as_posix())
            included.append(path.relative_to(revision).as_posix())
with zipfile.ZipFile(archive) as package:
    assert package.testzip() is None
assert pdf.read_bytes() == (revision / 'design_preview.pdf').read_bytes()
print(json.dumps({'pdf': str(pdf), 'source_zip': str(archive), 'zip_files': len(included), 'code_hashes_unchanged': len(inventory), 'pdf_bytes': pdf.stat().st_size}, indent=2))
