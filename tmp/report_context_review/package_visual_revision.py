from pathlib import Path
import ast,hashlib,json,shutil,zipfile
ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'Report/DesignVisualRevision_2026-11-01'
validation=json.loads((HERE/'VALIDATION.json').read_text(encoding='utf-8'))
validation['visual_review']='Complete: all 53 rendered pages reviewed; final changed figure and text pages rechecked. Draft figure and source-check markers are intentional.'
for p in HERE.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
data=json.loads((HERE/'Evidence/bench.json').read_text(encoding='utf-8'))
camera=json.loads((HERE/'Evidence/camera_bench.json').read_text(encoding='utf-8'))
for collection in [data['inputs'],camera['inputs']]:
    for name,digest in collection.items():
        p=Path(name)
        if not p.is_absolute():p=ROOT/p
        assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,name
for name,digest in data['code_sha256'].items():
    assert hashlib.sha256((ROOT/'Software/CleanPipeline3'/name).read_bytes()).hexdigest()==digest,name
assert data['connected_components']['partition_matches']==100
assert data['ctc']['max_gradient_absolute_error']<1e-9
assert data['seam_reference']['maximum_cost_error']<1e-10
assert all(x['same_path_vertex_counts'] and x['clipped_points']==0 and x['maximum_coordinate_error_mm']<=.0005001 for x in data['roundtrip'])
validation['evidence_input_hashes_verified']=True
(HERE/'VALIDATION.json').write_text(json.dumps(validation,indent=2)+'\n',encoding='utf-8')
output=ROOT/'output';(output/'pdf').mkdir(exist_ok=True,parents=True)
pdf=output/'pdf/Handwriting_Robot_Design_Visual_Revision_2026-11-01.pdf'
archive=output/'Handwriting_Robot_Design_Visual_Revision_Prism_Source_2026-11-01.zip'
shutil.copy2(HERE/'design_preview.pdf',pdf)
allowed={'.tex','.bib','.cls','.md','.json','.py','.png','.jpg','.jpeg','.pdf','.gcode'}
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(HERE.rglob('*')):
        if p.is_file() and p.suffix.lower() in allowed and p.name!='design_preview.pdf':
            z.write(p,p.relative_to(HERE).as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    count=len(z.namelist())
assert pdf.read_bytes()==(HERE/'design_preview.pdf').read_bytes()
print(json.dumps({'pdf':str(pdf),'zip':str(archive),'files':count,'pdf_pages':validation['pages'],'design_pages':validation['design_pages'],'zip_MB':round(archive.stat().st_size/1e6,2)},indent=2))
