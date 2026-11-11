from pathlib import Path
import zipfile,json,re,shutil
r=Path.cwd();b=r/'Report/IntegratedReport_2026-11-01';a=r/'Report/ReportWorkingArchive_2026-11-01_VisualTests'
log=(b/'.build/full_report.log').read_text(errors='replace');bad=re.findall(r'[^\n]*(?:Overfull|undefined|multiply defined|Float too large)[^\n]*',log);assert not bad,bad
fls=(b/'.build/full_report.fls').read_text(errors='replace');assert 'ReportWorkingArchive' not in fls
pdf=r/'output/pdf/Handwriting_Robot_Full_Report_Visuals_2026-11-01.pdf';shutil.copy2(b/'.build/full_report.pdf',pdf)
zpath=r/'output/Handwriting_Robot_Clean_Compile_Source_2026-11-01.zip'
files=sorted(p for p in b.rglob('*') if p.is_file() and '.build' not in p.parts)
with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
 for p in files:z.write(p,p.relative_to(b).as_posix())
with zipfile.ZipFile(zpath) as z:
 assert z.testzip() is None
 assert 'full_report.tex' in z.namelist() and '2_Project_Definition/project_proposal.pdf' in z.namelist()
 assert not any(n.endswith(('.aux','.log','.npz','.json')) for n in z.namelist())
record={'clean_build_passed':True,'source_files':len(files),'source_zip':str(zpath),'output_pdf':str(pdf),'archive':str(a),'archived_files':536,'visual_test_figures':12,'all_original_images_reviewed':41,'visual_inspection':'all twelve included figures and their adjacent explanatory pages inspected; no clipping, overlap or unresolved references observed'}
(a/'FINAL_CLEAN_BUILD_CHECK.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
p=r/'Report/REPORT_COLLABORATION_CONTEXT.md';s=p.read_text(encoding='utf-8');note='''## Current authoritative report: full assembly with visual tests and clean compile folder

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

''';p.write_text(s.replace('# Report collaboration context\n','# Report collaboration context\n\n'+note,1) if s.startswith('# Report collaboration context') else s.replace('# Handwriting Robot report context\n','# Handwriting Robot report context\n\n'+note,1),encoding='utf-8')
print(json.dumps(record,indent=2))
