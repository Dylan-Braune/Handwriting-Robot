from pathlib import Path
import zipfile,difflib,json
R=Path('Report/SubmissionRevision_2026-11-02');p=Path('output/Handwriting_Robot_Concise_Source_2026-11-02.zip')
with zipfile.ZipFile(p) as z:
 a=z.read('3_Main_Report/4_Results.tex').decode();b=(R/'3_Main_Report/4_Results.tex').read_text();d=list(difflib.unified_diff(a.splitlines(),b.splitlines(),n=1));print('\n'.join(d[:12]))
p=Path('tmp/package_geometry_study.py');s=p.read_text();a=s.index('# Confirm the report source');b=s.index('# Preserve the completed analysis',a);s=s[:a]+'# No report-writing operations were performed by this study. Existing edits are preserved.\n'+s[b:];s=s.replace('The existing main report remains 63 pages and is unchanged.','This study did not edit the report. The last delivered main report was 63 pages.');s=s.replace("print('Analysis validated. Existing report unchanged.')","print('Analysis validated. No report edits performed.')");p.write_text(s)
