from pathlib import Path
import shutil
r=Path.cwd().resolve();src=(r/'Report/IntegratedReport_2026-11-01/.build').resolve();dst=(r/'Report/ReportWorkingArchive_2026-11-01_VisualTests/CleanBuildVerification').resolve()
assert src.is_relative_to(r/'Report/IntegratedReport_2026-11-01') and dst.is_relative_to(r/'Report/ReportWorkingArchive_2026-11-01_VisualTests') and not dst.exists()
shutil.move(str(src),str(dst));print('Build intermediates archived; source folder contains',sum(p.is_file() for p in (r/'Report/IntegratedReport_2026-11-01').rglob('*')),'files.')
