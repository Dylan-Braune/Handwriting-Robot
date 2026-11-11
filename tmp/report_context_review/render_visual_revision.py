from pathlib import Path
here=Path(__file__).resolve().parent
s=(here/'render_design.py').read_text(encoding='utf-8')
s=s.replace('Report/DesignRevision_2026-11-01','Report/DesignVisualRevision_2026-11-01').replace('tmp/report_context_review/design_render','tmp/report_context_review/visual_revision_render')
exec(compile(s,str(here/'render_design.py'),'exec'))
