from pathlib import Path
HERE=Path(__file__).resolve().parent
s=(HERE/'validate_visual_revision.py').read_text(encoding='utf-8')
s=s.replace('Report/DesignVisualRevision_2026-11-01','Report/DesignPolished_2026-11-01').replace('Visual evidence and pseudocode pass, 1 November 2026','Full Design review and polish, 1 November 2026')
(HERE/'validate_polished_revision.py').write_text(s,encoding='utf-8')
s=(HERE/'render_design.py').read_text(encoding='utf-8').replace('Report/DesignRevision_2026-11-01','Report/DesignPolished_2026-11-01').replace('tmp/report_context_review/design_render','tmp/report_context_review/polished_render')
(HERE/'render_polished_revision.py').write_text(s,encoding='utf-8')
s=(HERE/'package_visual_revision.py').read_text(encoding='utf-8').replace('Report/DesignVisualRevision_2026-11-01','Report/DesignPolished_2026-11-01').replace('Handwriting_Robot_Design_Visual_Revision','Handwriting_Robot_Design_Polished').replace('all 53 rendered pages reviewed','all final rendered pages reviewed')
(HERE/'package_polished_revision.py').write_text(s,encoding='utf-8')
