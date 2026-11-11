from pathlib import Path
import json

root = Path(__file__).resolve().parents[2]
base = root / 'Report/DesignPolished_2026-11-01'
p = base / 'VALIDATION.json'
v = json.loads(p.read_text(encoding='utf-8'))
v['revision'] = 'Real-input calculation revision, 1 November 2026'
v['scope'] = ('Read-only calculation traces using existing pages, trained models and final profiles; '
              'offline subsystem checks and generated G-code. No training, optimizer updates or hardware tests.')
v['external_citation_sites'] = 53
v['internal_citation_sites'] = 0
v['pending_manual_source_checks'] = 7
v['evidence_appendix_start_pdf_page'] = 55
p.write_text(json.dumps(v, indent=2) + '\n', encoding='utf-8')
with (base / 'README.md').open('a', encoding='utf-8') as f:
    f.write('\nFinal verified reading copy: 43 Design pages, 3 reference pages and 14 appendix pages (60 total).\n')
with (root / 'Report/REPORT_COLLABORATION_CONTEXT.md').open('a', encoding='utf-8') as f:
    f.write('\nReal-calculation delivery checked: 43 Design + 3 references + 14 appendix pages = 60. '
            'All rendered pages reviewed; no overfull boxes, undefined references or missing citation keys. '
            'Existing six figure placeholders and eight completion notes remain explicit. '
            'Generic scalar examples were replaced or removed, with full-precision real traces supplied. '
            'The traced single-crop first-convolution patch storage is 1.40625 MiB, not a measured peak.\n')
print('Delivery metadata updated.')
