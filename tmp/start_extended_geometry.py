from pathlib import Path
import ast
B=Path('Report/GeometricFidelity_Extended_2026-11-02');old=Path('Report/GeometricFidelity_2026-11-02/measure_geometry.py').read_text();tree=ast.parse(old)
functions='\n\n'.join(ast.get_source_segment(old,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['canvas','compare'])
(B/'geometry_core.py').write_text("from pathlib import Path\nimport sys\nimport numpy as np\nfrom scipy.ndimage import distance_transform_edt\nROOT=Path(__file__).resolve().parents[2]\nsys.path.insert(0,str(ROOT/'Software/CleanPipeline3'))\nimport segment as SEG\nTOLS=[.05,.10,.15,.20]\n\n"+functions+'\n')
(B/'PROTOCOL.md').write_text('''# Extended reference-variant comparison, 2 November 2026

This is a new experiment. The previous fixed-occurrence study is preserved.
Writers are Dylan, Yeukita, Owen, Abhinav and IAM 150. Keep the old scoring rule,
48-pixel x-height, isotropic scale, bounded translation and primary F at 0.10
x-height. Do not change tolerance to achieve a desired result.

Two descriptions must remain distinct.

1. Reference-bank fit. Test every visually verified, geometrically separable
   character available from the physical source-text blocks. Compare it to up to
   eight verified examples of the SAME character and SAME writer in original pages.
   The bank is selected by page diversity and source order, before geometry scores.
   Take the largest F-score within this fixed bank, recording which original was
   selected. This is nearest-exemplar reconstruction fit, an optimistic reference
   comparison, not a new blind recognition accuracy or a generalisation guarantee.
2. Best-case visual examples. Select the strongest valid pair for each writer and
   show its score, source and letter explicitly. The equal-writer mean of five such
   maxima is a selected-example mean, not overall system fidelity.

Preserve the fixed-reference results as a baseline. Also compare same-letter other-
writer banks with identical processing and matched bank sizes wherever possible.
Report the number of reference variants and evaluation characters. Report per-writer,
character-weighted and equally writer-weighted averages without omitting low scores.
Only annotate glyph identity/separability and exclude contaminated or mislabelled
crops, not low-scoring shapes. All edits and exclusions are recorded. No neural
classifier or model training is used. Do not edit report LaTeX during this study.
''')
