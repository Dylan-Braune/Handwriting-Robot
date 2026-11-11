# Integrated Design, Results and Discussion revision

Entry point: `report_preview.tex`. Build with `python build_report.py` from this
directory, using a Python environment with NumPy, Pillow, Matplotlib and MiKTeX
on PATH. The original epr400 class is retained unchanged.

This review assembles Sections 3, 4 and 5 plus references and supporting
appendices. It is not the full final submission: the author's Literature,
Approach, Conclusion and front matter must be assembled separately. Their
addition will require an overall page-budget pass against the 70-page main-text
cap. Do not mistake this section review for a completed 70-page whole report.

## Current author decisions

- Selected demonstration: library-trained joint reader and library-trained
  eleven-writer model, normalisation width 0.17. Numerical-route models are
  evaluated alongside them. No deployment setting was changed by this revision.
- Use the author's supplied hardware diagram unchanged. It remains embedded
  from `Evidence/user_hardware_diagram_input.png`.
- Treat the current glyph libraries as final; do not insert a manual-curation
  feature or a narrative about curation.
- Signed consent is confirmed for all five contributors. Signatures remain
  private. Institutional ethics clearance/exemption is still to be supplied.
- The seven physical geometry readings are confirmed measurements of the final
  calibrated machine. The instrument is a metal non-digital vernier caliper;
  model and least count are pending.
- Approximately R4000 reported build cost; borrowed ODROID at no purchase cost,
  returned after the project. Reconcile the later R750 list before final costing.

## Measurement and analysis

`ResultsEvidence/measurements/RESULTS_PROVENANCE.md` is Claude's original archive,
preserved rather than rewritten. `ResultsEvidence/REVIEWED_METRICS.json` contains
the counts recalculated for this revision. `review_results.py` rechecks the saved
records; it does not train models or operate hardware. `make_results_figures.py`
uses preserved local JSON paths, without a session-temp-directory dependency.

`measure_physical_style.py` is a new retrospective geometric study on the real
pages and original reference crops. Its method, all descriptor rows, input/code
hashes, exclusions and resampling results are in
`ResultsEvidence/physical_style/results.json`. It uses 156 original lines and
184 physical lines across eleven writers, without exclusions. It requires the
original repository data and SciPy/OpenCV; it runs offline and writes only here.
The seven-feature shape coefficient is 0.6967, nine-feature coefficient 0.7127.

Four measured model files are preserved under unambiguous names in
`ResultsEvidence/checkpoints/`, with their full hashes. These are evidence
copies, never replacements of the checkpoint names resolved by the server.
Some original scripts in the preserved archive retain historic absolute paths;
their source is documentary. Use this revision's portable review/plotting
scripts for aggregate reconstruction, and the recorded source inputs to repeat
model inference. The raw original images and large personal cache remain in the
repository, identified by hashes rather than duplicated in the report bundle.

## Important corrections from v6

- The selected joint reader gives 94.06% pooled character accuracy. The evaluated
  numerical reader gives 94.39%; they are not silently interchanged.
- The reader edit counts are 2719 and 2567 on 45747 reference characters.
- The word score is positional word match, not standard word edit accuracy.
- Earlier r=0.714 is seven shape features on ten writers and software renders;
  nine features give r=0.701. The new physical study is a separate experiment.
- Primary R3 timing uses 25 console records with explicit completion markers.
  It excludes stopped and zero-character jobs; maximum remains 3.875 s/char.
- Batching and the framework backend changed together in the 2.76x experiment.
  Its host speed factor is not credited as a measured ODROID pass.
- The 0.5 mm geometry comparison concerns feature lengths. Absolute coordinate
  accuracy and repeated-word overlay require their own reference measurements.
- Proposal field conditions are 400--1000 lux and stable, level, vibration-free
  support. They are not replaced with paper/pen choices from the v6 summary.

## Remaining author records

The overlay figure is explicitly reserved; no passing result is invented.
Numeric lighting readings, the caliper identification/least count, printed-part
material and institutional ethics statement are pending. The existing Design
figure and hardware completion slots remain visible. Source-page flags inherited
from the Design are retained for the author's manual checks. POPIA sections were
checked against the official PDF preserved in `ResultsEvidence/sources/`.

Legacy Design audit notes copied into this directory describe prior Design-only
passes. The current integrated status is `INTEGRATED_REVIEW.json` and this README.
