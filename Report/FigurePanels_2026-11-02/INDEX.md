# separate panel images, 2 November 2026

every figure the report generates, split into one image file per panel.
the folder name is the test or figure name, and the files inside are
numbered left to right and top to bottom as they appear in the composite.

two kinds of file:

- `NN_<panel>.png` the panel as it is drawn in the report, with its own
  title and axis labels, at 300 dpi
- `NN_<panel>_image.png` where a panel is a single photograph, crop or
  feature map, the bare pixels at full resolution, with no axes. this is
  the one to re-crop or re-scale

## figures used in the report

| folder | panels | bare images | where it appears |
|---|---|---|---|
| `D25_seam_design` | 3 | 3 | Design 3.4, seam (short version, installed as design-seam-two-panels) |
| `D32_first_features` | 5 | 5 | Design 3.5, first features (short version, installed as design-owen-first-features) |
| `design-candidate-selection` | 3 | 0 | Design 3.6, candidate selection |
| `design-glyph-alignment` | 11 | 0 | Design 3.6, glyph alignment |
| `design-physical-page-example` | 5 | 0 | Design 3.7, physical page |
| `design-segmentation-stages` | 6 | 0 | Design 3.4, conditioning stages |
| `design-segmentation-variation` | 6 | 0 | Design 3.4, segmentation under three captures |
| `design-stroke-normalisation` | 8 | 0 | Design 3.5, stroke normalisation |
| `design-style-gcode-examples` | 8 | 0 | Design 3.6, trajectory and G-code |
| `results_geometry_errors` | 1 | 0 | Results, geometry errors |
| `results_holdout_confusion` | 1 | 1 | Results, held-out writer confusion |
| `results_physical_examples` | 4 | 4 | Results, reproduced lines |
| `results_physical_style_scatter` | 9 | 0 | Results, style scatter |
| `results_r3_confirmed` | 1 | 0 | Results, R3 |
| `results_r4_stages` | 1 | 0 | Results, R4 stages |
| `results_reproduced_confusion` | 1 | 1 | Results, reproduced-line writer confusion |
| `results_writer_performance` | 1 | 0 | Results, writer performance |
| `T25_seam_cost_field` | 6 | 6 | Appendix B (extended seam figure) |
| `T27_ownership_corrections` | 6 | 6 | Design 3.4, ink ownership |
| `T32_feature_stack` | 18 | 18 | Appendix B (extended feature figure) |
| `T36_ctc_lattice` | 4 | 3 | Design 3.5, CTC lattice |
| `T38_gradient_check` | 2 | 0 | Design 3.5, gradient check |
| `T40_stroke_width_sweep` | 21 | 20 | Design 3.5, stroke-width sweep |
| `T41_skeleton_trace_simplify` | 4 | 4 | Design 3.6, skeleton and pen paths |
| `T43_glyph_coordinates` | 6 | 1 | Design 3.6, glyph coordinates |
| `T45_variant_pool_medoid` | 12 | 0 | Design 3.6, variant pool |
| `T49_connectedness` | 4 | 4 | Design 3.6, joinedness |
| `T52_bresenham_staircase` | 2 | 0 | Design 3.3, stepper staircase |
| `T56_time_budget` | 2 | 0 | Design 3.7, time budget |

## the rest of the visual tests

produced by the same scripts and kept for reference; these are not placed
in the report.

| folder | panels | bare images |
|---|---|---|
| `T15_software_block_diagram` | 1 | 0 |
| `T19_glyph_quality_gates` | 66 | 0 |
| `T20_joining_by_writer` | 5 | 0 |
| `T21_otsu_criterion` | 5 | 3 |
| `T22_deskew_sweep` | 4 | 3 |
| `T23_quarter_turn_margin` | 2 | 0 |
| `T24_profile_pitch_centres` | 3 | 1 |
| `T26_seam_weight_sensitivity` | 2 | 0 |
| `T29_fixed_canvas_fit` | 8 | 8 |
| `T30_convolution_arithmetic` | 5 | 4 |
| `T31_learned_kernels` | 32 | 32 |
| `T33_max_pooling` | 4 | 4 |
| `T35_frame_to_pixel` | 2 | 2 |
| `T37_frame_posteriors` | 2 | 1 |
| `T39_augmentation_set` | 20 | 20 |
| `T42_cut_cost_refinement` | 3 | 2 |
| `T46_shape_descriptor` | 8 | 4 |
| `T47_slant_search` | 2 | 1 |
| `T48_jitter_ladder` | 4 | 0 |
| `T50_style_calibration` | 6 | 0 |
| `T51_candidate_selection_dylan` | 6 | 5 |
| `T51_candidate_selection_owen` | 6 | 5 |
| `T53_directional_pulse_penalty` | 2 | 0 |
| `T54_stroke_order` | 7 | 0 |
| `T55_decimation_ladder` | 6 | 0 |
| `T57_machine_placement` | 1 | 0 |
| `T59_page_vote` | 4 | 1 |
| `T60_round_trip` | 9 | 3 |
| `T61_job_states` | 1 | 0 |

## how these were produced

the generators were re-run with `FIG_PANELS` set to this folder, which is
the only thing that switches the panel export on. every one of the 27
generated figures the report uses was re-rendered and compared with the
installed copy pixel by pixel: all identical, so nothing in the report
moved. the report was rebuilt afterwards and is still 139 pages with no
overfull boxes and no undefined references.

the scripts that write the panels are:

- `Report/DesignVisualTests_2026-11-01/scripts/vt_*.py` for `T*` and `D*`
- `Report/DesignResultsDiscussion_2026-11-01/run_visual_bench.py` for the
  six subsystem illustrations
- `Report/DesignResultsDiscussion_2026-11-01/make_design_examples.py` for
  stroke normalisation
- `Report/DesignResultsDiscussion_2026-11-01/make_results_figures.py` and
  `tmp/redraw_confusion.py` for the results figures
