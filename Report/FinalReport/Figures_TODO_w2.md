# Figures still to be supplied (writer 2, Sections 3.3 - 3.5)

| Label | Where | What it must show | Source |
|---|---|---|---|
| fig:seg-illum | 3.3 | 4 panels of one photograph: colour image (2400 px), luma grey G, background estimate B (heat map), corrected image I_c | run ProcessPage stages on `Software/CNN/NOGIT/dylan/preprocessing_deskew_ruleline_border_cleanup.jpg`, save intermediate arrays |
| fig:seg-mask | 3.3 | 2x3 panel of page-mask construction (Otsu candidates, largest component, filled/closed, hull repair, eroded mask on photo) | stage 2 of ProcessPage (DetectPageMask) |
| fig:seg-skewmeas | 3.3 | plot of J(theta) (coarse and fine grid) with chosen angle, plus page before/after rotation | call EstimateSkew scoring for each angle on a tilted photo |
| fig:seg-result | 3.3 | deskewed page with green TEXT / orange MESS boxes and order labels, plus 3 line crops (one before/after per-line deskew) | existing `*_segpreview.png` in `Software/CNN/NOGIT/dylan` and `.../yeukita`; crops from ProcessPage |
| fig:wid-strokereal | 3.5 | real stroke normalisation strip: grey crop, Otsu mask, core band with x_h, skeleton, re-inked line, final canvas; one IAM writer and one personal writer | StrokeNormalize in `Software/CNN/TrainAuthor.py` on crops from `Data/Datasets/IAMpages10` and `NOGIT/dylan` |

Optional additions (not placeholders in the text): CTC output heat map (160x75 log-probabilities) of a real line from the numpy recogniser; bar chart of the accuracies of Table rec-results; confusion matrix of the shape classifier (123 validation lines).

Other placeholders (text, not figures): see `TO BE COMPLETED BY STUDENT` in 3_1_summary.tex (training computer; rig details; gantry CAD/BOM; tests R3-R5), 3_2_blockdiagram.tex (component models, supply, pen-motor switching, level shifting), 3_3_segmentation.tex (ODROID timing; segmentation accuracy), 3_4_textrec.tex (ODROID recogniser time), 3_5_writerid.tex (ODROID classifier time).

Note: after condensation the placeholders fig:seg-illum, fig:seg-mask, fig:seg-skewmeas (Section 3.3) and fig:wid-strokereal (Section 3.5) are in 4_Appendices/Appendix_w2.tex; fig:seg-result stays in the main text.
