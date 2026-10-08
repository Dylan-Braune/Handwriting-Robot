# Figures still to be supplied (writer 3, Sections 3.6 - 3.10)

| Label | Where | What it must show | Source |
|---|---|---|---|
| fig:sty-stages | Appendix (cited in 3.6) | One real line of a database writer and one personal line: grey crop, ink mask with rules removed and core band, Zhang-Suen skeleton, traced chains (colours) with territory cuts, one extracted glyph in the normalised frame | call BinarizeLine/CoreBand/Skeletonize/TracePolylines/ExtractLineGlyphs from `Software/CNN/AuthorReproductionStuff/BuildStyleProfile.py` on crops from `Data/Datasets/IAMpages10` and `Software/CNN/NOGIT/dylan` |
| fig:gan-photo | Appendix (cited in 3.9) | Photograph of the assembled gantry (axes, end-stops, pen holder with toggle motor, paper holder) and of the scanning rig with camera; CAD renderings and a wiring diagram with component values | student supplies (the Hardware folder of the repo is empty) |

Real images already included (copied from the repo, files in `Figures/`): `w3_glyphs153_ae.png` (crop of `Software/CNN/NOGIT/WebCache/glyphs_153.png`), `w3_synth_150.png` and `w3_synth_153.png` (from `Software/CNN/NOGIT/EndToEnd/150_synth.png` and `153_synth.png`, rescaled).

Optional additions (not placeholders in the text): plot of S(theta) for slant estimation of a real line; heat map of p_j and swap probability P_swap(lambda, d); histogram of the joint score over N candidates (data in the brief A4, section 5.4); photographs of a drawn sentence next to the preview image; overlay of the same word drawn twice (R5 test).

Text placeholders (`TO BE COMPLETED BY STUDENT`) are in: 3_6 (R6 demonstration, storage device), 3_7 (final evaluation run and N), 3_8 (G-code verification run), 3_9 (Table gan-hw: motor, driver, jumpers, transmission, calibrated steps/mm, switches, pen motor drive, power supply, camera; R5 test; maximum feed; uncertainty of L_X and L_Y; 3.3 V source of the switches), 3_10 (code submission, network and OS configuration, timing table rows).
