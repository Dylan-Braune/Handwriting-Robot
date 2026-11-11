# Handwriting Robot Live System Evaluation Guide
Code controls and expected effects for classification and reproduction
11 November 2026

This guide identifies changes an evaluator could reasonably ask to see in the running handwriting robot system. The aim is to predict the effect of a change, demonstrate the relevant stage, and explain the observation. It covers classification, segmentation, handwriting synthesis, profile selection, layout and physical execution using existing trained models. Training changes are outside the live comparison.

The central question is whether changing one assumption produces the effect expected from the algorithm. For every control, identify where it acts, what should change, what remains comparable, which later stage can override it, and what observation would contradict the prediction.

## How to use this guide

Geometry changes are usually the most predictable: larger spacing separates words and more shear increases slant. Recognition changes involve trade-offs: a stricter threshold may remove noise and faint strokes together, so accuracy need not change monotonically. Physical execution changes can affect time, ink deposition and mechanical tracking while the digital preview remains identical.

The identifiers and defaults below refer to the inspected Software/CleanPipeline3 implementation and its web server. Some controls are existing request fields, some require changing an argument or constant at its active call site, and some require a small logic edit. An evaluator request is not evidence that a coefficient has an established optimum. Expected effects are algorithmic predictions, not new measured results.

For a controlled comparison, keep the source photograph or text, writer, checkpoint, preprocessing, size and seed fixed except for the selected change. Display the intermediate result nearest that change before interpreting the final score. A fixed seed improves reproducibility, but a changed branch can consume random values differently and alter subsequent choices.

## Twelve comparisons to prioritise

1. Disable and enable joins while retaining the same writer profile.
2. Compare upright and slanted writing.
3. Compare small and large writing and explain automatic fitting.
4. Compare narrow and wide word spacing.
5. Compare zero jitter with increased jitter.
6. Compare personal variants with stronger anchoring.
7. Compare one candidate with multiple candidates and repair off with repair on.
8. Compare the ink and shape writer models with preparation stated.
9. Compare raw and stroke-normalised writer classification.
10. Change a segmentation threshold or seam setting and show the intermediate crops.
11. Compare slower and faster pen-down feed on the same generated geometry.
12. Disable and enable physical endpoint joining and compare pen lifts and added connectors.

For each comparison, state the expected effect before generating or classifying the result. Record whether the outcome matches the prediction. Use a temporary profile copy for profile experiments and restore the baseline afterwards.

## Connectedness and cursive joining

Connectedness is not a smooth universal cursive slider. Several gates act before a connector is produced. A profile may change from 0.1 to 0.3 without any synthesis joins appearing. Crossing 0.5 can change fallback letterforms as well as connector eligibility.

| Control or change | Evaluator question | Expected effect and qualifications |
| --- | --- | --- |
| profile['connectedness'] within 0 to 0.44 | Why does increasing connectedness show no joins | Automatic inter-letter joins remain suppressed below JOIN_PRINT_CONN_FLOOR. Changing 0.1 to 0.3 may therefore show no joining difference. |
| Cross JOIN_PRINT_CONN_FLOOR = 0.45 | When does joining begin | Eligible pairs become candidates for joining. Distance, source compatibility, letter type and attachment geometry must still pass. |
| Cross CORE_CURSIVE_CONN = 0.5 | Why did the letter shapes change too | Generic fallback selection can switch towards cursive forms. This changes more than the number of connectors. |
| Increase connectedness above the gates | Can you make this hand more connected | More eligible joins are likely. Even connectedness of 1 does not force every adjacent character to connect. |
| JOIN_PROB_CAP = 0 versus 1 | Show only the contribution of added joins | Zero suppresses probabilistic synthesis joins while retaining the profile's print or cursive classification. This is a cleaner isolation than crossing the profile thresholds. |
| MAX_JOIN_GAP_XH | Stop long unnatural connections | Reducing it rejects longer horizontal reaches between letters. Increasing it permits more distant eligible attachments. Other gates still apply. |

Use the same text, writer and seed with joining capped at zero and then enabled. Compare different profile connectedness values separately. This distinguishes adding connectors from changing fallback style. Source S1 identifies the synthesis implementation.

## Reproduction geometry and variation

These controls change the generated handwriting or layout without neural-network training. They are strong candidates for live explanation because their geometric intent can be inspected directly.

| Control | Evaluator question | Expected effect and qualifications |
| --- | --- | --- |
| profile['slantDeg'] | Make the writer upright and then more slanted | Increasing magnitude increases horizontal shear with height, moving tall strokes more than baseline points. Slant caps and style calibration can limit or modify the result. |
| mmPerXh | Write the same text at two sizes | Larger physical x-height enlarges glyphs and advances, generally increases path length and causes earlier wrapping. Ruling and bounds fitting can override the apparent requested size. |
| lineWidthMm | Fit the paragraph into a narrower area | Narrower width causes more line breaks. It does not simply squeeze every glyph horizontally. |
| profile['wordSpaceXh'] | Separate the words more clearly | Larger values increase word gaps and can cause additional wrapping. They change layout rather than the captured character shapes. |
| MIN_LETTER_GAP_XH and MAX_LETTER_GAP_XH | Make the writing tighter or looser | These constrain the gap used in advancing between stored glyphs. Tightening can increase collisions; loosening can fragment the appearance of words and increase width. |
| COLLISION_MIN_GAP_XH = 0.05 | What prevents overlapping letters | Increasing it pushes overlapping glyph extents farther apart. Reducing it permits closer placement. The guard uses geometric extents rather than semantic understanding of letters. |
| jitter | Make the writing mechanical or variable | Increasing it increases positional, angular and scale perturbations and baseline drift. It does not control only one kind of variation. |
| Position jitter multiplier 0.022 | Change position variation alone | Increasing it increases glyph displacement without directly changing the rotation and scale multipliers. |
| Rotation jitter multiplier 2.2 | Vary individual letter angles | Increasing it increases local angular variation, distinct from the writer's overall slant. |
| Scale jitter multiplier 0.05 | Vary individual character sizes | Increasing it increases local size variation, potentially changing clearance and appearance. |
| Drift persistence 0.85 | Make the baseline wander gradually | Higher persistence retains drift velocity longer and produces more correlated movement. It acts on velocity, not simply on baseline position. |
| Drift bound plus or minus 0.25 x-heights | Keep the writing near its baseline | Smaller bounds restrict wander. Larger bounds permit greater vertical displacement and can reduce clearance between lines. |
| Line clearance addition 0.85 | Increase separation between lines | Increasing the addition in (ascender - descender + clearance) times x-height increases synthesis pitch. Later ruling placement can alter final spacing. |
| seed | Repeat exactly the same generation | Fixes pseudorandom choices for the same inputs and execution path. Changing it changes variants and perturbations. Branch changes can still alter random-number consumption. |
| USE_STYLE_CALIBRATION | Show the uncorrected extracted style | Disabling it removes the automatic shear and extent correction. The visible effect depends on the writer's stored reference measurements and cache state. |

Measured profile style is not always the final visible style. Calibration, clamping, fallback selection and machine placement can all modify it. An unchanged output does not establish that a parameter is irrelevant until those later stages have been checked. See sources S1 and S2.

## Glyph selection and automatic repair

These comparisons explain whether the system is reusing captured handwriting, substituting representative shapes or optimising candidates for the existing classifiers.

| Control or change | Evaluator question | Expected effect and qualifications |
| --- | --- | --- |
| legibility or profile['legibilityLambda'] | Prefer personal variants and then readable fallbacks | Higher anchoring strength generally favours cleaner representative or fallback forms. It can reduce personal variation and changes the permitted slant. It is not a universal vertex interpolation. |
| _AUTO_LAM_GOOD = 0.24 and _AUTO_LAM_BAD = 0.32 | Why does this profile use many anchors | Moving the thresholds changes how median prior distance maps to automatic anchoring. An explicit stored legibilityLambda can bypass the automatic calculation. |
| IDEAL_SOURCE | Use a real representative instead of a prototype | Selects the preferred stored representative, medoid or prototype, where available. Missing or rejected representatives still trigger fallback. |
| IDEAL_ANCHOR_MAX_PRIORD = 0.30 | Accept more of the writer's own shapes | Increasing the limit permits less typical representatives. Decreasing it can cause more generic fallback use. |
| Variant pool for one character | Use only one version of the letter a | Reducing the pool makes the character more repetitive. The retained glyph's quality matters more than the count alone. |
| Remove one character from a temporary profile | What happens when this letter was never captured | Exercises missing-character fallback. The output may remain readable while becoming less specific to that writer. |
| perCharLam | Repair one ambiguous letter | Applies stronger anchoring at selected text positions. Neighbouring joins and spacing can change too. |
| nTries | Generate quickly and then search more | More trials provide more candidate opportunities and cost more inference time. The internal best score cannot worsen if the original candidate set is retained, but physical quality need not improve. |
| repair False or True | What does automatic repair contribute | Disabling it reveals the best initial candidate. Enabling it permits targeted regeneration accepted by the full joint score. |
| repairRounds | Allow more correction attempts | Increases the repair budget without guaranteeing improvement or a global optimum. |
| repairWorstK | Repair one weak position or several | Larger values modify more positions per trial, potentially improving readability while changing more personal geometry. |
| Joint score weighting or form | Prioritise text over writer identity | Requires a small scoring-logic edit. A weighted or alternative combination can select different candidates from the same pool without changing trained weights. |

The current joint objective is the harmonic mean of character score and requested-writer probability. A comparison of text-only selection, writer-only selection and joint selection is meaningful if the candidate pool is held fixed. Changing both the candidate pool and scoring rule obscures the cause of a different winner. Sources S1 and S3 locate the relevant logic.

## Live classification with existing models

Preprocessing, model selection and decision logic are the useful live controls. Recognition accuracy need not change monotonically when one is adjusted. A confident prediction can still be wrong, and confidence is not geometric agreement.

| Control or small edit | Evaluator question | Expected effect and qualifications |
| --- | --- | --- |
| PAGE_NAMER | Compare the ink and shape classifiers | Changes which saved checkpoint judges photographed lines. Labels and confidence can change without retraining. Use that model's corresponding class mapping. |
| PAGE_NAMER_STROKE_NORM | Does identity depend on pen thickness | Enabling normalisation removes much of the original stroke-width information. It can improve robustness or discard useful cues, depending on the checkpoint and image. |
| StrokeNormalize penWidthXh argument | Classify the same skeleton at different widths | Larger widths thicken the reconstructed image and can close gaps. Smaller widths can lose small details after resizing. Changes occur in discrete kernel-size steps. |
| Text checkpoint selection | Compare saved text recognition models | Changes the existing reader and potentially its transcript and confidence. The model object must actually be reloaded; a filename edit alone is insufficient. |
| Ink weighted versus equal line pooling | Why should a long line count more | Equal weighting increases short lines' influence. Ink weighting increases dense or long lines' influence. The page result can change while every line prediction stays unchanged. |
| Lines included in the page vote | What if a crop is poor | Excluding a crop changes the evidence and pooled probabilities. Keep the excluded crop visible rather than presenting removal as an unexplained accuracy gain. |
| Pooling ink threshold currently below 128 | What counts as evidence for weighting | Raising the threshold counts more grey pixels as ink and can change relative line weights. It does not itself alter the classifier input pixels. |
| Whole line versus a shorter crop | How much handwriting is needed | Less content generally supplies less evidence, but confidence is not guaranteed to fall smoothly. Keep crop preparation consistent. |
| Explicit blur contrast or brightness transformation | What happens with a poor photograph | Can reveal recognition sensitivity without retraining. Stronger transformations may erase strokes, close gaps or change background estimates. These are explicit input edits rather than existing universal request sliders. |
| Confidence based abstention | Can the system say it does not know | Requires decision-logic addition. Reject below a threshold instead of always naming the maximum-softmax class. Raising the threshold increases abstention without improving the probabilities themselves. |

The inspected server uses SHAPE_MODEL with stroke normalisation disabled for photographed-page identification. Synthesis has a separate shape-based judging path. Changing one path does not automatically change the other. Model objects are loaded during server startup. See S4, S5 and S6.

## Page segmentation before classification

Segmentation can change the whole classification result before the network sees a crop. Show the mask, centres, seam overlay, line count and final crops. Predict the intermediate effect first, such as retaining more foreground or producing fewer centres, before predicting a change in final accuracy.

| Control | Evaluator question | Expected effect and qualifications |
| --- | --- | --- |
| Background estimation window radius | Handle a stronger local shadow | Smaller windows follow local illumination more closely but can follow ink itself. Larger windows preserve broader structures but miss rapid lighting changes. |
| Ink ratio threshold adjustment | Retain faint handwriting | A higher ratio threshold generally retains more foreground, including noise. A lower threshold can erase faint strokes. Requires a local decision edit rather than a currently exposed request field. |
| Darkness exponent 0.55 | Make the crop darker or lighter | For darkness between zero and one, reducing the exponent strengthens intermediate darkness. Increasing it weakens darkness. The exponent acts on darkness, not directly on brightness. |
| AXIS_TURN_MARGIN = 1.35 | Why did the page rotate by ninety degrees | Lowering the margin accepts weaker quarter-turn evidence. Raising it makes turning more conservative. |
| Deskew range minus four to plus four degrees | Read a more tilted page | A wider search handles greater global tilt at extra work, but may admit other objective maxima. It does not fix perspective or curved baselines. |
| Deskew increment 0.5 degrees | Straighten the page more precisely | Smaller increments provide finer angular search and more computation. They do not guarantee a better estimate on an ambiguous page. |
| Peak floor 0.06 times maximum | Recover a faint line | Lowering it admits weaker peaks and possible false lines. Raising it can drop faint genuine lines. |
| Centre separation 0.6P | Separate two close lines | Lowering it allows closer centres but can split one line's ascender, body and descender bands. Raising it can merge nearby true lines. P is estimated line pitch. |
| mergeValley or LINE_MERGE_VALLEY | Stop splitting one line into several | Lowering the threshold makes shallow-valley merging more aggressive. At 1.0, this merging loop is effectively disabled. |
| PEAK_MIN_PITCH_XH | Avoid mistaking letter height for line spacing | Raising the pitch floor rejects more letter-sized peak gaps, but can reject genuinely tight line spacing. |
| Run limits 1.25P and 1.4P | Remove ruling while keeping handwriting | Smaller limits remove shorter straight structures and may remove genuine underlines or strokes. Larger limits retain more printed marks. |
| Seam ink penalty 4.0 | Make the cut avoid ink more strongly | Increasing it gives closed-mask ink a stronger penalty relative to other costs. The path may make longer detours. |
| Seam midpoint weight 0.012 | Keep boundaries near the middle | Increasing it resists detours. Reducing it allows low-ink routes farther from the midpoint. |
| Seam movement penalty 0.02 | Make the boundary smoother | Increasing it penalises vertical movement more. Reducing it permits locally jagged routes. |
| Caller seam step allowance 7 | Bend around a narrow tall stroke | Increasing it permits steeper row changes between columns. Reducing it limits how quickly the seam can react. The called function's default is not necessarily the active caller value. |
| Closing radius near 0.07P | Stop cuts threading through tiny stroke gaps | Larger closing seals more gaps but can join neighbouring structures. Smaller closing preserves narrow passages. |
| Ownership substantial overlap fraction 0.15 | Why was a fragment assigned to this line | Raising it recognises fewer overlaps as substantial. Lowering it lets smaller overlaps affect reference ownership. This is not simply a more-correction slider. |
| Minimum crop ink | Keep short lines or punctuation | Lowering it retains smaller candidates and more artefacts. Raising it can discard genuine short lines. |
| Solid crop limits 15 percent and three times median | Why was a dark block rejected | Lowering limits makes rejection more aggressive. Raising them retains denser crops. Both conditions must pass; an all-rejected fallback returns the original candidates. |

Equal line counts do not establish correct ink ownership. A change can recover a missing line while corrupting another. Inspect the content of each crop and retain failed cases when interpreting recognition. Sources S7, S8 and S9 identify the implementation.

## Machine placement and physical execution

These controls act at different points from synthesis. Distinguish a generated path, its final machine placement, a preview image and the physical ink. A new generated job is needed for settings stored in G-code; changing a constant does not rewrite a saved job.

| Control | Evaluator question | Expected effect and qualifications |
| --- | --- | --- |
| drawFeedMmMin | Draw the same path more slowly | Changes pen-down command feed. Digital geometry should remain the same; physical time, ink deposition and tracking can change. |
| travelFeedMmMin | Spend less time between strokes | Changes pen-up feed. It should preserve intended drawn geometry, but carriage dynamics still matter. |
| penDownDwellMs | Start before the pen settles | Shorter dwell reduces waiting and can produce weak or missing stroke starts. It does not directly command contact pressure. |
| penUpDwellMs | Travel sooner after lifting | Shorter dwell reduces waiting but can produce travel marks if clearance has not been achieved. |
| joinStrokesMm = 0.6 | Avoid unnecessary pen lifts | Increasing it keeps the pen down over more short endpoint gaps, reducing toggles and adding connectors. This is separate from synthesis connectedness. |
| closeRingMm = 0.6 | Close an almost complete loop | Increasing it closes more near-loops. Excessive values can add unwanted closing strokes. |
| minSegmentMm = 0.12 | Reduce tiny movement commands | Increasing it removes more short moves through decimation and can lose detail. It is not the extraction-stage Douglas Peucker tolerance. |
| Stroke order optimisation | Reduce pen-up travel | Can reorder or reverse strokes while retaining much of the visible geometry. It changes execution order and potentially joins and physical ink effects. |
| pageMarginXmm and pageMarginYmm | Move writing away from an edge | Changes placement and usable space. A narrower usable width can also change wrapping. |
| line or lineOffsetMm | Put the next result below the previous one | Moves the job vertically. It is a placement change rather than a writer-style change. |
| lineHeightMm and rulingXhFrac | Fit a different paper ruling | Changes ruling-based sizing. Line height can affect both letter size and placement. |
| ignoreRuling | Compare natural clearance with ruled spacing | Changes line placement. Tighter ruling can bring adjacent ascenders and descenders together. |
| Bounds fitting behaviour | What happens when text does not fit | Automatic fitting can shrink the output and mask an earlier size increase. Inspect the reported placement and scale. |
| PEN_UP_OVERRUN_S | Why run the cam after the switch changes | Alters extra motor run after confirmed pen-up. It is distinct from the generated G-code settling dwell. |

The server already accepts several request settings, including feed rates, dwells, line placement, ruling behaviour and trial count. Other controls require changing a configuration field or active call site. A command feed is not a guarantee of exact measured speed; software pulse generation and hardware behaviour still affect execution. Sources S10 and S11 locate this path.

## Profile extraction controls that require regeneration

These change reproduction only after rebuilding and reloading the affected profile. They do not require neural-network training, but regeneration may be too slow for a short comparison. Editing these constants while continuing to use an existing profile will usually demonstrate nothing.

| Extraction control | Expected effect after regeneration |
| --- | --- |
| SimplifyPolyline tolerance 0.6 source pixels | Larger tolerance retains fewer vertices and can remove fine curvature. Smaller tolerance retains more raster detail and noise. |
| Junction continuation angle 75 degrees | Larger angles permit sharper chain reconnections. Smaller angles leave more separate strokes. |
| Body band threshold 45 percent | Changes estimated baseline and x-height, affecting many downstream normalised measurements. |
| Cut cost height weight 1.15 | Increasing it favours cuts through lower connectors more strongly. |
| Minimum cell width 0.08 times x-height | Increasing it prevents narrow cells but can force boundaries away from the best local gap. |
| Glyph width limits 2.1 and 2.9 x-heights | Larger limits admit broad valid forms and more multi-letter bad cuts. Smaller limits can reject genuine wide glyphs. Both numbers are width limits. |
| MAX_VARIANTS_PER_CHAR | Larger pools preserve more variety and potentially poor variants. Smaller pools increase repetition. |
| Prior and consensus filtering strength | Stricter filtering can improve consistency while discarding valid personal variation. |
| Smoothing step and passes | More smoothing can remove jitter and flatten distinctive corners or small features. |

Use source S12 for the extraction path. Changing extraction also changes the measurements feeding later heuristics, so existing connectedness or anchoring thresholds may no longer have the same empirical meaning.

## Settings that are not useful ordinary live controls

Training learning rate, batch size, dropout probability, epochs and optimiser do not change ordinary inference from fixed saved weights. A change to a training constant is not a live classification comparison.

CNN channel counts, LSTM hidden size and alphabet order are tied to checkpoint shapes and label semantics. They are not compatible ordinary tuning parameters. Input height and width alter preparation, but also change the input distribution and frame geometry; they are poor first choices for an interpretable live adjustment.

The inspected GantryConfig declares accelMmS2, but the inspected motion path does not consume that field. Do not promise an acceleration demonstration by changing it alone. SQUARENESS_ERROR_DEG is likewise declared without being applied as a motion correction.

Changing letterAdvance alone may have little effect on normal stored glyphs because they use their own width, lead and advance through _Advance. The summary table can matter more for fallback cases.

Render pen width changes a raster preview or judging input. It does not change the physical nib width or contact force. Steps per millimetre, microstepping and measured switch travel encode calibration or hardware configuration; arbitrary edits can change scale without constituting a valid calibration.

## Reloads caches and later stages

The server loads PROFILES, the text model and writer models into memory. Editing JSON or a checkpoint path on disk does not necessarily update those objects. Know whether the active code path needs a reload, restart, cache invalidation or new generated job.

| Dependency | What can go wrong | What to establish before interpreting a comparison |
| --- | --- | --- |
| PROFILES in server memory | A JSON edit has no immediate effect | The active in-memory profile is the intended temporary version. |
| Loaded model objects | A checkpoint filename changes while the old weights remain loaded | The actual model object and its class mapping have been replaced consistently. |
| _STYLE_CAL cache | An edited profile reuses an older correction | The cache key uses author and rendering scale, not every profile value. Recompute the affected correction where required. |
| Profile _gapRange cache | New spacing limits or glyph metadata appear ineffective | The cached gap range has been recalculated for the comparison. |
| Function default arguments | Changing a module variable does not change a bound default | The active call passes the intended value, or the function has been redefined through an appropriate reload. |
| Existing G-code job | New settings do not affect a saved job | A new job was generated with the changed settings. |
| Candidate search and repair | Automatic selection compensates for a deliberate change | The direct comparison uses one candidate and repair disabled before full-system search is restored. |
| Shared pseudorandom generator | A branch change alters subsequent random choices | The seed is fixed, while recognising that this does not guarantee every later glyph choice remains identical. |
| Ruling and bounds fitting | An explicit size or pitch change is rescaled later | Final machine placement and scale are inspected as well as synthesis output. |

For direct explanation, start with one candidate, repair disabled and a fixed seed. Then restore the automatic search to show how the complete system responds. A clear result includes both the local effect and any compensating downstream effect. Sources S2, S3, S4 and S13 locate these dependencies.

## Questions to answer for every requested change

1. Where in the pipeline does this value act, and what are its units?
2. Is it an existing request field, a call argument, a constant, a small logic edit or a profile-regeneration setting?
3. What should increase, decrease or remain unchanged?
4. Which guard, cap, cache or later stage can override the change?
5. What intermediate image, path, crop, label or timing would make the effect visible?
6. Does the comparison keep the model, input, seed and unrelated settings consistent?
7. Does a higher internal score represent independent evidence of a better physical result?
8. What observation would show that the initial prediction was wrong?
9. How is the original baseline restored?

An unchanged final output can be informative: it may reveal a disabled branch, cached value, eligibility gate, quantised kernel, inactive configuration field or downstream compensation. Explain that mechanism rather than repeatedly changing coefficients without checking the active path.

## Code reference index

Paths are relative to the Handwriting Robot repository root. Line numbers refer to the inspected code and can move after edits. These references support the implementation-specific controls; the expected effects are explanations of that code rather than a record of new demo tests.

S1  Software/CleanPipeline3/synthesise.py line 916  SynthesizeText, placement, connectedness gates, variation and layout.
S2  Software/CleanPipeline3/synthesise.py line 707  USE_STYLE_CALIBRATION, _STYLE_CAL and StyleCalibration.
S3  Software/CleanPipeline3/synthesise.py line 1324  SynthesizeJointBestOf and accepted repair.
S4  Software/CleanPipeline3/web/server.py line 178  Loaded models, page writer settings and profile loading near line 234.
S5  Software/CleanPipeline3/web/server.py line 420  NameWriter and ink-weighted page probabilities.
S6  Software/CleanPipeline3/recognise.py line 283  StrokeNormalize; ClassifyImage near line 525.
S7  Software/CleanPipeline3/segment.py line 587  InkMask and background ratio; Deskew near line 778.
S8  Software/CleanPipeline3/segment.py line 892  LineCentres, valley merging and SeamCuts.
S9  Software/CleanPipeline3/segment.py line 1347  DropSolidCrops; SegmentLines near line 1368.
S10  Software/CleanPipeline3/web/server.py line 695  ApiGenerate, request settings, sizing and job generation.
S11  Software/CleanPipeline3/reproduce.py line 499  WriteGcode, loop closure and endpoint joins; configuration near line 37.
S12  Software/CleanPipeline3/style.py line 476  ExtractLineGlyphs; SimplifyPolyline near line 393 and profile construction near line 1072.
S13  Software/CleanPipeline3/synthesise.py line 535  _GapRange and profile spacing cache.

The expanded code theory manual provides the longer derivations and coefficient evidence discussion. Its source is Software/CleanPipeline2/testing/results/CodeTheory/main.pdf, with a delivery copy in output/Code_Theory_Expanded_2026-11-10.
