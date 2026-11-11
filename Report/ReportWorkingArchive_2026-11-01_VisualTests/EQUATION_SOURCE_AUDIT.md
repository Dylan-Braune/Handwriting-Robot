# Equation and source audit

Citations in the report contain only a source and a verified page locator, or only the source where a locator is unavailable. A research citation is not added to an invented project formula merely to make it look externally validated. Derived geometry, units and implementation rules remain explicitly identified.

The author-hosted LeCun PDF is numbered 1-46; its consulted pagination (for example pp. 6-8) is used, rather than guessed offsets into the journal edition. Bradley, Graves, Haines and similar author copies likewise use the consulted document pagination. Source-page and version notes are kept here rather than in the report.

| Equation label | Basis | Supporting source or derivation |
|---|---|---|
| `eq:res-char` | Published error convention | Graves et al. 2006 p. 2; the report explicitly defines its complement as character accuracy. |
| `eq:res-apex` | Project geometric derivation | Subtracting the two squared-distance equations yields the stated apex offset; no experimental cause is inferred solely from it. |
| `eq:des-pulley` | Derived transmission geometry | Stepper transmission principles: acarnley2002; tooth pitch and circumference applied to the installed pulley. |
| `eq:des-stepres` | Derived command resolution | A4988 datasheet p. 6 and the installed 200-step motor and transmission. |
| `eq:des-rpm` | Dimensional derivation | Feed divided by travel per revolution, using the preceding transmission equation. |
| `eq:des-torque` | Derived mechanical load model | Standard force, torque and rotary-inertia relationships; acarnley2002. No verified page available. |
| `eq:des-pulsetime` | Project timing model | Derived from the actual separate-axis pulse calls and speed settings; not a formula borrowed from Bresenham. |
| `eq:des-basecurrent` | Standard circuit relationship | ROHM low-side transistor article; voltage drop across the base resistor. Source has no pages. |
| `eq:des-calibration` | Project calibration definition | Measured counts divided by measured travel; absolute counts use the defined origin offset. |
| `eq:des-scaleerror` | Derived sensitivity | First-order uncertainty propagation, JCGM 100:2008 pp. 18-19, applied to the project scale model. |
| `eq:des-current` | Published component equation | A4988 datasheet p. 9. |
| `eq:des-ratio` | Adapted image operation | Local-mean thresholding principle: Bradley and Roth pp. 2-3; ratio image and histogram threshold are the project adaptation. |
| `eq:des-integral` | Published method in project indexing | Bradley and Roth p. 2; exclusive-prefix convention is stated in the report. |
| `eq:des-otsu` | Published method | Otsu 1979. Source cited without an unverified page. |
| `eq:des-skew` | Adapted projection criterion | Projection-profile skew principle: Postl 1986; squared-profile objective and search grid are the implemented choices. |
| `eq:des-seam` | Adapted dynamic program | Avidan and Shamir p. 3; local cost, movement limit and penalty are project choices. |
| `eq:des-seamcost` | Project heuristic | Explicit engineering cost whose coefficients are examined in the retained sensitivity experiment. |
| `eq:des-fit` | Derived geometry | A common scale must satisfy both height and width bounds; their minimum preserves aspect ratio. |
| `eq:des-conv` | Standard network operation | LeCun et al. author version pp. 6-8; cross-correlation and patch-matrix conventions stated separately. |
| `eq:des-bn` | Published method | Ioffe and Szegedy p. 4. |
| `eq:des-lstm` | Published recurrence | Graves 2013 p. 5. |
| `eq:des-ctc` | Published objective | Graves et al. 2006 pp. 2-3. |
| `eq:des-ctcgrad` | Adapted derivative convention | Graves et al. 2006 pp. 5-6; occupancy and batching conventions are stated. |
| `eq:des-strokenorm` | Established operations plus project width | Zhang-Suen thinning p. 237 and morphology in Haralick p. 533; 0.17 width is empirical project selection. |
| `eq:des-viterbi` | Adapted alignment recursion | Graves et al. 2006 pp. 4-5; maximisation replaces path summation for known-transcript alignment. |
| `eq:des-cutcost` | Project heuristic | Implemented low-ink and baseline preference, illustrated with a real extracted glyph. |
| `eq:des-glyphcoords` | Project coordinate convention | Shear normalisation principle: Vinciarelli and Luettin; origin, baseline, scale and post-trim lead are project definitions. |
| `eq:des-connectedness` | Project descriptor definition | Counted skeleton crossings divided by eligible adjacent pairs; Bulacu motivates shape descriptors but does not supply this exact rule. |
| `eq:des-placement` | Derived inverse transform | Inverse of the project glyph-storage transform with the stated physical scale and placement offsets. |
| `eq:des-anchor` | Project blend definition | Configured interpolation between variants and an anchor; not a universal learned-model rule. |
| `eq:des-jointscore` | Published error convention plus project objective | Graves et al. 2006 p. 2 supports edit error; using a harmonic mean with writer posterior is the project selection rule. |
| `eq:des-machinecoords` | Project coordinate convention | Page bounds, margins, axis direction and uniform scale explicitly determine the affine transform. |
| `eq:des-timebudget` | Dimensional job model | Distance divided by feed, plus configured pen-transition dwells; not a measured execution-time equation. |
| `eq:des-pagevote` | Project fusion rule | Ink-weighted mean of line posteriors, followed by the largest mean. Its weights are defined and a real vote is calculated. |
| `eq:app-bnback` | Derived published-method derivative | Ioffe and Szegedy p. 4, with the chain-rule simplification stated. |
| `eq:app-lstmback` | Derived recurrence derivatives | Graves 2013 pp. 4-5 and the implemented forward LSTM equations. |
| `eq:app-ctcforward` | Published recurrence in log space | Graves et al. 2006 p. 4. |
| `eq:app-rmsprop` | Adapted optimiser | Tieleman and Hinton p. 29; the code coefficients and coupled decay are explicit. |
| `eq:app-adamw` | Published optimisers | Kingma and Ba p. 2; Loshchilov and Hutter p. 3. |
| `eq:app-thinning` | Published method | Zhang and Suen p. 237. |
| `eq:des-convback` | Derived network derivatives | Matrix convolution and the stated chain rule; Chellapilla and LeCun sources support the underlying operations. |
| `eq:des-gradcheck` | Derived central difference | The two Taylor expansions are shown; JCGM pp. 18-19 supports finite-difference sensitivity evaluation. |
