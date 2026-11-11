# Author completion list and source checks

These items were removed from the typeset report at the author's request. Removing a note does not establish the missing measurement or fact. Existing limitations and the scope of the reported evidence remain explicit.

## Records and figures removed from the manuscript

### 1. Author record or check: 3_Main_Report\5_Discussion.tex

Insert the university's confirmed ethics-clearance or exemption statement and reference. Participant consent is confirmed; do not infer institutional clearance from the signatures.

### 2. Figure: 3_Main_Report\design\3_3_hardware.tex

\begin{figure}[!htbp]\centering
\designasset{design-gantry-assembly.jpg}{0.85}{Photograph or assembled CAD view with the custom brackets, cross-shaft, belt tensioning, rail bearings and paper restraint labelled.}
\caption{Mechanical assembly and the components locating the pen relative to the page.}
\label{fig:des-assembly}
\end{figure}

### 3. Figure: 3_Main_Report\design\3_3_hardware.tex

\begin{figure}[!htbp]\centering
\designasset{design-complete-circuit.pdf}{0.98}{Complete circuit with GPIO offsets/physical pins, 24 V motor rail, ODROID supply, pen supply, driver logic rails, switch resistors and protection components.}
\caption{Complete electrical circuit and power distribution.}
\label{fig:des-complete-circuit}
\end{figure}

### 4. Figure: 3_Main_Report\design\3_3_hardware.tex

\begin{figure}[!htbp]\centering
\designasset{design-camera-arrangement.jpg}{0.70}{Photograph of the actual camera and page mount, labelled with working distance, page area and illumination direction.}
\caption{Page capture arrangement used to supply images to the processing pipeline.}
\label{fig:des-capture}
\end{figure}

### 5. Author record or check: 3_Main_Report\design\3_3_hardware.tex

Confirm the 0.950\,kg mass and which axis it represents; measure carriage resistance and obtain the torque--speed or starting-rate data for the installed 42BYGH40-01 at the actual driver setting. A curve for a different motor cannot establish its torque margin.

### 6. Author record or check: 3_Main_Report\design\3_3_hardware.tex

Measure pen-motor running/startup current and confirm the 3.3 V rail's available capacity. The user reports 10 kOhm pull-downs; confirm whether one is fitted at the transistor base as well as at switch inputs. Replace the prototype image with correct DC notation and actual installed components. Record the reason the motor-suppression diode was omitted before final submission.

### 7. Author record or check: 3_Main_Report\design\3_3_hardware.tex

Confirm each driver's sense-resistor marking, measured Vref and cooling arrangement; record the ODROID adapter's DC output rating, the operating state during the 0.35 A reading, and fitted decoupling.

### 8. Author record or check: 3_Main_Report\design\3_3_hardware.tex

Verify the supplied camera specification against its label and saved frame dimensions; add working distance and mounting dimensions. Describe only the lighting arrangement actually used; do not claim a measured lux range without a measurement.

### 9. Author record or check: 3_Main_Report\design\3_5_classification.tex

Record reader split counts and the source pages used to construct final glyph profiles. Keep the reported writer holdout distinct from a wholly independent capture-session test.

### 10. Author record or check: 3_Main_Report\design\3_5_classification.tex

Add a matched ink-versus-shape design comparison when available; do not compare different writer populations as if only preprocessing changed. Verify the selected checkpoint's preprocessing in the final submitted configuration.

### 11. Author record or check: 3_Main_Report\design\3_5_classification.tex

Confirm applicability of the archived fixed-parameter comparison to the submitted implementation. The new common-material trained-model comparison is given in Table~\ref{tab:dis-training}; it is distinct from a controlled equal-initialisation training experiment.

### 12. Figure: 3_Main_Report\design\3_6_reproduction.tex

\begin{figure}[!htbp]\centering
\designasset{design-glyph-quality.pdf}{0.91}{One real writer's character variants, identifying retained forms, neighbour-contaminated or fragmented cuts, the medoid and any accepted denoised prototype. Include the rejection criterion.}
\caption{Example of glyph-library filtering and representative selection.}
\label{fig:des-glyphquality}
\end{figure}

### 13. Figure: 3_Main_Report\design\3_6_reproduction.tex

\begin{figure}[!htbp]\centering
\designasset{design-glyph-joining.pdf}{0.94}{Enlarged real glyph pair with body, dot/crossbar, entry and exit labelled; show accepted connector and one rejected large-gap or incompatible-source case.}
\caption{Assembly of eligible glyph bodies into connected pen paths.}
\label{fig:des-joins}
\end{figure}

### 14. Author record or check: Appendices\design_implementation_details.tex

Confirm the completion entries against the final submitted deliverables, and identify the computer used for training and profile construction. The proposal's embedded-processing requirement must remain distinguishable from the actual training arrangement.

### 15. Physical overlay experiment: Appendices\results_evidence.tex

\subsection{Planned repeated-word and physical overlay record}
\label{app:res-overlay}
\begin{figure}[!htbp]\centering
\designasset{results-repeated-word-overlay.pdf}{0.90}{Two executions of the same word using fixed glyph choices, seed, size and layout, shown against one coordinate reference. Include the original scans, scale, point-registration method, maximum measured displacement, repeated-draw count and measurement uncertainty. This is a pending measurement, not a predicted passing result.}
\caption{Reserved repeated-word overlay required to extend the gantry qualification.}\label{fig:res-overlay-pending}\end{figure}
The overlay should keep machine placement visible: unconstrained image translation or warping would hide the absolute-position error being tested. A separate original-versus-reproduced handwriting illustration should state whether the text, scale and baseline are matched, and distinguish qualitative resemblance from a measured geometric metric.



## Sources without verified page locators

The report cites these sources without a page number where no verified locator was available. Check the relevant edition or paper manually; no page has been invented.

- `acarnley2002`
- `bc337ds`
- `chellapilla2006`
- `douglas1973`
- `marti2002`
- `otsu1979`
- `postl1986`

## Other final records

- Institutional ethics-clearance or exemption reference, if applicable.
- Reconciled budget, printed-part material, caliper make/model/least count, and measured lighting record.
- Final whole-report assembly, including front matter and Conclusion, and its overall 70-page main-text budget.
- Missing figures are excluded rather than presented as empty or completed figures. Their labels and instructions are preserved above.

## Citation-only sources retained without a page

`acarnley2002`, `bc337ds`, `chellapilla2006`, `douglas1973`, `he2020`, `iqs2024`, `marti2002`, `odroidn2ds`, `otsu1979`, `pascanu2013`, `postl1986`, `pytorchctc`, `rohmlowside2026`, `vinciarelli2001`, `wigington2017`.

Where an article was checked only through its abstract or bibliographic record, no precise page is supplied. In particular, obtain the cited edition for Acarnley, BC337, Chellapilla, Douglas-Peucker, Otsu, Postl, Marti, Vinciarelli and the IQS technical page before adding locators.
