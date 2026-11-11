# Visual and pseudocode follow-up

The following prior coverage audit still applies. This pass adds 11 total
line-numbered pseudocode blocks (eight design procedures and three independent
verification procedures), fills six figure slots and adds two appendix figures.
New evidence is recorded in `Evidence/bench.json` and `Evidence/camera_bench.json`.
It includes failures and explicitly separates software inference, synthetic
perturbations, archived real captures and existing physical robot output.
The code's current `rulingMaxExtent` default is 1.25; the previously stated 1.60
was corrected in the Design. No pipeline code, model or final profile was changed.

# Design coverage audit — second pass, 1 November 2026

## Conclusion and scope

The revised chapter covers the project's principal design paths and the substantial
operations behind them. It is a materially stronger engineering explanation than
the first pass. Coverage does not establish submission readiness: complete figures
are assumed for this review, and the factual/evidence gaps below still matter.
No grade is predicted. No training, model inference or hardware execution was used
to validate performance during this report revision.

The review used the complete Design chapter; Appendix 4's Design requirements and
relevant adjacent guidance; the approved-proposal snapshot; the Design material in
Sreekumar's u23531003 report and Aston's unfinished report; the code-explanation
index and relevant detailed explanations; and the current main code paths listed
below. Current source takes precedence over old theory text when they disagree.
This is a report-to-code coverage audit, not a proof of correctness for every line
of repository code or a cover-to-cover audit of the 335-page historical manual.

## Assessment and reference-report check

| Criterion | Treatment in revised package | Remaining condition |
|---|---|---|
| Mandatory 3.1, three-column task summary | Preserved; maps to proposal deliverables and tasks | Confirm completion labels at submission, including verification tasks |
| Mandatory 3.2 final system overview | Hardware and processing diagrams with system narrative | Complete and check actual figures |
| Clear engineering storyline | Page → line → text/writer; known transcript → glyph profile; new text → machine | Retain this chain during final compression |
| Core proposal work receives technical detail | Segmentation, numerical networks, extraction, synthesis and motion explicitly explained | No requirement-compliance claim without Results evidence |
| Mathematics beyond code descriptions | Image criteria, seams, CTC, derivatives, glyph frames, candidate score, calibration, torque and timing | Fill installed electrical values/measurements |
| First-principles contribution is distinguishable | Numerical operations, caches, derivatives and optimiser calculations; framework role stated | Final supplied weights were PyTorch-trained; do not reinstate old no-library claim |
| Complete circuit with component values | Hardware prose and final circuit slot, latest supply details incorporated | Verify Vref, Rsense, motor load, adapter rating, protection and decoupling |
| Readable process descriptions and short algorithms | Eight numbered algorithms and representation-contract table | Executable source should be submitted separately; do not claim AMS upload before it occurs |
| Sreekumar's useful presentation practices | Worked calculations, explicit assumptions, implementation checks and meaningful trade-offs | Do not borrow another project's numerical results or reasons for decisions |
| Aston's useful presentation practices | Real stored glyph example, staged algorithm explanations and linked equations | Draft placeholders/unverified measurements in Aston's report are not authorities |
| Appendix supports rather than hides core design | Derivative conventions, optimisers, thinning and selected field meanings moved to appendix | Import the appendix with the Design source to preserve references |
| Design verification separate from Results | Numerical check and width-selection validation are local design evidence | Formal accuracy, stylistic correlation, runtime and QTP tables remain in Results |
| Honest weaknesses and engineering remedies | Open-loop error, pulse scheduling, missing diode, data separation, library/context loss, internal judge bias | Actual reasons for retaining limitations cannot be invented |

Appendix 4 PDF pp. 12–16 are the main Design authority. Its example software heading
does not require reorganising this user's accepted three-stage explanation into a
single generic Software design subsection. The main section heading, 3.1 and 3.2
remain as required. Pictures do not replace calculations or measured verification.

## Current-code coverage map

| Module and important function families | Where explained | Review outcome |
|---|---|---|
| `segment`: Integral, BoxSum/Mean, morphology, Otsu | 3.4.2; implementation appendix | Prefix statistics and ratio threshold criterion retained |
| `Resize`, `Halve`, `FitFixedCanvas`, Rotate/Warp/Elastic | 3.4.1/3.4.5; 3.5 training | Geometry, padding and augmentation roles covered; no need for every interpolation loop in main text |
| `PaperRegionMask`, RemovePrintedStructure, DropArtifactPieces | 3.4.2; segmentation algorithm | Paper/rule/artefact stages distinguished |
| LineAxisScore, UprightPage, Deskew | 3.4.3 | Quarter turn, bounded skew and reader-based 180-degree choice separated |
| LinePitch, LineCentres, LeastCostSeam, SeamCuts | 3.4.4 | Scale estimate, centre selection and dynamic programme explained |
| LabelComponents, PieceBounds, AssignOwnership and helpers | 3.4.5; ownership algorithm | Eight-connectivity, substantial-band reference, large-piece preservation, dots/rings/fragments corrected from broad first-pass account |
| `_BuildLineCrop`, DropSolidCrops | 3.4.5 | Full-resolution recovery, nearest ownership flags, contrast and dual rejection guard added |
| `recognise`: numerical layers, LSTM and two model classes | 3.5.2–3.5.4; appendix | Tensor shapes, heads, recurrent state and inference statistics covered |
| StrokeNormalize | 3.5.4 | Quantisation and actual selected 0.17 configuration distinguished from default |
| DecodeCtc, CtcForcedAlign, FrameXToPixel | 3.5.3; 3.6.1 | Collapse order, repeats, max versus sum, and inverse resize explained |
| CharAcc, Levenshtein, ClassifyImage | 3.6.6; 3.7.1 | Edit score, writer posterior and page vote are different measures |
| `train`: derivatives, loss and caches | 3.5.3/3.5.5/3.5.6; appendix | CTC occupancy, convolution overlap-add, BN/LSTM derivatives and pooling conventions covered |
| RMSprop, AdamW, schedules, ClipGradNorm | 3.5.5; appendix | Defaults are implementation settings, not recovered historical training settings |
| Data views, joint batching, split builders, augmentation | 3.5.1/3.5.5 | Writer split counts and reader corpus distinguished; unknown reader counts retained |
| `trainlibrary`, parameter export | 3.5.6 | Production weights explicitly framework-trained; CTC reduction mismatch stated |
| `strokenorm_prepare`, `strokenorm_train` | 3.5.4 | Cached deterministic preparation, repeated random augmentation and four-width study added |
| `style`: graph tracers, spurs, rejoining, SimplifyPolyline | 3.6.2; appendix | Corrected default: split THEN prune/rejoin; alternate tracer not claimed active |
| ExtractLineGlyphs, cut cost, crossbar/dot ownership | 3.6.1/3.6.2; profile algorithm | Source transcript, ordered cuts, whole-line thinning and bar exception covered |
| ShapeGrid, prior, consensus, medoid, prototype | 3.6.3; appendix | Distinct distances, actual medoid and robust compatible prototype explained |
| RenderRefStats, ExtractAuthorRaw, BuildAuthorProfile | 3.6.3/3.6.4 | Measurements and raw-cache versus final-profile roles retained |
| Historical `curate` tooling | Private source review only | User will remove the operating feature; current final profiles are taken as given in the report |
| `synthesise`: variant/core selection, AutoLegibility, advance | 3.6.3/3.6.5 | Anchoring, rank/entry weighting and ambiguous core source tag added |
| SynthesizeText, layout, jitter, body/join rules, StyleCalibration | 3.6.5 | Baseline drift, collision/joins and measured-probe calibration retained |
| RenderTrajectory, WeakChars, SynthesizeJointBestOf | 3.6.6 | Actual harmonic score, bounded search and accepted-repair state shown |
| SynthesizeLegible | 3.6.6 | Reader-only helper's print escalation not confused with joint route |
| `reproduce`: XhForRuling, TallestLine, ToMachine | 3.6.7 | X-height/pitch, robust extents, ruled/free spacing, offsets and fit added |
| OptimizeStrokeOrder, Decimate, Clamp, WriteGcode | 3.6.7 | Row-constrained ordering, physical thresholds, modal feeds, clipping and nominal timing covered |
| G-code parsing, linear execution and optional arc input | 3.6.7; 3.3 drive | Generated writing uses linear segments; arc support is an interpreter capability, not an essential synthesis stage |
| Connect, Calibrate, switch debounce, BresenhamMove, SetPen | 3.3 | Electrical roles, count calibration, interpolation and actual delay/pen control explained |
| Abort, end-stop fault, recovery and file execution | 3.7.2 | Software stopping distinguished from independent power interruption |
| `retrace`: PageStrokes, PruneSpecks, RetracePage | 3.7.3 | Diagnostic bypass added; not falsely described as isolating mechanics from all image processing |
| `web/server`: ingestion, ReadAllLines, NameWriter, generation and jobs | 3.4.3; 3.7 | Orientation reuse, ink-weighted pooling, file/job separation and interfaces covered |
| `web/camera`, `timing`, `paths` | 3.3.6; 3.7 | UVC/frame settling, stage timings and writer-map consistency explained; no UI/database credit padding |

Ancillary CLI parsing, HTTP response formatting, plotting scripts, cache byte
encodings and every low-level variable do not each warrant a main-text subsection.
Their engineering roles are covered through these operations; a function catalogue
would add length without demonstrating understanding.

## New concrete material

- Eight algorithms: coordinated stepping; page segmentation; ink ownership; CTC
  loss/gradient; numerical training batch; profile construction; joint search and
  repair; trajectory-to-machine preparation.
- Real stored Dylan `a` example: nine final variants, four displayed alongside
  representatives, without an initial/final comparison or manual operating feature.
  Actual geometry is read from stored data; source hashes accompany the figure.
- Stroke-normalisation illustration rebuilt at 0.17 using existing real crop segments.
- Derived examples labelled as examples: 8:3 interpolation, CTC repeated letters,
  nominal x-height 3.864 mm, first-convolution patch storage and job-time budget.
- Supporting appendix with derivatives, optimiser updates, thinning, selected field
  meanings and coordinate conventions.

## Material facts still needing evidence

1. Final approved proposal copy and completion entries; computer used for historical
   reader training/profile extraction versus the intended embedded operation.
2. Reader train/validation counts, epochs, actual historical hyperparameters and
   source-line separation. A writer holdout does not exclude the same line from
   a separately sourced reader corpus. Earlier handover reports overlap; reconcile
   manifests before any final independent-performance claim.
3. Final installed reader/writer/checkpoint/preprocessing identity. Report the selected
   eleven-writer configuration without claiming that old web defaults validate it.
4. Matched final NumPy/framework learning comparison, if such performance is claimed.
   Fixed-weight verification uses matched loss scaling; current trainers differ.
5. Original-versus-shape comparison controlling writer population and inputs; the
   eleven-class width sweep alone does not prove an advantage over an older
   thirteen-class ink or shape model.
6. Driver Vref, confirmed sense markings, cooling and decoupling; pen-motor load,
   3.3 V capacity, transistor-base pull-down and ODROID adapter rating.
7. Valid axis load/friction/torque data and dimensional calibration measurements.
8. Actual constraints behind unimplemented remedies, especially motor suppression,
   pulse scheduling, alignment/context improvements and source-data limitations.

Training metadata and sweep values supplied by the user on 1 November have been
incorporated as records, not remeasured. No `.pt` files were deserialised to execute
their contents, and no checkpoint was changed. The selected sweep model's use in
the eleven-writer evaluation is consistent with the existing project handover;
historical shipped files and selected evaluation files are not interchangeable.

## Literature check made during this pass

PyTorch's official CTCLoss documentation confirms target-length normalisation for
default `mean` reduction and the `zero_infinity` option:
https://docs.pytorch.org/docs/2.15/generated/torch.nn.CTCLoss.html
The checked convention is cited in the chapter. The subsequent sourcing pass is
documented in `SOURCE_AUDIT.md`: 16 external sources inspected for relevant claims,
eight references explicitly flagged for manual checking, and an internal code
record. The current baseline/revised main-report union is 40 cited keys. This is
not a blanket audit of sources used elsewhere or every unused bibliography entry.
