# Project design records and their provenance

These unpublished records identify the evidence behind in-text project citations.
They are not independent publications or a substitute for missing measurements.

## Hardware clarifications

The author supplied these facts in this report-editing conversation, 31 October–
1 November 2026: installed motors 42BYGH40-01, 40 mm body, 1.8 degree full step,
1.68 A/phase; operation at 24 V; supply indication 0.35 A at an unspecified state,
current limit 2.11 A. Camera supplied specification: IMX179, 8 MP USB autofocus,
78 degree field, 3264 by 2488 (dimensions need checking against actual frame).
ODROID uses its supplied mains adapter, output rating unknown. Dedicated 3.3 V
supply pin powers the pen motor; GPIO supplies its transistor control signal.
No flyback diode fitted; 10 kOhm pull-downs reported. R10 marking is tentative;
Vref is unknown. No peak motor load, verified torque margin or lighting lux is
claimed. The draft's 0.950 kg mass remains to be confirmed.

## Training records supplied by the author

The text reader is `paper_cnn_bilstm_ctc_joint_best.npz`, using Teklia pre-cut IAM
lines and five photographed writers. Exact reader split counts, final epoch and
host are unknown. Final supplied reader and writer weights originated in PyTorch;
NumPy inference and a separately implemented NumPy trainer must not be conflated.

The selected eleven-writer shape model is `author_shape_11_pw170_weights.npz`,
normalisation fraction 0.17, best saved epoch 43, held-out accuracy 0.9935.
The eleven-writer corpus has 1186 lines: 517 IAM plus 669 personal, 1031 training
and 155 holdout. IAM holds out the last labelled page; personal data use a seeded
15 percent line split (seed 0). This is not a wholly independent capture-session
split. Width runs all had a 50-epoch budget, PyTorch on Colab T4:

| Fraction | Best saved epoch | Holdout accuracy |
|---|---:|---:|
| 0.06 | 38 | 0.9806 |
| 0.09 | 43 | 0.9742 |
| 0.11 | 39 | 0.9806 |
| 0.17 | 43 | 0.9935 |

The original 13-output shape checkpoint has different provenance and is not the
eleven-writer sweep winner. Historical learning rates, batch sizes and epochs
cannot be inferred from current defaults or an optimiser's end state.

## Reproduction decision supplied by the author

Available personal data comprise a few offline photographed pages per writer;
there are no measured time-ordered pen trajectories. Avoiding a separately trained
first-principles generator on the ODROID and reusing the observed glyph geometry
motivated the library. No quantitative ODROID generator training time was measured.
Few-shot generators may require little new-writer reference data after broader
training: the report explicitly distinguishes these two quantities.

Final stored glyph libraries are taken as the demonstration inputs. The report
does not describe manual curation as an operating feature. It does not claim that
one general generator needs many pages from every new writer.

## Reproducible subsystem evidence

`Evidence/bench.json` and `Evidence/camera_bench.json` identify input paths,
SHA-256 hashes, configurations and retained successes/failures. Associated scripts
are supplied in this revision. They include independent component, seam, CTC and
G-code checks, original crop alignment, candidate examples and archived captures.
The archived CleanPipeline2 numerical comparison is identified as historical,
not rerun or automatically equivalent to all submitted code.

`EQUATION_AUDIT.json` maps every numbered equation to its implementation/input
basis and adjacent numerical example. `WORKED_CALCULATION_CHECKS.json` retains
unrounded checks of constructed calculations. Chosen example parameters do not
become recorded hardware or trained-model measurements through inclusion here.

## New coefficient sensitivity, 1 November 2026

`Evidence/coefficient_sensitivity.json` records 48 seam computations: eight
settings on the first, middle and last gaps of Owen01 and Yeukita01. The
parameterised reference matches the current default on all six seams. This
post-design study varies midpoint attraction and movement penalty separately;
it neither recovers a historical selection experiment nor establishes a global
optimum. Every setting retains the same two raw-mask intersections. No training,
inference, checkpoint change, coefficient change or hardware execution was needed.
