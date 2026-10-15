# Code Explainer — Index

Four deep, function-by-function explainers (purpose / inputs / outputs / maths / literature
connection, per function) covering every file in the live pipeline plus the training code, written
to support understanding your own pipeline and cross-referencing it against the literature review.
Total ~4,000 lines across the four files. Read alongside `ReportContentPlan.md` for how this maps
into the report's Literature Study and Design & Implementation sections.

| Block | File | Covers | Key worked examples / literature hooks |
|---|---|---|---|
| A — Segmentation | [CodeExplained_A_Segmentation.md](CodeExplained_A_Segmentation.md) | `SegmentLean.py` (production), `SegmentPage.py` (legacy, still used by `BuildStyleProfile.py`'s personal-author path) | Otsu 1979, DP seam = seam-carving lineage (Avidan & Shamir 2007), union-find CC (Rosenfeld & Pfaltz 1966 / Tarjan 1975), summed-area tables (Crow 1984), the "express" thin-stroke bug fixed via closing-based cost map, explained mathematically |
| B — Training & Classification | [CodeExplained_B_TrainingClassification.md](CodeExplained_B_TrainingClassification.md) | `TrainText.py`, `TrainAuthor.py`, `ExtractIAMLines.py`, `authors_config.py`, `np_inference/{weight_io,layers,text_model,author_model}.py` | CTC (Graves et al. 2006), LSTM (Hochreiter & Schmidhuber 1997), BiLSTM (Schuster & Paliwal 1997), BatchNorm (Ioffe & Szegedy 2015), im2col/GEMM conv (Chellapilla et al. 2006), the frozen-backbone writer-ID bug as the central worked example |
| C — Reproduction & G-code | [CodeExplained_C_ReproductionGcode.md](CodeExplained_C_ReproductionGcode.md) | `BuildStyleProfile.py`, `ProfileIO.py`, `SynthesizeHandwriting.py`, `server.py`, `Evaluate.py`, `web_render_helpers.py`, `camera_capture.py` | Douglas-Peucker (1973) closed-curve collapse bug, CTC forced alignment vs. greedy decoding distinction, Zhang-Suen skeletonization, harmonic-mean best-of-N scoring derivation |
| D — Gantry Hardware | [CodeExplained_D_GantryHardware.md](CodeExplained_D_GantryHardware.md) | `odroid_direct_drive.py` | Bresenham 1965 (exactly on-topic — plotter control), G-code dispatch, homing state machine, pen-lift pulse timing |

## Things flagged by the agents worth acting on before/while writing the report

1. **No off-hardware simulation path** (Block D): `gpiod=None` raises `GantryNotConnectedError` rather
   than falling back to a mock GPIO — if your report's Discussion claims testing happened "off
   hardware," that needs rephrasing; testing without the physical gantry means the G-code
   generation/round-trip path (`GcodeRoundTrip` in Block C), not a simulated motor driver.
2. **`run_gcode_file` position-tracking inconsistency** (Block D): local position tracking uses
   nominal (not calibrated) steps/mm and doesn't clamp — doesn't affect real motion (motor control
   always recomputes from true step counters) but could drift in relative-mode edge cases. Candidate
   for Discussion §5.1.3 "Unsolved problems."
3. **`authors_config.PERSONAL_AUTHORS` count ambiguity** (Block B): lists 5 entries vs. 2 described
   elsewhere as part of the live 10-author set — reconcile against the actual live checkpoint before
   writing Table 2's "10 users stored" row (ties to `ReportContentPlan.md`'s open item 4).
4. **G20 (inch units) is parsed but not actually converted** (Block D) — a real functional gap, not
   just missing documentation; decide whether to fix, or note as a known limitation.
5. **No velocity ramping/acceleration profiling** (Block D) — plain fixed-delay stepping; legitimate
   "Suggestions for future work" (§6.4) item.
