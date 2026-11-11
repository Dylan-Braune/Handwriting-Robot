# Sweep and coefficient-study inventory

## Included with adequate records

1. **Stroke-normalisation training width:** 0.06, 0.09, 0.11, 0.17; same
   eleven-class population/holdout, 50-epoch budgets, best saved epochs 38/43/39/43,
   validation accuracies 0.9806/0.9742/0.9806/0.9935. Supplied training records
   identify Colab T4/PyTorch. The selected width is 0.17; validation selection is
   not independent final test performance. Main Design contains the table.
2. **New seam sensitivity:** midpoint weights 0, 0.006, 0.012, 0.024, 0.048 at
   movement penalty 0.02; movement penalties 0, 0.01, 0.04 additionally at midpoint
   weight 0.012. Eight settings, six fixed gaps, 48 runs. Exact default agreement
   on six paths; input/code hashes and all outputs retained. Main explanation
   plus detailed appendix table. This did not determine the historical settings.
3. **Conversion fixtures:** three offsets × three geometric scales, nine cases.
   This checks coordinate serialization across sizes, not coefficient selection.
   The existing evidence appendix retains vertex and clipping/error outcomes.
4. **Capture/perturbation cases and three fixed synthesis candidates:** useful
   subsystem illustrations, not a hyperparameter search. Their settings and
   failures remain explicit; they are not labelled tuning studies.

## Retained artifacts requiring more metadata before use as quantitative evidence

- `Software/CNN/NOGIT/EndToEnd/nsweep_abhinav_n01_*` through `n05`, `n10`,
  `n20`, `n30`, and `n1520_abhinav_N15_*` / `N20_*`: actual rendered files exist,
  but no associated transcript/checkpoint/configuration/score log was recovered
  in that directory's text/JSON/Markdown/Python records. Filenames alone do not
  establish a controlled nTries study or justify a selected search budget.
- `Software/CleanPipeline/NOGIT/experiments/cursive_anchor/px_sweep.py` and
  `new_results/px_sweep_yeukita.png`: historical rendering-resolution experiment
  using older module names and models. Its apparent motivation must not be
  transferred to the final eleven-writer / 0.17 pipeline without recovering the
  actual output table and exact run configuration.
- Historical legibility-level/anchor images and investigation notes use earlier
  profiles and populations. A script's existence or an agent's interpretation
  does not establish a final controlled coefficient-selection result.

These artifacts were not fabricated into new result tables or silently presented
as current configuration evidence. Recover their logs/settings before inclusion.
