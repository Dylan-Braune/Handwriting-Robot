# Coefficient justification and parameter sweeps

Every numerical constant in the submitted code, with its basis, its reason, its
alternatives, a worked example, and — where it can be exercised offline — a
five-value sweep with a picture of what changes.

**The submitted code is never modified.** A sweep reads
`Software/u23539195_BrauneDR_Code/*.py`, rewrites exactly one token into a
throwaway copy in the scratch directory, and runs that copy in its own process.
A SHA-256 digest of all seven submitted files is checked before and after every
batch and the run fails if any file has changed. At the submitted value nothing
is rewritten at all, so the panel marked *submitted value* in every figure is
the submitted code itself.

## The document

`latex/main.pdf` — build it with `./build.sh`.

## Live demonstration

From `sweep/`:

```
python demo.py                          # list every coefficient
python demo.py darkness_gamma --info    # what it is and why, runs nothing
python demo.py darkness_gamma           # run its five recorded values
python demo.py darkness_gamma 0.2 0.9   # run the submitted value plus these
```

`demo.py` prints the justification, runs the values, writes
`figures/<key>.png`, and opens it.

To change a coefficient's standing sweep values permanently, edit its `values`
list in the matching `sweep/coeffs_*.py` and re-run `python run.py <key>`.

## Everything else

```
python run.py --list        # the keys
python run.py               # every sweep (about an hour)
python run.py <key> ...     # named sweeps only
python validate.py          # every patch target is unique and the default is a no-op
python checkpatterns.py     # the wider set of candidate tokens, 212 of them
python checkcites.py        # every \cite key exists in the bibliography
python archfacts.py         # parameter counts, receptive field, CTC columns, splits
python mkdoc.py             # regenerate the LaTeX from results/
```

## Layout

| path | what it is |
| --- | --- |
| `sweep/coeffs_*.py` | the registry: one entry per coefficient, with its patch and its written justification |
| `sweep/patch.py` | builds the throwaway patched copy; verifies the submitted code is untouched |
| `sweep/worker.py` | runs one pipeline stage inside one patched copy |
| `sweep/run.py` | drives a sweep and composes the figures |
| `sweep/compose.py` | every panel is drawn here, so the study has one look |
| `sweep/figstyle.py` | Times New Roman, 10 pt floor, black/silver/white/blue |
| `sweep/probes.py` | the fixed page, line and sentence every sweep uses |
| `figures/` | `<key>.png` five-panel figure, `<key>_trend.png` metric against value |
| `results/` | `<key>.json` every run, its metrics, and the source digests |
| `source_snapshot/` | the submitted code as it stood when this study was made |
| `latex/` | the document |

## What this establishes, and what it does not

It shows what each coefficient does and that the submitted value is reasonable.

It does **not** show any value is optimal. These are one-at-a-time sweeps on a
small number of fixed inputs; they do not explore interactions, and a sweep on
one page is not a measurement over the dataset. Where a number comes out better
away from the submitted value it is reported as it stands.

No network was retrained, no deployed style profile was overwritten, no G-code
was sent to the machine and the gantry was not run. Coefficients that need
retraining or hardware are documented, and the reason they are not swept is
stated rather than filled in with a plausible-looking experiment.
