# measurements/

The audit trail for every number in Section 4 (Results) and Section 5
(Discussion) of the final report.

**Start with [`RESULTS_PROVENANCE.md`](RESULTS_PROVENANCE.md).** For each result
it gives what was measured, the script that measured it, where the input data
lives, what came out, what it means, and the table or figure in the report that
carries it.

```
RESULTS_PROVENANCE.md   the document to read
scripts/                23 measurement scripts, copied out of the working
                        session so they survive it
outputs/                the JSON and logs those scripts produced; these are
                        what make_results_figures.py reads
raw/                    odroid_server_log.txt -- the ODROID N2+ server log,
                        the primary record for R3 and R4
```

Scripts expect to be run with `python -I` from the repository root, and they
read the live data paths named in the provenance document rather than copies.

Two warnings, both set out in full in §13 of the provenance document:

1. **Three of the checkpoints behind these results live only in
   `C:/Users/braun/Downloads/`**, and one is shadowed by a different file of the
   same name inside the repository. SHA-256 values for the files actually used
   are in §13a. They should be copied into version control.
2. **The configuration measured is not the configuration the server loads.**
   `recognise.py` resolves a different reader, and `server.py` feeds
   un-normalised crops to a classifier trained on normalised ones. Both are
   one-line changes; see §13b.

`make_results_figures.py` (in the parent directory) hard-codes a temporary
session path in its `SCR` constant. To regenerate the figures from this
archive, point `SCR` at `measurements/outputs/`.
