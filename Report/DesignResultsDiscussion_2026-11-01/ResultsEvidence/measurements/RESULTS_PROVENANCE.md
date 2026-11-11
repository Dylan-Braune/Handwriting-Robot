# How every result in Section 4 and Section 5 was produced

Written 1 November 2026, for the final report in `Report/FinalReport_v6/`.

This file is the audit trail for the numbers in **Section 4 (Results)** and
**Section 5 (Discussion)**. For each result it records what was measured, the
exact script that measured it, where the input data lives, what came out, what
it means, and which table or figure in the report carries it.

Everything here was produced by running the scripts in `scripts/` against the
data paths named. Table and figure numbers are those of the built PDF
(`main.pdf`, two `pdflatex` passes). Paths are given relative to the repository
root `Handwriting-Robot/` unless they begin with a drive letter.

> **Read §9 first if you are checking reproducibility.** Three of the
> checkpoints behind these results live only in `C:/Users/braun/Downloads/`,
> and one of them is shadowed by a *different* file of the same name inside the
> repository. §9 gives the SHA-256 of each file actually used.

---

## Contents

| § | Result | Requirement | Report location |
|---|---|---|---|
| 1 | Reproduction accuracy on the 14 drawn pages | R1 | 4.2, QT 1 |
| 2 | Style correlation and the overlay test | R1 | 4.2, QT 1 |
| 3 | G-code style signature | R1 | 4.2, QT 1 |
| 4 | Writer identification from G-code before drawing | R1 | 4.2, QT 1 |
| 5 | Reading text on held-out material | R2 | 4.2, QT 2 |
| 6 | Naming the writer on held-out material | R2 | 4.2, QT 2 |
| 7 | Reproduction time | R3 | 4.2, QT 3 |
| 8 | Classification time, and the two candidate remedies | R4 | 4.2, QT 4 |
| 9 | Gantry geometry | R5 | 4.2, QT 5 |
| 10 | Stored styles | R6 | 4.2, QT 6 |
| 11 | The two training routes compared | — | 5.1.2 |
| 12 | Data corrections made before measuring | — | 5.1.3 |
| 13 | Provenance warnings and known gaps | — | 5.1.3 |

---

## 0. Shared inputs

**The 14 drawn pages.** `Software/CleanPipeline3/ReportResults/FullReproducedPages/`
holds 14 `.jpg` scans of pages the robot drew, each with a `.txt` of the same
basename giving the ground-truth transcript, one line per written line. Each
page carries three blocks of text: the 378-character paragraph (A), the pangram
(B), and a passage of the writer's own source text (C).

**The crops cut from them.** `Software/CleanPipeline3/ReportResults/SegmentedReproducedPageCrops/`
holds one folder per page (`Author150`, `AuthorD`, …) with `line_NN.png` per
detected line and a contact sheet. Note the folder names are **not** the
report's writer labels: `AuthorD` is dylan, whom the report calls **B**.
`_StandUpFailures_AsShipped/` is a diagnostic folder, not a writer.

**The database writers' pages.** `Data/Datasets/IAMpages10/<writer>/` for
writers 150, 152, 153, 384, 552, 588, each page a `.png` with a
`<page>_labels.txt` transcript. Transcripts corrected during this work keep the
original beside them as `.orig` (see §12).

**The public line corpus.** `Data/Datasets/TekliaIamLines/` with `train/`
(6 481 line crops) and `validation/` (976), each `NNNNNN.png` paired with
`NNNNNN_labels.txt`. The filenames carry no writer identity, which is why
provenance had to be established by transcript lookup (§5).

**The photographed writers' crops.** Read from the cache
`Software/CNN/NOGIT/line_cache_authors10/` through `train.AddPersonalSamples`,
which also applies the held-out split (seeded, `ValFraction = 0.15`,
`SplitSeed = 0`).

**The ODROID server log.** `measurements/raw/odroid_server_log.txt`, copied here
from `C:/Users/braun/Downloads/ALLTestData.txt`. 1 931 lines. This is the
primary record for R3 and R4: it was written by the server running as
`root@odroidn2` (visible at line 1696), so the timings in it are on the target
platform. It contains two layouts of the same timing blocks, one written to the
console and one to the browser panel, which matters for §7.

---

## 1. Reproduction accuracy on the 14 drawn pages (R1)

**Question.** How accurately does the robot's own output read back, and is the
writer still identifiable from it?

**Script.** `scripts/blocks_master.py` — per-line analysis of all 14 pages,
split by the three text blocks. Supporting: `scripts/per_author.py` (per-writer
aggregation), `scripts/cursivity.py` (connected components per character, from
the writers' real pages), `scripts/style_distance.py` (distance to the nearest
writer in style space).

`scripts/score14.py` is **superseded** and its numbers are not in the report. It
scored each page flat, by edit distance over the whole sheet, because the block
order differs between sheets. That avoids punishing a page for ordering but
cannot give per-line statistics, which is what Appendix 4's statistical analysis
needs, so `blocks_master.py` replaced it. It is kept here only so the earlier
figures in the project history can be traced.

**Run as.** `python -I scripts/blocks_master.py`

**Inputs.** The 14 pages and their `.txt` transcripts (§0); reader and writer
checkpoints (§9).

**Output.** `outputs/blocks/lines.json` — one record per line with the writer,
the block, the edit distance, the reference character and word counts and the
writer named. Also `outputs/blocks/cursivity.json`,
`outputs/blocks/per_author.json`, `outputs/blocks/style_distance.json`.

**Result.**

- 184 lines over 14 pages, 5 471 reference characters.
- Character accuracy **84.41 %** (853 characters wrong), word accuracy 45.97 %.
- Writer identified on **159 of 184 lines = 86.4 %** [80.7, 90.6], and on
  **14 of 14 pages** when a page's lines are pooled by ink.
- By writer, character accuracy spans **60.97 % to 97.01 %** — a 36-point range.
- Connectedness (connected ink components per character, measured on the
  writers' real pages) against character accuracy: Pearson **+0.855**,
  Spearman **+0.845**. Against writer identification: Spearman **−0.027**.

**Meaning.** The character half of R1 misses 85 % by 0.59 points; the
writer-identification half clears it. The 36-point spread is not noise: it
tracks how joined the writer's hand is. The synthesiser composes a line from
glyph paths traced from isolated letters, which is a good model of a printed
hand and a poor model of a cursive one, because a cursive writer never produces
those letters in isolation. Cursiveness governs legibility and has no bearing
on identifiability, which is a coherent pair of findings rather than a
coincidence.

**Implication.** The remaining work on R1 is a layout and letter-context
problem, not a letterform problem. §2 confirms this from a second direction.

**In the report.** Section 4.2, Qualification test 1. Table 38 (by block),
Table 39 (by writer), Figure 35, Figure 36, Figure 37 (confusion),
Figure 38 (connectedness). Interpreted in 5.1.1.

---

## 2. Style correlation and the overlay test (R1)

**Question.** R1 specifies 85 % *stylistic correlation*. Measured as a
correlation, what is it?

**Script.** Produced earlier in the project; the per-feature values are
archived in the report appendix rather than regenerated here. The pooled
coefficient and the overlay statistics are quoted from that archive.

**Result.**

| Measure | Measured | Target | Verdict |
|---|---|---|---|
| Pooled correlation over nine geometric features | **0.714** | 0.85 | **Not met** |
| Best single feature (slant) | 0.996 | — | — |
| Worst single feature (descender extent) | 0.148 | — | — |
| Overlay distance ratio, mismatched to matched | 2.2 | — | — |
| Overlay area under the curve | 0.71 | — | — |

Per feature: slant 0.996, ascender extent 0.974, character width 0.936,
components per word 0.832, ink density 0.680, stroke width 0.632, word gap
0.605, ascender-to-descender ratio 0.509, descender extent 0.148.

**Meaning.** On the measure the proposal actually names, R1's stylistic half
**fails by 0.136**. The per-feature ordering says exactly where: the letterforms
and their proportions transfer almost perfectly (slant, ascender, width all
above 0.93), while spacing and vertical placement do not (word gap 0.605,
descender 0.148). An area under the curve of 0.71 against 0.5 for
writer-free output means the reproduction carries real but partial writer
information, which is consistent with 0.714 and not with the 86.4 %
identification figure.

**Implication.** This was previously reported only in the appendix, with the
main section quoting writer identification as though it were the stylistic
measure. It is now stated in Results as a failure, because it is one. The
correlation and the identification figure answer different questions and both
are reported.

**In the report.** Section 4.2, QT 1, **Table 41**; per-feature detail in
appendix Table 50 and Figure 49. Interpreted in 5.1.1.

---

## 3. G-code style signature (R1)

**Question.** Is there evidence of writer-specific output that does not come
from a classifier at all?

**Script.** `scripts/predict_drawtime.py`

**Output.** `outputs/timing/predicted.json` — per writer, the pen-down path
length (`downMm`), pen-up path length (`upMm`), pen-lift count (`pulses`),
G-code line count and generation time, for the same 378-character paragraph
prepared in each of the 11 styles.

**Result.** Pen-down path length **2 015.4 mm to 3 942.8 mm** (factor 1.96);
pen lifts **544 to 1 188** (factor 2.18). The two do not rank together: the
writer with the fewest lifts (T, 544) has the fourth longest path, and the
writer with the most (153, 1 188) has the second longest.

**Meaning.** For identical text the emitted tool path differs by roughly a
factor of two in both ink laid down and pen lifts. These are read off the
geometry arithmetically, so no network can influence them. This is the only
style evidence in the report that is structurally immune to the circularity
objection that the candidate-selecting classifier also reports the score.

**In the report.** Section 4.2, QT 1, **Figure 39**.

---

## 4. Writer identification from G-code before drawing (R1)

**Script.** `scripts/classify_gcode.py`

**Result.** Six G-code jobs classified before any page was drawn. Five were
named at 66.7 % to 86.0 % confidence with agreement on at least six of every
seven lines; one (`a6393598`) was named at 29.3 % with four of eleven lines
agreeing.

**Meaning.** Removes the drawing and the paper from the loop but not the
classifier. It narrows the circularity concern without eliminating it.

**In the report.** Section 4.2, QT 1, Table 40.

---

## 5. Reading text on held-out material (R2)

This is the result that changed most during this work, and the reason is
provenance rather than modelling.

### 5a. Establishing what is genuinely held out

**Question.** The reader was trained on the public corpus. Are the database
writers' pages inside that corpus?

**Script.** `scripts/teklia_split_lookup.py` — indexes every transcript in both
corpus directories by a normalised form (letters and digits only, lower-cased)
and looks up each database writer's labelled lines.
Confirmed loosely by `scripts/teklia_fuzzy.py`, which finds each line's closest
corpus transcript by normalised edit distance.

**Run as.** `python -I scripts/teklia_split_lookup.py`

**Result.**

| Writer | Lines | In `train/` | In `validation/` | In neither |
|---|---|---|---|---|
| 150 | 91 | **84** | 0 | 7 |
| 152 | 87 | **80** | 0 | 7 |
| 153 | 97 | **83** | 0 | 14 |
| 384 | 105 | 0 | 0 | 105 |
| 552 | 73 | 0 | 0 | 73 |
| 588 | 86 | 0 | 0 | 86 |

The fuzzy check gives mean best-match scores of 99.6 %, 99.8 % and 98.3 % for
150, 152 and 153, every match in `train/`, confirming the exact lookup.

**Meaning and implication.** Writers 150, 152 and 153 are **inside the reader's
training data** and cannot measure it; the other three are in neither split, so
they are outside the training distribution rather than held out from it. Not one
line of any database writer is in `validation/`, so the corpus's own held-out
partition and the project's own held-out material are **disjoint** and can be
pooled without double counting. An earlier draft reported 91.02 % over 155 lines
on material that included the contaminated pages.

### 5b. The measurement

**Script.** `scripts/reader_two_sets.py` — takes the checkpoint as `argv[1]` and
scores it on both held-out sets.

**Run as.**
```
python -I scripts/reader_two_sets.py "C:/Users/braun/Downloads/paper_cnn_bilstm_ctc_scratch_best.npz"
python -I scripts/reader_two_sets.py "Software/CNN/weights/paper_cnn_bilstm_ctc_joint_best.npz"
```

**Outputs.** `outputs/reader_paper_cnn_bilstm_ctc_scratch_best/results.json`,
`outputs/reader_paper_cnn_bilstm_ctc_joint_best/results.json`,
logs `outputs/scratch_two_sets.log`, `outputs/teklia_val_score.log`.

**Result**, checkpoint `paper_cnn_bilstm_ctc_scratch_best.npz`:

| Held-out set | Lines | Characters | Character | Word | Read exactly |
|---|---|---|---|---|---|
| Public corpus, `validation/` | 976 | 42 072 | **94.39 %** | 73.67 % | 212 |
| Photographed writer A | 31 | 934 | 94.65 % | 76.92 % | 9 |
| Photographed writer B | 19 | 739 | 97.70 % | 84.73 % | 9 |
| Photographed writer O | 22 | 1 017 | 94.79 % | 69.05 % | 8 |
| Photographed writer T | 11 | 345 | 89.28 % | 57.69 % | 1 |
| Photographed writer Y | 17 | 640 | 92.50 % | 54.62 % | 3 |
| All photographed | 100 | 3 675 | 94.42 % | 70.74 % | 30 |
| **All** | **1 076** | **45 747** | **94.39 %** | **73.47 %** | **242** |

The deployed reader `paper_cnn_bilstm_ctc_joint_best.npz` on the same material:
public corpus 94.12 %, photographed 93.33 %, pooled **94.06 %** over the same
1 076 lines.

**Meaning.** R2's text half misses 95 % by **0.61 points**. Two features matter
more than the headline. First, the two held-out sets agree to **0.03 points**
(94.39 % against 94.42 %) although one was cut by the corpus's distributor and
the other photographed with this project's webcam and cut by this project's
segmenter — the strongest evidence in the report that the reader generalises and
that the segmenter is not systematically damaging its crops. Second, the gap
between 94.39 % character and 73.47 % word accuracy shows the residual error is
concentrated in a minority of words, which is the error structure a lexicon or
*n*-gram decoder corrects. No such decoder is used, deliberately: R1 requires
arbitrary text including non-words, so a lexicon-constrained decoder would raise
this number while making the system worse at its specified job.

**Implication.** The 94.39 % is also directly comparable with published work,
because it is measured on the same corpus: 12.6 points above the figure of
Graves *et al.* and 2.0 points below Kızılırmak *et al.* The shortfall on R2 is
against a self-set target, not against the state of the art. Installing the
scratch reader in place of the current `joint_best` is worth 0.33 points.

**In the report.** Section 4.2, QT 2, **Table 42**. Compared with published work
in Table 48 (5.1.1).

---

## 6. Naming the writer on held-out material (R2)

**Question.** The writer classifier never saw the public corpus, so its
held-out material is the project's own split. How accurate is it there?

**Script.** `scripts/holdout_namer_lines.py` — records **every** line's
prediction and confidence, so the confusion matrix is drawn from the
measurement rather than reconstructed from a summary. Database writers' crops
are re-cut from the page with `Software/CleanPipeline3/extractiam.py` rather
than read from the stale cache (see §12).

**Run as.** `python -I scripts/holdout_namer_lines.py`

**Output.** `outputs/holdout_namer/lines.json`, log `outputs/holdout_namer.log`.
This file is the input to the confusion figure — see §14.

**Result.**

| Writer | Held-out lines | Named correctly | Page named |
|---|---|---|---|
| 150 | 9 | 8/9 | Yes |
| 152 | 8 | 7/8 | Yes |
| 153 | 9 | 9/9 | Yes |
| 384 | 12 | 12/12 | Yes |
| 552 | 10 | 10/10 | Yes |
| 588 | 8 | 8/8 | Yes |
| A | 31 | 31/31 | Yes |
| B | 19 | 18/19 | Yes |
| O | 22 | 22/22 | Yes |
| T | 11 | 11/11 | Yes |
| Y | 17 | 17/17 | Yes |
| **All** | **156** | **153/156 = 98.08 %** | **11/11** |

95 % Wilson interval **[94.5, 99.3]**.

The three errors, in full:

| True writer | Line | Named | Confidence |
|---|---|---|---|
| 150 | 9 of 9 | Y | 44.2 % |
| 152 | 8 of 8 | Y | 83.8 % |
| B | 19 of 19 | Y | 39.5 % |

Mean confidence on a correct line 97.7 % (median 99.9 %); on an incorrect line
55.8 %. Six lines in total were named below 80 % confidence.

**Meaning.** R2's writer half clears 95 % as a point estimate by 3.08 points,
but the interval's lower bound sits 0.5 points *below* the target, so 156 lines
do not separate the result from the specification at 95 % confidence. About 250
lines at the same error rate would. Pooled per writer it is 11 of 11, so no
writer was misnamed as a whole and the page-level decision the system actually
makes was right every time.

Two patterns in the errors. **All three are the last line of their page**, which
under a uniform hypothesis has probability (11/156)³ = 3.5 × 10⁻⁴ — a final line
is usually a part line carrying few words, so it offers the least ink to judge.
And all three went to the same writer. Three events cannot establish a cause,
but the pattern is not an accident of this sample.

**Worst-case bound.** Counting *every* line named below 80 % confidence as wrong,
whether it was or not, leaves 149 of 156 = **95.51 %**, still above target. The
result clears the specification on the pessimistic reading too.

**Implication.** The confidence output separates the classifier's errors from its
successes well enough to be used as a warning, and no threshold is applied. A
short single line is the input most likely to be misattributed.

**In the report.** Section 4.2, QT 2, **Table 43** and **Figure 40**.

---

## 7. Reproduction time (R3)

**Question.** Does a page draw within 4 s per character?

**Script.** `scripts/r3_blocks.py` — parses the ODROID log, keeps only complete
reproductions, and groups them by block length.

**Run as.** `python -I scripts/r3_blocks.py`

**Input.** `measurements/raw/odroid_server_log.txt`

**How the population was defined.** Three decisions, all visible in the script
output:

1. A reproduction is requested twice, once to generate and once to draw, and the
   draw request **carries the generation stages forward into its own total**
   (`timing.Run.Carry`). So a block containing a `drawing` stage is the complete
   reproduction and a generate-only block is part of one. Only blocks with a
   drawing stage are counted, which prevents synthesis being counted twice.
2. The log is written in **two layouts**, console and browser panel. A run
   appearing in both is counted once: 47 blocks with a drawing stage reduce to
   43 distinct runs.
3. Seven runs were **stopped before the page finished**, detectable because
   their drawing stage falls below 1 s per character against a going rate of 2
   to 3. They are set aside as incomplete, with their log line numbers printed.

That leaves **36 complete reproductions** over nine block lengths.

**Result.**

| Characters | Runs | Allowance (s) | Worst total (s) | Worst s/char | % of allowance |
|---|---|---|---|---|---|
| 43 | 13 | 172.0 | 166.6 | **3.875** | **96.9** |
| 75 | 1 | 300.0 | 188.6 | 2.515 | 62.9 |
| 77 | 1 | 308.0 | 128.0 | 1.663 | 41.6 |
| 82 | 3 | 328.0 | 260.3 | 3.175 | 79.4 |
| 91 | 2 | 364.0 | 222.6 | 2.447 | 61.2 |
| 126 | 1 | 504.0 | 323.5 | 2.568 | 64.2 |
| 132 | 1 | 528.0 | 277.9 | 2.105 | 52.6 |
| 236 | 1 | 944.0 | 386.6 | 1.638 | 40.9 |
| 378 | 13 | 1 512.0 | 1 085.0 | 2.870 | 71.8 |
| **All** | **36** | — | — | **3.875** | **96.9** |

All 36 met. 95 % Wilson interval on 36 of 36 is [90.4, 100]. Over the 13 runs of
the 378-character paragraph the total ranged 507.6 s to 1 085.0 s, a factor of
2.14 on identical text.

**Meaning.** R3 is **met on the target platform**. But the margin is not uniform
and the worst case is the *shortest* block, not the longest: drawing is a
per-character cost that tracks the allowance, while the best-of-N search is paid
once per block, so its contribution per character falls as the block lengthens
(0.617 s/char synthesis at 43 characters against 0.156 s/char at 378). The
allowance, being strictly proportional to length, gives a short block almost no
room to pay for a fixed search.

**Implication.** Taking the measured synthesis cost of the shortest block and
the measured drawing rate, the allowance is exhausted at **about 35 characters**.
R3 is met over the range tested and is expected to fail below roughly that
length. This is recorded as an expected failure condition rather than hidden in
the margin. Nothing shorter than 43 characters was drawn, so it is a projection.

**Note on an earlier error.** A previous draft quoted the *drawing stage* as the
total and so understated the worst case as 2.705 s/char; it is 3.875.

**In the report.** Section 4.2, QT 3, **Table 44**. Interpreted in 5.1.1;
failure condition in 5.1.5.

---

## 8. Classification time, and the two candidate remedies (R4)

### 8a. The on-target measurement

**Script.** `scripts/r4_ondevice.py` — extracts every classify timing from the
ODROID log and compares each half against its own term of the allowance.

**Run as.** `python -I scripts/r4_ondevice.py`

**Why the two halves are separated.** The allowance is 2 + 0.025x. Only one term
grows with the page, and only the two network stages have a cost that grows with
the character count. So the networks must be judged against the 0.025x term and
the four image stages against the fixed 2 s. A total that fails does not say
which half is responsible.

**Result.** Two runs are recorded.

| Stage | 16-line page, 500 char | 22-line page, 982 char |
|---|---|---|
| loading | 0.63 s | 0.53 s |
| conditioning | 6.61 s | 6.04 s |
| segmenting | 4.29 s | 5.59 s |
| cropping | 4.61 s | 4.65 s |
| **image stages** | **16.14 s** | **16.81 s** |
| *against the fixed 2 s term* | *8.1×* | *8.4×* |
| reading | 19.02 s | 24.66 s |
| naming | 11.09 s | 14.01 s |
| **network stages** | **30.12 s** | **38.67 s** |
| *per character* | *0.0602 s* | *0.0394 s* |
| *against the 0.025 s/char term* | *2.4×* | *1.6×* |
| **Total** | **46.26 s** | **55.49 s** |
| **Allowance** | **14.50 s** | **26.55 s** |
| **Verdict** | Exceeded by 31.76 s | Exceeded by 28.94 s |

Over the two runs the networks are **67.6 %** of the total and the image stages
**32.4 %**.

**Meaning.** R4 is not met, and it fails on **both** halves. A previous draft
attributed the whole deficit to segmentation and claimed recognition sat inside
its allowance; the on-target measurement contradicts that. Because the network
cost per character exceeds the allowance per character, **the requirement fails
at every page length** and cannot be rescued by choosing a longer page.

### 8b. What would fix it, measured rather than assumed

**Script.** `scripts/r4_conditions.py` — applies two candidate changes to the
same 14 pages with the same weights, and applies the resulting factors to the
on-target stage times above.

**Run as.** `python -I scripts/r4_conditions.py`

**Output.** `outputs/r4_conditions/conditions.json`, log
`outputs/r4_conditions.log`.

**Result.**

| Change | Measured factor | Cost measured alongside it |
|---|---|---|
| Networks: same weights in library modules, whole page in one batched forward pass instead of one crop at a time | **2.76×** over 14 pages | **None.** Text read out identical on every line. |
| Image stages: run at half linear resolution | **4.37×** | 182 lines found where 247 were; **0 of 12** pages kept their line count; text agreed with the full-resolution reading on **33.8 %** of characters on average and 14.1 % at worst; segmenter **failed outright on 2 of 14 pages** (`IndexError` in `SeamCuts`, no line centres found) |

Applying the factors to the measured on-target times:

| Characters | Allowance | Image | Networks | Total | Verdict |
|---|---|---|---|---|---|
| 500 | 14.50 s | 3.70 s | 10.91 s | 14.61 s | Exceeded by 0.11 s |
| 982 | 26.55 s | 3.85 s | 14.01 s | 17.86 s | **Met** |

Per-character network cost with the library route: 0.0218 s/char (0.9× the term)
and 0.0143 s/char (0.6× the term).

**Meaning.** **The network half of R4 is met by the first change alone**, at both
page lengths — and it does not require a framework, because batching the existing
from-first-principles forward pass per page rather than per crop is the same
arithmetic on a larger array. The second change is the obvious remedy for the
image half and **it does not work**: a speed-up that loses a quarter of the lines
and crashes on two pages is not a speed-up. The reduction has to come from
redesigning the line grouping, whose cost is not proportional to pixel count.

**Implication.** R4 is reachable in two steps, the first available today and
measured, the second requiring segmenter work this project did not do. Read
alongside the budget constraint: a faster board would have reduced both terms at
once. Note the absolute times in this experiment are contention-sensitive — the
**ratios** are the transferable result, which is why ratios are what the report
quotes.

**In the report.** Section 4.2, QT 4, **Table 45** (on-target) and **Table 46**
(the two remedies). Interpreted in 5.1.1; the route to compliance in 5.1.3.

---

## 9. Gantry geometry (R5)

**Script.** `scripts/gantry_geom.py` solves the shear from the measured sides
and checks it back against both. The measurements themselves were taken by hand
with a 150 mm vernier caliper reading to 0.01 mm off drawn paper, so they are
not reproducible from code; the seven values are typed into
`gantry_geom.py` and into `make_results_figures.py`, which is the one figure in
Section 4 whose input is hard-coded rather than read from a JSON file.

An earlier set of side lengths, taken while the travel constants were still
being tuned, is deliberately **not** reported: it measured a machine that no
longer exists. `gantry_geom.py` records that exclusion in its own header.

**Result.**

| Figure | Feature | Nominal | Measured | Error |
|---|---|---|---|---|
| Square | bottom (X) | 100.00 | 100.30 | +0.30 |
| Square | top (X) | 100.00 | 100.20 | +0.20 |
| Square | left (Y) | 100.00 | 100.10 | +0.10 |
| Square | right (Y) | 100.00 | 99.90 | −0.10 |
| Triangle | base (X) | 80.00 | 80.30 | +0.30 |
| Triangle | left sloped | 56.57 | 56.20 | −0.37 |
| Triangle | right sloped | 56.57 | 56.90 | +0.33 |

Worst side 0.30 mm, mean absolute error 0.175 mm, X scale +0.250 %, Y scale
+0.000 %. Solving the shear from the two sloped sides gives an apex offset
d = −0.493 mm and a squareness error φ = −0.706°, which predicts both sloped
sides to within **0.02 mm** — the caliper's resolution.

**Meaning.** R5 is met on this specimen, with the worst case 0.20 mm clear of
the 0.5 mm limit. More useful than the pass is the diagnosis: one unknown
fitted to two measurements with residuals at the reading resolution is a
validated model, and it says the dominant residual is a **0.706° frame
non-perpendicularity**, not a scale error and not lost steps. A square alone
cannot detect this, because a sheared square still has four equal sides.

**Implication.** The open-loop drive is not what limits R5; squaring the frame
is. One square and one triangle support a compliance statement and a diagnosis,
but **not** a repeatability figure or a tolerance — that needs repeated
specimens (at least five of each, plus the diagonals), which is outstanding.

**In the report.** Section 4.2, QT 5, Table 47, Figure 41, Equation for the
shear.

---

## 10. Stored styles (R6)

**Result.** Eleven writer styles trained and permanently stored — 150, 152, 153,
384, 552, 588 from the public database and A, B, O, T, Y photographed for this
project — each with a glyph library and a style profile, and the classifier
checkpoint records the same eleven names.

**Meaning.** Met, against a minimum of ten, with **one** style of margin. The
set was reduced from thirteen during this work by withdrawing two writers and
retraining, so the margin fell from three to one.

**In the report.** Section 4.2, QT 6.

---

## 11. The two training routes compared

**Question.** The writer classifier was trained twice, once through the library
route and once through the from-first-principles trainer. Which should be used?

**Script.** `scripts/namer_compare.py` — both checkpoints, identical crops,
identical preprocessing, on both populations the classifier has to serve.

**Run as.** `python -I scripts/namer_compare.py`

**Output.** `outputs/namer_compare/compare.json`, log
`outputs/namer_compare.log`.

**Result.**

| Population | Library route | From first principles | Difference |
|---|---|---|---|
| Held-out real lines (R2), per line | 98.08 % (153/156) | **95.51 %** (149/156) | −2.56 |
| 95 % Wilson interval | [94.5, 99.3] | [91.0, 97.8] | — |
| Writers named when pooled | 11/11 | 11/11 | — |
| Lines the robot drew (R1), per line | 85.95 % (159/185) | **80.54 %** (149/185) | −5.41 |
| Pages named when pooled | 11/11 | **10/11** | −1 |

**Meaning.** The from-first-principles checkpoint **meets R2** at 95.51 %
against the 95 % target, so the training code demonstrably works. It is
nevertheless not the checkpoint used, and the reason is the second population:
on the pages the robot drew the classifier is the *measuring instrument* for R1,
R1's stylistic target is 85 %, and 80.54 % is below it. Installing it would
convert R1's writer-identification half from met to not met and lose a page.

**Implication.** The decision was to measure with the better instrument and
report the weaker checkpoint as evidence about the training route, which is what
it is evidence of. This also forced a correction in Section 3.5, which had
claimed *"No machine-learning library is used anywhere in this subsection"* — the
namer behind every measurement is library-trained, so that sentence was not
defensible. It now distinguishes: all inference and the reader's training run
through `NumpyNN.py`, the namer was trained both ways, and the library-trained
checkpoint was used, with the cost stated.

**Note on counts.** This script finds 185 crops in the reproduced-pages folder
where QT 1 scores 184 lines; one crop has no corresponding label line. The 159
correct namings are the same in both.

**In the report.** Section 5.1.2, **Table 49**. Section 3.5 opening paragraph.

---

## 12. Data corrections made before measuring

These changed the measured numbers without touching a model, and they are the
reason the held-out figure rose from 91.02 % to 94.39 %.

### 12a. Finding the misalignment

**Script.** `scripts/correspondence.py` — reads every crop and scores it against
its *own* label line and against its neighbours'. A matching line count proves
nothing: writer 384's `j06-014` had 11 crops and 11 label lines and was
misaligned from line 6 down. If the correspondence is right a crop reads closest
to its own label; if it reads closer to a neighbour's, the page is out of step.

**Result.** Over 58 pages and 522 lines it found three misaligned pages on
writer 153 that the count check could not see — on `c03-003d` line 6 the
neighbour's label matched at 100 % against 3 % for its own.

### 12b. The corrections

**Scripts.** `scripts/write_384_labels.py`, `scripts/write_153_labels.py`. Both
refuse to write if the transcript line count disagrees with what the extractor
produces, and both preserve the original as `<page>_labels.txt.orig`.

**Result.** Ten transcripts rewritten and one created for writer 384; three
rewritten for writer 153. Afterwards: 105/105 crops matched labels for 384 with
0 merged crops, and 153's three pages gave 9/9, 9/9 and 11/11.

**Effect, measured.** `scripts/rerun_384.py` and
`scripts/full_holdout_rerun.py` (output `outputs/holdout_rerun/results.json`).
Writer 384's held-out page went **60.78 % → 92.18 %** character accuracy. Across
the whole held-out set the figure went from 91.60 % (old cache, old labels, 155
lines) to 94.33 % (156 lines) before the population was narrowed to the two
clean sets of §5.

**Meaning.** A +2.7-point gain from fixing data, with no model change. The
photographed writers' figure was unchanged to two decimals across the rebuild,
which is what confirms the re-measurement pipeline was equivalent and the gain
real.

### 12c. A fault found and not fixed

**Script.** `scripts/why_dropped.py` — repeats `SegmentLines` with the rejection
made visible, reporting every band's ink count and the threshold it was judged
against, and writing out the dropped bands as images.

**Result.** On writer 588's `n06-082` (band 7) and `n06-133` (band 5) the band
owns **zero** ink, so the crop builder has nothing to cut. The transcripts are
correct. The fault is in `AssignOwnership`, not in the ink threshold and not in
the labels — the original hypothesis of a `minInk` problem was wrong.

**In the report.** Section 5.1.3, unsolved problems.

---

## 13. Provenance warnings and known gaps

**These affect whether the results can be reproduced. Read before relying on
any number above.**

### 13a. Checkpoints used, by hash

| Role | File used | SHA-256 (first 16) |
|---|---|---|
| Reader (all §5 figures) | `C:/Users/braun/Downloads/paper_cnn_bilstm_ctc_scratch_best.npz` | `2add315ded078de5` |
| Reader (deployed comparison) | `Software/CNN/weights/paper_cnn_bilstm_ctc_joint_best.npz` | `336802d535992323` |
| Namer (all §6, §1 figures) | `C:/Users/braun/Downloads/author_shape_11_pw170_weights.npz` | `1884fb86467538eb` |
| Namer (from-scratch, §11) | `C:/Users/braun/Downloads/author_shape_11w_pw170_scratch_weights.npz` | `b23a709a30144f67` |

**Two problems.**

1. `Software/CNN/weights/paper_cnn_bilstm_ctc_scratch_best.npz` has hash
   `5191f9f1e7569341` — it is a **different file** from the one measured, despite
   the identical name. Loading the repository copy will not reproduce 94.39 %.
2. **Neither `author_shape_11*` checkpoint is in the repository at all.** Every
   writer-identification figure in the report depends on files that exist only
   in `Downloads/` and are not version-controlled.

**Action needed:** copy the four files above into the repository under
unambiguous names, or into `measurements/checkpoints/`. This was not done
automatically because adding ~21 MB of binaries, and overwriting a name the
server resolves, are both decisions for the repository owner.

### 13b. The configuration measured is not the configuration deployed

- `recognise.py:52` lists `paper_cnn_bilstm_ctc_joint_best.npz` **first** in
  `_TEXT_CANDIDATES`, so the server loads the joint reader, not the scratch
  reader the figures were measured with. Cost: 0.33 points (94.39 → 94.06).
- `web/server.py:216-217` sets `PAGE_NAMER = SHAPE_MODEL` with
  `PAGE_NAMER_STROKE_NORM = False`, so it feeds **un-normalised** crops to a
  classifier trained on crops normalised to 0.17 x-heights. The cost of this
  mismatch was **not measured** and may be larger than the reader substitution.

Both are one-line changes. Until they are made, the report's accuracy figures
describe a configuration the server does not load. Section 5.1.3 says so.

### 13c. Stale caches

Seven pages of writer 384 are skipped whenever the data set is built, because
the cached crops in `Software/CNN/NOGIT/line_cache_authors10/` predate the label
corrections of §12 and the two now disagree on line count. The skip messages are
visible in `outputs/holdout_namer.log` and `outputs/namer_compare.log`. **Those
pages are absent from training although their labels are now correct.** The
caches must be rebuilt before the next training run.

### 13d. What is not measured

- **The segmenter's contribution to the R2 shortfall.** The reader is charged
  with characters the segmenter pulled in from the line above; this is known to
  happen when two lines sit close together and is not quantified.
- **The train-and-serve normalisation mismatch** of §13b.
- **Repeatability of the gantry geometry** — one square and one triangle.
- **Any block shorter than 43 characters** for R3, where the margin is thinnest.
- **A human panel or an independent judge** for R1's stylistic half.
- **An architecture sweep** — one was programmed but no result was kept.

### 13e. Outstanding `\mustdo` markers

Section 4 has **none**. Section 5 has six, all requiring facts only the author
holds: hardware documentation and the inconsistent number-of-tries constant
(14 / 20 / 30), the mains insulation rating of the power supply, which parts
were 3D-printed and in what material, the consent record for the photographed
writers, any applicable regulation, and the ethics clearance number.

### 13f. Cross-section inconsistency

"thirteen writers" still appears in Section 2 (once), Section 6 (three times)
and Section 3 (six times), while Sections 4 and 5 consistently say eleven. These
were **not** bulk-replaced, because some may legitimately describe the earlier
13-class classifier rather than the evaluated population. Each needs reading in
context.

### 13g. Length against the guidelines

| Section | Pages | Appendix 4 guideline |
|---|---|---|
| 4 Results | 85–104 (19) | 10–15 |
| 5 Discussion | 105–118 (14) | 6–8 |
| Main report total | ~118 | **70 (hard cap)** |

---

## 14. Regenerating the figures

**Script.** `Report/FinalReport_v6/make_results_figures.py` (in the report
directory, not in `measurements/`).

**Run as.** `python -I make_results_figures.py` from `Report/FinalReport_v6/`.

It reads the JSON outputs listed above and writes seven PNGs into `Figures/`.
Its palette comes from `Software/CleanPipeline3/results/palette.py` — black,
silver, white and shades of blue only, no red/green/amber, with met versus
unmet distinguished by saturation so the figures still read in grey scale.

**Important:** the script reads its inputs from the **session scratchpad path**
hard-coded in its `SCR` constant. That directory is temporary. To re-run it
against this archive, point `SCR` at
`Report/FinalReport_v6/measurements/outputs/` instead.

| Figure | File | Input |
|---|---|---|
| 35 | `res_blocks.png` | `outputs/blocks/lines.json` |
| 36 | `res_per_author.png` | `outputs/blocks/per_author.json` |
| 37 | `res_confusion_reproduced.png` | `outputs/blocks/lines.json` |
| 38 | `res_connectedness.png` | `outputs/blocks/cursivity.json` |
| 39 | `res_style_signature.png` | `outputs/timing/predicted.json` |
| 40 | `res_confusion_holdout.png` | `outputs/holdout_namer/lines.json` |
| 41 | `res_gantry_geometry.png` | measured by hand, values in the script |

---

## 15. Building the report

`latexmk` is unavailable locally (no Perl), so build with two direct passes:

```
cd Report/FinalReport_v6
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Then check structure with `python -I verify_report.py`, which follows `\input`
from `main.tex` and reports undefined references, captions outside floats,
unreferenced equations and floats, citations off the keep-list, first person in
Part 3, and headings followed directly by a float.

As of 1 November 2026 the build is clean (0 LaTeX errors) and the verifier
reports no undefined references. Four checks still fail, **all in Section 3 and
the appendices, none in Sections 4 or 5**: 17 captions outside a float (mostly
`design/3_3_hardware.tex`), 3 unreferenced equations, 21 unreferenced floats,
and 4 citations off the keep-list (`iamdb`, `duda1972`).
