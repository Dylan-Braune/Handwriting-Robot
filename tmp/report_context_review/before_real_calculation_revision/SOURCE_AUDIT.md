# Research citation audit — 1 November 2026

## Current policy, following the user's correction

The manuscript uses research papers and relevant manufacturer/standards material
to support borrowed methods and external technical facts. The 285 repeated code
and internal-record citation sites have been removed, as have their two bibliography
entries. They have not been mechanically replaced with unrelated papers.

Project-specific parameter values, circuit construction, measured observations,
training records, sweeps, heuristics and coordinate transforms are described as
this project's work. Their supporting files remain in the source package. A paper
cannot establish what was wired, trained or measured in this project. Constructed
calculation examples remain labelled and follow the equation they evaluate; they
do not need another citation to the source file after every numerical substitution.

The few module pointers in ordinary prose are reading aids, not numbered scholarly
references. The detailed evidence-manifest paths are introduced once in the evidence
appendix and retained with the raw results. Main-text figure, table and algorithm
captions no longer carry repetitive self-citations.

Verified printed page numbers remain in the in-text cite commands. Author copies
without printed pagination explicitly use PDF page numbers. Unpaginated manufacturer
or framework documentation uses section headings. Unavailable source/page checks
retain `page to verify` and a visible blue source flag; no page was invented.

## Verified research and technical sources

| Key | Inspected material and in-text locator | Claim supported and scope |
|---|---|---|
| `teikari2016` | Local thesis, printed p. 6, Fig. 4 | Example Cartesian robot arrangement; the project's kinematic simplification is its own geometric argument |
| `a4988ds` | Official rev. 8 datasheet, p. 6 Table 1 and p. 9 | Microstep selection, current chopping and Vref/current-limit relation |
| `bresenham1965` | Original paper scan, printed p. 28 | Integer error recurrence and octant handling; actual project abort/counter loop remains its own implementation |
| `rosenfeld1966` | Author-hosted journal scan, pp. 476–479 | Sequential labels and equivalence handling; the project's run-based eight-connected variant is described as its implementation |
| `avidan2007` | Inspected author-paper PDF, PDF p. 3 Sec. 3 | Minimum-cost seam dynamic programme; project cost weights, horizontal use and seven-row step are adaptations |
| `kizilirmak2023` | arXiv paper, pp. 4–5 | 12-convolution stage structure, feature sequence and two 256-unit BiLSTMs; project dimensions/head/decoder differences distinguished |
| `ioffe2015` | Official PMLR reading PDF, PDF pp. 4–5, Alg. 1 | Batch-normalisation forward, derivative and inference-statistic conventions; copy has no printed proceedings pagination |
| `graves2006` | Author-hosted ICML reading PDF, PDF pp. 2–6 | CTC path collapse, lattice, likelihood and derivative; PDF p. 2 defines insertion/deletion/substitution error |
| `graves2013` | arXiv v5 reading copy, pp. 4–6, especially p. 5 Eqs. 9–11 and p. 6 Eqs. 13–14 | Modern cell/memory recurrence, BPTT and softmax loss gradient; project omits source's peephole terms |
| `srivastava2014` | Journal paper, printed pp. 1931 and 1934 | Dropping units during training and expected-output scaling; project uses inverted dropout, explicitly distinguished |
| `zhang1984` | Original journal scan, printed p. 237 | Two thinning subpasses and neighbour tests, directly checked against implementation |
| `kingma2015` | ICLR author copy, p. 2 Alg. 1 | Moment estimates, bias correction and Adam update |
| `loshchilov2019` | ICLR paper copy, p. 3 Alg. 2 | Decoupled AdamW decay; project coefficients and schedule remain implementation choices |
| `tieleman2012` | Author-hosted combined Lecture 6 slide PDF, PDF p. 29 | RMSprop moving-square-gradient principle; lecture uses 0.9/0.1, code selects 0.99/0.01 |
| `pytorchctc` | Official CTCLoss documentation, `Parameters` | Default target-length-normalised mean and zero-infinity option; unpaginated web source |
| `rohmlowside2026` | ROHM manufacturer article, sections `Why NPN Devices Are Used in Low-Side Switches` and `Protection and Wiring Techniques` | Base-resistor relation and flyback recirculation; unpaginated, not a substitute for BC337-specific ratings |


| `bradley2007` | Author-hosted six-page reading copy, PDF pp. 2–3, Secs. 2.2–3, Eq. (2); complete pages rendered and read | Local-average illumination comparison and four-corner integral-image sum. Project forms a ratio and uses Otsu; it does not claim to reproduce Bradley's threshold rule. Replaces the unavailable Crow citation at this site. |
| `brink2012` | Author-hosted journal copy, printed p. 162; complete page rendered and read | Ink width is useful for writer identification and depends on pen and style. Does not prove which features the initial project CNN used. |
| `haralick1987` | Author-hosted original scan, printed pp. 533 and 541, Defs. 1 and 36; complete pages rendered and read | Binary dilation and closing, including closing narrow breaks. Project kernel-size choices remain engineering choices. |
| `haines2016` | Author-preprint PDF p. 4, Sec. 4 | Approximately four A4 sample pages in that study; no universal sample minimum. |
| `dai2024` | ECCV paper, printed p. 9 | One generation reference distinguished from the 339-writer training corpus and four-GPU setup; no ODROID timing inferred. |

Graves 2013 was additionally inspected at printed p. 18 for online pen positions
and end-of-stroke markers. The project has offline photographs, so temporal order
is not measured. Haines and One-DM already appear in the user's Literature Review.
The three newly used research entries above were already in the supplied bibliography;
their reading-copy URLs have been recorded. No additional bibliography entry was needed.

## Referenced sources still requiring the user's manual check

| Key | Citation site | What must be checked | Why left flagged |
|---|---|---|---|
| `acarnley2002` | 3.3.2 load/torque discussion | Appropriate page on load torque and operating-speed characteristics | No accessible full book copy inspected |
| `bc337ds` | 3.3.3 transistor drive | Exact installed-device datasheet, saturation conditions and relevant page | Manufacturer PDF request returned access denial; generic ROHM theory does not verify this part's ratings |
| `otsu1979` | 3.4.2 threshold criterion | Between-class variance and maximising threshold page | Located original scan but complete source access failed; old audit's p. 63 was not reused without inspection |
| `postl1986` | 3.4.3 projection-based deskew context | Original projection method, exact relevance and page | Full source not inspected; project's squared-row objective remains explicitly its implemented criterion |
| `chellapilla2006` | 3.5.2 matrix convolution | Unrolled convolution/matrix product and page | Repository responded with a non-PDF challenge page |
| `douglas1973` | 3.6.2 and appendix simplification | Endpoint-fit procedure and precise page; original journal title | Located scan could not be opened reliably; project's closed-loop adaptation is separately described |
| `marti2002` | 3.5.1 IAM dataset identification | Exact dataset-paper page and relevance to used release | Existing user Literature Review source; no full source inspected in this pass, no release counts inferred from it |


Crow is no longer a pending manuscript source: Bradley and Roth supply an inspected
explanation of the exact rectangular-sum identity used here.

## Equations and original contribution

The established CTC, LSTM, batch-normalisation, thinning, RMSprop, Adam/AdamW,
seam dynamic programme and driver-current relation retain their research or
manufacturer citations next to the explanation. Matrix derivatives and the central
finite-difference expression now include their algebraic derivation. The remaining
project geometry, scale conversions, weighted vote, connector/selection heuristics
and time budgets are explicitly presented as derived calculations or project definitions.
This avoids attributing a custom formula or empirical sweep result to an unrelated paper.

All forty numbered equations and their numerical substitutions remain. Eleven
algorithm blocks, the recorded width sweep and the 48-run seam-coefficient study
remain unchanged in content and measured results. The equation audit records the
basis of each formula separately from a code locator.

## Current citation counts

The Design and appendices cite **28** distinct external sources at **53** citation sites. The union with the current baseline main-report sections is **41** sources. **7** source/page checks remain flagged. There are **zero** code/internal-record bibliography citations in the manuscript.
