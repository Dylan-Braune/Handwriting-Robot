# Design source and page-reference audit — 1 November 2026

## Citation policy applied

Borrowed methods and technical facts are cited in the text where used. Printed
page numbers are used when visible in the inspected copy. When an inspected author
PDF has no printed pagination, the citation explicitly says **PDF p.** rather than
inventing a proceedings page number. Bibliography URLs identify those reading
copies. A verified web page without pagination uses its exact section heading;
it cannot honestly be assigned a page number.

Uninspected references retain their numbered citation and carry a blue
**verify source/page: key** marker at the exact citation site. The user requested
these flags for manual checking. No locator has been borrowed from an older
agent audit or inferred solely from the article's bibliographic page range.
Remove a marker only after checking both the claim and the locator in the actual
source. This audit concerns the revised Design and supporting appendix, not a new
audit of every source used elsewhere in the report.

## Verified sources and locators used

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

Relevant complete source pages were inspected as rendered images as well as text.
The Levenshtein 1966 scan was also inspected, but its deletion/insertion/reversal
coding treatment is not the clearest support for the project's substitution-aware
reader metric. The existing Graves 2006 reference supports that exact convention,
so an additional historical citation was unnecessary. The original LSTM paper was
not used to support later forget-gate equations; the user's existing Graves 2013
source supplies the needed form with the peephole difference stated.

## Referenced sources for the user's manual check

| Key | Citation site | What must be checked | Why left flagged |
|---|---|---|---|
| `acarnley2002` | 3.3.2 load/torque discussion | Appropriate page on load torque and operating-speed characteristics | No accessible full book copy inspected |
| `bc337ds` | 3.3.3 transistor drive | Exact installed-device datasheet, saturation conditions and relevant page | Manufacturer PDF request returned access denial; generic ROHM theory does not verify this part's ratings |
| `crow1984` | 3.4.2 summed-area tables | Four-corner sum and constant-window-work discussion | Full PDF could not be obtained from located link |
| `otsu1979` | 3.4.2 threshold criterion | Between-class variance and maximising threshold page | Located original scan but complete source access failed; old audit's p. 63 was not reused without inspection |
| `postl1986` | 3.4.3 projection-based deskew context | Original projection method, exact relevance and page | Full source not inspected; project's squared-row objective remains explicitly its implemented criterion |
| `chellapilla2006` | 3.5.2 matrix convolution | Unrolled convolution/matrix product and page | Repository responded with a non-PDF challenge page |
| `douglas1973` | 3.6.2 and appendix simplification | Endpoint-fit procedure and precise page; original journal title | Located scan could not be opened reliably; project's closed-loop adaptation is separately described |
| `marti2002` | 3.5.1 IAM dataset identification | Exact dataset-paper page and relevance to used release | Existing user Literature Review source; no full source inspected in this pass, no release counts inferred from it |

These are **referenced but unverified sources**, not assertions that their claims
were checked. The citation flags in the reading PDF intentionally remain visible.

## Project facts and derivations

- Motor identity, supplied camera specification, supply readings, absent diode and
  wiring details come from the user's hardware clarifications. Installed wiring
  and measured values require the final circuit/photos/lab record, not a generic
  publication. Exact motor and camera manufacturer documents still need attaching
  or checking before final specification claims.
- Sweep settings, computer, split counts and checkpoint provenance were supplied
  by the user and agree with relevant archived project context. They were not
  remeasured. The table is validation/model-selection evidence, not an independent
  final-system accuracy test.
- Source code is cited once as a repository snapshot (`braunepipeline2026`) and
  identified by modules/functions throughout the package. It has no genuine page
  numbering. Tensor sizes, numerical recipes, ownership rules, fallback logic and
  file conventions are code observations, not borrowed empirical facts.
- Pulley conversion, resolution, torque decomposition, calibration sensitivity,
  inverse image mapping, glyph transforms, harmonic score examples and timing
  arithmetic are worked project derivations. They are introduced as calculations,
  not falsely attributed verbatim to a publication. Standard algorithms underlying
  them receive the appropriate source citations.
- BN, CTC and LSTM derivatives in the appendix are the project's algebraic
  implementation of the cited operations. Exact stochastic/precision conventions
  are checked separately from mathematical description.

## Bibliography size and reuse

The supplied Literature Review cites **21** distinct sources. The revised Design
and its appendix cite **25**, including the project source record. Their union is
**40**; the union with the other current baseline main-report sections is also 40.
This is a count of cited keys, not all unused entries in the `.bib` file, and does
not include a future completed Results/Discussion bibliography.

Six Design references are already cited in the user's supplied Literature Review:
Douglas–Peucker, Graves 2006, Graves 2013, Kizilirmak, Marti and Teikari. Other
reused entries already existed in the supplied bibliography. Only **two new
external entries** were introduced: official PyTorch CTC documentation for the
reduction convention and ROHM's switching explanation for the pen circuit. One
internal project-code record was added. No unrelated general textbook was added.

Machine-readable counts are in `SOURCE_AUDIT_COUNTS.json`. Source downloads and
rendered audit sheets remain in `tmp/report_context_review/sources` and adjacent
temporary files, not in the Prism source ZIP.

## Additional representation-rationale checks in the full polish pass

- `haines2016`: inspected complete author-preprint PDF page 4, section 4,
  including the approximate four-A4-page sample statement. This reading copy is
  a 19-page preprint with inconsistent journal-header metadata; the citation
  explicitly uses **PDF p. 4**, not an inferred page in the final 18-page article.
  The present project does not inherit that paper's manual annotation procedure,
  scan resolution or collection time merely by citing its sample size.
- `graves2013`: additionally inspected complete printed page 18, which identifies
  the data as online pen positions and end-of-stroke markers. It supports the
  missing-data distinction, not an assertion that every generator needs online data.
- `dai2024`: inspected complete printed page 9: 339 training writers, one style
  reference during generation, and training on four RTX 3090 GPUs. These describe
  that study, not measured ODROID performance or a universal data minimum.

Both newly cited keys were already in the user's Literature Review. The corpus
versus per-writer-reference distinction is intentional. Few photographed pages,
no online trajectories and avoiding a separate ODROID generative training task
are the user's supplied reasons; explicit geometric control is a design argument
grounded in the current stored-path and conversion implementation.

## Citation and calculation pass requested during final review

Every numbered equation has an immediately adjacent citation to its implemented
convention, input record or primary source, and a numerical substitution (existing
substitutions are retained). Borrowed mathematical derivations also retain their
primary references and inspected page locators in the neighbouring explanation.
Unverified pages now explicitly say `page to verify` inside the cite command as
well as retaining the visible blue source flag. No page was fabricated to make
the report appear complete.

Implementation claims throughout the prose and captions now cite the project
source with file/function locators; recorded hardware/model/experiment facts cite
the accompanying unpublished project records. This introduces one internal
evidence reference, not another outside publication. Chosen calculation examples
are explicitly identified and must not be described as measurements.

The new coefficient sweep is in Evidence/coefficient_sensitivity.json and its
script. It is post-design sensitivity, not proof of historical selection or an
independent segmentation-accuracy result. Input and code hashes are retained.
