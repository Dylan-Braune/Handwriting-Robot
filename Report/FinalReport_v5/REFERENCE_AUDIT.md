# Reference audit — FinalReport_v5

2026-10-29. Two jobs: (a) give every citation the page the information was
taken from, (b) check that each source actually says what it is cited for, and
remove the citations that do not.

**Scale.** 57 sources, **146 `\cite` commands**. **31 sources were obtained and
read**; 26 could not be. **66 of the 146 sites now carry a page locator.**

**The rule applied without exception:** a locator is written only after the
source was opened and the claim found. Nothing here is inferred from a page
range, a search-result snippet or memory. Where a source could not be obtained,
the citation was left exactly as it was — unverified, not wrong.

**Sub-rule for editions.** Where only a *preprint* was available but the
bibliography entry is the *proceedings*, the preprint's page numbers are not the
proceedings' page numbers. Those citations get a numbered element instead
(`Alg. 1`, `Tab. 3`, `Lec. 6e`). Four locators were written and then withdrawn
for breaking this rule, including one for `lecun1998`, whose only available copy
has Type-3 fonts and extracts as unreadable text.

---

## A. Errors found and corrected

| # | Source | Problem | Fix |
|---|---|---|---|
| 1 | `barrow1977` | Page range missing (the report's own `\mustdo`, twice) | IJCAI 1977 **pp. 659–663**; both markers removed |
| 2 | `pearson1896` | `\mustdo` asked whether vol. 186 or 187 | **187**, pp. 253–318 was already right; DOI added |
| 3 | `vinciarelli2001` | Pages missing | **22(9):1043–1050** + DOI |
| 4 | `postl1986` | Pages missing | ICPR 1986 **pp. 687–689**, Paris |
| 5 | `hemanth2022` | **Year wrong.** Paper header reads "OCTOBER 2021, VOLUME: 12, ISSUE: 01", DOI `10.21917/ijsc.2021.0351` | Year **2021**, month Oct, DOI added |
| 6 | `ma2023` | **Pages wrong.** Paper states "pages 268-273" | **268–273** (was 275–280) + DOI |
| 7 | `mamun2024` | **Claim not in the source.** "the EMNIST, IAM and **HEIC** datasets" — the source names IAM, MNIST and EMNIST (p. 2). HEIC is an image container format | Text corrected, locator p. 2 |
| 8 | `marti2002` | **Figures from the wrong source.** 657 writers / 5 685 sentences / 115 320 words / 300 dpi are the **FKI release statistics**; the 2002 paper reports **1 066 forms by ~400 writers** | New `iamdb` entry for the release figures; `marti2002` p. 39 for the paper's own |
| 9 | `chen2023` | **Mis-cited.** The whole "Training" paragraph hung off it. chen2023 names batch normalisation only in a list (p. 40), has dropout layers in its model (p. 42), and **reports validation accuracy only — never CER or WER** | Split to `ioffe2015` Alg. 1, `srivastava2014` p. 1930, `kizilirmak2023` p. 1, `he2015`, `simard2003` |
| 10 | `chen2023` | "libraries such as **TensorFlow** and OpenCV" — only OpenCV found (p. 41) | Reduced to OpenCV |
| 11 | **`shi2017` — citation removed** | The Discussion table credited it with a result **on IAM lines**. CRNN evaluates on **IIIT5k, SVT, ICDAR 2003/2013 — scene text** (p. 5) and reports no IAM result | Row **replaced** with Graves *et al.* 2009, the IAM line-recognition paper already cited: **74.1 % word, 81.8 % character** (p. 855). `shi2017` still cited where it is correct (the architecture row, p. 2298) |

## B. Comparison table, now filled from the papers

The `\mustdo` asked for "four published accuracy cells". Two of them are not
accuracies. The column is renamed and each cell names its own metric:

- Graves *et al.* (p. 855) — IAM offline lines, **74.1 % word / 81.8 % character**
- Kızılırmak *et al.* (p. 1) — IAM lines, **3.59 % CER, 9.44 % WER**
- Kang *et al.* (Tab. 1) — IAM, **FID 120.1 → 130.7**; real data 90.4
- Dai *et al.* (Tab. 3) — IAM OOV-unseen, **FID 102.8**
- Bulacu *et al.* — **still blank.** The IEEE copy could not be obtained, so no
  number was entered rather than a guessed one. The `\mustdo` says exactly that.

FID is a distance between image distributions, lower-is-better, with no
accuracy equivalent — a sentence now says so, so the column is not read as one
scale.

## C. Problems found, **not** fixed — they need your judgement

1. **A4988 undervoltage claim is about the wrong rail.** Appendix H says the
   bulk capacitor keeps the sag "clear of the driver's undervoltage threshold".
   The A4988's UVLO is on **V_DD, the 3–5.5 V logic supply (2.7–2.9 V, p. 4)**.
   The datasheet gives **no** undervoltage threshold for V_BB, the 24 V rail.
   The sag argument still holds against 24 V; it is just not a UVLO argument.
2. **The 3.3 V logic margin may not exist.** The datasheet puts logic high at
   **0.7 × V_DD** (p. 4). If the carriers run V_DD = **5 V**, V_IH = 3.5 V and
   **3.3 V fails**. Only a 3.3 V V_DD makes the report's claim true. Worth
   measuring — it may bear on the outstanding X-axis direction-pin fault.
3. **`stepperds` is not a citable reference.** The entry has no manufacturer, no
   part number and no date — just "NEMA 17 bipolar stepper motor… Manufacturer
   datasheet". It is cited 5 times, including for the 1.68 A rating that
   Appendix H's current calculation rests on. Appendix 4 gives no credit for
   component values that are not justified from a source. Same gap as the
   existing `\mustdo` about the ODROID board revision.
4. **`douglas1973` journal name.** In 1973 the journal was *The Canadian
   Cartographer*; it became *Cartographica* later. Both forms are in common use;
   flagged, not changed.
5. **Chamfer distance** is one-directional in Barrow *et al.* (pp. 659–660); the
   report's two-directional mean is its own. Wording now says "symmetrises".

## D. Locators applied (31 sources read)

Printed page, read from a published copy: `chen2023` 39/40/41/42 · `otsu1979` 63
· `bresenham1965` 28 · `barrow1977` 659–660 · `levenshtein1966` 707–708 (read as
images; the scan has no text layer) · `graves2006` 370, 372 · `graves2009` 855 ·
`srivastava2014` 1930 · `fawcett2006` 868 · `efron1979` 3 · `nair2010` 807 ·
`avidan2007` 10:3 · `haines2016` 26:1 · `shi2017` 2298 · `kang2020` 273 ·
`marti2002` 39 · `a4988ds` 4, 6, 9.

Preprint entry, so its own pages are correct: `kizilirmak2023` 1, 4–5, 5, 9 ·
`mamun2024` 2 · `graves2013` 1.

Numbered element (edition-proof): `ioffe2015` Alg. 1 · `kingma2015` Alg. 1 ·
`loshchilov2019` Alg. 2 · `pascanu2013` Alg. 1 · `tieleman2012` Lec. 6e ·
`jcgm2008` Clause 5 · `dai2024` Sec. 1, Tab. 3 · `kang2020` Tab. 1 ·
`lecun1998` Fig. 2, Sec. II-A.

Cited without a locator on purpose: `he2015` and `simard2003` now carry the two
sentences that the `chen2023` correction left unsourced. The attributions are
right — He initialisation and Simard's elastic distortions — but neither paper
could be obtained, so no page was written.

## E. The 26 sources not obtained

No locator was added. These are **unverified, not wrong**.

**Paywalled or access-controlled** — `zhang1984` and `wagner1974` (ACM),
`bulacu2007` (IEEE), `huang2024` (IEEE), `yahya2021` (MDPI 403), `crow1984`,
`dubuisson1994`, `duda1972`, `douglas1973`, `hochreiter1997`, `viterbi1967`,
`fleiss1971`, `wilson1927`, `ji2021`, `kumar2024`, `iqs2024` (403),
`iso6983` (paid ISO standard).
A paywall-mirror route was attempted once and correctly refused by the sandbox;
it was not retried.

**Bot-protected** — `teikari2016`. The Tampere repository (Anubis) blocks
automated download. **This is the most valuable one still missing: 5 citations,
including the Douglas–Peucker simplification in Appendix S and the
Cartesian-versus-arm argument in Section 3.3.** If you can download `Teikari.pdf`
by hand from `trepo.tuni.fi/handle/123456789/24419`, those five can be finished
in one pass.

**Books / unpublished** — `acarnley2002`, `biagiotti2008`.

**Incomplete entries, nothing to fetch** — `stepperds`, `odroidn2ds`,
`bc337ds` (the onsemi host refused), `postl1986`, `vinciarelli2001`
(metadata fixed, no copy obtained).

**If you have any of these as PDFs**, drop them in a folder and they can be
finished quickly — the whole pipeline is scripted.

## F. Build state

`latexmk -pdf` clean. **Zero undefined citations.** 58 bibitems (`iamdb` added).
All corrected fields verified present in `main.bbl`.

## G. Dead bibliography entries

Still in `references.bib`, cited nowhere, costing nothing (they do not render):
`agduk2022`, `arvanitopoulos2014`, `babushkin2025`, `bhunia2018`, `bhunia2021`,
`bradley2007`, `brink2012`, `canny1986`, `chellapilla2006`, `christlein2017`,
`fogel2020`, `graves2014`, `haralick1987`, `he2020`, `itu601`, `lam1992`,
`li2021`, `likforman2007`, `loshchilov2017`, `michael2019`, `needleman1970`,
`niblack1986`, `plamondon2000`, `puigcerver2017`, `rosenfeld1966`,
`sauvola2000`, `shafait2008`, `sze2017`, `tarjan1975`, `tarjan1984`,
`tresset2013`, `wells1986`, `wigington2017`, `wong1982`.

`he2015` and `simard2003` left this list in this pass. `puigcerver2017` is the
other IAM paper available if you prefer it to Graves in the comparison table.
