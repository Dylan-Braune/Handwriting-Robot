# EPR400/EPR402 Final Report — Condensed Requirements
Distilled from `FinalReportTemplate_2026.doc`, `FinalReportTemplate_NOTES.pdf`, and `APPENDIX4_2026.pdf` (Appendix 4 of the study guide, last revised 4 Sept 2026). Read this instead of the source PDFs. Where this file and the Appendix 4 PDF conflict, the PDF wins — but this was written to preserve every strict/compulsory rule, so that should be rare.

**Goal of this document:** give Claude (or the student) everything needed to write a final report that (a) will not be penalised or marked "incomplete", and (b) is as close as possible to the maximum allowed length (70 pages of main report) with every page full of credit-worthy content, so marks are maximised.

---

## 0. Non-negotiable top-level rules

- **Two submission items, both required** (incomplete submission = automatic Project fail):
  1. **The final report** — a single PDF, submitted on the AMS.
  2. **All code** developed (any language/platform: C/C++/C#/Python, Matlab/Octave/Simulink, SPICE/Comsol/NEC/FEKO/PSIM, VHDL, DSP/microcontroller code, PCB design files) — submitted separately on the AMS under "Submission item 2".
- **Deadlines are examination deadlines** — treated exactly like a written exam. **No late submissions under any circumstances.** The AMS closes after the grace period.
- **Preliminary report uploads required**: at 14 days, 7 days, and 3 days before the final deadline (check Project Clickup for exact dates). These aren't graded but (a) back up your work, (b) prove a paper trail of progress. In the final 3 days you may re-upload updated drafts under "final report"; only the last upload before the deadline counts.
- **Use the official template only.** Do NOT build your own template — this is an automatic fail of GA6 (the formatting graduate attribute).
- **No marketing/spin, no foreseeing/falsifying results.** Report must reflect the ABSOLUTE TRUTH of the project's status on the day of submission. If something doesn't work, say so.
- **No revisions after submission.** What you submit is final.
- Poor formatting/grammar/language quality can lead to **examination refusal, a supplementary exam, or failing the module.**

---

## 1. Overall report structure

The final report PDF contains, in this order:

| Part | Content | Page numbering |
|---|---|---|
| Part 1. Preamble | Title page + declarations + ToC + abbreviations | Roman numerals (i, ii, iii…) |
| Part 2. Project definition | Unaltered, approved Project Proposal (landscape) | No page numbers |
| Part 3. Main report | Literature study → Approach → Design & Implementation → Results → Discussion → Conclusion → References | Normal numerals, starting at 1 |
| Appendix (optional) | Additional results only | Continues normal numerals, excluded from page-count limits |

Each "Part" and each top-level numbered section (1–7) starts on a **new page**.

### Page-count target (main report = Part 3 only)
- **Minimum 50 pages, maximum 70 pages.** Fewer than 50 ⇒ "usually incomplete and probably not accepted." More than 70 ⇒ **not allowed under any circumstances.**
- Page count **excludes** Part 2 (Project Proposal) and **excludes** the References section and the optional Appendix.
- **Target 68–70 pages** to maximise credit while staying safely under the hard cap — leave a small buffer (~1–2 pages) in case of last-minute additions, but do not pad with fluff (see §6, credit rules).
- Section-length guidelines (not hard rules — a rough budget to hit ~70 pages total):

| Section | Guideline length |
|---|---|
| 1. Literature study | 4–8 pages |
| 2. Approach | 1–2 pages |
| 3. Design and implementation | 30–40 pages (**the core — spend most of your page budget here**) |
| 4. Results (4.2 Qualification tests especially) | 10–15 pages incl. figures |
| 5. Discussion | 6–8 pages |
| 6. Conclusion | 1–4 pages (at least 1, no more than 4) |
| 7. References | excluded from count |

If you're significantly outside a guideline, something is wrong — either you're under-detailing or padding. To use the full 70 pages, maximise depth in **Design and Implementation** (richest section) and **Results**, not by inflating Discussion/Conclusion/Literature beyond their natural length.

---

## 2. Part 1 — Preamble (page i onward, Roman numerals)

Written in **12 pt** body font (template says 11 or 12 pt serif — e.g. Times New Roman, Cambria, Baskerville, Caledonia; final formatting table says use 12 pt — follow the template's 12 pt and NOT sans-serif anywhere in body text). **This is the ONLY part of the report that may be written in the first person.**

Must contain, in this exact order, with these exact (adaptable) headings and near-exact wording:

1. **Opening one-sentence description** of the work, first person, e.g.:
   > This report describes work that I have completed in my final year project, developing a low power communication system for long distance data transmission.
2. **"1. Project proposal and technical documentation"** — include *exactly*:
   > This main report contains an unaltered copy of the approved Project Proposal (as Part 2 of the report).
   > All the code that I developed appears as a separate submission on the AMS.
3. **"2. Project history"** — must explicitly state whether/how this project builds on a prior project, citing by author/year, and explicitly distinguish what was done by you vs others. Mandatory paragraph, cannot be skipped. Example of the required content/style:
   > This project extends a previous project by Owens (2025), but my implementation is completely new. I borrowed some ideas from Owens, but only regarding the filter design procedure and nothing else.
   > I used library code for low level hardware integration between the DSP board and the camera and did not develop this myself.
   - If your project does **not** build on a prior project, say that explicitly instead (do not omit the paragraph).
4. **"3. Language editing"** declaration — must name a real language editor and include their **signature** (handwritten/scanned/electronic) and date, OR attach a confirmation email from the editor at the end of the report (and write "proof of language editing is attached" in place of the signature). Exact required wording (only the blank is filled in):
   > This document has been language edited by a knowledgeable person. By submitting this document in its present form, **I declare that this is the written material that I wish to be examined on**.
   > My language editor was ______________.
   **Report is rejected without this.**
5. **"4. Plagiarism declaration"** — must appear **verbatim** (only the blank for your name filled in):
   > I, [NAME] understand what plagiarism is and have carefully studied the plagiarism policy of the University. I hereby declare that all the work described in this report is my own, except where explicitly indicated otherwise. Although I may have discussed the design and investigation with my study leader, fellow students or consulted various books, articles or the internet, the design/investigative work is my own. I have mastered the design and I have made all the required calculations in my lab book (and/or they are reflected in this report) to authenticate this. I am not presenting a complete solution of someone else or that I found somewhere.
   > Wherever I have used information from other sources, I have given credit by proper and complete referencing of the source material so that it can be clearly discerned what is my own work and what was quoted from other sources. I acknowledge that failure to comply with the instructions regarding referencing will regarded as plagiarism. If there is any doubt about the authenticity of my work, I understand that I may have to attend an oral ancillary examination/evaluation about the work.
6. **"5. AI usage declaration"** — must appear, **modified to reflect actual facts**:
   > The author confirms that no artificial intelligence tools, including language models or code-generation tools, were used in the preparation, writing, or any analysis that appears in this report. All content, design, explanations, data analysis, and code were developed solely by the author.
   - ⚠️ **This must be truthful.** If AI tools were in fact used anywhere (including for this report-drafting process), that fact must be disclosed here instead of the boilerplate denial — fabricating this declaration is itself a plagiarism/integrity violation. Confirm with the student exactly what to state before finalising.
7. **"6. Software library usage declaration"** — default boilerplate (modify to list actual exceptions, e.g. hardware-interfacing libraries, FFT library, with source):
   > Unless explicitly stated otherwise in the next paragraph, the author confirms that no software library functions, packages, or toolboxes were used in the design, implementation, or analysis presented in this report. All code and computations were developed entirely by the author.
   - If there **are** exceptions, add a follow-up paragraph naming the exact library function(s) used and their source. Appendix 4's own examples of the required style:
     > Example 1: The only exception to the above is that library functions were used for software interfacing between the camera and the embedded platform. The specific library functions that were used were … and these may be found at…
     > Example 2: The only exception was that the FFT was implemented by using … that may be found here: …
8. **"7. Unaltered Project Proposal"** — verbatim:
   > I certify that the Project Proposal appearing as the Introduction section of the report is an unaltered copy of the approved Project Proposal.
9. **Signature block** — "With my signature below, I hereby certify that all statements, declarations, and information presented above are complete." + signature + date. **Report rejected without this signature.**
10. **Table of Contents** (new page, still Roman numerals — typically page ii/iii). Lists only what follows the Preamble (not the Preamble itself, and not the ToC's own entry). Structure, exactly in this order:
    ```
    Part 2. Project definition: approved Project Proposal        (no page numbers listed for this Part)
        1. Project description
        2. Technical challenges in this project
        3. Functional analysis
        4. System requirements and specifications
        5. Field conditions
        6. Student tasks

    Part 3. Main report                                           (page numbers listed from here on)
        1. Literature study ........................................ 1
            1.1 <your subheading> ................................... 1
                1.1.1 <your sub-subheading> ......................... 1
        2. Approach ................................................. 4
            2.1 <your subheading> ................................... 4
        3. Design and implementation ................................ 7
            3.1 Design summary ....................................... 7
            3.2 System block diagram .................................. 8
            3.3 <your subheading> .................................... 12
            ...
        4. Results .................................................. 34
            4.1 Summary of results achieved .......................... 34
            4.2 Qualification tests .................................. 34
                Qualification test 1: <title>
                    Objectives of the test or experiment ............ 34
                    ... (all 7 sub-subheadings, each with page no.)
                Qualification test 2: <title>
                    ...
        5. Discussion ............................................... 50
            5.1 Critical evaluation of the design .................... 50
                5.1.1 Interpretation of results ...................... 50
                5.1.2 Critical evaluation ............................ 54
                5.1.3 Unsolved problems .............................. 58
                5.1.4 Strong points of the design ..................... 60
                5.1.5 Expected failure conditions ..................... 61
            5.2 Considerations in the design ........................... 50 (or correct page)
                5.2.1 Ergonomics ..................................... 61
                5.2.2 Health and safety .............................. 61
                5.2.3 Environmental impact ............................ 62
                5.2.4 Social and legal impact ......................... 63
                5.2.5 Ethics clearance ................................ 63
        6. Conclusion ................................................ 64
            6.1 Summary of the work completed ......................... 64
            6.2 Summary of the observations and findings ............... 65
            6.3 Contribution ......................................... 67
            6.4 Suggestions for future work ........................... 69
        7. References ................................................ 70
        Appendix (if included) ....................................... 72
    ```
    - Headings shown in black above are fixed/required main headings (your own wording must match them, except where the spec says "student-defined"); your own sub-/sub-subheadings (blue in the official template) are freely named by you but must still be listed with correct page numbers.
11. **List of Abbreviations** (new page, Roman numerals). Define *every* abbreviation used in the report, including common ones (SNR, rms, DC, AC, RAM, FPGA, GSM, etc.) — and also define each on first use in the body text. Also define any special notation here (e.g., scalar vs. bold vector notation).

Formatting note: page header on every page = your initials+surname (left) and section name (right), per the template.

---

## 3. Part 2 — Project definition (landscape, no page numbers)

- Heading page: just the heading **"Part 2. Project definition: approved Project Proposal"** followed by exactly this sentence, rest of page blank:
  > This section contains the problem identification in the form of the complete, approved Project Proposal, unchanged from the final approved version that appears on the AMS.
- Next pages: paste in your **final approved Project Proposal exactly as submitted/approved**, in **landscape orientation**, unchanged formatting, unchanged section/page numbering.
- Allowed edits only: (i) fix grammar errors, (ii) fix formatting glitches. **Do NOT renumber sections or change any specification/content** — altering specifications at this point can be construed as fraud.
- Remove all highlighting/comments **except** anything in the "Feedback" block on page 1 of the proposal — that must be retained and must appear in the final report.
- If an Engineering Change Proposal was submitted, use the final approved (post-ECP) proposal.

---

## 4. Part 3 — Main report (starts page 1, normal numerals)

Heading page: just "Part 3. Main report", rest blank. Then literature study starts on the next page at page 1.

### 4.1 Section "1. Literature study" (4–8 pages)
- Goal: show you acquired/evaluated the requisite background knowledge and placed your project in context — an *extension* of the first-semester literature study.
- Structure:
  (i) **Background and context** (3–5 pages) — literature-grounded discussion of the problem domain and how others have solved similar/related problems; organise as a unified narrative/argument, **not** an article-by-article list. Use subheadings as needed (e.g., "1.1 Background and context", "1.1.1 Digital modulation techniques", …).
  (ii) **Summary of how you applied this background** (1–3 pages) — what you learned and how it informed your design.
- **At least six references to technical journal articles** (IEEE etc. preferred), plus you may cite good technical books/websites (not amateur sites). One-to-one correspondence between in-text citations and the reference list (no orphan references, no citations missing from the list).
- Referencing style: IEEE-style or an author-date style (APA/Harvard) — consistent throughout. Confirm your study leader's preference.

### 4.2 Section "2. Approach" (1–2 pages)
- Brief, high-level "storyline" explaining **why** and **how** you approached the problem — bridges Part 2 (requirements) and Section 3 (deep technical detail). No deep maths/circuit detail here.
- Cover: concept-design alternatives considered, trade-offs, your eventual preferred/chosen solution, and how the functional blocks from the Project Proposal map onto your first concept design. Use high-level block diagrams/flow diagrams if useful.

### 4.3 Section "3. Design and implementation" (30–40 pages — THE most important section, deepest technical content, where most of your marks and page budget should go)

**Compulsory subsection 3.1 — "Design summary"** (heading text must appear exactly):
- One intro sentence, exact required wording:
  > This section summarises the project design tasks and how they were implemented (see Table 1).
- **Compulsory Table 1 — "Table 1. Design summary."** (caption below table, bold, this exact wording) with **exactly 3 columns**, these exact headers:

  | Deliverable or task | Implementation | Completion of deliverable or task, and section in the report |
  |---|---|---|

  - **Column 1 — Deliverable or task:** one specific task per row (new row per task), worded as a short task name, e.g. *"Design of a PCB for the main electronics"*, *"Development of optimisation routine"*, *"DC-DC converters had to be designed and implemented by the student"*. Rows must correspond to (i) **every deliverable** listed in line 5 of the Section 4 table of your approved Project Proposal, AND (ii) **every design/implementation task** listed in Section 6.1 of your Project Proposal — both sets must be covered, so cross-check against those two lists before finalising the table.
  - **Column 2 — Implementation:** one or two sentences of *what you actually did* to complete that task, and explicitly flag if anything was off-the-shelf/toolbox-assisted rather than first-principles, e.g. *"The PCB design was completed using SimpleCAD. This was done from first principles"*, or *"Optimisation was completed in Matlab, but the Optimization Toolbox was used, and while some code was developed from first principles, numerical methods for optimisation were taken off the shelf"*, or *"The student did not complete the inverter. The design was completed and simulated, but the implementation in hardware did not work correctly."*
  - **Column 3 — Completion of deliverable or task, and section in the report:** literally the word **"Completed"** or **"Incomplete"** (truthful as of submission day — never guess what might work by the oral exam), followed by the exact subsection number in the report where the full detail can be found, e.g. *"Completed. Section 4.2.8"* or *"Incomplete. Section 4.2.9"*.
  - Only include your own design/implementation tasks — never off-the-shelf/non-design tasks like "write report" or "source components".
  - **Minimal copy-paste template:**
    ```
    Table 1. Design summary.
    | Deliverable or task                                   | Implementation                                                                 | Completion of deliverable or task, and section in the report |
    |--------------------------------------------------------|---------------------------------------------------------------------------------|----------------------------------------------------------------|
    | <deliverable 1, from Proposal §4 line 5 / §6.1>         | <what you did, first-principles vs toolbox, flagged honestly>                   | Completed / Incomplete. Section <X.Y>                          |
    | <deliverable 2>                                         | <...>                                                                           | Completed / Incomplete. Section <X.Y>                          |
    | <one row per deliverable/task — do not skip any>         | <...>                                                                           | Completed / Incomplete. Section <X.Y>                          |
    ```

**Compulsory subsection 3.2 — "System block diagram"**:
- Full system-level block diagram (hardware and/or software, separate or combined) reflecting your **final** design (related to, but may differ from, the Project Proposal's functional block diagram), with an accompanying systems-level text description of how it works.

**Subsections 3.3 onward (your own headings/order, but must include Hardware design and/or Software design as applicable):**
- Typical pattern (adapt names/order/numbering to your project): Theoretical analysis → Modelling → Simulation → Hardware design → Hardware implementation → Software design → Software implementation → Optimisation → Statistical analysis.
- **Hardware design** (if applicable): complete circuit diagrams (ideally one master diagram showing all modules, plus detail diagrams for complex sub-circuits). Component names/values in figures ≥ **6-point font**. Must accompany every circuit diagram with **design equations and calculated component values** and descriptive text of circuit operation — a schematic with no equations/values earns no credit.
- **Software design** (if applicable): flowcharts/process diagrams (UML if used), text description of modules, representative code **snippets** (not full listings — full code goes in the separate code submission; code snippets in the report are formatted as figures, max ~1 page each unless the whole program is only a few pages). State explicitly: "Complete code has been submitted separately on the AMS."
- **Give ALL necessary technical detail**: theoretical analysis/mathematics that derive the solution, design equations, calculated component values, simulations performed, experiments conducted, software/hardware developed. Two canned examples of *insufficient* detail to avoid: (a) describing an image-recognition system only via code/flowcharts without the underlying maths; (b) showing a circuit layout with no design equations/component values.
- **Credit rule (maximise marks):** you only get credit for *your own, first-principles design/implementation work*. Off-the-shelf solutions, datasheet copy-paste, library code, copied UI/data — **zero credit**, so don't waste page budget describing them in detail (mention briefly, cite source, move on). Spend the bulk of your space on your most important/highest-priority design aspects (ordered as per Project Proposal Section 4 requirement priority). No credit for explaining basic textbook concepts (e.g. Ohm's law) — focus only on your own contribution and anything not already covered in other modules.
- Always explain **why** a design choice/test was made, not just what — avoid a "disconnected list of paragraphs"; keep one clear storyline/golden thread through the section. State all assumptions explicitly.

### 4.4 Section "4. Results" (10–15 pages incl. figures)

**Compulsory subsection 4.1 — "Summary of results achieved"**:
- **Compulsory Table 2 — "Table 2. Summary of results achieved."** (caption below table, bold), **exactly 3 columns**, these exact headers:

  | Intended outcome | Actual outcome | Location in report |
  |---|---|---|

  - Rows are split into **two bold group-header rows inside the table** (not separate tables), in this exact order:
    1. **"Core mission requirements and specifications"** — one row per (i) core mission requirement of the product and (ii) its corresponding target specification, taken directly from **Section 4** of the approved Project Proposal.
    2. **"Field condition requirements and specifications"** — one row per field condition documented in **Section 5** of the approved Project Proposal (e.g. operating under actual environmental conditions, real-time noisy data, etc.).
  - **Column 1 — Intended outcome:** the requirement/specification/field condition exactly as stated (or a close paraphrase) in the approved Project Proposal — a text description, or text + measurable specification, e.g. *"The motor should be able to rotate at a speed that can propel the vehicle forward at at least 10 km/h."*
  - **Column 2 — Actual outcome:** the real, measured result for that row — **must be literally true on submission day**, including honestly reporting failures/unmet specs, e.g. *"The measured maximum speed was 12.6 km/h."* or *"The system could not deliver the required power into the load. The system could deliver 800 Watts into the load before overheating."*
  - **Column 3 — Location in report:** cross-reference to the exact subsection with the full qualification-test detail, e.g. *"Section 4.2.6"*.
  - **Minimal copy-paste template:**
    ```
    Table 2. Summary of results achieved.
    | Intended outcome                                            | Actual outcome                          | Location in report |
    |--------------------------------------------------------------|------------------------------------------|---------------------|
    | Core mission requirements and specifications (bold row, spans all 3 columns) |
    | <requirement/spec 1, from Proposal §4>                        | <measured/actual result, honest>         | Section <X.Y.Z>     |
    | <requirement/spec 2>                                          | <...>                                     | Section <X.Y.Z>     |
    | Field condition requirements and specifications (bold row, spans all 3 columns) |
    | <field condition 1, from Proposal §5>                         | <measured/actual result, honest>         | Section <X.Y.Z>     |
    | <field condition 2>                                           | <...>                                     | Section <X.Y.Z>     |
    ```

**Subsection 4.2 — "Qualification tests"** (detailed — the bulk of this section's pages):
- Expands on *how* each result in Table 2 was obtained/verified. Document qualification tests **in order of importance**, matching the priority order of requirements in Project Proposal Section 4 (requirement 1 first, etc.).
- Tests must verify the **integrated system as a whole**, not isolated subsystems — after reading this section an examiner should have no doubt whether the full system meets its requirements.
- Typically **5–6 tests max** (sometimes only 2–3), each test block formatted **exactly** like this (bold heading with NO section number, sub-subheadings in italic with no number/bullet, each followed by prose/figures/tables):

  ```
  Qualification test 1: Measurement of closed-loop controlled vehicle speed
      Objectives of the test or experiment
          <prose>
      Equipment used
          <prose — exact equipment make/model, e.g. "Tektronix TDS2002C oscilloscope">
      Test setup and experimental parameters
          <prose>
      Steps followed in the test or experiment
          <prose, numbered steps if useful>
      Results or measurements
          <figures/tables/graphs of real measured data>
      Observations
          <objective description only — no interpretation>
      Statistical analysis
          <where appropriate; omit heading+content together if genuinely not applicable to this test>

  Qualification test 2: <next title, matching next-priority requirement>
      Objectives of the test or experiment
      Equipment used
      Test setup and experimental parameters
      Steps followed in the test or experiment
      Results or measurements
      Observations
      Statistical analysis
  ```
  - The 7 sub-subheadings above are: *Objectives of the test or experiment · Equipment used · Test setup and experimental parameters · Steps followed in the test or experiment · Results or measurements · Observations · Statistical analysis*.
  - Under **Equipment used**, give exact make/model (e.g. "Tektronix TDS2002C oscilloscope"); include any test jig/software you built specifically for testing — this may instead warrant its own subsection before 4.2, e.g. "4.2 Test equipment developed", pushing qualification tests to 4.3.
  - Under **Results or measurements**, present graphically/tabularly wherever possible (captured scope traces, BER-vs-SNR graphs, error-vs-time plots, etc.).
  - Under **Observations**, strictly **objective** description of what any engineer would see in the data; **no interpretation/conclusions here** (save that for Discussion §5).
  - Under **Statistical analysis**, include where appropriate, omit where not relevant to that particular test.

### 4.5 Section "5. Discussion" (6–8 pages — critically important, heavily weighted in marking/ECSA GA9)

⚠️ Do **not** write "everything works perfectly" — there is no perfect design, and examiners specifically assess your ability to critically self-evaluate (this is graded, part of ECSA Graduate Attribute 9). Be properly motivated and honest, not falsely modest or falsely boastful — e.g. "my system is good at X but not so good at Y" with justification.

**5.1 Critical evaluation of the design** — must include at least these compulsory sub-subheadings (italic, you may add your own):
- *Interpretation of results* (compulsory) — what the results mean, how good/acceptable/poor they are, why they turned out that way, consequences of external factors, why specs were missed where relevant. Expand on design alternatives/trade-offs briefly flagged in Approach.
- *Critical evaluation* (compulsory) — evaluate in hindsight the choices you made (implementation choices, design procedures, methods, experimental protocols) and their technical/economic feasibility.
- *Unsolved problems* (compulsory) — any remaining issues; what you'd do differently with more time/money; explicitly explain any deviation from the Project Proposal's plan.
- *Strong points of the design* (compulsory) — what worked especially well and why.
- *Expected failure conditions* (compulsory) — design limitations; under what conditions/inputs the system will fail, and whether it fails gracefully or catastrophically.

**5.2 Considerations in the design** — compulsory sub-subheadings, each must appear (brief, but a motivated explicit statement even where "not applicable"):
- *Ergonomics* (compulsory) — user interaction aspects: UI, panel layout, control placement, installation ease, reliability.
- *Health and safety* (compulsory) — electrical safety, burn hazards, hearing protection, etc.; if genuinely not applicable, say so with motivation.
- *Environmental impact* (compulsory) — pollution, acoustic/electromagnetic noise, end-of-life recyclability, and how these were addressed.
- *Social and legal impact* (compulsory) — applicable legislation/regulations and compliance; social impact (e.g. contributions to poverty/health/infrastructure access, job creation, cultural issues).
- *Ethics clearance* (compulsory) — state the ethics clearance number if required, or explicitly state that ethics clearance was not required.

### 4.6 Section "6. Conclusion" (1–4 pages; minimum 1, hard max 4)
- Written in **third person** like the rest of the report (never first person, never personal-experience language like "I learnt a lot" — unacceptable).
- Must use exactly these 4 compulsory subheadings:
  - **6.1 Summary of the work completed** — brief factual summary of what was done.
  - **6.2 Summary of observations and findings** — concise summary of key results/interpretation.
  - **6.3 Contribution** — explicitly state what was genuinely *new to you*: new software/theory/hardware mastered beyond prior modules; your own contribution vs. others' (study leader guidance, peer help, code from elsewhere) — be specific and honest about provenance of every component.
  - **6.4 Suggestions for future work** — concrete, specific suggestions for extending/improving the work (design alternatives not attempted due to time/budget).
- Be completely honest about what was and wasn't achieved.
- **Exact worked examples from Appendix 4** (match this level of specificity and technical, third-person register — never emotional/personal language):
  > **6.1 Summary of the work completed**
  > This report describes work carried out on the design of a voice communication system, with the objective of using very low power to achieve reliable wireless communication over long distances.
  > A literature survey was completed on modern communication system design. The hardware and software for a low power communication system was then designed from first principles. At the core of the system is an existing DSP board, and all additional hardware subsystems were designed and implemented. Python code was developed to simulate the system, as well as C code for the DSP, and assembly language code for a PIC processor that resides on the hardware platform that was designed. The system was implemented and several field tests were carried out. A voice communications channel was set up between the University and a site in Midrand. The main result is shown in the BER graph in section 3.3 of the report.
  >
  > **6.2 Summary of observations and findings**
  > Successful voice communication of good quality was achieved over a distance of more than 50 km. It was found that it is possible to achieve long distance communication with low power by employing an appropriate design. Figure 1 in section 3 shows that the bit error rate (BER) is dependent on communication distance.
  > An important discovery was that the modulation scheme plays a major role in determining the relationship between distance achieved and power required. One implication of this is that the carrier frequency and modulation scheme have to be designed carefully, while considering the tradeoff between BER and power.
  >
  > **6.3 Contribution**
  > New software that had to be mastered to complete this project was Comsol. Comsol is a finite element package, which is not something that undergraduate students would usually have any knowledge of and that is not covered in any undergraduate modules.
  > Specifically, the Comsol multiphysics module was used. In order to use this, a thorough understanding of turbulent water flow had to be developed. Literature was consulted, and especially chapter 4 in the textbook by Nicholson (2012) was mastered. The mathematics used in Comsol largely reflects the underlying theory explained in this chapter of the textbook.
  > No new code was developed, but multiple simulations were developed in Comsol. From these, an entirely new implementation was developed without any inputs from the study leader. Once the hardware implementation was completed and several errors were discovered, the study leader provided guidance with investigating the origin of these errors.
  > Code was mostly developed by the student without any reliance on existing libraries. Only code that relates to hardware interfacing was taken directly from existing libraries. Friends in class initially helped with coding in R for statistical analyses, as this was complex and completely new to the student.
  >
  > **6.4 Suggestions for future work**
  > The communication protocol did not work as well as expected and was not immune against noise. Future work should include the investigation of methods that will improve the signal-to-noise-ratio.
  > The processing platform used, the Xeroid S7, was not an optimal choice. Particular difficulties to overcome when selecting a new platform would be the low sampling rate of the on-board ADC.
  > Finally, more experiments are needed to confirm that image recognition is in fact an improvement over existing literature under all noise conditions.

### 4.7 Section "7. References" (new page, excluded from page-count limit)
- **Minimum six references to technical journal articles** (as per Literature study), plus other high-quality technical sources as needed. No amateur sources (e.g. Hackaday, Instructables).
- Every reference in the list must be cited somewhere in the report body, and vice versa (exact one-to-one correspondence).
- Never cite your study leader as "Prof. X" informally — if quoting them, cite as you would any other expert source, per your chosen referencing format.
- Consistent IEEE or author-date (APA/Harvard) style throughout — confirm study leader's preferred format.

### 4.8 Appendix (optional, new page, excluded from page-count limit)
- Use only for **additional results** beyond what's in the main Results section (e.g., more repetitions/examples of a test already summarised in Section 4). The main report must still stand alone with a valid, representative picture of successes *and* failures — don't offload important/required content here just to save page count.

---

## 5. Formatting rules (Section C of Appendix 4) — strict, heavily penalised if wrong

| Item | Rule |
|---|---|
| Template | Use the official provided template only (`FinalReportTemplate_2026.doc`) — do not design your own. |
| Submission format | Single PDF via AMS. |
| Body font | 12 pt serif font (Times New Roman, Baskerville, or Caledonia — template also allows 11 pt per its own note, but formatting table says 12 pt: **use 12 pt to be safe**). Never sans-serif (no Arial/Helvetica) in body text. One font/size throughout body — do not mix. |
| Figures/tables text | Same font consistently across all figures; **min. 8 pt**; sans-serif allowed here. |
| Figure/table captions | Bold; 10–12 pt; sans-serif allowed; captions always appear **below** the figure/table, never above, and must always be referred to in the body text. |
| Headings | Same font family as body. Part headings 24 pt bold lowercase/sentence case; section headings (1.–7.) 18 pt bold Sentence case, underlined margin-to-margin; subheadings 14 pt bold UPPERCASE; sub-subheadings 12 pt bold italic; deeper levels: lowercase 12pt bold italic, then (a)/(b) numbering in bold italic with no further numbering. Headings may be coloured if applied consistently per level. Left-justified. |
| Body justification | Full justification for all body text, table/figure captions. Headings left-justified only. |
| Line spacing | Single. |
| Margins | 25 mm all sides (guideline — template takes priority if it differs slightly; be internally consistent). |
| Paragraphs | No first-line indent; leave one blank line between paragraphs. |
| Page numbers | Bottom right. Roman numerals through List of Abbreviations; normal numerals from Part 3 page 1 onward. Title page has no number. Part 2 (Project Proposal) has no page numbers. |
| Header | Every page: your initials+surname at left margin, section name at right margin. |
| Print | One side of paper only (even though it's a PDF, format as if single-sided — don't design double-sided layouts). |
| New pages | Every main section (1 through 7) and every Part starts on a new page; never start a section/page with a figure — always at least one introductory sentence first. |
| Equations | Numbered sequentially (1), (2)… in round brackets, right-justified (build via a borderless 2-column 1-row table). Centre equation. Define every symbol (italic for symbols). Leave a blank line after (and optionally before). Reference every equation in text as "Equation 1" or "Eq. 1" — never lowercase "equation (1)". Punctuate equations as part of the sentence (end with full stop if sentence ends there). Exact worked example from the spec: `The equation for a straight line is` / `y = mx + c,` right-justified `(1)` / `where m is the slope and c the y-axis intersection. x is the independent variable.` |
| Figures/tables sizing | A4-oriented figures preferred; don't oversize; typical min. size ~8×8 cm; never split a table/figure across two pages. |
| Figure caption exact format | Caption goes **below** the figure, number + caption text **in bold**, e.g.: `Figure 1.` / `The figure shows data as measured for the protection circuitry. Note that the current limits at 10 A.` — figures and tables are numbered in separate sequences starting at 1; never a caption above the figure; always refer to the figure by number in the body text. |
| Figures/tables orientation | Match body text orientation, or rotate with the bottom facing the outer page edge; caption always stays upright regardless. |
| Figures/tables numbering | Number figures and tables in **separate** sequences (Figure 1, Figure 2…; Table 1, Table 2… — not shared numbering). No frames/borders around figures. |
| Code listings | No long code listings in the main report (full code → separate AMS code submission only). Short snippets as figures, ≤1 page each, unless the whole program is only a few pages (then it may appear in the main report). |
| Footnotes | Superscript number right after the word; explanation at bottom of same page below a margin-to-margin line, 8 pt font, numbered sequentially through the whole document. |
| Spelling/grammar | Must be flawless — have someone competent (the declared language editor) check it. Study leader will **not** proofread. |
| Tense | Past tense throughout (project is complete by the time it's read), except where present tense is grammatically correct (e.g. stating specifications). |
| Person/voice | Third person, formal technical register throughout — **except** the Preamble, which is first person. Never conversational/SMS style. Never a chronological personal narrative ("first I did X, then I did Y") — present facts technically, not autobiographically. |

---

## 6. How to maximise marks AND fill up to ~70 pages honestly

1. **Depth over breadth, concentrated in Section 3 (Design and implementation).** Every extra page should come from genuine first-principles technical content you haven't yet shown: additional design equations, derivations, extra circuit/software sub-modules, more simulation results, more detailed component selection justification. Never pad with restated requirements, filler prose, or oversized figures/whitespace.
2. **Only your own, first-principles work earns credit.** Do not spend pages describing off-the-shelf parts, library functions, or datasheet content in detail — a one-line mention + citation is enough; redirect that space to your own derivations/implementation.
3. **Every circuit diagram needs accompanying design equations and calculated component values** — diagrams alone earn zero credit. This is one of the highest-value ways to add substantial, legitimate page count.
4. **Qualification tests (§4.2) are a second high-value area** — document each of your 5–6 most important tests completely (all 7 required sub-subheadings, with graphs/tables of real measured data), rather than being terse.
5. **Discussion (§5) is heavily weighted for ECSA GA9** — do not under-write it. Genuinely critical, well-motivated self-evaluation across all 10 compulsory sub-subheadings is both required for passing and a legitimate way to add strong, high-mark-density pages.
6. **Do not exceed 70 pages under any circumstances** — it is a hard, unconditional cap. Aim for 68–70, not exactly 70, to leave a small safety margin for last-minute edits without tipping over.
7. **Never fabricate or guess future results.** All data, "Completed"/"Incomplete" statuses, and specification compliance must be literally true as of the exact day you submit — this is graded as an integrity matter, not just a technical one, and consequences for dishonesty "may be severe."
8. **Truthfully report the AI usage and software-library declarations** (§2, items 5–6) — do not let maximising marks compromise these; a false declaration is itself a serious violation, independent of report quality.

---

## 7. Quick compulsory-headings checklist (for self-verification before submission)

- [ ] Preamble: 1 Project proposal & documentation · 2 Project history · 3 Language editing (+ signature) · 4 Plagiarism declaration (verbatim) · 5 AI usage declaration (truthful) · 6 Software library usage declaration (truthful) · 7 Unaltered Project Proposal · Signature · ToC · List of Abbreviations
- [ ] Part 2: unaltered, approved, landscape Project Proposal, Feedback block retained
- [ ] 1. Literature study (≥6 journal references, background + application subsections)
- [ ] 2. Approach
- [ ] 3. Design and implementation → 3.1 Design summary (Table 1, 3 cols) · 3.2 System block diagram · Hardware and/or Software design subsections with equations/values
- [ ] 4. Results → 4.1 Summary of results achieved (Table 2, 3 cols) · 4.2 Qualification tests (5–6 tests, 7 sub-subheadings each, in priority order)
- [ ] 5. Discussion → 5.1 Critical evaluation (Interpretation of results · Critical evaluation · Unsolved problems · Strong points · Expected failure conditions) · 5.2 Considerations (Ergonomics · Health and safety · Environmental impact · Social and legal impact · Ethics clearance)
- [ ] 6. Conclusion → 6.1 Summary of work completed · 6.2 Summary of observations and findings · 6.3 Contribution · 6.4 Suggestions for future work
- [ ] 7. References (≥6 journal articles, 1:1 citation correspondence)
- [ ] Appendix (optional, additional results only)
- [ ] Main report (Part 3) page count: 50–70 pages, target ~68–70
- [ ] Code submitted separately on AMS (Submission item 2)
- [ ] Formatting table (§5 above) followed exactly
