"""Contact sheet of the glyph libraries: one column per character, no stats.

Writes results/GlyphSheet.docx

Each column is a character, the glyphs under it are its stored variants from
most typical downwards, which is lowest prior distance first.  Short columns
just stop early.  Drawing and ranking come from glyphs.py so both documents
show the same shapes in the same order.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

from docx import Document                                       # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT                  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH                   # noqa: E402
from docx.shared import Inches, Pt                              # noqa: E402

import glyphs as G                                              # noqa: E402
import paths                                                    # noqa: E402
import synthesise as syn                                        # noqa: E402

OUT = HERE / "results"
SHEET = OUT / "images" / "sheet"
SHEET.mkdir(parents=True, exist_ok=True)
DOCX = OUT / "GlyphSheet.docx"

TOP = 8              # glyphs shown per character
COLS = 12            # characters per table
PER_PAGE = 2         # tables before a page break
THUMB = 0.40         # inches wide per glyph
TALL = 0.46          # inches tall at most, so a narrow letter like i or l does
                     # not stretch its whole row down the page


def Order(ch):
    """a to z first, then capitals, then digits, punctuation last"""
    for i, ok in enumerate((ch.islower(), ch.isupper(), ch.isdigit())):
        if ok:
            return (i, ch.lower(), ch)
    return (3, ch, ch)


def Block(doc, writer, prof, chars):
    """one table: a column per character, its variants stacked underneath"""
    lib = prof["glyphs"]
    t = doc.add_table(rows=TOP + 1, cols=len(chars))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, ch in enumerate(chars):
        cands = lib[ch]
        ranked = sorted(cands, key=lambda g: g.get("priorD", 0.5))[:TOP]
        core = syn._CoreGlyph(ch, prof)
        frame = G.Frame([g["strokes"] for g in ranked]
                        + ([core[0]] if core else []))
        cell = t.cell(0, j)                       # the character itself on top
        cell.text = ""
        par = cell.paragraphs[0]
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.paragraph_format.space_after = Pt(0)
        r = par.add_run(ch)
        r.bold, r.font.size = True, Pt(13)
        for i in range(TOP):
            cell = t.cell(i + 1, j)
            cell.text = ""
            par = cell.paragraphs[0]
            par.alignment = WD_ALIGN_PARAGRAPH.CENTER
            par.paragraph_format.space_after = Pt(0)
            if i >= len(ranked):
                continue                          # short column, leave it blank
            p = SHEET / ("%s_%s_%d.png" % (writer, G._Safe(ch), i))
            pw, ph = G.Draw(ranked[i]["strokes"], frame, p)
            run = par.add_run()
            if ph / pw > TALL / THUMB:          # tall and narrow, fit by height
                run.add_picture(str(p), height=Inches(TALL))
            else:
                run.add_picture(str(p), width=Inches(THUMB))
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def Main():
    doc = Document()
    for sec in doc.sections:
        sec.left_margin = sec.right_margin = Inches(0.55)
        sec.top_margin = sec.bottom_margin = Inches(0.6)
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(9.5)

    doc.add_heading("Glyph libraries, every writer", 0)
    p = doc.add_paragraph()
    p.add_run("D.R. Braune, 23539195.  Each column is one character, the "
              "glyphs below it are that writer's stored variants from most "
              "typical downwards, up to %d.  Fewer are shown where fewer were "
              "kept.  Drawn from the stored stroke coordinates, with a faint "
              "baseline, by testing/glyphsheet.py" % TOP).italic = True

    for w in paths.AllWriters:
        fp = paths.Profiles / ("%s.json" % w)
        if not fp.exists():
            continue
        prof = json.load(open(fp, encoding="utf-8"))
        chars = sorted((c for c in prof.get("glyphs", {}) if prof["glyphs"][c]),
                       key=Order)
        doc.add_page_break()
        doc.add_heading("Writer %s" % w, 1)
        doc.add_paragraph("%d characters, %d stored variants."
                          % (len(chars),
                             sum(len(v) for v in prof["glyphs"].values())))
        blocks = [chars[i:i + COLS] for i in range(0, len(chars), COLS)]
        for k, b in enumerate(blocks):
            Block(doc, w, prof, b)
            if (k + 1) % PER_PAGE == 0 and k + 1 < len(blocks):
                doc.add_page_break()
        print("  %-9s %d characters in %d tables" % (w, len(chars), len(blocks)))

    doc.save(DOCX)
    print("\nwrote %s" % DOCX)


if __name__ == "__main__":
    Main()
