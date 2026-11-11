from pathlib import Path
import re
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output' / 'Live_System_Evaluator_Guide_2026-11-11'
OUT.mkdir(parents=True, exist_ok=True)
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
sec.top_margin, sec.bottom_margin = Cm(1.8), Cm(1.8)
sec.left_margin, sec.right_margin = Cm(1.7), Cm(1.7)
sec.footer_distance = Cm(0.8)
styles = doc.styles
for name in ['Normal', 'Title', 'Subtitle', 'Heading 1', 'Heading 2', 'List Number']:
    st = styles[name]
    st.font.name = 'Calibri'
    st.font.color.rgb = RGBColor(0, 0, 0)
    st.font.size = Pt(10.5)
    st.paragraph_format.space_after = Pt(6)
    st.paragraph_format.line_spacing = 1.06
styles['Title'].font.size = Pt(24)
styles['Title'].font.bold = True
styles['Title'].paragraph_format.space_after = Pt(10)
styles['Subtitle'].font.size = Pt(12)
styles['Heading 1'].font.size = Pt(15)
styles['Heading 1'].font.bold = True
styles['Heading 1'].paragraph_format.space_before = Pt(15)
styles['Heading 1'].paragraph_format.space_after = Pt(7)
styles['Heading 1'].paragraph_format.keep_with_next = True
for st in styles:
    if st.type == 1 and ('Heading' in st.name or st.name in ['Title', 'Subtitle']):
        st.font.color.rgb = RGBColor(0, 0, 0)
for element in styles.element.iter():
    for border in list(element.findall(qn('w:pBdr'))):
        element.remove(border)

footer = sec.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
footer.add_run('Handwriting Robot  |  ')
fld = OxmlElement('w:fldSimple')
fld.set(qn('w:instr'), 'PAGE')
footer._p.append(fld)
for r in footer.runs:
    r.font.size = Pt(8)

def shade(cell, value):
    sh = OxmlElement('w:shd')
    sh.set(qn('w:fill'), value)
    cell._tc.get_or_add_tcPr().append(sh)

def table(lines):
    parsed = [[v.strip() for v in line.strip().strip('|').split('|')] for line in lines]
    rows = [parsed[0]] + parsed[2:]
    t = doc.add_table(rows=0, cols=len(rows[0]))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    widths = [4.2, 4.2, 9.2] if len(rows[0]) == 3 else [5.8, 11.8]
    for col, width in zip(t.columns, widths):
        col.width = Cm(width)
    borders = OxmlElement('w:tblBorders')
    for side in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
        b = OxmlElement('w:' + side)
        for key, val in [('val', 'single'), ('sz', '4'), ('color', 'D9D9D9')]:
            b.set(qn('w:' + key), val)
        borders.append(b)
    t._tbl.tblPr.append(borders)
    for i, values in enumerate(rows):
        row = t.add_row()
        no_split = OxmlElement('w:cantSplit')
        row._tr.get_or_add_trPr().append(no_split)
        if i == 0:
            repeat = OxmlElement('w:tblHeader')
            row._tr.get_or_add_trPr().append(repeat)
        for j, (cell, value) in enumerate(zip(row.cells, values)):
            cell.width = Cm(widths[j])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            pr = cell._tc.get_or_add_tcPr()
            margins = OxmlElement('w:tcMar')
            for side in ['top', 'left', 'bottom', 'right']:
                m = OxmlElement('w:' + side)
                m.set(qn('w:w'), '90')
                m.set(qn('w:type'), 'dxa')
                margins.append(m)
            pr.append(margins)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.line_spacing = 1.03
            run = p.add_run(value)
            run.font.size = Pt(9)
            if i == 0:
                p.paragraph_format.keep_with_next = True
                shade(cell, '284658')
                run.bold = True
                run.font.color.rgb = RGBColor(255,255,255)
            elif i % 2 == 0:
                shade(cell, 'F3F6F8')
    doc.add_paragraph().paragraph_format.space_after = Pt(2)

lines = (Path(__file__).parent / 'content.md').read_text(encoding='utf-8').splitlines()
i = 0
while i < len(lines):
    line = lines[i].strip()
    if not line:
        i += 1
        continue
    if line.startswith('|'):
        block = []
        while i < len(lines) and lines[i].strip().startswith('|'):
            block.append(lines[i]); i += 1
        table(block)
        continue
    if line.startswith('# '):
        doc.add_paragraph(line[2:], 'Title')
    elif line.startswith('## '):
        doc.add_paragraph(line[3:], 'Heading 1')
    elif re.match(r'^\d+\. ', line):
        p = doc.add_paragraph(line)
        p.paragraph_format.left_indent = Cm(0.6)
        p.paragraph_format.first_line_indent = Cm(-0.6)
        p.paragraph_format.space_after = Pt(2)
    else:
        p = doc.add_paragraph(line)
        if i in [1, 2]:
            p.style = styles['Subtitle']
        if re.match(r'^S\d+  ', line):
            p.paragraph_format.space_after = Pt(5)
            for r in p.runs:
                r.font.size = Pt(9)
        if line.startswith('Paths are relative'):
            p.paragraph_format.keep_with_next = True
    i += 1

doc.core_properties.title = 'Handwriting Robot Live System Evaluation Guide'
doc.core_properties.subject = 'Inference and reproduction controls with expected effects and evaluator questions'
doc.core_properties.author = 'D R Braune'
doc.core_properties.keywords = 'handwriting robot, evaluation, classification, reproduction, coefficients'
output = OUT / 'Handwriting_Robot_Live_System_Evaluation_Guide.docx'
doc.save(output)
print(output)
print('Tables:', len(doc.tables), 'Control rows:', sum(len(t.rows)-1 for t in doc.tables))
