"""Illustrate final stored glyph geometry; no model or hardware calls."""
from pathlib import Path
import hashlib
import json
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
import pypdfium2 as pdfium

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROFILES = ROOT / 'Software/CNN/NOGIT/StyleProfiles10'
FIGS = HERE / 'Figures'

def signature(g):
    return json.dumps(g['strokes'], separators=(',', ':'))

def main():
    raw_path = PROFILES / 'dylan.json'
    curated_path = PROFILES / 'dylan_curated.json'
    raw = json.loads(raw_path.read_text(encoding='utf-8'))
    curated = json.loads(curated_path.read_text(encoding='utf-8'))
    ch = 'a'
    variants = curated['glyphs'][ch]
    panels = [(f'Final variant {i}', g) for i,g in enumerate(variants[:4])]
    panels += [('Final real medoid', curated['idealMedoid'][ch]),
               ('Final median prototype', curated['idealGlyphs'][ch])]
    all_pts = [p for _,g in panels for s in g['strokes'] for p in s]
    xlo, xhi = min(p[0] for p in all_pts), max(p[0] for p in all_pts)
    ylo, yhi = min(p[1] for p in all_pts), max(p[1] for p in all_pts)
    xlo, xhi = min(0,xlo)-.10, max(1,xhi)+.10
    ylo, yhi = min(-.1,ylo)-.08, max(1.1,yhi)+.08
    FIGS.mkdir(exist_ok=True)
    dest = FIGS / 'design-glyph-profile-example.pdf'
    c = canvas.Canvas(str(dest), pagesize=(720, 400))
    c.setTitle('Author D stored glyph variants and curation')
    c.setFont('Helvetica-Bold', 16)
    c.drawString(14, 379, f"Author D: stored character '{ch}'")
    c.setFont('Helvetica', 12)
    c.drawString(14, 363, f"Final profile: {len(variants)} variants. Four examples and two representatives, all at the same scale.")
    for j,(title,g) in enumerate(panels):
        left, bottom = 14+(j%3)*237, 197-(j//3)*170
        c.setFillColor(HexColor('#111111'))
        c.setFont('Helvetica-Bold', 12)
        c.drawString(left, bottom+139, title)
        k = min(205/(xhi-xlo), 112/(yhi-ylo))
        ox, oy = left+15-xlo*k, bottom+17-ylo*k
        c.setStrokeColor(HexColor('#b0b7bf')); c.setDash(3,3)
        for y in (0,1):
            c.line(left+10, oy+y*k, left+217, oy+y*k)
        c.setDash()
        c.setFillColor(HexColor('#66717e')); c.setFont('Helvetica', 11)
        c.drawString(left+11, oy+2, '0')
        c.drawString(left+11, oy+k+2, '1 xh')
        c.setLineWidth(1.4); c.setStrokeColor(HexColor('#173f67'))
        for s in g['strokes']:
            if len(s)<2:
                continue
            p=c.beginPath(); p.moveTo(ox+s[0][0]*k,oy+s[0][1]*k)
            for x,y in s[1:]:
                p.lineTo(ox+x*k,oy+y*k)
            c.drawPath(p)
        c.setFillColor(HexColor('#111111')); c.setFont('Helvetica', 11)
        c.drawString(left, bottom+3, f"prior distance {g.get('priorD',float('nan')):.3f}; {len(g['strokes'])} strokes")
    c.setFont('Helvetica', 11)
    c.drawString(14, 10, 'The medoid selects an actual example; the prototype combines compatible paths.')
    c.save()
    pdf=pdfium.PdfDocument(str(dest))
    pdf[0].render(scale=1.8).to_pil().save(dest.with_suffix('.png'))
    facts={
        'writer':'dylan', 'character':ch, 'final_variants':len(variants),
        'panels':[{'label':title,'priorD':g.get('priorD'),'strokes':len(g['strokes'])} for title,g in panels],
        'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (raw_path,curated_path)},
        'meaning':'Actual final stored profile geometry; not a new performance test.'
    }
    (HERE/'PROFILE_EXAMPLE_PROVENANCE.json').write_text(json.dumps(facts,indent=2),encoding='utf-8')
    print(json.dumps(facts,indent=2))

if __name__=='__main__':
    main()
