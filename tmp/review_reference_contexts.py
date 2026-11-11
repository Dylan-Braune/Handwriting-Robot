from pathlib import Path
import json
from PIL import Image,ImageDraw
B=Path('Report/GeometricFidelity_Extended_2026-11-02');x=json.loads((B/'candidates.json').read_text());ids=['R00200','R00241','R00303','R00398','R00366','R00817','R01325'];rows=[r for r in x['references'] if r['id'] in ids];sheet=Image.new('RGB',(1500,len(rows)*190),'white');dr=ImageDraw.Draw(sheet)
for i,r in enumerate(rows):
 a=Image.open(r['source']).convert('RGB');a.thumbnail((1450,110));sheet.paste(a,(20,i*190+35));dr.text((20,i*190+5),f"{r['id']} {r['writer']} expected {r['char']} in {r['word']} word# {r['word_index']} x={r['word_box']} source {r['source']}",fill='black')
sheet.save(B/'review/questionable_contexts.jpg')
print([(r['id'],r['word'],r['char'],r['source'],r['word_box']) for r in rows])
