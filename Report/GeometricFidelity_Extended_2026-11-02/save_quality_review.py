from pathlib import Path
import json
B=Path(__file__).resolve().parent
physical={
'joined glyphs or word fragments':['E00116','E00073','E00084','E00092','E00148','E00149','E00151','E00152','E00154','E00156','E00158','E00160','E00166','E00168','E00171','E00194','E00208','E00211']}
reference={
'joined neighbouring letters':'''R00143 R00177 R00289 R00316 R00370 R00379 R00157 R00168 R00187 R00171 R00150 R00278 R00281 R00331 R00371 R00141 R00367 R00146 R02572 R00819 R00573 R00110 R02392 R00055 R02464 R00084 R02387 R02052 R02562 R00053 R00079 R02498 R01574 R00014 R01408 R01802 R01778 R01323 R01471 R01537 R00015'''.split(),
'incomplete fragment or isolated noise, not an unambiguous complete character':'''R00175 R00287 R00280 R00166 R00368 R00176 R00288 R00315 R00366 R00378 R00262 R00390 R00169 R00186 R00969 R00555 R00996 R01153 R00817 R00136 R00832 R00102 R02441 R00072 R02391 R02386 R02179 R02463 R02146 R02561 R00071 R02249 R01317 R01623 R01580 R01775 R01344 R01223 R01538 R01575 R01579 R01801 R01363'''.split()}
x=json.loads((B/'candidates.json').read_text());known={r['id'] for r in x['physical']+x['references']};bad=[]
for side,groups in [('physical',physical),('reference',reference)]:
 for reason,ids in groups.items():
  for uid in ids:
   assert uid in known,uid;bad.append(dict(id=uid,kind=side,reason=reason))
# Identity and complete-glyph reviews were made before geometry scores were calculated.
# Circle-shaped i dots in Abhinav's real handwriting were inspected in full-line context
# and retained. Unusual but complete letterforms are not removed for visual quality.
(B/'quality_review.json').write_text(json.dumps({'stage':'before measurement','excluded':bad,'notes':['Every proposed physical glyph and every selected bank glyph was visually reviewed.','Unusual letterforms are retained when the labelled source context confirms them, including Abhinav circle-shaped i dots.','Damaged but correctly isolated physical strokes, such as the blot on Dylan adaptive a, are retained as reproduction errors.','No reference replacement or exclusion is based on F-score.']},indent=2))
reject={r['id'] for r in bad};accepted={k:[r for r in x[k] if r['id'] not in reject] for k in ['physical','references']};accepted['protocol']='PROTOCOL.md';accepted['quality_review']='quality_review.json'
(B/'accepted_candidates.json').write_text(json.dumps(accepted,indent=2));print('Accepted',len(accepted['physical']),len(accepted['references']),'Excluded',len(bad))
