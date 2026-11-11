from pathlib import Path
import json
B=Path('Report/GeometricFidelity_2026-11-02')
ann={
'original_owen_09':{'reason':'Visually attach the comma to free, rather than easily. No shape score was inspected.','word_overrides':{'7':{'x1':1601,'y1':113},'8':{'x0':1632}}},
'original_yeukita_04':{'reason':'Exclude the printed/right-page boundary after measurement.','word_overrides':{'12':{'x1':2210}}},
'original_abhinav_03':{'reason':'The cached line contains in from the next line beneath velocity. Remove only that foreign region, leaving the y descender further right.','mask_exclude_rects':[[0,125,150,199]]},
'original_abhinav_04':{'reason':'Visual correspondence shows in is missing from this crop. Its a is at x156..233, and a foreign fragment is near x280. Keep complete later words and remove the right-margin line.','word_overrides':{'1':{'x0':156,'x1':233,'y0':61,'y1':96},'5':{'x1':1800}},'exclude_words':{'0':'The word in belongs to the previous crop and is not intact here.'}}}
(B/'word_annotations.json').write_text(json.dumps(ann,indent=2))
p=B/'annotate_words.py';s=p.read_text();s=s.replace("  if key in ann:cuts=ann[key]['cuts']", "  if key in ann and 'cuts' in ann[key]:cuts=ann[key]['cuts']")
s=s.replace("  rows=[]", "  if key in ann:\n   for l,t,rt,bt in ann[key].get('mask_exclude_rects',[]):mask[t:bt,l:rt]=False\n  rows=[]")
s=s.replace("  out.extend(rows)","  for row in rows:\n   overrides=ann.get(key,{}).get('word_overrides',{}).get(str(row['word_index']),{});row.update(overrides)\n   reason=ann.get(key,{}).get('exclude_words',{}).get(str(row['word_index']))\n   if reason:row['excluded']=reason\n  out.extend(rows)")
p.write_text(s)
