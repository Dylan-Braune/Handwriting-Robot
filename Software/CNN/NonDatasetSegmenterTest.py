"""
NonDatasetSegmenterTest.py

Strict test harness for the two non-dataset page segmenters:

  * NonDatasetSegmenterCV.py -- full-library implementation (OpenCV)
  * NonDatasetSegmenterFP.py -- first-principles implementation (numpy only;
    PIL used just for file decode/encode)

Scoring is a STRICT 1:1 ORDERED MATCH against the label files in
NOGIT/NonDatasetImages: the k-th detected box must have the same tag
(TEXT, or MESS for a diagram/sketch block) as the k-th label row.  A page
scores matched_rows / max(expected, detected).

Usage (from software/CNN):
    python NonDatasetSegmenterTest.py            # run BOTH versions
    python NonDatasetSegmenterTest.py cv         # library version only
    python NonDatasetSegmenterTest.py fp         # first-principles only

Outputs per version under NOGIT/NonDatasetTestOutput/<cv|fp>/:
    previews/<page>_preview.png     colour-coded boxes (green TEXT, orange MESS)
    crops/<page>/line_NN_TAG.png    raw per-line crops (white background,
                                    NON-rectangular: each line owns exactly its
                                    own ink, so overlapping ascenders/
                                    descenders of neighbouring lines are
                                    excluded)
    crops_model/<page>/...          the same crops scaled to fit the trainer's
                                    input size (INPUT_H x INPUT_W below) with
                                    the aspect ratio KEPT and the remainder
                                    padded white, so the writing retains its
                                    natural shape
    report.json                     per-page accuracy summary

Label file format (unchanged from the existing convention):
    <index><TAB><line text>          one row per physical text line
    <index><TAB>MESS                 one row where a diagram/sketch sits
"""

import os
import sys
import json
import glob
import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(SCRIPT_DIR, 'NOGIT', 'NonDatasetImages')
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, 'NOGIT', 'NonDatasetTestOutput')

# keep in sync with train_paper_cnn_bilstm_ctc.py INPUT_HEIGHT / INPUT_WIDTH
INPUT_H, INPUT_W = 64, 640


def ReadLabels(imgPath):
    base = os.path.splitext(imgPath)[0]
    labelPath = base + '_labels.txt'
    if not os.path.exists(labelPath):
        return None
    rows = []
    with open(labelPath, encoding='utf-8') as f:
        for ln in f:
            ln = ln.rstrip('\n')
            if not ln.strip():
                continue
            parts = ln.split('\t', 1)
            rows.append(parts[1] if len(parts) == 2 else ln)
    return rows


def Score(detected, labelRows):
    expTags = ['MESS' if r.strip() == 'MESS' else 'TEXT' for r in labelRows]
    detTags = [r['tag'] for r in detected]
    n = max(len(expTags), len(detTags))
    match = sum(1 for i in range(min(len(expTags), len(detTags)))
                if expTags[i] == detTags[i])
    return (match / n if n else 1.0), expTags, detTags


def RunVersion(moduleName, outSub):
    mod = __import__(moduleName)
    outRoot = os.path.join(OUTPUT_ROOT, outSub)
    os.makedirs(os.path.join(outRoot, 'previews'), exist_ok=True)

    imagePaths = sorted(p for ext in ('*.png', '*.jpg', '*.jpeg', '*.JPG', '*.JPEG')
                        for p in glob.glob(os.path.join(IMAGES_DIR, ext)))
    if not imagePaths:
        print(f'No images found in {IMAGES_DIR}')
        return []

    pageReports = []
    for p in imagePaths:
        name = os.path.splitext(os.path.basename(p))[0]
        results, preview, meta = mod.ProcessPage(p)
        preview.save(os.path.join(outRoot, 'previews', name + '_preview.png'))

        cropDir = os.path.join(outRoot, 'crops', name)
        modelDir = os.path.join(outRoot, 'crops_model', name)
        for d in (cropDir, modelDir):
            if os.path.isdir(d):
                for f in glob.glob(os.path.join(d, '*.png')):
                    os.remove(f)
            os.makedirs(d, exist_ok=True)
        for r in results:
            fname = f"line_{r['order']:02d}_{r['tag']}.png"
            img = Image.fromarray(r['raw_crop'])
            img.save(os.path.join(cropDir, fname))
            # model-format: scale to fit INPUT_H x INPUT_W KEEPING the aspect
            # ratio, then pad the remainder with white -- the writing keeps
            # its natural shape instead of being stretched to the full width
            g = img.convert('L')
            s = min(INPUT_W / g.width, INPUT_H / g.height)
            g = g.resize((max(1, int(g.width * s)), max(1, int(g.height * s))),
                         Image.Resampling.BILINEAR)
            canvas = Image.new('L', (INPUT_W, INPUT_H), 255)
            canvas.paste(g, (0, (INPUT_H - g.height) // 2))
            canvas.save(os.path.join(modelDir, fname))

        labels = ReadLabels(p)
        rep = dict(page=name, detected=len(results),
                   detected_mess=sum(1 for r in results if r['tag'] == 'MESS'),
                   skew_deg=round(meta.get('skew', 0.0), 2),
                   text_height_px=round(meta.get('textH', 0.0), 1))
        if labels is None:
            print(f'{name}: no label file -- segmented {len(results)} boxes '
                  f'(no accuracy score)')
        else:
            acc, expTags, detTags = Score(results, labels)
            rep.update(expected=len(expTags),
                       expected_mess=expTags.count('MESS'),
                       accuracy=round(acc, 4))
            print(f"{name}: acc={acc*100:.1f}%  expected {len(expTags)} rows "
                  f"({expTags.count('MESS')} MESS) | detected {len(detTags)} "
                  f"({detTags.count('MESS')} MESS) | skew={rep['skew_deg']}deg")
            if acc < 1.0:
                for i in range(max(len(expTags), len(detTags))):
                    e = expTags[i] if i < len(expTags) else '--'
                    d = detTags[i] if i < len(detTags) else '--'
                    flag = '' if e == d else '   <<< MISMATCH'
                    lbl = labels[i][:50] if i < len(labels) else ''
                    print(f'   {i:2d}  exp={e:4s} det={d:4s}  {lbl}{flag}')
        pageReports.append(rep)

    scored = [r['accuracy'] for r in pageReports if 'accuracy' in r]
    if scored:
        print(f'== {moduleName} OVERALL: {np.mean(scored)*100:.1f}% ==')

    with open(os.path.join(outRoot, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump(pageReports, f, indent=2)
    return pageReports


if __name__ == '__main__':
    which = sys.argv[1].lower() if len(sys.argv) > 1 else 'both'
    if which in ('cv', 'both'):
        print('--- library (OpenCV) version ---')
        RunVersion('NonDatasetSegmenterCV', 'cv')
    if which in ('fp', 'both'):
        print('--- first-principles (numpy) version ---')
        RunVersion('NonDatasetSegmenterFP', 'fp')
