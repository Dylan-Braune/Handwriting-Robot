"""
make_demo_check.py -- regenerate every visual artifact for manual checking
with the CURRENT settings, into one folder: NOGIT/DemoCheck/

Produces, per author:
  <a>_1_real.png      a REAL held-out line by that author (ground truth)
  <a>_2_synth.png     the SAME words synthesized in that author's style
  <a>_3_gcode.png     a NOVEL sentence, rendered from the emitted G-code
  big_<a>.png         the same G-code line, large, for reading letter by letter
and overall:
  sheet_all.png       every author stacked: real / synth / gcode
  readback.txt        what the text recognizer read back from each

The G-code images are re-parsed from the actual .gcode file, so they show
what the gantry would draw, not the idealised trajectory.

Run:
    python make_demo_check.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gcode_writer as GW
import style_profile as SP
import synthesize_handwriting as SY
import verify_end_to_end as V
from train_paper_cnn_bilstm_ctc import IAMLineDatasetRaw, _decode_png

OUT = Path(__file__).resolve().parent / "NOGIT" / "DemoCheck"
NOVEL = "My robot copies handwriting from ten authors"


def Main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cpu")
    reader = V.LoadTextModel(dev)
    profiles = SY.LoadAllProfiles()
    cfg = GW.GantryConfig()
    base = IAMLineDatasetRaw(root_dir=str(SP.DATA_DIR), cache_dir=str(SP.CACHE_DIR))

    hold = {}
    for s in base.samples:
        if s["is_holdout"] and 22 <= len(s["text"]) <= 52:
            hold.setdefault(s["page_key"].split("/")[0], []).append(s)

    lines, rows = [], []
    for a in sorted(profiles):
        prof = profiles[a]
        item = hold.get(a, [None])[0]
        realImg = _decode_png(item["image_png"]).convert("L") if item else None
        realTxt = item["text"] if item else ""

        # same words as the real line, in this author's style
        synImg = None
        if realTxt:
            tj = SY.SynthesizeLegible(realTxt, prof, nTries=4, seed=5,
                                      lineWidthMm=10_000.0, reader=reader,
                                      device=dev)
            synImg = SY.RenderTrajectory(tj, pxPerMm=18.0, profile=prof)

        # a novel sentence, taken all the way through G-code
        tjN = SY.SynthesizeLegible(NOVEL, prof, nTries=4, seed=3,
                                   lineWidthMm=10_000.0, reader=reader,
                                   device=dev)
        gpath = OUT / ("%s.gcode" % a)
        gImg = V.GcodeRoundTrip(tjN, cfg, str(gpath), prof)

        if realImg:
            realImg.save(OUT / ("%s_1_real.png" % a))
        if synImg:
            synImg.save(OUT / ("%s_2_synth.png" % a))
        if gImg:
            gImg.save(OUT / ("%s_3_gcode.png" % a))
            big = gImg.resize((int(gImg.width * 150 / gImg.height), 150),
                              Image.LANCZOS)
            big.save(OUT / ("big_%s.png" % a))

        gotSyn = V.ReadText(reader, synImg, dev) if synImg else ""
        gotG = V.ReadText(reader, gImg, dev) if gImg else ""
        lines.append(
            "--- author %s ---\n"
            "  real line text : %s\n"
            "  synth read as  : %s\n"
            "  novel wanted   : %s\n"
            "  gcode read as  : %s\n" % (a, realTxt, gotSyn, NOVEL, gotG))
        rows.append((a, realImg, synImg, gImg, realTxt))

    with open(OUT / "readback.txt", "w", encoding="utf-8") as f:
        f.write("What the frozen text recognizer reads back.\n"
                "'synth' = same words as the real line; 'gcode' = a novel\n"
                "sentence rendered from the emitted G-code file.\n\n")
        f.writelines(lines)

    # stacked sheet
    H = 70
    def fit(im):
        if im is None:
            return Image.new("L", (10, H), 255)
        return im.resize((max(1, int(im.width * H / im.height)), H),
                         Image.LANCZOS)
    fitted = [(a, fit(r), fit(s), fit(g), t) for a, r, s, g, t in rows]
    maxW = min(1500, max(max(r.width, s.width, g.width)
                         for _, r, s, g, _ in fitted))
    lab = 96
    sheet = Image.new("L", (lab + maxW + 12, sum(3 * H + 34 for _ in fitted) + 8), 246)
    d = ImageDraw.Draw(sheet)
    y = 4
    for a, r, s, g, t in fitted:
        d.text((6, y + 30), a, fill=0)
        for k, tag in enumerate(("real", "synth", "GCODE")):
            d.text((6, y + 48 + k * H), tag, fill=110)
        d.text((lab, y + 1), "real: %s   |   gcode: %s" % (t[:52], NOVEL), fill=130)
        for k, im in enumerate((r, s, g)):
            sheet.paste(im.crop((0, 0, min(maxW, im.width), H)), (lab, y + 12 + k * H))
        d.line([(0, y + 3 * H + 26), (sheet.width, y + 3 * H + 26)], fill=205)
        y += 3 * H + 34
    sheet.save(OUT / "sheet_all.png")
    print("wrote %d files to %s" % (len(list(OUT.iterdir())), OUT))
    print("start with: sheet_all.png, then big_<author>.png, then readback.txt")


if __name__ == "__main__":
    Main()
