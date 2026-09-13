"""Draw the character cuts back onto the author's real line images.

This is the check that the glyph library is actually made of that person's
letters: every stored variant comes from one of these boxes, so if a box is
in the wrong place the glyph filed under that character is not that
character.
"""
import argparse

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import _env

import BuildStyleProfile as SP
from TrainText import (CHARSET, PaperCRNN, IAMLineDatasetRaw, _decode_png)


def main(author, nLines=4, scale=2.0, out=None):
    device = torch.device("cpu")
    model = PaperCRNN(num_classes=len(CHARSET) + 1).to(device)
    sd = torch.load(SP.TEXT_WEIGHTS, map_location=device, weights_only=False)
    model.load_state_dict(sd.get("model_state_dict", sd)
                          if isinstance(sd, dict) else sd)
    model.eval()

    base = IAMLineDatasetRaw(root_dir=str(SP.DATA_DIR), cache_dir=str(SP.CACHE_DIR))
    samples = [s for s in base.samples
               if s["page_key"].split("/")[0] == author][:nLines]
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\consola.ttf", 15)
    except OSError:
        font = ImageFont.load_default()

    panels = []
    for s in samples:
        gray = np.array(_decode_png(s["image_png"]).convert("L"))
        glyphs, stats = SP.ExtractLineGlyphs(gray, s["text"], model, device)
        if glyphs is None:
            print(f"  {s['page_key']}: alignment failed")
            continue
        im = Image.fromarray(gray).convert("L").convert("RGB")
        w, h = im.size
        im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        band = SP.CoreBand(SP.BinarizeLine(gray))
        base_y = band[1] * scale if band else h * scale * 0.7
        for (i, ch, left, right) in stats.get("cuts", []):
            if ch == " ":
                continue
            x0, x1 = left * scale, right * scale
            kept = glyphs[i] is not None
            col = (0, 150, 0) if kept else (220, 0, 0)
            d.rectangle([x0, 2, x1, im.size[1] - 16], outline=col, width=2)
            d.text((x0 + 2, im.size[1] - 15), ch, fill=col, font=font)
        conf = stats["alignConf"]
        nk = sum(1 for g in glyphs if g is not None)
        print(f"  {s['page_key']}: conf {conf:.3f}, {nk} glyphs kept  "
              f"'{s['text'][:48]}'")
        panels.append(im)

    if not panels:
        return
    W = max(p.width for p in panels) + 8
    H = sum(p.height + 14 for p in panels) + 8
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    y = 4
    for p in panels:
        sheet.paste(p, (4, y))
        y += p.height + 14
    path = _env.OUT_DIR / (out or f"cuts_{author}.png")
    sheet.save(path)
    print(f"-> {path}  {sheet.size}")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("author")
    ap.add_argument("--lines", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    main(args.author, nLines=args.lines, out=args.out)
