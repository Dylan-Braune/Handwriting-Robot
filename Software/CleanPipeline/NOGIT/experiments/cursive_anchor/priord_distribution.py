import sys
sys.path.insert(0, r"C:\Users\braun\OneDrive\Documents\GitHub\Handwriting-Robot\Software\CleanPipeline")
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROF_DIR = Path(r"C:\Users\braun\OneDrive\Documents\GitHub\Handwriting-Robot\Software\CNN\NOGIT\StyleProfiles10")
AUTHORS = ["abhinav", "yeukita", "thiya"]


def font(size, bold=False):
    try:
        return ImageFont.truetype("C:/Windows/Fonts/consolab.ttf" if bold else "C:/Windows/Fonts/consola.ttf", size)
    except Exception:
        return ImageFont.load_default()


stats = {}
for a in AUTHORS:
    p = json.load(open(PROF_DIR / f"{a}.json"))
    vals = [g.get("priorD", None) for vs in p["glyphs"].values() for g in vs]
    vals = [v for v in vals if v is not None]
    arr = np.array(vals)
    stats[a] = dict(conn=p["connectedness"], n=len(arr), mean=float(arr.mean()),
                     p50=float(np.percentile(arr, 50)), p90=float(np.percentile(arr, 90)),
                     frac_clean=float((arr <= 0.12).mean()), frac_garbage=float((arr >= 0.55).mean()),
                     arr=arr)
    print(a, {k: v for k, v in stats[a].items() if k != "arr"})

# histogram image
W, H = 900, 420
pad_l, pad_b, pad_t = 60, 50, 40
plot_w, plot_h = W - pad_l - 20, H - pad_b - pad_t
bins = np.linspace(0, 1.0, 41)
colors = {"abhinav": (0, 110, 200), "yeukita": (0, 160, 60), "thiya": (200, 30, 30)}

img = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(img)
f = font(13)
fb = font(15, bold=True)

maxcount = 0
hists = {}
for a in AUTHORS:
    h, _ = np.histogram(stats[a]["arr"], bins=bins)
    h = h / stats[a]["n"]  # normalize to fraction, counts differ per author
    hists[a] = h
    maxcount = max(maxcount, h.max())

# axes
draw.line([(pad_l, pad_t), (pad_l, pad_t + plot_h)], fill=(0, 0, 0))
draw.line([(pad_l, pad_t + plot_h), (pad_l + plot_w, pad_t + plot_h)], fill=(0, 0, 0))
for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
    x = pad_l + frac * plot_w
    draw.line([(x, pad_t + plot_h), (x, pad_t + plot_h + 5)], fill=(0, 0, 0))
    draw.text((x - 10, pad_t + plot_h + 8), f"{frac:.2f}", fill=(0, 0, 0), font=f)
draw.text((pad_l + plot_w // 2 - 60, H - 20), "priorD (distance from cross-author letter shape)", fill=(0, 0, 0), font=f)
draw.text((4, 4), "fraction of this author's variants", fill=(0, 0, 0), font=f)

for ai, a in enumerate(AUTHORS):
    h = hists[a]
    pts = []
    for i, v in enumerate(h):
        x = pad_l + (bins[i] + bins[i + 1]) / 2 / 1.0 * plot_w
        y = pad_t + plot_h - (v / maxcount) * plot_h
        pts.append((x, y))
    draw.line(pts, fill=colors[a], width=2)
    draw.text((W - 150, pad_t + ai * 20), f"{a} (conn={stats[a]['conn']:.2f})", fill=colors[a], font=fb)

for thresh, label in ((0.12, "clean <=0.12"), (0.55, "garbage >=0.55")):
    x = pad_l + thresh * plot_w
    draw.line([(x, pad_t), (x, pad_t + plot_h)], fill=(180, 180, 180), width=1)
    draw.text((x + 2, pad_t), label, fill=(120, 120, 120), font=f)

out = Path(__file__).resolve().parent / "priord_distribution.png"
img.save(out)
print("wrote", out)

print()
print(f"{'author':10s} {'conn':>6s} {'n':>5s} {'mean':>6s} {'p50':>6s} {'p90':>6s} {'%clean':>7s} {'%garbage':>9s}")
for a in AUTHORS:
    s = stats[a]
    print(f"{a:10s} {s['conn']:6.2f} {s['n']:5d} {s['mean']:6.3f} {s['p50']:6.3f} {s['p90']:6.3f} {100*s['frac_clean']:6.1f}% {100*s['frac_garbage']:8.1f}%")
