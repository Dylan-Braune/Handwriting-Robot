import json
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent / "hershey_cursive.json"
data = json.load(open(DATA))
chars = data["chars"]


def parse(d):
    strokes = []
    for piece in d.split("M")[1:]:
        nums = re.findall(r"-?\d+\.?\d*", piece)
        pts = [(float(nums[i]), float(nums[i + 1])) for i in range(0, len(nums) - 1, 2)]
        if pts:
            strokes.append(pts)
    return strokes


def get(ch):
    return chars[ord(ch) - 33]


# calibration from the x-height band letters (n,m,u,v,w main strokes sit at y in [13,22])
BASELINE_Y = 22.0
XH_TOP_Y = 13.0
SCALE = BASELINE_Y - XH_TOP_Y  # = 9.0


def to_local(strokes):
    # x shifted so the glyph starts at local x=0; y flipped (hershey y grows down) and scaled to x-heights
    x0 = strokes[0][0][0]
    out = []
    for s in strokes:
        out.append([((x - x0) / SCALE, (BASELINE_Y - y) / SCALE) for (x, y) in s])
    return out


HERSHEY_CURSIVE = {}
for ch in "abcdefghijklmnopqrstuvwxyz":
    g = get(ch)
    strokes = parse(g["d"])
    if strokes:
        HERSHEY_CURSIVE[ch] = to_local(strokes)

out = Path(__file__).resolve().parent / "hershey_cursive_normalized.json"
json.dump(HERSHEY_CURSIVE, open(out, "w"))
print("wrote", out, "chars:", sorted(HERSHEY_CURSIVE))
for ch in "abcdefghijklmnopqrstuvwxyz":
    if ch in HERSHEY_CURSIVE:
        allpts = [p for s in HERSHEY_CURSIVE[ch] for p in s]
        ys = [p[1] for p in allpts]
        print(ch, "y range", round(min(ys), 2), round(max(ys), 2), "strokes:", len(HERSHEY_CURSIVE[ch]))
