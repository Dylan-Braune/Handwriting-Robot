#every classify timing recorded ON THE ODROID N2+, which is the target
#platform.  the log that holds them was written by the server running as
#root@odroidn2, so these are on-target measurements and not laptop figures.
#
#R4 allows 2 + 0.025x seconds for a page of x characters.  the stage names are
#the server's own: loading, conditioning, segmenting, cropping, reading,
#naming.  the first four are image work and the last two are the networks.
#
#the point of separating them is that the allowance has only ONE term that
#grows with the page, 0.025x, and the networks are the only stages that scale
#with the character count.  everything else is per-page work that has to fit
#inside the fixed 2 s.
import re
from pathlib import Path

SRC = Path(r"C:\Users\braun\Downloads\ALLTestData.txt")
lines = SRC.read_text(encoding="utf-8", errors="replace").splitlines()
HEAD = re.compile(r"^timing for classify: ([\d.]+) s total, (\d+) characters, (\d+) lines")
STAGE = re.compile(r"^\s+(\w[\w ]*?)\s+([\d.]+) s")

runs = []
for i, ln in enumerate(lines):
    m = HEAD.match(ln)
    if not m:
        continue
    total, chars, nl = float(m.group(1)), int(m.group(2)), int(m.group(3))
    st = {}
    for j in range(i+1, min(i+12, len(lines))):
        s = STAGE.match(lines[j])
        if not s:
            break
        st[s.group(1).strip()] = float(s.group(2))
    runs.append({"total": total, "chars": chars, "lines": nl, "stages": st,
                 "allow": 2+0.025*chars, "srcLine": i+1})

IMAGE = ("loading", "conditioning", "segmenting", "cropping")
NET = ("reading", "naming")

print("%d classify runs recorded on the ODROID N2+\n" % len(runs))
print("%-7s %-7s %-7s %-9s %-9s %-10s %-10s %-9s %s"
      % ("line", "lines", "chars", "allow s", "total s", "image s", "network s",
         "over s", "verdict"))
print("-"*92)
for r in runs:
    img = sum(r["stages"].get(k, 0.0) for k in IMAGE)
    net = sum(r["stages"].get(k, 0.0) for k in NET)
    print("%-7d %-7d %-7d %-9.2f %-9.2f %-10.2f %-10.2f %-9.2f %s"
          % (r["srcLine"], r["lines"], r["chars"], r["allow"], r["total"],
             img, net, r["total"]-r["allow"],
             "MET" if r["total"] <= r["allow"] else "EXCEEDED"))

print("\n" + "="*92)
nMet = sum(1 for r in runs if r["total"] <= r["allow"])
print("%d of %d runs met R4" % (nMet, len(runs)))
tot = sum(r["total"] for r in runs)
print("\nshare of the total time, summed over all runs:")
for k in IMAGE+NET:
    v = sum(r["stages"].get(k, 0.0) for r in runs)
    print("   %-14s %7.2f s  %5.1f %%" % (k, v, 100*v/tot))
img = sum(sum(r["stages"].get(k, 0.0) for k in IMAGE) for r in runs)
net = sum(sum(r["stages"].get(k, 0.0) for k in NET) for r in runs)
print("   %-14s %7.2f s  %5.1f %%" % ("image work", img, 100*img/tot))
print("   %-14s %7.2f s  %5.1f %%" % ("networks", net, 100*net/tot))

#---the two cost terms against the two terms of the allowance--------------
#the allowance is 2 s plus 0.025 s per character.  the networks are the only
#stages whose cost grows with the character count, so they are the only ones
#the 0.025x term can pay for; the image stages must fit in the fixed 2 s
print("\nper-character cost of the network stages against the 0.025 s/char term:")
for r in runs:
    net = sum(r["stages"].get(k, 0.0) for k in NET)
    print("   %4d characters: %.4f s/char  (%.1f x the allowance term)"
          % (r["chars"], net/r["chars"], (net/r["chars"])/0.025))

print("\nper-page cost of the image stages against the fixed 2 s term:")
for r in runs:
    img = sum(r["stages"].get(k, 0.0) for k in IMAGE)
    print("   %2d lines: %.2f s  (%.1f x the 2 s term)" % (r["lines"], img, img/2))

print("\nthe smallest page that could comply, if the image stages were free:")
for r in runs:
    net = sum(r["stages"].get(k, 0.0) for k in NET)
    rate = net/r["chars"]
    if rate > 0.025:
        print("   a page reading at %.4f s/char can never comply: the network "
              "cost alone grows faster than the allowance" % rate)
    else:
        print("   at %.4f s/char the network cost is inside the allowance term"
              % rate)
