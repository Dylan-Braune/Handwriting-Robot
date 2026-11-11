#u23539195 coefficient justification: report any line that runs into the margin
#a box more than about 20 pt over is visible on the page and has to be fixed
import re
import sys
from pathlib import Path

LOG = Path(__file__).resolve().parent.parent/"latex"/"build3.log"
PAT = re.compile(r"Overfull \\hbox \(([0-9.]+)pt too wide\) in paragraph at "
                 r"lines (\d+)--(\d+)\n(.{0,160})",re.S)
LIMIT = 20.0


def Main():
    if not LOG.exists():
        print("no build log at %s" % LOG)
        return 1
    log = LOG.read_text(encoding="utf-8",errors="replace")
    bad = []
    for m in PAT.finditer(log):
        pt = float(m.group(1))
        if pt > LIMIT:
            bad.append((pt,m.group(2),m.group(3),
                        m.group(4).replace("\n"," ")[:110]))
    bad.sort(reverse=True)
    for (pt,a,b,ctx) in bad:
        print("%8.1fpt  lines %s--%s  %s" % (pt,a,b,ctx))
    print("\n%d box(es) over %.0f pt; %d overfull in total"
          % (len(bad),LIMIT,log.count("Overfull")))
    print("undefined references: %d, citation warnings: %d"
          % (log.count("undefined"),log.count("Warning: Citation")))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(Main())
