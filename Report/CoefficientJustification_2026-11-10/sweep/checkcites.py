#u23539195 coefficient justification: every \cite key used must exist in the bibliography
import re
import sys
from pathlib import Path

LATEX = Path(__file__).resolve().parent.parent/"latex"

CITE = re.compile(r"\\cite(?:\[[^\]]*\])?\{([^}]+)\}")
ENTRY = re.compile(r"^@\w+\{([^,]+),",re.M)


def Main():
    bibText = (LATEX/"references.bib").read_text(encoding="utf-8",errors="replace")
    bib = set(ENTRY.findall(bibText))
    cited = set()
    for name in ("coefficients.tex","main.tex"):
        p = LATEX/name
        if not p.exists():
            continue
        for m in CITE.findall(p.read_text(encoding="utf-8")):
            for k in m.split(","):
                cited.add(k.strip())
    missing = sorted(cited-bib)
    print("bibliography entries : %d" % len(bib))
    print("distinct keys cited  : %d" % len(cited))
    print("present in bib       : %d" % len(cited & bib))
    if missing:
        print("MISSING FROM BIB     : %s" % ", ".join(missing))
    else:
        print("every cited key is in the bibliography")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(Main())
