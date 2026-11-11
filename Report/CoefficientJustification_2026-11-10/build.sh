#!/bin/sh
#u23539195 coefficient justification: regenerate the tex from the recorded runs and build the pdf
set -e
cd "$(dirname "$0")/sweep"
python mkdoc.py
python checkcites.py
cd ../latex
pdflatex -interaction=nonstopmode main.tex >build1.log 2>&1 || true
bibtex main >bibtex.log 2>&1 || true
pdflatex -interaction=nonstopmode main.tex >build2.log 2>&1 || true
pdflatex -interaction=nonstopmode main.tex >build3.log 2>&1 || true
echo "pages: $(python -c "import re;print(re.findall(r'(\d+) pages',open('build3.log',encoding='utf-8',errors='replace').read())[-1])")"
grep -c "Warning: Citation" build3.log || true
