#!/usr/bin/env bash
# Build paper/main.pdf and an arXiv-ready source bundle.
#   bash paper/build.sh          -> paper/main.pdf, paper/arxiv-src.tar.gz
# Needs pdflatex + bibtex (TeX Live). arXiv compiles the bundle itself from main.tex + main.bbl + figures/.
set -euo pipefail
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null
bibtex main > /dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null
echo "main.pdf: $(pdfinfo main.pdf 2>/dev/null | awk '/^Pages/ {print $2}') pages"
grep -E "Overfull|undefined|Citation .* undefined" main.log | sed 's/^/  warn: /' || true
# arXiv wants sources, not the PDF: tex, the pre-built .bbl (it does not run bibtex), figures.
tar czf arxiv-src.tar.gz main.tex main.bbl figures/*.png
echo "arxiv-src.tar.gz: $(tar tzf arxiv-src.tar.gz | tr '\n' ' ')"
