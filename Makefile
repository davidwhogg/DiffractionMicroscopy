LATEX       = pdflatex
CHECK_RERUN = grep Rerun $*.log

all: gaussian.pdf

vc.tex: ../.git/logs/HEAD
	echo "%%% This file is generated by the Makefile." > vc.tex
	echo '\\newcommand{\giturl}{\url{%' >> vc.tex
	git config --get remote.origin.url | sed "s/\.git//" | sed "s/https/http/" >> vc.tex
	echo '}}' >> vc.tex
	git log -1 --date=short --format="format:\\newcommand{\\githash}{%h}\\newcommand{\\gitdate}{%ad}\\newcommand{\\gitauthor}{%an}" >> vc.tex

%.pdf: %.tex vc.tex
	${LATEX} $<
