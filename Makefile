DATASETS ?= bank77,clinc,tweet
METHODS  ?= all

# MacTeX installs to /Library/TeX/texbin — not always on make's PATH
TEXBIN   ?= /Library/TeX/texbin
PDFLATEX  = PATH="$(TEXBIN):$$PATH" pdflatex -interaction=nonstopmode
BIBTEX    = PATH="$(TEXBIN):$$PATH" bibtex

run:
	python -m main.run_experiments --datasets $(DATASETS) --methods $(METHODS)

bank77:
	python -m main.run_experiments --datasets bank77 --methods $(METHODS)

clinc:
	python -m main.run_experiments --datasets clinc --methods $(METHODS)

tweet:
	python -m main.run_experiments --datasets tweet --methods $(METHODS)

kmeans:
	python -m main.run_experiments --datasets $(DATASETS) --methods kmeans

jose:
	python -m main.run_experiments --datasets $(DATASETS) --methods jose

pairwise:
	python -m main.run_experiments --datasets $(DATASETS) --methods pairwise

correction:
	python -m main.run_experiments --datasets $(DATASETS) --methods correction

keyphrase:
	python -m main.run_experiments --datasets $(DATASETS) --methods keyphrase

normalization:
	python -m main.run_experiments --datasets $(DATASETS) --methods normalization

paraphrase:
	python -m main.run_experiments --datasets $(DATASETS) --methods paraphrase

clusterllm:
	python -m main.run_experiments --datasets $(DATASETS) --methods clusterllm

smoke:
	python smoke_test.py

figures:
	python figures/plot_figures.py

report: figures
	cd report && $(PDFLATEX) report.tex && $(BIBTEX) report && $(PDFLATEX) report.tex && $(PDFLATEX) report.tex

.PHONY: run bank77 clinc tweet kmeans jose pairwise correction keyphrase normalization paraphrase clusterllm smoke figures report
