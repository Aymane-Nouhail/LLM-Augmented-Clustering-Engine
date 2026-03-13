DATASETS ?= bank77,clinc,tweet
METHODS  ?= all

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

smoke:
	python smoke_test.py

.PHONY: run bank77 clinc tweet kmeans jose pairwise correction keyphrase smoke
