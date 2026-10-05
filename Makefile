# Use the IPE environment's interpreter, e.g. `make collect PY=/mnt/dlabscratch1/zxu/envs/ipe/bin/python`.
PY ?= python
PORT ?= 8000

.PHONY: collect report serve test

collect:  ## Rebuild docs/data from IPE outputs
	$(PY) -m collector

report:   ## Rebuild and print the model pairing table
	$(PY) -m collector --report

serve:    ## Preview the site at http://localhost:$(PORT)
	cd docs && $(PY) -m http.server $(PORT)

test:
	$(PY) -m unittest discover tests
