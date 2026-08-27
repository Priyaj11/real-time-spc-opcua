# Every command this project needs, in one place.
#
# Run `make` on its own to see the list. The point of this file is that nobody
# has to remember pytest marker expressions or coverage flags, and that the
# command CI runs is the same command you run locally.

PYTHON := python

.DEFAULT_GOAL := help
.PHONY: help fast test slow cov lint fix dash data clean

help:  ## Show this list
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-8s\033[0m %s\n", $$1, $$2}'

fast:  ## The loop you run while coding: no sockets, no long studies
	$(PYTHON) -m pytest -m "not integration and not slow"

test:  ## Everything. What CI runs, and what must pass before a commit
	$(PYTHON) -m pytest

slow:  ## Only the statistical studies, with their measured numbers printed
	$(PYTHON) -m pytest -m slow -s

cov:  ## Coverage report over the deterministic tests
	# Deliberately excludes the integration lane. Coverage traces every line
	# executed, which slows the server's publishing loop enough that a test
	# pacing at 100x simulated time can no longer keep up, and the client's
	# subscription handshake starves. Same failure as speed_factor=200 in
	# Milestone 10, from a different cause. Those tests are verified by
	# `make test`; measuring them is not worth breaking them.
	$(PYTHON) -m pytest -m "not integration" --cov --cov-report=term-missing

lint:  ## Static checks: unused names, import order, docstrings, likely bugs
	$(PYTHON) -m ruff check .

fix:  ## Apply the lint fixes that are safe to apply automatically
	$(PYTHON) -m ruff check . --fix

dash:  ## Launch the operator dashboard
	$(PYTHON) -m streamlit run spc_opcua/dashboard/app.py

data:  ## Collect a fresh sample_data.csv and subgroups.csv over OPC UA
	$(PYTHON) -m spc_opcua.pipeline

clean:  ## Remove caches and coverage artefacts
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +