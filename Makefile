PYTHON ?= python

.PHONY: install verify lint typecheck test audit bench

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .[dev]

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy --strict --config-file pyproject.toml sih26147

test:
	$(PYTHON) -m pytest --cov=sih26147 --cov-report=term-missing --cov-fail-under=85

audit:
	$(PYTHON) -m pip_audit -r requirements.txt

verify: lint typecheck test audit
