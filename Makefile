.PHONY: format
format:
	ruff check airow --select I --fix
	ruff format airow

.PHONY: lint
lint:
	ruff check airow
	mypy airow

.PHONY: tests
tests:
	pytest --cov=airow -vv tests/
