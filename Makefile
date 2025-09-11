.PHONY: install lint format check clean help

PYTHON := python
PIP := pip

help:
	@echo "Targets: install, lint, format, check, clean"

install:
	$(PIP) install -e .[test]
	$(PIP) install pre-commit black isort flake8
	pre-commit install || true

# Run all pre-commit hooks against the entire repo
lint:
	pre-commit run --all-files || true

# Auto-format code (fix whitespace with Black, sort imports)
format:
	black .
	isort . --profile=black --line-length=100

# Verify formatting/lint without modifying files
check:
	black --check .
	isort . --check-only --profile=black --line-length=100
	flake8 . --max-line-length=100

# Remove caches and generated artifacts (keeps folders)
clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} + || true
	rm -rf .pytest_cache .mypy_cache .pytype .cache || true
	rm -rf build dist *.egg-info || true
	find reports/figures -mindepth 1 -type f -delete 2>/dev/null || true
	find reports/metrics -mindepth 1 -type f -delete 2>/dev/null || true
	find models -mindepth 1 -type f -delete 2>/dev/null || true
