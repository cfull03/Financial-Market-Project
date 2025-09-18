# ============================
# File: Makefile
# ============================
SHELL := /bin/bash

# ---- Config ----
CONFIG ?= configs/default.yaml
# Slugify project_name from CONFIG; fallback to "dataset"
NAME ?= $(shell python - <<'PY'
import re, sys, yaml
cfg=yaml.safe_load(open('$(CONFIG)'))
n=cfg.get('project_name') or 'dataset'
print(re.sub(r'[^A-Za-z0-9]+','_',n).strip('_').lower() or 'dataset')
PY
)

# First CSV under data/raw as default (override with CSV=/path/to.csv)
CSV ?= $(shell ls -1 data/raw/*.csv 2>/dev/null | head -n1)

# Sample settings
N ?= 500
STAGE ?= processed

# ---- Dev tools ----
.PHONY: help install lint format check clean

help:
	@echo "Targets:"
	@echo "  ingest      - copy a source CSV into data/raw/ (stable filename)"
	@echo "  split       - split raw -> processed train/test (timestamped)"
	@echo "  clean       - clean latest train/test -> *_clean (timestamped)"
	@echo "  validate    - validate latest processed train against schema"
	@echo "  eda         - generate quick EDA figures"
	@echo "  sample      - sample N rows from latest dataset (stage=$(STAGE))"
	@echo "  install/lint/format/check/clean"

install:
	pip install -e .[test]
	pip install pre-commit black isort flake8
	pre-commit install || true

lint:
	pre-commit run --all-files

format:
	black .
	isort . --profile=black --line-length=100

check:
	black --check .
	isort . --check-only --profile=black --line-length=100
	flake8 . --max-line-length=100

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} + || true
	rm -rf .pytest_cache .mypy_cache .cache || true
	rm -rf build dist *.egg-info || true
	find reports/figures -mindepth 1 -type f -delete 2>/dev/null || true
	find reports/metrics -mindepth 1 -type f -delete 2>/dev/null || true
	find models -mindepth 1 -type f -delete 2>/dev/null || true

# ---- CLI wrappers ----
.PHONY: ingest split clean-data validate eda sample

ingest:
	@if [ -z "$(CSV)" ]; then \
		echo "No CSV found in data/raw. Set CSV=/path/to.csv"; exit 1; \
	fi
	dsproj ingest --config $(CONFIG) --csv "$(CSV)" --name $(NAME)

split:
	dsproj split --config $(CONFIG) --name $(NAME)

clean-data:
	dsproj clean --config $(CONFIG) --name $(NAME) --add-timeparts --drop-constants

validate:
	dsproj validate --config $(CONFIG) --name $(NAME)

eda:
	dsproj eda-report --config $(CONFIG) --name $(NAME)

sample:
	dsproj make-sample --config $(CONFIG) --n $(N) --stage $(STAGE) --name $(NAME)
