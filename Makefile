# NeuroGraphRAG — developer entrypoints.
# On Windows without `make`, run the underlying commands directly (see README),
# or use Git Bash / WSL where `make` is available.

PYTHON ?= python
PKG    := neurographrag

.PHONY: help install install-extras dev-install index eval conformal report demo api web test lint fmt clean

help:
	@echo "NeuroGraphRAG targets:"
	@echo "  install         Install core runtime deps"
	@echo "  install-extras  Install optional accelerators (ST/LoRA/faiss/ragas)"
	@echo "  dev-install     Editable install with dev tools"
	@echo "  index           Build the knowledge graph + retrieval indexes"
	@echo "  eval            Run the reproducible evaluation (ablations)"
	@echo "  conformal       Run the risk-controlled (conformal) retrieval study"
	@echo "  report          Regenerate results tables/figures from the last eval"
	@echo "  demo            One-shot: index + eval + report"
	@echo "  api             Launch the FastAPI backend (serves the static UI)"
	@echo "  web             Launch the Vite/React frontend dev server"
	@echo "  test            Run the test suite"
	@echo "  lint / fmt      Ruff lint / format"
	@echo "  clean           Remove build + run artifacts"

install:
	$(PYTHON) -m pip install -r requirements.txt

install-extras:
	$(PYTHON) -m pip install -r requirements-extras.txt

dev-install:
	$(PYTHON) -m pip install -e ".[api,dev]"

index:
	$(PYTHON) -m $(PKG).cli index --config configs/default.yaml

eval:
	$(PYTHON) -m $(PKG).cli eval --config configs/default.yaml

conformal:
	$(PYTHON) -m $(PKG).cli conformal --config configs/default.yaml

report:
	$(PYTHON) -m $(PKG).cli report --config configs/default.yaml

demo:
	$(PYTHON) -m $(PKG).cli demo --config configs/default.yaml

api:
	$(PYTHON) -m $(PKG).cli serve --config configs/default.yaml

web:
	cd frontend && npm install && npm run dev

test:
	$(PYTHON) -m pytest

lint:
	ruff check src tests

fmt:
	ruff format src tests

clean:
	$(PYTHON) -c "import shutil,glob,os; [shutil.rmtree(p,ignore_errors=True) for p in ['artifacts','runs','.pytest_cache','.ruff_cache','build','dist']]"
