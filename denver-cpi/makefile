# Denver CPI – Kaggle dataset analysis

# Sync dependencies (install/update from pyproject.toml and uv.lock)
.PHONY: sync
sync:
	uv sync

# Download dataset from Kaggle (requires kaggle CLI + credentials)
.PHONY: download
download:
	kaggle datasets download bls/denver-cpi -p data --unzip

# Run Jupyter notebook (executes all cells top to bottom)
.PHONY: run
run:
	uv run jupyter nbconvert --execute --to notebook --inplace denver-cpi-trends.ipynb

# Open notebook in browser for interactive use
.PHONY: notebook
notebook:
	uv run jupyter notebook denver-cpi-trends.ipynb
