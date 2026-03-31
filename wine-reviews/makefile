# Wine Reviews – Kaggle dataset analysis (zynicide/wine-reviews)

# Sync dependencies (install/update from pyproject.toml and uv.lock)
.PHONY: sync
sync:
	uv sync

# Download dataset from Kaggle (requires kaggle CLI + credentials)
.PHONY: download
download:
	kaggle datasets download zynicide/wine-reviews -p data --unzip

# Run Jupyter notebook (executes all cells top to bottom)
.PHONY: run
run:
	uv run jupyter nbconvert --execute --to notebook --inplace wine-reviews-analysis.ipynb

# Open notebook in browser for interactive use
.PHONY: notebook
notebook:
	uv run jupyter notebook wine-reviews-analysis.ipynb
