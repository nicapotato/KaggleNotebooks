# Global employment / unemployment — Kaggle dataset analysis
# Dataset: lucalullo/global-employment-unemployment-rates-1991-2025

.PHONY: sync
sync:
	uv sync

.PHONY: download
download:
	kaggle datasets download lucalullo/global-employment-unemployment-rates-1991-2025 -p data --unzip

.PHONY: run-eda
run-eda:
	uv run jupyter nbconvert --execute --to notebook --inplace unemployment-global-eda.ipynb

.PHONY: run-clustering
run-clustering:
	uv run jupyter nbconvert --execute --to notebook --inplace unemployment-global-clustering.ipynb

.PHONY: run
run: run-eda run-clustering

.PHONY: notebook
notebook:
	uv run jupyter notebook unemployment-global-eda.ipynb
