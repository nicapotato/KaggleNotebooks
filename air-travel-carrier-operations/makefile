# Air travel carrier operations — Kaggle dataset analysis

.PHONY: sync
sync:
	uv sync

.PHONY: download
download:
	kaggle datasets download zkskhurram/airline-ticket-prices-vs-oil-and-fuel-costs -p data --unzip

.PHONY: run
run:
	uv run jupyter nbconvert --execute --to notebook --inplace air-travel-carrier-analysis.ipynb

.PHONY: notebook
notebook:
	uv run jupyter notebook air-travel-carrier-analysis.ipynb
