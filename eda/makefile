# EDA notebooks (UV) — OSRS skills, global war cost, scientific redactions
.PHONY: data data-osrs notebook notebook-osrs notebook-war notebook-redactions \
	execute execute-osrs execute-war execute-redactions execute-all notebook-server

# --- Data (optional local downloads; many notebooks use kagglehub inline) ---
data: data-osrs

data-osrs:
	kaggle datasets download samanthakamal33122/old-school-runescape-level-99-skill-dataset -p data --unzip

# --- Jupyter: open a specific notebook ---
notebook: notebook-osrs

notebook-osrs:
	uv run jupyter notebook osrs-skills-eda.ipynb

notebook-war:
	uv run jupyter notebook global-war-and-cost.ipynb

notebook-redactions:
	uv run jupyter notebook redactions-eda.ipynb

# --- Execute notebooks in place (clears then runs; saves outputs) ---
execute: execute-osrs

execute-osrs:
	uv run jupyter nbconvert --execute --to notebook --inplace osrs-skills-eda.ipynb

execute-war:
	uv run jupyter nbconvert --execute --to notebook --inplace global-war-and-cost.ipynb

execute-redactions:
	uv run jupyter nbconvert --execute --to notebook --inplace redactions-eda.ipynb

execute-all: execute-osrs execute-war execute-redactions

notebook-server:
	uv run jupyter notebook
