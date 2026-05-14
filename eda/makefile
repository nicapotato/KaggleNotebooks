# EDA notebooks (UV) — OSRS skills, global war cost, scientific redactions
.PHONY: data data-osrs notebook notebook-osrs notebook-war notebook-redactions notebook-ai notebook-usda notebook-spotify \
	execute execute-osrs execute-war execute-redactions execute-ai execute-usda execute-spotify execute-all notebook-server

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

notebook-ai:
	uv run jupyter notebook ai-workforce-displacement.ipynb

notebook-usda:
	uv run jupyter notebook usda-food-analysis.ipynb

notebook-spotify:
	uv run jupyter notebook spotify-eda-2014-2026.ipynb

# --- Execute notebooks in place (clears then runs; saves outputs) ---
execute: execute-osrs

execute-osrs:
	uv run jupyter nbconvert --execute --to notebook --inplace osrs-skills-eda.ipynb

execute-war:
	uv run jupyter nbconvert --execute --to notebook --inplace global-war-and-cost.ipynb

execute-redactions:
	uv run jupyter nbconvert --execute --to notebook --inplace redactions-eda.ipynb

execute-ai:
	uv run jupyter nbconvert --execute --to notebook --inplace ai-workforce-displacement.ipynb

execute-usda:
	uv run jupyter nbconvert --execute --to notebook --inplace usda-food-analysis.ipynb

execute-spotify:
	uv run jupyter nbconvert --execute --to notebook --inplace spotify-eda-2014-2026.ipynb

execute-all: execute-osrs execute-war execute-redactions execute-ai execute-usda execute-spotify

notebook-server:
	uv run jupyter notebook
