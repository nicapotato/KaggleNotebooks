# EDA notebooks (UV) — OSRS skills, global war cost, scientific redactions, AI workforce, teen mental health, AI dependency students, student performance, gaming vs academic performance, food delivery
.PHONY: data data-osrs notebook notebook-osrs notebook-war notebook-redactions notebook-ai notebook-ai-student notebook-student-perf notebook-usda notebook-spotify notebook-teen-mh notebook-gen-z notebook-gaming-vs-academic notebook-food-delivery \
	execute execute-osrs execute-war execute-redactions execute-ai execute-ai-student execute-student-perf execute-usda execute-spotify execute-teen-mh execute-gen-z execute-gaming-vs-academic execute-food-delivery execute-all notebook-server

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

notebook-teen-mh:
	uv run jupyter notebook teen-mental-health-eda.ipynb

notebook-ai-student:
	uv run jupyter notebook ai-dependency-student.ipynb

notebook-student-perf:
	uv run jupyter notebook student-performance-prediction-eda.ipynb

notebook-gen-z:
	uv run jupyter notebook gen-z-social-media-usage-eda.ipynb

notebook-gaming-vs-academic:
	uv run jupyter notebook gaming-vs-academic-performance-eda.ipynb

notebook-food-delivery:
	uv run jupyter notebook food-delivery-operations-and-customer-analytics-eda.ipynb

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

execute-teen-mh:
	uv run jupyter nbconvert --execute --to notebook --inplace teen-mental-health-eda.ipynb

execute-ai-student:
	uv run jupyter nbconvert --execute --to notebook --inplace ai-dependency-student.ipynb

execute-student-perf:
	uv run jupyter nbconvert --execute --to notebook --inplace student-performance-prediction-eda.ipynb

execute-gen-z:
	uv run jupyter nbconvert --execute --to notebook --inplace gen-z-social-media-usage-eda.ipynb

execute-gaming-vs-academic:
	uv run jupyter nbconvert --execute --to notebook --inplace gaming-vs-academic-performance-eda.ipynb

execute-food-delivery:
	uv run jupyter nbconvert --execute --to notebook --inplace food-delivery-operations-and-customer-analytics-eda.ipynb

execute-all: execute-osrs execute-war execute-redactions execute-ai execute-usda execute-spotify execute-teen-mh execute-ai-student execute-student-perf execute-gaming-vs-academic execute-food-delivery

notebook-server:
	uv run jupyter notebook
