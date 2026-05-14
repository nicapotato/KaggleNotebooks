# EDA notebooks (UV) — OSRS skills, global war cost, scientific redactions, AI workforce, teen mental health, AI dependency students, student performance, gaming vs academic performance, food delivery
.PHONY: execute execute-osrs execute-war execute-redactions execute-ai execute-ai-student execute-student-perf execute-usda execute-spotify execute-teen-mh execute-gen-z execute-gaming-vs-academic execute-food-delivery execute-all notebook-server
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
