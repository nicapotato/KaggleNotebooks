# EDA notebooks (UV) — OSRS skills, OSRS money-making guides, global war cost, global poverty/inequality, banking fraud, company reviews/salaries, scientific redactions, AI workforce, teen mental health, AI dependency students, student performance, student AI tools, gaming vs academic performance, food delivery, Hacker News, speed dating, guided speed dating
.PHONY: execute execute-osrs execute-osrs-mmg execute-osrs-mmg-daily execute-war execute-global-poverty execute-banking-fraud execute-companie-reviews execute-redactions execute-ai execute-ai-student execute-student-perf execute-student-ai-tools execute-usda execute-spotify execute-teen-mh execute-gen-z execute-gaming-vs-academic execute-food-delivery execute-hacker-news execute-hacker-news-daily execute-speed-dating execute-guided-speed-dating execute-all notebook-server
# --- Execute notebooks in place (clears then runs; saves outputs) ---
execute: execute-osrs

execute-osrs:
	uv run jupyter nbconvert --execute --to notebook --inplace osrs-skills-eda.ipynb

execute-osrs-mmg:
	uv run jupyter nbconvert --execute --to notebook --inplace osrs-money-making-guides-eda.ipynb

execute-osrs-mmg-daily:
	uv run jupyter nbconvert --execute --to notebook --inplace osrs-money-makers-daily-scrape-eda.ipynb

execute-war:
	uv run jupyter nbconvert --execute --to notebook --inplace global-war-and-cost.ipynb

execute-global-poverty:
	uv run jupyter nbconvert --execute --to notebook --inplace global-poverty-and-inequality.ipynb

execute-banking-fraud:
	uv run jupyter nbconvert --execute --to notebook --inplace banking-fraud-detection-eda.ipynb

execute-companie-reviews:
	uv run jupyter nbconvert --execute --to notebook --inplace companie-reviews-salary.ipynb

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

execute-student-ai-tools:
	uv run jupyter nbconvert --execute --to notebook --inplace student-ai-tools.ipynb

execute-gen-z:
	uv run jupyter nbconvert --execute --to notebook --inplace gen-z-social-media-usage-eda.ipynb

execute-gaming-vs-academic:
	uv run jupyter nbconvert --execute --to notebook --inplace gaming-vs-academic-performance-eda.ipynb

execute-food-delivery:
	uv run jupyter nbconvert --execute --to notebook --inplace food-delivery-operations-and-customer-analytics-eda.ipynb

execute-hacker-news:
	uv run jupyter nbconvert --execute --to notebook --inplace hacker-news-eda.ipynb

execute-hacker-news-daily:
	uv run jupyter nbconvert --execute --to notebook --inplace hacker-news-daily-scrape-eda.ipynb

execute-speed-dating:
	uv run jupyter nbconvert --execute --to notebook --inplace speed-dating-attraction-eda.ipynb

execute-guided-speed-dating:
	uv run jupyter nbconvert --execute --to notebook --inplace guided-speed-dating-attraction-exploration.ipynb

execute-all: execute-osrs execute-osrs-mmg execute-osrs-mmg-daily execute-war execute-global-poverty execute-banking-fraud execute-companie-reviews execute-redactions execute-ai execute-usda execute-spotify execute-teen-mh execute-ai-student execute-student-perf execute-student-ai-tools execute-gen-z execute-gaming-vs-academic execute-food-delivery execute-hacker-news execute-hacker-news-daily execute-speed-dating execute-guided-speed-dating

notebook-server:
	uv run jupyter notebook
