# OSRS skills EDA — same pattern as denver-cpi
.PHONY: data notebook execute

data:
	kaggle datasets download samanthakamal33122/old-school-runescape-level-99-skill-dataset -p data --unzip

notebook:
	uv run jupyter notebook osrs-skills-eda.ipynb

execute:
	uv run jupyter nbconvert --execute --to notebook --inplace osrs-skills-eda.ipynb
