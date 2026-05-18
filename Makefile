.PHONY: demo data train test

demo:
	streamlit run src/ui/app.py

data:
	python -m src.data.generator

train:
	python -m src.models.failure_prob

test:
	pytest -q
