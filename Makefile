.PHONY: install test run train

install:
	pip install -r requirements.txt

test:
	PYTHONPATH=src pytest -q

run:
	PYTHONPATH=src uvicorn jane_street_portfolio.api:app --host 0.0.0.0 --port 8000

train:
	PYTHONPATH=src python -m jane_street_portfolio.cli train --data data/train.csv --target action --out artifacts/baseline_model.joblib
