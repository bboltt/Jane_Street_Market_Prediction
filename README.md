# Jane Street Market Prediction — Production-Ready Portfolio

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![API](https://img.shields.io/badge/API-FastAPI-009688)
![Tests](https://img.shields.io/badge/tests-pytest-success)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-deployable-brightgreen)

This repository now combines **research-grade modeling** with **deployment-grade engineering** for the Jane Street Market Prediction use case.

---

## Why this project stands out

- Notebook-first exploration preserved in `janestreet.ipynb`
- Reusable Python package under `src/jane_street_portfolio`
- Model service abstraction for training, saving, loading, and inference
- FastAPI app with health checks and batch prediction endpoint
- Containerized deployment path and CI pipeline

---

## System architecture

```mermaid
flowchart TD
    A[CSV / Batch Market Data] --> B[Feature Matrix Builder]
    B --> C[Model Service]
    C --> D[Sklearn Pipeline: Imputer + Logistic Regression]
    D --> E[Model Artifact (.joblib)]
    E --> F[FastAPI Inference Service]
    F --> G[/predict]
    F --> H[/health]
```

---

## Repository layout

```text
.
├── janestreet.ipynb
├── requirements.txt
├── Dockerfile
├── Makefile
├── .github/workflows/ci.yml
├── src/
│   └── jane_street_portfolio/
│       ├── __init__.py
│       ├── api.py
│       ├── cli.py
│       ├── config.py
│       ├── features.py
│       ├── modeling.py
│       └── service.py
└── tests/
    ├── test_api.py
    ├── test_features.py
    ├── test_modeling.py
    └── test_service.py
```

---

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
pytest -q
```

---

## Train and persist a model artifact

Assuming your CSV includes `feature_*` columns and a binary target column (`action` by default):

```bash
export PYTHONPATH=src
python -m jane_street_portfolio.cli train --data path/to/train.csv --target action --out artifacts/baseline_model.joblib
```

---

## Run API locally

```bash
export PYTHONPATH=src
uvicorn jane_street_portfolio.api:app --host 0.0.0.0 --port 8000
```

Example requests:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"rows": [{"feature_0": 0.12, "feature_1": 1.4}, {"feature_0": 0.77, "feature_1": 1.8}]}'
```

> Load a previously trained artifact via `POST /load` before inference when starting a fresh service instance.

---

## Docker deployment

Build and run:

```bash
docker build -t jane-street-service:latest .
docker run --rm -p 8000:8000 jane-street-service:latest
```

---

## CI pipeline

GitHub Actions workflow runs on each push/PR to validate installation, lint-free import paths, and tests.

---

## License

MIT License.
