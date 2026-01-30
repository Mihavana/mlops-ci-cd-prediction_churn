# mlops-ci-cd-prediction_churn

Application ML Ops pour la prédiction du churn client avec modèle XGBoost, API FastAPI et pipeline CI/CD.

## Description

Ce projet est une application complète de Machine Learning Ops (MLOps) qui :
- **Entraîne** un modèle XGBoost pour prédire le churn client
- **Expose** une API REST FastAPI pour les prédictions
- **Déploie** l'application via Docker et Docker Compose
- **Automatise** le CI/CD avec Jenkins

Dataset utilisé: [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)

## Démarrage rapide

### Entraîner le modèle
```bash
python src/train.py
```

### Exécuter l'API
```bash
python -m uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```

### Avec Docker
```bash
docker-compose up --build
```

## API Endpoints

- `GET /health` - Vérifier la santé
- `POST /predict` - Prédiction simple
- `POST /predict-batch` - Prédictions en batch
- `GET /features` - Lister les features

Documentation: http://localhost:8000/docs

## Tests
```bash
pytest
```

## Dépendances
- fastapi
- uvicorn
- xgboost
- scikit-learn
- pandas
- numpy
- pytest
