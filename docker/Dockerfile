# Utiliser une image Python 3.13 légère
FROM python:3.13-slim

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copier les requirements
COPY requirements.txt .

# Installer les dépendances Python
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copier l'application
COPY src/ /app/src/
COPY dataset/ /app/dataset/
RUN mkdir -p /app/model/ || true
COPY model/full_pipeline_xgb_optimized.pkl /app/model/full_pipeline_xgb_optimized.pkl

RUN chmod -R 755 /app/model /app/dataset

ENV PYTHONPATH=/app/src

# Exposer le port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Commande pour démarrer l'API
CMD ["python", "-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
