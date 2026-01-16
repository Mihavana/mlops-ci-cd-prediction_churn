"""
API FastAPI pour la prédiction de churn client.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import pandas as pd
import logging

from predict import load_model, predict_single, predict_batch, preprocess_input

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialisation de l'application FastAPI
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API pour prédire le churn client avec un modèle XGBoost optimisé",
    version="1.0.0"
)

# Charger le modèle au démarrage
try:
    MODEL, FEATURES_ORDER = load_model()
    logger.info("✓ Modèle chargé avec succès au démarrage")
except Exception as e:
    logger.error(f"✗ Erreur lors du chargement du modèle: {e}")
    MODEL = None
    FEATURES_ORDER = None


# ============ MODÈLES PYDANTIC ============

class ClientData(BaseModel):
    """Modèle pour les données d'un client"""
    gender: Optional[str] = Field(None, example="Male")
    SeniorCitizen: Optional[int] = Field(None, example=0)
    Partner: Optional[str] = Field(None, example="Yes")
    Dependents: Optional[str] = Field(None, example="No")
    tenure: Optional[int] = Field(None, example=12)
    PhoneService: Optional[str] = Field(None, example="Yes")
    MultipleLines: Optional[str] = Field(None, example="No")
    InternetService: Optional[str] = Field(None, example="Fiber optic")
    OnlineSecurity: Optional[str] = Field(None, example="No")
    OnlineBackup: Optional[str] = Field(None, example="No")
    DeviceProtection: Optional[str] = Field(None, example="No")
    TechSupport: Optional[str] = Field(None, example="No")
    StreamingTV: Optional[str] = Field(None, example="No")
    StreamingMovies: Optional[str] = Field(None, example="No")
    Contract: Optional[str] = Field(None, example="Month-to-month")
    PaperlessBilling: Optional[str] = Field(None, example="Yes")
    PaymentMethod: Optional[str] = Field(None, example="Electronic check")
    MonthlyCharges: Optional[float] = Field(None, example=65.5)
    TotalCharges: Optional[float] = Field(None, example=786.0)


class PredictionResponse(BaseModel):
    """Modèle pour la réponse de prédiction"""
    prediction: int = Field(..., description="0: No Churn, 1: Churn")
    churn: str = Field(..., description="Yes ou No")
    probability: float = Field(..., description="Probabilité de churn (0-1)")
    confidence: float = Field(..., description="Confiance de la prédiction")


class HealthResponse(BaseModel):
    """Modèle pour la réponse de santé"""
    status: str = Field(..., example="healthy")
    model_loaded: bool = Field(..., example=True)
    features_count: Optional[int] = Field(None, example=19)


# ============ ROUTES ============

@app.get("/", tags=["General"])
async def root():
    """Route racine avec information sur l'API"""
    return {
        "message": "API de prédiction de churn client",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Vérifier la santé de l'API et du modèle"""
    features_count = len(FEATURES_ORDER) if FEATURES_ORDER else None
    
    return {
        "status": "healthy" if MODEL is not None else "unhealthy",
        "model_loaded": MODEL is not None,
        "features_count": features_count
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(client_data: ClientData):
    """
    Prédire le churn pour un client.
    
    Prend en entrée les données du client et retourne la prédiction.
    """
    if MODEL is None or FEATURES_ORDER is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé. Service indisponible."
        )
    
    try:
        # Convertir le modèle Pydantic en dictionnaire
        client_dict = client_data.model_dump(exclude_none=True)
        
        # Effectuer la prédiction
        result = predict_single(client_dict, MODEL, FEATURES_ORDER)
        
        return result
    
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Erreur dans les données: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        raise HTTPException(
            status_code=500,
            detail="Erreur interne lors de la prédiction"
        )


@app.post("/predict-batch", tags=["Prediction"])
async def predict_batch_endpoint(clients: List[ClientData]):
    """
    Prédire le churn pour plusieurs clients en batch.
    
    Prend en entrée une liste de clients et retourne les prédictions.
    """
    if MODEL is None or FEATURES_ORDER is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé. Service indisponible."
        )
    
    if not clients:
        raise HTTPException(
            status_code=400,
            detail="La liste des clients ne peut pas être vide"
        )
    
    try:
        # Convertir la liste de modèles Pydantic en DataFrame
        clients_list = [client.model_dump(exclude_none=True) for client in clients]
        df = pd.DataFrame(clients_list)
        
        # Effectuer les prédictions en batch
        result = predict_batch(df, MODEL, FEATURES_ORDER)
        
        # Retourner seulement les colonnes pertinentes
        response = result[['churn', 'churn_probability']].to_dict('records')
        
        return {
            "count": len(response),
            "predictions": response
        }
    
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction batch: {e}")
        raise HTTPException(
            status_code=500,
            detail="Erreur interne lors de la prédiction batch"
        )


@app.get("/features", tags=["Info"])
async def get_features():
    """Retourner la liste des features utilisées par le modèle"""
    if FEATURES_ORDER is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé. Service indisponible."
        )
    
    return {
        "features": FEATURES_ORDER,
        "count": len(FEATURES_ORDER)
    }


# ============ EXCEPTION HANDLERS ============

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Gestionnaire d'exception générale"""
    logger.error(f"Erreur non gérée: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Erreur interne du serveur"}
    )


# ============ ÉVÉNEMENTS DE STARTUP/SHUTDOWN ============

@app.on_event("startup")
async def startup_event():
    """Événement de démarrage"""
    logger.info("Application démarrée")
    if MODEL is not None:
        logger.info(f"✓ Modèle prêt avec {len(FEATURES_ORDER)} features")
    else:
        logger.warning("⚠ Modèle non disponible")


@app.on_event("shutdown")
async def shutdown_event():
    """Événement d'arrêt"""
    logger.info("Application arrêtée")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
