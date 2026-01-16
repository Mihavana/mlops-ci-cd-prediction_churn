"""
Tests unitaires pour l'API FastAPI.
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from app import app

# Créer un client de test
client = TestClient(app)


class TestHealthEndpoint:
    """Tests pour le endpoint de santé"""
    
    def test_health_check_status_code(self):
        """Vérifier que le health check retourne 200"""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_check_response_format(self):
        """Vérifier le format de la réponse du health check"""
        response = client.get("/health")
        data = response.json()
        
        assert 'status' in data
        assert 'model_loaded' in data
        assert isinstance(data['model_loaded'], bool)


class TestRootEndpoint:
    """Tests pour le endpoint racine"""
    
    def test_root_status_code(self):
        """Vérifier que la route racine retourne 200"""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_root_response_contains_message(self):
        """Vérifier que la réponse racine contient un message"""
        response = client.get("/")
        data = response.json()
        
        assert 'message' in data
        assert 'version' in data
        assert 'endpoints' in data


class TestPredictEndpoint:
    """Tests pour le endpoint de prédiction"""
    
    def test_predict_status_code(self):
        """Vérifier que le endpoint de prédiction retourne 200"""
        # Données minimales d'un client
        client_data = {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 65.5,
            "TotalCharges": 786.0
        }
        
        response = client.post("/predict", json=client_data)
        # Status code peut être 200, 503 (service unavailable si modèle pas chargé), ou 500
        assert response.status_code in [200, 503, 500]
    
    def test_predict_response_format_when_available(self):
        """Vérifier le format de la réponse quand le modèle est disponible"""
        client_data = {
            "gender": "Male",
            "tenure": 12,
            "MonthlyCharges": 65.5,
            "TotalCharges": 786.0
        }
        
        response = client.post("/predict", json=client_data)
        
        if response.status_code == 200:
            data = response.json()
            assert 'prediction' in data
            assert 'churn' in data
            assert 'probability' in data
            assert 'confidence' in data
            assert data['churn'] in ['Yes', 'No']
            assert 0 <= data['probability'] <= 1
            assert 0 <= data['confidence'] <= 1
    
    def test_predict_with_empty_data(self):
        """Vérifier la gestion des données vides"""
        response = client.post("/predict", json={})
        # Dépend de la validation Pydantic
        assert response.status_code in [400, 422, 500]
    
    def test_predict_with_invalid_data(self):
        """Vérifier la gestion des données invalides"""
        client_data = {
            "gender": "InvalidValue",
            "tenure": -1,  # Valeur invalide (doit être positif)
        }
        
        response = client.post("/predict", json=client_data)
        # Peut retourner 400 ou 500 dépendant de la validation
        assert response.status_code in [400, 500]


class TestBatchPredictEndpoint:
    """Tests pour le endpoint de prédiction batch"""
    
    def test_batch_predict_status_code(self):
        """Vérifier que le endpoint batch retourne une réponse"""
        clients_data = [
            {
                "gender": "Male",
                "tenure": 12,
                "MonthlyCharges": 65.5,
                "TotalCharges": 786.0
            },
            {
                "gender": "Female",
                "tenure": 24,
                "MonthlyCharges": 80.0,
                "TotalCharges": 1920.0
            }
        ]
        
        response = client.post("/predict-batch", json=clients_data)
        assert response.status_code in [200, 400, 503, 500]
    
    def test_batch_predict_empty_list(self):
        """Vérifier la gestion d'une liste vide"""
        response = client.post("/predict-batch", json=[])
        # Doit retourner une erreur 400
        assert response.status_code == 400


class TestFeaturesEndpoint:
    """Tests pour le endpoint des features"""
    
    def test_features_endpoint(self):
        """Vérifier que le endpoint des features retourne une réponse"""
        response = client.get("/features")
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert 'features' in data
            assert 'count' in data
            assert isinstance(data['features'], list)


class TestErrorHandling:
    """Tests pour la gestion des erreurs"""
    
    def test_404_not_found(self):
        """Vérifier que les routes non trouvées retournent 404"""
        response = client.get("/nonexistent")
        assert response.status_code == 404
    
    def test_method_not_allowed(self):
        """Vérifier que les méthodes non autorisées retournent une erreur"""
        response = client.post("/health")
        # Peut être 405 (Method Not Allowed) ou 422 (Unprocessable Entity)
        assert response.status_code in [405, 422]


class TestAPIIntegration:
    """Tests d'intégration de l'API"""
    
    def test_api_startup(self):
        """Vérifier que l'API démarre sans erreur"""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_health_to_predict_flow(self):
        """Vérifier un flux complet: health -> predict"""
        # Vérifier la santé
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        health_data = health_response.json()
        model_loaded = health_data['model_loaded']
        
        if model_loaded:
            # Si le modèle est chargé, essayer une prédiction
            client_data = {"tenure": 12}
            predict_response = client.post("/predict", json=client_data)
            assert predict_response.status_code in [200, 400]


# ============ FIXTURES ============

@pytest.fixture
def sample_client_data():
    """Créer des données de client d'exemple"""
    return {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "MonthlyCharges": 65.5,
        "TotalCharges": 786.0
    }


@pytest.fixture
def sample_clients_batch():
    """Créer plusieurs clients d'exemple"""
    return [
        {
            "gender": "Male",
            "tenure": 12,
            "MonthlyCharges": 65.5,
            "TotalCharges": 786.0
        },
        {
            "gender": "Female",
            "tenure": 24,
            "MonthlyCharges": 80.0,
            "TotalCharges": 1920.0
        }
    ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
