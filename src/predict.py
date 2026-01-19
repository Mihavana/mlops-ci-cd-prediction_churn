"""
Module de prédiction pour le churn client.
Charge le modèle entraîné et effectue les prédictions.
"""

import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Le chemin par défaut pour le développement local sur votre PC
BASE_DIR = Path(__file__).parent.parent 
# Le chemin dans Docker, via la variable d'environnement (si elle existe)
DOCKER_APP_DIR = os.environ.get("APP_DIR", "")

# Définition du chemin du modèle
MODEL_FILENAME = 'full_pipeline_xgb_optimized.pkl'

# Choisir le chemin approprié
if DOCKER_APP_DIR:
    MODEL_PATH = Path(DOCKER_APP_DIR) / "model" / MODEL_FILENAME
else:
    MODEL_PATH = BASE_DIR / "model" / MODEL_FILENAME

# Seuil optimal déterminé lors de l'entraînement (F1-score)
OPTIMAL_THRESHOLD = 0.5172


def load_model(model_path=None):
    """
    Charge le modèle XGBoost optimisé et ses métadonnées.
    
    Args:
        model_path: Chemin vers le modèle. Si None, utilise le chemin par défaut.
    
    Returns:
        tuple: (model, features_order)
    """
    if model_path is None:
        model_path = MODEL_PATH
    
    if not os.path.exists(model_path):
        print(f"DEBUG: Le chemin {model_path} n'existe pas!")
        raise FileNotFoundError(f"Le fichier modèle est introuvable à : {model_path}")
    
    
    metadata = joblib.load(model_path)
    
    if isinstance(metadata, dict):
        model = metadata['model']
        features_order = metadata['features_order']
    else:
        # Rétrocompatibilité si le modèle est sauvegardé directement
        model = metadata
        features_order = None
    
    return model, features_order


def preprocess_input(data, features_order=None):
    """
    Prétraite les données d'entrée avant la prédiction.
    
    Args:
        data: DataFrame ou dict avec les features du client
        features_order: Ordre des features utilisé lors de l'entraînement
    
    Returns:
        pd.DataFrame: Données prétraitées
    """
    # Convertir en DataFrame si dictionnaire
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data.copy()
    
    # Conversion de TotalCharges si présent
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # S'assurer que l'ordre des colonnes correspond au modèle
    if features_order is not None:
        # Vérifier que toutes les colonnes requises sont présentes
        missing_cols = set(features_order) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Colonnes manquantes: {missing_cols}")
        
        # Réordonner les colonnes
        df = df[features_order]
    
    return df


def predict_single(client_data, model=None, features_order=None, threshold=OPTIMAL_THRESHOLD):
    """
    Effectue une prédiction pour un seul client.
    
    Args:
        client_data: Dictionnaire ou DataFrame avec les données du client
        model: Modèle à utiliser (si None, charge le modèle par défaut)
        features_order: Ordre des features
        threshold: Seuil de décision pour la prédiction binaire
    
    Returns:
        dict: {
            'prediction': 0 ou 1 (0=No Churn, 1=Churn),
            'churn': 'Yes' ou 'No',
            'probability': probabilité de churn (0-1),
            'confidence': confiance de la prédiction
        }
    """
    if model is None or features_order is None:
        model, features_order = load_model()
    
    # Prétraiter les données
    df = preprocess_input(client_data, features_order)
    
    # Effectuer la prédiction
    probability = model.predict_proba(df)[0][1]  # Probabilité de churn (classe 1)
    prediction = 1 if probability >= threshold else 0
    
    # Calculer la confiance (distance au seuil 0.5)
    confidence = max(probability, 1 - probability)
    
    return {
        'prediction': int(prediction),
        'churn': 'Yes' if prediction == 1 else 'No',
        'probability': float(probability),
        'confidence': float(confidence)
    }


def predict_batch(data, model=None, features_order=None, threshold=OPTIMAL_THRESHOLD):
    """
    Effectue des prédictions en batch pour plusieurs clients.
    
    Args:
        data: DataFrame avec les données des clients
        model: Modèle à utiliser (si None, charge le modèle par défaut)
        features_order: Ordre des features
        threshold: Seuil de décision pour la prédiction binaire
    
    Returns:
        pd.DataFrame: DataFrame avec colonnes supplémentaires:
            - 'churn_prediction': 0 ou 1
            - 'churn_probability': probabilité de churn
            - 'churn': 'Yes' ou 'No'
    """
    if model is None or features_order is None:
        model, features_order = load_model()
    
    # Prétraiter les données
    df = preprocess_input(data, features_order)
    
    # Effectuer les prédictions
    probabilities = model.predict_proba(df)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    
    # Ajouter les résultats au DataFrame
    result = data.copy()
    result['churn_probability'] = probabilities
    result['churn_prediction'] = predictions
    result['churn'] = result['churn_prediction'].map({0: 'No', 1: 'Yes'})
    
    return result


# ============ SCRIPT STANDALONE ============

if __name__ == "__main__":
    
    print("\n" + "="*60)
    print("TEST DE PRÉDICTION DU MODULE")
    print("="*60)
    
    # Créer des données de test
    test_data = {
        # --- Colonnes Numériques ---
        'tenure':         [3.0, 70.0],
        'MonthlyCharges': [105.00, 20.00],
        'TotalCharges':   [315.00, 1400.00],
        'SeniorCitizen':  [0, 1],

        # --- Colonnes Catégorielles ---
        'gender':          ['Male', 'Female'],
        'Partner':         ['No', 'Yes'],
        'Dependents':      ['No', 'Yes'],
        'PhoneService':    ['Yes', 'Yes'],
        'MultipleLines':   ['Yes', 'No'],
        'InternetService': ['Fiber optic', 'No'],
        'OnlineSecurity':  ['No', 'Yes'],
        'OnlineBackup':    ['No', 'Yes'],
        'DeviceProtection':['No', 'Yes'],
        'TechSupport':     ['No', 'Yes'],
        'StreamingTV':     ['Yes', 'No'],
        'StreamingMovies': ['Yes', 'No'],
        'Contract':        ['Month-to-month', 'Two year'],
        'PaperlessBilling':['Yes', 'No'],
        'PaymentMethod':   ['Electronic check', 'Bank transfer (automatic)'],
    }
    
    df_test = pd.DataFrame(test_data)
    
    try:
        # Charger le modèle
        loaded_model, required_columns = load_model()
        print(f"✓ Modèle chargé avec {len(required_columns)} features")
        
        # Test 1: Prédiction simple
        print("\n--- TEST 1: Prédiction Simple (Client 1) ---")
        client_1 = {k: v[0] for k, v in test_data.items()}
        result_single = predict_single(client_1, loaded_model, required_columns)
        print(f"Prédiction: {result_single['churn']}")
        print(f"Probabilité: {result_single['probability']:.4f}")
        print(f"Confiance: {result_single['confidence']:.4f}")
        
        # Test 2: Prédictions en batch
        print("\n--- TEST 2: Prédictions en Batch (2 clients) ---")
        results_batch = predict_batch(df_test, loaded_model, required_columns)
        print(results_batch[['churn', 'churn_probability']].to_string())
        
        print("\n✓ Tous les tests de prédiction réussis!")
        
    except FileNotFoundError as e:
        print(f"✗ Erreur: {e}")
        print("\nACTION REQUISE: Exécutez train.py pour entraîner et sauvegarder le modèle.")
