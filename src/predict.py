import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model')
MODEL_FILENAME = 'full_pipeline_xgb_optimized.pkl'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

OPTIMAL_THRESHOLD = 0.5172
def load_model_and_metadata(path: str = MODEL_PATH):
    """Charge le modèle et la liste des colonnes."""
    metadata = joblib.load(path)
    return metadata['model'], metadata['features_order']

def make_prediction(data_raw: pd.DataFrame, model, required_columns: list, threshold: float = OPTIMAL_THRESHOLD):
    # 1. Vérification des colonnes existantes
    missing_cols = set(required_columns) - set(data_raw.columns)
    if missing_cols:
        raise ValueError(f"Données manquantes : {missing_cols}. Le modèle nécessite ces colonnes.")

    # 2. Garantie de l'ordre correct
    # Le DataFrame d'entrée est réordonné pour correspondre exactement à l'ordre d'entraînement
    data_ordered = data_raw[required_columns]

    """Effectue des prédictions de churn."""
    # Le pipeline (model) gère l'application du prétraitement sur data_raw
    proba_churn = model.predict_proba(data_raw)[:, 1]
    prediction_binary = (proba_churn >= threshold).astype(int)
    
    results = pd.DataFrame({
        'Churn_Probability': proba_churn,
        'Churn_Prediction': prediction_binary
    }, index=data_raw.index)
    
    return results

if __name__ == "__main__":
    
    # --- 1. Création de l'Exemple de Données Brutes ---
    
    # Nous créons des profils clients qui représentent des cas de churn (Client 1)
    # et de non-churn (Client 2) potentiels.
    
    data = {
        # --- Colonnes Numériques ---
        'tenure':         [3.0],    # P1 (3 mois), P2 (70 mois)
        'MonthlyCharges': [100.00], # P1 (Haut), P2 (Faible)
        'TotalCharges':   [140.00],
        'SeniorCitizen':  [0],         # P1 (Non Senior), P2 (Senior)

        # --- Colonnes Catégorielles ---
        'gender':          ['Male'],
        'Partner':         ['No'],
        'Dependents':      ['No'],
        'PhoneService':    ['Yes'],
        'MultipleLines':   ['Yes'],
        'InternetService': ['Fiber optic'],
        'OnlineSecurity':  ['No'],
        'OnlineBackup':    ['No'],
        'DeviceProtection':['No'],
        'TechSupport':     ['No'],
        'StreamingTV':     ['Yes'],
        'StreamingMovies': ['Yes'],
        'Contract':        ['month-to-month'],
        'PaperlessBilling':['Yes'],
        'PaymentMethod':   ['Electronic check'],
    }
    
    df_test = pd.DataFrame(data)
    
    # IMPORTANT : Assurez-vous que l'ordre des colonnes de df_test est le même 
    # que l'ordre des colonnes utilisé dans X_train
    
    try:
        loaded_model, required_columns = load_model_and_metadata()

        print("\n--- TEST DE PRÉDICTION DÉMARRE ---")

        predictions = make_prediction(
            data_raw=df_test, 
            model=loaded_model, 
            required_columns=required_columns, # <-- L'argument est passé ici !
            threshold=OPTIMAL_THRESHOLD
        )
        
        print("\n--- RÉSULTATS DE PRÉDICTION ---")
        print(predictions)
        
    except FileNotFoundError as e:
        print(e)
        print("\nACTION REQUISE : Exécutez train.py pour sauvegarder le modèle optimisé.")