"""
Module d'entraînement du modèle de prédiction de churn client.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.model_selection import GridSearchCV
import joblib
from pathlib import Path
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chemins par défaut
DATASET_PATH = Path(__file__).parent.parent / "dataset" / "Telco-Customer-Churn.csv"
MODEL_PATH = Path(__file__).parent.parent / "model" / "full_pipeline_xgb_optimized.pkl"

# Créer le répertoire model s'il n'existe pas
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# Features catégorielles
CATEGORIAL_FEATURES = [
    'gender', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]


def load_data(dataset_path=None):
    """
    Charge les données du dataset.
    
    Args:
        dataset_path: Chemin vers le dataset. Si None, utilise le chemin par défaut.
    
    Returns:
        pd.DataFrame: Les données brutes
    """
    if dataset_path is None:
        dataset_path = DATASET_PATH
    
    logger.info(f"Chargement des données depuis {dataset_path}")
    df = pd.read_csv(dataset_path)
    logger.info(f"Données chargées: {df.shape}")
    return df


def clean_data(df):
    """
    Nettoie les données.
    
    Args:
        df: DataFrame brut
    
    Returns:
        pd.DataFrame: DataFrame nettoyé
    """
    logger.info("Nettoyage des données")
    
    # Supprimer customerID
    df = df.drop(columns=['customerID'], errors='ignore')
    
    # Convertir TotalCharges en numérique
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Supprimer les lignes avec valeurs manquantes
    df_clean = df.dropna()
    
    logger.info(f"Données nettoyées: {len(df_clean)} lignes")
    return df_clean


def create_features(df_clean):
    """
    Crée les features X et la cible y.
    
    Args:
        df_clean: DataFrame nettoyé
    
    Returns:
        tuple: (X, y_encoded)
    """
    logger.info("Création des features")
    
    X = df_clean.drop('Churn', axis=1)
    y = df_clean['Churn']
    
    # Encoder la variable cible
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    logger.info(f"Features créées: {X.shape[1]} features, {len(y)} samples")
    return X, y_encoded


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Sépare les données en train et test.
    
    Args:
        X: Features
        y: Cible
        test_size: Proportion du test set
        random_state: Seed pour la reproductibilité
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    logger.info(f"Split des données (test_size={test_size})")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def create_preprocessor():
    """
    Crée le préprocesseur pour les données catégorielles.
    
    Returns:
        ColumnTransformer: Le préprocesseur
    """
    logger.info("Création du préprocesseur")
    
    categorial_pipeline = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorial_pipeline, CATEGORIAL_FEATURES)
        ],
        remainder='passthrough'
    )
    
    return preprocessor


def create_pipeline(preprocessor):
    """
    Crée le pipeline complet avec préprocesseur et modèle XGBoost.
    
    Args:
        preprocessor: Le préprocesseur ColumnTransformer
    
    Returns:
        Pipeline: Le pipeline complet
    """
    logger.info("Création du pipeline")
    
    xgb_model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        scale_pos_weight=2.7,  # Ratio des classes
        random_state=42
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb_model)
    ])
    
    return pipeline


def train_model(X_train, y_train):
    """
    Entraîne le modèle avec GridSearch.
    
    Args:
        X_train: Features d'entraînement
        y_train: Cible d'entraînement
    
    Returns:
        Pipeline: Le modèle optimisé
    """
    logger.info("Création du modèle et entraînement")
    
    preprocessor = create_preprocessor()
    pipeline = create_pipeline(preprocessor)
    
    # GridSearch
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.05, 0.1, 0.2],
        'classifier__scale_pos_weight': [2.5, 2.7, 3.0]
    }
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=5,
        verbose=1,
        n_jobs=-1
    )
    
    logger.info("Démarrage de la recherche par grille...")
    grid_search.fit(X_train, y_train)
    
    logger.info(f"Meilleur score AUC-ROC: {grid_search.best_score_:.4f}")
    logger.info(f"Meilleurs hyperparamètres: {grid_search.best_params_}")
    
    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test):
    """
    Évalue le modèle.
    
    Args:
        model: Le modèle entraîné
        X_test: Features de test
        y_test: Cible de test
    
    Returns:
        dict: Métriques d'évaluation
    """
    logger.info("Évaluation du modèle")
    
    # Prédictions probabilistes
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Score AUC-ROC
    auc_score = roc_auc_score(y_test, y_pred_proba)
    logger.info(f"AUC-ROC: {auc_score:.4f}")
    
    # Déterminer le seuil optimal
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx]
    logger.info(f"Seuil optimal (F1): {best_threshold:.4f}")
    
    return {
        'auc_score': auc_score,
        'best_threshold': best_threshold,
        'y_pred_proba': y_pred_proba
    }


def save_model(model, features, model_path=None):
    """
    Sauvegarde le modèle et ses métadonnées.
    
    Args:
        model: Le modèle entraîné
        features: Liste des noms de features
        model_path: Chemin de sauvegarde. Si None, utilise le chemin par défaut.
    """
    if model_path is None:
        model_path = MODEL_PATH
    
    logger.info(f"Sauvegarde du modèle vers {model_path}")
    
    metadata = {
        'model': model,
        'features_order': list(features)
    }
    
    joblib.dump(metadata, model_path)
    logger.info("✓ Modèle sauvegardé avec succès")


def train_full_pipeline(dataset_path=None, model_path=None):
    """
    Exécute le pipeline complet d'entraînement.
    
    Args:
        dataset_path: Chemin vers le dataset
        model_path: Chemin de sauvegarde du modèle
    """
    # Chargement et nettoyage
    df = load_data(dataset_path)
    df_clean = clean_data(df)
    
    # Création des features
    X, y = create_features(df_clean)
    
    # Split
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Entraînement
    best_model = train_model(X_train, y_train)
    
    # Évaluation
    metrics = evaluate_model(best_model, X_test, y_test)
    
    # Sauvegarde
    save_model(best_model, X_train.columns, model_path)
    
    logger.info("✓ Entraînement complété avec succès!")
    
    return best_model, metrics


# ============ SCRIPT PRINCIPAL ============

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Entraînement du modèle de prédiction de churn")
    logger.info("=" * 60)
    
    try:
        best_model, metrics = train_full_pipeline()
        logger.info(f"\nAUC-ROC final: {metrics['auc_score']:.4f}")
        logger.info(f"Seuil optimal: {metrics['best_threshold']:.4f}")
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement: {e}", exc_info=True)
        exit(1)
