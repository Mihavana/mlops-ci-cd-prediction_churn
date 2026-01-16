"""
Tests unitaires pour le module d'entraînement.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from train import (
    load_data,
    clean_data,
    create_features,
    split_data,
    train_model,
    evaluate_model
)


class TestDataLoading:
    """Tests pour le chargement des données"""
    
    def test_load_data(self):
        """Test le chargement des données"""
        df = load_data()
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'Churn' in df.columns
    
    def test_data_shape(self):
        """Vérifier la forme des données"""
        df = load_data()
        assert df.shape[0] > 0  # Au moins une ligne
        assert df.shape[1] > 5  # Au moins plusieurs colonnes


class TestDataCleaning:
    """Tests pour le nettoyage des données"""
    
    def test_clean_data_removes_null(self):
        """Vérifier que le nettoyage supprime les valeurs NULL"""
        df = load_data()
        df_clean = clean_data(df)
        assert df_clean.isnull().sum().sum() == 0
    
    def test_clean_data_removes_customerid(self):
        """Vérifier que customerID est supprimé"""
        df = load_data()
        df_clean = clean_data(df)
        assert 'customerID' not in df_clean.columns
    
    def test_clean_data_converts_totalcharges(self):
        """Vérifier que TotalCharges est converti en numérique"""
        df = load_data()
        df_clean = clean_data(df)
        if 'TotalCharges' in df_clean.columns:
            assert pd.api.types.is_numeric_dtype(df_clean['TotalCharges'])
    
    def test_clean_data_reduces_size(self):
        """Vérifier que le nettoyage réduit la taille des données"""
        df = load_data()
        df_clean = clean_data(df)
        assert len(df_clean) <= len(df)


class TestFeatureCreation:
    """Tests pour la création de features"""
    
    def test_features_created(self):
        """Vérifier que les features sont créées"""
        df = load_data()
        df_clean = clean_data(df)
        X, y = create_features(df_clean)
        
        assert X is not None
        assert y is not None
        assert len(X) == len(y)
    
    def test_target_variable_binary(self):
        """Vérifier que la variable cible est binaire"""
        df = load_data()
        df_clean = clean_data(df)
        X, y = create_features(df_clean)
        
        unique_values = np.unique(y)
        assert len(unique_values) == 2
        assert all(val in [0, 1] for val in unique_values)


class TestDataSplitting:
    """Tests pour le split train/test"""
    
    def test_split_data(self):
        """Vérifier le split des données"""
        df = load_data()
        df_clean = clean_data(df)
        X, y = create_features(df_clean)
        
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        assert X_train is not None
        assert X_test is not None
        assert len(X_train) > 0
        assert len(X_test) > 0
    
    def test_split_ratio(self):
        """Vérifier le ratio du split (80/20)"""
        df = load_data()
        df_clean = clean_data(df)
        X, y = create_features(df_clean)
        
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        total = len(X_train) + len(X_test)
        train_ratio = len(X_train) / total
        
        assert 0.75 < train_ratio < 0.85  # Permettre une petite marge


class TestModelTraining:
    """Tests pour l'entraînement du modèle"""
    
    def test_model_trained(self):
        """Vérifier que le modèle est entraîné"""
        df = load_data()
        df_clean = clean_data(df)
        X, y = create_features(df_clean)
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        model = train_model(X_train, y_train)
        
        assert model is not None
    
    def test_model_can_predict(self):
        """Vérifier que le modèle peut faire des prédictions"""
        df = load_data()
        df_clean = clean_data(df)
        X, y = create_features(df_clean)
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        model = train_model(X_train, y_train)
        predictions = model.predict(X_test[:5])
        
        assert len(predictions) == 5
        assert all(pred in [0, 1] for pred in predictions)


class TestModelEvaluation:
    """Tests pour l'évaluation du modèle"""
    
    def test_model_evaluation(self):
        """Vérifier que le modèle est évalué"""
        df = load_data()
        df_clean = clean_data(df)
        X, y = create_features(df_clean)
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        model = train_model(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        
        assert metrics is not None
        assert 'auc_score' in metrics
    
    def test_auc_score_valid(self):
        """Vérifier que le score AUC est valide"""
        df = load_data()
        df_clean = clean_data(df)
        X, y = create_features(df_clean)
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        model = train_model(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        
        auc_score = metrics['auc_score']
        assert 0 <= auc_score <= 1


# ============ FIXTURES ============

@pytest.fixture
def sample_data():
    """Créer un DataFrame d'exemple pour les tests"""
    return pd.DataFrame({
        'gender': ['Male', 'Female'],
        'SeniorCitizen': [0, 1],
        'tenure': [12, 24],
        'MonthlyCharges': [65.5, 80.0],
        'TotalCharges': [786.0, 1920.0],
        'Churn': ['No', 'Yes']
    })


# ============ TESTS UTILITAIRES ============

def test_imports():
    """Vérifier que tous les imports fonctionnent"""
    try:
        import train
        assert hasattr(train, 'load_data')
        assert hasattr(train, 'clean_data')
        assert hasattr(train, 'create_features')
    except ImportError:
        pytest.fail("Impossible d'importer le module train")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
