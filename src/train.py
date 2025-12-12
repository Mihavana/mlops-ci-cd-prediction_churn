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


df = pd.read_csv('/mnt/44D2A11AD2A1116A/Studies/INSI/M1/CI-CD/Final_Project/mlops-ci-cd-prediction_churn/dataset/Telco-Customer-Churn.csv')

# Nettoyage des données

df = df.drop(columns=['customerID'])
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

df_clean = df.dropna()

# Assignation de X et y

X = df_clean.drop('Churn', axis = 1)
y = df_clean['Churn']

le = LabelEncoder()

y_encoded = le.fit_transform(y)

# Repartir train et test

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Encodage des Données Catégorielles avec OneHotEncoder

CATEGORIAL_FEATURES = [
    'gender', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

categorial_pipeline = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorial_pipeline, CATEGORIAL_FEATURES)
    ],
    remainder='passthrough'
)

# Entrainement avex XGBoost

xgb_model = XGBClassifier(
    objective = 'binary:logistic',
    eval_metric = 'auc',
    scale_pos_weight = 2.7, # nombre non-churns/nombre churns
    random_state = 42
)

full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb_model)
])

# Entrainement
full_pipeline.fit(X_train, y_train)

# Test de prédiction sur les données tests
y_pred_proba = full_pipeline.predict_proba(X_test)[:, 1]

# Evaluation de perf

# auc_score = roc_auc_score(y_test, y_pred_proba)
# print(f"Le score AUC-ROC est : {auc_score:.4f}")

# GridSearch
param_grid = {
    # 1. Ajuster l'Ancienneté (Nombre d'arbres)
    'classifier__n_estimators': [100, 200, 300],
    
    # 2. Ajuster la Complexité (Profondeur max)
    'classifier__max_depth': [3, 5, 7],
    
    # 3. Ajuster le Taux d'Apprentissage
    'classifier__learning_rate': [0.05, 0.1, 0.2],
    
    # 4. Ajuster le Poids pour le Déséquilibre (si vous voulez tester d'autres valeurs)
    # C'est souvent mieux de le calculer et le fixer, mais on peut le faire varier
    'classifier__scale_pos_weight': [2.5, 2.7, 3.0] 
}

grid_search = GridSearchCV(
    estimator=full_pipeline, 
    param_grid=param_grid, 
    scoring='roc_auc', 
    cv=5, 
    verbose=3, 
    n_jobs=-1  # Utiliser tous les coeurs du processeur
)

print("Démarrage de la recherche par grille...")
# La recherche prend en entrée l'ensemble X_train et y_train
grid_search.fit(X_train, y_train)

# Afficher le meilleur score AUC-ROC trouvé
best_auc = grid_search.best_score_
print(f"\nMeilleur AUC-ROC trouvé : {best_auc:.4f}")

# Afficher la combinaison d'hyperparamètres qui a donné ce score
print("Meilleurs hyperparamètres :")
print(grid_search.best_params_)

# Récupérer le modèle final optimisé
best_model = grid_search.best_estimator_

# Utiliser ce meilleur modèle pour la prédiction finale sur X_test
final_y_pred_proba = best_model.predict_proba(X_test)[:, 1]
final_auc_test = roc_auc_score(y_test, final_y_pred_proba)

print(f"AUC-ROC final sur les données de test (avec le meilleur modèle) : {final_auc_test:.4f}")