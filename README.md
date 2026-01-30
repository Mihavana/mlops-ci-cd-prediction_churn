# 📊 Telco Customer Churn Prediction – CI/CD & MLOps Project

Ce projet a pour objectif de concevoir et d’industrialiser un modèle de
**prédiction du churn client** à partir du dataset **Telco Customer Churn**,
en appliquant les principes du **DevOps** et du **MLOps**.

La solution intègre :
- un modèle de Machine Learning basé sur **XGBoost**
- une API de prédiction
- une chaîne **CI/CD avec Jenkins**
- une infrastructure **Dockerisée**
- un registre d’images **Harbor local**

---

## 🎯 Objectifs du projet

- Prédire le churn client (classification binaire)
- Industrialiser le cycle de vie d’un modèle ML
- Automatiser les tests, le build et le déploiement
- Garantir la reproductibilité de l’environnement
- Illustrer un cas concret MLOps niveau Master

---

## 🧠 Modèle de Machine Learning

- Type : Classification binaire (churn / non churn)
- Algorithme : **XGBoost**
- Librairies : scikit-learn, xgboost
- Métriques :
  - Accuracy
  - Precision
  - Recall (prioritaire pour la classe churn)
  - F1-score

---

## 🏗️ Architecture du projet

```
    dataset/
    docker/
    │ └── Dockerfile
    Docker jenkins/
    │ ├── Dockerfile
    │ └── docker-compose.yml
    docs/
    │ ├── architecture.md
    │ ├── cahier_des_charges.md
    │ └── rapport_final.md
    model/
    src/
    │ ├── app.py
    │ ├── predict.py
    │ └── train.py
    tests/
    │ ├── test_api.py
    │ └── test_train.py
    docker-compose.yml
    Jenkinsfile
    requirements.txt
    README.md
```


---

## ⚙️ Prérequis

Avant de démarrer le projet, les outils suivants doivent être installés
et fonctionnels localement :

- **Docker**
- **Docker Compose**
- **Jenkins (Docker)**
- **Harbor (registre Docker local)**
- **Git**
- Environnement Linux recommandé

⚠️ Le projet suppose que **Harbor est accessible localement**
pour le stockage des images Docker générées par le pipeline CI/CD.

---

## 🚀 Démarrage du projet

### 1️⃣ Cloner le dépôt
```bash
git clone https://github.com/Mihavana/mlops-ci-cd-prediction_churn.git
cd mlops-ci-cd-prediction_churn
