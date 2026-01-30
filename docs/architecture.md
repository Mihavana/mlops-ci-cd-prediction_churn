# Architecture du projet CI/CD – Pipeline ML

## 1. Vue d’ensemble
Ce projet met en place une chaîne complète CI/CD pour une application
de Machine Learning, incluant l’entraînement du modèle, les tests,
la conteneurisation et le déploiement automatisé via Jenkins.

L’objectif est d’industrialiser le cycle de vie d’un modèle ML,
de la donnée jusqu’à l’API de prédiction.

---

## 2. Arborescence du projet

dataset/  
→ Données utilisées pour l’entraînement du modèle

model/  
→ Modèles entraînés et sauvegardés

src/  
- train.py : entraînement du modèle  
- predict.py : logique de prédiction  
- app.py : API exposant le modèle  

tests/  
- test_train.py : tests du pipeline d’entraînement  
- test_api.py : tests de l’API  

docker/  
- Dockerfile : image de l’application ML  

Docker jenkins/  
- Dockerfile : image Jenkins  
- docker-compose.yml : orchestration Jenkins  

docs/  
- architecture.md  
- cahier_des_charges.md  
- rapport_final.md  

---

## 3. Composants techniques

- Langage : Python
- Modèle de classification : XGBoost
- Framework API : Flask / FastAPI
- Tests : Pytest
- Conteneurisation : Docker
- Orchestration : Docker Compose
- CI/CD : Jenkins
- Gestion des dépendances : requirements.txt

---

## 4. Pipeline CI/CD (Jenkins)

Le pipeline CI/CD est défini dans le fichier `Jenkinsfile`.

Étapes principales :
1. Récupération du code source
2. Installation des dépendances
3. Exécution des tests unitaires
4. Build de l’image Docker
5. Lancement de l’application via Docker Compose

---

## 5. Conteneurisation

- L’application ML est encapsulée dans une image Docker dédiée
- Jenkins est exécuté dans un conteneur séparé
- Docker Compose permet de reproduire l’environnement localement

---

## 6. Sécurité et bonnes pratiques

- Isolation des services via Docker
- Tests automatisés avant le déploiement
- Pipeline versionné (CI/CD as Code)
- Séparation claire entre données, code et infrastructure

---

## 7. Justification des choix techniques

- Jenkins : outil CI/CD robuste et largement utilisé en entreprise
- Docker : reproductibilité et portabilité
- Tests automatisés : garantie de qualité et stabilité du modèle
