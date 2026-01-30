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

Le pipeline CI/CD est défini dans le fichier `Jenkinsfile`
et implémente une approche **CI/CD as Code**.

Il est déclenché automatiquement lors d’un push sur la branche `main`
et permet d’automatiser l’ensemble du cycle de vie de l’application
de prédiction du churn.

### Étapes du pipeline

1. **Préparation de l’environnement**
   - Installation de Python et des outils nécessaires
   - Création d’un environnement virtuel (venv)
   - Installation des dépendances du projet

2. **Contrôle qualité du code**
   - Vérification du formatage avec *Black*
   - Analyse statique du code avec *Pylint*
   - Ces étapes permettent d’assurer la maintenabilité du code
     sans bloquer le pipeline en cas d’avertissements mineurs

3. **Tests unitaires**
   - Tests du pipeline d’entraînement du modèle
   - Tests des endpoints de l’API de prédiction
   - Génération de rapports de couverture de code  
   *(Étape optionnelle activable selon le contexte)*

4. **Construction de l’image Docker**
   - Build de l’image applicative à partir du Dockerfile
   - Taggage de l’image avec le numéro de build Jenkins
   - Génération d’un tag `latest`

5. **Analyse de sécurité**
   - Scan de l’image Docker avec *Trivy*
   - Détection des vulnérabilités critiques et élevées
   - Échec du pipeline en cas de vulnérabilités bloquantes

6. **Publication de l’image**
   - Push de l’image Docker vers un registre privé **Harbor**
   - Authentification sécurisée via les credentials Jenkins

7. **Déploiement automatisé**
   - Déploiement de l’application via Docker Compose
   - Arrêt des conteneurs existants
   - Redémarrage avec la nouvelle version de l’image
   - Vérification de la disponibilité de l’API via un healthcheck

8. **Nettoyage**
   - Suppression de l’environnement virtuel
   - Nettoyage des images Docker temporaires
   - Optimisation de l’espace disque sur l’agent Jenkins

---

### Déclenchement conditionnel

Les étapes critiques (build, scan, push, déploiement) sont exécutées
uniquement lors des commits sur la branche `main`,
afin de respecter une stratégie de déploiement contrôlée.

---

### Sécurité

- Secrets gérés via Jenkins Credentials
- Authentification sécurisée au registre Harbor
- Scan de vulnérabilités intégré au pipeline (Trivy)

Ce pipeline illustre une approche **CI/CD orientée MLOps et DevSecOps**,
adaptée à un contexte académique de niveau Master.


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
