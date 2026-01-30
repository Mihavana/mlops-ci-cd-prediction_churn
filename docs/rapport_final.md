# Rapport final  
## Projet CI/CD – Prédiction du churn client (Telco Customer Churn)

---

## 1. Introduction

Dans un contexte de forte concurrence, la fidélisation des clients constitue
un enjeu stratégique majeur pour les entreprises de télécommunications.
La prédiction du churn client permet d’anticiper les départs et de mettre
en place des actions de rétention ciblées.

L’objectif de ce projet est de concevoir et d’industrialiser un modèle
de Machine Learning capable de prédire le churn client à partir du dataset
**Telco Customer Churn**, en intégrant une chaîne **CI/CD** basée sur les
principes du DevOps et du MLOps.

---

## 2. Présentation du dataset et du problème

Le dataset *Telco Customer Churn* contient des informations sur les clients
d’un opérateur télécom, telles que :
- données démographiques
- types de contrats
- services souscrits
- informations de facturation

La variable cible est binaire :
- `Churn = Yes` : le client quitte l’entreprise
- `Churn = No` : le client reste

Le problème traité est donc un **problème de classification binaire supervisée**.

---

## 3. Approche et méthodologie

### 3.1 Prétraitement des données
Les étapes suivantes ont été réalisées :
- nettoyage des données
- gestion des valeurs manquantes
- encodage des variables catégorielles
- normalisation des variables numériques

Ces étapes sont intégrées dans le pipeline d’entraînement afin d’assurer
la reproductibilité du modèle.

---

### 3.2 Modélisation

Plusieurs algorithmes de classification ont été envisagés afin de répondre
au problème de prédiction du churn. Le modèle retenu est **XGBoost
(eXtreme Gradient Boosting)**, reconnu pour ses performances élevées
sur des données tabulaires et des problèmes de classification binaire.

XGBoost repose sur un ensemble d’arbres de décision entraînés de manière
itérative, chaque nouvel arbre visant à corriger les erreurs des précédents.
Ce choix est particulièrement adapté au problème du churn client,
car il permet de modéliser des relations non linéaires complexes
entre les variables explicatives.

Les hyperparamètres du modèle ont été ajustés afin d’obtenir un compromis
entre performance et généralisation.

---

## 4. Architecture et implémentation

Le projet repose sur une architecture modulaire :
- scripts d’entraînement du modèle (`train.py`)
- logique de prédiction (`predict.py`)
- API exposant le modèle (`app.py`)
- tests unitaires (`tests/`)
- conteneurisation avec Docker
- orchestration via Docker Compose
- automatisation CI/CD avec Jenkins

L’ensemble de l’infrastructure est versionnée afin de garantir
la reproductibilité et la traçabilité des évolutions.

---

## 5. Pipeline CI/CD

Le pipeline CI/CD est défini dans un `Jenkinsfile` et comprend les étapes suivantes :
1. Récupération du code source
2. Installation des dépendances
3. Exécution des tests unitaires
4. Construction de l’image Docker
5. Déploiement automatisé de l’application

Cette approche permet de valider automatiquement chaque modification
du code avant son déploiement.

---

## 6. Tests et validation

Des tests unitaires ont été mis en place afin de :
- vérifier le bon fonctionnement du pipeline d’entraînement
- valider les prédictions du modèle
- tester les endpoints de l’API

L’exécution automatique des tests dans le pipeline CI/CD garantit
la stabilité et la qualité de la solution.

---

## 7. Difficultés rencontrées

Plusieurs difficultés ont été identifiées au cours du projet :
- gestion des dépendances Machine Learning dans un environnement Docker
- tests automatisés d’un modèle de Machine Learning
- intégration de Jenkins avec Docker
- reproductibilité des résultats entre les environnements

Ces difficultés ont nécessité une attention particulière sur
la configuration des environnements et l’organisation du code.

---

## 8. Solutions apportées

Pour répondre à ces problématiques :
- utilisation d’un environnement Dockerisé
- séparation claire des responsabilités dans le code
- automatisation des tests via Pytest
- définition du pipeline CI/CD comme code (Jenkinsfile)

Ces solutions ont permis d’obtenir une chaîne CI/CD stable et reproductible.

---

## 9. Résultats obtenus

Les résultats du projet sont les suivants :
- modèle de prédiction du churn fonctionnel
- API de prédiction opérationnelle
- pipeline CI/CD automatisé
- environnement reproductible
- documentation technique complète

Le projet démontre la faisabilité de l’industrialisation
d’un modèle de Machine Learning dans un contexte réaliste.

---

## 10. Limites du projet

Certaines limites ont été identifiées :
- absence de monitoring des performances en production
- absence de détection de dérive des données
- versioning manuel des modèles
- déploiement limité à un environnement unique

Ces limites sont principalement liées au cadre académique
et aux contraintes de temps.

---

## 11. Perspectives d’amélioration

Dans un contexte professionnel, plusieurs améliorations pourraient être envisagées :
- mise en place d’un outil de versioning des modèles (MLflow)
- ajout de monitoring et de détection de dérive
- déploiement multi-environnements (staging / production)
- ajout de tests de performance et de sécurité

---

## 12. Conclusion

Ce projet a permis de mettre en œuvre une solution complète
d’industrialisation d’un modèle de Machine Learning appliqué
à la prédiction du churn client.

Il illustre l’importance des pratiques DevOps et MLOps pour
garantir la fiabilité, la maintenabilité et la reproductibilité
des modèles de Machine Learning en production, conformément
aux attentes d’un niveau Master.
