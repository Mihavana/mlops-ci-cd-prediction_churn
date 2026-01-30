# Cahier des charges  
## Projet CI/CD – Prédiction du churn client (Telco)

---

## 1. Contexte et problématique

Dans le secteur des télécommunications, la perte de clients (*churn*)
représente un enjeu stratégique majeur. Anticiper le départ des clients
permet aux entreprises de mettre en place des actions de rétention ciblées
et de réduire les pertes financières.

Cependant, les modèles de prédiction de churn sont souvent développés
de manière expérimentale et déployés manuellement, sans automatisation,
ce qui limite leur maintenabilité, leur reproductibilité et leur fiabilité.

Ce projet vise à répondre à cette problématique en mettant en place
une chaîne **CI/CD** permettant l’**industrialisation d’un modèle de Machine Learning**
de prédiction du churn, à partir du dataset **Telco Customer Churn**.

---

## 2. Objectifs du projet

### Objectif principal
Concevoir et déployer une solution automatisée permettant de prédire
le churn client à l’aide d’un modèle de Machine Learning,
intégré dans un pipeline CI/CD.

### Objectifs spécifiques
- Entraîner un modèle de prédiction du churn client
- Automatiser les phases de tests, de build et de déploiement
- Exposer le modèle via une API de prédiction
- Garantir la reproductibilité de l’environnement d’exécution
- Appliquer les principes du DevOps et du MLOps

---

## 3. Périmètre du projet

### Inclus dans le périmètre
- Exploitation du dataset Telco Customer Churn
- Entraînement d’un modèle de classification binaire (churn / non churn)
- Développement d’une API de prédiction
- Mise en place de tests unitaires (modèle et API)
- Conteneurisation avec Docker
- Mise en place d’un pipeline CI/CD avec Jenkins
- Documentation technique complète

### Hors périmètre
- Monitoring en temps réel des performances du modèle
- Détection automatique de dérive des données
- Versioning avancé des modèles (MLflow, etc.)
- Déploiement multi-environnements

---

## 4. Exigences fonctionnelles

- Le pipeline CI/CD doit se déclencher automatiquement à chaque modification du code
- Le modèle doit être entraîné à partir du dataset Telco Customer Churn
- L’API doit permettre de prédire la probabilité de churn d’un client
- Les tests unitaires doivent valider le bon fonctionnement du pipeline ML
- Le déploiement doit être automatisé après validation des tests

---

## 5. Exigences techniques

- Langage : Python
- Librairies ML : XGBoost
- Tests : Pytest
- API : Flask / FastAPI
- Conteneurisation : Docker
- Orchestration locale : Docker Compose
- CI/CD : Jenkins
- Environnement : Linux

---

## 6. Contraintes

- Données fournies limitées au dataset Telco Customer Churn
- Temps de développement limité (cadre académique)
- Ressources matérielles limitées
- Environnement d’exécution contrôlé

---

## 7. Livrables attendus

- Code source du projet
- Modèle de prédiction du churn entraîné
- API de prédiction fonctionnelle
- Pipeline CI/CD (Jenkinsfile)
- Images Docker
- Documentation :
  - architecture.md
  - cahier_des_charges.md
  - rapport_final.md

---

## 8. Critères d’évaluation

- Pertinence du modèle de prédiction du churn
- Qualité du pipeline CI/CD
- Reproductibilité de la solution
- Qualité des tests automatisés
- Justification des choix techniques
- Capacité d’analyse critique
- Qualité de la documentation

---

## 9. Acteurs du projet

- Étudiant(e) : conception, développement, documentation
- Jury académique : évaluation du projet

---

## 10. Conclusion

Ce projet a pour objectif de démontrer la capacité à industrialiser
un modèle de Machine Learning appliqué à un cas métier réel
(prédiction du churn client), en intégrant des pratiques DevOps
et MLOps, conformément aux exigences d’un niveau Master.
