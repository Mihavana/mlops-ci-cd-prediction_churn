# 🚀 Guide de Démarrage - Jenkins CI/CD

## 📋 Résumé

Votre projet MLOps est maintenant prêt pour **Jenkins CI/CD**!

```
✅ Application complète (train + API)
✅ 30 tests unitaires (tous passants)
✅ Dockerfile et docker-compose
✅ Jenkinsfile configuré
✅ Documentation Jenkins
```

## ⚡ 5 Étapes pour démarrer

### 1️⃣ Vérifier Jenkins est installé
```bash
jenkins --version
# ou
java -jar jenkins.war
```

### 2️⃣ Configurer les credentials Docker Hub
Dans Jenkins UI:
- **Manage Jenkins** → **Manage Credentials**
- **New Credentials** → **Secret text**
- ID: `docker-username`
- Secret: `votre_username_docker`

Répéter avec:
- ID: `docker-password`
- Secret: `votre_token_docker`

### 3️⃣ Créer un nouveau Job Pipeline
- **New Item** → **Pipeline**
- Nom: `mlops-churn-prediction`
- **Pipeline** section:
  - Definition: `Pipeline script from SCM`
  - SCM: `Git`
  - Repository: `https://github.com/votre-user/repo.git`
  - Script Path: `Jenkinsfile`

### 4️⃣ Déclencher un build
```bash
git push origin main  # Déclenche automatiquement
```

Ou cliquer **Build Now** dans Jenkins

### 5️⃣ Monitorer le résultat
- Jenkins UI → Console Output
- Vérifier chaque stage: ✅ ou ❌

## 📊 Pipeline Stages

| # | Stage | Durée | Condition |
|---|-------|-------|-----------|
| 1 | 🔄 Checkout | 30s | Toujours |
| 2 | 🔧 Setup | 60s | Toujours |
| 3 | ✨ Linting | 45s | Toujours |
| 4 | 🧪 Tests | 240s | Toujours |
| 5 | 📦 Build Docker | 120s | Branche `main` |
| 6 | 🔐 Push Registry | 60s | Branche `main` |
| 7 | 🚀 Deploy | 30s | Branche `main` |
| **Total** | | **~7 min** | |

## 🎯 Que se passe-t-il?

### Sur une branche de feature (ex: `feature/new-model`):
```
1. Tests exécutés ✓
2. Linting ✓
3. Build Docker → SKIPPED (pas main)
4. Deploy → SKIPPED (pas main)

Résultat: Build SUCCESS
Pas de déploiement
```

### Sur la branche main:
```
1. Tests exécutés ✓
2. Linting ✓
3. Build Docker ✓
4. Push vers Docker Hub ✓
5. Deploy en production ✓

Résultat: Build SUCCESS + Application en prod!
```

## 🔗 Intégration avec GitHub/GitLab

### Pour webhooks automatiques:
1. Jenkins: **Configure job** → **Build Triggers**
2. Cocher **GitHub push trigger** (ou GitLab push)
3. Copier l'URL Jenkins: `http://jenkins.example.com/github-webhook/`
4. GitHub: **Settings** → **Webhooks** → **Add webhook**
5. Payload URL: `http://jenkins.example.com/github-webhook/`
6. Content type: `application/json`

Maintenant chaque `git push` déclenche le pipeline! 🚀

## 📈 Visualiser les résultats

### Coverage Report
- Après le build, aller à: **Coverage Report**
- Voir la couverture des tests par fichier

### Test Results
- Aller à: **Test Results**
- Voir tous les tests: PASS/FAIL
- Détails d'erreur si FAIL

### Console Output
- Cliquer sur un build → **Console Output**
- Voir logs complets de chaque stage

## 🔐 Variables Importantes

À configurer dans le Jenkinsfile:

```groovy
DOCKER_REGISTRY = 'docker.io'  // Votre registry
IMAGE_NAME = 'mlops-churn-prediction'  // Nom du projet
```

## ✅ Checklist d'installation

- [ ] Jenkins en fonctionnement
- [ ] Plugins Pipeline installés
- [ ] Credentials Docker créés
- [ ] Job Pipeline créé
- [ ] Jenkinsfile au root du repo
- [ ] Git repository pointant vers le bon URL
- [ ] Webhooks GitHub/GitLab configurés (optionnel)
- [ ] Premier build lancé avec succès

## 🚨 Problèmes courants & Solutions

### ❌ "Docker command not found"
```bash
sudo usermod -aG docker jenkins
sudo systemctl restart jenkins
```

### ❌ Tests échouent
```bash
# Vérifier localement d'abord:
pytest tests/ -v
```

### ❌ Push Docker échoue
```bash
# Vérifier credentials:
docker login -u votre_user
# Vérifier token Docker Hub valide
```

### ❌ Deploy échoue
```bash
# Vérifier port 8000 disponible:
lsof -i :8000
# Vérifier docker daemon:
docker ps
```

## 📚 Documentation complète

- [`Jenkinsfile`](Jenkinsfile) - Configuration pipeline
- [`JENKINS_SETUP.md`](JENKINS_SETUP.md) - Guide détaillé setup
- [`docs/jenkins-pipeline.md`](docs/jenkins-pipeline.md) - Architecture pipeline

## 🎉 Résumé

Votre projet a maintenant:

✅ **Développement**: Train + API complète
✅ **Testing**: 30 tests automatisés
✅ **CI/CD**: Pipeline Jenkins complet
✅ **Docker**: Image prête pour production
✅ **Documentation**: Guide complet

**Vous êtes prêt pour la production!**

---

**Questions?** Consulter `JENKINS_SETUP.md` pour le guide détaillé.
