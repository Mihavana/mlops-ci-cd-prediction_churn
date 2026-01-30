# Pipeline Jenkins CI/CD - Architecture

## Vue d'ensemble du Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                     JENKINS CI/CD PIPELINE                       │
└──────────────────────────────────────────────────────────────────┘

1️⃣  GIT PUSH (Developer)
    ↓
2️⃣  CHECKOUT (Clone repository)
    ↓
3️⃣  SETUP (Install dependencies)
    ↓
4️⃣  LINTING (Black, Pylint)
    ↓
5️⃣  TESTS (pytest, coverage)
    ├─ test_train.py ✓
    ├─ test_api.py ✓
    └─ Coverage report
    ↓
6️⃣  BUILD DOCKER (Only on main)
    ├─ Build image
    ├─ Tag image
    └─ Push to registry
    ↓
7️⃣  DEPLOY (Only on main)
    ├─ Pull latest image
    ├─ Update docker-compose
    └─ Health check
```

## Stages du Pipeline

### Stage 1: Checkout
- Clone le repository
- Utilise les credentials configurés
- Récupère le code source

### Stage 2: Setup Environment
- Crée un virtualenv Python 3.13
- Installe toutes les dépendances
- Installe les outils dev (black, pylint, pytest-cov)

### Stage 3: Code Quality - Linting
- **Black**: Formate le code Python
- **Pylint**: Analyse statique du code
- Exit code 0 (warnings non bloquants)

### Stage 4: Tests - Unit Tests
- Exécute `test_train.py` (15 tests)
  - Chargement données
  - Nettoyage
  - Features
  - Entraînement
  - Évaluation
- Exécute `test_api.py` (15 tests)
  - Health check
  - Prédictions
  - Batch processing
  - Error handling
- Génère rapport de couverture

**Sortie:**
```
test_train.py::... PASSED
test_api.py::... PASSED
Coverage: 85%+
```

### Stage 5: Build Docker Image
**Condition:** Branche `main` uniquement

Actions:
- Build l'image Docker
- Tag avec numéro de build: `mlops-churn:123`
- Tag avec `latest`

```bash
docker build -f docker/Dockerfile -t docker.io/user/mlops-churn:123 .
docker tag docker.io/user/mlops-churn:123 docker.io/user/mlops-churn:latest
```

### Stage 6: Push to Registry
**Condition:** Branche `main` uniquement

Actions:
- Login au Docker Registry
- Push `mlops-churn:123`
- Push `mlops-churn:latest`
- Logout

### Stage 7: Deploy
**Condition:** Branche `main` uniquement

Actions:
- Arrête containers existants
- Pull latest image
- Lance `docker-compose up -d`
- Vérifie health check `/health`
- Confirme déploiement

## Workflow Complet

```
Developer              Git Repository         Jenkins Server          Docker Registry
    │                       │                      │                        │
    ├─── git push ─────────>│                      │                        │
    │                       │                      │                        │
    │                       ├─── webhook ─────────>│                        │
    │                       │                      │                        │
    │                       │      Checkout        │                        │
    │                       │<─────────────────────┤                        │
    │                       │                      │                        │
    │                       │      Run Tests       │                        │
    │                       │<─────────────────────┤                        │
    │                       │                      │                        │
    │                       │      Build Docker    │                        │
    │                       │<─────────────────────┤                        │
    │                       │                      │                        │
    │                       │                      ├──── docker push ──────>│
    │                       │                      │                        │
    │                       │      Deploy          │<──── docker pull ─────┤
    │                       │<─────────────────────┤                        │
    │                       │                      │                        │
```

## Sécurité

### Credentials
- Docker credentials en Secrets
- Pas d'hardcoding des tokens
- Rotation des tokens recommandée

### Permissions
- Build uniquement sur `main`
- Déploiement automatique + manuel
- Audit trail de tous les builds

## Métriques & Reports

### Coverage Report
```
Statement Coverage: 85%+
Branch Coverage: 80%+
```

### Test Results
```
Passed: 30/30
Failed: 0
Skipped: 0
```

### Build Duration
Temps estimé:
- Tests: ~4 minutes
- Build Docker: ~2 minutes
- Push & Deploy: ~1 minute
- **Total: ~7 minutes**

## Customisation

### Modifier le registry Docker
```groovy
DOCKER_REGISTRY = 'registry.gitlab.com/votre-groupe/projet'
```

### Ajouter des stages
```groovy
stage('Custom Stage') {
    steps {
        sh 'your-command'
    }
}
```

### Changer la branche de déploiement
```groovy
when {
    branch 'develop'  // Au lieu de 'main'
}
```

### Ajouter des notifications
```groovy
slackSend(channel: '#jenkins', message: '...')
emailext(subject: '...', to: 'email@example.com')
```

## Monitoring

Consulter dans Jenkins:
1. **Build Status**: ✅ Success / ❌ Failed / ⚠️ Unstable
2. **Console Output**: Logs détaillés de chaque stage
3. **Coverage Report**: Couverture des tests
4. **Test Results**: Résultats JUnit
5. **Artifacts**: Images Docker, rapports

## Dépannage

| Problème | Solution |
|----------|----------|
| Docker not found | `sudo usermod -aG docker jenkins` |
| Python not found | Installer Python 3.13 sur Jenkins |
| Tests failing | Vérifier logs dans Console Output |
| Push échoue | Vérifier credentials Docker |
| Health check échoue | API pas démarrée, check docker logs |

## Checklist de déploiement

- [ ] Jenkins installé et configuré
- [ ] Plugins nécessaires installés
- [ ] Credentials Docker créés
- [ ] Repository Git connecté
- [ ] Jenkinsfile au root du repo
- [ ] Docker démon accessible
- [ ] Python 3.13 disponible
- [ ] Premier test build réussi
- [ ] Notifications configurées (optionnel)

---

**Prêt pour le CI/CD en production!**
