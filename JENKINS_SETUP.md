# Configuration Jenkins - Guide de Setup

## 📋 Prérequis

1. **Jenkins** installé et en fonctionnement
2. **Docker** installé sur le serveur Jenkins
3. **Python 3.13** disponible sur le serveur Jenkins
4. **Git** configuré
5. Accès à un **Docker Registry** (Docker Hub, AWS ECR, GitLab Registry, etc.)

## 🔧 Configuration Jenkins

### 1. Installer les plugins Jenkins

Aller à **Manage Jenkins** → **Plugin Manager** → Installer:
- `Pipeline`
- `Git`
- `Docker Pipeline`
- `Email Extension Plugin` (optionnel pour notifications)
- `Cobertura Plugin` (pour les rapports de couverture)
- `JUnit Plugin`

### 2. Créer les credentials

Aller à **Manage Jenkins** → **Manage Credentials**

**Créer 2 credentials "Secret text":**
- `docker-username`: Votre nom d'utilisateur Docker Hub
- `docker-password`: Votre token/password Docker Hub

**Pour AWS ECR** (si applicable):
- Créer AWS credentials avec access key + secret key

### 3. Créer un nouveau Job Pipeline

1. Cliquer sur **New Item**
2. Choisir **Pipeline**
3. Nom: `mlops-churn-prediction`
4. Description: `MLOps Pipeline - Churn Prediction`

### 4. Configurer le Pipeline

Dans la section **Pipeline**:

**Option A: Pipeline script from SCM** (Recommandé)
```
Definition: Pipeline script from SCM
SCM: Git
Repository URL: https://github.com/votre-user/mlops-ci-cd-prediction_churn.git
Credentials: [Vos credentials GitHub]
Branch: */main
Script Path: Jenkinsfile
```

**Option B: Pipeline script** (Si pas de SCM)
Copier-coller le contenu du Jenkinsfile directement

### 5. Build Triggers (optionnel)

Pour déclencher automatiquement:
- **GitHub push**: Webhooks GitHub
- **Poll SCM**: `H H * * *` (vérifier toutes les heures)
- **Build periodically**: `H H * * 0` (chaque dimanche)

### 6. Configurer les logs

Dans **General**:
- ✓ Cocher "Discard old builds"
- Garder 10 derniers builds

## 🐳 Configuration Docker Registry

### Pour Docker Hub:
```bash
# Sur la machine Jenkins
docker login -u votre_username -p votre_password
```

Puis utiliser dans Jenkinsfile:
```groovy
DOCKER_REGISTRY = 'docker.io'
```

### Pour AWS ECR:
```groovy
DOCKER_REGISTRY = '123456789.dkr.ecr.eu-west-1.amazonaws.com'
```

### Pour GitLab Registry:
```groovy
DOCKER_REGISTRY = 'registry.gitlab.com/votre-groupe/projet'
```

## 🔐 Paramétrer les variables d'environnement

Modifier les variables en début du Jenkinsfile:

```groovy
environment {
    DOCKER_REGISTRY = 'votre-registry.com'
    DOCKER_USERNAME = credentials('docker-username')
    DOCKER_PASSWORD = credentials('docker-password')
    IMAGE_NAME = 'mlops-churn-prediction'
}
```

## 📊 Monitorer le Pipeline

Après chaque commit:
1. Jenkins clone le repo
2. Exécute les tests
3. Build l'image Docker
4. Push vers le registry
5. Déploie en production

Consulter les logs dans: **Build History** → **Console Output**

## 🔔 Notifications

### Email (optionnel)

Décommenter dans `post` section du Jenkinsfile et configurer:

```groovy
mail to: 'votre-email@example.com',
    subject: "✅ Build réussi: ${env.JOB_NAME}",
    body: "Build #${env.BUILD_NUMBER} réussi!\n${env.BUILD_URL}"
```

### Slack (optionnel)

Installer plugin `Slack Notification` et ajouter:

```groovy
slackSend(
    channel: '#jenkins-builds',
    message: "✅ Build réussi: ${env.JOB_NAME} #${env.BUILD_NUMBER}"
)
```

## 📈 Rapports et Métriques

Le Jenkinsfile génère:
- **Coverage Report**: `htmlcov/index.html`
- **Test Results**: JUnit XML format
- **Build Logs**: Console output

Consulter dans l'interface Jenkins:
- **Coverage Report** (onglet dans le build)
- **Test Results** (onglet dans le build)

## 🚨 Dépannage

**Erreur: "Docker not found"**
```bash
# Sur le serveur Jenkins
sudo usermod -aG docker jenkins
sudo systemctl restart jenkins
```

**Erreur: "Permission denied"**
```bash
# Vérifier les credentials Docker
docker login -u votre_username
```

**Erreur: "Python not found"**
```bash
# Vérifier que Python 3.13 est installé
python3.13 --version
```

## ✅ Vérifier que tout fonctionne

1. Créer une branche test
2. Faire un commit
3. Voir le build déclencher automatiquement
4. Vérifier les logs dans Jenkins
5. Confirmer le déploiement

---

**Configuration complète!** 🎉
