pipeline {
    agent {
        docker {
            image 'python:3.11-slim'
            args '-v /var/run/docker.sock:/var/run/docker.sock'
        }
    }
    
    environment {
        // Configuration Docker
        DOCKER_REGISTRY = 'docker.io'  // Remplacer par votre registry (DockerHub, ECR, etc.)
        DOCKER_USERNAME = credentials('docker-username')  // Créer dans Jenkins
        DOCKER_PASSWORD = credentials('docker-password')
        IMAGE_NAME = 'mlops-churn-prediction'
        IMAGE_TAG = "${BUILD_NUMBER}"
        FULL_IMAGE = "${DOCKER_REGISTRY}/${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}"
        
        // Python
        PYTHON_VERSION = '3'
    }
    
    options {
        // Garder les 10 derniers builds
        buildDiscarder(logRotator(numToKeepStr: '10'))
        // Timeout de 30 minutes
        timeout(time: 30, unit: 'MINUTES')
        // Timestamps dans les logs
        timestamps()
    }
    
    stages {
        stage('🔄 Checkout') {
            steps {
                script {
                    echo '========== CHECKOUT CODE =========='
                }
                checkout scm
            }
        }
        
        stage('🔧 Setup Environment') {
            steps {
                script {
                    echo '========== SETUP PYTHON ENVIRONMENT =========='
                    sh '''
                        python${PYTHON_VERSION} -m venv venv
                        . venv/bin/activate
                        pip install --upgrade pip setuptools wheel
                        pip install -r requirements.txt
                        pip install black pylint pytest-cov
                    '''
                }
            }
        }
        
        stage('✨ Code Quality - Linting') {
            steps {
                script {
                    echo '========== CODE LINTING (BLACK & PYLINT) =========='
                }
                sh '''
                    . venv/bin/activate
                    
                    echo "--- Formatting avec Black ---"
                    black --check src/ tests/ || true
                    
                    echo "--- Linting avec Pylint ---"
                    pylint src/ --exit-zero || true
                '''
            }
        }
        
        stage('🧪 Tests - Unit Tests') {
            steps {
                script {
                    echo '========== RUNNING UNIT TESTS =========='
                }
                sh '''
                    . venv/bin/activate
                    
                    echo "--- Test d'entraînement ---"
                    pytest tests/test_train.py -v --tb=short --cov=src/train --cov-report=xml
                    
                    echo "--- Test API ---"
                    pytest tests/test_api.py -v --tb=short --cov=src/app --cov-report=xml
                    
                    echo "--- Rapport de couverture ---"
                    pytest tests/ --cov=src --cov-report=html --cov-report=term
                '''
            }
        }
        
        stage('📦 Build Docker Image') {
            when {
                branch 'main'  // Uniquement sur la branche main
            }
            steps {
                script {
                    echo '========== BUILDING DOCKER IMAGE =========='
                }
                sh '''
                    docker build -f docker/Dockerfile -t ${FULL_IMAGE} .
                    docker tag ${FULL_IMAGE} ${DOCKER_REGISTRY}/${DOCKER_USERNAME}/${IMAGE_NAME}:latest
                '''
            }
        }
        
        stage('🔐 Push to Registry') {
            when {
                branch 'main'
            }
            steps {
                script {
                    echo '========== PUSHING TO DOCKER REGISTRY =========='
                }
                sh '''
                    echo "${DOCKER_PASSWORD}" | docker login -u ${DOCKER_USERNAME} --password-stdin ${DOCKER_REGISTRY}
                    docker push ${FULL_IMAGE}
                    docker push ${DOCKER_REGISTRY}/${DOCKER_USERNAME}/${IMAGE_NAME}:latest
                    docker logout ${DOCKER_REGISTRY}
                '''
            }
        }
        
        stage('🚀 Deploy') {
            when {
                branch 'main'
            }
            steps {
                script {
                    echo '========== DEPLOYMENT =========='
                }
                sh '''
                    # Option 1: Déploiement local avec Docker Compose
                    docker-compose down || true
                    docker-compose pull
                    docker-compose up -d
                    
                    # Option 2: Attendre que l'API soit prête
                    sleep 5
                    curl -f http://localhost:8000/health || exit 1
                    
                    echo "✓ API déployée avec succès!"
                '''
            }
        }
        stage('🧹 Cleanup') {
            when {
                expression {
                    return true  // Toujours exécuter
                }
            }
            steps {
                script {
                    echo '========== CLEANUP =========='
                }
                sh '''
                    rm -rf venv || true
                    docker system prune -f || true
                    echo "✓ Cleanup complété"
                '''
            }
        }
    }
    
    post {
        always {
            script {
                echo '========== BUILD FINISHED =========='
                echo "Build status: ${currentBuild.result}"
            }
        }
        
        success {
            script {
                echo '✅ PIPELINE RÉUSSI'
            }
        }
        
        failure {
            script {
                echo '❌ PIPELINE ÉCHOUÉ'
            }
        }
        
        unstable {
            script {
                echo '⚠️ PIPELINE INSTABLE'
            }
        }
    }
}
