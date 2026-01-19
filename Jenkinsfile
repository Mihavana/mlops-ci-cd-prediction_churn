pipeline {
    agent any
    
    environment {
        // Configuration Docker
        DOCKER_REGISTRY = 'docker.io'
        IMAGE_NAME = 'mlops-churn-prediction'
        IMAGE_TAG = "${BUILD_NUMBER}"
    }
    
    options {
        // Garder les 10 derniers builds
        buildDiscarder(logRotator(numToKeepStr: '10'))
        // Timeout de 90 minutes
        timeout(time: 90, unit: 'MINUTES')
        // Timestamps dans les logs
        timestamps()
    }
    
    stages {

        stage('Setup Environment') {
            steps {
                script {
                    echo '========== SETUP PYTHON ENVIRONMENT =========='
                    sh '''
                        apt-get update || true
                        apt-get install -y python3 python3-pip python3-venv || true
                        python3 -m venv venv
                        . venv/bin/activate
                        pip install --upgrade pip setuptools wheel
                        pip install -r requirements.txt
                        pip install black pylint pytest-cov
                    '''
                }
            }
        }
        
        stage('Code Quality - Linting') {
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
        
        stage('Tests - Unit Tests') {
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
        
        stage('Build Docker Image') {
            when {
                expression {
                    env.GIT_BRANCH?.endsWith('main')
                }
            }
            steps {
                script {
                    echo '========== BUILDING DOCKER IMAGE =========='
                }
                sh '''
                    docker build -f docker/Dockerfile -t ${IMAGE_NAME}:${IMAGE_TAG} .
                    docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${IMAGE_NAME}:latest
                '''
            }
        }

        stage('Deploy') {
            when {
                expression {
                    env.GIT_BRANCH?.endsWith('main')
                }
            }
            steps {
                script {
                    echo '========== DEPLOYMENT =========='
                }
                sh '''
                    docker compose down || true
                    docker compose up -d

                    echo "Waiting for API to be ready..."
                    # On exécute le health check depuis le conteneur directement
                    for i in 1 2 3 4 5 6 7 8 9 10; do
                        if docker exec churn-prediction-api python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" > /dev/null 2>&1; then
                            echo "✓ API is ready!"
                            success=1
                            break
                        else
                            echo "Retry $i/10... (API not ready yet)"
                            sleep 5
                        fi
                    done
                '''
            }
        }
        stage('Cleanup') {
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

                // --- NETTOYAGE ---
                echo "Suppression de l'image de build pour économiser l'espace ..."
                // On supprime l'image taguée avec le numéro (ex: mlops-churn-prediction:45)
                sh "docker rmi ${IMAGE_NAME}:${IMAGE_TAG} || true"
                
                // On nettoie les couches de build résiduelles
                sh "docker image prune -f"
                // ------------------------------
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
