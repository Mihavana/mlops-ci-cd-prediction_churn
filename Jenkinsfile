pipeline {
    agent any
    
    environment {
        // Configuration Docker
        DOCKER_REGISTRY = 'docker.io'
        IMAGE_NAME = 'mlops-churn-prediction'
        IMAGE_TAG = "${BUILD_NUMBER}"

        // Configuration Harbor
        HARBOR_REGISTRY = '192.168.1.201'
        PROJECT_NAME = 'mlops-project'
        REGISTRY_PATH = "${HARBOR_REGISTRY}/${PROJECT_NAME}/${IMAGE_NAME}"        
        // Identifiants Jenkins
        HARBOR_CREDS = credentials('harbor-creds')
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
        
        // stage('Tests - Unit Tests') {
        //     steps {
        //         script {
        //             echo '========== RUNNING UNIT TESTS =========='
        //         }
        //         sh '''
        //             . venv/bin/activate
                    
        //             echo "--- Test d'entraînement ---"
        //             pytest tests/test_train.py -v --tb=short --cov=src/train --cov-report=xml
                    
        //             echo "--- Test API ---"
        //             pytest tests/test_api.py -v --tb=short --cov=src/app --cov-report=xml
                    
        //             echo "--- Rapport de couverture ---"
        //             pytest tests/ --cov=src --cov-report=html --cov-report=term
        //         '''
        //     }
        // }
        
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
                    export DOCKER_BUILDKIT=1
                    
                    docker build -f docker/Dockerfile -t ${IMAGE_NAME}:${IMAGE_TAG} -t ${IMAGE_NAME}:latest .
                '''
            }
        }

        stage('Security Scan - Trivy') {
            steps {
                script {
                    echo '========== SCANNING IMAGE WITH TRIVY (DOCKER) =========='
                    // On lance un conteneur Trivy qui scanne l'image construite à l'étape précédente
                    // On monte le socket docker pour que Trivy puisse accéder aux images locales

                    sh """
                        mkdir -p /tmp/trivy-cache
                        chmod -R 777 /tmp/trivy-cache
                    """

                    sh """
                        docker run --rm \
                            --privileged \
                            -v /var/run/docker.sock:/var/run/docker.sock \
                            -v /tmp/trivy-cache:/tmp/trivy-cache:z \
                            -e TRIVY_CACHE_DIR=/tmp/trivy-cache \
                            aquasec/trivy:latest image \
                            --scanners vuln \
                            --ignore-unfixed \
                            --pkg-types os,library \
                            --severity HIGH,CRITICAL \
                            --exit-code 1 \
                            ${IMAGE_NAME}:${IMAGE_TAG}
                    """
                }
            }
        }

        stage('Push to Harbor') {
            when { branch 'main' }
            steps {
                script {
                    echo '========== PUSHING TO HARBOR =========='

                    sh "docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${REGISTRY_PATH}:${IMAGE_TAG}"
                    sh "docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${REGISTRY_PATH}:latest"
                    
                    sh """
                        echo "${HARBOR_CREDS_PSW}" | docker login ${HARBOR_REGISTRY} -u "${HARBOR_CREDS_USR}" --password-stdin
                        docker push ${REGISTRY_PATH}:${IMAGE_TAG}
                        docker push ${REGISTRY_PATH}:latest
                        docker logout ${HARBOR_REGISTRY}
                    """
                }
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
                    return true
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
                // Supprimer l'image taguée avec le numéro (ex: mlops-churn-prediction:45)
                sh "docker rmi ${IMAGE_NAME}:${IMAGE_TAG} || true"
                
                // Nettoyer les couches de build résiduelles
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
