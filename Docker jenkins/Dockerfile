# Utilise l'image Jenkins LTS recommandée
FROM jenkins/jenkins:lts-jdk17 

# Passe à l'utilisateur root pour l'installation
USER root

# Installe le client Docker CLI (Docker-CE-CLI)
# Ces commandes installent les outils nécessaires pour télécharger et le client Docker
RUN apt-get update && \
    apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release && \
    curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    apt-get update && \
    apt-get install -y docker-ce-cli 

RUN apt-get install -y openssh-client    

RUN mkdir -p /root/.ssh && \
    chmod 700 /root/.ssh && \
    ssh-keyscan -t ed25519 github.com >> /root/.ssh/known_hosts && \
    chmod 600 /root/.ssh/known_hosts && \
    chown -R root:root /root/.ssh

# Reviens à l'utilisateur jenkins
USER jenkins