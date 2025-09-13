FROM python:3.11

# 기본 도구 설치
RUN apt-get update && apt-get update --allow-unauthenticated && \
    apt-get install -y --no-install-recommends \
        git \
        curl \
        wget \
        sudo && \
    rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /apps

# Clone the Git repository. Here we dynamically specify the repository name using the variable defined earlier.
RUN git clone -b main https://github.com/yoju-lab/koreatech_BuildingDeployingMCPAIAgent koreatech_BuildingDeployingMCPAIAgent

# Changes the working directory to /app/${REPO_NAME}. This uses the variable to dynamically set the directory path.
WORKDIR /apps/koreatech_BuildingDeployingMCPAIAgent

RUN pip install --no-cache-dir -r ./requirements.txt

RUN rm -rf .git
