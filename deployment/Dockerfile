# Text2Loc Visionary - Docker部署配置
# 支持CPU和GPU两种运行模式

# 阶段1: 基础构建阶段
FROM python:3.10-slim as builder

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 创建工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .
COPY pyproject.toml .
COPY setup.py .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 阶段2: 生产镜像 - CPU版本
FROM python:3.10-slim as cpu-runtime

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    TEXT2LOC_PORT=8080 \
    OLLAMA_URL=http://host.docker.internal:11434 \
    MOCK_MODE=false \
    CUDA_VISIBLE_DEVICES=""

# 安装运行时系统依赖
RUN apt-get update && apt-get install -y \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 创建非root用户
RUN useradd -m -u 1000 -s /bin/bash text2loc && \
    mkdir -p /app && chown -R text2loc:text2loc /app

# 切换用户
USER text2loc

# 设置工作目录
WORKDIR /app

# 从构建阶段复制已安装的Python包
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 复制项目文件
COPY --chown=text2loc:text2loc . .

# 创建必要的目录
RUN mkdir -p logs data checkpoints config

# 暴露端口
EXPOSE 8080

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/api/v1/health || exit 1

# 默认启动命令（API服务）
CMD ["python", "-m", "api.server"]

# 阶段3: 生产镜像 - GPU版本（基于CUDA）
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as gpu-runtime

# 安装Python 3.10
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-distutils \
    python3-pip \
    python3.10-dev \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.10 /usr/local/bin/python

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    TEXT2LOC_PORT=8080 \
    OLLAMA_URL=http://host.docker.internal:11434 \
    MOCK_MODE=false \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# 创建非root用户
RUN useradd -m -u 1000 -s /bin/bash text2loc && \
    mkdir -p /app && chown -R text2loc:text2loc /app

# 切换用户
USER text2loc

# 设置工作目录
WORKDIR /app

# 从构建阶段复制已安装的Python包
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 复制项目文件
COPY --chown=text2loc:text2loc . .

# 安装PyTorch with CUDA 11.8
RUN pip install --no-cache-dir \
    torch==1.11.0+cu113 \
    torchvision==0.12.0+cu113 \
    -f https://download.pytorch.org/whl/torch_stable.html

# 创建必要的目录
RUN mkdir -p logs data checkpoints config

# 暴露端口
EXPOSE 8080

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/api/v1/health || exit 1

# 默认启动命令（API服务）
CMD ["python", "-m", "api.server"]

# 阶段4: 开发镜像
FROM python:3.10-slim as development

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    TEXT2LOC_PORT=8080 \
    OLLAMA_URL=http://localhost:11434 \
    MOCK_MODE=true \
    PYTHONPATH=/app

# 安装开发工具
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    vim \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 复制项目文件
COPY . .

# 安装开发依赖
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy \
    ipython

# 创建必要的目录
RUN mkdir -p logs data checkpoints config

# 暴露端口
EXPOSE 8080 6001

# 开发模式启动命令
CMD ["bash", "-c", "python -m api.server & python -m http.server 6001 --directory frontend"]
