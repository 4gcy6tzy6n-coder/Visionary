# Text2Loc Visionary 部署指南

## 项目概述

Text2Loc Visionary 是 Text2Loc 系统的增强版，支持通过自然语言描述在 3D 点云地图中进行精确位置定位。本系统集成了先进的自然语言理解（NLU）和向量检索技术，提供更智能、更灵活的位置查询能力。

**核心特性**:
- 增强的自然语言理解（支持 qwen3-vl:2b 模型）
- 向量检索系统（支持 qwen3-embedding:0.6b 模型）
- 混合检索算法（结合模板匹配和向量相似度）
- 完整的监控和日志系统
- 支持多环境部署（开发/测试/生产）

## 部署选项

### 1. 快速部署（推荐）
使用 Docker Compose 快速启动所有服务，适合开发和测试环境。

### 2. 手动部署
手动安装所有依赖并配置服务，适合定制化部署。

### 3. 云服务部署
在云平台（AWS、Azure、GCP）上部署，适合生产环境。

### 4. 混合部署
结合容器化和传统部署方式。

## 快速开始

### 使用 Docker Compose（最简单的方法）

1. **克隆项目**:
```bash
git clone https://github.com/Yan-Xia/Text2Loc.git
cd Text2Loc/Text2Loc visionary
```

2. **复制环境变量文件**:
```bash
cp deployment/.env.example deployment/.env
# 根据需要编辑 .env 文件
```

3. **启动开发环境**:
```bash
cd deployment
docker-compose --profile development up -d
```

4. **验证部署**:
```bash
# 检查服务状态
docker-compose ps

# 测试API健康状态
curl http://localhost:8080/api/v1/health
```

5. **访问服务**:
- API服务: http://localhost:8080
- 前端界面: http://localhost:80 (如果已部署)
- 监控面板: http://localhost:3000 (如果已部署监控)

### 环境变量配置

创建 `.env` 文件并配置以下关键变量:

```env
# 基本配置
API_PORT=8080
FRONTEND_PORT=80
TEXT2LOC_ENV=production
LOG_LEVEL=INFO

# 安全配置
API_KEY=your_secure_api_key_here
REDIS_PASSWORD=secure_redis_password

# 模型配置
MOCK_MODE=false
OLLAMA_URL=http://ollama:11434

# 性能配置
CACHE_ENABLED=true
BATCH_SIZE=10
MAX_CACHE_SIZE=1000
```

## 详细部署指南

### 1. 系统要求

**最低要求**:
- CPU: 4核
- 内存: 8GB
- 存储: 20GB
- Docker: 20.10+
- Docker Compose: 2.0+

**推荐配置（生产环境）**:
- CPU: 8核或更多
- 内存: 16GB或更多
- 存储: 100GB SSD
- GPU: NVIDIA GPU（可选，用于加速推理）

### 2. Docker 部署

#### 2.1 构建镜像

```bash
# 构建CPU版本
docker build -f deployment/Dockerfile --target cpu-runtime -t text2loc-visionary:cpu .

# 构建GPU版本（需要CUDA）
docker build -f deployment/Dockerfile --target gpu-runtime -t text2loc-visionary:gpu .

# 构建开发版本
docker build -f deployment/Dockerfile --target development -t text2loc-visionary:dev .
```

#### 2.2 运行容器

**单容器运行**:
```bash
# 运行API服务
docker run -d \
  --name text2loc-api \
  -p 8080:8080 \
  -e MOCK_MODE=false \
  -e OLLAMA_URL=http://host.docker.internal:11434 \
  text2loc-visionary:cpu
```

**使用Docker Compose运行完整环境**:
```bash
# 开发环境（最小化服务）
docker-compose --profile development up -d

# 测试环境（完整服务）
docker-compose --profile test up -d

# 生产环境（带监控）
docker-compose --profile production up -d

# 完整环境（所有服务）
docker-compose up -d
```

### 3. 手动部署

#### 3.1 安装依赖

```bash
# 安装系统依赖
sudo apt-get update
sudo apt-get install -y \
  python3.10 \
  python3-pip \
  redis-server \
  nginx \
  curl \
  git

# 安装Python依赖
pip install -r requirements.txt

# 安装Ollama（用于模型服务）
curl -fsSL https://ollama.com/install.sh | sh

# 拉取所需模型
ollama pull qwen3-vl:2b
ollama pull qwen3-embedding:0.6b
```

#### 3.2 配置服务

**配置Redis**:
```bash
sudo nano /etc/redis/redis.conf
# 设置密码和内存限制
requirepass your_redis_password
maxmemory 512mb
maxmemory-policy allkeys-lru
```

**配置Nginx**:
```bash
sudo cp deployment/nginx.conf /etc/nginx/nginx.conf
sudo nginx -t
sudo systemctl restart nginx
```

#### 3.3 启动服务

```bash
# 启动Redis
sudo systemctl start redis
sudo systemctl enable redis

# 启动Ollama
ollama serve &

# 启动Text2Loc API
cd /path/to/text2loc
python -m api.server &

# 启动监控服务（可选）
cd deployment
docker-compose --profile monitoring up -d
```

### 4. 云平台部署

#### 4.1 AWS ECS 部署

**创建任务定义**:
```json
{
  "family": "text2loc-visionary",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "containerDefinitions": [
    {
      "name": "text2loc-api",
      "image": "your-ecr-repo/text2loc-visionary:latest",
      "portMappings": [
        {"containerPort": 8080, "protocol": "tcp"}
      ],
      "environment": [
        {"name": "TEXT2LOC_ENV", "value": "production"},
        {"name": "MOCK_MODE", "value": "false"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/text2loc-visionary",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### 4.2 Kubernetes 部署

**创建Deployment**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: text2loc-visionary
spec:
  replicas: 3
  selector:
    matchLabels:
      app: text2loc-visionary
  template:
    metadata:
      labels:
        app: text2loc-visionary
    spec:
      containers:
      - name: text2loc-api
        image: text2loc-visionary:latest
        ports:
        - containerPort: 8080
        env:
        - name: TEXT2LOC_ENV
          value: "production"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /api/v1/health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
```

## 配置说明

### 配置文件结构

```
deployment/
├── Dockerfile              # 多阶段构建配置
├── docker-compose.yml      # 多服务编排
├── nginx.conf             # 反向代理配置
├── prometheus.yml         # 监控配置
├── .env.example           # 环境变量示例
└── README.md              # 本文件
```

### 关键配置项

#### API服务配置
```yaml
# config.yaml 或环境变量
api:
  port: 8080
  host: 0.0.0.0
  workers: 4
  timeout: 30

nlu:
  model: "qwen3-vl:2b"
  ollama_url: "http://ollama:11434"
  confidence_threshold: 0.7

vector_db:
  model: "qwen3-embedding:0.6b"
  index_type: "FlatL2"
  vector_weight: 0.6

cache:
  enabled: true
  redis_url: "redis://redis:6379"
  ttl: 3600
```

#### 性能调优
```bash
# 调整API工作进程数
export TEXT2LOC_WORKERS=4

# 调整批处理大小
export BATCH_SIZE=16

# 启用响应缓存
export CACHE_ENABLED=true
export CACHE_TTL=300

# 调整连接池大小
export DB_POOL_SIZE=20
export DB_MAX_OVERFLOW=40
```

## 监控和运维

### 1. 健康检查

**API健康端点**:
```bash
# 基础健康检查
curl http://localhost:8080/api/v1/health

# 详细系统状态
curl http://localhost:8080/api/v1/status

# 性能指标
curl http://localhost:8080/api/v1/metrics
```

### 2. 日志管理

**查看日志**:
```bash
# Docker容器日志
docker logs text2loc-api

# 跟随日志输出
docker logs -f text2loc-api

# 查看特定服务的日志
docker-compose logs text2loc-api

# 查看错误日志
docker-compose logs text2loc-api | grep ERROR
```

**日志级别设置**:
```bash
# 开发环境 - 详细日志
export LOG_LEVEL=DEBUG

# 测试环境 - 信息日志
export LOG_LEVEL=INFO

# 生产环境 - 警告和错误
export LOG_LEVEL=WARNING
```

### 3. 性能监控

**启用Prometheus监控**:
```bash
cd deployment
docker-compose --profile monitoring up -d
```

**访问监控面板**:
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090

**关键监控指标**:
- API响应时间（P95, P99）
- 查询成功率
- 系统资源使用（CPU、内存、磁盘）
- 缓存命中率
- 模型推理时间

### 4. 备份和恢复

**数据备份**:
```bash
# 备份Redis数据
docker exec text2loc-redis redis-cli --rdb /data/dump.rdb
docker cp text2loc-redis:/data/dump.rdb ./backup/redis-backup.rdb

# 备份配置文件
cp -r config backup/config-$(date +%Y%m%d)

# 备份模型检查点
cp -r checkpoints backup/checkpoints-$(date +%Y%m%d)
```

**数据恢复**:
```bash
# 恢复Redis数据
docker cp backup/redis-backup.rdb text2loc-redis:/data/
docker exec text2loc-redis redis-cli --rdb /data/redis-backup.rdb

# 恢复配置文件
cp -r backup/config-20240101/* config/
```

## 故障排除

### 常见问题

#### 1. 服务无法启动

**问题**: Docker容器无法启动或立即退出

**解决方案**:
```bash
# 检查容器日志
docker logs text2loc-api

# 检查端口占用
netstat -tulpn | grep :8080

# 检查环境变量配置
docker-compose config

# 重新构建镜像
docker-compose build --no-cache
```

#### 2. API响应慢

**问题**: 查询响应时间过长

**解决方案**:
```bash
# 检查系统资源
docker stats

# 增加缓存大小
export MAX_CACHE_SIZE=5000

# 调整批处理大小
export BATCH_SIZE=20

# 检查网络延迟
docker exec text2loc-api ping ollama

# 启用GPU加速（如果可用）
export DOCKER_TARGET=gpu-runtime
```

#### 3. 模型服务连接失败

**问题**: 无法连接到Ollama服务

**解决方案**:
```bash
# 检查Ollama服务状态
curl http://ollama:11434/api/tags

# 确保模型已下载
docker exec text2loc-ollama ollama list

# 重新拉取模型
docker exec text2loc-ollama ollama pull qwen3-vl:2b

# 检查网络连接
docker network inspect text2loc-network
```

#### 4. 内存不足

**问题**: 容器因内存不足被杀死

**解决方案**:
```bash
# 增加内存限制
# 在docker-compose.yml中增加：
services:
  text2loc-api:
    deploy:
      resources:
        limits:
          memory: 4G

# 减少缓存大小
export MAX_CACHE_SIZE=1000

# 减少工作进程数
export TEXT2LOC_WORKERS=2
```

### 调试工具

**交互式调试**:
```bash
# 进入运行中的容器
docker exec -it text2loc-api bash

# 启动Python交互式环境
docker exec -it text2loc-api python -c "
from api.text2loc_api import Text2LocAPI
api = Text2LocAPI()
print(api.get_capabilities())
"

# 测试特定功能
docker exec -it text2loc-api python -c "
from enhancements.nlu.engine import NLUEngine
nlu = NLUEngine()
result = nlu.parse_direction('红色大楼的北侧')
print(result)
"
```

**性能分析**:
```bash
# 使用cProfile进行性能分析
docker exec text2loc-api python -m cProfile -o profile.stats api/server.py

# 分析性能数据
docker exec text2loc-api python -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('time').print_stats(10)
"
```

## 安全最佳实践

### 1. 网络安全

```bash
# 使用防火墙限制访问
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 8080/tcp from 192.168.1.0/24  # 仅限内部网络

# 禁用不必要的端口
sudo ufw deny 22/tcp  # 如果不需要SSH
```

### 2. API安全

```yaml
# 启用API密钥验证
security:
  api_key_enabled: true
  allowed_origins:
    - https://yourdomain.com
    - https://app.yourdomain.com
  
  rate_limiting:
    enabled: true
    requests_per_minute: 60
    burst_limit: 100
```

### 3. 数据安全

```bash
# 加密敏感数据
export REDIS_PASSWORD=$(openssl rand -hex 32)
export API_KEY=$(openssl rand -hex 32)

# 定期轮换密钥
# 在crontab中添加：
0 0 1 * * /path/to/rotate-keys.sh
```

### 4. 容器安全

```dockerfile
# 使用非root用户运行容器
USER text2loc

# 只读文件系统（除了必要目录）
VOLUME ["/app/logs", "/app/data"]
```

## 扩展和定制

### 1. 添加新模型

**添加新的NLU模型**:
```python
# 在 enhancements/nlu/engine.py 中添加
class NLUConfig:
    # 添加新模型选项
    MODEL_CHOICES = {
        "qwen3-vl:2b": "Qwen3-VL 2B",
        "qwen2.5:0.5b": "Qwen2.5 0.5B",
        "your-new-model": "Your New Model"
    }
```

**更新docker-compose.yml**:
```yaml
services:
  ollama:
    environment:
      - OLLAMA_MODELS=qwen3-vl:2b,qwen3-embedding:0.6b,your-new-model
```

### 2. 自定义检索算法

```python
# 创建自定义检索器
from enhancements.vector_db.base import BaseRetriever

class CustomRetriever(BaseRetriever):
    def retrieve(self, query, candidates, top_k=10):
        # 实现自定义检索逻辑
        pass
```

### 3. 添加新功能模块

1. 在新目录中创建模块：`enhancements/new_feature/`
2. 实现必要的接口
3. 在配置中启用模块
4. 在适配器中集成

## 支持与贡献

### 获取帮助

- **问题反馈**: 在GitHub Issues中报告问题
- **文档**: 查看项目Wiki获取详细文档
- **社区**: 加入Discord/Slack社区

### 贡献代码

1. Fork项目仓库
2. 创建特性分支
3. 提交更改
4. 创建Pull Request

### 测试贡献

```bash
# 运行测试套件
cd /path/to/text2loc
python -m pytest tests/ -v

# 运行性能测试
python performance_test.py --real

# 运行集成测试
python phase4_comprehensive_test.py
```

## 版本历史

### v1.0.0 (2026-01-29)
- 初始发布版本
- 支持基本的位置查询功能
- 集成qwen3-vl和qwen3-embedding模型
- 提供Docker容器化部署
- 包含完整的监控系统

### v1.1.0 (计划中)
- 支持更多语言模型
- 改进的缓存机制
- 增强的错误处理
- 更多的部署选项

## 许可证

本项目基于MIT许可证开源。详细信息请查看LICENSE文件。

## 免责声明

本软件按"原样"提供，不提供任何明示或暗示的担保，包括但不限于适销性、特定用途适用性和非侵权性的担保。在任何情况下，作者或版权持有人均不对因软件或软件的使用或其他交易而产生、引起或与之相关的任何索赔、损害赔偿或其他责任承担责任。

---

**最后更新**: 2026-01-29  
**维护者**: Text2Loc Visionary Team  
**联系方式**: text2loc-support@example.com