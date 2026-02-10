# Text2Loc Visionary API 文档

## 概述

Text2Loc Visionary 提供 REST API 接口，支持自然语言位置查询、增强检索和系统监控。

## 快速开始

### 启动服务器

```bash
# Windows
start.bat

# 或手动启动
python -m api.server --port 8080
```

### 基础查询

```bash
curl -X POST http://localhost:8080/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "我站在红色大楼的北侧约5米处",
    "top_k": 5,
    "enable_enhanced": true
  }'
```

## API 端点

### 健康检查

```http
GET /health
```

**响应**:
```json
{
  "status": "healthy",
  "service": "text2loc-visionary",
  "port": 8080,
  "version": "1.0.0"
}
```

### 处理查询

```http
POST /api/v1/query
Content-Type: application/json
```

**请求体**:
```json
{
  "query": "自然语言描述",
  "top_k": 5,
  "enable_enhanced": true,
  "return_debug_info": false
}
```

**响应**:
```json
{
  "query_id": "query_20260128_120000_1",
  "status": "success",
  "query_analysis": {
    "object_name": "大楼",
    "direction": "北",
    "color": "红色",
    "confidence": 0.85
  },
  "retrieval_results": [...],
  "final_result": {...},
  "processing_time_ms": 45.2
}
```

### 获取系统状态

```http
GET /api/v1/status
```

### 获取配置

```http
GET /api/v1/config
```

### 获取性能指标

```http
GET /api/v1/metrics
```

**响应**:
```json
{
  "status": "healthy",
  "uptime_seconds": 3600.5,
  "timestamp": "2026-01-28T12:00:00",
  "metrics": {
    "modules": {...},
    "summary": {
      "total_operations": 150,
      "success_rate": 0.98,
      "avg_query_time_ms": 45.2,
      "p95_query_time_ms": 120.5
    }
  },
  "errors": {...},
  "feedback": {...}
}
```

### 获取错误信息

```http
GET /api/v1/errors
```

### 提交用户反馈

```http
POST /api/v1/feedback
Content-Type: application/json
```

**请求体**:
```json
{
  "query_id": "query_xxx",
  "rating": 4,
  "comment": "定位结果准确",
  "query_text": "我站在红色大楼的北侧",
  "result_quality": "positive"
}
```

**响应**:
```json
{
  "status": "success",
  "message": "感谢您的反馈"
}
```

## 使用示例

### Python 示例

```python
import requests

API_URL = "http://localhost:8080/api/v1/query"

def query_location(description, top_k=5):
    """查询位置"""
    response = requests.post(API_URL, json={
        "query": description,
        "top_k": top_k,
        "enable_enhanced": True
    })
    return response.json()

# 使用
result = query_location("红色大楼北侧5米")
print(result)
```

### JavaScript 示例

```javascript
async function queryLocation(description) {
  const response = await fetch('http://localhost:8080/api/v1/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: description,
      top_k: 5,
      enable_enhanced: true
    })
  });
  return response.json();
}
```

## 配置

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| TEXT2LOC_PORT | 8080 | API服务端口（使用6000+端口） |
| TEXT2LOC_HOST | 0.0.0.0 | 主机地址 |

### 配置文件

```yaml
enhancement:
  enabled: true
  fallback_threshold: 0.6

modules:
  nlu:
    enabled: true
    provider: "qwen2b"
    cache_ttl: 3600

  retrieval:
    mode: "hybrid"
    weights:
      template: 0.3
      vector: 0.7
```

## 迁移指南

### 从原有系统迁移

原有系统使用固定格式查询：

```json
{
  "object_label": "building",
  "object_color": "red",
  "direction": "north"
}
```

增强版支持自然语言：

```json
{
  "query": "我站在红色大楼的北侧约5米处",
  "enable_enhanced": true
}
```

### 兼容性

- 原有API端点保持不变
- 新增 `enable_enhanced` 参数控制是否使用增强功能
- 默认使用增强模式（enable_enhanced=true）

## 端口说明

服务使用6000+端口避免冲突：

| 服务 | 默认端口 | 说明 |
|------|----------|------|
| API服务 | 8080 | REST API服务器 |
| 前端开发 | 6001 | 开发服务器 |
| 前端生产 | 6002 | 生产服务器 |

## 错误处理

| 状态码 | 说明 |
|--------|------|
| 200 | 成功 |
| 400 | 请求参数错误 |
| 500 | 服务器内部错误 |

## 监控指标

- **processing_time_ms**: 查询处理时间
- **success_rate**: 成功率
- **p95_query_time_ms**: 95%请求响应时间
- **error_rate**: 错误率
