"""
Qwen3-Embedding客户端 - Text2Loc增强版

基于Ollama集成的qwen3-embedding:0.6b模型，提供文本向量化和相似度计算
"""

import json
import time
import logging
import hashlib
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import requests
import numpy as np
from numpy.linalg import norm

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """嵌入结果"""
    text: str  # 原始文本
    embedding: np.ndarray  # 嵌入向量
    dimension: int  # 向量维度
    model: str  # 使用的模型
    cached: bool = False  # 是否来自缓存
    error: Optional[str] = None  # 错误信息


class EmbeddingModel(Enum):
    """支持的嵌入模型"""
    QWEN3_EMBEDDING_06B = "qwen3-embedding:0.6b"  # 主要使用的嵌入模型
    QWEN2_5_05B = "qwen2.5:0.5b"  # 备用模型


class EmbeddingClient:
    """Qwen3-Embedding客户端"""

    def __init__(self,
                 ollama_url: str = "http://localhost:11434",
                 model_name: Union[str, EmbeddingModel] = EmbeddingModel.QWEN3_EMBEDDING_06B,
                 timeout: int = 30,
                 mock_mode: bool = False,
                 cache_enabled: bool = True,
                 max_cache_size: int = 1000):
        """
        初始化嵌入客户端

        Args:
            ollama_url: ollama服务地址
            model_name: 模型名称或EmbeddingModel枚举
            timeout: API调用超时时间（秒）
            mock_mode: 模拟模式，用于测试（不实际调用API）
            cache_enabled: 是否启用缓存
            max_cache_size: 最大缓存条目数
        """
        self.ollama_url = ollama_url
        self.model_name = model_name.value if isinstance(model_name, EmbeddingModel) else model_name
        self.timeout = timeout
        self.mock_mode = mock_mode
        self.cache_enabled = cache_enabled
        self.max_cache_size = max_cache_size
        self.session = requests.Session() if not mock_mode else None

        # 缓存系统
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # 性能统计
        self.total_calls = 0
        self.total_time = 0.0

        # 验证连接（如果不是模拟模式）
        if not mock_mode:
            self._test_connection()

        logger.info(f"EmbeddingClient初始化完成: model={self.model_name}, mock={mock_mode}")

    def _test_connection(self):
        """测试ollama连接"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info(f"成功连接到ollama服务: {self.ollama_url}")
                # 检查模型是否可用
                models = response.json().get("models", [])
                model_available = any(model["name"] == self.model_name for model in models)
                if not model_available:
                    logger.warning(f"模型 {self.model_name} 未找到，使用第一个可用模型")
                    if models:
                        self.model_name = models[0]["name"]
                        logger.info(f"使用模型: {self.model_name}")
            else:
                logger.warning(f"ollama服务连接异常: {response.status_code}")
        except Exception as e:
            logger.error(f"无法连接到ollama服务: {e}")
            raise ConnectionError(f"无法连接到ollama服务: {e}")

    def _get_cache_key(self, text: str) -> str:
        """生成缓存键"""
        return hashlib.md5(f"{self.model_name}:{text}".encode('utf-8')).hexdigest()

    def _get_from_cache(self, text: str) -> Optional[np.ndarray]:
        """从缓存获取嵌入"""
        if not self.cache_enabled:
            return None

        cache_key = self._get_cache_key(text)
        if cache_key in self.embedding_cache:
            self.cache_hits += 1
            return self.embedding_cache[cache_key]

        self.cache_misses += 1
        return None

    def _add_to_cache(self, text: str, embedding: np.ndarray):
        """添加嵌入到缓存"""
        if not self.cache_enabled:
            return

        # 如果缓存过大，清理最旧的一半
        if len(self.embedding_cache) >= self.max_cache_size:
            keys_to_remove = list(self.embedding_cache.keys())[:self.max_cache_size // 2]
            for key in keys_to_remove:
                del self.embedding_cache[key]
            logger.info(f"清理缓存，移除{len(keys_to_remove)}个条目")

        cache_key = self._get_cache_key(text)
        self.embedding_cache[cache_key] = embedding

    def _call_ollama_embed_api(self, text: str) -> np.ndarray:
        """
        调用ollama嵌入API

        Args:
            text: 要嵌入的文本

        Returns:
            嵌入向量
        """
        if self.mock_mode:
            # 模拟模式返回随机向量（固定维度）
            return self._mock_embedding(text)

        try:
            payload = {
                "model": self.model_name,
                "prompt": text
            }

            response = self.session.post(
                f"{self.ollama_url}/api/embeddings",
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()
                embedding = np.array(result.get("embedding", []), dtype=np.float32)

                if len(embedding) == 0:
                    raise ValueError("返回的嵌入向量为空")

                # 归一化嵌入向量
                embedding = embedding / norm(embedding)

                return embedding
            else:
                logger.error(f"嵌入API调用失败: {response.status_code}, {response.text}")
                raise Exception(f"嵌入API调用失败: {response.status_code}")

        except requests.exceptions.Timeout:
            logger.error(f"嵌入API调用超时: {self.timeout}秒")
            raise TimeoutError(f"嵌入API调用超时: {self.timeout}秒")
        except Exception as e:
            logger.error(f"嵌入API调用异常: {e}")
            raise

    def _mock_embedding(self, text: str, dimension: int = 1024) -> np.ndarray:
        """模拟嵌入生成（用于测试）"""
        # 使用文本哈希作为随机种子，确保相同文本产生相同向量
        seed = int(hashlib.md5(text.encode('utf-8')).hexdigest()[:8], 16)
        np.random.seed(seed)

        # 生成随机向量并归一化
        embedding = np.random.randn(dimension).astype(np.float32)
        embedding = embedding / norm(embedding)

        return embedding

    def embed_text(self, text: str, use_cache: bool = True) -> EmbeddingResult:
        """
        为单个文本生成嵌入

        Args:
            text: 要嵌入的文本
            use_cache: 是否使用缓存

        Returns:
            EmbeddingResult对象
        """
        logger.debug(f"生成文本嵌入: {text[:50]}...")

        start_time = time.time()

        try:
            # 1. 检查缓存
            if use_cache:
                cached_embedding = self._get_from_cache(text)
                if cached_embedding is not None:
                    logger.debug(f"缓存命中: {text[:30]}...")
                    return EmbeddingResult(
                        text=text,
                        embedding=cached_embedding,
                        dimension=len(cached_embedding),
                        model=self.model_name,
                        cached=True
                    )

            # 2. 调用API生成嵌入
            embedding = self._call_ollama_embed_api(text)

            # 3. 添加到缓存
            if use_cache:
                self._add_to_cache(text, embedding)

            elapsed_time = time.time() - start_time
            logger.debug(f"嵌入生成完成: {len(embedding)}维, 耗时{elapsed_time:.3f}秒")

            # 更新统计
            self.total_calls += 1
            self.total_time += elapsed_time

            return EmbeddingResult(
                text=text,
                embedding=embedding,
                dimension=len(embedding),
                model=self.model_name,
                cached=False
            )

        except Exception as e:
            logger.error(f"嵌入生成失败: {e}")
            elapsed_time = time.time() - start_time
            self.total_calls += 1
            self.total_time += elapsed_time

            return EmbeddingResult(
                text=text,
                embedding=np.array([]),
                dimension=0,
                model=self.model_name,
                cached=False,
                error=str(e)
            )

    def batch_embed_texts(self, texts: List[str], batch_size: int = 10, use_cache: bool = True) -> List[EmbeddingResult]:
        """
        批量生成文本嵌入

        Args:
            texts: 文本列表
            batch_size: 批次大小（注意：ollama API逐个处理，这里用于控制并发）
            use_cache: 是否使用缓存

        Returns:
            EmbeddingResult列表
        """
        logger.info(f"批量生成嵌入: {len(texts)}个文本")

        results = []

        for i, text in enumerate(texts):
            # 添加小延迟避免API过载
            if i > 0 and not self.mock_mode:
                time.sleep(0.05)

            result = self.embed_text(text, use_cache=use_cache)
            results.append(result)

            # 进度日志
            if (i + 1) % 10 == 0:
                logger.info(f"批量嵌入进度: {i + 1}/{len(texts)}")

        return results

    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        计算两个嵌入向量的余弦相似度

        Args:
            embedding1: 第一个嵌入向量
            embedding2: 第二个嵌入向量

        Returns:
            余弦相似度 (-1到1之间)
        """
        if len(embedding1) == 0 or len(embedding2) == 0:
            return 0.0

        # 确保向量是numpy数组
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)

        # 归一化
        emb1_norm = emb1 / norm(emb1)
        emb2_norm = emb2 / norm(emb2)

        # 计算余弦相似度
        similarity = np.dot(emb1_norm, emb2_norm)

        # 确保在[-1, 1]范围内
        return float(np.clip(similarity, -1.0, 1.0))

    def find_most_similar(self, query_embedding: np.ndarray,
                         candidate_embeddings: List[np.ndarray],
                         top_k: int = 5) -> List[Tuple[int, float]]:
        """
        查找最相似的嵌入向量

        Args:
            query_embedding: 查询向量
            candidate_embeddings: 候选向量列表
            top_k: 返回前k个结果

        Returns:
            [(索引, 相似度), ...] 列表
        """
        similarities = []

        for i, candidate in enumerate(candidate_embeddings):
            sim = self.cosine_similarity(query_embedding, candidate)
            similarities.append((i, sim))

        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 返回前k个
        return similarities[:top_k]

    def get_cache_stats(self) -> Dict[str, any]:
        """获取缓存统计信息"""
        return {
            "cache_size": len(self.embedding_cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0.0
        }

    def get_stats(self) -> Dict[str, any]:
        """获取性能统计信息"""
        avg_time = self.total_time / self.total_calls if self.total_calls > 0 else 0

        return {
            "total_calls": self.total_calls,
            "total_time": self.total_time,
            "average_time": avg_time,
            "cache_stats": self.get_cache_stats(),
            "model": self.model_name
        }

    def clear_cache(self):
        """清理缓存"""
        self.embedding_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("嵌入缓存已清理")
