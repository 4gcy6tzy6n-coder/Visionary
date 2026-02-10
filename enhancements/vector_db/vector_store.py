"""
向量存储管理器 - Text2Loc增强版

统一管理嵌入生成、索引构建和检索功能
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from .embedding_client import EmbeddingClient
from .faiss_index import FaissIndex
from .hybrid_retriever import HybridRetriever, RetrievalResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VectorStoreConfig:
    """向量存储配置"""
    embedding_model: str = "qwen3-embedding:0.6b"
    index_type: str = "FlatL2"
    vector_weight: float = 0.6
    template_weight: float = 0.4
    cache_enabled: bool = True
    mock_mode: bool = False


class VectorStore:
    """向量存储管理器"""

    def __init__(self, config: Optional[VectorStoreConfig] = None):
        """
        初始化向量存储

        Args:
            config: 配置参数
        """
        self.config = config or VectorStoreConfig()

        # 初始化组件
        self.embedding_client = EmbeddingClient(
            model_name=self.config.embedding_model,
            mock_mode=self.config.mock_mode,
            cache_enabled=self.config.cache_enabled
        )

        self.index = None
        self.retriever = None

        # 数据存储
        self.items = []  # 原始数据
        self.metadata_list = []  # 元数据列表

        logger.info(f"向量存储初始化完成: model={self.config.embedding_model}")

    def _ensure_index(self, dimension: int):
        """确保索引已初始化"""
        if self.index is None:
            self.index = FaissIndex(
                dimension=dimension,
                index_type=self.config.index_type,
                mock_mode=self.config.mock_mode
            )

        if self.retriever is None:
            self.retriever = HybridRetriever(
                vector_weight=self.config.vector_weight,
                template_weight=self.config.template_weight
            )

    def add_item(self, text: str, metadata: Dict[str, Any]):
        """
        添加单个项目

        Args:
            text: 文本描述
            metadata: 元数据
        """
        # 生成嵌入
        embedding_result = self.embedding_client.embed_text(text)

        if embedding_result.error:
            logger.error(f"生成嵌入失败: {embedding_result.error}")
            return False

        # 确保索引已初始化
        self._ensure_index(embedding_result.dimension)

        # 添加到索引
        vector = embedding_result.embedding.reshape(1, -1)
        metadata_with_text = metadata.copy()
        metadata_with_text["text"] = text

        self.index.add(vector, [metadata_with_text])

        # 存储原始数据
        self.items.append(text)
        self.metadata_list.append(metadata_with_text)

        logger.debug(f"添加项目: '{text[:30]}...'")
        return True

    def add_items_batch(self, texts: List[str], metadata_list: List[Dict[str, Any]]):
        """
        批量添加项目

        Args:
            texts: 文本列表
            metadata_list: 元数据列表
        """
        if len(texts) != len(metadata_list):
            raise ValueError("文本列表和元数据列表长度不匹配")

        logger.info(f"批量添加项目: {len(texts)}个")

        # 批量生成嵌入
        embedding_results = self.embedding_client.batch_embed_texts(texts)

        # 收集有效结果
        valid_vectors = []
        valid_metadata = []

        for i, (text, metadata, result) in enumerate(zip(texts, metadata_list, embedding_results)):
            if result.error is None:
                valid_vectors.append(result.embedding)
                metadata_with_text = metadata.copy()
                metadata_with_text["text"] = text
                valid_metadata.append(metadata_with_text)
            else:
                logger.warning(f"项目{i}生成嵌入失败: {result.error}")

        if valid_vectors:
            # 确保索引已初始化
            self._ensure_index(len(valid_vectors[0]))

            # 添加到索引
            vectors_array = np.array(valid_vectors)
            self.index.add(vectors_array, valid_metadata)

            # 存储原始数据
            self.items.extend(texts)
            self.metadata_list.extend(valid_metadata)

            logger.info(f"成功添加 {len(valid_vectors)}/{len(texts)} 个项目到索引")
        else:
            logger.warning("没有有效项目添加到索引")

    def search(self, query_text: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        搜索最相似的项目

        Args:
            query_text: 查询文本
            top_k: 返回前k个结果

        Returns:
            检索结果列表
        """
        if self.index is None or self.retriever is None:
            logger.warning("索引未初始化，无法搜索")
            return []

        logger.info(f"搜索查询: '{query_text[:30]}...'")

        # 生成查询嵌入
        embedding_result = self.embedding_client.embed_text(query_text)

        if embedding_result.error:
            logger.error(f"生成查询嵌入失败: {embedding_result.error}")
            return []

        # 向量检索
        vector_results = self.index.search(embedding_result.embedding, top_k=top_k * 2)

        if not vector_results:
            logger.warning("向量检索没有返回结果")
            return []

        # 混合检索
        retrieval_results = self.retriever.retrieve(
            query_text=query_text,
            query_embedding=embedding_result.embedding,
            vector_results=vector_results,
            top_k=top_k
        )

        return retrieval_results

    def batch_search(self, query_texts: List[str], top_k: int = 5) -> List[List[RetrievalResult]]:
        """
        批量搜索

        Args:
            query_texts: 查询文本列表
            top_k: 返回前k个结果

        Returns:
            批量检索结果列表
        """
        logger.info(f"批量搜索: {len(query_texts)}个查询")

        results = []

        for i, query_text in enumerate(query_texts):
            result = self.search(query_text, top_k=top_k)
            results.append(result)

            if (i + 1) % 5 == 0:
                logger.info(f"批量搜索进度: {i + 1}/{len(query_texts)}")

        return results

    def get_item(self, index: int) -> Optional[Dict[str, Any]]:
        """
        获取单个项目

        Args:
            index: 索引位置

        Returns:
            项目信息
        """
        if 0 <= index < len(self.items):
            return {
                "text": self.items[index],
                "metadata": self.metadata_list[index]
            }
        return None

    def get_all_items(self) -> List[Dict[str, Any]]:
        """获取所有项目"""
        return [
            {
                "text": self.items[i],
                "metadata": self.metadata_list[i]
            }
            for i in range(len(self.items))
        ]

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        embedding_stats = self.embedding_client.get_stats() if self.embedding_client else {}
        index_stats = self.index.get_info() if self.index else {}
        retriever_stats = self.retriever.get_stats() if self.retriever else {}

        return {
            "total_items": len(self.items),
            "embedding_stats": embedding_stats,
            "index_stats": index_stats,
            "retriever_stats": retriever_stats,
            "config": {
                "embedding_model": self.config.embedding_model,
                "index_type": self.config.index_type,
                "vector_weight": self.config.vector_weight,
                "template_weight": self.config.template_weight
            }
        }

    def save(self, path: str):
        """
        保存向量存储到文件

        Args:
            path: 文件路径
        """
        import pickle

        try:
            # 保存索引
            if self.index:
                self.index.save(path + "_index")

            # 保存数据
            data = {
                "items": self.items,
                "metadata_list": self.metadata_list,
                "config": self.config
            }
            with open(path + "_data.pkl", 'wb') as f:
                pickle.dump(data, f)

            logger.info(f"向量存储已保存到: {path}")

        except Exception as e:
            logger.error(f"保存向量存储失败: {e}")
            raise

    def load(self, path: str):
        """
        从文件加载向量存储

        Args:
            path: 文件路径
        """
        import pickle

        try:
            # 加载数据
            with open(path + "_data.pkl", 'rb') as f:
                data = pickle.load(f)

            self.items = data["items"]
            self.metadata_list = data["metadata_list"]
            self.config = data["config"]

            # 重新初始化组件
            self.embedding_client = EmbeddingClient(
                model_name=self.config.embedding_model,
                mock_mode=self.config.mock_mode,
                cache_enabled=self.config.cache_enabled
            )

            # 加载索引
            if self.items:
                # 确保索引已初始化
                embedding_result = self.embedding_client.embed_text(self.items[0])
                if embedding_result.error is None:
                    self._ensure_index(embedding_result.dimension)
                    if self.index:
                        self.index.load(path + "_index")

            logger.info(f"向量存储已从 {path} 加载")

        except Exception as e:
            logger.error(f"加载向量存储失败: {e}")
            raise

    def clear(self):
        """清空向量存储"""
        self.items.clear()
        self.metadata_list.clear()

        if self.index:
            self.index.clear()

        logger.info("向量存储已清空")

    def update_config(self, **kwargs):
        """
        更新配置

        Args:
            kwargs: 配置参数
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"更新配置: {key}={value}")

        # 如果权重变化，更新检索器
        if "vector_weight" in kwargs or "template_weight" in kwargs:
            if self.retriever:
                self.retriever.vector_weight = self.config.vector_weight
                self.retriever.template_weight = self.config.template_weight
