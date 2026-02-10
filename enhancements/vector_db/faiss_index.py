"""
Faiss向量索引管理器 - Text2Loc增强版

提供高效的向量索引构建、更新和检索功能
"""

import logging
import pickle
import time
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("Faiss未安装，将使用模拟模式")


@dataclass
class IndexStats:
    """索引统计信息"""
    num_vectors: int  # 向量数量
    dimension: int  # 向量维度
    index_type: str  # 索引类型
    memory_usage: float  # 内存使用（MB）
    build_time: float  # 构建时间（秒）
    last_update: Optional[str] = None  # 最后更新时间


class FaissIndex:
    """Faiss向量索引管理器"""

    def __init__(self,
                 dimension: int,
                 index_type: str = "FlatL2",
                 metric: str = "l2",
                 mock_mode: bool = False):
        """
        初始化Faiss索引

        Args:
            dimension: 向量维度
            index_type: 索引类型（FlatL2, IVF_FLAT, IVF_PQ等）
            metric: 距离度量（l2, inner_product, cosine）
            mock_mode: 模拟模式（无Faiss时使用）
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.mock_mode = mock_mode or not FAISS_AVAILABLE

        self.index = None
        self.vector_to_id = {}  # 向量ID到元数据的映射
        self.id_to_vector = {}  # 元数据到向量ID的映射
        self.next_id = 0

        # 性能统计
        self.stats = IndexStats(
            num_vectors=0,
            dimension=dimension,
            index_type=index_type,
            memory_usage=0.0,
            build_time=0.0
        )

        # 模拟索引（当Faiss不可用时）
        self.mock_index = []
        self.mock_metadata = []

        if not self.mock_mode:
            self._build_index()
            logger.info(f"Faiss索引初始化完成: {dimension}维, 类型={index_type}")
        else:
            logger.warning("Faiss未安装或使用模拟模式，将使用内存列表模拟索引")
            logger.info(f"模拟索引初始化完成: {dimension}维")

    def _build_index(self):
        """构建Faiss索引"""
        if self.mock_mode:
            return

        start_time = time.time()

        try:
            if self.index_type == "FlatL2":
                self.index = faiss.IndexFlatL2(self.dimension)
            elif self.index_type == "IVF_FLAT":
                # 使用IVF_FLAT索引，需要预训练
                nlist = 100  # 质心数量
                quantizer = faiss.IndexFlatL2(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            elif self.index_type == "IVF_PQ":
                # 使用IVF_PQ索引，压缩存储
                nlist = 100
                m = 8  # 子向量数量
                quantizer = faiss.IndexFlatL2(self.dimension)
                self.index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, m)
            else:
                # 默认使用FlatL2
                self.index = faiss.IndexFlatL2(self.dimension)

            elapsed_time = time.time() - start_time
            self.stats.build_time = elapsed_time

            logger.info(f"Faiss索引构建完成: {self.index_type}, 耗时{elapsed_time:.3f}秒")

        except Exception as e:
            logger.error(f"Faiss索引构建失败: {e}")
            raise

    def add(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]):
        """
        向索引中添加向量

        Args:
            vectors: 向量数组，形状为(n, dimension)
            metadata: 每个向量的元数据列表
        """
        if len(vectors) == 0:
            return

        start_time = time.time()

        try:
            if self.mock_mode:
                # 模拟模式：存储在列表中
                for vec, meta in zip(vectors, metadata):
                    self.mock_index.append(vec)
                    self.mock_metadata.append(meta)
                    self.vector_to_id[self.next_id] = meta
                    self.id_to_vector[meta.get("id", self.next_id)] = self.next_id
                    self.next_id += 1

                self.stats.num_vectors = len(self.mock_index)
                logger.info(f"添加向量到模拟索引: {len(vectors)}个")

            else:
                # 真实Faiss模式
                vectors = np.ascontiguousarray(vectors, dtype=np.float32)

                # 如果是IVF索引，需要先训练
                if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                    logger.info("训练IVF索引...")
                    self.index.train(vectors)

                # 添加向量
                self.index.add(vectors)

                # 存储元数据
                for i, meta in enumerate(metadata):
                    vector_id = self.index.ntotal - len(vectors) + i
                    self.vector_to_id[vector_id] = meta
                    self.id_to_vector[meta.get("id", vector_id)] = vector_id

                self.stats.num_vectors = self.index.ntotal
                logger.info(f"添加向量到Faiss索引: {len(vectors)}个，总计{self.index.ntotal}个")

            elapsed_time = time.time() - start_time
            self.stats.last_update = time.strftime("%Y-%m-%d %H:%M:%S")

            # 更新内存使用估算
            self._update_memory_usage()

            logger.debug(f"添加向量耗时: {elapsed_time:.3f}秒")

        except Exception as e:
            logger.error(f"添加向量失败: {e}")
            raise

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        搜索最相似的向量

        Args:
            query_vector: 查询向量
            top_k: 返回前k个结果

        Returns:
            [(元数据, 相似度), ...] 列表
        """
        start_time = time.time()

        try:
            if self.mock_mode:
                # 模拟模式：计算所有向量的相似度
                similarities = []
                for i, vector in enumerate(self.mock_index):
                    sim = self._cosine_similarity(query_vector, vector)
                    similarities.append((i, sim))

                # 按相似度排序
                similarities.sort(key=lambda x: x[1], reverse=True)

                # 返回前k个结果
                results = []
                for idx, sim in similarities[:top_k]:
                    if idx < len(self.mock_metadata):
                        results.append((self.mock_metadata[idx], sim))

                elapsed_time = time.time() - start_time
                logger.debug(f"模拟搜索耗时: {elapsed_time:.3f}秒")

                return results

            else:
                # 真实Faiss模式
                query_vector = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1)

                if self.metric == "cosine":
                    # 余弦相似度需要归一化
                    query_vector = query_vector / np.linalg.norm(query_vector)

                distances, indices = self.index.search(query_vector, top_k)

                results = []
                for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
                    if idx >= 0 and idx in self.vector_to_id:
                        # Faiss返回的距离是L2距离，需要转换为相似度
                        if self.metric == "l2":
                            # L2距离越小越相似，转换为0-1的相似度
                            similarity = 1.0 / (1.0 + dist)
                        else:
                            similarity = 1.0 - dist

                        results.append((self.vector_to_id[idx], similarity))

                elapsed_time = time.time() - start_time
                logger.debug(f"Faiss搜索耗时: {elapsed_time:.3f}秒")

                return results

        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """计算余弦相似度"""
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        return float(np.dot(v1_norm, v2_norm))

    def _update_memory_usage(self):
        """更新内存使用估算"""
        if self.mock_mode:
            # 模拟模式：估算内存使用
            vector_size = self.dimension * 4  # float32
            metadata_size = 100  # 估算每个元数据100字节
            self.stats.memory_usage = (len(self.mock_index) * (vector_size + metadata_size)) / (1024 * 1024)
        else:
            # 真实Faiss模式
            if self.index is not None:
                # 估算Faiss索引内存使用
                vector_size = self.dimension * 4  # float32
                index_overhead = 1.5  # Faiss索引开销系数
                self.stats.memory_usage = (self.index.ntotal * vector_size * index_overhead) / (1024 * 1024)

    def save(self, path: str):
        """
        保存索引到文件

        Args:
            path: 文件路径
        """
        try:
            if self.mock_mode:
                # 保存模拟索引
                data = {
                    "mock_index": self.mock_index,
                    "mock_metadata": self.mock_metadata,
                    "vector_to_id": self.vector_to_id,
                    "id_to_vector": self.id_to_vector,
                    "next_id": self.next_id,
                    "stats": self.stats
                }
                with open(path, 'wb') as f:
                    pickle.dump(data, f)
            else:
                # 保存Faiss索引
                faiss.write_index(self.index, path + ".index")

                # 保存元数据
                data = {
                    "vector_to_id": self.vector_to_id,
                    "id_to_vector": self.id_to_vector,
                    "next_id": self.next_id,
                    "stats": self.stats
                }
                with open(path + ".meta", 'wb') as f:
                    pickle.dump(data, f)

            logger.info(f"索引已保存到: {path}")

        except Exception as e:
            logger.error(f"保存索引失败: {e}")
            raise

    def load(self, path: str):
        """
        从文件加载索引

        Args:
            path: 文件路径
        """
        try:
            if self.mock_mode:
                # 加载模拟索引
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                self.mock_index = data["mock_index"]
                self.mock_metadata = data["mock_metadata"]
                self.vector_to_id = data["vector_to_id"]
                self.id_to_vector = data["id_to_vector"]
                self.next_id = data["next_id"]
                self.stats = data["stats"]
            else:
                # 加载Faiss索引
                self.index = faiss.read_index(path + ".index")

                # 加载元数据
                with open(path + ".meta", 'rb') as f:
                    data = pickle.load(f)
                self.vector_to_id = data["vector_to_id"]
                self.id_to_vector = data["id_to_vector"]
                self.next_id = data["next_id"]
                self.stats = data["stats"]

            logger.info(f"索引已从 {path} 加载")

        except Exception as e:
            logger.error(f"加载索引失败: {e}")
            raise

    def clear(self):
        """清空索引"""
        if self.mock_mode:
            self.mock_index.clear()
            self.mock_metadata.clear()
        else:
            if self.index is not None:
                self.index.reset()

        self.vector_to_id.clear()
        self.id_to_vector.clear()
        self.next_id = 0
        self.stats.num_vectors = 0

        logger.info("索引已清空")

    def get_stats(self) -> IndexStats:
        """获取索引统计信息"""
        self._update_memory_usage()
        return self.stats

    def get_info(self) -> Dict[str, Any]:
        """获取索引详细信息"""
        return {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "num_vectors": self.stats.num_vectors,
            "memory_usage_mb": self.stats.memory_usage,
            "build_time": self.stats.build_time,
            "last_update": self.stats.last_update,
            "mock_mode": self.mock_mode,
            "faiss_available": FAISS_AVAILABLE
        }
