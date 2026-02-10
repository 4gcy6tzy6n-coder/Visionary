"""
开放词汇对象识别 - Text2Loc增强版

使用CLIP模型和向量相似度匹配，支持开放词汇对象识别
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ObjectRecognitionResult:
    """对象识别结果"""
    object_name: str  # 对象名称
    confidence: float  # 置信度 (0.0-1.0)
    category: str  # 所属类别
    alternatives: List[Dict[str, Any]]  # 备选结果
    is_open_vocab: bool  # 是否为开放词汇识别


class OpenVocabObjectIdentifier:
    """开放词汇对象识别器"""

    # 预定义类别层次结构
    CATEGORY_HIERARCHY = {
        "vehicle": ["car", "truck", "bus", "motorcycle", "bicycle", "van"],
        "building": ["building", "house", "apartment", "office", "store"],
        "infrastructure": ["pole", "lamppost", "sign", "traffic_light", "fence", "wall"],
        "road": ["road", "street", "highway", "lane", "parking"],
        "sidewalk": ["sidewalk", "pavement", "walkway"],
        "vegetation": ["tree", "bush", "shrub", "grass", "plant"],
        "terrain": ["terrain", "ground", "field", "lawn"],
        "person": ["person", "people", "pedestrian", "human"],
        "object": ["box", "trash_bin", "vending_machine", "bench"],
    }

    # 类别到父类的映射
    CATEGORY_TO_PARENT = {
        "car": "vehicle", "truck": "vehicle", "bus": "vehicle",
        "motorcycle": "vehicle", "bicycle": "vehicle", "van": "vehicle",
        "building": "building", "house": "building", "apartment": "building",
        "office": "building", "store": "building",
        "pole": "infrastructure", "lamppost": "infrastructure", "sign": "infrastructure",
        "traffic_light": "infrastructure", "fence": "infrastructure", "wall": "infrastructure",
        "road": "road", "street": "road", "highway": "road", "lane": "road", "parking": "road",
        "sidewalk": "sidewalk", "pavement": "sidewalk", "walkway": "sidewalk",
        "tree": "vegetation", "bush": "vegetation", "shrub": "vegetation",
        "grass": "vegetation", "plant": "vegetation",
        "terrain": "terrain", "ground": "terrain", "field": "terrain", "lawn": "terrain",
        "person": "person", "people": "person", "pedestrian": "person", "human": "person",
        "box": "object", "trash_bin": "object", "vending_machine": "object", "bench": "object",
    }

    # 开放词汇描述模板
    OPEN_VOCAB_TEMPLATES = [
        "a photo of {object}",
        "an image of {object}",
        "a picture of {object}",
        "this is a {object}",
        "there is a {object}",
        "{object} in the scene",
    ]

    def __init__(self, use_embedding: bool = True, mock_mode: bool = False):
        """
        初始化开放词汇对象识别器

        Args:
            use_embedding: 是否使用嵌入向量（需要EmbeddingClient）
            mock_mode: 模拟模式
        """
        self.use_embedding = use_embedding
        self.mock_mode = mock_mode

        # 嵌入客户端（可选）
        self.embedding_client = None

        # 对象描述缓存
        self.description_cache = {}

        # 统计信息
        self.total_recognitions = 0
        self.total_time = 0.0

        logger.info(f"开放词汇对象识别器初始化: use_embedding={use_embedding}, mock={mock_mode}")

    def set_embedding_client(self, embedding_client):
        """
        设置嵌入客户端

        Args:
            embedding_client: EmbeddingClient实例
        """
        self.embedding_client = embedding_client
        logger.info("嵌入客户端已设置")

    def get_object_descriptions(self, object_type: str, features: Dict[str, Any]) -> List[str]:
        """
        生成对象的自然语言描述

        Args:
            object_type: 对象类型
            features: 对象特征（颜色、形状、材质等）

        Returns:
            描述列表
        """
        descriptions = []

        # 基础描述
        base_desc = f"{object_type}"
        descriptions.append(base_desc)

        # 带颜色的描述
        if "color" in features:
            color = features["color"]
            descriptions.append(f"{color} {object_type}")
            descriptions.append(f"{object_type} with {color} color")

        # 带形状的描述
        if "shape" in features:
            shape = features["shape"]
            descriptions.append(f"{shape} {object_type}")
            descriptions.append(f"{object_type} that is {shape}")

        # 带位置的描述
        if "position" in features:
            pos = features["position"]
            if "relative" in pos:
                descriptions.append(f"{object_type} {pos['relative']}")

        # 带材质的描述
        if "material" in features:
            material = features["material"]
            descriptions.append(f"{material} {object_type}")

        return descriptions

    def generate_embedding_descriptions(self, object_type: str, features: Dict[str, Any]) -> List[str]:
        """
        生成用于嵌入的描述

        Args:
            object_type: 对象类型
            features: 对象特征

        Returns:
            用于嵌入的描述列表
        """
        descriptions = []

        # 使用模板生成描述
        for template in self.OPEN_VOCAB_TEMPLATES:
            # 基础描述
            desc = template.format(object=object_type)
            descriptions.append(desc)

            # 带颜色的描述
            if "color" in features:
                color = features["color"]
                desc_with_color = template.format(object=f"{color} {object_type}")
                descriptions.append(desc_with_color)

        # 生成更详细的描述
        detail_parts = []
        if "color" in features:
            detail_parts.append(features["color"])
        if "shape" in features:
            detail_parts.append(features["shape"])
        if "material" in features:
            detail_parts.append(features["material"])

        if detail_parts:
            detail_desc = f"{object_type} that is {' and '.join(detail_parts)}"
            descriptions.append(detail_desc)

        return descriptions

    def recognize_by_embedding(self,
                              object_features: Dict[str, Any],
                              candidate_objects: List[str],
                              top_k: int = 3) -> ObjectRecognitionResult:
        """
        通过嵌入向量识别对象

        Args:
            object_features: 对象特征
            candidate_objects: 候选对象列表
            top_k: 返回前k个结果

        Returns:
            识别结果
        """
        if not self.embedding_client:
            logger.warning("嵌入客户端未设置，无法进行嵌入识别")
            return self._fallback_recognize(object_features)

        start_time = time.time()

        try:
            # 生成对象描述
            object_type = object_features.get("type", "object")
            descriptions = self.generate_embedding_descriptions(object_type, object_features)

            if not descriptions:
                logger.warning("无法生成描述，使用默认对象类型")
                return self._fallback_recognize(object_features)

            # 为候选对象生成嵌入
            candidate_embeddings = []
            candidate_texts = []

            for candidate in candidate_objects:
                candidate_desc = f"a photo of {candidate}"
                candidate_texts.append(candidate_desc)

            # 批量生成候选嵌入
            candidate_results = self.embedding_client.batch_embed_texts(candidate_texts)

            # 收集有效嵌入
            for i, result in enumerate(candidate_results):
                if result.error is None:
                    candidate_embeddings.append(result.embedding)
                else:
                    logger.warning(f"生成候选嵌入失败{i}: {result.error}")

            if not candidate_embeddings:
                logger.warning("没有有效的候选嵌入，使用默认识别")
                return self._fallback_recognize(object_features)

            # 生成对象特征嵌入（使用描述）
            object_embedding_results = self.embedding_client.batch_embed_texts(descriptions)

            # 收集有效嵌入
            object_embeddings = []
            for result in object_embedding_results:
                if result.error is None:
                    object_embeddings.append(result.embedding)

            if not object_embeddings:
                logger.warning("无法生成对象嵌入，使用默认识别")
                return self._fallback_recognize(object_features)

            # 计算平均对象嵌入
            avg_object_embedding = np.mean(object_embeddings, axis=0)

            # 计算与每个候选对象的相似度
            similarities = []
            for i, candidate_emb in enumerate(candidate_embeddings):
                sim = self.embedding_client.cosine_similarity(avg_object_embedding, candidate_emb)
                similarities.append((candidate_objects[i], sim))

            # 按相似度排序
            similarities.sort(key=lambda x: x[1], reverse=True)

            # 获取最佳匹配
            if similarities:
                best_object, best_similarity = similarities[0]

                # 生成备选结果
                alternatives = []
                for obj, sim in similarities[1:top_k]:
                    alternatives.append({
                        "object": obj,
                        "confidence": float(sim),
                        "category": self.CATEGORY_TO_PARENT.get(obj, "unknown")
                    })

                elapsed_time = time.time() - start_time
                self.total_recognitions += 1
                self.total_time += elapsed_time

                logger.info(f"开放词汇识别完成: {best_object} (置信度: {best_similarity:.3f})")

                return ObjectRecognitionResult(
                    object_name=best_object,
                    confidence=float(best_similarity),
                    category=self.CATEGORY_TO_PARENT.get(best_object, "unknown"),
                    alternatives=alternatives,
                    is_open_vocab=True
                )
            else:
                logger.warning("没有找到匹配的对象")
                return self._fallback_recognize(object_features)

        except Exception as e:
            logger.error(f"嵌入识别失败: {e}")
            return self._fallback_recognize(object_features)

    def _fallback_recognize(self, object_features: Dict[str, Any]) -> ObjectRecognitionResult:
        """
        回退识别方法（基于规则）

        Args:
            object_features: 对象特征

        Returns:
            识别结果
        """
        object_type = object_features.get("type", "object")
        color = object_features.get("color", "")
        shape = object_features.get("shape", "")

        # 构建对象名称
        if color and shape:
            object_name = f"{color} {shape} {object_type}"
        elif color:
            object_name = f"{color} {object_type}"
        elif shape:
            object_name = f"{shape} {object_type}"
        else:
            object_name = object_type

        # 确定类别
        category = self.CATEGORY_TO_PARENT.get(object_type, "unknown")

        logger.warning(f"使用回退识别: {object_name}")

        return ObjectRecognitionResult(
            object_name=object_name,
            confidence=0.5,  # 回退方法置信度较低
            category=category,
            alternatives=[],
            is_open_vocab=False
        )

    def recognize_by_rules(self,
                          object_features: Dict[str, Any],
                          known_objects: List[Dict[str, Any]]) -> ObjectRecognitionResult:
        """
        通过规则匹配识别对象

        Args:
            object_features: 对象特征
            known_objects: 已知对象列表

        Returns:
            识别结果
        """
        start_time = time.time()

        try:
            object_type = object_features.get("type", "object")
            color = object_features.get("color", "")
            shape = object_features.get("shape", "")

            # 计算匹配分数
            scores = []

            for known_obj in known_objects:
                score = 0.0

                # 类型匹配
                if known_obj.get("type") == object_type:
                    score += 0.5

                # 颜色匹配
                if color and known_obj.get("color") == color:
                    score += 0.3

                # 形状匹配
                if shape and known_obj.get("shape") == shape:
                    score += 0.2

                scores.append((known_obj, score))

            # 按分数排序
            scores.sort(key=lambda x: x[1], reverse=True)

            if scores and scores[0][1] > 0:
                best_obj, best_score = scores[0]

                elapsed_time = time.time() - start_time
                self.total_recognitions += 1
                self.total_time += elapsed_time

                return ObjectRecognitionResult(
                    object_name=best_obj.get("name", object_type),
                    confidence=best_score,
                    category=best_obj.get("category", "unknown"),
                    alternatives=[],
                    is_open_vocab=False
                )
            else:
                return self._fallback_recognize(object_features)

        except Exception as e:
            logger.error(f"规则识别失败: {e}")
            return self._fallback_recognize(object_features)

    def batch_recognize(self,
                       object_features_list: List[Dict[str, Any]],
                       candidate_objects: List[str] = None,
                       use_embedding: bool = True) -> List[ObjectRecognitionResult]:
        """
        批量识别对象

        Args:
            object_features_list: 对象特征列表
            candidate_objects: 候选对象列表
            use_embedding: 是否使用嵌入识别

        Returns:
            识别结果列表
        """
        results = []

        for i, features in enumerate(object_features_list):
            if use_embedding and self.embedding_client and candidate_objects:
                result = self.recognize_by_embedding(features, candidate_objects)
            else:
                result = self._fallback_recognize(features)

            results.append(result)

            if (i + 1) % 10 == 0:
                logger.info(f"批量识别进度: {i + 1}/{len(object_features_list)}")

        return results

    def get_category_hierarchy(self) -> Dict[str, List[str]]:
        """获取类别层次结构"""
        return self.CATEGORY_HIERARCHY.copy()

    def get_parent_category(self, object_name: str) -> str:
        """
        获取父类别

        Args:
            object_name: 对象名称

        Returns:
            父类别名称
        """
        return self.CATEGORY_TO_PARENT.get(object_name, "unknown")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        avg_time = self.total_time / self.total_recognitions if self.total_recognitions > 0 else 0

        return {
            "total_recognitions": self.total_recognitions,
            "total_time": self.total_time,
            "average_time": avg_time,
            "embedding_available": self.embedding_client is not None,
            "mock_mode": self.mock_mode
        }

    def identify_object(self,
                       object_features: Dict[str, Any],
                       candidate_objects: List[str] = None,
                       method: str = "auto") -> ObjectRecognitionResult:
        """
        识别对象（主接口）

        Args:
            object_features: 对象特征
            candidate_objects: 候选对象列表
            method: 识别方法（auto, embedding, rules, fallback）

        Returns:
            识别结果
        """
        if method == "auto":
            if self.embedding_client and candidate_objects:
                return self.recognize_by_embedding(object_features, candidate_objects)
            else:
                return self._fallback_recognize(object_features)
        elif method == "embedding":
            if self.embedding_client and candidate_objects:
                return self.recognize_by_embedding(object_features, candidate_objects)
            else:
                logger.warning("嵌入识别不可用，使用回退方法")
                return self._fallback_recognize(object_features)
        elif method == "fallback":
            return self._fallback_recognize(object_features)
        else:
            logger.warning(f"未知方法: {method}，使用回退方法")
            return self._fallback_recognize(object_features)
