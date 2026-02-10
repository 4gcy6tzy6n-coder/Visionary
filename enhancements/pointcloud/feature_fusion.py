"""
特征融合优化 - Text2Loc增强版

融合几何、语义和空间特征，提供统一的对象表示
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FusionMethod(Enum):
    """融合方法"""
    CONCAT = "concat"  # 拼接
    ATTENTION = "attention"  # 注意力加权
    WEIGHTED = "weighted"  # 加权平均
    MLP = "mlp"  # 多层感知机


@dataclass
class FusedFeature:
    """融合后的特征"""
    feature_vector: np.ndarray  # 特征向量
    dimension: int  # 维度
    components: Dict[str, np.ndarray]  # 各组件特征
    weights: Dict[str, float]  # 各组件权重


class FeatureFusion:
    """特征融合器"""

    def __init__(self, fusion_method: FusionMethod = FusionMethod.WEIGHTED):
        """
        初始化特征融合器

        Args:
            fusion_method: 融合方法
        """
        self.fusion_method = fusion_method

        # 默认权重配置
        self.default_weights = {
            "geometric": 0.3,      # 几何特征
            "semantic": 0.4,       # 语义特征
            "spatial": 0.2,        # 空间特征
            "appearance": 0.1,     # 外观特征
        }

        logger.info(f"特征融合器初始化: 方法={fusion_method.value}")

    def fuse_geometric_features(self,
                               features: Dict[str, Any]) -> np.ndarray:
        """
        融合几何特征

        Args:
            features: 几何特征字典

        Returns:
            几何特征向量
        """
        # 提取几何特征
        position = features.get("position", [0, 0, 0])
        bbox = features.get("bbox", [0, 0, 0, 1, 1, 1])
        size = features.get("size", [1, 1, 1])
        num_points = features.get("num_points", 0)

        # 计算几何特征
        # 位置（归一化）
        pos_norm = np.array(position) / 100.0 if max(position) > 1 else np.array(position)

        # 包围盒特征
        bbox_center = [(bbox[0] + bbox[3]) / 2, (bbox[1] + bbox[4]) / 2, (bbox[2] + bbox[5]) / 2]
        bbox_size = [bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2]]

        # 归一化包围盒特征
        bbox_center_norm = np.array(bbox_center) / 100.0 if max(bbox_center) > 1 else np.array(bbox_center)
        bbox_size_norm = np.array(bbox_size) / 100.0 if max(bbox_size) > 1 else np.array(bbox_size)

        # 归一化尺寸
        size_norm = np.array(size) / 100.0 if max(size) > 1 else np.array(size)

        # 归一化点数
        num_points_norm = min(num_points / 10000.0, 1.0)

        # 拼接几何特征
        geometric_features = np.concatenate([
            pos_norm,
            bbox_center_norm,
            bbox_size_norm,
            size_norm,
            [num_points_norm]
        ])

        return geometric_features

    def fuse_semantic_features(self,
                              features: Dict[str, Any]) -> np.ndarray:
        """
        融合语义特征

        Args:
            features: 语义特征字典

        Returns:
            语义特征向量
        """
        # 提取语义特征
        category = features.get("category", "unknown")
        color = features.get("color", "unknown")
        shape = features.get("shape", "unknown")
        material = features.get("material", "unknown")

        # 类别编码（简化版）
        category_mapping = {
            "building": [1, 0, 0, 0, 0, 0, 0, 0, 0],
            "vehicle": [0, 1, 0, 0, 0, 0, 0, 0, 0],
            "infrastructure": [0, 0, 1, 0, 0, 0, 0, 0, 0],
            "road": [0, 0, 0, 1, 0, 0, 0, 0, 0],
            "vegetation": [0, 0, 0, 0, 1, 0, 0, 0, 0],
            "person": [0, 0, 0, 0, 0, 1, 0, 0, 0],
            "object": [0, 0, 0, 0, 0, 0, 1, 0, 0],
            "terrain": [0, 0, 0, 0, 0, 0, 0, 1, 0],
            "unknown": [0, 0, 0, 0, 0, 0, 0, 0, 1],
        }

        category_vector = category_mapping.get(category, category_mapping["unknown"])

        # 颜色编码（简化版）
        color_mapping = {
            "red": [1, 0, 0, 0, 0, 0, 0, 0],
            "green": [0, 1, 0, 0, 0, 0, 0, 0],
            "blue": [0, 0, 1, 0, 0, 0, 0, 0],
            "yellow": [0, 0, 0, 1, 0, 0, 0, 0],
            "gray": [0, 0, 0, 0, 1, 0, 0, 0],
            "black": [0, 0, 0, 0, 0, 1, 0, 0],
            "white": [0, 0, 0, 0, 0, 0, 1, 0],
            "unknown": [0, 0, 0, 0, 0, 0, 0, 1],
        }

        color_vector = color_mapping.get(color, color_mapping["unknown"])

        # 形状编码（简化版）
        shape_mapping = {
            "box": [1, 0, 0, 0, 0],
            "cylinder": [0, 1, 0, 0, 0],
            "sphere": [0, 0, 1, 0, 0],
            "plane": [0, 0, 0, 1, 0],
            "unknown": [0, 0, 0, 0, 1],
        }

        shape_vector = shape_mapping.get(shape, shape_mapping["unknown"])

        # 材质编码（简化版）
        material_mapping = {
            "metal": [1, 0, 0, 0],
            "plastic": [0, 1, 0, 0],
            "wood": [0, 0, 1, 0],
            "concrete": [0, 0, 0, 1],
            "unknown": [0, 0, 0, 0],
        }

        material_vector = material_mapping.get(material, material_mapping["unknown"])

        # 拼接语义特征
        semantic_features = np.concatenate([
            category_vector,
            color_vector,
            shape_vector,
            material_vector
        ])

        return semantic_features

    def fuse_spatial_features(self,
                             features: Dict[str, Any],
                             context_objects: List[Dict[str, Any]] = None) -> np.ndarray:
        """
        融合空间特征

        Args:
            features: 空间特征字典
            context_objects: 上下文对象列表

        Returns:
            空间特征向量
        """
        # 提取空间特征
        position = features.get("position", [0, 0, 0])
        direction = features.get("direction", "unknown")
        distance = features.get("distance", 0.0)

        # 方向编码
        direction_mapping = {
            "north": [1, 0, 0, 0, 0],
            "east": [0, 1, 0, 0, 0],
            "south": [0, 0, 1, 0, 0],
            "west": [0, 0, 0, 1, 0],
            "on_top": [0, 0, 0, 0, 1],
            "unknown": [0, 0, 0, 0, 0],
        }

        direction_vector = direction_mapping.get(direction, direction_mapping["unknown"])

        # 距离归一化
        distance_norm = min(distance / 100.0, 1.0)

        # 位置归一化
        pos_norm = np.array(position) / 100.0 if max(position) > 1 else np.array(position)

        # 上下文特征（如果有）
        context_features = np.zeros(5)  # 默认5维上下文特征
        if context_objects:
            # 计算与上下文对象的平均距离
            avg_distance = 0.0
            for obj in context_objects:
                if "position" in obj:
                    obj_pos = obj["position"]
                    dist = np.sqrt(
                        (position[0] - obj_pos[0]) ** 2 +
                        (position[1] - obj_pos[1]) ** 2 +
                        (position[2] - obj_pos[2]) ** 2
                    )
                    avg_distance += dist

            if context_objects:
                avg_distance /= len(context_objects)
                context_features[0] = min(avg_distance / 100.0, 1.0)

        # 拼接空间特征
        spatial_features = np.concatenate([
            pos_norm,
            direction_vector,
            [distance_norm],
            context_features
        ])

        return spatial_features

    def fuse_appearance_features(self,
                                features: Dict[str, Any]) -> np.ndarray:
        """
        融合外观特征

        Args:
            features: 外观特征字典

        Returns:
            外观特征向量
        """
        # 提取外观特征
        color = features.get("color", "unknown")
        texture = features.get("texture", "unknown")
        brightness = features.get("brightness", 0.5)

        # 颜色编码（RGB）
        if color == "red":
            color_vector = [1, 0, 0]
        elif color == "green":
            color_vector = [0, 1, 0]
        elif color == "blue":
            color_vector = [0, 0, 1]
        else:
            color_vector = [0.5, 0.5, 0.5]

        # 纹理编码
        texture_mapping = {
            "smooth": [1, 0, 0],
            "rough": [0, 1, 0],
            "patterned": [0, 0, 1],
            "unknown": [0.33, 0.33, 0.33],
        }

        texture_vector = texture_mapping.get(texture, texture_mapping["unknown"])

        # 亮度归一化
        brightness_norm = max(0.0, min(1.0, brightness))

        # 拼接外观特征
        appearance_features = np.concatenate([
            color_vector,
            texture_vector,
            [brightness_norm]
        ])

        return appearance_features

    def fuse_features(self,
                     features: Dict[str, Any],
                     context_objects: List[Dict[str, Any]] = None,
                     method: Optional[FusionMethod] = None) -> FusedFeature:
        """
        融合所有特征

        Args:
            features: 特征字典
            context_objects: 上下文对象列表
            method: 融合方法

        Returns:
            融合后的特征
        """
        if method is None:
            method = self.fusion_method

        # 提取各组件特征
        components = {}

        # 几何特征
        if any(key in features for key in ["position", "bbox", "size", "num_points"]):
            components["geometric"] = self.fuse_geometric_features(features)

        # 语义特征
        if any(key in features for key in ["category", "color", "shape", "material"]):
            components["semantic"] = self.fuse_semantic_features(features)

        # 空间特征
        if any(key in features for key in ["position", "direction", "distance"]):
            components["spatial"] = self.fuse_spatial_features(features, context_objects)

        # 外观特征
        if any(key in features for key in ["color", "texture", "brightness"]):
            components["appearance"] = self.fuse_appearance_features(features)

        if not components:
            logger.warning("没有可融合的特征")
            return FusedFeature(
                feature_vector=np.array([]),
                dimension=0,
                components={},
                weights={}
            )

        # 根据融合方法进行融合
        if method == FusionMethod.CONCAT:
            # 拼接
            fused = np.concatenate(list(components.values()))
            weights = {k: 1.0 for k in components.keys()}

        elif method == FusionMethod.WEIGHTED:
            # 加权平均
            weights = {}
            weighted_features = []

            for key, feature in components.items():
                weight = self.default_weights.get(key, 0.25)
                weights[key] = weight
                weighted_features.append(feature * weight)

            fused = np.sum(weighted_features, axis=0)

        elif method == FusionMethod.ATTENTION:
            # 注意力加权（简化版）
            # 基于特征方差计算注意力权重
            weights = {}
            attention_scores = []

            for key, feature in components.items():
                # 使用方差作为注意力分数
                score = np.var(feature) if len(feature) > 0 else 0.0
                attention_scores.append(score)

            # 归一化注意力分数
            total = sum(attention_scores)
            if total > 0:
                attention_scores = [s / total for s in attention_scores]
            else:
                attention_scores = [1.0 / len(components)] * len(components)

            # 加权求和
            weighted_features = []
            for i, (key, feature) in enumerate(components.items()):
                weight = attention_scores[i]
                weights[key] = weight
                weighted_features.append(feature * weight)

            fused = np.sum(weighted_features, axis=0)

        elif method == FusionMethod.MLP:
            # MLP融合（简化版：加权平均）
            logger.warning("MLP融合未完全实现，使用加权平均替代")
            weights = {}
            weighted_features = []

            for key, feature in components.items():
                weight = self.default_weights.get(key, 0.25)
                weights[key] = weight
                weighted_features.append(feature * weight)

            fused = np.sum(weighted_features, axis=0)

        else:
            logger.warning(f"未知融合方法: {method}，使用加权平均")
            weights = {}
            weighted_features = []

            for key, feature in components.items():
                weight = self.default_weights.get(key, 0.25)
                weights[key] = weight
                weighted_features.append(feature * weight)

            fused = np.sum(weighted_features, axis=0)

        # 归一化融合特征
        norm = np.linalg.norm(fused)
        if norm > 0:
            fused = fused / norm

        return FusedFeature(
            feature_vector=fused,
            dimension=len(fused),
            components=components,
            weights=weights
        )

    def batch_fuse_features(self,
                           features_list: List[Dict[str, Any]],
                           context_objects_list: List[List[Dict[str, Any]]] = None,
                           method: Optional[FusionMethod] = None) -> List[FusedFeature]:
        """
        批量融合特征

        Args:
            features_list: 特征字典列表
            context_objects_list: 上下文对象列表列表
            method: 融合方法

        Returns:
            融合特征列表
        """
        if context_objects_list is None:
            context_objects_list = [None] * len(features_list)

        results = []

        for i, (features, context_objects) in enumerate(zip(features_list, context_objects_list)):
            result = self.fuse_features(features, context_objects, method)
            results.append(result)

            if (i + 1) % 10 == 0:
                logger.info(f"批量融合进度: {i + 1}/{len(features_list)}")

        return results

    def update_weights(self, new_weights: Dict[str, float]):
        """
        更新权重配置

        Args:
            new_weights: 新权重字典
        """
        for key, weight in new_weights.items():
            if key in self.default_weights:
                self.default_weights[key] = weight
                logger.info(f"更新权重: {key}={weight}")

    def get_fusion_info(self) -> Dict[str, Any]:
        """获取融合信息"""
        return {
            "fusion_method": self.fusion_method.value,
            "weights": self.default_weights,
            "component_names": list(self.default_weights.keys())
        }
