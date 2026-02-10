"""
空间关系提取器 - Text2Loc增强版

提取对象间的方向、距离和拓扑关系
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SpatialRelation:
    """空间关系"""
    relation_type: str  # 关系类型（direction, distance, topology）
    relation_name: str  # 关系名称（如north, near, between）
    source_object: str  # 源对象
    target_object: str  # 目标对象
    confidence: float  # 置信度 (0.0-1.0)
    distance: Optional[float] = None  # 距离（米）
    direction_vector: Optional[Tuple[float, float, float]] = None  # 方向向量


class SpatialRelationExtractor:
    """空间关系提取器"""

    def __init__(self, distance_threshold: float = 5.0, angle_threshold: float = 45.0):
        """
        初始化空间关系提取器

        Args:
            distance_threshold: 距离阈值（米），用于判断"near"
            angle_threshold: 角度阈值（度），用于判断方向
        """
        self.distance_threshold = distance_threshold
        self.angle_threshold = angle_threshold

        logger.info(f"空间关系提取器初始化: 距离阈值={distance_threshold}m, 角度阈值={angle_threshold}°")

    def extract_direction_relation(self,
                                  source_pos: Tuple[float, float, float],
                                  target_pos: Tuple[float, float, float],
                                  source_name: str,
                                  target_name: str) -> SpatialRelation:
        """
        提取方向关系

        Args:
            source_pos: 源对象位置 (x, y, z)
            target_pos: 目标对象位置 (x, y, z)
            source_name: 源对象名称
            target_name: 目标对象名称

        Returns:
            方向关系
        """
        # 计算方向向量
        dx = target_pos[0] - source_pos[0]
        dy = target_pos[1] - source_pos[1]
        dz = target_pos[2] - source_pos[2]

        # 计算水平距离
        horizontal_distance = np.sqrt(dx ** 2 + dy ** 2)

        if horizontal_distance < 0.01:  # 非常接近
            # 检查垂直关系
            if abs(dz) > 0.1:
                if dz > 0:
                    relation_name = "above"
                else:
                    relation_name = "below"
                confidence = 0.9
            else:
                relation_name = "on_top"
                confidence = 0.95
        else:
            # 计算水平角度（相对于正北方向）
            angle = np.degrees(np.arctan2(dx, dy))  # 0°=北, 90°=东

            # 归一化角度到0-360
            if angle < 0:
                angle += 360

            # 确定方向
            directions = [
                ("north", 0, 22.5),
                ("north_east", 22.5, 67.5),
                ("east", 67.5, 112.5),
                ("south_east", 112.5, 157.5),
                ("south", 157.5, 202.5),
                ("south_west", 202.5, 247.5),
                ("west", 247.5, 292.5),
                ("north_west", 292.5, 337.5),
                ("north", 337.5, 360),
            ]

            relation_name = "unknown"
            confidence = 0.5

            for dir_name, min_angle, max_angle in directions:
                if min_angle <= angle < max_angle:
                    relation_name = dir_name
                    # 置信度基于角度范围
                    center_angle = (min_angle + max_angle) / 2
                    angle_diff = abs(angle - center_angle)
                    confidence = 1.0 - (angle_diff / (self.angle_threshold / 2))
                    confidence = max(0.5, min(1.0, confidence))
                    break

        # 创建方向关系
        relation = SpatialRelation(
            relation_type="direction",
            relation_name=relation_name,
            source_object=source_name,
            target_object=target_name,
            confidence=confidence,
            distance=horizontal_distance,
            direction_vector=(dx, dy, dz)
        )

        return relation

    def extract_distance_relation(self,
                                 source_pos: Tuple[float, float, float],
                                 target_pos: Tuple[float, float, float],
                                 source_name: str,
                                 target_name: str) -> SpatialRelation:
        """
        提取距离关系

        Args:
            source_pos: 源对象位置
            target_pos: 目标对象位置
            source_name: 源对象名称
            target_name: 目标对象名称

        Returns:
            距离关系
        """
        # 计算欧氏距离
        dx = target_pos[0] - source_pos[0]
        dy = target_pos[1] - source_pos[1]
        dz = target_pos[2] - source_pos[2]
        distance = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        # 确定距离关系
        if distance < 1.0:
            relation_name = "very_near"
            confidence = 0.9
        elif distance < self.distance_threshold:
            relation_name = "near"
            confidence = 0.8
        elif distance < self.distance_threshold * 2:
            relation_name = "moderately_far"
            confidence = 0.7
        else:
            relation_name = "far"
            confidence = 0.6

        relation = SpatialRelation(
            relation_type="distance",
            relation_name=relation_name,
            source_object=source_name,
            target_object=target_name,
            confidence=confidence,
            distance=distance,
            direction_vector=(dx, dy, dz)
        )

        return relation

    def extract_topology_relation(self,
                                 source_pos: Tuple[float, float, float],
                                 target_pos: Tuple[float, float, float],
                                 source_bbox: Tuple[float, float, float, float, float, float],
                                 target_bbox: Tuple[float, float, float, float, float, float],
                                 source_name: str,
                                 target_name: str) -> SpatialRelation:
        """
        提取拓扑关系（包含、相邻、相交）

        Args:
            source_pos: 源对象位置
            target_pos: 目标对象位置
            source_bbox: 源对象包围盒 (xmin, ymin, zmin, xmax, ymax, zmax)
            target_bbox: 目标对象包围盒
            source_name: 源对象名称
            target_name: 目标对象名称

        Returns:
            拓扑关系
        """
        # 检查包含关系
        if self._is_contained(source_bbox, target_bbox):
            relation_name = "contains"
            confidence = 0.9
        elif self._is_contained(target_bbox, source_bbox):
            relation_name = "contained_in"
            confidence = 0.9
        # 检查相交关系
        elif self._is_intersecting(source_bbox, target_bbox):
            relation_name = "intersects"
            confidence = 0.8
        # 检查相邻关系
        elif self._is_adjacent(source_bbox, target_bbox):
            relation_name = "adjacent"
            confidence = 0.7
        else:
            relation_name = "separate"
            confidence = 0.5

        relation = SpatialRelation(
            relation_type="topology",
            relation_name=relation_name,
            source_object=source_name,
            target_object=target_name,
            confidence=confidence
        )

        return relation

    def _is_contained(self, bbox1: Tuple[float, float, float, float, float, float],
                     bbox2: Tuple[float, float, float, float, float, float]) -> bool:
        """
        检查bbox1是否完全包含在bbox2内

        Args:
            bbox1: 包围盒1
            bbox2: 包围盒2

        Returns:
            是否包含
        """
        return (bbox1[0] >= bbox2[0] and bbox1[1] >= bbox2[1] and bbox1[2] >= bbox2[2] and
                bbox1[3] <= bbox2[3] and bbox1[4] <= bbox2[4] and bbox1[5] <= bbox2[5])

    def _is_intersecting(self, bbox1: Tuple[float, float, float, float, float, float],
                        bbox2: Tuple[float, float, float, float, float, float]) -> bool:
        """
        检查两个包围盒是否相交

        Args:
            bbox1: 包围盒1
            bbox2: 包围盒2

        Returns:
            是否相交
        """
        return not (bbox1[3] < bbox2[0] or bbox1[0] > bbox2[3] or
                    bbox1[4] < bbox2[1] or bbox1[1] > bbox2[4] or
                    bbox1[5] < bbox2[2] or bbox1[2] > bbox2[5])

    def _is_adjacent(self, bbox1: Tuple[float, float, float, float, float, float],
                    bbox2: Tuple[float, float, float, float, float, float],
                    threshold: float = 0.1) -> bool:
        """
        检查两个包围盒是否相邻

        Args:
            bbox1: 包围盒1
            bbox2: 包围盒2
            threshold: 阈值

        Returns:
            是否相邻
        """
        # 计算最小距离
        dx = max(0, bbox1[0] - bbox2[3], bbox2[0] - bbox1[3])
        dy = max(0, bbox1[1] - bbox2[4], bbox2[1] - bbox1[4])
        dz = max(0, bbox1[2] - bbox2[5], bbox2[2] - bbox1[5])

        distance = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        return distance <= threshold

    def extract_all_relations(self,
                             objects: List[Dict[str, Any]]) -> List[SpatialRelation]:
        """
        提取所有对象间的空间关系

        Args:
            objects: 对象列表，每个对象包含name, position, bbox等信息

        Returns:
            空间关系列表
        """
        relations = []

        if len(objects) < 2:
            return relations

        # 对每对对象提取关系
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects[i+1:], i+1):
                # 提取方向关系
                dir_rel = self.extract_direction_relation(
                    obj1["position"], obj2["position"],
                    obj1["name"], obj2["name"]
                )
                relations.append(dir_rel)

                # 提取距离关系
                dist_rel = self.extract_distance_relation(
                    obj1["position"], obj2["position"],
                    obj1["name"], obj2["name"]
                )
                relations.append(dist_rel)

                # 提取拓扑关系（如果有bbox）
                if "bbox" in obj1 and "bbox" in obj2:
                    topo_rel = self.extract_topology_relation(
                        obj1["position"], obj2["position"],
                        obj1["bbox"], obj2["bbox"],
                        obj1["name"], obj2["name"]
                    )
                    relations.append(topo_rel)

        logger.info(f"提取了{len(relations)}个空间关系")
        return relations

    def extract_descriptive_relations(self,
                                     reference_object: Dict[str, Any],
                                     target_objects: List[Dict[str, Any]]) -> List[SpatialRelation]:
        """
        提取描述性关系（用于生成自然语言描述）

        Args:
            reference_object: 参考对象
            target_objects: 目标对象列表

        Returns:
            描述性关系列表
        """
        relations = []

        for target in target_objects:
            # 提取方向关系
            dir_rel = self.extract_direction_relation(
                reference_object["position"], target["position"],
                reference_object["name"], target["name"]
            )

            # 提取距离关系
            dist_rel = self.extract_distance_relation(
                reference_object["position"], target["position"],
                reference_object["name"], target["name"]
            )

            # 组合成描述性关系
            if dir_rel.confidence > 0.6 and dist_rel.confidence > 0.6:
                # 创建组合关系
                combined_relation = SpatialRelation(
                    relation_type="descriptive",
                    relation_name=f"{dir_rel.relation_name}_{dist_rel.relation_name}",
                    source_object=reference_object["name"],
                    target_object=target["name"],
                    confidence=(dir_rel.confidence + dist_rel.confidence) / 2,
                    distance=dist_rel.distance,
                    direction_vector=dir_rel.direction_vector
                )
                relations.append(combined_relation)

        return relations

    def get_relation_statistics(self,
                               relations: List[SpatialRelation]) -> Dict[str, Any]:
        """
        获取关系统计信息

        Args:
            relations: 空间关系列表

        Returns:
            统计信息
        """
        if not relations:
            return {}

        # 统计关系类型
        type_counts = {}
        name_counts = {}
        total_confidence = 0.0

        for rel in relations:
            type_counts[rel.relation_type] = type_counts.get(rel.relation_type, 0) + 1
            name_counts[rel.relation_name] = name_counts.get(rel.relation_name, 0) + 1
            total_confidence += rel.confidence

        avg_confidence = total_confidence / len(relations)

        return {
            "total_relations": len(relations),
            "relation_types": type_counts,
            "relation_names": name_counts,
            "average_confidence": avg_confidence,
            "unique_relation_types": len(type_counts),
            "unique_relation_names": len(name_counts)
        }

    def update_thresholds(self, distance_threshold: float = None, angle_threshold: float = None):
        """
        更新阈值参数

        Args:
            distance_threshold: 距离阈值
            angle_threshold: 角度阈值
        """
        if distance_threshold is not None:
            self.distance_threshold = distance_threshold
            logger.info(f"更新距离阈值: {distance_threshold}m")

        if angle_threshold is not None:
            self.angle_threshold = angle_threshold
            logger.info(f"更新角度阈值: {angle_threshold}°")
