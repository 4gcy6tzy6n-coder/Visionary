"""
格式转换器 - Text2Loc增强版

提供新旧格式之间的双向转换
"""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OldFormat:
    """旧格式（原有Text2Loc）"""
    object_label: str  # 对象标签
    object_color: str  # 对象颜色
    direction: str  # 方向
    offset: np.ndarray  # 偏移量
    cell_id: str  # 单元格ID
    pose_id: str  # 位姿ID
    description: str  # 描述文本


@dataclass
class NewFormat:
    """新格式（增强版）"""
    object_name: str  # 对象名称（增强）
    object_color: str  # 对象颜色（增强）
    direction: str  # 方向（增强）
    direction_confidence: float  # 方向置信度
    object_confidence: float  # 对象置信度
    color_confidence: float  # 颜色置信度
    offset: np.ndarray  # 偏移量
    cell_id: str  # 单元格ID
    pose_id: str  # 位姿ID
    description: str  # 描述文本
    enhanced_description: Optional[str] = None  # 增强描述
    vector_embedding: Optional[np.ndarray] = None  # 向量嵌入
    spatial_relations: Optional[List[Dict[str, Any]]] = None  # 空间关系


class FormatConverter:
    """格式转换器"""

    def __init__(self):
        """初始化格式转换器"""
        logger.info("格式转换器初始化完成")

    def old_to_new(self, old_format: OldFormat) -> NewFormat:
        """
        旧格式转新格式

        Args:
            old_format: 旧格式数据

        Returns:
            新格式数据
        """
        # 基础字段映射
        new_format = NewFormat(
            object_name=old_format.object_label,
            object_color=old_format.object_color,
            direction=old_format.direction,
            direction_confidence=1.0,  # 旧格式默认置信度
            object_confidence=1.0,
            color_confidence=1.0,
            offset=old_format.offset,
            cell_id=old_format.cell_id,
            pose_id=old_format.pose_id,
            description=old_format.description,
            enhanced_description=None,
            vector_embedding=None,
            spatial_relations=None
        )

        logger.debug(f"旧格式转新格式: {old_format.object_label} -> {new_format.object_name}")
        return new_format

    def new_to_old(self, new_format: NewFormat) -> OldFormat:
        """
        新格式转旧格式

        Args:
            new_format: 新格式数据

        Returns:
            旧格式数据
        """
        # 基础字段映射
        old_format = OldFormat(
            object_label=new_format.object_name,
            object_color=new_format.object_color,
            direction=new_format.direction,
            offset=new_format.offset,
            cell_id=new_format.cell_id,
            pose_id=new_format.pose_id,
            description=new_format.description
        )

        logger.debug(f"新格式转旧格式: {new_format.object_name} -> {old_format.object_label}")
        return old_format

    def batch_old_to_new(self, old_formats: List[OldFormat]) -> List[NewFormat]:
        """
        批量旧格式转新格式

        Args:
            old_formats: 旧格式列表

        Returns:
            新格式列表
        """
        return [self.old_to_new(old) for old in old_formats]

    def batch_new_to_old(self, new_formats: List[NewFormat]) -> List[OldFormat]:
        """
        批量新格式转旧格式

        Args:
            new_formats: 新格式列表

        Returns:
            旧格式列表
        """
        return [self.new_to_old(new) for new in new_formats]

    def merge_formats(self,
                     old_format: OldFormat,
                     new_format: NewFormat) -> NewFormat:
        """
        合并新旧格式（保留旧格式的必要信息）

        Args:
            old_format: 旧格式数据
            new_format: 新格式数据

        Returns:
            合并后的格式
        """
        merged = NewFormat(
            object_name=new_format.object_name if new_format.object_name else old_format.object_label,
            object_color=new_format.object_color if new_format.object_color else old_format.object_color,
            direction=new_format.direction if new_format.direction else old_format.direction,
            direction_confidence=new_format.direction_confidence,
            object_confidence=new_format.object_confidence,
            color_confidence=new_format.color_confidence,
            offset=new_format.offset if new_format.offset is not None else old_format.offset,
            cell_id=new_format.cell_id if new_format.cell_id else old_format.cell_id,
            pose_id=new_format.pose_id if new_format.pose_id else old_format.pose_id,
            description=new_format.description if new_format.description else old_format.description,
            enhanced_description=new_format.enhanced_description,
            vector_embedding=new_format.vector_embedding,
            spatial_relations=new_format.spatial_relations
        )

        logger.debug(f"合并格式: {old_format.object_label} + {new_format.object_name}")
        return merged

    def validate_old_format(self, old_format: OldFormat) -> Tuple[bool, list]:
        """
        验证旧格式

        Args:
            old_format: 旧格式数据

        Returns:
            (是否有效, 错误列表)
        """
        errors = []

        if not old_format.object_label:
            errors.append("object_label不能为空")
        if not old_format.object_color:
            errors.append("object_color不能为空")
        if not old_format.direction:
            errors.append("direction不能为空")
        if old_format.offset is None:
            errors.append("offset不能为空")
        if not old_format.cell_id:
            errors.append("cell_id不能为空")
        if not old_format.pose_id:
            errors.append("pose_id不能为空")

        return len(errors) == 0, errors

    def validate_new_format(self, new_format: NewFormat) -> Tuple[bool, list]:
        """
        验证新格式

        Args:
            new_format: 新格式数据

        Returns:
            (是否有效, 错误列表)
        """
        errors = []

        if not new_format.object_name:
            errors.append("object_name不能为空")
        if not new_format.object_color:
            errors.append("object_color不能为空")
        if not new_format.direction:
            errors.append("direction不能为空")
        if new_format.offset is None:
            errors.append("offset不能为空")
        if not new_format.cell_id:
            errors.append("cell_id不能为空")
        if not new_format.pose_id:
            errors.append("pose_id不能为空")

        # 验证置信度
        if not (0.0 <= new_format.direction_confidence <= 1.0):
            errors.append("direction_confidence必须在0-1之间")
        if not (0.0 <= new_format.object_confidence <= 1.0):
            errors.append("object_confidence必须在0-1之间")
        if not (0.0 <= new_format.color_confidence <= 1.0):
            errors.append("color_confidence必须在0-1之间")

        return len(errors) == 0, errors

    def convert_to_description(self, new_format: NewFormat) -> str:
        """
        将新格式转换为描述文本

        Args:
            new_format: 新格式数据

        Returns:
            描述文本
        """
        # 使用增强信息生成描述
        if new_format.enhanced_description:
            return new_format.enhanced_description

        # 使用基础信息生成描述
        direction = new_format.direction
        color = new_format.object_color
        obj = new_format.object_name

        # 生成多种描述变体
        descriptions = [
            f"The pose is {direction} of a {color} {obj}.",
            f"A {color} {obj} is located to the {direction} of the pose.",
            f"To the {direction} of the pose, there is a {color} {obj}.",
        ]

        # 返回第一个描述
        return descriptions[0]

    def convert_to_batch_descriptions(self, new_formats: List[NewFormat]) -> List[str]:
        """
        批量转换为描述文本

        Args:
            new_formats: 新格式列表

        Returns:
            描述文本列表
        """
        return [self.convert_to_description(fmt) for fmt in new_formats]

    def extract_features(self, new_format: NewFormat) -> Dict[str, Any]:
        """
        从新格式提取特征

        Args:
            new_format: 新格式数据

        Returns:
            特征字典
        """
        features = {
            "object_name": new_format.object_name,
            "object_color": new_format.object_color,
            "direction": new_format.direction,
            "offset": new_format.offset.tolist() if isinstance(new_format.offset, np.ndarray) else new_format.offset,
            "cell_id": new_format.cell_id,
            "pose_id": new_format.pose_id,
            "description": new_format.description,
        }

        # 添加置信度信息
        features["direction_confidence"] = new_format.direction_confidence
        features["object_confidence"] = new_format.object_confidence
        features["color_confidence"] = new_format.color_confidence

        # 添加增强信息
        if new_format.enhanced_description:
            features["enhanced_description"] = new_format.enhanced_description

        if new_format.vector_embedding is not None:
            features["vector_embedding"] = new_format.vector_embedding.tolist()

        if new_format.spatial_relations:
            features["spatial_relations"] = new_format.spatial_relations

        return features

    def batch_extract_features(self, new_formats: List[NewFormat]) -> List[Dict[str, Any]]:
        """
        批量提取特征

        Args:
            new_formats: 新格式列表

        Returns:
            特征列表
        """
        return [self.extract_features(fmt) for fmt in new_formats]

    def create_from_dict(self, data: Dict[str, Any]) -> NewFormat:
        """
        从字典创建新格式

        Args:
            data: 字典数据

        Returns:
            新格式数据
        """
        # 提取必要字段
        object_name = data.get("object_name") or data.get("label") or data.get("object_label", "")
        object_color = data.get("object_color") or data.get("color", "")
        direction = data.get("direction", "")
        offset = data.get("offset")
        if isinstance(offset, list):
            offset = np.array(offset)
        elif offset is None:
            offset = np.array([0.0, 0.0])

        cell_id = data.get("cell_id", "")
        pose_id = data.get("pose_id", "")
        description = data.get("description", "")

        # 提取置信度
        direction_confidence = data.get("direction_confidence", 1.0)
        object_confidence = data.get("object_confidence", 1.0)
        color_confidence = data.get("color_confidence", 1.0)

        # 提取增强信息
        enhanced_description = data.get("enhanced_description")
        vector_embedding = data.get("vector_embedding")
        if vector_embedding is not None and isinstance(vector_embedding, list):
            vector_embedding = np.array(vector_embedding)

        spatial_relations = data.get("spatial_relations")

        return NewFormat(
            object_name=object_name,
            object_color=object_color,
            direction=direction,
            direction_confidence=direction_confidence,
            object_confidence=object_confidence,
            color_confidence=color_confidence,
            offset=offset,
            cell_id=cell_id,
            pose_id=pose_id,
            description=description,
            enhanced_description=enhanced_description,
            vector_embedding=vector_embedding,
            spatial_relations=spatial_relations
        )

    def to_dict(self, new_format: NewFormat) -> Dict[str, Any]:
        """
        将新格式转换为字典

        Args:
            new_format: 新格式数据

        Returns:
            字典数据
        """
        data = {
            "object_name": new_format.object_name,
            "object_color": new_format.object_color,
            "direction": new_format.direction,
            "direction_confidence": new_format.direction_confidence,
            "object_confidence": new_format.object_confidence,
            "color_confidence": new_format.color_confidence,
            "offset": new_format.offset.tolist() if isinstance(new_format.offset, np.ndarray) else new_format.offset,
            "cell_id": new_format.cell_id,
            "pose_id": new_format.pose_id,
            "description": new_format.description,
        }

        if new_format.enhanced_description:
            data["enhanced_description"] = new_format.enhanced_description

        if new_format.vector_embedding is not None:
            data["vector_embedding"] = new_format.vector_embedding.tolist()

        if new_format.spatial_relations:
            data["spatial_relations"] = new_format.spatial_relations

        return data
