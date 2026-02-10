"""
集成适配器 - Text2Loc增强版

将增强模块与原有Text2Loc系统集成
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加父目录到路径以支持datapreparation导入
_parent_dir = Path(__file__).parent.parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

try:
    from datapreparation.kitti360pose.select import get_direction
    from datapreparation.kitti360pose.utils import COLOR_NAMES
except ImportError:
    get_direction = None
    COLOR_NAMES = None
    logger.warning("datapreparation模块未找到，将使用模拟模式")


@dataclass
class IntegrationConfig:
    """集成配置"""
    # 功能开关
    enable_enhanced_direction: bool = True
    enable_enhanced_color: bool = True
    enable_enhanced_object: bool = True
    enable_vector_search: bool = True
    enable_hybrid_retrieval: bool = True
    
    # 回退配置
    fallback_on_error: bool = True
    fallback_on_low_confidence: bool = True
    confidence_threshold: float = 0.7
    
    # 性能配置
    cache_enabled: bool = True
    batch_size: int = 10
    
    # 模拟模式
    mock_mode: bool = True


class Text2LocAdapter:
    """Text2Loc集成适配器"""

    def __init__(self, config: Optional[IntegrationConfig] = None):
        """
        初始化集成适配器

        Args:
            config: 集成配置
        """
        self.config = config or IntegrationConfig()

        # 增强模块（可选，按需初始化）
        self.enhanced_direction_parser = None
        self.enhanced_color_mapper = None
        self.enhanced_object_identifier = None
        self.vector_store = None

        # 统计信息
        self.total_adaptations = 0
        self.enhanced_usage_count = 0
        self.fallback_count = 0

        logger.info(f"Text2Loc集成适配器初始化完成")

    def set_enhanced_direction_parser(self, parser):
        """设置增强方向解析器"""
        self.enhanced_direction_parser = parser
        logger.info("增强方向解析器已设置")

    def set_enhanced_color_mapper(self, mapper):
        """设置增强颜色映射器"""
        self.enhanced_color_mapper = mapper
        logger.info("增强颜色映射器已设置")

    def set_enhanced_object_identifier(self, identifier):
        """设置增强对象识别器"""
        self.enhanced_object_identifier = identifier
        logger.info("增强对象识别器已设置")

    def set_vector_store(self, vector_store):
        """设置向量存储"""
        self.vector_store = vector_store
        logger.info("向量存储已设置")

    def adapt_direction(self,
                       obj,
                       pose: np.ndarray,
                       original_direction: str) -> Tuple[str, float, bool]:
        """
        适配方向解析

        Args:
            obj: 对象
            pose: 位置
            original_direction: 原始方向

        Returns:
            (方向, 置信度, 是否使用增强)
        """
        self.total_adaptations += 1

        # 检查是否启用增强
        if not self.config.enable_enhanced_direction or self.enhanced_direction_parser is None:
            return original_direction, 1.0, False

        # 尝试使用增强解析
        try:
            # 生成描述文本（简化）
            description = f"object at position relative to pose"

            # 使用增强解析器
            if hasattr(self.enhanced_direction_parser, 'parse_direction'):
                result = self.enhanced_direction_parser.parse_direction(description)

                if result.use_enhanced and result.confidence >= self.config.confidence_threshold:
                    self.enhanced_usage_count += 1
                    logger.debug(f"使用增强方向解析: {original_direction} -> {result.standard_direction}")
                    return result.standard_direction, result.confidence, True

            # 回退到原始解析
            if self.config.fallback_on_error or self.config.fallback_on_low_confidence:
                self.fallback_count += 1
                logger.debug(f"回退到原始方向解析: {original_direction}")
                return original_direction, 0.5, False

        except Exception as e:
            logger.error(f"增强方向解析失败: {e}")
            if self.config.fallback_on_error:
                self.fallback_count += 1
                return original_direction, 0.5, False

        return original_direction, 1.0, False

    def adapt_color(self,
                   rgb: Tuple[int, int, int],
                   original_color: str) -> Tuple[str, float, bool]:
        """
        适配颜色识别

        Args:
            rgb: RGB值
            original_color: 原始颜色

        Returns:
            (颜色, 置信度, 是否使用增强)
        """
        self.total_adaptations += 1

        # 检查是否启用增强
        if not self.config.enable_enhanced_color or self.enhanced_color_mapper is None:
            return original_color, 1.0, False

        # 尝试使用增强识别
        try:
            result = self.enhanced_color_mapper.get_color_name(rgb)

            if result.confidence >= self.config.confidence_threshold:
                self.enhanced_usage_count += 1
                logger.debug(f"使用增强颜色识别: {original_color} -> {result.color_name}")
                return result.color_name, result.confidence, True

            # 回退
            if self.config.fallback_on_low_confidence:
                self.fallback_count += 1
                return original_color, 0.5, False

        except Exception as e:
            logger.error(f"增强颜色识别失败: {e}")
            if self.config.fallback_on_error:
                self.fallback_count += 1
                return original_color, 0.5, False

        return original_color, 1.0, False

    def adapt_object(self,
                    obj_features: Dict[str, Any],
                    original_label: str) -> Tuple[str, float, bool]:
        """
        适配对象识别

        Args:
            obj_features: 对象特征
            original_label: 原始标签

        Returns:
            (对象名称, 置信度, 是否使用增强)
        """
        self.total_adaptations += 1

        # 检查是否启用增强
        if not self.config.enable_enhanced_object or self.enhanced_object_identifier is None:
            return original_label, 1.0, False

        # 尝试使用增强识别
        try:
            # 准备候选对象（从原始标签推断）
            candidate_objects = [original_label]

            # 使用增强识别器
            result = self.enhanced_object_identifier.identify_object(
                obj_features,
                candidate_objects,
                method="auto"
            )

            if result.confidence >= self.config.confidence_threshold:
                self.enhanced_usage_count += 1
                logger.debug(f"使用增强对象识别: {original_label} -> {result.object_name}")
                return result.object_name, result.confidence, True

            # 回退
            if self.config.fallback_on_low_confidence:
                self.fallback_count += 1
                return original_label, 0.5, False

        except Exception as e:
            logger.error(f"增强对象识别失败: {e}")
            if self.config.fallback_on_error:
                self.fallback_count += 1
                return original_label, 0.5, False

        return original_label, 1.0, False

    def adapt_description(self,
                         description_text: str,
                         original_description: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool]:
        """
        适配描述解析

        Args:
            description_text: 描述文本
            original_description: 原始描述

        Returns:
            (解析结果, 置信度, 是否使用增强)
        """
        self.total_adaptations += 1

        # 检查是否启用向量检索
        if not self.config.enable_vector_search or self.vector_store is None:
            return original_description, 1.0, False

        # 尝试使用向量检索
        try:
            # 在向量存储中搜索
            results = self.vector_store.search(description_text, top_k=1)

            if results:
                best_result = results[0]
                if best_result.score >= self.config.confidence_threshold:
                    self.enhanced_usage_count += 1
                    logger.debug(f"使用向量检索增强: 置信度={best_result.score:.3f}")

                    # 合并结果
                    enhanced_description = original_description.copy()
                    enhanced_description.update(best_result.metadata)

                    return enhanced_description, best_result.score, True

            # 回退
            if self.config.fallback_on_low_confidence:
                self.fallback_count += 1
                return original_description, 0.5, False

        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            if self.config.fallback_on_error:
                self.fallback_count += 1
                return original_description, 0.5, False

        return original_description, 1.0, False

    def adapt_cell_retrieval(self,
                            text_description: str,
                            cell_objects: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float, bool]:
        """
        适配单元格检索

        Args:
            text_description: 文本描述
            cell_objects: 单元格对象列表

        Returns:
            (检索结果, 置信度, 是否使用增强)
        """
        self.total_adaptations += 1

        # 检查是否启用混合检索
        if not self.config.enable_hybrid_retrieval or self.vector_store is None:
            return cell_objects, 1.0, False

        # 尝试使用混合检索
        try:
            # 在向量存储中搜索
            results = self.vector_store.search(text_description, top_k=len(cell_objects))

            if results:
                # 提取元数据
                retrieved_objects = [r.metadata for r in results]

                # 计算平均置信度
                avg_confidence = np.mean([r.score for r in results])

                if avg_confidence >= self.config.confidence_threshold:
                    self.enhanced_usage_count += 1
                    logger.debug(f"使用混合检索增强: 平均置信度={avg_confidence:.3f}")
                    return retrieved_objects, avg_confidence, True

            # 回退
            if self.config.fallback_on_low_confidence:
                self.fallback_count += 1
                return cell_objects, 0.5, False

        except Exception as e:
            logger.error(f"混合检索失败: {e}")
            if self.config.fallback_on_error:
                self.fallback_count += 1
                return cell_objects, 0.5, False

        return cell_objects, 1.0, False

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        enhancement_rate = 0.0
        if self.total_adaptations > 0:
            enhancement_rate = self.enhanced_usage_count / self.total_adaptations

        fallback_rate = 0.0
        if self.total_adaptations > 0:
            fallback_rate = self.fallback_count / self.total_adaptations

        return {
            "total_adaptations": self.total_adaptations,
            "enhanced_usage_count": self.enhanced_usage_count,
            "fallback_count": self.fallback_count,
            "enhancement_rate": enhancement_rate,
            "fallback_rate": fallback_rate,
            "config": {
                "enable_enhanced_direction": self.config.enable_enhanced_direction,
                "enable_enhanced_color": self.config.enable_enhanced_color,
                "enable_enhanced_object": self.config.enable_enhanced_object,
                "enable_vector_search": self.config.enable_vector_search,
                "enable_hybrid_retrieval": self.config.enable_hybrid_retrieval,
                "confidence_threshold": self.config.confidence_threshold
            },
            "modules_available": {
                "direction_parser": self.enhanced_direction_parser is not None,
                "color_mapper": self.enhanced_color_mapper is not None,
                "object_identifier": self.enhanced_object_identifier is not None,
                "vector_store": self.vector_store is not None,
            }
        }

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

    def process_query(self, query: str, candidates: List[Dict[str, Any]], 
                      enable_enhanced: bool = True) -> Dict[str, Any]:
        """
        处理用户查询并返回定位结果

        Args:
            query: 用户自然语言查询
            candidates: 候选位置列表
            enable_enhanced: 是否使用增强功能

        Returns:
            Dict包含查询分析、检索结果和最终定位结果
        """
        import time
        start_time = time.time()
        
        self.total_adaptations += 1
        
        query_analysis = {
            "original_query": query,
            "object_name": None,
            "direction": None,
            "color": None,
            "distance": None,
            "enhanced_used": False
        }
        
        try:
            if enable_enhanced and self.enhanced_direction_parser:
                direction_result = self.adapt_direction(query)
                if direction_result:
                    query_analysis["direction"] = direction_result
                    query_analysis["enhanced_used"] = True
                    self.enhanced_usage_count += 1
            
            if enable_enhanced and self.enhanced_color_mapper:
                color_result = self.adapt_color(query)
                if color_result:
                    query_analysis["color"] = color_result
                    query_analysis["enhanced_used"] = True
                    self.enhanced_usage_count += 1
            
            if enable_enhanced and self.enhanced_object_identifier:
                object_result = self.adapt_object(query)
                if object_result:
                    query_analysis["object_name"] = object_result
                    query_analysis["enhanced_used"] = True
                    self.enhanced_usage_count += 1
            
            if not query_analysis["enhanced_used"]:
                for candidate in candidates:
                    if "description" in candidate:
                        query_analysis["object_name"] = candidate.get("object_label", "")
                        break
            
            retrieval_results = []
            for candidate in candidates:
                cell_id = candidate.get("cell_id", "")
                description = candidate.get("description", "")
                pose_id = candidate.get("pose_id", "")
                object_label = candidate.get("object_label", "")
                object_color = candidate.get("object_color", "")
                direction = candidate.get("direction", "")
                
                relevance_score = 0.5
                if query_analysis["object_name"]:
                    if query_analysis["object_name"].lower() in object_label.lower():
                        relevance_score += 0.3
                if query_analysis["direction"]:
                    if query_analysis["direction"].lower() == direction.lower():
                        relevance_score += 0.2
                if query_analysis["color"]:
                    if query_analysis["color"].lower() == object_color.lower():
                        relevance_score += 0.1
                
                relevance_score = min(relevance_score, 1.0)
                
                retrieval_results.append({
                    "cell_id": cell_id,
                    "description": description,
                    "pose_id": pose_id,
                    "object_label": object_label,
                    "object_color": object_color,
                    "direction": direction,
                    "relevance_score": relevance_score,
                    "retrieval_method": "enhanced" if query_analysis["enhanced_used"] else "fallback"
                })
            
            retrieval_results.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            final_result = None
            confidence = 0.0
            if retrieval_results:
                best = retrieval_results[0]
                confidence = best["relevance_score"]
                final_result = {
                    "cell_id": best["cell_id"],
                    "description": best["description"],
                    "pose_id": best["pose_id"],
                    "confidence": confidence,
                    "retrieval_method": best["retrieval_method"]
                }
            
            elapsed_time = (time.time() - start_time) * 1000
            
            return {
                "query_analysis": query_analysis,
                "retrieval_results": retrieval_results,
                "final_result": final_result,
                "confidence": confidence,
                "processing_time_ms": elapsed_time,
                "status": "success" if final_result else "no_match"
            }
            
        except Exception as e:
            logger.error(f"查询处理失败: {e}")
            if self.config.fallback_on_error:
                self.fallback_count += 1
                fallback_result = {
                    "query_analysis": query_analysis,
                    "retrieval_results": [],
                    "final_result": None,
                    "confidence": 0.0,
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "status": "error",
                    "error": str(e),
                    "fallback_used": True
                }
                return fallback_result
            raise
