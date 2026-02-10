"""
DynamicTemplateGenerator - 动态模板生成器

负责生成多种Text2Loc兼容的模板变体
支持动态模板选择、质量评估、模板学习和优化
"""

import json
import random
import time
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TemplateVariant:
    """模板变体"""

    template: str  # 原始模板格式
    filled_text: str  # 填充后的文本
    quality_score: float  # 质量评分 0.0-1.0
    template_type: str  # 模板类型: base, natural, hybrid, minimal
    compatibility_level: str  # 兼容层级: full, partial, minimal
    confidence: float = 0.8  # 置信度
    usage_score: float = 0.0  # 使用评分（基于历史成功率）

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "template": self.template,
            "filled_text": self.filled_text,
            "quality_score": self.quality_score,
            "template_type": self.template_type,
            "compatibility_level": self.compatibility_level,
            "confidence": self.confidence,
            "usage_score": self.usage_score
        }


@dataclass
class ParsedResult:
    """解析结果"""

    direction: Optional[str] = None
    color: Optional[str] = None
    object: Optional[str] = None
    relation: Optional[str] = None
    distance: Optional[float] = None

    completeness_score: float = 0.0
    confidence_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "direction": self.direction,
            "color": self.color,
            "object": self.object,
            "relation": self.relation,
            "distance": self.distance,
            "completeness_score": self.completeness_score,
            "confidence_scores": self.confidence_scores
        }


@dataclass
class TemplatePerformance:
    """模板性能统计"""

    template_type: str
    total_usage: int = 0
    successful_usage: int = 0
    total_time_ms: float = 0.0
    avg_quality: float = 0.0

    @property
    def success_rate(self) -> float:
        """成功率"""
        return self.successful_usage / self.total_usage if self.total_usage > 0 else 0.0

    @property
    def avg_time_ms(self) -> float:
        """平均时间"""
        return self.total_time_ms / self.total_usage if self.total_usage > 0 else 0.0


class TemplateQualityScorer:
    """模板质量评分器"""

    # 质量评分权重
    WEIGHTS = {
        "completeness": 0.3,  # 完整性
        "specificity": 0.25,  # 特异性
        "readability": 0.2,   # 可读性
        "compatibility": 0.25 # 兼容性
    }

    def __init__(self):
        # 标准对象类别（22种）
        self.standard_objects = [
            "box", "bridge", "building", "fence", "garage", "guard rail",
            "lamp", "pad", "parking", "pole", "road", "sidewalk",
            "smallpole", "stop", "terrain", "traffic light", "traffic sign",
            "trash bin", "tunnel", "vegetation", "vending machine", "wall"
        ]

        # 标准颜色（8种）
        self.standard_colors = [
            "dark-green", "gray", "gray-green", "bright-gray",
            "black", "green", "beige"
        ]

        # 标准方向（5种）
        self.standard_directions = ["north", "south", "east", "west", "on-top"]

    def score(
        self,
        template: str,
        filled_text: str,
        parsed_result: ParsedResult
    ) -> float:
        """
        评估模板质量

        Args:
            template: 模板格式
            filled_text: 填充后的文本
            parsed_result: 解析结果

        Returns:
            质量评分 0.0-1.0
        """
        scores = {}

        # 1. 完整性评分
        scores["completeness"] = self._score_completeness(parsed_result)

        # 2. 特异性评分
        scores["specificity"] = self._score_specificity(filled_text)

        # 3. 可读性评分
        scores["readability"] = self._score_readability(filled_text)

        # 4. 兼容性评分
        scores["compatibility"] = self._score_compatibility(filled_text)

        # 加权平均
        total_score = 0.0
        for factor, weight in self.WEIGHTS.items():
            total_score += scores.get(factor, 0.5) * weight

        return total_score

    def _score_completeness(self, parsed_result: ParsedResult) -> float:
        """评估完整性"""
        score = 0.5  # 基础分

        # 根据关键元素存在情况加分
        if parsed_result.object:
            score += 0.2
        if parsed_result.direction:
            score += 0.15
        if parsed_result.color:
            score += 0.1
        if parsed_result.relation:
            score += 0.05

        return min(score, 1.0)

    def _score_specificity(self, text: str) -> float:
        """评估特异性（越具体分数越高）"""
        score = 0.3  # 基础分

        # 检查是否包含标准颜色
        for color in self.standard_colors:
            if color in text:
                score += 0.2
                break

        # 检查是否包含标准对象
        for obj in self.standard_objects:
            # 近似匹配，因为文本中可能有"的"、"色"等
            if obj in text or obj.replace(' ', '') in text:
                score += 0.3
                break

        # 检查数字（距离）
        if any(char.isdigit() for char in text):
            score += 0.1

        # 检查方向词
        for direction in self.standard_directions:
            if direction in text:
                score += 0.1
                break

        return min(score, 1.0)

    def _score_readability(self, text: str) -> float:
        """评估可读性"""
        score = 0.5

        # 基础分（假设大部分模板可读性尚可）
        score = 0.7

        # 简单启发式：检查长度是否适中
        text_length = len(text)
        if 5 <= text_length <= 50:
            score += 0.2
        elif 50 < text_length <= 100:
            score += 0.1
        else:
            score -= 0.1

        return max(0.1, min(score, 1.0))

    def _score_compatibility(self, text: str) -> float:
        """评估与Text2Loc的兼容性"""
        # Text2Loc原系统偏好简单、明确的描述
        score = 0.5

        # 检查是否包含基础颜色词（Text2Loc兼容）
        base_colors = ["red", "green", "blue", "gray", "black", "white"]
        if any(color in text for color in base_colors):
            score += 0.3

        # 检查是否包含标准方向
        base_directions = ["north", "south", "east", "west", "on-top"]
        if any(dir in text for dir in base_directions):
            score += 0.2

        return min(score, 1.0)


class DynamicTemplateGenerator:
    """动态模板生成器"""

    # 基础模板库（保持与原有Text2Loc兼容）
    BASE_TEMPLATES = [
        # 英文模板
        "The pose is {direction} of a {color} {object}.",
        "A {color} {object} is located to the {direction} of the pose.",
        "To the {direction} of the pose, there is a {color} {object}.",
        "Located {direction} of the {color} {object}.",
        "Position is {direction} relative to a {color} {object}.",

        # 中文自然模板
        "在{color}{object}的{direction}侧",
        "位于{direction}方向的{color}{object}附近",
        "{direction}边有一个{color}色的{object}",
        "在{color}{object}{relation}，方向为{direction}",

        # 混合模板
        "Position: {direction} of {color} {object}",
        "{object} ({color}) - {direction} direction",
        "{direction} | {color} {object}",

        # 简化模板（兼容性最高）
        "{direction} {color} {object}",
        "{object} {direction}",
        "{color} {object} {direction}",

        # 空间关系模板
        "{relation} {color} {object} with {direction} orientation",
        "At {distance} meters {direction} of the {color} {object}",
        "{direction} of the {object}, {distance}m away",
    ]

    # 模板分类
    TEMPLATE_CATEGORIES = {
        "base": {
            "description": "基础Text2Loc兼容模板",
            "priority": 1.0,
            "templates": [
                "The pose is {direction} of a {color} {object}.",
                "A {color} {object} is located to the {direction} of the pose.",
                "To the {direction} of the pose, there is a {color} {object}.",
                "Position is {direction} relative to a {color} {object}.",
            ]
        },
        "natural": {
            "description": "自然语言模板（中文）",
            "priority": 0.9,
            "templates": [
                "在{color}{object}的{direction}侧",
                "位于{direction}方向的{color}{object}附近",
                "{direction}边有一个{color}色的{object}",
                "在{color}{object}{relation}，方向为{direction}",
            ]
        },
        "hybrid": {
            "description": "混合语言模板",
            "priority": 0.8,
            "templates": [
                "Position: {direction} of {color} {object}",
                "{object} ({color}) - {direction} direction",
                "{direction} | {color} {object}",
            ]
        },
        "minimal": {
            "description": "极简模板（高兼容性）",
            "priority": 0.7,
            "templates": [
                "{direction} {color} {object}",
                "{object} {direction}",
                "{color} {object} {direction}",
            ]
        },
        "spatial": {
            "description": "带空间关系的模板",
            "priority": 0.6,
            "templates": [
                "{relation} {color} {object} with {direction} orientation",
                "At {distance} meters {direction} of the {color} {object}",
                "{direction} of the {object}, {distance}m away",
            ]
        },
    }

    # 字段映射（用于模板填充）
    FIELD_MAP = {
        "direction": ["direction", "orient", "pos", "location_dir"],
        "color": ["color", "col", "c"],
        "object": ["object", "obj", "target", "t"],
        "relation": ["relation", "rel", "near", "beside"],
        "distance": ["distance", "dist", "d", "meters"],
    }

    def __init__(self):
        """
        初始化动态模板生成器
        """
        self.quality_scorer = TemplateQualityScorer()

        # 模板性能统计
        self.template_performance: Dict[str, TemplatePerformance] = defaultdict(
            lambda: TemplatePerformance(template_type="unknown")
        )

        # 模板学习历史
        self.learning_history: List[Dict] = []

        # 使用统计
        self.total_generations = 0
        self.total_time_ms = 0.0

        logger.info("DynamicTemplateGenerator 初始化完成")

    def generate(
        self,
        parsed_result: ParsedResult,
        n_variants: int = 5,
        prefer_compatibility: bool = True
    ) -> List[TemplateVariant]:
        """
        生成多种模板变体

        根据：
        1. 解析置信度选择模板类型
        2. 要素完整性决定模板复杂度
        3. 历史效果选择最优模板

        Args:
            parsed_result: 解析结果
            n_variants: 生成变体数量
            prefer_compatibility: 是否优先兼容性

        Returns:
            模板变体列表
        """
        start_time = time.time()
        self.total_generations += 1

        try:
            templates = []

            # 根据要素完整性选择模板集
            completeness = parsed_result.completeness_score

            if completeness > 0.9:
                # 完整查询：使用所有模板类型
                category_keys = ["base", "natural", "hybrid", "minimal", "spatial"]
            elif completeness > 0.7:
                # 较完整：基础+自然+极简
                category_keys = ["base", "natural", "minimal"]
            elif completeness > 0.5:
                # 一般完整：基础+极简
                category_keys = ["base", "minimal"]
            else:
                # 不完整：仅极简
                category_keys = ["minimal"]

            # 为模板选择添加随机性（避免总是相同结果）
            if prefer_compatibility:
                # 如果需要兼容性，优先极简模板
                if "minimal" not in category_keys:
                    category_keys.insert(0, "minimal")

            # 生成模板变体
            for category_key in category_keys:
                category = self.TEMPLATE_CATEGORIES.get(category_key)
                if not category:
                    continue

                # 从每个类别中选择2个模板
                template_list = category["templates"]
                selected_templates = random.sample(
                    template_list,
                    min(2, len(template_list))
                )

                for template in selected_templates:
                    try:
                        # 填充模板
                        filled = self._fill_template(template, parsed_result)

                        if not filled:
                            continue

                        # 评估质量
                        score = self.quality_scorer.score(
                            template, filled, parsed_result
                        )

                        # 获取性能评分
                        perf_score = self._get_template_performance_score(template)

                        # 综合评分
                        final_score = (score * 0.7) + (perf_score * 0.3)

                        # 检查兼容性
                        compatibility = self._check_compatibility(filled)

                        variant = TemplateVariant(
                            template=template,
                            filled_text=filled,
                            quality_score=final_score,
                            template_type=category_key,
                            compatibility_level=compatibility,
                            confidence=parsed_result.completeness_score
                        )

                        templates.append(variant)

                    except Exception as e:
                        logger.warning(f"填充模板失败: {e}")
                        continue

                    if len(templates) >= n_variants:
                        break

                if len(templates) >= n_variants:
                    break

            # 如果没有生成足够的模板，使用基础模板补全
            if len(templates) < n_variants:
                for template in self.TEMPLATE_CATEGORIES["minimal"]["templates"]:
                    try:
                        filled = self._fill_template(template, parsed_result)
                        if filled:
                            score = self.quality_scorer.score(
                                template, filled, parsed_result
                            )
                            variant = TemplateVariant(
                                template=template,
                                filled_text=filled,
                                quality_score=score,
                                template_type="minimal",
                                compatibility_level="full",
                                confidence=parsed_result.completeness_score
                            )
                            templates.append(variant)
                    except Exception:
                        continue

                    if len(templates) >= n_variants:
                        break

            # 按质量排序
            templates.sort(key=lambda x: x.quality_score, reverse=True)

            elapsed = time.time() - start_time
            self.total_time_ms += elapsed * 1000

            logger.info(f"✅ 模板生成完成: {len(templates)}个变体")
            logger.info(f"   用时: {elapsed*1000:.1f}ms")

            return templates

        except Exception as e:
            logger.error(f"模板生成失败: {e}")
            return self._fallback_templates(parsed_result, n_variants)

    def _fill_template(self, template: str, parsed_result: ParsedResult) -> Optional[str]:
        """
        填充模板

        Args:
            template: 模板字符串
            parsed_result: 解析结果

        Returns:
            填充后的字符串
        """
        fields = {
            "direction": parsed_result.direction,
            "color": parsed_result.color,
            "object": parsed_result.object,
            "relation": parsed_result.relation,
            "distance": parsed_result.distance,
        }

        # 移除空值字段
        filled_fields = {
            k: v for k, v in fields.items() if v is not None
        }

        # 检查模板中需要的字段
        required_fields = []
        for field_name in fields.keys():
            if f"{{{field_name}}}" in template:
                required_fields.append(field_name)

        # 如果缺少必要字段，返回None
        for field in required_fields:
            if field not in filled_fields:
                return None

        # 填充模板
        try:
            filled = template.format(**filled_fields)
            return filled
        except KeyError as e:
            logger.debug(f"模板字段缺失: {e}")
            return None

    def _check_compatibility(self, text: str) -> str:
        """
        检查与Text2Loc的兼容性

        Args:
            text: 填充后的文本

        Returns:
            兼容层级: full, partial, minimal
        """
        # Text2Loc原系统使用22种标准对象和8种标准颜色
        # 兼容性检查基于此

        compatibility_score = 0.0

        # 检查包含标准对象（全兼容）
        standard_objects = [
            "box", "bridge", "building", "fence", "garage",
            "lamp", "parking", "pole", "road", "sidewalk",
            "smallpole", "stop", "terrain", "traffic",
            "trash", "tunnel", "vegetation", "vending", "wall"
        ]

        for obj in standard_objects:
            if obj in text.lower():
                compatibility_score += 0.4
                break

        # 检查标准颜色
        standard_colors = ["dark-green", "gray", "bright-gray", "black", "green", "beige"]
        for color in standard_colors:
            if color in text.lower():
                compatibility_score += 0.3
                break

        # 检查标准方向
        standard_directions = ["north", "south", "east", "west", "on-top"]
        for direction in standard_directions:
            if direction in text.lower():
                compatibility_score += 0.3
                break

        # 评分转兼容层级
        if compatibility_score >= 0.8:
            return "full"
        elif compatibility_score >= 0.5:
            return "partial"
        else:
            return "minimal"

    def _get_template_performance_score(self, template: str) -> float:
        """
        获取模板性能评分

        历史成功率越高，得分越高

        Args:
            template: 模板字符串

        Returns:
            性能评分 0.0-1.0
        """
        if template not in self.template_performance:
            return 0.5  # 默认分

        perf = self.template_performance[template]

        if perf.total_usage == 0:
            return 0.5

        # 基于成功率和使用次数的性能分数
        success_rate = perf.success_rate
        usage_log = min(perf.total_usage / 100, 1.0)  # 使用次数越多越可靠

        performance_score = (success_rate * 0.7) + (usage_log * 0.3)

        return performance_score

    def update_template_performance(
        self,
        template: str,
        template_type: str,
        success: bool,
        time_ms: float
    ):
        """
        更新模板性能统计

        Args:
            template: 模板字符串
            template_type: 模板类型
            success: 是否成功
            time_ms: 用时
        """
        if template not in self.template_performance:
            self.template_performance[template] = TemplatePerformance(
                template_type=template_type
            )

        perf = self.template_performance[template]
        perf.total_usage += 1

        if success:
            perf.successful_usage += 1

        perf.total_time_ms += time_ms

        # 更新平均质量
        perf.avg_quality = (
            perf.avg_quality * (perf.total_usage - 1) + (1.0 if success else 0.5)
        ) / perf.total_usage

        # 记录学习历史
        self.learning_history.append({
            "timestamp": datetime.now().isoformat(),
            "template": template,
            "template_type": template_type,
            "success": success,
            "time_ms": time_ms,
            "total_usage": perf.total_usage,
            "success_rate": perf.success_rate
        })

        logger.debug(f"更新模板性能: {template[:50]}... success={success}")

    def _fallback_templates(
        self,
        parsed_result: ParsedResult,
        n_variants: int
    ) -> List[TemplateVariant]:
        """
        备用模板生成（失败时使用）

        Args:
            parsed_result: 解析结果
            n_variants: 需要的变体数量

        Returns:
            模板变体列表
        """
        templates = []

        # 使用最简单的模板
        simple_templates = [
            "{direction} {object}",
            "{color} {object}",
            "{direction} {color} {object}",
        ]

        for template in simple_templates:
            try:
                filled = self._fill_template(template, parsed_result)
                if filled:
                    variant = TemplateVariant(
                        template=template,
                        filled_text=filled,
                        quality_score=0.3,
                        template_type="fallback",
                        compatibility_level="full",
                        confidence=parsed_result.completeness_score
                    )
                    templates.append(variant)
            except Exception:
                continue

            if len(templates) >= n_variants:
                break

        return templates

    def learn_from_experience(
        self,
        template: str,
        success: bool,
        quality_score: float,
        use_case: str = "general"
    ):
        """
        从经验中学习

        Args:
            template: 模板字符串
            success: 是否成功
            quality_score: 质量评分
            use_case: 使用场景
        """
        # 更新性能统计
        if template in self.template_performance:
            perf = self.template_performance[template]
            if success:
                perf.successful_usage += 1
            perf.total_usage += 1

            # 更新平均质量
            perf.avg_quality = (
                perf.avg_quality * (perf.total_usage - 1) + quality_score
            ) / perf.total_usage

        # 记录学习记录
        self.learning_history.append({
            "timestamp": datetime.now().isoformat(),
            "template": template,
            "success": success,
            "quality_score": quality_score,
            "use_case": use_case,
            "learning_type": "experience"
        })

        logger.debug(f"经验学习: template={template[:50]}... success={success}")

    def get_best_template(
        self,
        parsed_result: ParsedResult,
        mode: str = "balanced"
    ) -> Optional[TemplateVariant]:
        """
        获取最佳模板

        Args:
            parsed_result: 解析结果
            mode: 模式
                - "fast": 快速返回第一个
                - "balanced": 平衡质量和速度
                - "high_quality": 优先高质量
                - "compatible": 优先兼容性

        Returns:
            最佳模板变体
        """
        variants = self.generate(parsed_result, n_variants=5)

        if not variants:
            return None

        if mode == "fast":
            return variants[0]

        elif mode == "balanced":
            # 综合评分 = 0.7*质量 + 0.3*性能
            for variant in variants:
                perf_score = self._get_template_performance_score(variant.template)
                variant.quality_score = variant.quality_score * 0.7 + perf_score * 0.3
            variants.sort(key=lambda x: x.quality_score, reverse=True)
            return variants[0]

        elif mode == "high_quality":
            # 仅按质量排序
            variants.sort(key=lambda x: x.quality_score, reverse=True)
            return variants[0]

        elif mode == "compatible":
            # 优先兼容性
            compatibility_order = {"full": 3, "partial": 2, "minimal": 1}
            variants.sort(
                key=lambda x: (
                    compatibility_order.get(x.compatibility_level, 0),
                    x.quality_score
                ),
                reverse=True
            )
            return variants[0]

        return variants[0]

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""

        # 按类型统计
        type_stats = defaultdict(lambda: {
            "total_usage": 0,
            "successful_usage": 0,
            "success_rate": 0.0,
            "avg_quality": 0.0,
            "avg_time_ms": 0.0
        })

        for template, perf in self.template_performance.items():
            type_key = perf.template_type
            stats = type_stats[type_key]

            stats["total_usage"] += perf.total_usage
            stats["successful_usage"] += perf.successful_usage
            stats["avg_quality"] += perf.avg_quality * perf.total_usage
            stats["avg_time_ms"] += perf.avg_time_ms * perf.total_usage

        # 计算平均值
        for key in type_stats.keys():
            stats = type_stats[key]
            if stats["total_usage"] > 0:
                stats["success_rate"] = stats["successful_usage"] / stats["total_usage"]
                stats["avg_quality"] /= stats["total_usage"]
                stats["avg_time_ms"] /= stats["total_usage"]

        # 常用模板
        top_templates = sorted(
            self.template_performance.items(),
            key=lambda x: x[1].total_usage,
            reverse=True
        )[:10]

        return {
            "total_generations": self.total_generations,
            "total_time_ms": self.total_time_ms,
            "avg_time_ms": (self.total_time_ms / self.total_generations
                          if self.total_generations > 0 else 0.0),
            "unique_templates": len(self.template_performance),
            "type_stats": dict(type_stats),
            "top_templates": [
                {
                    "template": t[:100] + "..." if len(t) > 100 else t,
                    "type": p.template_type,
                    "usage": p.total_usage,
                    "success_rate": p.success_rate,
                    "avg_quality": p.avg_quality
                }
                for t, p in top_templates
            ],
            "learning_history_count": len(self.learning_history),
            "recent_learnings": self.learning_history[-5:] if self.learning_history else []
        }

    def get_template_families(self) -> Dict[str, List[str]]:
        """获取模板家族分类"""
        families = {}

        for category_key, category_info in self.TEMPLATE_CATEGORIES.items():
            families[category_key] = {
                "description": category_info["description"],
                "templates": category_info["templates"],
                "count": len(category_info["templates"])
            }

        return families

    def export_learnings(self) -> Dict[str, Any]:
        """导出学习结果"""
        return {
            "template_performance": {
                template: {
                    "template_type": perf.template_type,
                    "total_usage": perf.total_usage,
                    "successful_usage": perf.successful_usage,
                    "avg_quality": perf.avg_quality,
                    "avg_time_ms": perf.avg_time_ms,
                }
                for template, perf in self.template_performance.items()
            },
            "learning_history": self.learning_history[-100:],  # 最近100条
            "statistics": {
                "total_generations": self.total_generations,
                "total_time_ms": self.total_time_ms,
                "unique_templates": len(self.template_performance)
            }
        }

    def import_learnings(self, data: Dict[str, Any]):
        """导入学习结果"""
        if "template_performance" in data:
            for template, perf_data in data["template_performance"].items():
                perf = TemplatePerformance(
                    template_type=perf_data["template_type"]
                )
                perf.total_usage = perf_data["total_usage"]
                perf.successful_usage = perf_data["successful_usage"]
                perf.avg_quality = perf_data["avg_quality"]
                perf.avg_time_ms = perf_data["avg_time_ms"]
                self.template_performance[template] = perf

        if "learning_history" in data:
            self.learning_history = data["learning_history"]

        if "statistics" in data:
            stats = data["statistics"]
            self.total_generations = stats.get("total_generations", 0)
            self.total_time_ms = stats.get("total_time_ms", 0.0)

        logger.info(f"导入了 {len(data.get('template_performance', {}))} 个模板的学习结果")


# 测试函数
def test_dynamic_template_generator():
    """测试动态模板生成器"""
    print("=" * 60)
    print("测试 DynamicTemplateGenerator")
    print("=" * 60)

    generator = DynamicTemplateGenerator()

    # 测试用例1：完整的查询
    print("\n测试用例1: 完整查询")
    parsed_result1 = ParsedResult(
        direction="north",
        color="red",
        object="building",
        relation="near",
        distance=10.0,
        completeness_score=0.95,
        confidence_scores={
            "direction": 0.9,
            "color": 0.95,
            "object": 0.92,
            "relation": 0.8,
            "distance": 0.7
        }
    )

    templates1 = generator.generate(parsed_result1, n_variants=5)
    print(f"生成了 {len(templates1)} 个变体")

    for i, variant in enumerate(templates1, 1):
        print(f"  {i}. [{variant.template_type}] {variant.filled_text}")
        print(f"     质量: {variant.quality_score:.2f}, 兼容性: {variant.compatibility_level}")

    # 测试用例2：部分查询
    print("\n测试用例2: 部分查询（只有方向和对象）")
    parsed_result2 = ParsedResult(
        direction="east",
        object="car",
        completeness_score=0.5,
        confidence_scores={
            "direction": 0.8,
            "object": 0.7
        }
    )

    templates2 = generator.generate(parsed_result2, n_variants=4)
    print(f"生成了 {len(templates2)} 个变体")

    for i, variant in enumerate(templates2, 1):
        print(f"  {i}. [{variant.template_type}] {variant.filled_text}")
        print(f"     质量: {variant.quality_score:.2f}, 兼容性: {variant.compatibility_level}")

    # 测试用例3：最低完整性查询
    print("\n测试用例3: 最低完整性查询（只有对象）")
    parsed_result3 = ParsedResult(
        object="parking",
        completeness_score=0.3,
        confidence_scores={
            "object": 0.6
        }
    )

    templates3 = generator.generate(parsed_result3, n_variants=3)
    print(f"生成了 {len(templates3)} 个变体")

    for i, variant in enumerate(templates3, 1):
        print(f"  {i}. [{variant.template_type}] {variant.filled_text}")
        print(f"     质量: {variant.quality_score:.2f}, 兼容性: {variant.compatibility_level}")

    # 测试最佳模板选择
    print("\n测试最佳模板选择:")

    for mode in ["fast", "balanced", "high_quality", "compatible"]:
        best = generator.get_best_template(parsed_result1, mode=mode)
        if best:
            print(f"  {mode}: {best.filled_text}")

    # 性能统计
    print(f"\n{'='*60}")
    print("性能统计")
    print(f"{'='*60}")
    stats = generator.get_performance_stats()

    print(f"总生成次数: {stats['total_generations']}")
    print(f"总用时: {stats['total_time_ms']:.1f}ms")
    print(f"平均用时: {stats['avg_time_ms']:.1f}ms")
    print(f"独特模板数: {stats['unique_templates']}")

    if stats['type_stats']:
        print(f"\n按类型统计:")
        for type_key, type_stats in stats['type_stats'].items():
            print(f"  {type_key}:")
            print(f"    使用次数: {type_stats['total_usage']}")
            print(f"    成功率: {type_stats['success_rate']:.2f}")
            print(f"    平均质量: {type_stats['avg_quality']:.2f}")

    if stats['top_templates']:
        print(f"\n最常用模板（前5）:")
        for i, t in enumerate(stats['top_templates'][:5], 1):
            print(f"  {i}. {t['template']}")
            print(f"     类型: {t['type']}, 使用: {t['usage']}, 成功率: {t['success_rate']:.2f}")

    # 模板家族
    print(f"\n{'='*60}")
    print("模板家族分类")
    print(f"{'='*60}")

    families = generator.get_template_families()
    for family_name, family_info in families.items():
        print(f"\n{family_name}: {family_info['description']}")
        print(f"  模板数量: {family_info['count']}")
        print(f"  示例: {family_info['templates'][0] if family_info['templates'] else 'None'}")


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)

    # 运行测试
    test_dynamic_template_generator()
