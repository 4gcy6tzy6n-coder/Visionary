"""
扩展颜色识别系统 - Text2Loc增强版

从8种预定义颜色扩展到100+种常见颜色，支持HSV颜色空间分析
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ColorResult:
    """颜色识别结果"""
    rgb: Tuple[int, int, int]  # RGB值
    hsv: Tuple[float, float, float]  # HSV值
    color_name: str  # 颜色名称
    confidence: float  # 置信度 (0.0-1.0)
    alternatives: List[str]  # 备选颜色名称


class ColorSpace(Enum):
    """颜色空间"""
    RGB = "rgb"
    HSV = "hsv"
    LAB = "lab"


class ColorLibrary:
    """扩展颜色库"""

    # 扩展的颜色库（100+种常见颜色）
    EXPANDED_COLORS = {
        # 红色系
        "red": (255, 0, 0),
        "dark_red": (139, 0, 0),
        "light_red": (255, 102, 102),
        "crimson": (220, 20, 60),
        "scarlet": (255, 36, 0),
        "maroon": (128, 0, 0),
        "burgundy": (128, 0, 32),
        "coral": (255, 127, 80),
        "salmon": (250, 128, 114),
        "tomato": (255, 99, 71),
        "cherry": (222, 49, 99),
        
        # 橙色系
        "orange": (255, 165, 0),
        "dark_orange": (255, 140, 0),
        "light_orange": (255, 200, 100),
        "amber": (255, 191, 0),
        "golden": (255, 215, 0),
        "peach": (255, 218, 185),
        "apricot": (251, 206, 177),
        "tangerine": (255, 165, 0),
        
        # 黄色系
        "yellow": (255, 255, 0),
        "dark_yellow": (204, 204, 0),
        "light_yellow": (255, 255, 153),
        "lemon": (255, 250, 205),
        "gold": (255, 215, 0),
        "mustard": (255, 219, 88),
        "banana": (255, 255, 153),
        "corn": (251, 236, 93),
        
        # 绿色系
        "green": (0, 255, 0),
        "dark_green": (0, 100, 0),
        "light_green": (144, 238, 144),
        "lime": (50, 205, 50),
        "forest_green": (34, 139, 34),
        "olive": (128, 128, 0),
        "emerald": (0, 201, 87),
        "mint": (152, 255, 152),
        "sea_green": (84, 255, 159),
        "spring_green": (0, 255, 127),
        "chartreuse": (127, 255, 0),
        "pale_green": (152, 251, 152),
        
        # 青色系
        "cyan": (0, 255, 255),
        "dark_cyan": (0, 139, 139),
        "light_cyan": (224, 255, 255),
        "teal": (0, 128, 128),
        "aqua": (0, 255, 255),
        "turquoise": (64, 224, 208),
        "aquamarine": (127, 255, 212),
        
        # 蓝色系
        "blue": (0, 0, 255),
        "dark_blue": (0, 0, 139),
        "light_blue": (173, 216, 230),
        "navy": (0, 0, 128),
        "royal_blue": (65, 105, 225),
        "sky_blue": (135, 206, 235),
        "steel_blue": (70, 130, 180),
        "dodger_blue": (30, 144, 255),
        "cornflower": (100, 149, 237),
        "midnight": (25, 25, 112),
        "slate_blue": (106, 90, 205),
        
        # 紫色系
        "purple": (128, 0, 128),
        "dark_purple": (75, 0, 130),
        "light_purple": (216, 191, 216),
        "violet": (238, 130, 238),
        "indigo": (75, 0, 130),
        "magenta": (255, 0, 255),
        "plum": (221, 160, 221),
        "lavender": (230, 230, 250),
        "orchid": (218, 112, 214),
        "thistle": (216, 191, 216),
        
        # 粉色系
        "pink": (255, 192, 203),
        "dark_pink": (255, 20, 147),
        "light_pink": (255, 182, 193),
        "hot_pink": (255, 105, 180),
        "rose": (255, 0, 127),
        "blush": (222, 93, 131),
        
        # 棕色系
        "brown": (165, 42, 42),
        "dark_brown": (101, 67, 33),
        "light_brown": (181, 136, 99),
        "chocolate": (210, 105, 30),
        "sienna": (160, 82, 45),
        "saddle_brown": (139, 69, 19),
        "tan": (210, 180, 140),
        "beige": (245, 245, 220),
        "wheat": (245, 222, 179),
        "burlywood": (222, 184, 135),
        
        # 灰色系
        "gray": (128, 128, 128),
        "dark_gray": (64, 64, 64),
        "light_gray": (211, 211, 211),
        "dim_gray": (105, 105, 105),
        "slate_gray": (112, 128, 144),
        "charcoal": (54, 69, 79),
        "silver": (192, 192, 192),
        "platinum": (229, 228, 226),
        
        # 白色系
        "white": (255, 255, 255),
        "ivory": (255, 255, 240),
        "eggshell": (240, 234, 214),
        "cream": (255, 253, 208),
        
        # 黑色系
        "black": (0, 0, 0),
        "jet": (52, 52, 52),
        "obsidian": (25, 25, 25),
        
        # 原Text2Loc颜色（保持兼容）
        "dark-green": (0, 100, 0),
        "gray-green": (100, 120, 100),
        "bright-gray": (200, 200, 200),
        "beige": (245, 245, 220),
    }

    # 颜色别名映射
    COLOR_ALIASES = {
        "red": ["scarlet", "crimson", "cherry"],
        "orange": ["tangerine"],
        "yellow": ["lemon", "banana"],
        "green": ["lime", "emerald", "mint"],
        "blue": ["sky", "navy", "royal"],
        "purple": ["violet", "indigo", "magenta"],
        "pink": ["rose", "blush"],
        "brown": ["chocolate", "sienna", "tan"],
        "gray": ["grey", "silver", "charcoal"],
        "black": ["jet", "obsidian"],
        "white": ["ivory", "cream", "eggshell"],
    }

    @classmethod
    def get_color_rgb(cls, color_name: str) -> Optional[Tuple[int, int, int]]:
        """
        获取颜色的RGB值

        Args:
            color_name: 颜色名称

        Returns:
            RGB元组，如果不存在则返回None
        """
        # 直接查找
        if color_name in cls.EXPANDED_COLORS:
            return cls.EXPANDED_COLORS[color_name]

        # 查找别名
        for main_color, aliases in cls.COLOR_ALIASES.items():
            if color_name in aliases:
                return cls.EXPANDED_COLORS.get(main_color)

        return None

    @classmethod
    def get_all_color_names(cls) -> List[str]:
        """获取所有颜色名称"""
        return list(cls.EXPANDED_COLORS.keys())

    @classmethod
    def find_similar_colors(cls, rgb: Tuple[int, int, int], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        查找最相似的颜色

        Args:
            rgb: RGB值
            top_k: 返回前k个

        Returns:
            [(颜色名称, 相似度), ...]
        """
        similarities = []

        for color_name, color_rgb in cls.EXPANDED_COLORS.items():
            # 计算欧氏距离
            distance = np.sqrt(
                (rgb[0] - color_rgb[0]) ** 2 +
                (rgb[1] - color_rgb[1]) ** 2 +
                (rgb[2] - color_rgb[2]) ** 2
            )

            # 转换为相似度（0-1）
            max_distance = np.sqrt(3 * 255 ** 2)
            similarity = 1.0 - (distance / max_distance)

            similarities.append((color_name, similarity))

        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]


class EnhancedColorMapper:
    """增强颜色映射器"""

    def __init__(self, use_hsv: bool = True, use_lab: bool = False):
        """
        初始化颜色映射器

        Args:
            use_hsv: 是否使用HSV颜色空间
            use_lab: 是否使用LAB颜色空间
        """
        self.use_hsv = use_hsv
        self.use_lab = use_lab

        logger.info(f"增强颜色映射器初始化: HSV={use_hsv}, LAB={use_lab}")

    def rgb_to_hsv(self, rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """
        RGB转HSV

        Args:
            rgb: RGB值 (0-255)

        Returns:
            HSV值 (H: 0-360, S: 0-100, V: 0-100)
        """
        r, g, b = [x / 255.0 for x in rgb]

        max_val = max(r, g, b)
        min_val = min(r, g, b)
        delta = max_val - min_val

        # 计算H
        if delta == 0:
            h = 0
        elif max_val == r:
            h = 60 * (((g - b) / delta) % 6)
        elif max_val == g:
            h = 60 * (((b - r) / delta) + 2)
        else:
            h = 60 * (((r - g) / delta) + 4)

        # 计算S
        if max_val == 0:
            s = 0
        else:
            s = (delta / max_val) * 100

        # 计算V
        v = max_val * 100

        return (h, s, v)

    def rgb_to_lab(self, rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """
        RGB转LAB（简化版）

        Args:
            rgb: RGB值 (0-255)

        Returns:
            LAB值
        """
        # 简化的RGB到LAB转换
        r, g, b = rgb

        # RGB到XYZ（简化）
        x = 0.412453 * r + 0.357580 * g + 0.180423 * b
        y = 0.212671 * r + 0.715160 * g + 0.072169 * b
        z = 0.019334 * r + 0.119193 * g + 0.950227 * b

        # XYZ到LAB（简化）
        # 这里使用简化版本，实际应该使用完整的转换公式
        l = 116 * y / 100 - 16 if y > 0.008856 else 903.3 * y
        a = 500 * (x - y)
        b = 200 * (y - z)

        return (l, a, b)

    def get_color_name(self,
                      rgb: Tuple[int, int, int],
                      confidence_threshold: float = 0.7) -> ColorResult:
        """
        获取颜色名称（增强版）

        Args:
            rgb: RGB值
            confidence_threshold: 置信度阈值

        Returns:
            颜色识别结果
        """
        # 转换为HSV
        hsv = self.rgb_to_hsv(rgb)

        # 查找最相似的颜色
        similar_colors = ColorLibrary.find_similar_colors(rgb, top_k=5)

        if not similar_colors:
            # 如果没有匹配，返回默认值
            return ColorResult(
                rgb=rgb,
                hsv=hsv,
                color_name="unknown",
                confidence=0.0,
                alternatives=[]
            )

        # 获取最佳匹配
        best_color, best_similarity = similar_colors[0]

        # 计算置信度（基于相似度）
        confidence = best_similarity

        # 如果使用HSV，可以进一步提高准确性
        if self.use_hsv:
            # 检查色调区间
            h, s, v = hsv

            # 如果饱和度很低，可能是灰色
            if s < 20:
                # 在低饱和度下，优先选择灰色系
                if best_similarity < 0.7:
                    # 重新在灰色系中查找
                    gray_colors = [c for c in ColorLibrary.get_all_color_names() if "gray" in c or "grey" in c]
                    if gray_colors:
                        best_color = "gray"
                        confidence = 0.8

        # 获取备选颜色
        alternatives = [color for color, sim in similar_colors[1:5] if sim > 0.6]

        # 如果最佳匹配的置信度低于阈值，标记为不确定
        if confidence < confidence_threshold:
            logger.warning(f"颜色识别置信度低: {rgb}, 最佳匹配: {best_color}, 置信度: {confidence:.3f}")

        return ColorResult(
            rgb=rgb,
            hsv=hsv,
            color_name=best_color,
            confidence=confidence,
            alternatives=alternatives
        )

    def batch_get_color_names(self, rgb_list: List[Tuple[int, int, int]]) -> List[ColorResult]:
        """
        批量获取颜色名称

        Args:
            rgb_list: RGB值列表

        Returns:
            颜色识别结果列表
        """
        results = []

        for rgb in rgb_list:
            result = self.get_color_name(rgb)
            results.append(result)

        return results

    def get_color_statistics(self, color_results: List[ColorResult]) -> Dict[str, Any]:
        """
        获取颜色统计信息

        Args:
            color_results: 颜色识别结果列表

        Returns:
            统计信息
        """
        if not color_results:
            return {}

        # 统计颜色分布
        color_counts = {}
        total_confidence = 0.0
        high_confidence_count = 0

        for result in color_results:
            color_name = result.color_name
            color_counts[color_name] = color_counts.get(color_name, 0) + 1
            total_confidence += result.confidence

            if result.confidence >= 0.8:
                high_confidence_count += 1

        avg_confidence = total_confidence / len(color_results)

        return {
            "total_colors": len(color_results),
            "unique_colors": len(color_counts),
            "color_distribution": color_counts,
            "average_confidence": avg_confidence,
            "high_confidence_rate": high_confidence_count / len(color_results),
            "most_common_color": max(color_counts.items(), key=lambda x: x[1])[0] if color_counts else None
        }
