"""
增强点云处理模块 - Text2Loc增强版

提供扩展的颜色识别、开放词汇对象识别和空间关系提取功能
"""

from .colors import EnhancedColorMapper, ColorLibrary
from .open_vocab import OpenVocabObjectIdentifier
from .spatial import SpatialRelationExtractor
from .feature_fusion import FeatureFusion

__all__ = [
    "EnhancedColorMapper",
    "ColorLibrary",
    "OpenVocabObjectIdentifier",
    "SpatialRelationExtractor",
    "FeatureFusion",
]
