"""
Text2Loc增强版核心模块

提供自然语言理解、向量检索、增强点云处理和系统集成功能
"""

from .nlu.engine import NLUEngine, NLUConfig, NLUResult
from .vector_db import (
    EmbeddingClient, EmbeddingResult, EmbeddingModel,
    FaissIndex, HybridRetriever, RetrievalResult, VectorStore
)
from .pointcloud import (
    EnhancedColorMapper, ColorLibrary, OpenVocabObjectIdentifier,
    SpatialRelationExtractor, FeatureFusion
)
from .integration import (
    Text2LocAdapter, IntegrationConfig, ConfigManager,
    FormatConverter, NewFormat, OldFormat
)

__all__ = [
    # NLU模块
    "NLUEngine",
    "NLUConfig",
    "NLUResult",
    
    # 向量检索模块
    "EmbeddingClient",
    "EmbeddingResult",
    "EmbeddingModel",
    "FaissIndex",
    "HybridRetriever",
    "RetrievalResult",
    "VectorStore",
    
    # 增强点云处理模块
    "EnhancedColorMapper",
    "ColorLibrary",
    "OpenVocabObjectIdentifier",
    "SpatialRelationExtractor",
    "FeatureFusion",
    
    # 集成适配器模块
    "Text2LocAdapter",
    "IntegrationConfig",
    "ConfigManager",
    "FormatConverter",
    "NewFormat",
    "OldFormat",
]
