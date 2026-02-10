"""
集成适配器模块 - Text2Loc增强版

提供与原有Text2Loc系统的集成适配器
"""

from .adapter import Text2LocAdapter, IntegrationConfig
from .config_manager import ConfigManager
from .format_converter import FormatConverter, NewFormat, OldFormat

__all__ = [
    "Text2LocAdapter",
    "IntegrationConfig",
    "ConfigManager",
    "FormatConverter",
    "NewFormat",
    "OldFormat",
]
