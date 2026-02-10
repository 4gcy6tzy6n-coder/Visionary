"""
配置管理器 - Text2Loc增强版

统一管理系统配置，支持多环境配置
"""

import yaml
import json
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SystemConfig:
    """系统配置"""
    # 模型配置
    nlu_model: str = "qwen3-vl:2b"
    embedding_model: str = "qwen3-embedding:0.6b"
    ollama_url: str = "http://localhost:11434"
    api_timeout: int = 30

    # 功能开关
    enable_nlu: bool = True
    enable_vector_search: bool = True
    enable_enhanced_color: bool = True
    enable_enhanced_object: bool = True
    enable_hybrid_retrieval: bool = True

    # 性能配置
    cache_enabled: bool = True
    max_cache_size: int = 1000
    batch_size: int = 10
    mock_mode: bool = False

    # 置信度阈值
    nlu_confidence_threshold: float = 0.7
    direction_confidence_threshold: float = 0.7
    color_confidence_threshold: float = 0.6
    object_confidence_threshold: float = 0.6
    retrieval_confidence_threshold: float = 0.7

    # 回退配置
    fallback_enabled: bool = True
    fallback_on_error: bool = True
    fallback_on_low_confidence: bool = True

    # 索引配置
    index_type: str = "FlatL2"
    vector_weight: float = 0.6
    template_weight: float = 0.4

    # 空间关系配置
    distance_threshold: float = 5.0
    angle_threshold: float = 45.0

    # 日志配置
    log_level: str = "INFO"
    log_file: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemConfig':
        """从字典创建配置"""
        return cls(**data)


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器

        Args:
            config_path: 配置文件路径
        """
        self.config = SystemConfig()
        self.config_path = config_path
        self.config_history = []

        if config_path and Path(config_path).exists():
            self.load_from_file(config_path)
        else:
            logger.info("使用默认配置")

    def load_from_file(self, config_path: str):
        """
        从文件加载配置

        Args:
            config_path: 配置文件路径
        """
        try:
            path = Path(config_path)

            if path.suffix.lower() == '.yaml' or path.suffix.lower() == '.yml':
                with open(path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
            elif path.suffix.lower() == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                logger.error(f"不支持的配置文件格式: {path.suffix}")
                return

            if data:
                self.config = SystemConfig.from_dict(data)
                self.config_history.append(("load", self.config.to_dict()))
                logger.info(f"配置已从 {config_path} 加载")
            else:
                logger.warning(f"配置文件为空: {config_path}")

        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            raise

    def save_to_file(self, config_path: str, format: str = "yaml"):
        """
        保存配置到文件

        Args:
            config_path: 文件路径
            format: 文件格式（yaml或json）
        """
        try:
            data = self.config.to_dict()

            if format.lower() == "yaml":
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            elif format.lower() == "json":
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"不支持的格式: {format}")

            self.config_history.append(("save", self.config.to_dict()))
            logger.info(f"配置已保存到 {config_path}")

        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            raise

    def update(self, **kwargs):
        """
        更新配置

        Args:
            kwargs: 配置参数
        """
        old_config = self.config.to_dict()

        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"更新配置: {key}={value}")
            else:
                logger.warning(f"未知配置项: {key}")

        self.config_history.append(("update", old_config, self.config.to_dict()))

    def reset(self):
        """重置为默认配置"""
        self.config = SystemConfig()
        self.config_history.append(("reset", None))
        logger.info("配置已重置为默认值")

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值

        Args:
            key: 配置键
            default: 默认值

        Returns:
            配置值
        """
        return getattr(self.config, key, default)

    def get_all(self) -> Dict[str, Any]:
        """获取所有配置"""
        return self.config.to_dict()

    def get_config_for_module(self, module_name: str) -> Dict[str, Any]:
        """
        获取模块特定配置

        Args:
            module_name: 模块名称

        Returns:
            模块配置
        """
        all_config = self.config.to_dict()

        if module_name == "nlu":
            return {
                "model": all_config["nlu_model"],
                "ollama_url": all_config["ollama_url"],
                "timeout": all_config["api_timeout"],
                "confidence_threshold": all_config["nlu_confidence_threshold"],
                "cache_enabled": all_config["cache_enabled"],
                "mock_mode": all_config["mock_mode"],
                "fallback_enabled": all_config["fallback_enabled"],
            }
        elif module_name == "vector_db":
            return {
                "model": all_config["embedding_model"],
                "ollama_url": all_config["ollama_url"],
                "timeout": all_config["api_timeout"],
                "index_type": all_config["index_type"],
                "cache_enabled": all_config["cache_enabled"],
                "max_cache_size": all_config["max_cache_size"],
                "mock_mode": all_config["mock_mode"],
            }
        elif module_name == "pointcloud":
            return {
                "enable_enhanced_color": all_config["enable_enhanced_color"],
                "enable_enhanced_object": all_config["enable_enhanced_object"],
                "distance_threshold": all_config["distance_threshold"],
                "angle_threshold": all_config["angle_threshold"],
            }
        elif module_name == "integration":
            return {
                "enable_nlu": all_config["enable_nlu"],
                "enable_vector_search": all_config["enable_vector_search"],
                "enable_enhanced_color": all_config["enable_enhanced_color"],
                "enable_enhanced_object": all_config["enable_enhanced_object"],
                "enable_hybrid_retrieval": all_config["enable_hybrid_retrieval"],
                "confidence_threshold": min(
                    all_config["nlu_confidence_threshold"],
                    all_config["direction_confidence_threshold"],
                    all_config["color_confidence_threshold"],
                    all_config["object_confidence_threshold"],
                    all_config["retrieval_confidence_threshold"]
                ),
                "fallback_enabled": all_config["fallback_enabled"],
                "fallback_on_error": all_config["fallback_on_error"],
                "fallback_on_low_confidence": all_config["fallback_on_low_confidence"],
            }
        else:
            return all_config

    def validate(self) -> Tuple[bool, list]:
        """
        验证配置

        Returns:
            (是否有效, 错误列表)
        """
        errors = []

        # 验证模型配置
        if not self.config.nlu_model:
            errors.append("nlu_model不能为空")
        if not self.config.embedding_model:
            errors.append("embedding_model不能为空")
        if not self.config.ollama_url:
            errors.append("ollama_url不能为空")

        # 验证阈值
        if not (0.0 <= self.config.nlu_confidence_threshold <= 1.0):
            errors.append("nlu_confidence_threshold必须在0-1之间")
        if not (0.0 <= self.config.direction_confidence_threshold <= 1.0):
            errors.append("direction_confidence_threshold必须在0-1之间")
        if not (0.0 <= self.config.color_confidence_threshold <= 1.0):
            errors.append("color_confidence_threshold必须在0-1之间")
        if not (0.0 <= self.config.object_confidence_threshold <= 1.0):
            errors.append("object_confidence_threshold必须在0-1之间")
        if not (0.0 <= self.config.retrieval_confidence_threshold <= 1.0):
            errors.append("retrieval_confidence_threshold必须在0-1之间")

        # 验证权重
        if not (0.0 <= self.config.vector_weight <= 1.0):
            errors.append("vector_weight必须在0-1之间")
        if not (0.0 <= self.config.template_weight <= 1.0):
            errors.append("template_weight必须在0-1之间")
        if abs(self.config.vector_weight + self.config.template_weight - 1.0) > 0.01:
            errors.append("vector_weight + template_weight必须等于1.0")

        # 验证性能配置
        if self.config.batch_size <= 0:
            errors.append("batch_size必须大于0")
        if self.config.max_cache_size <= 0:
            errors.append("max_cache_size必须大于0")

        is_valid = len(errors) == 0
        return is_valid, errors

    def get_history(self) -> list:
        """获取配置历史"""
        return self.config_history

    def export_to_default(self):
        """导出为默认配置文件"""
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)

        # 导出YAML
        yaml_path = config_dir / "default.yaml"
        self.save_to_file(str(yaml_path), "yaml")

        # 导出JSON
        json_path = config_dir / "default.json"
        self.save_to_file(str(json_path), "json")

        logger.info(f"默认配置已导出到 {config_dir}")

    def create_environment_config(self, env_name: str, overrides: Dict[str, Any]) -> str:
        """
        创建环境特定配置

        Args:
            env_name: 环境名称（dev, test, prod）
            overrides: 覆盖配置

        Returns:
            配置文件路径
        """
        # 创建配置副本
        env_config = SystemConfig.from_dict(self.config.to_dict())

        # 应用覆盖
        for key, value in overrides.items():
            if hasattr(env_config, key):
                setattr(env_config, key, value)

        # 保存配置
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)

        config_path = config_dir / f"{env_name}.yaml"
        data = env_config.to_dict()

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"环境配置已创建: {config_path}")
        return str(config_path)

    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        config = self.config.to_dict()

        return {
            "models": {
                "nlu": config["nlu_model"],
                "embedding": config["embedding_model"],
                "ollama_url": config["ollama_url"],
            },
            "features": {
                "nlu": config["enable_nlu"],
                "vector_search": config["enable_vector_search"],
                "enhanced_color": config["enable_enhanced_color"],
                "enhanced_object": config["enable_enhanced_object"],
                "hybrid_retrieval": config["enable_hybrid_retrieval"],
            },
            "performance": {
                "cache_enabled": config["cache_enabled"],
                "max_cache_size": config["max_cache_size"],
                "batch_size": config["batch_size"],
                "mock_mode": config["mock_mode"],
            },
            "thresholds": {
                "nlu": config["nlu_confidence_threshold"],
                "direction": config["direction_confidence_threshold"],
                "color": config["color_confidence_threshold"],
                "object": config["object_confidence_threshold"],
                "retrieval": config["retrieval_confidence_threshold"],
            },
            "fallback": {
                "enabled": config["fallback_enabled"],
                "on_error": config["fallback_on_error"],
                "on_low_confidence": config["fallback_on_low_confidence"],
            },
        }
