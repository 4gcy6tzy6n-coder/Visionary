"""
模型配置管理 API
支持 DeepSeek、Ollama、OpenAI 兼容接口
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import requests
import time

logger = logging.getLogger(__name__)

# 配置文件路径
CONFIG_FILE = os.path.join(os.path.dirname(__file__), '..', 'config', 'model_config.json')


@dataclass
class ModelConfig:
    """模型配置"""
    provider: str = "deepseek"  # deepseek, ollama, openai
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    timeout: int = 30
    temperature: float = 0.7
    max_tokens: int = 500
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self.config = ModelConfig()
        self._load_config()
    
    def _load_config(self):
        """从文件加载配置"""
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.config = ModelConfig.from_dict(data)
                    logger.info(f"✅ 已加载模型配置: {self.config.provider}")
            else:
                # 使用环境变量作为默认值
                self._load_from_env()
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            self._load_from_env()
    
    def _load_from_env(self):
        """从环境变量加载配置"""
        # 检查 DeepSeek
        if os.environ.get("DEEPSEEK_API_KEY"):
            self.config.provider = "deepseek"
            self.config.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
            self.config.base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            self.config.model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
            logger.info("✅ 从环境变量加载 DeepSeek 配置")
        # 检查 Ollama
        elif os.environ.get("OLLAMA_URL"):
            self.config.provider = "ollama"
            self.config.base_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
            self.config.model = os.environ.get("OLLAMA_MODEL", "qwen3-vl:2b")
            logger.info("✅ 从环境变量加载 Ollama 配置")
    
    def _save_config(self):
        """保存配置到文件"""
        try:
            os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"✅ 配置已保存: {self.config.provider}")
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            raise
    
    def get_config(self) -> Dict[str, Any]:
        """获取当前配置（隐藏 API Key）"""
        config_dict = self.config.to_dict()
        # 隐藏 API Key
        if config_dict.get("api_key"):
            config_dict["api_key"] = config_dict["api_key"][:8] + "..."
        return config_dict
    
    def get_full_config(self) -> ModelConfig:
        """获取完整配置"""
        return self.config
    
    def update_config(self, config_data: Dict[str, Any]) -> bool:
        """更新配置"""
        try:
            # 验证必需的字段
            provider = config_data.get("provider")
            if provider not in ["deepseek", "ollama", "openai"]:
                raise ValueError(f"不支持的提供商: {provider}")
            
            # 更新配置
            self.config.provider = provider
            
            if "api_key" in config_data:
                self.config.api_key = config_data["api_key"]
            if "url" in config_data:
                self.config.base_url = config_data["url"]
            if "model" in config_data:
                self.config.model = config_data["model"]
            if "timeout" in config_data:
                self.config.timeout = config_data["timeout"]
            if "temperature" in config_data:
                self.config.temperature = config_data["temperature"]
            if "max_tokens" in config_data:
                self.config.max_tokens = config_data["max_tokens"]
            
            # 设置默认值
            if not self.config.base_url:
                if provider == "deepseek":
                    self.config.base_url = "https://api.deepseek.com"
                elif provider == "ollama":
                    self.config.base_url = "http://localhost:11434"
            
            if not self.config.model:
                if provider == "deepseek":
                    self.config.model = "deepseek-chat"
                elif provider == "ollama":
                    self.config.model = "qwen3-vl:2b"
            
            self._save_config()
            return True
            
        except Exception as e:
            logger.error(f"更新配置失败: {e}")
            raise
    
    def reset_config(self):
        """重置配置"""
        self.config = ModelConfig()
        if os.path.exists(CONFIG_FILE):
            os.remove(CONFIG_FILE)
        logger.info("✅ 配置已重置")
    
    def test_connection(self) -> Dict[str, Any]:
        """测试连接"""
        start_time = time.time()
        
        try:
            if self.config.provider == "deepseek":
                return self._test_deepseek()
            elif self.config.provider == "ollama":
                return self._test_ollama()
            elif self.config.provider == "openai":
                return self._test_openai()
            else:
                return {"success": False, "error": "未知的提供商"}
                
        except Exception as e:
            logger.error(f"测试连接失败: {e}")
            return {"success": False, "error": str(e)}
    
    def _test_deepseek(self) -> Dict[str, Any]:
        """测试 DeepSeek 连接"""
        try:
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.config.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            }
            
            start = time.time()
            response = requests.post(
                f"{self.config.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.config.timeout
            )
            elapsed = (time.time() - start) * 1000
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "model": data.get("model", self.config.model),
                    "response_time": elapsed
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _test_ollama(self) -> Dict[str, Any]:
        """测试 Ollama 连接"""
        try:
            # 首先检查服务是否运行
            start = time.time()
            response = requests.get(
                f"{self.config.base_url}/api/tags",
                timeout=self.config.timeout
            )
            elapsed = (time.time() - start) * 1000
            
            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"无法连接到 Ollama 服务 (HTTP {response.status_code})"
                }
            
            # 检查模型是否存在
            data = response.json()
            models = [m.get("name") for m in data.get("models", [])]
            
            if self.config.model not in models:
                return {
                    "success": False,
                    "error": f"模型 '{self.config.model}' 不存在。可用模型: {', '.join(models[:5])}..."
                }
            
            # 测试生成
            test_payload = {
                "model": self.config.model,
                "prompt": "Hello",
                "stream": False,
                "options": {"num_predict": 10}
            }
            
            start = time.time()
            gen_response = requests.post(
                f"{self.config.base_url}/api/generate",
                json=test_payload,
                timeout=self.config.timeout
            )
            gen_elapsed = (time.time() - start) * 1000
            
            if gen_response.status_code == 200:
                return {
                    "success": True,
                    "model": self.config.model,
                    "response_time": gen_elapsed,
                    "available_models": models[:10]
                }
            else:
                return {
                    "success": False,
                    "error": f"生成测试失败 (HTTP {gen_response.status_code})"
                }
                
        except requests.exceptions.ConnectionError:
            return {"success": False, "error": "无法连接到 Ollama 服务，请检查服务是否运行"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _test_openai(self) -> Dict[str, Any]:
        """测试 OpenAI 兼容接口"""
        try:
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.config.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            }
            
            start = time.time()
            response = requests.post(
                f"{self.config.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.config.timeout
            )
            elapsed = (time.time() - start) * 1000
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "model": data.get("model", self.config.model),
                    "response_time": elapsed
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}


# 全局配置管理器实例
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """获取配置管理器实例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
