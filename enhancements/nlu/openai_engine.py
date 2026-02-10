"""
OpenAI 兼容 NLU 引擎
支持 OpenAI 及兼容接口（如 OneAPI、NewAPI 等）
"""

import json
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OpenAIConfig:
    """OpenAI 配置"""
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-3.5-turbo"
    timeout: int = 30
    temperature: float = 0.7
    max_tokens: int = 500
    enabled: bool = True


@dataclass
class NLUResult:
    """NLU 解析结果"""
    text: str
    components: Dict[str, Any]
    confidence: float
    model: str = ""
    parse_time: float = 0.0
    enhanced_used: bool = False
    intent: str = ""
    need_clarification: bool = False
    clarification_question: Optional[str] = None
    error: Optional[str] = None


class OpenAINLUEngine:
    """
    OpenAI 兼容 NLU 引擎
    
    支持 OpenAI 及兼容接口进行自然语言理解
    """
    
    SYSTEM_PROMPT = """你是一个专业的位置描述解析助手。你的任务是将用户的中英文位置描述转换为结构化的标准格式。

# 输出格式
请严格按照以下 JSON 格式输出：
{
    "direction": "方向词(东/南/西/北/东北/西北/东南/西南/无)",
    "color": "颜色词(红/蓝/绿/黄/灰/黑/白/紫/橙/棕/无)",
    "object": "目标对象(大楼/建筑/树/山/路/灯/车/桥/河/无)",
    "relation": "关系词(靠近/旁边/对面/附近/在...北边/在...东边/无)",
    "distance": "距离词(近/远/附近/无)",
    "confidence": 0.85,
    "need_clarification": false
}

# 方向识别规则
- "北/北边/北面/向北/朝北" → "north"
- "南/南边/南面/向南/朝南" → "south"
- "东/东边/东面/向东/朝东" → "east"
- "西/西边/西面/向西/朝西" → "west"
- "东北/东北边" → "northeast"
- "西北/西北边" → "northwest"
- "东南/东南边" → "southeast"
- "西南/西南边" → "southwest"
- 背朝太阳(中午) → 方向是 "south"（背朝南=朝北）
- 背朝太阳(早晨) → 方向是 "west"（背朝东=朝西）
- 面向太阳(中午) → 方向是 "north"（面朝南=朝南）

# 颜色识别
支持: 红、蓝、绿、黄、灰、黑、白、紫、橙、棕

# 对象识别
支持: 大楼、建筑、树、山、路、灯、车、桥、河等

# 示例
输入: "我在红色大楼的北边"
输出: {"direction": "north", "color": "red", "object": "building", "relation": "near", "distance": "none", "confidence": 0.95, "need_clarification": false}

输入: "背朝太阳，周围有黑色的房子"
输出: {"direction": "south", "color": "black", "object": "house", "relation": "near", "distance": "none", "confidence": 0.9, "need_clarification": false}

只输出 JSON，不要其他内容。"""
    
    def __init__(self, config: OpenAIConfig):
        self.config = config
        self.session = None
    
    def _get_session(self):
        """获取请求会话"""
        if self.session is None:
            import requests
            self.session = requests.Session()
            self.session.headers.update({
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            })
        return self.session
    
    def parse(self, text: str) -> NLUResult:
        """
        解析位置描述
        
        Args:
            text: 用户输入的自然语言
            
        Returns:
            NLUResult: 解析结果
        """
        start_time = time.time()
        
        if not self.config.enabled:
            return NLUResult(
                text=text,
                components={"error": "OpenAI 引擎未启用"},
                confidence=0.0,
                error="OpenAI disabled"
            )
        
        if not self.config.api_key:
            return NLUResult(
                text=text,
                components={"error": "未配置 API Key"},
                confidence=0.0,
                error="No API key configured"
            )
        
        try:
            session = self._get_session()
            
            payload = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": text}
                ],
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature
            }
            
            # 某些兼容接口需要额外的参数
            if "response_format" not in payload:
                payload["response_format"] = {"type": "json_object"}
            
            response = session.post(
                f"{self.config.base_url}/chat/completions",
                json=payload,
                timeout=self.config.timeout
            )
            
            response.raise_for_status()
            data = response.json()
            
            content = data["choices"][0]["message"]["content"]
            
            # 解析 JSON
            result = self._extract_json(content)
            
            parse_time = time.time() - start_time
            
            return NLUResult(
                text=text,
                components={
                    "direction": result.get("direction", "none"),
                    "color": result.get("color", "none"),
                    "object": result.get("object", "none"),
                    "relation": result.get("relation", "none"),
                    "distance": result.get("distance", "none"),
                    "confidence": result.get("confidence", 0.8),
                    "need_clarification": result.get("need_clarification", False)
                },
                confidence=result.get("confidence", 0.8),
                model=self.config.model,
                parse_time=parse_time,
                enhanced_used=True,
                intent="location_query"
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"OpenAI JSON 解析错误: {e}")
            return NLUResult(
                text=text,
                components={"error": f"JSON解析失败: {str(e)}"},
                confidence=0.0,
                error=str(e)
            )
        except Exception as e:
            logger.error(f"OpenAI API 错误: {e}")
            return NLUResult(
                text=text,
                components={"error": str(e)},
                confidence=0.0,
                error=str(e)
            )
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """从文本中提取 JSON"""
        import re
        
        # 尝试直接解析
        try:
            return json.loads(text)
        except:
            pass
        
        # 查找 JSON 代码块
        patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'\{.*\}'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except:
                    continue
        
        # 默认返回
        return {
            "direction": "none",
            "color": "none",
            "object": "none",
            "relation": "none",
            "distance": "none",
            "confidence": 0.0,
            "need_clarification": True
        }
    
    def parse_batch(self, texts: List[str]) -> List[NLUResult]:
        """批量解析"""
        return [self.parse(text) for text in texts]
    
    def test_connection(self) -> Dict[str, Any]:
        """测试 API 连接"""
        try:
            result = self.parse("test connection")
            return {
                "success": result.error is None,
                "model": self.config.model,
                "error": result.error
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
